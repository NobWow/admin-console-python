"""
Embeddable asynchronous console command prompt with extension support
Example:

import asyncio
from admin_console import AdminCommandExecutor, basic_command_set

async def main():
    ace = AdminCommandExecutor()
    basic_command_set(ace)
    await ace.load_extensions()
    print("Terminal. Type help to see the list of commands")
    await ace.prompt_loop()
    print("Exit from command prompt")


if __name__ == "__main__":
    asyncio.run(main())
"""
import asyncio
import traceback
from .ainput import AsyncRawInput, colors
import json
import os
import importlib
import importlib.util
import types
import datetime
import logging
import re
import warnings
from math import ceil
from typing import Union, Sequence, Tuple, Mapping, MutableMapping, Awaitable, Dict, List, Set, Optional, Type, Callable, Coroutine, Any
from collections import ChainMap, defaultdict
from aiohndchain import AIOHandlerChain


ArgumentType = Union[str, int, float, bool, None]
argsplitter = re.compile('(?<!\\\\)".*?(?<!\\\\)"|(?<!\\\\)\'.*?(?<!\\\\)\'|[^ ]+')
backslasher = re.compile(r'(?<!\\)\\(.)')
allescapesplitter = re.compile(r'(\\\\|\\x[0-9a-fA-F]{2}|\\u[0-9a-fA-F]{4}|\\[0-7]{1,3}|\\[abfnrtv])')
octal = re.compile(r'\\[0-7]{1,3}')
quoted = re.compile(r'(?P<quote>["\']).*?(?P=quote)')
hexescapefinder = re.compile('(?<!\\\\)\\\\x[0-9a-fA-F]{2}')
single_char_escaper = {
    "a": "\a",
    "b": "\b",
    "f": "\f",
    "n": "\n",
    "r": "\r",
    "t": "\t",
    "v": "\v",
    " ": " "
}


class InvalidArgument(Exception):
    pass


class NotEnoughArguments(Exception):
    pass


def parse_escapes(inp: str):
    """Convert all escape code sequences into actual symbols
    The list of the escape codes is similar to Python string escape sequences"""
    body = allescapesplitter.split(inp)
    for i in range(len(body)):
        if body[i] == "\\\\":
            body[i] = "\\"
        elif body[i].startswith('\\x'):
            body[i] = chr(int(body[i][2:], base=16))
        elif body[i].startswith('\\u'):
            body[i] = chr(int(body[i][2:], base=16))
        elif octal.fullmatch(body[i]):
            body[i] = chr(int(body[i][1:], base=8))
        elif len(body[i]) == 2 and body[i][0] == "\\" and body[i][1] in single_char_escaper:
            body[i] = single_char_escaper[body[i][1:]]
    return ''.join(body)


def validate_convert_args(argtypes: Tuple[ArgumentType, str], args: Sequence) -> Union[Sequence[object], int]:  # return list if arguments are valid, int if error (index of invalid arg)
    """Validate and cast string variables into their specified type from argtypes"""
    cargs = []
    for i in range(len(args)):
        assert type(args[i]) is str, 'didn\'t you forget that you pass arguments as str because its input from the console? You just try to convert it to the type of arg...'
        try:
            cargs.append(argtypes[i](args[i]))
        except ValueError:
            return i


def paginate_list(list_: Sequence[object], elemperpage: int, cpage: int) -> (int, Sequence[object]):
    """Extract a page from a list
    Parameters
    ----------
    list_ : list
        List of the elements
    elemperpage : int
        Amount of elements per page
    cpage : int
        Number of selected page

    Returns
    -------
    int, list
        amount of pages
        the list page"""
    maxpages = ceil(len(list_) / elemperpage)
    cpage = max(1, min(cpage, maxpages))
    start_at = max((cpage - 1) * elemperpage, 0)
    end_at = min(cpage * elemperpage, len(list_))
    return maxpages, list_[start_at:end_at]


def paginate_range(count: int, elemperpage: int, cpage: int) -> (int, int, int):
    """Similar to the paginate_list(list, elemperpage, cpage), but creates a range instead of the list
    Parameters
    ----------
    count : int
        Amount of all elements
    elemperpage : int
        Amount of elements per page
    cpage : int
        Number of selected page

    Returns
    -------
    int, int, int
        amount of pages
        ID of the first element in the page
        ID of the last element in the page"""
    maxpages = ceil(count / elemperpage)
    cpage = max(1, min(cpage, maxpages))
    start_at = max((cpage - 1) * elemperpage, 0)
    end_at = min(cpage * elemperpage, count)
    return maxpages, start_at, end_at


class AdminCommand():
    """Represents a console command.
    To add a command of an extension, use AdminCommandExtension.add_command(
        afunc, name, args, optargs, description) instead
    Emits AdminCommandExecutor.cmdexec_event, AdminCommandExecutor.cmdtab_event
    """
    def __init__(self, afunc, name: str, args: Sequence[Tuple[ArgumentType, str]], optargs: Sequence[Tuple[ArgumentType, str]], description: str = '', atabcomplete: Optional[Awaitable[Sequence[str]]] = None):
        """
        Parameters
        ----------
        afunc : coroutine
            await afunc(AdminCommandExecutor, *args)
            Coroutine function that represents the command functor and receives parsed arguments
        name : str
            Name of the command
        args : list
            [(type : type, name : str), ...]
            List of the mandatory arguments of the command

            Possible types are: str, int, float, bool, None
            -------------------
            str: a word or a "string"
            int: valid number, 123
            float: a number with floating point, 123.456
            bool: switch argument, can be yes / no, true / false, y / n
            None: raw string, such an argument could be only the last argument of the command
                Cannot be used with at least 1 argument from optargs
            -------------------
        optargs : list
            [(type : class, name : str), ...]
            List of the optional arguments of the command
            Types are described in args
        description : str
            Description of what this command does. Shows in help
        atabcomplete : coroutine
            await atabcomplete(AdminCommandExecutor, *args)
            Coroutine function that is called when a tab with the last incomplete or empty argument is specified
            Must return None or a collection of suggested arguments
        """
        self.name = name
        self.args = args  # tuple (type, name)
        self.optargs = optargs  # tuple (type, name)
        # type can be str, int, float, None
        # if type is None then it is the last raw-string argument
        self.description = description
        self.afunc = afunc  # takes AdminCommandExecutor and custom args
        self.atabcomplete = atabcomplete

    async def execute(self, executor, args: Sequence[object]):
        """Shouldn't be overriden, use afunc to assign a functor to the command"""
        async with executor.cmdexec_event.emit_and_handle(self, executor, args) as handle:
            try:
                if handle()[0] is not False:
                    await self.afunc(executor, *args)
                if not executor.prompt_dispatching:
                    return
            except Exception:
                handle(False)
                raise

    async def tab_complete(self, executor, args: Sequence[object]) -> Union[tuple, None]:
        """Shouldn't be overriden, use atabcomplete to assign a tab complete handler"""
        async with executor.cmdtab_event.emit_and_handle(self, executor, args) as handle:
            if self.atabcomplete is not None:
                try:
                    _res, _args, _kwargs = handle()
                    if 'override' in _kwargs:
                        return _kwargs['override']
                    elif _res:
                        return await self.atabcomplete(executor, *args)
                except Exception:
                    handle(False)
                    raise


class AdminCommandExtension():
    """Extension data class. Constructed by AdminCommandExecutor.
    In extension scripts the instance is passed into:
        async def extension_init(AdminCommandExtension)
            called when an extension is loaded
        async def extension_cleanup(AdminCommandExtension)
            called when an extension is unloaded
    """
    def __init__(self, ACE, name: str, module: types.ModuleType, logger: logging.Logger = logging.getLogger('main')):
        """
        Parameters
        ----------
        ACE : AdminCommandExecutor
            instance of the AdminCommandExecutor class holding this extension.
        name : str
            name of the extension or extensionless filename of the extension
        module : types.ModuleType
            Python importlib module of the extension
        logger : logging.Logger
            logging.Logger instance attached to an extension
        """
        self.ace = ACE
        self.tasks: MutableMapping[str, asyncio.Task] = {
            # 'task_name': asyncio.Task
        }
        self.module = module
        self.commands: MutableMapping[str, AdminCommand] = {
            # 'cmdname': AdminCommand()
        }
        self.data = {}
        self.name = name
        self.logger = logger

    def sync_local_commands(self, overwrite=False) -> bool:
        """DEPRECATED: since version 1.1.0 admin_console uses collections.ChainMap for command overrides
        Adds all the extension commands into AdminCommandExecutor commands list
        Parameters
        ----------
        overwrite : bool
            If True, already existing commands will be replaced
            If False, fails in case when any of the extension commands are overlapping with already existing.

        Returns
        -------
        bool
            Success
        """
        return False

    def add_command(self, afunc: Callable[[Any], Coroutine[Any, Any, Any]], name: str, args: Sequence[Tuple[ArgumentType, str]] = tuple(), optargs: Sequence[Tuple[ArgumentType, str]] = tuple(), description: str = '', replace=False) -> bool:
        """Registers a command and adds it to the AdminCommandExecutor.
        Constructs an AdminCommand instance with all the arguments passed.
        Doesn't require sync_local_commands() to be run

        Note
        ----
        This function will be transformed into an async function in the future versions

        Parameters
        ----------
        see AdminCommand
        replace : bool
            Whether or not the command should be replaced if already exists. Defaults to False

        Returns
        -------
        bool
            Success
        """
        asyncio.create_task(self.ace.cmdadd_event(name, args, optargs, description))
        cmd = AdminCommand(afunc, name, args, optargs, description)
        if name not in self.ace.commands or (name in self.ace.commands and replace):
            self.commands[name] = cmd
            return True
        else:
            return False

    def remove_command(self, name: str, remove_native=False) -> bool:
        """Unregisters a command from the AdminCommandExtension and/or from an AdminCommandExecutor
        If remove_native is True, it doesn't check whether or not this command is owned by this extension
        Parameters
        ----------
        remove_native : bool
            If False, delete only if this command is owned by self
            Assign a command to AdminCommandExecutor.disabledCmd in case of it already existing

        Returns
        -------
        bool
            Success
        """
        asyncio.create_task(self.ace.cmdrm_event(name))
        if remove_native and name in self.ace.commands:
            self.commands[name] = self.ace.disabledCmd
            return True
        else:
            if name not in self.commands:
                return False
            if name in self.commands:
                del self.commands[name]
            return True
        return False

    def clear_commands(self) -> bool:
        """Clear all the commands registered by this extension

        Returns
        -------
        bool
            Success
        """
        self.commands.clear()
        return True

    def msg(self, msg: str):
        """Show message in the console with the extension prefix"""
        self.ace.print('[%s] %s' % (self.name, msg))

    def logmsg(self, msg: str, level: int = logging.INFO):
        """Write a message into the log"""
        self.logger.log(level, '[%s] %s' % (self.name, msg))


class AdminCommandExecutor():
    """This is the main class of the library. It handles command execution and extension load/unload
    Firstly, an extensions should be loaded before prompting for commands.

    ace = AdminCommandExecutor()
    await ace.load_extensions()

    Launching a prompt. Waits until exit command is invoked.
    await ace.prompt_loop()

    Perform a cleanup
    await ace.full_cleanup()

    Members
    -------
    AdminCommandExecutor.stuff : dict
        Arbitrary data storage. Can be used to share data between extensions
    AdminCommandExecutor.commands : collections.ChainMap
        dictionary of a commands and its overrides (extension commands)
            {"name": AdminCommand}, {"extensioncmd": AdminCommand}, ....
    AdminCommandExecutor.lang : dict
        dictionary of formattable strings
        {
            'nocmd': '%s: unknown command',
            'usage': 'Usage: %s',
            'invalidarg': '%s is invalid, check your command arguments.',
            'toomanyargs': 'warning: this command receives %s arguments, you provided %s or more',
            'notenoughargs': 'not enough arguments: the command receives %s arguments, you provided %s.'
        }
    AdminCommandExecutor.types : dict
        dictionary of typenames
        {
            str: 'word',
            int: 'n',
            float: 'n.n',
            bool: 'yes / no',
            None: 'text...'
        }
    AdminCommandExecutor.extpath : str
    self.prompt_dispatching : bool = True
        Set this variable to False to stop prompt_loop() after a command dispatch
    self.promptheader : str
        Command prompt header. Defaults to 'nothing'
    self.promptarrow : str
        Command prompt arrow. Defaults to '>'
    Command prompt header and arrow are combined into a prompt
    self.history : list
        Last executed commands, can be cycled with an arrow keys.
    self.ainput : AsyncRawInput(history=self.history)
        AsyncRawInput class for reading user input in raw tty mode
    self.ainput.ctrl_c = self.full_cleanup
        Async function for handling Ctrl + C
    self.disabledCmd : AdminCommand
        AdminCommand that alerts the user that this command is disabled by the system
        Used by AdminCommandExtension.remove_command(name)
    self.prompt_format = {'bold': True, 'fgcolor': colors.GREEN}
        Formatting of the prompt header and arrow.
    self.input_format = {'fgcolor': 10}
        Formatting of the user input in terminal
    self.tab_complete_lastinp = ''
        Contains last input on last tabcomplete call
    self.tab_complete_tuple = tuple()
        Contains last argument suggestions on tab complete call
    self.tab_complete_id = 0
        Contains currently cycled element ID in self.tab_complete_tuple

    Events
    self.events : collections.defaultdict(AIOHandlerChain)
        Main pool of events. Can be used to store custom events.
    self.cmdexec_event : AIOHandlerChain
        Arguments: (cmd: AdminCommand, executor: Union[AdminCommandExecutor, AdminCommandEWrapper], args: Sequence[Any])
        Emits when a command is executed through specific executor and with parsed arguments. Cancellable.
    self.cmdtab_event : AIOHandlerChain
        Arguments: (cmd: AdminCommand, executor: Union[AdminCommandExecutor, AdminCommandEWrapper], args: Sequence[Any])
        Keyword arguments are received from handlers: {'override': Sequence[str]}
        Emits when TAB key is pressed with this command. Arguments are parsed until the text cursor.
        Cancellable. Set 'override' in keyword argument dictionary to explicitly set the list of suggested strings.
    self.cmdadd_event : AIOHandlerChain
        Arguments: (name: str, args: Sequence[(Type, name)], optargs: Sequence[(Type, name)], description: str)
        Emits when an extension adds the command or self.add_command is called. Emits asynchronously since AdminCommandExtension.add_command is not async. Not cancellable.
    self.cmdrm_event : AIOHandlerChain
        Arguments: (name: str)
        Emits when an extension removes the command or self.remove_command is called. Emits asynchronously since AdminCommandExtension.remove_command is not async. Not cancellable.
    self.extload_event : AIOHandlerChain
        Arguments: (name: str)
        Emits when an extension is loading. Cancellable.
    self.extunload_event : AIOHandlerChain
        Arguments: (name: str)
        Emits when an extension is unloading. Cancellable.

    Others:
    self.print = self.ainput.writeln
    self.logger = logger
    """
    def __init__(self, stuff: Optional[Mapping] = None, use_config=True, logger: logging.Logger = None, extension_path='extensions/'):
        """
        Parameters
        ----------
        stuff : dict
            Arbitrary data storage
        use_config : bool
            Whether or not a config.json file should be created or loaded in a working directory. Defaults to True
        logger : logging.Logger
            logging.Logger instance to handle all the log messages
        extension_path : str
            Relative or absolute path to the extensions directory. Defaults to "extensions/"
        """
        if stuff is None:
            stuff = {}
        self.stuff = stuff
        self.use_config = use_config
        self.commands: Union[Dict[str, AdminCommand], ChainMap] = ChainMap()
        self.lang: Dict[str, str] = {
            'nocmd': '%s: unknown command',
            'usage': 'Usage: %s',
            'invalidarg': '%s is invalid, check your command arguments.',
            'toomanyargs': 'warning: this command receives %s arguments, you provided %s or more',
            'notenoughargs': 'not enough arguments: the command receives %s arguments, you provided %s.'
        }
        self.types: Dict[ArgumentType, str] = {
            str: 'word',
            int: 'n',
            float: 'n.n',
            bool: 'yes / no',
            None: 'text...'
        }
        self.tasks: Dict[str, asyncio.Task] = {
            # 'task_name': asyncio.Task()
        }
        self.extensions: Dict[str, AdminCommandExtension] = {
            # 'extension name': AdminCommandExtension()
        }
        self.full_cleanup_steps: Set[Awaitable] = set(
            # awaitable functions
        )
        self.extpath = extension_path
        self.prompt_dispatching = True
        self.promptheader = 'nothing'
        self.promptarrow = '>'
        self.history: List[str] = []
        self.ainput = AsyncRawInput(history=self.history)
        self.ainput.ctrl_c = self.full_cleanup
        self.prompt_format = {'bold': True, 'fgcolor': colors.GREEN}
        self.input_format = {'fgcolor': 10}
        self.logger = logger
        self.tab_complete_lastinp = ''
        self.tab_complete_tuple: Sequence[str] = tuple()
        self.tab_complete_id = 0
        # events
        self.events = defaultdict(lambda: AIOHandlerChain())
        self.events['cmdadd_event'] = self.cmdadd_event = AIOHandlerChain(cancellable=False)
        self.events['cmdrm_event'] = self.cmdrm_event = AIOHandlerChain(cancellable=False)
        self.events['cmdexec_event'] = self.cmdexec_event = AIOHandlerChain()
        self.events['cmdtab_event'] = self.cmdtab_event = AIOHandlerChain()
        self.events['extload_event'] = self.extload_event = AIOHandlerChain()
        self.events['extunload_event'] = self.extunload_event = AIOHandlerChain()

        async def disabledCmd(cmd: AdminCommandExecutor, *args):
            cmd.error("This command is disabled by system")

        self.disabledCmd = AdminCommand(disabledCmd, 'disabled', tuple(), tuple())
        if use_config:
            self.load_config()

    def add_command(self, afunc: Callable[[Any], Coroutine[Any, Any, Any]], name: str, args: Sequence[Tuple[Type, str]] = tuple(), optargs: Sequence[Tuple[Type, str]] = tuple(), description: str = '', atabcomplete: Optional[Callable[[Any], Coroutine[Any, Any, Any]]] = None) -> bool:
        """
        Constructs an AdminCommand instance with all the arguments passed.
        Adds the command to the first layer of the chainmap
        Emits cmdadd_event asynchronously.

        Note
        ----
        This function will be transformed into an async function in the future versions

        Parameters
        ----------
        see AdminCommand

        Returns
        -------
        bool
            Success
        """
        if name in self.commands:
            return False
        asyncio.create_task(self.cmdadd_event(name, args, optargs, description))
        self.commands.maps[0][name] = AdminCommand(afunc, name, args, optargs, description, atabcomplete)
        return True

    def remove_command(self, name: str):
        """
        Permanently removes the command from the chainmap.
        Emits cmdrm_event asynchronously.

        Note
        ----
        This function will be transformed into an async function in the future versions

        Parameters
        ----------
        name : str
            name of the command

        Returns
        -------
        bool
            Success
        """
        if name not in self.commands:
            return False
        asyncio.create_task(self.cmdrm_event(name))
        del self.commands[name]
        return True

    def print(self, *value, sep=' ', end='\n'):
        """Prints a message in the console, preserving the prompt and user input, if any
        Partially copies the Python print() command"""
        str_ = sep.join(str(element) for element in value)
        if end == '\n':
            self.ainput.writeln(str_)
        else:
            self.ainput.write(str_)

    def error(self, msg: str):
        """Shows a red error message in the console and logs.
        ERROR: msg

        Parameters
        ----------
        msg : str
            Message"""
        if self.logger:
            self.logger.error(msg)
        else:
            self.ainput.writeln('ERROR: %s' % msg, fgcolor=colors.RED)

    def info(self, msg: str):
        """Shows a regular info message in the console and logs.

        Parameters
        ----------
        msg : str
            Message
        """
        if self.logger:
            self.logger.info(msg)
        else:
            self.ainput.writeln('INFO: %s' % msg)

    def log(self, msg: str, level=10):
        """Shows a log message in the console and logs

        Parameters
        ----------
        msg : str
            Message
        level : int
            Level of the log message between 0 and 50
        """
        if self.logger:
            self.logger.log(level, msg)
        elif level >= self.logger.getEffectiveLevel():
            levels = {0: 'NOTSET', 10: 'DEBUG', 20: 'INFO', 30: 'WARNING', 40: 'ERROR', 50: 'CRITICAL'}
            self.ainput.writeln('%s: %s' % (levels[level], msg))

    async def load_extensions(self):
        """Loads extensions from an extension directory specified in AdminCommandExecutor.extpath"""
        extlist = []
        if os.path.exists(os.path.join(self.extpath, 'extdep.txt')):
            with open(os.path.join(self.extpath, 'extdep.txt'), 'r') as f:
                for x in f.readlines():
                    # Light vulnerability fix: path traversal
                    if '../' in x or x.startswith('/'):
                        continue
                    extlist.append('%s.py' % x.strip())
        else:
            self.print('Note: create extdep.txt in the extensions folder to sequentally load modules')
        if not os.path.exists(self.extpath):
            try:
                os.makedirs(self.extpath)
            except OSError as exc:
                self.error('Failed to create extension directory: %s: %s' % (type(exc).__name__, exc))
                return
        with os.scandir(self.extpath) as extpath:
            for file in extpath:
                if file.name.endswith('.py') and file.is_file() and file.name not in extlist:
                    extlist.append(file.name)
        for name in extlist:
            if not os.path.exists(os.path.join(self.extpath, name)):
                self.error('Module file %s not found' % name)
            await self.load_extension(name.split('.')[0])

    def load_config(self, path: str = 'config.json') -> bool:
        """Loads a configuration from a JSON file

        Parameters
        ----------
        path : str
            Absolute or relative path to the config file. Defaults to "config.json"

        Returns
        -------
        bool
            Success
        """
        try:
            file = open(path, 'r')
            self.info('')
            self.config = json.loads(file.read())
            file.close()
            return True
        except (json.JSONDecodeError, OSError):
            self.error("Error occurred during load of the config: \n%s" % traceback.format_exc())
            return False
        except FileNotFoundError:
            self.error("Configuration file is not found")
            return False

    def save_config(self, path: str = 'config.json') -> bool:
        """Saves a configuration into a JSON file

        Parameters
        ----------
        path : str
            Absolute or relative path to the config file. Defaults to "config.json"

        Returns
        -------
        bool
            Success
        """
        try:
            file = open(path, 'w')
            file.write(json.dumps(self.config, indent=True))
            file.close()
            return True
        except OSError:
            self.error('Failed to save configuration at %s' % path)
            return False

    async def load_extension(self, name: str) -> bool:
        """Loads a Python script as an extension from its extension path and call await extension_init(AdminCommandExtension)
        Emits extload_event.

        Parameters
        ----------
        name : str
            The extensionless name of a file containing a Python script

        Returns
        -------
        bool
            Success
        """
        async with self.extload_event.emit_and_handle(name, before=False) as handle:
            if not handle()[0]:
                return False
            if name in self.extensions:
                self.error("Failed to load extension %s: This extension is already loaded." % name)
                return False
            _path = os.path.join(self.extpath, name + '.py')
            # Light vulnerability fix: path traversal
            if '../' in _path or _path.startswith('/'):
                self.error("Failed to load extension %s: path should not be absolute or contain \"..\"" % name)
                return False
            spec = importlib.util.spec_from_file_location(name, _path)
            module = importlib.util.module_from_spec(spec)
            try:
                spec.loader.exec_module(module)
                if 'extension_init' not in module.__dict__ or 'extension_cleanup' not in module.__dict__:
                    self.error("Cannot load %s: missing extension_init or extension_cleanup" % name)
                    return
                extension = AdminCommandExtension(self, name, module, logger=self.logger)
                self.info('Loading extension %s' % name)
                await module.extension_init(extension)
                self.extensions[name] = extension
                self.commands.maps.insert(0, extension.commands)
                return True
            except BaseException:
                self.error("Failed to load extension %s:\n%s" % (name, traceback.format_exc()))
                return False

    async def unload_extension(self, name: str, keep_dict=False) -> bool:
        """Unload an extension and call await extension_cleanup(AdminCommandExtension)
        Emits extunload_event.

        Parameters
        ----------
        name : str
            The name of an extension
        keep_dict : bool
            DEPRECATED: ignored in version 1.1.0
            Whether or not this extension should be kept in the list. Defaults to False

        Returns
        -------
        bool
            Success
        """
        assert self.extunload_event._lock != self.cmdexec_event._lock
        async with self.extunload_event.emit_and_handle(name, before=False) as handle:
            if not handle()[0]:
                return False
            if name not in self.extensions:
                handle(False)
                return None
            for task in self.extensions[name].tasks:
                self.extensions[name].tasks[task].cancel()
            try:
                extension = self.extensions[name]
                if not keep_dict:
                    warnings.warn("keep_dict is ignored in version 1.1.0", DeprecationWarning)
                del self.extensions[name]
                await extension.module.extension_cleanup(extension)
                self.commands.maps.remove(extension.commands)
                return True
            except Exception:
                self.error("Failed to call cleanup in %s:\n%s" % (name, traceback.format_exc()))
                handle(False)
                return False

    def parse_args(self, argl: str, argtypes: Sequence[Tuple[ArgumentType, str]] = None, opt_argtypes: Sequence[Tuple[ArgumentType, str]] = None, *, raise_exc=True) -> (list, str):
        """
        Smart split the argument line and convert all the arguments to its types
        Raises InvalidArgument(argname) if one of the arguments couldn't be converted
        Raises NotEnoughArguments(len(args)) if there isn't enough arguments provided

        Parameters
        ----------
        argl : str
            Argument line with raw space-separated arguments
        argtypes : list
            Any collection containing a tuples (type, name) representing an argument name and type
            The arguments that are listed in there are mandatory.
        opt_argtypes : list
            Same as argtypes, but those arguments are parsed after mandatory and are optional.
            Doesn't cause NotEnoughArguments to be raise if there is not enough optional arguments
        raise_exc : bool
            Whether or not exceptions are raised. If False, data is returned as it is

        Returns
        -------
        list, str
            A list of converted arguments and the remnant
            If remnant isn't an empty string, then there is probably too many arguments provided
        """
        args = []
        if argtypes:
            for argtype, argname in argtypes:
                if argtype is None:
                    args.append(parse_escapes(argl))
                    argl = ''
                    break
                # arg, _, argl = argl.partition(' ')
                argmatch = argsplitter.search(argl)
                if not argmatch:
                    if raise_exc:
                        raise NotEnoughArguments(len(argtypes), len(args))
                    else:
                        return args, argl
                arg = argl[argmatch.start():argmatch.end()]
                argl = argl[argmatch.end():]
                if quoted.fullmatch(arg) is not None:
                    arg = arg[1:-1]
                try:
                    if argtype is bool:
                        args.append(True if arg.lower() in ['true', 'yes', 'y', '1'] else False)
                    elif argtype is str:
                        args.append(parse_escapes(arg))
                    else:
                        args.append(argtype(arg))
                except ValueError:
                    if raise_exc:
                        raise InvalidArgument(argname)
                    else:
                        args.append(arg)
                        return args, argl
        if opt_argtypes:
            for argtype, argname in opt_argtypes:
                if argtype is None and argl:
                    args.append(argl)
                    argl = ''
                    break
                elif not argl:
                    break
                argmatch = argsplitter.search(argl)
                arg = argl[argmatch.start():argmatch.end()]
                argl = argl[argmatch.end():]
                if quoted.fullmatch(arg) is not None:
                    arg = arg[1:-1]
                try:
                    if argtype is bool:
                        args.append(True if arg.lower() in ['true', 'yes', 'y', '1'] else False)
                    elif argtype is str:
                        args.append(parse_escapes(arg))
                    else:
                        args.append(argtype(arg))
                except ValueError:
                    if raise_exc:
                        raise InvalidArgument(argname)
                    else:
                        args.append(arg)
                        return args, argl
        return args, argl

    async def dispatch(self, cmdline: str) -> bool:
        """Executes a command. Shouldn't be used explicitly. Use prompt_loop() instead.

        Parameters
        ----------
        cmdline : str
            A whole line representing a command

        Returns
        -------
        bool
            Success
        """
        cmdname, _, argl = cmdline.partition(' ')
        argl = argl.strip()
        if cmdname not in self.commands:
            self.print(self.lang['nocmd'] % cmdname)
            return
        cmd: Union[AdminCommand, None] = self.commands[cmdname]
        if cmd is None or cmd is self.disabledCmd:
            self.print(self.lang['nocmd'] % cmdname)
            return
        try:
            args, argl = self.parse_args(argl, cmd.args, cmd.optargs)
            if argl:
                self.print(self.lang['toomanyargs'] % (len(cmd.args) + len(cmd.optargs), len(args) + 1))
                self.print(self.usage(cmdname))
            await cmd.execute(self, args)
        except NotEnoughArguments as exc:
            self.print(self.lang['notenoughargs'] % (len(cmd.args), exc.args[0]))
            self.print(self.usage(cmdname))
        except InvalidArgument as exc:
            self.print(self.lang['invalidarg'] % exc.args[0])
            self.print(self.usage(cmdname))
        except Exception as exc:
            self.print("An exception just happened in this command, check logs or smth...")
            self.error("Command execution failed: %s" % exc)

    def usage(self, cmdname: str, lang=True) -> str:
        """Get a formatted usage string for the command
        Raises KeyError if the command doesn't exist.

        Parameters
        ----------
        cmdname : str
            Name of the command
        lang : bool
            Whether or not returned string should be formatted through lang['usage']

        Returns
        -------
        str
            Formatted string representing a usage of the command
        """
        cmd = self.commands[cmdname]
        if cmd.args:
            mandatory_args = ['<%s: %s>' % (x[1], self.types[x[0]]) for x in cmd.args]
        else:
            mandatory_args = ''
        if cmd.optargs:
            optional_args = ['[%s: %s]' % (x[1], self.types[x[0]]) for x in cmd.optargs]
        else:
            optional_args = ''
        usage = '%s %s %s' % (cmdname, ' '.join(mandatory_args), ' '.join(optional_args))
        if lang:
            return self.lang['usage'] % (usage)
        else:
            return usage

    async def prompt_loop(self):
        """Asynchronously prompt for commands and dispatch them
        Blocks asynchronously until the prompt is closed.
        Prefer launching in asyncio.Task wrapper.
        """
        self.ainput.prepare()
        self.ainput.add_keystroke('\t', self._tab_complete)
        # loop = asyncio.get_event_loop()
        # loop.add_signal_handler(2, self.ctrl_c)
        while self.prompt_dispatching:
            inp = await self.ainput.prompt_line('%s%s ' % (self.promptheader, self.promptarrow), prompt_formats=self.prompt_format, input_formats=self.input_format)
            if not inp:
                continue
            if self.prompt_dispatching:
                self.tasks['prompt_cmd'] = asyncio.create_task(self.dispatch(inp))
                try:
                    await self.tasks['prompt_cmd']
                except asyncio.CancelledError:
                    pass
        self.ainput.remove_keystroke('\t')
        self.ainput.end()

    async def full_cleanup(self):
        """Perform cleanup steps in AdminCommandExecutor.full_cleanup_steps
        """
        self.prompt_dispatching = False
        self.ainput.is_reading = False
        for func in self.full_cleanup_steps:
            await func(self)
        for extension in tuple(self.extensions.keys()):
            await self.unload_extension(extension)

    async def _tab_complete(self):
        """This is callback function for TAB key event
           Uses AsyncRawInput data to handle the text update
        """
        inp = ''.join(self.ainput.read_lastinp[0:self.ainput.cursor])
        if not inp or not self.prompt_dispatching:
            return
        if self.tab_complete_lastinp != inp:
            # generate new suggestions and replace last argument
            cmdname, sep, argl = inp.partition(' ')
            if not sep:
                # generate list of command suggestions
                self.tab_complete_tuple = []
                for cmdname in self.commands:
                    if cmdname.startswith(inp) and self.commands[cmdname] is not self.disabledCmd:
                        self.tab_complete_tuple.append(cmdname)
                self.tab_complete_id = 0
                self.ainput.writeln('\t'.join(self.tab_complete_tuple), fgcolor=colors.GREEN)
            else:
                argl = argl.strip()
                if cmdname not in self.commands:
                    return
                cmd: AdminCommand = self.commands[cmdname]
                if cmd is self.disabledCmd:
                    return
                try:
                    args, argl = self.parse_args(argl, cmd.args, cmd.optargs, raise_exc=False)
                    if argl:
                        self.print(self.lang['toomanyargs'] % (len(cmd.args) + len(cmd.optargs), len(args) + 1))
                        self.print(repr(argl))
                        self.print(self.usage(cmdname))
                    suggestions = await cmd.tab_complete(self, args)
                    if suggestions is not None:
                        self.ainput.writeln('\t'.join(suggestions), fgcolor=colors.GREEN)
                    self.tab_complete_tuple = suggestions
                    self.tab_complete_id = 0
                except asyncio.CancelledError:
                    if self.prompt_dispatching:
                        self.print("Process has been cancelled")
            self.tab_complete_lastinp = inp
        elif self.tab_complete_tuple:
            # cycle through the suggestions and replace last argument
            self.tab_complete_id = (self.tab_complete_id + 1) % len(self.tab_complete_tuple)
        # replace last argument
        if self.tab_complete_tuple:
            _, _, last_arg = inp.rpartition(' ')
            target = self.tab_complete_tuple[self.tab_complete_id]
            if not last_arg:
                self.ainput.read_lastinp.extend(target)
            else:
                self.ainput.read_lastinp[-len(last_arg):] = target
            self.ainput.cursor = len(self.ainput.read_lastinp)
            self.ainput.redraw_lastinp(1)
            inp = ''.join(self.ainput.read_lastinp[0:self.ainput.cursor])
            self.tab_complete_lastinp = inp


class FakeAsyncRawInput():
    """
    A helper class for emulating real user input. Used to pipe the IO through custom ways.
    Doesn't handle any ANSI escape codes at all. Just lets the data go through
    Works best with AdminCommandEWrapper.
    See ainput.AsyncRawInput for more info
    """
    def __init__(self, ace: AdminCommandExecutor, *args, **kwargs):
        self.ace: AdminCommandExecutor = ace
        super().__init__(*args, **kwargs)

    def __del__(self):
        # end() was here...
        pass

    def prepare(self):
        """Do something before handling IO"""
        # useless!
        pass

    def end(self):
        """Do something after IO"""
        # useless too!
        pass

    def set_interrupt_handler(self, awaitable):
        """Ctrl + C event in the real terminal. Useless here"""
        # I don't support that...
        self.ace.log("set_interrupt_handler: attempt to change an interrupting callback inside of the fake ARI, which is not supported.", logging.WARNING)

    def add_keystroke(self, keystroke: str, awaitable):
        """Add a key press event"""
        # I don't support that either...
        self.ace.log("add_keystroke: attempt to add a keystroke %s handler, which is not supported by a fake ARI." % repr(keystroke))

    def remove_keystroke(self, keystroke: str):
        """Remove key press event"""
        self.ace.log("remove_keystroke: attempt to remove a keystroke %s handler, which is not supported by a fake ARI." % repr(keystroke))

    def write(self, msg: str, **formats):
        """Send raw data without end of line"""
        raise NotImplementedError

    def writeln(self, msg: str, **formats):
        """Send a formatted message line with EOL"""
        raise NotImplementedError

    def redraw_lastinp(self, at: int):
        # it's a dummy function here
        pass

    def prompt_line(self, prompt="", echo=True, history_disabled=False, prompt_formats: Optional[MutableMapping] = None, input_formats: Optional[MutableMapping] = None):
        """Ask a user for input"""
        raise NotImplementedError

    def prompt_keystroke(self, prompt="", echo=True):
        """Ask a user to press a key"""
        raise NotImplementedError


class AdminCommandEWrapper(AdminCommandExecutor):
    """
    AdminCommandExecutor wrapper. It overlays all the functions from the AdminCommandExecutor, but anything could be replaced to change command behavior, like redirecting its output or providing a different way of IO.
    It operates with an existing AdminCommandExecutor, if a special keyword argument is specified.
    Usually it is passed into the command callback instead of the AdminCommandExecutor, like when a command output needs to be handled differently.
    Useful for a remote console server, or GUI, where command output is sent to a remote client or is shown in a GUI.
    A proper way to use it is to change its members, for example assigning a different AsyncRawInput or logger implementation.
    WARNING: this is NOT a sandbox solution for the extensions! This class is used to execute commands in a different way
    """
    def __init__(self, *args, ace: AdminCommandExecutor, **kwargs):
        self.master = ace

    def __getattribute__(self, name: str, /):
        try:
            attr = object.__getattribute__(self, name)
            return attr
        except AttributeError:
            pass
        return getattr(object.__getattribute__(self, 'master'), name)

    def __setattr__(self, name: str, value, /):
        master = object.__getattribute__(self, 'master')
        if name in object.__getattribute__(self, '__dict__') and name not in master.__dict__:
            object.__setattr__(self, name, value)
        else:
            setattr(master, name, value)

    def override(self, name: str, value):
        object.__setattr__(self, name, value)


def basic_command_set(ACE: AdminCommandExecutor):
    commands_ = ACE.commands.maps[0]

    async def exitquit(self: AdminCommandExecutor):
        self.print("Command prompt is closing")
        self.prompt_dispatching = False
        try:
            await self.full_cleanup()
        except Exception:
            self.print(traceback.format_exc())
    commands_['exit'] = commands_['quit'] = \
        AdminCommand(exitquit, 'exit', [], [], 'Exit from command prompt and gracefully shutdown')

    async def help_(self: AdminCommandExecutor, cpage: int = 1):
        usagedesc = []
        cmdnorepeat = []
        for cmdname in self.commands:
            cmd = self.commands[cmdname]
            if cmd in cmdnorepeat:
                continue
            cmdnorepeat.append(cmd)
            usagedesc.append('| %s -> %s' % (self.usage(cmdname, False), cmd.description))
        maxpage, list_ = paginate_list(usagedesc, 6, cpage)
        self.print(('Help (page %s out of %s)\n' % (cpage, maxpage)) + '\n'.join(list_))
    commands_['help'] = commands_['?'] = \
        AdminCommand(help_, 'help', [], [(int, 'page')], 'Show all commands')

    async def extlist(self: AdminCommandExecutor, cpage: int = 1):
        ls = list(self.extensions.keys())
        maxpage, pgls = paginate_list(ls, 7, cpage)
        cpage = max(min(maxpage, cpage), 1)
        self.print('Extensions (page %s of %s):\n%s' % (cpage, maxpage, '\n'.join(pgls)))
    commands_['extlist'] = \
        commands_['extls'] = \
        commands_['lsext'] = \
        commands_['exts'] = \
        commands_['extensions'] = \
        AdminCommand(extlist, 'extlist', [], [(int, 'page')], 'List loaded extensions')

    async def extload(self: AdminCommandExecutor, name: str):
        if not await self.load_extension(name):
            self.print("%s failed to load." % name)
            return
        else:
            self.print("%s loaded." % name)
    commands_['extload'] = \
        commands_['extensionload'] = \
        commands_['loadextension'] = \
        AdminCommand(extload, 'extload', [(None, 'filename (without .py)')], [], 'Load an extension (from extensions directory)')

    async def extunload(self: AdminCommandExecutor, name: str):
        res = await self.unload_extension(name)
        if res is None:
            self.print("%s is not loaded or non-existant extension." % name)
        elif res is False:
            self.print("%s is unloaded but failed to cleanup." % name)
        else:
            self.print("%s unloaded." % name)

    async def extunload_tab(self: AdminCommandExecutor, *args):
        name = args[0]
        res = []
        for extname in self.extensions:
            if extname.startswith(name):
                res.append(extname)
        return res
    commands_['extunload'] = \
        commands_['unloadext'] = \
        commands_['extremove'] = \
        commands_['removeext'] = \
        commands_['extensionremove'] = \
        commands_['removeextension'] = \
        AdminCommand(extunload, 'extunload', [(None, 'name (without .py)')], [], 'Unload an extension', extunload_tab)

    async def extreload(self: AdminCommandExecutor, name: str):
        res = await self.unload_extension(name)
        if res is None:
            self.print("%s is not loaded or non-existant extension." % name)
        elif res is False:
            self.print("%s is unloaded but failed to cleanup." % name)
        else:
            self.print("%s unloaded." % name)
            if not await self.load_extension(name):
                self.print("%s failed to load." % name)
                return
            else:
                self.print("%s loaded." % name)
    commands_['extreload'] = \
        commands_['reloadext'] = \
        commands_['extrestart'] = \
        commands_['reloadextension'] = \
        commands_['extensionreload'] = \
        commands_['restartextension'] = \
        commands_['extensionrestart'] = \
        commands_['relext'] = \
        commands_['extrel'] = \
        AdminCommand(extreload, 'extreload', [(None, 'name (without .py)')], [], 'Reload an extension', extunload_tab)

    async def date(self: AdminCommandExecutor):
        date = datetime.datetime.now()
        self.print("It's %s.%s.%s-%s:%s:%s right now" % (date.day, date.month, date.year, date.hour, date.minute, date.second))
    commands_['date'] = AdminCommand(date, 'date', [], [], 'Show current date')

    async def error(self: AdminCommandExecutor):
        self.error("Stderr!")
    commands_['error'] = AdminCommand(error, 'error', [], [], 'Test error')

    async def testoptargs(self: AdminCommandExecutor, mand1: str, opt1: int = 0, opt2: str = '', opt3: int = 0):
        self.print(mand1, opt1, opt2, opt3)

    async def testoptargs_tab(self: AdminCommandExecutor, *args):
        if len(args) == 0:
            return "example", "another", "main", "obsolete"
        elif len(args) == 2:
            return "optional", "not", "needed"
    commands_['testoptargs'] = AdminCommand(testoptargs, 'testoptargs', [(str, 'mandatory')], ((int, 'opt1'), (str, 'opt2'), (int, 'opt3')), 'See how argument parsing and tabcomplete works', testoptargs_tab)

    if ACE.extensions:
        warnings.warn("basic_command_set() should be initialized before loading extensions", RuntimeWarning)


async def main():
    # scoped functions

    ACE = AdminCommandExecutor(None, use_config=False)
    ACE.print("Plain command line")
    basic_command_set(ACE)
    await ACE.load_extensions()
    await ACE.prompt_loop()
if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
