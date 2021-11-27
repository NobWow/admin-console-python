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
import types
import datetime
import logging
import re
from math import ceil


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


def str_findall_unescaped(str_: str, char: int):
    list_ = []
    _next_pos = -1
    for i in range(str_.count(chr(char))):
        _next_pos = str_.find(chr(char), _next_pos + 1)
        if str_[_next_pos - 1] != chr(92):
            list_.append(_next_pos)
    return list_


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


def validate_convert_args(argtypes: tuple, args: list):  # return list if arguments are valid, int if error (index of invalid arg)
    """Validate and cast string variables into their specified type from argtypes"""
    cargs = []
    for i in range(len(args)):
        assert type(args[i]) is str, 'didn\'t you forget that you pass arguments as str because its input from the console? You just try to convert it to the type of arg...'
        try:
            cargs.append(argtypes[i](args[i]))
        except ValueError:
            return i


def paginate_list(list_: list, elemperpage: int, cpage: int) -> (int, list):
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

    def __init__(self, afunc, name: str, args: list, optargs: list, description: str = '', atabcomplete=None):
        """Represents a console command.
        To add a command of an extension, use AdminCommandExtension.add_command(
            afunc, name, args, optargs, description) instead
        Parameters
        ----------
        afunc : coroutine
            await afunc(AdminCommandExecutor, *args)
            Coroutine function that represents the command functor and receives parsed arguments
        name : str
            Name of the command
        args : list
            [(type : class, name : str), ...]
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
            Must return None or a list of suggested arguments
        """
        self.name = name
        self.args = args  # tuple (type, name)
        self.optargs = optargs  # tuple (type, name)
        # type can be str, int, float, None
        # if type is None then it is the last raw-string argument
        self.description = description
        self.afunc = afunc  # takes AdminCommandExecutor and custom args
        self.atabcomplete = atabcomplete

    async def execute(self, executor, args):
        """Shouldn't be overriden, use afunc to assign a functor to the command"""
        await self.afunc(executor, *args)

    async def tab_complete(self, executor, args):
        """Shouldn't be overriden, use atabcomplete to assign a tab complete handler"""


class AdminCommandExtension():
    def __init__(self, ACE, name: str, module: types.ModuleType, logger: logging.Logger = logging.getLogger('main')):
        """Extension data class. Constructed by AdminCommandExecutor.
        In extension scripts the instance is passed into:
            async def extension_init(AdminCommandExtension)
                called when an extension is loaded
            async def extension_cleanup(AdminCommandExtension)
                called when an extension is unloaded
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
        self.tasks = {
            # 'task_name': asyncio.Task
        }
        self.module = module
        self.commands = {
            # 'cmdname': AdminCommand()
        }
        self.data = {}
        self.name = name
        self.logger = logger

    def sync_local_commands(self, overwrite=False):
        """Adds all the extension commands into AdminCommandExecutor commands list
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
        if not self.commands:
            return False
        if overwrite:
            self.ace.commands.update(self.commands)
            return True
        if self.commands.keys().isdisjoint(self.ace.commands.keys()):
            self.ace.commands.update(self.commands)
            return True
        else:
            return False

    def add_command(self, afunc, name: str, args: list, optargs: list = [], description: str = '', replace=False):
        """Registers a command and adds it to the AdminCommandExecutor.
        Constructs an AdminCommand instance with all the arguments passed.
        Doesn't require sync_local_commands() to be run
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
        cmd = AdminCommand(afunc, name, args, optargs, description)
        if name not in self.ace.commands or (name in self.ace.commands and replace):
            self.commands[name] = cmd
            self.ace.commands[name] = cmd
            return True
        else:
            return False

    def remove_command(self, name: str, remove_native=False):
        """Unregisters a command from the AdminCommandExtension and/or from an AdminCommandExecutor
        If remove_native is True, it doesn't check whether or not this command is owned by this extension
        Parameters
        ----------
        remove_native : bool
            If False, delete only if this command is owned by self

        Returns
        -------
        bool
            Success
        """
        if remove_native:
            if name in self.ace.commands:
                del self.ace.commands[name]
                return True
            else:
                return False
        else:
            if name not in self.commands and name not in self.ace.commands:
                return False
            if name in self.commands:
                del self.commands[name]
            if name in self.ace.commands:
                del self.ace.commands[name]
            return True

    def clear_commands(self):
        """Clear all the commands registered by this extension

        Returns
        -------
        bool
            Success
        """
        for cmdname in self.commands:
            if cmdname in self.ace.commands:
                del self.ace.commands[cmdname]
        else:
            return False
        return True

    def msg(self, msg):
        """Show message in the console with the extension prefix"""
        self.ace.print('[%s] %s' % (self.name, msg))

    def logmsg(self, msg: str, level: int = logging.INFO):
        """Write a message into the log"""
        self.logger.log(1, '[%s] %s' % (self.name, msg))


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
    AdminCommandExecutor.commands : dict
        dictionary of a commands
            {"name": AdminCommand}
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
    self.prompt_format = {'bold': True, 'fgcolor': colors.GREEN}
        Formatting of the prompt header and arrow.
    self.input_format = {'fgcolor': 10}
        Formatting of the user input in terminal
    Others:
    self.print = self.ainput.writeln
    self.logger = logger
    """
    def __init__(self, stuff: dict = {}, use_config=True, logger: logging.Logger = None, extension_path='extensions/'):
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
        self.stuff = stuff
        self.use_config = use_config
        self.commands = {}
        self.lang = {
            'nocmd': '%s: unknown command',
            'usage': 'Usage: %s',
            'invalidarg': '%s is invalid, check your command arguments.',
            'toomanyargs': 'warning: this command receives %s arguments, you provided %s or more',
            'notenoughargs': 'not enough arguments: the command receives %s arguments, you provided %s.'
        }
        self.types = {
            str: 'word',
            int: 'n',
            float: 'n.n',
            bool: 'yes / no',
            None: 'text...'
        }
        self.tasks = {
            # 'task_name': asyncio.Task()
        }
        self.extensions = {
            # 'extension name': AdminCommandExtension()
        }
        self.full_cleanup_steps = set(
            # awaitable functions
        )
        self.extpath = extension_path
        self.prompt_dispatching = True
        self.promptheader = 'nothing'
        self.promptarrow = '>'
        self.history = []
        self.ainput = AsyncRawInput(history=self.history)
        self.ainput.ctrl_c = self.full_cleanup
        self.prompt_format = {'bold': True, 'fgcolor': colors.GREEN}
        self.input_format = {'fgcolor': 10}
        self.logger = logger
        if use_config:
            self.load_config()

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
        extlist = set()
        if os.path.exists(os.path.join(self.extpath, 'extdep.txt')):
            with open(os.path.join(self.extpath, 'extdep.txt'), 'r') as f:
                extlist.update('%s.py' % x for x in f.readlines())
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
                    extlist.add(file.name)
        for name in extlist:
            if not os.path.exists(os.path.join(self.extpath, name)):
                self.error('Module file %s not found' % name)
            await self.load_extension(name.split('.')[0])

    def load_config(self, path: str = 'config.json'):
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

    def save_config(self, path: str = 'config.json'):
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
            self.config = json.dumps(self.config, indent=True)
            file.close()
            return True
        except OSError:
            self.error('Failed to save configuration at %s' % path)
            return False

    async def load_extension(self, name: str):
        """Loads a Python script as an extension from its extension path and call await extension_init(AdminCommandExtension)

        Parameters
        ----------
        name : str
            The extensionless name of a file containing a Python script

        Returns
        -------
        bool
            Success
        """
        spec = importlib.util.spec_from_file_location(name, os.path.join(self.extpath, name + '.py'))
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
            return True
        except BaseException:
            self.error("Failed to load extension %s:\n%s" % (name, traceback.format_exc()))
            return False

    async def unload_extension(self, name: str, keep_dict=False):
        """Unload an extension and call await extension_cleanup(AdminCommandExtension)
        Parameters
        ----------
        name : str
            The name of an extension
        keep_dict : bool
            Whether or not this extension should be kept in the list. Defaults to False

        Returns
        -------
        bool
            Success
        """
        if name not in self.extensions:
            return None
        for task in self.extensions[name].tasks:
            self.extensions[name].tasks[task].cancel()
        try:
            extension = self.extensions[name]
            extension.clear_commands()
            if not keep_dict:
                del self.extensions[name]
            await extension.module.extension_cleanup(extension)
            return True
        except Exception:
            self.error("Failed to call cleanup in %s:\n%s" % (name, traceback.format_exc()))
            return False

    async def dispatch(self, cmdline: str):
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
        args = []
        cmdname, _, argl = cmdline.partition(' ')
        argl = argl.strip()
        if cmdname not in self.commands:
            self.print(self.lang['nocmd'] % cmdname)
            return
        cmd: AdminCommand = self.commands[cmdname]
        if cmd.args:
            for argtype, argname in cmd.args:
                if argtype is None:
                    args.append(parse_escapes(argl))
                    argl = ''
                    break
                # arg, _, argl = argl.partition(' ')
                argmatch = argsplitter.search(argl)
                if not argmatch:
                    self.print(self.lang['notenoughargs'] % (len(cmd.args), len(args)))
                    self.print(self.usage(cmdname))
                    return
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
                    self.print(self.lang['invalidarg'] % argname)
                    self.print(self.usage(cmdname))
                    return
        if cmd.optargs:
            for argtype, argname in cmd.optargs:
                if argtype is None and argl:
                    args.append(argl)
                    argl = ''
                    break
                elif not argl:
                    break
                argmatch = argsplitter.search(argl)
                arg = argl[argmatch.start():argmatch.end()]
                argl = argl[argmatch.end() - 1:]
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
                    self.print(self.lang['invalidarg'] % argname)
                    self.print(self.usage(cmdname))
                    return
        if argl:
            self.print(self.lang['toomanyargs'] % (len(cmd.args) + len(cmd.optargs), len(args) + 1))
            self.print(self.usage(cmdname))
        try:
            await cmd.execute(self, args)
        except Exception as exc:
            self.print("An exception just happened in this command, check logs or smth...")
            self.error("Command execution failed: %s" % exc)

    def usage(self, cmdname: str, lang=True):
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
        self.ainput.end()

    async def full_cleanup(self):
        """Perform cleanup steps in AdminCommandExecutor.full_cleanup_steps
        """
        self.prompt_dispatching = False
        self.ainput.is_reading = False
        for func in self.full_cleanup_steps:
            await func(self)

    async def tab_complete(self):
        """This is callback function for TAB key event
           Uses AsyncRawInput data to handle the text update
        """
        inp = self.ainput.read_lastinp[0:self.ainput.cursor]
        if not inp:
            return
        if self.prompt_dispatching:
            self.tasks['prompt_cmd'] = asyncio.create_task(self.dispatch(inp))
            try:
                await self.tasks['prompt_cmd']
            except asyncio.CancelledError:
                if self.prompt_dispatching:
                    self.print("Process has been cancelled")


def basic_command_set(ACE: AdminCommandExecutor):

    async def exitquit(self: AdminCommandExecutor):
        self.prompt_dispatching = False
        try:
            await self.full_cleanup()
        except Exception:
            self.print(traceback.format_exc())
    ACE.commands['exit'] = ACE.commands['quit'] = \
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
    ACE.commands['help'] = ACE.commands['?'] = \
        AdminCommand(help_, 'help', [], [(int, 'page')], 'Show all commands')

    async def extlist(self: AdminCommandExecutor, cpage: int = 1):
        ls = list(self.extensions.keys())
        maxpage, pgls = paginate_list(ls, 7, cpage)
        cpage = max(min(maxpage, cpage), 1)
        self.print('Extensions (page %s of %s): %s' % (cpage, maxpage, '\n'.join(pgls)))
    ACE.commands['extlist'] = \
        ACE.commands['extls'] = \
        ACE.commands['lsext'] = \
        ACE.commands['exts'] = \
        ACE.commands['extensions'] = \
        AdminCommand(extlist, 'extlist', [], [(int, 'page')], 'List loaded extensions')

    async def extload(self: AdminCommandExecutor, name: str):
        if not await self.load_extension(name):
            self.print("%s failed to load." % name)
            return
        else:
            self.print("%s loaded." % name)
    ACE.commands['extload'] = \
        ACE.commands['extensionload'] = \
        ACE.commands['loadextension'] = \
        AdminCommand(extload, 'extload', [(None, 'filename (without .py)')], [], 'Load an extension (from extensions directory)')

    async def extunload(self: AdminCommandExecutor, name: str):
        res = await self.unload_extension(name)
        if res is None:
            self.print("%s is not loaded or non-existant extension." % name)
        elif res is False:
            self.print("%s is unloaded but failed to cleanup." % name)
        else:
            self.print("%s unloaded." % name)
    ACE.commands['extunload'] = \
        ACE.commands['unloadext'] = \
        ACE.commands['extremove'] = \
        ACE.commands['removeext'] = \
        ACE.commands['extensionremove'] = \
        ACE.commands['removeextension'] = \
        AdminCommand(extunload, 'extunload', [(None, 'name (without .py)')], [], 'Unload an extension')

    async def date(self: AdminCommandExecutor):
        date = datetime.datetime.now()
        self.print("It's %s.%s.%s-%s:%s:%s right now" % (date.day, date.month, date.year, date.hour, date.minute, date.second))
    ACE.commands['date'] = AdminCommand(date, 'date', [], [], 'Show current date')

    async def error(self: AdminCommandExecutor):
        self.error("Stderr!")
    ACE.commands['error'] = AdminCommand(error, 'error', [], [], 'Test error')


async def main():
    # scoped functions

    ACE = AdminCommandExecutor(None, use_config=False)
    await ACE.load_extensions()
    ACE.print("Plain command line")
    basic_command_set(ACE)
    await ACE.prompt_loop()
if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
