import asyncio
import traceback
from ainput import AsyncRawInput, colors
import json
import os
import importlib
import types
import datetime
import logging
from math import ceil


def str_findall_unescaped(str_: str, char: int):
    list_ = []
    _next_pos = -1
    for i in range(str_.count(chr(char))):
        _next_pos = str_.find(chr(char), _next_pos + 1)
        if str_[_next_pos - 1] != chr(92):
            list_.append(_next_pos)
    return list_


def parse_escapes(inp: str):
    res = ''
    _last = -4
    for i in range(inp.count('\\x') + 1):
        _nlast = inp.find('\\x', _last + 4)
        if _nlast == -1:
            res += inp[_last + 4:len(inp)]
            break
        res += inp[max(_last + 4, 0):_nlast]
        try:
            hexdata = inp[_nlast + 2:_nlast + 4]
            res += bytes.fromhex(hexdata).decode()
        except IndexError:
            pass
        _last = _nlast
    if res:
        return res
    else:
        return inp


def validate_convert_args(argtypes: list, args: list):  # return list if arguments are valid, int if error (index of invalid arg)
    cargs = []
    for i in range(len(args)):
        assert type(args[i]) is str, 'didn\'t you forget that you pass arguments as str because its input from the console? You just try to convert it to the type of arg...'
        try:
            cargs.append(argtypes[i](args[i]))
        except ValueError:
            return i


def paginate_list(list_: list, elemperpage: int, cpage: int) -> (int, list):
    maxpages = ceil(len(list_) / elemperpage)
    cpage = max(1, min(cpage, maxpages))
    start_at = max((cpage - 1) * elemperpage, 0)
    end_at = min(cpage * elemperpage, len(list_))
    return maxpages, list_[start_at:end_at]


def paginate_range(count: int, elemperpage: int, cpage: int) -> (int, int, int):
    maxpages = ceil(count / elemperpage)
    cpage = max(1, min(cpage, maxpages))
    start_at = max((cpage - 1) * elemperpage, 0)
    end_at = min(cpage * elemperpage, count)
    return maxpages, start_at, end_at


class AdminCommand():

    def __init__(self, afunc, name: str, args: list, optargs: list, description: str = ''):
        self.name = name
        self.args = args  # tuple (type, name)
        self.optargs = optargs  # tuple (type, name)
        # type can be str, int, float, None
        # if type is None then it is the last raw-string argument
        self.description = description
        self.afunc = afunc  # takes AdminCommandExecutor and custom args

    async def execute(self, executor, args):
        await self.afunc(executor, *args)


class AdminCommandExtension():
    def __init__(self, ACE, name: str, module: types.ModuleType, logger: logging.Logger = logging.getLogger('main')):
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
        if not self.commands:
            return False
        if overwrite:
            self.commands.update(self.commands)
            return True
        if self.commands.keys().isdisjoint(self.ace.commands.keys()):
            self.ace.commands.update(self.commands)

    def add_command(self, afunc, name: str, args: list, optargs: list, description: str = '', replace=False):
        if name in self.ace.commands:
            return False
        cmd = AdminCommand(afunc, name, args, optargs, description)
        self.commands[name] = cmd
        self.ace.commands[name] = cmd
        return True

    def remove_command(self, name: str, remove_native=False):
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
        for cmdname in self.commands:
            if cmdname in self.ace.commands:
                del self.ace.commands[cmdname]
        else:
            return False
        return True

    def msg(self, msg):
        self.ace.print('[%s] %s' % (self.name, msg))

    def logmsg(self, msg: str, level: int = logging.INFO):
        self.logger.log(1, '[%s] %s' % (self.name, msg))


class AdminCommandExecutor():
    def __init__(self, stuff: dict, use_config=True, logger: logging.Logger = None, extension_path='extensions/'):
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
        self.print = self.ainput.writeln
        self.logger = logger
        if use_config:
            self.load_config()

    def error(self, msg):
        if self.logger:
            self.logger.error(msg)
        else:
            self.ainput.writeln('ERROR: %s' % msg, fgcolor=colors.RED)

    def info(self, msg):
        if self.logger:
            self.logger.info(msg)
        else:
            self.ainput.writeln('INFO: %s' % msg)

    def log(self, msg, level=10):
        if self.logger:
            self.logger.log(level, msg)
        else:
            levels = {0: 'NOTSET', 10: 'DEBUG', 20: 'INFO', 30: 'WARNING', 40: 'ERROR', 50: 'CRITICAL'}
            self.ainput.writeln('%s: %s' % (levels[level], msg))

    async def load_extensions(self):
        extlist = []
        if os.path.exists(os.path.join(self.extpath, 'extdep.txt')):
            with open(os.path.join(self.extpath, 'extdep.txt'), 'r') as f:
                extlist.extend('%s.py' % x for x in f.readlines())
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

    def load_config(self, path: str = 'config.json'):
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
        try:
            file = open(path, 'w')
            self.config = json.dumps(self.config, indent=True)
            file.close()
            return True
        except OSError:
            self.error('Failed to save configuration at %s' % path)

    async def load_extension(self, name: str):
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
        args = []
        cmdname, _, argl = cmdline.partition(' ')
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
                if not argl:
                    self.print(self.lang['notenoughargs'] % (len(cmd.args), len(args)))
                    self.print(self.usage(cmdname))
                    return
                arg, _, argl = argl.partition(' ')
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
                arg, _, argl = argl.partition(' ')
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
        self.ainput.prepare()
        # loop = asyncio.get_event_loop()
        # loop.add_signal_handler(2, self.ctrl_c)
        while self.prompt_dispatching:
            inp = await self.ainput.prompt_line('%s%s ' % (self.promptheader, self.promptarrow))
            if not inp:
                continue
            if self.prompt_dispatching:
                self.tasks['prompt_cmd'] = asyncio.create_task(self.dispatch(inp))
                try:
                    await self.tasks['prompt_cmd']
                except asyncio.CancelledError:
                    if self.prompt_dispatching:
                        self.print("Process has been cancelled")
        self.ainput.end()
        self.print('Exit from command prompt.')

    async def full_cleanup(self):
        self.prompt_dispatching = False
        self.ainput.is_reading = False
        for func in self.full_cleanup_steps:
            await func(self)


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
