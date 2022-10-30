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
import warnings
from enum import IntEnum, EnumMeta
from math import ceil, prod
from typing import Union, Sequence, MutableSequence, Tuple, Mapping, MutableMapping, Dict, List, Set, Optional, Type, Callable, Coroutine, Any
from types import ModuleType
from collections import ChainMap, defaultdict
from aiohndchain import AIOHandlerChain
from itertools import count as itercount
from itertools import chain, repeat, dropwhile, islice
from functools import partial


class CustomType:
    """Base class inherited by all custom types for commands. Has only one predefined method that all derivative types must have: getValue()"""
    _typename = 'custom'

    def __str__(self):
        return self.getRawValue()

    def getValue(self):
        """
        Obtain a value from this type wrapper
        """
        return self._value

    def getRawValue(self) -> str:
        """
        Obtain source value that this object is initialized from
        """
        return str(self._rawvalue)

    @classmethod
    def getTypeName(cls, ace) -> str:
        """
        Obtain a name for this type that is used in command usage
        Name is obtained from AdminCommandExecutor.lang first, then from
        self._typename if not found in former
        """
        return ace.lang.get(cls, cls._typename)

    def serialize(self) -> Union[str, int, float, bool, None]:
        """
        Obtain primitive value representing this object
        Can be async
        """
        return self._value

    @classmethod
    def deserialize(cls, data):
        """
        Construct an object from this raw data
        """
        raise NotImplementedError

    @classmethod
    def tabComplete(cls, value: Optional[str] = None) -> MutableSequence[str]:
        """
        Returns a list containing all available items from a starting value
        Can be async
        """
        return list()


ArgumentType = Union[str, int, float, bool, None, CustomType]
argsplitter = re.compile('(?<!\\\\)".*?(?<!\\\\)"|(?<!\\\\)\'.*?(?<!\\\\)\'|[^ ]+')
backslasher = re.compile(r'(?<!\\)\\(.)')
allescapesplitter = re.compile(r'(\\\\|\\x[0-9a-fA-F]{2}|\\u[0-9a-fA-F]{4}|\\[0-7]{1,3}|\\[abfnrtv"\'])')
octal = re.compile(r'\\[0-7]{1,3}')
quoted = re.compile(r'(?P<quote>["\']).*?(?P=quote)')
hexescapefinder = re.compile('(?<!\\\\)\\\\x[0-9a-fA-F]{2}')
zerostr = '0'
single_char_escaper = {
    "a": "\a",
    "b": "\b",
    "f": "\f",
    "n": "\n",
    "r": "\r",
    "t": "\t",
    "v": "\v",
    '"': '"',
    "'": "'",
    " ": " "
}
default_lang = {
    'date_fmt': '%m/%d/%y',
    'time_fmt': '%H:%M:%S',
    'datetime_fmt': '%a %b %e %H:%M:%S %Y',
    'nocmd': '{}: unknown command',
    'usage': 'Usage: {}',
    'invalidarg': '{} is invalid, check your command arguments.',
    'toomanyargs': 'warning: this command receives {0} arguments, you provided {1} or more',
    'notenoughargs': 'not enough arguments: the command receives {0} arguments, you provided {1}.',
    'cancelled': "Process has been cancelled",
    'command_error': "Command execution failed: {}",
    'tab_error': "Tabcomplete failed: {}",
    'help_cmd': 'Help (page {0} out of {1})\n{2}',
    'extensions_cmd': 'Extensions (page {0} of {1}):\n{2}',
    'extension_fail': '{} failed to load.',
    'extension_loaded': '{} loaded.',
    'extension_notfound': '{} is not loaded.',
    'extension_failcleanup': '{} unloaded but failed to cleanup.',
    'extension_unloaded': '{} unloaded.'
}
default_types = {
    str: 'word',
    int: 'n',
    float: 'n.n',
    bool: 'yes / no',
    None: 'text...',
}


class InvalidArgument(ValueError):
    """Raised by AdminCommandExecutor.parse_args() when one of the arguments cannot be parsed"""


class NotEnoughArguments(Exception):
    """Raised by AdminCommandExecutor.dispatch() when command string doesn't have enough arguments"""


# Additional derivative types for commands
class BaseDiscreteScale(int, CustomType):
    """Base class for defining discrete scale types. An inheritor must either implement getMin, getMax, getStep methods or
    set self._min, self._max and self._step variables accordingly.
    Default value for _step is 1"""
    _typename = "{0}<=n<={1}"
    _step = 1

    def __init__(self, value, /, *args, **kwargs):
        self._rawvalue = value
        if self._step == 0:
            raise ValueError("_step must not be zero")
        num = super(int).__init__(value, *args, **kwargs)
        _start, _end, _step = self.getMin(), self.getMax(), self.getStep()
        if num not in range(_start, _end, _step):
            raise InvalidArgument(f"an integer must be in range between {_start} and {_end} by +{_step} each")
        self._value = num

    def getValue(self) -> int:
        return self._value

    @classmethod
    def getTypeName(self, cmd) -> str:
        return super(CustomType).getTypeName(cmd).format(self.getMin(), self.getMax())

    @classmethod
    def getMin(cls) -> int:
        """Returns the lowest point of a scale"""
        return cls._min

    @classmethod
    def getMax(cls) -> int:
        """Returns the highest point of a scale"""
        return cls._max

    @classmethod
    def getStep(cls) -> int:
        """Returns the distance between each elements"""
        return cls._step

    @classmethod
    def tabComplete(cls, value: Optional[str] = None):
        return list(range(cls.getMin(), cls.getMax() + 1, cls.getStep()))


class FixedEnumType(BaseDiscreteScale):
    """
    Base class for defining a discrete scale of constant certain objects (e.g. words, literal phrases)
    This class must be subclassed with cls._enum of type IntEnum, An enum must be ordered and start from 0.
    If an enum starts from a number other than 0, specify keyword start_at with
    super().__init__(self, value: str, *args, start_at=N)
    """
    _typename = 'literal'

    def __init__(self, value: Optional[str] = None, *args, from_instance: Optional[IntEnum] = None, start_at: int = 0, **kwargs):
        self._enum: EnumMeta
        self._enuminstance: IntEnum = getattr(self._enum, value) if from_instance is not None else from_instance
        self._value = self._enuminstance.value
        self._min = start_at
        self._max = max(len(self._enum) - 1 + start_at, self._value)

    def getEnum(self):
        """
        Returns an underlying Enum instance
        """
        return self._enuminstance

    def getRawValue(self):
        return self._enuminstance.name

    @staticmethod
    def _startswith_predicate(value: str, item: IntEnum):
        return item.name.startswith(value)

    @classmethod
    def tabComplete(cls, value: str):
        return list(filter(partial(cls._startswith_predicate, value), iter(cls._enum)))


class BaseContinuousScale(float, CustomType):
    """
    Base class for defining types that are based on float, but has boundaries
    Unlike BaseDiscreteScale, this type doesn't have _step attribute as it is continuous.
    """
    _typename = '{0: .2f}<=n.n<={1: .2f}'
    _step = 1

    def __init__(self, value, /, *args, **kwargs):
        if self._step == 0:
            raise ValueError("_step must not be zero")
        num = super(float).__init__(value, *args, **kwargs)
        _start, _end, _step = self.getMin(), self.getMax(), self.getStep()
        if num not in range(_start, _end, _step):
            raise InvalidArgument("an integer must be in range between {0} and {1} by +{2} each".format(
                                  _start, _end, _step))
        self._value = num

    def getValue(self) -> float:
        return self._value

    @classmethod
    def getTypeName(cls, ace) -> str:
        return cls.getTypeName(ace).format(cls.getMin(), cls.getMax())

    @classmethod
    def getMin(cls) -> float:
        """Returns the lowest point of a scale"""
        return cls._min

    @classmethod
    def getMax(cls) -> float:
        """Returns the highest point of a scale"""
        return cls._max


class DateType(CustomType):
    date_expr = re.compile(
        r"(?P<year>\d{4})(?P<date_delimiter>[/.\-_ ])(?P<month>\d{1,2})(?P=date_delimiter)(?P<day>\d{1,2})"
    )
    _typename = 'date Y/m/d'
    date_reverse_expr = "%Y/%m/%d"

    def __init__(self, value: Optional[str] = None, *, raise_exc=True, from_date: Optional[datetime.date] = None):
        if from_date is not None:
            self._value = from_date
        else:
            self._value = self.parse(value, raise_exc=raise_exc)

    @classmethod
    def parse(cls, value: str, *, raise_exc=True):
        _data = cls.date_expr.fullmatch(value)
        return datetime.date(
            year=(int(_data.group('year')) if _data.group('year') is not None else 0),
            month=(int(_data.group('month')) if _data.group('month') is not None else 0),
            day=(int(_data.group('day')) if _data.group('day') is not None else 0)
        )

    @classmethod
    def tabComplete(cls, value: str):
        return [value + datetime.datetime.utcnow().strftime(cls.date_reverse_expr)]


class TimeType(CustomType):
    time_expr = re.compile(
        r"(?P<hour>\d{1,2})(?P<time_delimiter>[:.\-_ ])(?P<minute>\d{1,2})(?:(?P=time_delimiter)(?P<second>\d{1,2}))?"
    )
    _typename = 'time H:M:S'
    time_reverse_expr = "%H:%M:%S"

    def __init__(self, value: Optional[str] = None, *, raise_exc=True, from_time: Optional[datetime.time] = None):
        if from_time is not None:
            self._value = from_time
        else:
            self._value = self.parse(value, raise_exc=raise_exc)

    @classmethod
    def parse(cls, value: str, *, raise_exc=True):
        _data = cls.time_expr.fullmatch(value)
        return datetime.time(
            hour=(int(_data.group('hour')) if _data.group('hour') is not None else 0),
            minute=(int(_data.group('minute')) if _data.group('minute') is not None else 0),
            second=(int(_data.group('second')) if _data.group('second') is not None else 0)
        )

    def serialize(self):
        return self._value.second() + self._value.minute() * 60 + self._value.hour() * 3600

    @classmethod
    def deserialize(cls, data: int):
        # data is seconds
        _hour = data // 3600
        _r = data % 3600
        _minute = _r // 60
        _r %= 60
        return cls(from_time=datetime.time(_hour, _minute, _r))

    def tabComplete(cls, value: str):
        return [value + datetime.datetime.utcnow().strftime(cls.time_reverse_expr)]


class DateTimeType(CustomType):
    datetime_expr = re.compile(
        f"(?:{DateType.date_expr.pattern})?(?: *|[._+-]?)"
        f"(?:{TimeType.time_expr.pattern})?"
    )
    _typename = 'date+time Y/m/d H:M:S'
    datetime_reverse_expr = f"{DateType.date_reverse_expr}_{TimeType.time_reverse_expr}"

    def __init__(self, value: Optional[str] = None, *, raise_exc=True, from_datetime: Optional[datetime.datetime] = None):
        if from_datetime is not None:
            self._value = from_datetime
        else:
            self._value = self.parse(value, raise_exc=raise_exc)

    @classmethod
    def parse(cls, value: str, *, raise_exc=True):
        _data = cls.datetime_expr.fullmatch(value)
        return datetime.datetime(
            year=(int(_data.group('year')) if _data.group('year') is not None else 0),
            month=(int(_data.group('month')) if _data.group('month') is not None else 0),
            day=(int(_data.group('day')) if _data.group('day') is not None else 0),
            hour=(int(_data.group('hour')) if _data.group('hour') is not None else 0),
            minute=(int(_data.group('minute')) if _data.group('minute') is not None else 0),
            second=(int(_data.group('second')) if _data.group('second') is not None else 0)
        )

    def serialize(self):
        return int(self._value.timestamp())

    @classmethod
    def deserialize(cls, timestamp: int):
        return cls(from_datetime=datetime.datetime.fromtimestamp(float(timestamp)))

    @classmethod
    def tabComplete(cls, value: str):
        return [value + datetime.datetime.utcnow().strftime(cls.datetime_reverse_expr)]


class DurationType(CustomType):
    duration_expr = re.compile(
        r"(?:(?P<yrs>\d+) *y(?:(?:ea)?r)?s?)?{0} *"
        r"(?:(?P<mon>\d+) *(?:mon|M)(?:th)?s?)?{0} *"
        r"(?:(?P<wks>\d+) *w(?:(?:ee)?k)?s?)?{0} *"
        r"(?:(?P<ds>\d+) *d(?:ay)?s?)?{0} *"
        r"(?:(?P<hrs>\d+) *h(?:(?:ou)?rs?)?)?{0} *"
        r"(?:(?P<min>\d+) *m(?:in(?:ute)?s?)?)?{0} *"
        r"(?:(?P<sec>\d+) *s(?:ec(?:ond)?)?s?)?{0} *"
        r"(?:(?P<msec>\d+) *m(?:illi)?s(?:ec(?:ond)?)?s?)?{0} *"
        r"(?:(?P<mcsec>\d+) *(?:mi?cr?o?|u)s(?:ec(?:ond)?)?s?)? *"
        "".format(r"(?:[.,]| *and)?")
    )  # ---------------------------------------------------------
    _mul = (
        31557600,
        2629800,
        604800,
        86400,
        3600,
        60,
        1,
    )
    _typename = 'duration Ns,Nm,Nh...'

    def __init__(self, value: Optional[str] = None, *, raise_exc=True, from_timedelta: datetime.timedelta = None):
        if from_timedelta is not None:
            self._value = from_timedelta
        else:
            self._value = self.parse(value, raise_exc=raise_exc)

    @classmethod
    def parse(cls, value: str, *, raise_exc=True):
        _match = cls.duration_expr.fullmatch(value)
        if _match is None:
            if raise_exc:
                raise ValueError("invalid duration provided")
        _group_iter = (0 if num is None else int(num) for num in _match.groups())
        _seconds = sum(map(prod, zip(_group_iter, cls._mul)))
        _msec = next(_group_iter, 0)
        _usec = next(_group_iter, 0)
        return datetime.timedelta(
            seconds=_seconds,
            milliseconds=_msec,
            microseconds=_usec
        )

    def serialize(self):
        return self._value / datetime.timedelta(microseconds=1)

    @classmethod
    def deserialize(cls, usec: int):
        return cls(from_timedelta=datetime.timedelta(microseconds=usec))


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
    def __init__(self, afunc, name: str, args: Sequence[Tuple[ArgumentType, str]], optargs: Sequence[Tuple[ArgumentType, str]], description: str = '', atabcomplete: Optional[Callable[[str], Coroutine[Any, Any, Any]]] = None):
        """
        Parameters
        ----------
        afunc : coroutine function
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
        atabcomplete : coroutine function
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
        self.argchain = tuple(chain(args, optargs))

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

    async def tab_complete(self, executor, args: Sequence[object], argl: str = '') -> Union[MutableSequence[str], None]:
        """Shouldn't be overriden, use atabcomplete to assign a tab complete handler"""
        async with executor.cmdtab_event.emit_and_handle(self, executor, args, {'argl': argl, 'atabcomplete': self.atabcomplete}) as handle:
            try:
                _res, _args, _kwargs = handle()
                if 'override' in _kwargs:
                    return _kwargs['override']
                elif _res:
                    _len = len(args)
                    if argl or not _len:
                        # its when next command is tabcompleted
                        _len += 1
                        _last_type = self.argchain[_len - 1][0]
                        if _last_type in (None, str):
                            _last = ''
                        elif _last_type in (int, float):
                            _last = _last_type()
                        elif issubclass(_last_type, CustomType):
                            _last = None
                        _rlast = ''
                    else:
                        # already existing one
                        _last_type = self.argchain[_len - 1][0]
                        _last = args[-1]
                        if isinstance(_last, (int, float, str, bool)):
                            _rlast = str(_last)
                        elif isinstance(_last, CustomType):
                            _rlast = str(_last.getRawValue())
                    if self.atabcomplete is not None:
                        if asyncio.iscoroutinefunction(self.atabcomplete):
                            _res = await self.atabcomplete(executor, *args, argl=argl)
                            if _res:
                                return _res
                        else:
                            _res = self.atabcomplete(executor, *args, argl=argl)
                            if _res:
                                return _res
                    if issubclass(_last_type, CustomType):
                        # attempt to tabcomplete that one
                        if asyncio.iscoroutinefunction(_last_type.tabComplete):
                            return await _last_type.tabComplete(_rlast)
                        else:
                            return _last_type.tabComplete(_rlast)
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

    def add_command(self, afunc: Callable[[Any], Coroutine[Any, Any, Any]], name: str, args: Sequence[Tuple[ArgumentType, str]] = tuple(), optargs: Sequence[Tuple[ArgumentType, str]] = tuple(), description: str = '', atabcomplete: Optional[Callable[[str], Coroutine[Any, Any, Any]]] = None, replace=False) -> bool:
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
        cmd = AdminCommand(afunc, name, args, optargs, description, atabcomplete)
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
        self.ace.print('[{}] {}'.format(self.name, msg))

    def logmsg(self, msg: str, level: int = logging.INFO):
        """Write a message into the log"""
        self.logger.log(level, '[{}] {}'.format(self.name, msg))


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
            'nocmd': '{0}: unknown command',
            'usage': 'Usage: {0}',
            'invalidarg': '{0} is invalid, check your command arguments.',
            'toomanyargs': 'warning: this command receives {0} arguments, you provided {1} or more',
            'notenoughargs': 'not enough arguments: the command receives {0} arguments, you provided {1}.'
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
    self.tab_complete_seq = tuple()
        Contains last argument suggestions on tab complete call
    self.tab_complete_id = 0
        Contains currently cycled element ID in self.tab_complete_seq

    Events
    self.events : collections.defaultdict(AIOHandlerChain)
        Main pool of events. Can be used to store custom events.
    self.tab_event : AIOHandlerChain
        Arguments: (executor: Union[AdminCommandExecutor, AdminCommandEWrapper], suggestions: MutableSequence[str])
        Emits when user hits the TAB key without full command. Cancellable
        Event handlers must modify the suggestions list if needed. Otherwise, if event emission fails,
        this list is cleared and no results are shown
    self.cmdexec_event : AIOHandlerChain
        Arguments: (cmd: AdminCommand, executor: Union[AdminCommandExecutor, AdminCommandEWrapper], args: Sequence[Any])
        Emits when a command is executed through specific executor and with parsed arguments. Cancellable.
    self.cmdtab_event : AIOHandlerChain
        Arguments: (cmd: AdminCommand, executor: Union[AdminCommandExecutor, AdminCommandEWrapper], args: Sequence[Any], *, argl: str = "")
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
    def __init__(self, stuff: Optional[Mapping] = None, use_config=True, logger: logging.Logger = None, extension_path='extensions/', backend: Optional[AsyncRawInput] = None):
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
        self.lang: Union[Dict[str, str], ChainMap] = ChainMap({}, default_lang)
        self.types: Union[Dict[ArgumentType, str], ChainMap] = ChainMap({}, default_types)
        self.tasks: Dict[str, asyncio.Task] = {
            # 'task_name': asyncio.Task()
        }
        self.extensions: Dict[str, AdminCommandExtension] = {
            # 'extension name': AdminCommandExtension()
        }
        self.known_modules: Dict[str, ModuleType] = {}
        self.full_cleanup_steps: Set[Callable[[Any], Coroutine[Any, Any, Any]]] = set(
            # coroutine functions
        )
        self.extpath = extension_path
        self.prompt_dispatching = True
        self.promptheader = 'nothing'
        self.promptarrow = '>'
        self.history: List[str] = []
        if backend is None:
            self.ainput = AsyncRawInput(history=self.history)
        elif backend:
            self.ainput = backend
        self.ainput.ctrl_c = self.full_cleanup
        self.prompt_format = {'bold': True, 'fgcolor': colors.GREEN}
        self.input_format = {'fgcolor': 10}
        self.logger = logger
        self.tab_complete_lastinp = ''
        self.tab_complete_cursor: Optional[int] = None
        self.tab_complete_seq: Union[Sequence[str], MutableSequence[str]] = tuple()
        self.tab_complete_slices: MutableSequence[re.Match] = []
        self.tab_complete_argsfrom: Optional[int] = None
        self.tab_complete_id = 0
        # events
        self.events = defaultdict(lambda: AIOHandlerChain())
        self.events['tab_event'] = self.tab_event = AIOHandlerChain()
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

    def error(self, msg: str, log=True):
        """Shows a red error message in the console and logs.
        ERROR: msg

        Parameters
        ----------
        msg : str
            Message"""
        if self.logger and log:
            self.logger.error(msg)
        else:
            self.ainput.writeln('ERROR: {}'.format(msg), fgcolor=colors.RED)

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
            self.ainput.writeln('INFO: {}'.format(msg))

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
            self.ainput.writeln('{}: {}'.format(levels[level], msg))

    async def load_extensions(self):
        """
        Loads extensions from an extension directory specified in AdminCommandExecutor.extpath
        """
        extlist = []
        _initpy = '__init__.py'
        if os.path.exists(os.path.join(self.extpath, 'extdep.txt')):
            with open(os.path.join(self.extpath, 'extdep.txt'), 'r') as f:
                for x in f.readlines():
                    # Light vulnerability fix: path traversal
                    if '../' in x or x.startswith('/'):
                        continue
                    extlist.append('{}.py'.format(x.strip()))
        else:
            self.print('Note: create extdep.txt in the extensions folder to sequentally load modules')
        if not os.path.exists(self.extpath):
            try:
                os.makedirs(self.extpath)
            except OSError as exc:
                self.error('Failed to create extension directory: {}: {}'.format(type(exc).__name__, exc))
                return
        with os.scandir(self.extpath) as extpath:
            for file in extpath:
                if file.name.endswith('.py') and file.is_file() and file.name not in extlist and file.name != _initpy:
                    extlist.append(file.name)
        for name in extlist:
            if not os.path.exists(os.path.join(self.extpath, name)):
                self.error(f'Module file {name} not found')
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
            with open(path, 'r') as file:
                self.info('')
                self.config = json.loads(file.read())
            return True
        except (json.JSONDecodeError, OSError):
            self.error("Error occurred during load of the config: \n{}".format(traceback.format_exc()))
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
            with open(path, 'w') as file:
                file.write(json.dumps(self.config, indent=True))
            return True
        except OSError:
            self.error(f'Failed to save configuration at {path}')
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

        Note
        ----
        Due to the Python's importlib limitation, extensions cannot "import" other extensions properly
        and that way can't be implemented in any non-destructible way. Temporary modifying sys.modules
        may cause interference between multiple ACE instances (possible name collisions in
        sys.modules) and race conditions in multithreaded application.
        The only way one extension can make use of another is through AdminCommandExecutor.extensions['name'].*
        Extension objects must not be stored anywhere except by the AdminCommandExecutor owning it.
        Otherwise the garbage collection won't work and the extension wouldn't unload properly.
        Also, the Python module cannot be unloaded from the memory, in that particular case, the modules
        are cached, and if the same extension loads again, its module is taken from the known_modules and
        reloaded.
        """
        async with self.extload_event.emit_and_handle(name, before=False) as handle:
            if not handle()[0]:
                return False
            if name in self.extensions:
                self.error(f"Failed to load extension {name}: This extension is already loaded.")
                handle(False)
                return False
            _path = os.path.join(self.extpath, name + '.py')
            # Light vulnerability fix: path traversal
            if '../' in _path or os.path.isabs(_path):
                self.error(f"Failed to load extension {name}: path should not be absolute or contain \"..\"")
                handle(False)
                return False
            _modulename = os.path.relpath(os.path.join(self.extpath, name)).replace(os.sep, '.')
            try:
                if _modulename in self.known_modules:
                    module = importlib.reload(self.known_modules[_modulename])
                else:
                    module = importlib.import_module(_modulename)
                    self.known_modules[_modulename] = module
                if 'extension_init' not in module.__dict__ or 'extension_cleanup' not in module.__dict__:
                    self.error(f"Cannot initialize {name}: missing extension_init or extension_cleanup")
                    handle(False)
                    return
                extension = AdminCommandExtension(self, name, module, logger=self.logger)
                self.info(f'Loading extension {name}')
                await module.extension_init(extension)
                self.extensions[name] = extension
                self.commands.maps.insert(0, extension.commands)
                return True
            except Exception:
                self.error(f"Failed to load extension {name}:\n{traceback.format_exc()}")
                handle(False)
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

        Note
        ----
        Due to a Python limitation, it is impossible to unload a module from memory, but
        its possible to reload them with importlib.reload()
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
                # _name = extension.module.__name__
                # if _name in sys.modules:
                #     if sys.modules[_name] == extension:
                #         del sys.modules[_name]
                return True
            except Exception:
                self.error(f"Failed to call cleanup in {name}:\n{traceback.format_exc()}")
                handle(False)
                return False

    def parse_args(self, argl: str, argtypes: Sequence[Tuple[ArgumentType, str]] = None, opt_argtypes: Sequence[Tuple[ArgumentType, str]] = None, *, raise_exc=True, saved_matches: Optional[MutableSequence[re.Match]] = None, until: Optional[int] = None) -> (list, str):
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

        Keyword-only parameters:
        raise_exc : bool
            Whether or not exceptions are raised. If False, data is returned as it is
        saved_matches : list
            If specified, the parser will append the re.Match objects for each argument
            to this list except the last NoneType argument (if present in args),
            which does not have any boundaries
        until : int
            If specified, will stop parsing arguments when passed more than n symbols.
            Useful when passing sliced string isn't good due to storing argument boundaries
            in saved_matches.

        Returns
        -------
        list, str
            A list of converted arguments and the remnant
            If remnant isn't an empty string, then there is probably too many arguments provided
        """
        args = []
        remnant = argl
        arg_iterator = chain(argsplitter.finditer(argl), repeat(None))
        if argtypes:
            for argdata, argmatch in zip(argtypes, arg_iterator):
                argtype, argname = argdata
                if argtype is None:
                    args.append(parse_escapes(remnant.lstrip()))
                    remnant = ''
                    break
                if not argmatch:
                    if raise_exc:
                        raise NotEnoughArguments(len(argtypes), len(args))
                    else:
                        return args, remnant
                if saved_matches is not None:
                    saved_matches.append(argmatch)
                if until is not None:
                    _argend = min(argmatch.end(), until)
                else:
                    _argend = argmatch.end()
                if _argend <= argmatch.start():
                    return args, ''
                arg = argl[argmatch.start():_argend]
                remnant = argl[_argend:]
                if quoted.fullmatch(arg) is not None:
                    arg = arg[1:-1]
                try:
                    if issubclass(argtype, bool):
                        args.append(True if arg.lower() in ['true', 'yes', 'y', '1'] else False)
                    elif issubclass(argtype, str):
                        args.append(parse_escapes(arg))
                    else:
                        args.append(argtype(arg))
                except ValueError:
                    if raise_exc:
                        raise InvalidArgument(argname)
                    else:
                        args.append(arg)
                        return args, remnant
        if opt_argtypes:
            # continue iterating over the same arguments list
            for argdata, argmatch in zip(opt_argtypes, arg_iterator):
                argtype, argname = argdata
                if argtype is None and remnant:
                    args.append(parse_escapes(remnant.lstrip()))
                    remnant = ''
                    break
                elif not remnant:
                    break
                if argmatch is None:
                    return args, remnant
                if saved_matches is not None:
                    saved_matches.append(argmatch)
                if until is not None:
                    _argend = min(argmatch.end(), until)
                else:
                    _argend = argmatch.end()
                if _argend <= argmatch.start():
                    return args, ''
                arg = argl[argmatch.start():_argend]
                remnant = argl[_argend:]
                if quoted.fullmatch(arg) is not None:
                    arg = arg[1:-1]
                try:
                    if issubclass(argtype, bool):
                        args.append(True if arg.lower() in ['true', 'yes', 'y', '1'] else False)
                    elif issubclass(argtype, str):
                        args.append(parse_escapes(arg))
                    else:
                        args.append(argtype(arg))
                except ValueError:
                    if raise_exc:
                        raise InvalidArgument(argname)
                    else:
                        args.append(arg)
                        return args, remnant
        return args, remnant

    async def dispatch(self, cmdline: str) -> bool:
        """Executes a command. Shouldn't be used explicitly with input(). Use prompt_loop() instead.

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
            self.print(self.lang['nocmd'].format(cmdname))
            return
        cmd: Union[AdminCommand, None] = self.commands[cmdname]
        if cmd is None or cmd is self.disabledCmd:
            self.print(self.lang['nocmd'].format(cmdname))
            return
        self.ainput.ctrl_c = self.interrupt_command
        try:
            args, argl = self.parse_args(argl, cmd.args, cmd.optargs)
            if argl:
                self.print(self.lang['toomanyargs'].format(len(cmd.args) + len(cmd.optargs), len(args) + 1))
                self.print(self.usage(cmdname))
            await cmd.execute(self, args)
        except NotEnoughArguments as exc:
            self.print(self.lang['notenoughargs'].format(len(cmd.args), exc.args[0]))
            self.print(self.usage(cmdname))
        except InvalidArgument as exc:
            self.print(self.lang['invalidarg'].format(exc.args[0]))
            self.print(self.usage(cmdname))
        except Exception as exc:
            self.print("An exception just happened in this command, check logs or smth...")
            self.error(self.lang['command_error'].format(exc))
        finally:
            self.ainput.ctrl_c = self.full_cleanup

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
            mandatory_args = ['<{}: {}>'.format(x[1], self.types[x[0]] if x[0] in self.types else x[0].getTypeName(cmd)) for x in cmd.args]
        else:
            mandatory_args = ''
        if cmd.optargs:
            optional_args = ['[{}: {}]'.format(x[1], self.types[x[0]] if x[0] in self.types else x[0].getTypeName(cmd)) for x in cmd.optargs]
        else:
            optional_args = ''
        usage = '{} {} {}'.format(cmdname, ' '.join(mandatory_args), ' '.join(optional_args))
        if lang:
            return self.lang['usage'].format(usage)
        else:
            return usage

    def interrupt_command(self):
        """
        Cancels the command execution task.
        Ctrl + C is replaced during any command being executed.
        """
        if 'prompt_cmd' not in self.tasks:
            return
        if not self.tasks['prompt_cmd'].done():
            self.tasks['prompt_cmd'].cancel()

    async def cmd_interruptor(self):
        """
        Listens for Ctrl + C event during command execution.
        Cancels the command execution task if Ctrl + C pressed.
        """
        if 'prompt_cmd' not in self.tasks:
            return
        while not self.tasks['prompt_cmd'].done():
            # at this moment, if command prompts the user for something,
            # this function is interrupted and therefore exception
            # is thrown
            key = await self.ainput.prompt_keystroke('', echo=False)
            if key == '\3':
                self.tasks['prompt_cmd'].cancel()
                return

    def spawn_interruptor(self):
        """
        Spawn a listener for command cancellation keystroke.
        Useful when a command executes for an indefinite time and
        lets the user to bring back the prompt in emergency cases

        New in version 1.4.2
        """
        if 'cmd_interruptor' in self.tasks:
            if not self.tasks['cmd_interruptor'].done():
                self.tasks['cmd_interruptor'].cancel()
        self.tasks['cmd_interruptor'] = asyncio.create_task(self.cmd_interruptor())

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
            inp = await self.ainput.prompt_line('{}{} '.format(self.promptheader, self.promptarrow), prompt_formats=self.prompt_format, input_formats=self.input_format)
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

    async def _tab_predicate(self, cmdname: str) -> bool:
        """
        Must return True if this command can be tabcompleted
        """
        return True

    async def _tab_complete(self):
        """This is callback function for TAB key event
           Uses AsyncRawInput data to handle the text update
        """
        whole_inp = ''.join(self.ainput.read_lastinp)
        inp_beginning = whole_inp[0:self.ainput.cursor]
        rest = whole_inp[self.ainput.cursor:]
        if not inp_beginning or not self.prompt_dispatching:
            return
        if self.tab_complete_lastinp != whole_inp or self.tab_complete_cursor != self.ainput.cursor:
            self.tab_complete_slices.clear()
            # generate new suggestions and append or replace the argument
            cmdname, sep, argl = whole_inp.partition(' ')
            self.tab_complete_argsfrom = len(cmdname) + 1
            if not sep:
                # generate list of command suggestions
                self.tab_complete_seq = []
                for cmdname in self.commands:
                    if cmdname.startswith(inp_beginning) and self.commands[cmdname] is not self.disabledCmd and (await self._tab_predicate(cmdname)):
                        self.tab_complete_seq.append(cmdname)
                async with self.tab_event.emit_and_handle(self, self.tab_complete_seq) as handle:
                    _res, _args, _kwargs = handle()
                    if _res and self.tab_complete_seq:
                        self.ainput.writeln('\t'.join(self.tab_complete_seq), fgcolor=colors.GREEN)
                        self.tab_complete_id = 0
                    elif self.tab_complete_seq:
                        self.tab_complete_seq.clear()
            else:
                argl = argl.lstrip()
                if cmdname not in self.commands:
                    return
                cmd: AdminCommand = self.commands[cmdname]
                if cmd is self.disabledCmd:
                    return
                try:
                    args, argl = self.parse_args(argl, cmd.args, cmd.optargs, raise_exc=False,
                                                 saved_matches=self.tab_complete_slices,
                                                 until=self.ainput.cursor - self.tab_complete_argsfrom)
                    suggestions = await cmd.tab_complete(self, args, argl)
                    if suggestions is not None and suggestions:
                        self.ainput.writeln('\t'.join(suggestions), fgcolor=colors.GREEN)
                    self.tab_complete_seq = suggestions
                    self.tab_complete_id = 0
                except asyncio.CancelledError:
                    if self.prompt_dispatching:
                        self.print(self.lang['cancelled'])
                except InvalidArgument as exc:
                    # one of the arguments are invalid
                    self.print(self.lang['invalidarg'].format(exc.args[0]))
                    self.tab_complete_seq = tuple()
                except Exception as exc:
                    # other error
                    self.error(self.lang['tab_error'].format(str(exc)))
                    self.tab_complete_seq = tuple()
            self.tab_complete_lastinp = whole_inp
            self.tab_complete_cursor = self.ainput.cursor
        elif self.tab_complete_seq:
            # cycle through the suggestions
            self.tab_complete_id = (self.tab_complete_id + 1) % len(self.tab_complete_seq)
        # replace the n-th argument
        if self.tab_complete_seq:
            # pick the boundaries
            _bounds, _bound_id = next(
                dropwhile(self._cursor_boundary_predicate,
                          zip(self.tab_complete_slices, itercount())),
                (None, None)
            )
            _bounds: Optional[re.Match]
            _bound_id: Optional[int]
            _, _, last_arg = inp_beginning.rpartition(' ')
            target = self.tab_complete_seq[self.tab_complete_id]
            if not last_arg:
                # append the resulting argument
                self.ainput.read_lastinp.extend(target)
                self.tab_complete_cursor = self.ainput.cursor = len(self.ainput.read_lastinp)
            elif _bounds is not None:
                # replace the resulting argument
                # self.ainput.read_lastinp[-len(last_arg) - len(rest):] = target
                self.ainput.read_lastinp[_bounds.start() + self.tab_complete_argsfrom:
                                         _bounds.end() + self.tab_complete_argsfrom] = target
                self.tab_complete_cursor = self.ainput.cursor = _bounds.start() + self.tab_complete_argsfrom + len(target)
            else:
                self.ainput.read_lastinp[-len(last_arg) - len(rest):] = target
                self.tab_complete_cursor = self.ainput.cursor = len(self.ainput.read_lastinp)
            self.ainput.redraw_lastinp(1)
            whole_inp = ''.join(self.ainput.read_lastinp)
            self.tab_complete_lastinp = whole_inp
            if _bounds is not None:
                # also, after replacing the argument, reparse all the stuff
                _amount = len(self.tab_complete_slices)
                del self.tab_complete_slices[_bound_id:]
                self.tab_complete_slices.extend(
                    islice(argsplitter.finditer(whole_inp[self.tab_complete_argsfrom:], _bounds.start()), _amount)
                )

    def _cursor_boundary_predicate(self, id_match: Tuple[re.Match, int]) -> bool:
        match_, _ = id_match
        _argpos = self.ainput.cursor - self.tab_complete_argsfrom
        return match_.start() < _argpos and _argpos > match_.end()


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
        self.ace.log(f"add_keystroke: attempt to add a keystroke {repr(keystroke)} handler, which is not supported by a fake ARI.")

    def remove_keystroke(self, keystroke: str):
        """Remove key press event"""
        self.ace.log(f"remove_keystroke: attempt to remove a keystroke {repr(keystroke)} handler, which is not supported by a fake ARI.")

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
            return object.__getattribute__(self, name)
        except AttributeError:
            return getattr(object.__getattribute__(self, 'master'), name)

    def __setattr__(self, name: str, value, /):
        master = object.__getattribute__(self, 'master')
        if name in object.__getattribute__(self, '__dict__'):
            object.__setattr__(self, name, value)
        else:
            setattr(master, name, value)

    def override(self, name: str, value):
        """Change the member of the class, but apply it only for the wrapper, not the real command executor
        For example, replace ainput with something else if the command needs to query input and show messages
        in a different way, rather than doing this in the main console"""
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
            usagedesc.append('| {} -> {}'.format(self.usage(cmdname, False), cmd.description))
        maxpage, list_ = paginate_list(usagedesc, 6, cpage)
        self.print(self.lang['help_cmd'].format(cpage, maxpage, '\n'.join(list_)))
    commands_['help'] = commands_['?'] = \
        AdminCommand(help_, 'help', [], [(int, 'page')], 'Show all commands')

    async def extlist(self: AdminCommandExecutor, cpage: int = 1):
        ls = list(self.extensions.keys())
        maxpage, pgls = paginate_list(ls, 7, cpage)
        cpage = max(min(maxpage, cpage), 1)
        self.print(self.lang['extensions_cmd'].format(cpage, maxpage, '\n'.join(pgls)))
    commands_['extlist'] = \
        commands_['extls'] = \
        commands_['lsext'] = \
        commands_['exts'] = \
        commands_['extensions'] = \
        AdminCommand(extlist, 'extlist', [], [(int, 'page')], 'List loaded extensions')

    async def extload(self: AdminCommandExecutor, name: str):
        if not await self.load_extension(name):
            self.print(self.lang['extension_fail'].format(name))
            return
        else:
            self.print(self.lang['extension_loaded'].format(name))
    commands_['extload'] = \
        commands_['extensionload'] = \
        commands_['loadextension'] = \
        AdminCommand(extload, 'extload', [(None, 'filename (without .py)')], [], 'Load an extension (from extensions directory)')

    async def extunload(self: AdminCommandExecutor, name: str):
        res = await self.unload_extension(name)
        if res is None:
            self.print(self.lang['extension_notfound'].format(name))
        elif res is False:
            self.print(self.lang['extension_failcleanup'].format(name))
        else:
            self.print(self.lang['extension_unloaded'].format(name))

    async def extunload_tab(self: AdminCommandExecutor, *args, argl: str):
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
            self.print(self.lang['extension_notfound'].format(name))
        elif res is False:
            self.print(self.lang['extension_failcleanup'].format(name))
        else:
            self.print(self.lang['extension_unloaded'].format(name))
            if not await self.load_extension(name):
                self.print(self.lang['extension_fail'].format(name))
                return
            else:
                self.print(self.lang['extension_loaded'].format(name))
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
        self.print(date.strftime(self.lang['datetime_fmt']))
    commands_['date'] = AdminCommand(date, 'date', [], [], 'Show current date')

    async def error(self: AdminCommandExecutor):
        self.error("Stderr!")
    commands_['error'] = AdminCommand(error, 'error', [], [], 'Test error')

    async def testoptargs(self: AdminCommandExecutor, mand1: str, opt1: int = 0, opt2: str = '', opt3: int = 0):
        self.print(mand1, opt1, opt2, opt3)

    async def testoptargs_tab(self: AdminCommandExecutor, *args, argl: str):
        _len = len(args)
        if argl:
            _len += 1
        self.print('Arguments: ({})'.format(', '.join(str(x) for x in args)))
        self.print(f'Argument amount: {_len}')
        self.print(f'Remnant: "{argl}"')
        if _len <= 1:
            return "example", "another", "main", "obsolete"
        elif _len == 2:
            return '1', '2', '3', '4'
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
