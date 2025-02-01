"""
ainput.py
Tools for creating a simple text user interfaces in Unix-like systems with using asyncio
Doesn't support Windows or any non-POSIX terminals
"""

import asyncio
import os
import sys
from bisect import bisect_left
from operator import itemgetter
from enum import Enum
from typing import Union, Sequence, Tuple, MutableSequence, Optional, Callable, Coroutine, Any
import traceback

# raw mode features
from signal import SIGWINCH
from abc import ABC
import tty
import termios
import io
import re

# implementing logging handler
from logging import Handler, NOTSET, LogRecord, Formatter, getLogger
from logging import DEBUG, INFO, WARNING, ERROR, CRITICAL


# reset_format = '\x1b[00m'  # CSI and SGR 0
do_backspace = '\10\33[0K'


def carriage_return(arg) -> str:
    """Convert plain newline to CR + LF

       Returns
       -------
       str
           Converted string
    return arg.replace('\n', '\r\n')"""
    return arg.replace('\n', '\r\n')


def rawprint(*args, sep=' ', **kwargs):
    """Standard print wrapper, but with end='\r\n'"""
    print(sep.join(carriage_return(arg) for arg in args), end='\r\n', **kwargs)


ansi_escape = re.compile(r'(\x9B|\x1B\[)[0-?]*[ -\/]*[@-~]')
prev_word = re.compile("(\\w+) *$")
next_word = re.compile("\\w+\\W+(\\w+)")


def truelen(text: str) -> int:
    """Returns amount of visible-on-terminal characters in the string"""
    nocsi = ansi_escape.sub('', text)
    return sum(x.isprintable() for x in nocsi)


def truelen_list(seq: Sequence[str]) -> int:
    """Returns amount of visible-on-terminal characters in the sequence of strings"""
    return sum(char.isprintable() for char in seq)


def _isprintable_predicate(char: str) -> bool:
    return char.isprintable()


class colors(Enum):
    """ANSI terminal colors"""
    BLACK = 0
    RED = 1
    GREEN = 2
    YELLOW = 3
    BLUE = 4
    MAGENTA = 5
    CYAN = 6
    WHITE = 7


def format_term(*, bold=False, italic=False, underline=False, blink=False, fgcolor: Union[colors, int] = None, bgcolor: Union[colors, int] = None, **unused) -> (str, str):
    """
    Returns a tuple(ansi_escape_start: str, ansi_escape_end: str) for specified format
    Tip: "%stext%s" % format_term(args...)

    Parameters
    ----------
    bold : bool = False
        Formatted text appears bold (on some terminals also appears with bright color)
    italic : bool = False
        Formatted text appears italic
    underline : bool = False
        Formatted text appears with underline
    blink : bool = False
        Formatted text will blink fast if True
    fgcolor : colors or int = None
        Set foreground (font) color
        If enum colors specified, formatted text will have an 8-color format
        If integer specified, formatted text will have an 256-color format
    bgcolor : colors of int = None
        Set background (fill) color
        If enum colors specified, formatted text will have an 8-color format
        If integer specified, formatted text will have an 256-color format

    Example
    -------
        AsyncRawInput.writeln("%sHello world!%s" % format_term(italic=True, fgcolor=colors.GREEN, bgcolor=colors.BLACK))
    """
    start = set()
    end = set()
    if bold:
        start.add(1)
        end.add(22)
    if italic:
        start.add(3)
        end.add(23)
    if underline:
        start.add(4)
        end.add(24)
    if blink:
        start.add(5)
        end.add(25)
    if fgcolor is not None or bgcolor is not None:
        if isinstance(fgcolor, colors):
            start.add(30 + fgcolor.value)
            end.add(39)
        if isinstance(bgcolor, colors):
            start.add(40 + bgcolor.value)
            end.add(49)
        # 256 colors mode if integer is passed
        if isinstance(fgcolor, int):
            start.add('38;5;%s' % fgcolor)
            end.add(39)
        if isinstance(bgcolor, int):
            start.add('48;5;%s' % bgcolor)
            end.add(49)
    return '\33[%sm' % ';'.join(str(x) for x in start), \
           '\33[%sm' % ';'.join(str(x) for x in end)


class AbstractARI(ABC):
    """
    An abstract class for asynchronous CLI-interaction with user.
    Calls end() when destructed.
    """
    def __init__(self, history: list = None, history_limit: int = 30, *args, loop=None):
        """
        Parameters
        ----------
        history : list
            List of str containing previous user input
        history_limit : int = 30
            Max amount of elements in history list, when exceeded the old values gets deleted.
        """
        pass

    def __del__(self):
        self.end()

    def prepare(self):
        """
        Does whatever to initialize the terminal interaction.
        """
        pass

    def on_terminal_resize(self, *args, **kw):
        """
        Called when a terminal is resized. Implementation-specific.
        """
        pass

    def end(self):
        """
        Does whatever to finalize the terminal interaction.
        """
        pass

    def get_interrupt_handler(self) -> Callable[[Any], Coroutine[Any, Any, Any]]:
        """
        When Ctrl + C is pressed, the returned async function is called.
        """
        pass

    def set_interrupt_handler(self, callback):
        """
        Sets the callback for Ctrl + C keystroke

        Parameters
        ----------
        callback : coroutine or regular function
            (async) callback, called without arguments
        """
        pass

    def add_keystroke(self, keystroke: str, asyncfunction):
        """
        Add a new keystroke to the terminal

        Parameters
        ----------
        keystroke : str
            Raw keystroke code. For example, tab keystroke will be: "\\t", Ctrl + F will be "\\x06"
        asyncfunction : async function
            Async callback called without arguments
        """
        # self.keystrokes[keystroke] = asyncfunction
        pass

    def remove_keystroke(self, keystroke: str):
        """
        Remove a keystroke from the terminal

        Parameters : str
            Raw keystroke code.
        """
        # del self.keystrokes[keystroke]
        pass

    async def awrite(self, msg: str, **formats):
        """
        Write a formatted text to a terminal without CRLF.
        Don't use it when a user input is prompted

        Parameters
        ----------
        msg : str
            The text without CRLF.
        **formats : keyword arguments
            Formatting arguments passed as format_term(**formats)
        """
        pass

    async def awriteln(self, msg: str, *, error=False, **formats):
        """
        Show a message on the terminal, preserving a user prompt if any.

        Parameters
        ----------
        msg : str
            The message text.
        error : bool
            If True, message will be written to the error stream instead. Default is False.
        **formats : keyword arguments
            Formatting arguments passed as format_term(**formats)
        """
        pass

    def write(self, msg: str, *, error=False, **formats):
        """
        Same as awrite, but does it synchronously or applies the data into the queue
        """
        pass

    def writeln(self, msg: str, *, error=False, **formats):
        """
        Same as awriteln, but does it synchronously or applies the data into the queue
        """
        pass

    def get_terminal_size(self) -> Union[os.terminal_size, Tuple[int, int]]:
        """
        Obtain current resolution of the terminal. Returns tuple-like object: rows, lines
        """
        pass

    def move_cursor(self, at: int, *, flush=True, redraw=False):
        """
        Moves the cursor across the current line.
        Parameter at starts from 1, which means that at=1 is the first character of the terminal line
        """
        pass

    def move_input_cursor(self, at_char: int):
        """
        Sets the cursor's input position at specified character. Scrolls the input horizontally when necessary.
        """
        pass

    def redraw_lastinp(self, at: int, force_redraw_prompt=False):
        pass

    async def get_key(self):
        """
        Asynchronously obtain a single character or keystroke from
        """
        raise NotImplementedError

    async def prompt_line(self, prompt="> ", echo=True, history_disabled=False,
                          prompt_formats={}, input_formats={},
                          start_buffer: Optional[str] = None):
        """
        Start reading a single-line user input with prompt.
        Asynchronous version of input(prompt), handling the keystrokes.
        In addition to Python's input(prompt) function, the input is not wrapped
        into the new line when overflowed, instead it hides the leftmost characters,
        as well as handling the controlling terminal's resizing.
        To register a keystroke, use AsyncRawInput.add_keystroke(code, asyncfunction)

        Parameters
        ----------
        prompt : str
            The text that is displayed before user input
        echo : bool
            Whether or not a user input will be displayed. Set to False when prompting a password
        history_disabled : bool
            Whether or not a new entry should not be added on successful user input. Set to True when prompting a password
        prompt_formats : dict
            Dictionary of text formatting settings that are passed into format_term
            self.prompt_formats = format_term(**prompt_formats)
        input_formats : dict
            Dictionary of text formatting settings that are passed into format_term
            self.input_formats = format_term(**input_formats)
        start_buffer : Optional[str] = None
            If specified, the string will be used as a starting point after which the
            contents can be edited in the prompt line.
        """
        pass

    async def prompt_keystroke(self, prompt=': ', echo=True) -> str:
        """
        Start reading a single character from a terminal. Not handling the keystrokes.

        Parameters
        ----------
        prompt : str
            The text that is displayed before user input
        echo : bool
            Whether or not a user input will be displayed.

        Returns
        -------
        str
            Resulting pressed keystroke
        """
        pass


class AsyncRawInput(AbstractARI):
    """
    Unix implementation for asynchronous input and proper output handling

    Before using its features, self.prepare() should be called
    self.end() is called on object destruction handler

    self.loop : asyncio.BaseEventLoop
        Event loop attached to the IO handler
    self.is_reading : bool
        Whether or not the user input is currently receiving
    self.stdin : io.TextIOWrapper
        File-like object handling the terminal input
    self.stdout : io.TextIOWrapper
        File-like object handling the terminal output
    self.stderr : io.TextIOWrapper
        File-like object handling the error output
    self.read_lastinp : list
        List of str, containing each character for a mutable Unicode string
        Represents an unentered user input displaying on the terminal
    self.read_lastprompt : str
        A prompt. Text prepending the user input line.
        To format a prompt, set self.prompt_formats
    self.old_tcattrs : list
        List of control terminal arguments that have been set before prepare() called
    self.history : list
        List of previous user inputs
    self.history_limit : int
        Threshold of automatic entry deletion of old user inputs
    self.cursor : int
        Current position of the user terminal cursor.
    self.echo : bool = True
        Whether or not the user input is shown on the terminal. Don't modify it manually
    self.ctrl_c : (async) function
        Async callback that is called when Ctrl + C is pressed in the terminal
    self.keystrokes : list
        Sorted list of keystroke handlers (tuple).
        ( "raw keystroke code", async callable )
    self.prompt_formats : tuple(str, str)
        Formatting header, footer pair for displaying a prompt.
    self.input_formats : tuple(str, str)
        Formatting header, footer pair for displaying a user input
    """
    def __init__(self, history: list = None, history_limit: int = 30, stdin: io.TextIOWrapper = sys.stdin, stdout: io.TextIOWrapper = sys.stdout, stderr: io.TextIOWrapper = sys.stderr, *, loop=None):
        """
        Parameters
        ----------
        history : list
            List of str containing previous user input
        history_limit : int = 30
            Max amount of elements in history list, when exceeded the old values gets deleted.
        stdin : io.TextIOWrapper = sys.stdin
            File-like object handling standard input. Should be tty-like file
        stdout : io.TextIOWrapper = sys.stdout
            File-like object handling standard output.
        """
        self.loop = loop if loop is not None else asyncio.get_event_loop()
        self.is_reading = False
        self.stdin = stdin
        self.stdout = stdout
        self.stderr = stderr
        # Triggered every time a stream is available to be IO'ed.
        self.read_clk = asyncio.Event()
        self.write_clk = asyncio.Event()
        self.error_clk = asyncio.Event()
        self.read_lastkey: Optional[str] = None
        self.read_lastinp: MutableSequence[str] = []  # can only contain printable strings,
        # ...mutability but extra memory consumption
        self.read_lastprompt = ''
        self.old_tcattrs: MutableSequence[int] = None
        self.was_blocking: Optional[Tuple[bool, bool, bool]] = None
        self.prepared = False
        self.history = history
        self.history_limit = history_limit
        self.cursor = 0
        self._promptline_scroll = 0  # when terminal column-size exceeded, indicates the horizontal scroll
        self.current_input_buffer = []
        # horizontal scrolling behavior:
        # - draw only single line of input
        # - when at least 1 character of prompt can be shown, always format it, don't skip formatting
        # - scroll back when character is erased
        # - do not show dots or any symbol on the edges to make sure characters fill the entire line, sorry about that
        # - do not snap the scrolling, scroll char-by-char instead
        # - when scroll > 0, always redraw the prompt
        self.echo = True
        self.ctrl_c = None
        self.keystrokes: MutableSequence[Tuple[str, Callable[Any, Coroutine[Any, Any, Any]]]] = []
        self.default_kshandler: Optional[Callable[[str, list], Coroutine[Any, Any, Any]]] = None
        self.prompt_formats: Sequence[Tuple[str, str]] = ('', '')
        self.input_formats: Sequence[Tuple[str, str]] = ('', '')
        self._prompting_task: Optional[asyncio.Task] = None

    # Keystrokes that are handled in prompt_line
    def _enter_ks(self):
        pass

    def _backspace_ks(self):
        if self.read_lastinp and self.cursor != 0:
            self.read_lastinp.pop(self.cursor - 1)
            self.cursor -= 1
            if self.echo:
                if self._promptline_scroll > 0:
                    self._promptline_scroll -= 1
                    self.redraw_lastinp(self.cursor + 1, force_redraw_prompt=True)
                else:
                    self.redraw_lastinp(self.cursor + 1)

    def prepare(self):
        """
        Enables raw mode, saving the old TTY settings.
        Disables blocking mode for standard input, output and error output.
        Hooks up the SIGWINCH signal handler, which will redraw the prompt line if any.
        """
        if not self.prepared:
            self.old_tcattrs = termios.tcgetattr(self.stdin.fileno())
        self.prepared = True
        self.was_blocking = (
            os.get_blocking(self.stdin.fileno()),
            os.get_blocking(self.stdout.fileno()),
            os.get_blocking(self.stderr.fileno()),
        )
        os.set_blocking(self.stdin.fileno(), False)
        os.set_blocking(self.stdout.fileno(), False)
        os.set_blocking(self.stderr.fileno(), False)
        self.loop.add_signal_handler(SIGWINCH, self.on_terminal_resize)
        tty.setraw(self.stdin.fileno())

    def on_terminal_resize(self):
        cols, lines = self.get_terminal_size()
        self.move_input_cursor(self.cursor)

    def end(self):
        """
        Disables raw mode, restoring the old TTY settings for standard input
        Unhooks the SIGWINCH signal handler.
        """
        if self.is_reading and self.input_formats:
            self.stdout.write(self.input_formats[1])
            self.stdout.flush()
        termios.tcsetattr(self.stdin.fileno(), termios.TCSANOW, self.old_tcattrs)
        self.loop.remove_signal_handler(SIGWINCH)
        self.is_reading = False
        self.prepared = False
        os.set_blocking(self.stdin.fileno(), self.was_blocking[0])
        os.set_blocking(self.stdout.fileno(), self.was_blocking[1])
        os.set_blocking(self.stderr.fileno(), self.was_blocking[2])

    def get_interrupt_handler(self) -> Callable[[Any], Coroutine[Any, Any, Any]]:
        return self.ctrl_c

    def set_interrupt_handler(self, callback):
        self.ctrl_c = callback

    def add_keystroke(self, keystroke: str, asyncfunction):
        index = bisect_left(self.keystrokes, keystroke, key=itemgetter(0))
        try:
            if self.keystrokes[index] == keystroke:
                raise ValueError(f'keystroke {repr(keystroke)} already exists')
        except IndexError:
            pass
        self.keystrokes.insert(index, (keystroke, asyncfunction))

    def remove_keystroke(self, keystroke: str):
        index = bisect_left(self.keystrokes, keystroke, key=itemgetter(0))
        if self.keystrokes[index][0] == keystroke:
            del self.keystrokes[index]
        else:
            raise ValueError(f'keystroke {repr(keystroke)} not found')

    async def awrite(self, msg: str, *, error=False, **formats):
        """
        Mapped to write(), the asynchronous write will be implemented later.
        """
        self.write(msg, **formats)

    async def awriteln(self, msg: str, *, error=False, **formats):
        """
        Mapped to writeln(), the asynchronous writeln will be implemented later.
        """
        self.writeln(msg, **formats)

    def write(self, msg: str, *, error=False, **formats):
        stream = self.stderr if error else self.stdout
        _formats = format_term(**formats)
        stream.write(_formats[0] + carriage_return(msg) + _formats[1])
        stream.flush()

    def writeln(self, msg: str, *, error=False, **formats):
        formats_tuple = format_term(**formats)
        stream = self.stderr if error else self.stdout
        if len(formats_tuple) != 2:
            formats_tuple = ('', '')
        if self.is_reading:
            # disable input graphic rendition stuff
            self.stdout.write(self.input_formats[1])
            # clear the current line over the cursor
            self.stdout.write('\r\33[2K')
            # write the message
            stream.write('\r' + formats_tuple[0] + carriage_return(msg) + formats_tuple[1] + '\n')
            stream.flush()
            self.redraw_lastinp(0, force_redraw_prompt=True)
        else:
            stream.write('\r\33[0K' + formats_tuple[0] + carriage_return(msg) + formats_tuple[1] + '\n\r')
            stream.flush()

    def get_terminal_size(self) -> Union[os.terminal_size, Tuple[int, int]]:
        return os.get_terminal_size(self.stdout.fileno())

    def move_cursor(self, at: int, *, flush=True, redraw=False):
        """
        Moves the cursor across the current line.
        Parameter at starts from 1, which means that at=1 is the first character of the terminal line
        """
        self.stdout.write('\33[%sG' % at)
        if redraw:
            self.redraw_lastinp(0)
        if flush:
            self.stdout.flush()

    def move_input_cursor(self, at_char: int):
        """
        Sets the cursor's input position at specified character. Scrolls the input horizontally when necessary.
        """
        # clamped absolute position of the cursor
        at_char = max(min(len(self.read_lastinp), at_char), 0)
        cols, _ = self.get_terminal_size()
        _tlen = truelen(self.read_lastprompt)
        # N-th character is the character of the input buffer at the beginning of the line (bottom-left corner of the terminal)
        # Can be negative which means the prompt characters
        _tlen_scrolled = self._promptline_scroll - _tlen
        self.cursor = at_char
        if self.cursor == 0:
            self._promptline_scroll = 0
            self.move_cursor(_tlen, flush=False)
            self.redraw_lastinp(0, force_redraw_prompt=True)
        elif self.cursor < max(_tlen_scrolled, 0):
            self._promptline_scroll = at_char + _tlen
            self.move_cursor(1, flush=False)
            self.redraw_lastinp(0, force_redraw_prompt=True)
        elif self.cursor > _tlen_scrolled + cols - 1:
            self._promptline_scroll = max(0, at_char - cols + _tlen + 1)
            self.move_cursor(cols, flush=False)
            self.redraw_lastinp(0, force_redraw_prompt=True)
        else:
            # scrolling is not required in this case, thus redrawing
            self.cursor = at_char
            self.move_cursor(at_char - _tlen_scrolled + 1)

    def redraw_lastinp(self, at: int, force_redraw_prompt=False):
        """
        Redisplay a user input at specified position on current cursor line.
        If force_redraw_prompt is True, redraws the whole line entirely (including the prompt) regardless of scrolling state
        """
        # note: do not call move_cursor with redraw=True, this will cause an infinite recursion!
        cols, lines = self.get_terminal_size()
        if self._promptline_scroll or force_redraw_prompt:
            at = 0
            # move cursor at the beginning of the line
            self.move_cursor(1, flush=False)
            # clear the rest
            self.stdout.write('\33[0J')
            # reprint the prompt when possible
            t_len = truelen(self.read_lastprompt)
            visible_prompt_len = t_len - self._promptline_scroll
            if visible_prompt_len > 0:
                self.stdout.write('\r' + self.prompt_formats[0] + self.read_lastprompt[self._promptline_scroll:] + self.prompt_formats[1])
            if self.echo:
                # render part of user input and enable formats
                _start = max(0, self._promptline_scroll - t_len)
                _end = _start + cols - max(0, visible_prompt_len)
                self.stdout.write(self.input_formats[0] + ''.join(self.read_lastinp[_start:_end]))
                # restore cursor pos
                self.move_cursor(visible_prompt_len + self.cursor + 1, flush=False)
        elif self.echo:
            # move cursor at redrawing part
            t_len = truelen(self.read_lastprompt)
            self.move_cursor(t_len + max(at, 1), flush=False)
            # clear the rest
            self.stdout.write('\33[0K')
            # render part of user input and enable formats
            self.stdout.write(self.input_formats[0] + ''.join(self.read_lastinp[max(at - 1, 0):cols - t_len]))
            # restore cursor pos
            self.move_cursor(truelen(self.read_lastprompt) + self.cursor + 1, flush=False)
        self.stdout.flush()

    def _stdin_handler(self):
        try:
            self.current_input_buffer.clear()
            while True:
                symbol = self.stdin.read(1)
                if not symbol:
                    self.read_clk.set()
                    break
                self.current_input_buffer.append(symbol)
        except TypeError:
            self.read_clk.set()
        except BlockingIOError:
            self.read_clk.set()

    async def prompt_line(self, prompt="> ", echo=True, history_disabled=False,
                          prompt_formats={}, input_formats={}, start_buffer: Optional[str] = None):
        self._promptline_scroll = 0
        try:
            _task = asyncio.current_task()
            if self._prompting_task is not None and not self._prompting_task.done() and self._prompting_task is not _task and self.is_reading:
                self._prompting_task.cancel()
            self._prompting_task = _task
            self.is_reading = True
            if self.read_lastinp is None:
                self.read_lastinp = []
            elif start_buffer:
                self.read_lastinp = list(start_buffer)
            else:
                self.read_lastinp.clear()
            self.read_lastprompt = prompt
            self.prompt_formats = format_term(**prompt_formats)
            self.input_formats = format_term(**input_formats)
            if self.stdout.writable():
                self.stdout.write('\r' + self.prompt_formats[0] + self.read_lastprompt + self.prompt_formats[1])
                self.stdout.flush()
            self.cursor = 0
            self.echo = echo
            history_pos = 0
            history_incomplete = self.read_lastinp
            # absolute cursor position is calculated using truelen(prompt) + self.cursor - self._promptline_scroll
            # dye the user input by the specified color
            self.stdout.write(self.input_formats[0])
            self.stdout.flush()
            while self.is_reading:
                self.loop.add_reader(self.stdin.fileno(), self._stdin_handler)
                # suspend until a key is pressed
                await self.read_clk.wait()
                self.loop.remove_reader(self.stdin.fileno())
                self.read_clk.clear()
                keystroke = ''.join(self.current_input_buffer)
                index = max(0, min(len(self.keystrokes) - 1, bisect_left(self.keystrokes, keystroke, key=itemgetter(0))))
                # self.writeln(f'Keystroke: {index}, keystrokes: {repr(self.keystrokes)}')
                if self.keystrokes and self.keystrokes[index][0] == keystroke:
                    if asyncio.iscoroutinefunction(self.keystrokes[index][1]):
                        await self.keystrokes[index][1]()
                    else:
                        self.keystrokes[index][1]()
                    continue
                if not keystroke.isprintable():
                    # one of the characters are not printable, which means that this is a keystroke
                    if len(self.current_input_buffer) == 1:
                        if ord(self.current_input_buffer[0]) == 127 or ord(self.current_input_buffer[0]) == 8:
                            # do backspace
                            self._backspace_ks()
                            continue
                        elif ord(self.current_input_buffer[0]) == 3:
                            # Ctrl + C
                            if self.ctrl_c is None:
                                raise RuntimeError('Ctrl + C is not handled')
                            elif asyncio.iscoroutinefunction(self.ctrl_c):
                                await self.ctrl_c()
                            else:
                                self.ctrl_c()
                            continue
                        elif ord(self.current_input_buffer[0]) == 13 or ord(self.current_input_buffer[0]) == 10:
                            # submit the input
                            break
                    elif ord(self.current_input_buffer[0]) == 27 and len(self.current_input_buffer) >= 3:
                        # probably arrow keys or other keystrokes
                        if keystroke == '\33[3~':
                            # delete (frontspace)
                            if self.read_lastinp and self.cursor < len(self.read_lastinp):
                                self.read_lastinp.pop(self.cursor)
                                if echo:
                                    self.redraw_lastinp(self.cursor + 1)
                                self.move_input_cursor(self.cursor)
                            continue
                        elif keystroke == '\33[D':
                            # cursor left
                            if self.cursor > 0:
                                self.cursor -= 1
                                # self.stdout.write('\33[D')
                                if self.echo:
                                    self.move_input_cursor(self.cursor)
                            continue
                        elif keystroke == '\33[C':
                            # cursor right
                            if self.cursor < len(self.read_lastinp):
                                self.cursor += 1
                                # self.stdout.write('\33[C')
                                if self.echo:
                                    self.move_input_cursor(self.cursor)
                            continue
                        elif keystroke == '\33[A':
                            # move older history (cursor up)
                            if self.history and not history_disabled:
                                if history_pos < len(self.history):
                                    history_pos += 1
                                    # load previous command
                                    self.read_lastinp = list(self.history[-history_pos])
                                    self.cursor = len(self.read_lastinp)
                                    cols, _ = self.get_terminal_size()
                                    self._promptline_scroll = max(0, self.cursor - cols + truelen(self.read_lastprompt) + 1)
                                    if self.echo:
                                        self.redraw_lastinp(0)
                            continue
                        elif keystroke == '\33[B':
                            # move newer history (cursor)
                            if self.history and not history_disabled:
                                if history_pos == 1:
                                    history_pos = 0
                                    # load incomplete command
                                    self.read_lastinp = history_incomplete
                                    self.cursor = len(self.read_lastinp)
                                    cols, _ = self.get_terminal_size()
                                    self._promptline_scroll = max(0, self.cursor - cols + truelen(self.read_lastprompt) + 1)
                                    if self.echo:
                                        self.redraw_lastinp(0)
                                elif history_pos > 0:
                                    history_pos -= 1
                                    # load next command
                                    self.read_lastinp = list(self.history[-history_pos])
                                    self.cursor = len(self.read_lastinp)
                                    cols, _ = self.get_terminal_size()
                                    self._promptline_scroll = max(0, self.cursor - cols + truelen(self.read_lastprompt) + 1)
                                    self.redraw_lastinp(0)
                            continue
                        elif keystroke == '\33[H':
                            # home key
                            self.move_input_cursor(0)
                            continue
                        elif keystroke == '\33[F':
                            # end key
                            self.move_input_cursor(len(self.read_lastinp))
                            continue
                        # i am sorry for these workarounds... costs some CPU time
                        elif keystroke == '\33[1;5D':
                            # previous word
                            if self.cursor > 0 and self.read_lastinp:
                                _prevword_match = prev_word.search(''.join(self.read_lastinp), 0, self.cursor)
                                if _prevword_match is not None:
                                    self.move_input_cursor(_prevword_match.start(1))
                                else:
                                    self.move_input_cursor(0)
                            continue
                        elif keystroke == '\33[1;5C':
                            # next word
                            if self.cursor < len(self.read_lastinp) and self.read_lastinp:
                                _nextword_match = next_word.search(''.join(self.read_lastinp), self.cursor)
                                if _nextword_match is not None:
                                    self.move_input_cursor(_nextword_match.start(1))
                                else:
                                    self.move_input_cursor(len(self.read_lastinp))
                            continue
                        # else:
                        #     self.writeln('Unknown keystroke: %s' % ', '.join(repr(x) for x in key), fgcolor=colors.RED, bold=True)
                    if self.default_kshandler is not None:
                        if asyncio.iscoroutinefunction(self.default_kshandler):
                            await self.default_kshandler(keystroke, self.current_input_buffer)
                        else:
                            self.default_kshandler(keystroke, self.current_input_buffer)
                    # always reread after a keystroke dispatched
                    continue
                else:
                    # letter input
                    # for i in range(len(key)):
                    #    self.read_lastinp.insert(self.cursor + i, key[i])
                    self.read_lastinp[self.cursor:self.cursor] = keystroke
                    self.cursor += len(self.current_input_buffer)
                    if echo:
                        if self.cursor < len(self.read_lastinp):
                            # self.redraw_lastinp(self.cursor)
                            self.redraw_lastinp(0)
                        else:
                            self.stdout.write(''.join(self.current_input_buffer))
                        self.move_input_cursor(self.cursor)
                    self.stdout.flush()
            # remove input format
            result = ''.join(self.read_lastinp)
            if self.history is not None and not history_disabled:
                if not len(self.history):
                    self.history.append(result)
                elif result != self.history[-1]:
                    self.history.append(result)
                    if len(self.history) > self.history_limit:
                        self.history.pop(0)
            self.read_lastinp.clear()
            return result
        finally:
            self.loop.remove_reader(self.stdin.fileno())
            self.stdout.write(self.input_formats[1] + '\r\n')
            self.stdout.flush()
            self.is_reading = False

    async def prompt_keystroke(self, prompt=': ', echo=True) -> str:
        backup_inputformats = self.input_formats
        backup_promptformats = self.prompt_formats
        self.input_formats = ('', '')
        self.prompt_formats = ('', '')
        try:
            _task = asyncio.current_task()
            if self._prompting_task is not None and not self._prompting_task.done() and self._prompting_task is not _task and self.is_reading:
                self._prompting_task.cancel()
            self._prompting_task = _task
            self.read_lastprompt = prompt
            if self.stdout.writable():
                self.stdout.write(('\r' + self.read_lastprompt))
                self.stdout.flush()
            self.cursor = 0
            self.echo = echo
            self.is_reading = True
            self.loop.add_reader(self.stdin.fileno(), self._stdin_handler)
            await self.read_clk.wait()
            self.read_clk.clear()
            result = ''.join(self.current_input_buffer)
            if echo:
                self.stdout.write(''.join(result))
            return result
        finally:
            self.is_reading = False
            self.loop.remove_reader(self.stdin.fileno())
            if echo:
                self.stdout.write('\r\n')
                self.stdout.flush()
        self.input_formats = backup_inputformats
        self.prompt_formats = backup_promptformats


async def _background_task(inp: AsyncRawInput, amount: int):
    """
    This is debug function for simultaneous output
    """
    for i in range(amount):
        inp.writeln('Hello world! [%s]' % ', '.join(inp.history), fgcolor=colors.YELLOW)
        await asyncio.sleep(0.5)


class ARILogHandler(Handler):
    """
    Custom logging handler for displaying log messages in the terminal.
    """
    def __init__(self, ari: AsyncRawInput, level=NOTSET):
        super().__init__(level)
        self.ari = ari

    def emit(self, record: LogRecord):
        try:
            msg = self.format(record)
            formats = {}
            if record.levelno == DEBUG:
                formats['fgcolor'] = colors.YELLOW
            if record.levelno == WARNING:
                formats['fgcolor'] = colors.YELLOW
                formats['bold'] = True
            if record.levelno == ERROR:
                formats['fgcolor'] = colors.RED
            if record.levelno == CRITICAL:
                formats['bgcolor'] = colors.RED
                formats['fgcolor'] = colors.WHITE
            self.ari.writeln(msg, **formats)
        except Exception:
            self.handleError(record)


async def _log_testing(inp: AsyncRawInput, amount: int):
    """
    This is debug function for custom log handler testing
    """
    logger = getLogger("ranmame")
    logger.setLevel(INFO)
    handler = ARILogHandler(inp)
    formatter = Formatter("%(asctime)s [%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    sublog = logger.getChild('testeng')
    sublog.propagate = True
    # sublog.addHandler(NullHandler())
    for i in range(amount):
        logger.log((DEBUG, INFO, WARNING, ERROR)[i % 4], "This is a debug log!")
        sublog.log((INFO, WARNING, ERROR)[i % 3], "This is sublog!")
        await asyncio.sleep(1)


async def _process_inputs(amount: int):
    """
    This is debug function for input prompt testing
    """
    rawprint('handling %s inputs' % amount)
    inp = AsyncRawInput(history=[])

    def default_kshandler(keystroke: str, key: list):
        inp.writeln("Unknown keystroke: %s" % key)

    inp.prepare()
    inp.default_kshandler = default_kshandler

    def print_cursor_pos():
        inp.writeln("Cursor position: %s" % inp.cursor)

    def print_termsize():
        inp.writeln("Terminal size: %sx%s" % inp.get_terminal_size())

    def print_scrolling():
        inp.writeln("Scrolling: %s" % inp._promptline_scroll)

    def scroll_left():
        if inp._promptline_scroll > 0:
            inp._promptline_scroll -= 1
        inp.redraw_lastinp(0, True)

    def scroll_right():
        inp._promptline_scroll += 1
        inp.redraw_lastinp(0, True)

    inp.add_keystroke("\x10", print_cursor_pos)
    inp.add_keystroke("\x0c", print_termsize)
    inp.add_keystroke("\x13", print_scrolling)
    inp.add_keystroke("\x01", scroll_left)
    inp.add_keystroke("\x04", scroll_right)
    inp.remove_keystroke("\x04")
    asyncio.create_task(_background_task(inp, 10))
    asyncio.create_task(_log_testing(inp, 6))
    # response = await inp.prompt_keystroke('Are you sure? y/n: ')
    # if response.lower() == 'n':
    #    return
    try:
        for i in range(amount):
            # rawprint('\r\33[0K' + ', '.join(str(x) for x in await inp.prompt_line()))
            line = await inp.prompt_line("Prompt> ")
            inp.writeln(repr(inp.history) + '\n\r\33[0K' + line, bold=True, fgcolor=colors.GREEN)
            # rawprint(await rinput())
    finally:
        inp.end()
    rawprint('handling is complete')


async def _main():
    rawprint("Asyncio raw input example echoing, try press keys 30 seconds")
    rawprint('abc')
    rawprint('def')
    rawprint('With\nnewline')
    try:
        await asyncio.wait_for(_process_inputs(30), 30)
    except asyncio.TimeoutError:
        rawprint('Timeout, exiting!')
    except KeyboardInterrupt:
        rawprint("Inerrupt")
    except BaseException:
        rawprint(traceback.format_exc())


if __name__ == '__main__':
    loop = asyncio.new_event_loop()
    loop.run_until_complete(_main())
