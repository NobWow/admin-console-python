"""
ainput.py
Tools for creating a simple text user interfaces in Unix-like systems with using asyncio
Doesn't support Windows or any non-POSIX terminals
"""

import asyncio
import os
import sys
from enum import Enum
from typing import Union
import traceback

# raw mode features
import tty
import termios
import io
import re

# implementing logging handler
from logging import Handler, NOTSET, LogRecord, getLogger, Formatter
from logging import DEBUG, INFO, WARNING, ERROR, CRITICAL


loop = asyncio.get_event_loop()
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


def truelen(text: str) -> int:
    """Returns amount of visible-on-terminal characters in the string"""
    nocsi = ansi_escape.sub('', text)
    return len(''.join(x for x in nocsi if x.isprintable()))


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


class AsyncRawInput():
    """
    Main class for asynchronous input and proper output handling

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
    self.ctrl_c : async function
        Async callback that is called when Ctrl + C is pressed in the terminal
    self.keystrokes : dict
        Mapping of keystroke handlers.
        { "raw keystroke code": async callable }
    self.prompt_formats : tuple(str, str)
        Formatting header, footer pair for displaying a prompt.
    self.input_formats : tuple(str, str)
        Formatting header, footer pair for displaying a user input
    """
    def __init__(self, history: list = None, history_limit: int = 30, stdin: io.TextIOWrapper = sys.stdin, stdout: io.TextIOWrapper = sys.stdout, *, loop=None):
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
        self.loop = loop if loop else asyncio.get_event_loop()
        self.is_reading = False
        self.stdin = stdin
        self.stdout = stdout
        self.read_lastinp: list = []  # mutability but extra memory consumption
        self.read_lastprompt = ''
        self.old_tcattrs: list = None
        self.prepared = False
        self.history = history
        self.history_limit = history_limit
        self.cursor = 0
        self.echo = True
        self.ctrl_c = None
        self.keystrokes = {}
        self.prompt_formats = ('', '')
        self.input_formats = ('', '')

    def __del__(self):
        self.end()

    def prepare(self):
        """
        Enables raw mode, saving the old TTY settings. Disables blocking mode for standard input
        """
        if not self.prepared:
            self.old_tcattrs = termios.tcgetattr(self.stdin.fileno())
        self.prepared = True
        os.set_blocking(self.stdin.fileno(), False)
        tty.setraw(self.stdin.fileno())

    def end(self):
        """
        Disables raw mode, restoring the old TTY settings for standard input
        """
        if self.is_reading and self.input_formats:
            self.stdout.write(self.input_formats[1])
            self.stdout.flush()
        termios.tcsetattr(self.stdin.fileno(), termios.TCSANOW, self.old_tcattrs)
        self.is_reading = False
        self.prepared = False

    def set_interrupt_handler(self, awaitable):
        """
        Sets the callback for Ctrl + C keystroke

        Parameters
        ----------
        awaitable : coroutine function
            async callback, called without arguments
        """
        if not asyncio.iscoroutinefunction(awaitable):
            raise TypeError('awaitable should be a coroutine function')
        self.ctrl_c = awaitable

    def add_keystroke(self, keystroke: str, awaitable):
        """
        Add a new keystroke to the terminal

        Parameters
        ----------
        keystroke : str
            Raw keystroke code. For example, tab keystroke will be: "\\t", Ctrl + F will be "\\x06"
        awaitable : async function
            Async callback called without arguments
        """
        self.keystrokes[keystroke] = awaitable

    def remove_keystroke(self, keystroke: str):
        """
        Remove a keystroke from the terminal

        Parameters : str
            Raw keystroke code.
        """
        del self.keystrokes[keystroke]

    def write(self, msg: str, **formats):
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
        self.stdout.write(('\r%s' + carriage_return(msg) + '%s') % format_term(**formats))

    def writeln(self, msg: str, **formats):
        """
        Show a message on the terminal, preserving a user prompt if any.

        Parameters
        ----------
        msg : str
            The message text.
        **formats : keyword arguments
            Formatting arguments passed as format_term(**formats)
        """
        formats_tuple = format_term(**formats)
        if len(formats_tuple) != 2:
            formats_tuple = ('', '')
        if self.is_reading:
            # disable input graphic rendition stuff
            self.stdout.write(self.input_formats[1])
            # clear the current line over the cursor
            self.stdout.write('\r\33[2K')
            # write the message
            self.stdout.write('\r' + formats_tuple[0] + carriage_return(msg) + formats_tuple[1] + '\n')
            # reprint the prompt
            self.stdout.write('\r' + self.prompt_formats[0] + self.read_lastprompt + self.prompt_formats[1])
            if self.echo:
                # reprint last user input and re-enable graphic rendition for user input
                self.stdout.write(self.input_formats[0] + ''.join(self.read_lastinp))
                # move cursor to the last position
                self.stdout.write('\33[%sG' % (truelen(self.read_lastprompt) + self.cursor + 1))
            # stdout is a toilet
            self.stdout.flush()
        else:
            self.stdout.write('\r\33[0K' + formats_tuple[0] + carriage_return(msg) + formats_tuple[1] + '\n\r')
            self.stdout.flush()

    def redraw_lastinp(self, at: int):
        """
        Redisplay a user prompt at specified position on current cursor line.
        """
        # move cursor at redrawing part
        self.stdout.write('\33[%sG' % (truelen(self.read_lastprompt) + at))
        # clear the rest
        self.stdout.write('\33[0K')
        # render part of user input and enable formats
        self.stdout.write(self.input_formats[0] + ''.join(self.read_lastinp[at - 1:]))
        # restore cursor pos
        self.stdout.write('\33[%sG' % (truelen(self.read_lastprompt) + self.cursor + 1))
        self.stdout.flush()

    async def prompt_line(self, prompt="> ", echo=True, history_disabled=False, prompt_formats={}, input_formats={}):
        """
        Start reading a single-line user input with prompt from AsyncRawInput.stdin. Asynchronous version of input(prompt), handling the keystrokes.
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
        """
        try:
            self.is_reading = True
            if self.read_lastinp is None:
                self.read_lastinp = []
            else:
                self.read_lastinp.clear()
            self.read_lastprompt = prompt
            self.prompt_formats = format_term(**prompt_formats)
            self.input_formats = format_term(**input_formats)
            if self.stdout.writable():
                self.stdout.write(('\r%s' + self.read_lastprompt + '%s') % self.prompt_formats)
                self.stdout.flush()

            read_clk = asyncio.Event()
            key = []

            def stdin_reader():
                try:
                    key.clear()
                    while True:
                        symbol = self.stdin.read(1)
                        if not symbol:
                            read_clk.set()
                            break
                        key.append(symbol)
                except TypeError:
                    read_clk.set()
                except BlockingIOError:
                    read_clk.set()

            self.cursor = 0
            self.echo = echo
            history_pos = 0
            history_incomplete = self.read_lastinp
            # absolute cursor position is calculated using truelen(prompt) + self.cursor
            while self.is_reading:
                self.loop.add_reader(self.stdin.fileno(), stdin_reader)
                # dye the user input by the specified color
                self.stdout.write(self.input_formats[0])
                self.stdout.flush()
                # suspend until a key is pressed
                await read_clk.wait()
                self.loop.remove_reader(self.stdin.fileno())
                read_clk.clear()
                keystroke = ''.join(key)
                if keystroke in self.keystrokes:
                    await self.keystrokes[keystroke]()
                    continue
                if len(key) == 1:
                    if ord(key[0]) == 127 or ord(key[0]) == 8:
                        # do backspace
                        if self.read_lastinp:
                            self.read_lastinp.pop(self.cursor - 1)
                            self.cursor -= 1
                            if echo:
                                self.redraw_lastinp(self.cursor)
                        continue
                    elif ord(key[0]) == 3:
                        # Ctrl + C
                        if self.ctrl_c is None:
                            raise RuntimeError('Ctrl + C is not handled')
                        else:
                            await self.ctrl_c()
                        continue
                    elif ord(key[0]) == 13 or ord(key[0]) == 10:
                        # submit the input
                        break
                elif ord(key[0]) == 27 and len(key) == 3:
                    # probably arrow keys or other keystrokes
                    if keystroke == '\33[D':
                        # cursor left
                        if self.cursor > 0:
                            self.cursor -= 1
                            # self.stdout.write('\33[D')
                            if self.echo:
                                self.stdout.write('\33[%sG' % (truelen(self.read_lastprompt) + self.cursor + 1))
                                self.stdout.flush()
                    elif keystroke == '\33[C':
                        # cursor right
                        if self.cursor < len(self.read_lastinp):
                            self.cursor += 1
                            # self.stdout.write('\33[C')
                            if self.echo:
                                self.stdout.write('\33[%sG' % (truelen(self.read_lastprompt) + self.cursor + 1))
                                self.stdout.flush()
                    elif keystroke == '\33[A':
                        # move older history (cursor up)
                        if self.history and not history_disabled:
                            if history_pos < len(self.history):
                                history_pos += 1
                                # load previous command
                                self.read_lastinp = list(self.history[-history_pos])
                                self.cursor = len(self.read_lastinp)
                                if self.echo:
                                    self.redraw_lastinp(1)
                    elif keystroke == '\33[B':
                        # move newer history (cursor)
                        if self.history and not history_disabled:
                            if history_pos == 1:
                                history_pos = 0
                                # load incomplete command
                                self.read_lastinp = history_incomplete
                                self.cursor = len(self.read_lastinp)
                                if self.echo:
                                    self.redraw_lastinp(1)
                            elif history_pos > 0:
                                history_pos -= 1
                                # load next command
                                self.read_lastinp = list(self.history[-history_pos])
                                self.cursor = len(self.read_lastinp)
                                self.redraw_lastinp(1)
                    else:
                        self.writeln('Unknown keystroke: %s' % ', '.join(repr(x) for x in key), fgcolor=colors.RED, bold=True)
                    continue
                # letter input
                for i in range(len(key)):
                    self.read_lastinp.insert(self.cursor + i, key[i])
                self.cursor += len(key)
                if echo:
                    if self.cursor < len(self.read_lastinp):
                        self.redraw_lastinp(self.cursor)
                    else:
                        self.stdout.write(''.join(key))
                self.stdout.flush()
            # remove input format
            self.stdout.write(self.input_formats[1] + '\r\n')
            self.stdout.flush()
            self.is_reading = False
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
        except BaseException as exc:
            self.is_reading = False
            self.loop.remove_reader(self.stdin.fileno())
            self.stdout.write(self.input_formats[1])
            self.stdout.flush()
            raise exc

    async def prompt_keystroke(self, prompt=': ', echo=True):
        """
        Start reading a single character from a terminal. Not handling the keystrokes.

        Parameters
        ----------
        prompt : str
            The text that is displayed before user input
        echo : bool
            Whether or not a user input will be displayed.
        """
        backup_inputformats = self.input_formats
        backup_promptformats = self.prompt_formats
        self.input_formats = ('', '')
        self.prompt_formats = ('', '')
        try:
            char = []
            event = asyncio.Event()
            self.read_lastprompt = prompt
            if self.stdout.writable():
                self.stdout.write(('\r' + self.read_lastprompt))
                self.stdout.flush()
            self.cursor = 0
            self.echo = echo

            def char_reader():
                try:
                    while True:
                        symbol = self.stdin.read(1)
                        if not symbol:
                            event.set()
                            break
                        char.append(symbol)
                except TypeError:
                    event.set()
                except BlockingIOError:
                    event.set()

            self.is_reading = True
            self.loop.add_reader(self.stdin.fileno(), char_reader)
            await event.wait()
            self.loop.remove_reader(self.stdin.fileno())
            self.is_reading = False
            result = ''.join(char)
            self.stdout.write(''.join(result) + '\r\n')
            return result
        except BaseException as exc:
            self.is_reading = False
            self.loop.remove_reader(self.stdin.fileno())
            raise exc
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
    logger = getLogger("main")
    logger.setLevel(INFO)
    handler = ARILogHandler(inp)
    formatter = Formatter("%(asctime)s [%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    for i in range(amount):
        logger.log((DEBUG, INFO, WARNING, ERROR)[i % 4], "This is a debug log!")
        await asyncio.sleep(1)


async def _process_inputs(amount: int):
    """
    This is debug function for input prompt testing
    """
    rawprint('handling %s inputs' % amount)
    inp = AsyncRawInput(history=[])
    inp.prepare()
    asyncio.create_task(_background_task(inp, 10))
    asyncio.create_task(_log_testing(inp, 6))
    # response = await inp.prompt_keystroke('Are you sure? y/n: ')
    # if response.lower() == 'n':
    #    return
    try:
        for i in range(amount):
            # rawprint('\r\33[0K' + ', '.join(str(x) for x in await inp.prompt_line()))
            line = await inp.prompt_line()
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
    loop.run_until_complete(_main())
