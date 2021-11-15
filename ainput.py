import asyncio
import os
import sys
from enum import Enum
import traceback

# raw mode features
import tty
import termios
import io
import re

# implementing logging handler
from logging import Handler, NOTSET, LogRecord
from logging import DEBUG, WARNING, ERROR, CRITICAL


loop = asyncio.get_event_loop()
# reset_format = '\x1b[00m'  # CSI and SGR 0
do_backspace = '\10\33[0K'


def carriage_return(arg):
    return arg.replace('\n', '\r\n')


def rawprint(*args, sep=' ', **kwargs):
    print(sep.join(carriage_return(arg) for arg in args), end='\r\n', **kwargs)


ansi_escape = re.compile(r'(\x9B|\x1B\[)[0-?]*[ -\/]*[@-~]')


def truelen(text: str):
    nocsi = ansi_escape.sub('', text)
    return len(''.join(x for x in nocsi if x.isprintable()))


class colors(Enum):
    BLACK = 0
    RED = 1
    GREEN = 2
    YELLOW = 3
    BLUE = 4
    MAGENTA = 5
    CYAN = 6
    WHITE = 7


def format_term(*, bold=False, italic=False, underline=False, blink=False, fgcolor: colors = None, bgcolor: colors = None, **unused):
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
    def __init__(self, history: list = None, history_limit: int = 30, stdin: io.TextIOWrapper = sys.stdin, stdout: io.TextIOWrapper = sys.stdout, *, loop=None):
        self.loop = loop if loop else asyncio.get_event_loop()
        self.is_reading = False
        self.stdin = stdin
        self.stdout = stdout
        self.read_lastinp: list = []  # mutability but extra memory consumption
        self.read_lastprompt = ''
        self.old_tcattrs = termios.tcgetattr(0)
        self.history = history
        self.history_limit = history_limit
        self.cursor = 0
        self.echo = True
        self.ctrl_c = None
        self.keystrokes = {}
        self.prompt_formats = ('', '')
        self.input_formats = ('', '')
        if not stdin.isatty():
            raise RuntimeError("AsyncRawInput object should be attached to a tty-like fd, but non-tty file provided")

    def __del__(self):
        self.end()

    def prepare(self):
        self.old_tcattrs = termios.tcgetattr(self.stdin.fileno())
        os.set_blocking(self.stdin.fileno(), False)
        tty.setraw(self.stdin.fileno())

    def end(self):
        if self.is_reading and self.input_formats:
            self.stdout.write(self.input_formats[1])
            self.stdout.flush()
        termios.tcsetattr(self.stdin.fileno(), termios.TCSANOW, self.old_tcattrs)
        self.is_reading = False

    def set_interrupt_handler(self, awaitable):
        self.ctrl_c = awaitable

    def add_keystroke(self, keystroke: bytes, awaitable):
        self.keystrokes[keystroke] = awaitable

    def write(self, msg, **formats):
        self.stdout.write(('\r%s' + carriage_return(msg) + '%s') % format_term(**formats))

    def writeln(self, msg, **formats):
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

    def redraw_lastinp(self, at):
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
                    elif ord(key[0]) == 13:
                        # submit the input
                        break
                elif ord(key[0]) == 27 and len(key) == 3:
                    # probably arrow keys or other keystrokes
                    keystroke = ''.join(key)
                    if keystroke in self.keystrokes:
                        await self.keystrokes[keystroke]()
                        continue
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


async def background_task(inp, amount):
    for i in range(amount):
        inp.writeln('Hello world! [%s]' % ', '.join(inp.history), fgcolor=colors.YELLOW)
        await asyncio.sleep(0.5)


async def _process_inputs(amount: int):
    rawprint('handling %s inputs' % amount)
    inp = AsyncRawInput(history=[])
    inp.prepare()
    asyncio.create_task(background_task(inp, 10))
    # response = await inp.prompt_keystroke('Are you sure? y/n: ')
    # if response.lower() == 'n':
    #    return
    for i in range(amount):
        # rawprint('\r\33[0K' + ', '.join(str(x) for x in await inp.prompt_line()))
        line = await inp.prompt_line()
        inp.writeln('\r\33[0K' + line, bold=True, fgcolor=colors.GREEN)
        # rawprint(await rinput())
    rawprint('handling is complete')


class ARILogHandler(Handler):
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
