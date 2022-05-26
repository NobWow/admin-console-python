# admin_console
Interactive and asynchronous stdin console written in pure Python and asyncio with extension support
Can be used together with any Python project that uses __asyncio__, especially servers and web-based apps.
Only POSIX operating systems are supported, such as Linux, \*BSD, Solaris etc.
Windows support isn't coming soon.
## Example usage
Simple quickstart that will bring on the working command prompt in __asyncio__ environment:
```python3
import asyncio
from admin_console import AdminCommandExecutor, basic_command_set, colors

async def main():
    console = AdminCommandExecutor(use_config=False)  # do not load config.json in the current directory
    basic_command_set(console)  # exit, extlist, extload, extunload etc.
    # Custom prompt formatting
    console.promptheader = "Tutorial! "
    console.promptarrow = "->"
    console.prompt_format['bold'] = True
    console.prompt_format['fgcolor'] = colors.GREEN
    await console.load_extensions()  # will create an "extensions/" in the working directory
    await console.prompt_loop()  # blocks until > exit is invoked


if __name__ == "__main__":
    asyncio.run(main())
# Note: create extdep.txt in the extensions folder to sequentally load modules
# Tutorial! -> 
```
## Event handling
Example of how to handle command dispatching and how to cancel some commands
```python3
from admin_console import AdminCommand

# Add handler
async def handler(cmd: AdminCommand, executor: AdminCommandExecutor, *args):
    executor.print("Command executed! If this command's name contains bad, it is cancelled!")
    if 'bad' in cmd.name:
        return False  # cancel!

console.cmdexec_event.add_handler(handler)

# Wait for the event
await console.cmdexec_event.wait_for_successful()
console.print("Some command is executed!")

# Wait until an event happens and then determine if this event passes
async with console.cmdexec_event.wait_and_handle() as handle:
    # success is True if this event is not cancelled, False otherwise
    # args is the list and kwargs is the dictionary of the arguments of an event
    success, args, kwargs = handle()
    if not success:
        # event is already cancelled
        pass
    elif args[0] = "bad" and kwargs["user"] = "hacker":
        # cancel an event
        handle(False)

# Trigger an event and decide if it happened correctly
custom_event = console.events['some_custom_event']  # collections.defaultdict creates a new instance if there is no such an element
async with custom_event.emit_and_handle("good", "argument", 123, before=True) as handle:
    # putting before=True to evaluate the code before handlers
    handle(True)  # make this event uncancellable, next time handle(False) call won't have an effect
```
## Extensions
Extensions are importable Python script files (name.py) and they should be put in the `extensions/` directory of the script.
Each extension script should have `async def extension_init(self)` and `async def extension_cleanup(self)` async function definitions,
where `self` is an instance of `AdminCommandExtension`. Example:
```python3
from admin_console import AdminCommandExtension


async def extension_init(self: AdminCommandExtension):
    self.msg("Welcome here! Registering commands...")
    async def my_command(cmd: AdminCommandExecutor, arg1: str, arg2: int, arg3: bool, arg4: float, arg5: str):
        # do it yourself
        pass
    async def my_command_tab(cmd: AdminCommandExecutor, *args, argl: str):
        _len = len(args)
        if argl:
            _len += 1
        if _len == 0:
            # arg1 is being tabbed, which is a single word
            return "foo", "bar", "lore", "bug"
        elif _len == 1:
            # arg2 is being tabbed, which is int
            # do not tabcomplete an integer...
            pass
        elif _len == 2:
            # it is possible to tabcomplete a boolean, which is just yes or no
            return "yes", "no"
    self.add_command(my_command, 'my-command', ((str, 'some word'), (int, 'your amount'), (bool, 'yes or no?'), (float, 'write precise PI here'), (None, 'long line...')), my_command_tab)
    # do LITERALLY anything at the extension load time, but be careful: it stops the other extensions from loading while this function is running


async def extension_cleanup(self: AdminCommandExtension):
    self.msg("Goodbye...")
```
Make a file at the extensions directory with the contents above. The extension should be `.py`, otherwise the script won't be imported.
To tell the extension loader to load scripts in the specified order, make `extdep.txt` in the `extensions/` directory
and fill the file names (without .py) on each line.
## Documentation
Simple rendered reference is here: https://nobwow.github.io/admin_console.html
## Installation/Updating (from git)

`pip install -U git+https://github.com/NobWow/admin-console-python.git`
