# admin_console
Interactive and asynchronous stdin console written in pure Python and asyncio with extension support
Can be used together with any Python project that uses __asyncio__, especially servers and web-based apps.
Only POSIX operating systems are supported, such as Linux, \*BSD, Solaris etc.
Windows support isn't coming soon.
## Example usage
Simple quickstart that will bring on the working command prompt in __asyncio__ environment:
```python3
>>> import asyncio
>>> from admin_console import AdminCommandExecutor, basic_command_set, colors
>>>
>>> async def main():
...     cmd = AdminCommandExecutor(use_config=False)  # do not load config.json in the current directory
...     basic_command_set(cmd)  # exit, extlist, extload, extunload etc.
...     # Custom prompt formatting
...     cmd.promptheader = "Tutorial! "
...     cmd.promptarrow = "->"
...     cmd.prompt_formats['bold'] = True
...     cmd.prompt_formats['fgcolor'] = colors.GREEN
...     await cmd.load_extensions()  # will create an "extensions/" in the working directory
...     await cmd.prompt_loop()  # blocks until > exit is invoked
... 
>>> 
>>> if __name__ == "__main__":
...     asyncio.run(main())
Note: create extdep.txt in the extensions folder to sequentally load modules
Tutorial! -> 
```

Documentation: https://nobwow.github.io/admin_console.html
## Installation/Updating (from git)

`pip install -U git+https://github.com/NobWow/admin-console-python.git`
