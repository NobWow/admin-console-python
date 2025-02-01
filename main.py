import asyncio
import sys
import logging
from src.admin_console import AdminCommandExecutor, basic_command_set, colors
from src.admin_console.ainput import ARILogHandler
from collections import defaultdict


async def list_asyncio_tasks(console: AdminCommandExecutor):
    while True:
        console.ainput.writeln('-------------------', fgcolor=9)
        for task in asyncio.all_tasks():
            console.print("Task pending: %s" % task)
        console.ainput.writeln('-------------------', fgcolor=9)
        await asyncio.sleep(5)


def commands(cmd: AdminCommandExecutor):
    async def editing_prompt(cmd: AdminCommandExecutor):
        res = await cmd.ainput.prompt_line(history_disabled=True, start_buffer="edit me")
        cmd.print(res)
    cmd.add_command(editing_prompt, "editing-prompt", description="Test editing prompt")


async def main(args):
    if len(args) > 0:
        extpath = args[0]
    else:
        extpath = 'extensions/'
    logger = logging.getLogger('main')
    logger.setLevel(logging.DEBUG)
    console = AdminCommandExecutor(defaultdict(lambda: None), use_config=False, logger=logger)
    console.config = defaultdict(lambda: None)
    _handler = ARILogHandler(console.ainput, level=logging.DEBUG)
    logger.addHandler(_handler)
    _handler.setFormatter(logging.Formatter('-> [%(levelname)s] %(message)s'))
    basic_command_set(console)
    commands(console)
    if extpath != "no":
        await console.load_extensions()
    else:
        console.info("not loading extensions")
    console.promptheader = "~test console~ "
    console.promptarrow = ":"
    console.prompt_format = {'bold': True, "fgcolor": colors.BLUE}
    await console.prompt_loop()

if __name__ == "__main__":
    asyncio.run(main(sys.argv))
