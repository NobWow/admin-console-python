import asyncio
import sys
from src.admin_console import *


async def main(args):
    if len(args) > 0:
        extpath = args[0]
    else:
        extpath = 'extensions/'
    console = AdminCommandExecutor(use_config=False)
    if extpath != "no":
        await console.load_extensions()
    else:
        console.info("not loading extensions")
    basic_command_set(console)
    console.promptheader = "~test console~ "
    console.promptarrow = ":"
    console.prompt_format = {'bold': True, "fgcolor": colors.BLUE}
    await console.prompt_loop()

if __name__ == "__main__":
    asyncio.run(main(sys.argv))
