import sys
from io import StringIO

from langchain.utilities import PythonREPL


class CustomPythonREPL(PythonREPL):
    def run(self, command: str) -> str:
        """Run command with own globals/locals and returns anything printed."""
        old_stdout = sys.stdout
        sys.stdout = mystdout = StringIO()

        # NOTE: support multiple commands
        command_list: list[str] = [command]
        if isinstance(command, list):
            command_list: list[str] = command

        output: str = ""
        try:
            for cmd in command_list:
                exec(cmd, self.globals, self.locals)
                sys.stdout = old_stdout
                output += mystdout.getvalue() + "\n"
        except Exception as e:
            sys.stdout = old_stdout
            output = str(e)
        return output
