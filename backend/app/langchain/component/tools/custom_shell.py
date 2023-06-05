from typing import Any, Union

from langchain.tools import ShellTool


class CustomShellTool(ShellTool):
    def __init__(self) -> None:
        super().__init__()
        self.description += f"args {self.args}".replace("{", "{{").replace("}", "}}")

    def _parse_input(
        self,
        tool_input: Union[str, dict],
    ) -> Union[str, dict[str, Any]]:
        """Convert tool input to pydantic model."""
        input_args = self.args_schema
        if isinstance(tool_input, str):
            if input_args is not None:
                key_ = next(iter(input_args.__fields__.keys()))
                input_args.validate({key_: tool_input})
            return tool_input
        else:
            if input_args is not None:
                result = input_args.parse_obj(tool_input)
                return {k: v for k, v in result.dict().items() if k in tool_input}
        return tool_input

    # def _run(
    #     self,
    #     commands: Union[str, list[str]],
    #     run_manager: Optional[CallbackManagerForToolRun] = None,
    # ) -> str:
    #     """Run commands and return final output."""
    #     if isinstance(commands, list):
    #         pass

    #     return self.process.run(commands)
