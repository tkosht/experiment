from semantic_kernel import SKContext
from semantic_kernel.skill_definition import sk_function

from app.codeinterpreter.component.session import (
    CodeInterpreterResponse,
    CodeInterpreterSession,
)


class CodeInterpeterPython(object):
    def __init__(self, port=7890) -> None:
        self.session = CodeInterpreterSession(port=port)

    async def astart(self) -> None:
        await self.session.astart()

    @sk_function(
        name="code_interpreter_in_python",
        description="""Pythonのコードインタープリターです。指定されたPythonコードを実行して結果を返します。データ分析に関わるタスクに使うとよいでしょう。""",
        input_description="specify python notebook codes which are user input or previous output",
    )
    def run(self, context: SKContext) -> str:
        # request: str = context["input"]
        # response: CodeInterpreterResponse = await self.session.generate_response(
        #     request
        # )
        # return response
        return "AAA"

    async def astop(self):
        await self.session.astop()
