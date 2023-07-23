import base64
import json

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
        description="""グラフ化やデータ加工やモデル作成等のデータ分析タスクの結果を返します""",
        input_description="user input or previous output",
    )
    async def run(self, context: SKContext) -> str:
        request: str = context["input"]
        response: CodeInterpreterResponse = await self.session.generate_response(
            request
        )
        resdic = dict(
            text=response.content,
            images=[base64.b64encode(fl.content).decode() for fl in response.files],
        )
        res = json.dumps(resdic)
        return res

    async def astop(self):
        await self.session.astop()
