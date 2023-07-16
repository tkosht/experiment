from semantic_kernel import SKContext
from semantic_kernel.skill_definition import sk_function


class Answer:
    skill_name = "Answer"

    @sk_function(
        name="answer",
        description="LLMを使って回答を生成する際に利用します",
        input_description="query from user's input"
    )
    def answer(self, context: SKContext) -> str:
        return f"""「{context['input']}」の回答です ...
---
「NotebookLM」というこの機能は、ノート作成ソフトウェアに対するGoogleの答えであり、言語モデルをその中核に据えたものとなっている。NotebookLMは一般的なチャットボットとは異なり、ユーザーが既に持っているコンテンツをベースに、ユーザーのコンテンツ理解をAIによって深めていけるようにする。

Googleは同社ブログに「NotebookLMと従来のAIチャットボットの大きな違いは、NotebookLMではユーザーのノートや情報源を『土台にして』言語モデルが稼働するところにある」と記している。

例えば、NotebookLMに「Googleドキュメント」の文書をドロップすると、その要約が自動作成されるとともに、重要なトピックや、ユーザーがその内容をより深く理解できるようにするための質問が提示される。
"""
