import os

import openai
from dotenv import load_dotenv
from semantic_kernel import SKContext
from semantic_kernel.skill_definition import (sk_function,
                                              sk_function_context_parameter)

load_dotenv()
openai.api_key = os.environ["OPENAI_API_KEY"]


class Answer:
    skill_name = "AnswerSkill"

    def __init__(self, model_name: str = "gpt-4", temperature: float = 0.3, max_tokens: int = 2048) -> None:
        self.model_name: str = model_name
        self.temperature: float = temperature
        self.max_tokens: int = max_tokens

    def _params(self, messages: list[dict]):
        params = {
            "model": self.model_name,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": messages,
        }
        return params

    def _llm(self, message: str) -> str:
        params = self._params(messages=[{"role": "user", "content": message}])
        response: str = openai.ChatCompletion.create(**params)["choices"][0]["message"][
            "content"
        ]
        return response

    @sk_function(
        name="answer",
        description="任意のテキストに対して、大規模言語モデルを使用して要約やまとめなどの回答を生成する際に利用します",
        input_description="arbitrary text like user input or previous tool's/skill's output"
    )
    @sk_function_context_parameter(
        name="query",
        description="どのような回答を作成したいかを指示します。",
    )
    def make_answer(self, context: SKContext) -> str:
        input_text = context['input']
        query = context['query'] + " 特に、自ら文章を創作しないようにすること"
        print(f"Answer.make_answer: {input_text=}, {query=}")

        message = f"""以下のテキストに対して、以下の指示に従い文章を生成してください。
# テキスト
```
{input_text}
```

# 指示
```
{query}
```
"""

        response: str = self._llm(message=message)
        return response
