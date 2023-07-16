from semantic_kernel import SKContext
from semantic_kernel.skill_definition import sk_function


class SearchWeb:
    @sk_function(
        name="web",
        description="ニュース等のWeb上の情報を検索するときに使用します",
        input_description="user input or previous output"
    )
    def web(self, context: SKContext) -> str:
        query = context['input']
        result_dict = {
            "original_query": query,
            "source": f"https://search.test.com?query={query}",
            "content": "「ChatGPT」は米OpenAI社がリリースした大規模言語モデルの一種。特徴として自然言語でAIとやり取りできることが挙げられる。"
        }

        return str(result_dict)
