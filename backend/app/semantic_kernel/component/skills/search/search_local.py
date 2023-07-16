from semantic_kernel import SKContext
from semantic_kernel.skill_definition import sk_function


class SearchLocal:
    @sk_function(
        name="search",
        description="Webにはない専門的な内容や社内の情報について検索するときに使用します",
        input_description="user input or previous output"
    )
    def search(self, context: SKContext) -> str:
        query = context['input']
        print(f"SearchLocal.search: {query=}")
        result_dict = {
            "original_query": query,
            "source": "None",
            "content": "今日のニュースはわかりません・・"
        }

        return str(result_dict)
