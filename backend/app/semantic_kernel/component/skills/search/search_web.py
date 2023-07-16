from langchain.agents import load_tools
from langchain.tools.google_search.tool import GoogleSearchRun
from langchain.utilities.google_search import GoogleSearchAPIWrapper
from semantic_kernel import SKContext
from semantic_kernel.skill_definition import sk_function


class CustomGoogleSearchAPIWrapper(GoogleSearchAPIWrapper):
    def run(self, query: str) -> str:
        results = self._google_search_results(query, num=self.k)
        if len(results) == 0:
            return Exception("No good Google Search Result was found")
        return results


class SearchWeb(object):
    def __init__(self, n_search_results: int = 5) -> None:
        self.n_search_results = n_search_results
        self.tool: GoogleSearchRun = load_tools(["google-search"], llm=None)[0]
        self.tool.api_wrapper = CustomGoogleSearchAPIWrapper(k=n_search_results)

    @sk_function(
        name="web",
        description="ニュース等のWeb上の情報を検索するときに使用します",
        input_description="user input or previous output"
    )
    async def search(self, context: SKContext) -> str:
        query = context['input']
        print(f"SearchWeb.search: {query=}")
        searched_results = self.tool.run(tool_input=query)

        responses = []
        for res in searched_results:
            resdic = {
                "source_url": res["link"],
                "content": res["snippet"],
            }
            responses.append(resdic)

        return str(responses)
