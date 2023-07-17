import json
import os

import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import (  # AzureTextCompletion,; AzureTextEmbedding,; ; OpenAITextCompletion,   # noqa
    OpenAIChatCompletion,
    OpenAITextEmbedding,
)
from semantic_kernel.orchestration.sk_function_base import SKFunctionBase
from semantic_kernel.planning.basic_planner import BasicPlanner
from semantic_kernel.planning.plan import Plan
from typing_extensions import Self, Any


class SimpleRunner(object):
    def __init__(self, planner: BasicPlanner = BasicPlanner(), skill_dir=None) -> None:
        self.planner: BasicPlanner = planner
        self.kernel: sk.Kernel = None

        self.skills: list[dict[str, SKFunctionBase]] = []

        self.setup_kernel()
        self.setup_skills(skill_dir=skill_dir)

    def setup_kernel(self, model_name: str = "gpt-3.5-turbo") -> Self:
        kernel = sk.Kernel()

        api_key = os.environ.get("OPENAI_API_KEY")
        org_id = os.environ.get("OPENAI_ORG_ID")

        kernel.add_chat_service(
            "gpt", OpenAIChatCompletion(model_name, api_key, org_id)
        )
        self.kernel = kernel
        return self

    def setup_skills(self, skill_dir: str = "./skills") -> Self:
        from semantic_kernel.core_skills.http_skill import HttpSkill

        from app.semantic_kernel.component.skills.search.search_local import SearchLocal
        from app.semantic_kernel.component.skills.search.search_web import SearchWeb

        self.skills.append(self.kernel.import_skill(SearchLocal(), "SearchLocal"))
        self.skills.append(self.kernel.import_skill(SearchWeb(), "SearchWeb"))
        self.skills.append(self.kernel.import_skill(HttpSkill(), "HttpSkill"))
        self.skills.append(
            self.kernel.import_native_skill_from_directory(skill_dir, "math")
        )
        self.skills.append(
            self.kernel.import_native_skill_from_directory(skill_dir, "answer")
        )
        return self

    async def do_run(self, user_query: str, n_retries: int = 3) -> str:
        input_query = user_query
        for _ in range(n_retries):
            try:
                print("-" * 50)
                plan: Plan = await self.do_plan(input_query=input_query)
                print(f"generated_plan: {plan.generated_plan.result}")
                print(f"{json.loads(plan.generated_plan.result)}")
                print("-" * 25)
                response = await self.do_execute(plan)
                break
            except Exception as e:
                input_query = f"""
# ユーザの依頼
```
{user_query}
```

# あなたは、先程は以下のようなプランを実行しました
```
{plan.generated_plan.result}
```

# しかし以下のようなエラーが発生しました
```
{e}
```

# この結果を踏まえて、プランの見直しを検討してください
"""
                continue
        print(response)
        return response

    async def do_plan(
        self, input_query: str, prompt: str = None, n_retries: int = 3
    ) -> Plan:
        params = dict(goal=input_query, kernel=self.kernel)
        _params = params.copy()
        if prompt:
            params.update(dict(prompt=prompt))
        plan: Plan = await self.planner.create_plan_async(**_params)
        return plan

    async def do_execute(self, plan: Plan) -> Any:
        response = await self.planner.execute_plan_async(plan, self.kernel)
        return response
