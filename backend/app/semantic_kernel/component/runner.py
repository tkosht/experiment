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
    def __init__(
        self,
        planner: BasicPlanner = BasicPlanner(),
        skill_dir=None,
        model_name: str = "gpt-3.5-turbo",
    ) -> None:
        self.planner: BasicPlanner = planner
        self.skill_dir = skill_dir
        self.model_name = model_name

        self.kernel: sk.Kernel = None
        self.skills: list[dict[str, SKFunctionBase]] = []
        self.memory_store = sk.memory.VolatileMemoryStore()

        self.setup_kernel(model_name=model_name)

    def setup_kernel(self, model_name: str = "gpt-3.5-turbo") -> Self:
        kernel = sk.Kernel()

        api_key = os.environ.get("OPENAI_API_KEY")
        org_id = os.environ.get("OPENAI_ORG_ID")

        kernel.add_chat_service(
            "gpt", OpenAIChatCompletion(model_name, api_key, org_id)
        )
        kernel.add_text_embedding_generation_service(
            "ada", OpenAITextEmbedding("text-embedding-ada-002", api_key, org_id)
        )
        kernel.register_memory_store(memory_store=self.memory_store)
        self.kernel = kernel

        self.setup_skills(skill_dir=self.skill_dir)
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
        input_query = f"""[GOALここから]
以下の`- ユーザの依頼`について実行してください。ただし、最後の回答はユーザの依頼に正確に自然言語でお願いします

- ユーザの依頼
(((
{user_query}
)))
[GOALここまで]
"""
        for _ in range(n_retries):
            try:
                print("-" * 50)
                print("input_query:", input_query)
                plan: Plan = await self.do_plan(input_query=input_query)
                print(f"generated_plan: {plan.generated_plan.result}")
                print(f"{json.loads(plan.generated_plan.result)}")
                print("-" * 25)
                response = await self.do_execute(plan)
                break
            except Exception as e:
                input_query = f"""
- ユーザの依頼
(((
{user_query}
)))

- あなたは、直前に以下のようなプランを実行しました
(((
{plan.generated_plan.result}
)))

- しかし以下のようなエラーが発生しました
(((
{e}
)))

- 以上の結果を踏まえて、プランの見直しを検討してください
"""
                print(input_query)
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
