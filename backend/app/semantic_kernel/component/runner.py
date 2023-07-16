import os

import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import (  # AzureTextCompletion,; AzureTextEmbedding,; ; OpenAITextCompletion,   # noqa
    OpenAIChatCompletion, OpenAITextEmbedding)
from semantic_kernel.orchestration.sk_function_base import SKFunctionBase
from semantic_kernel.planning.basic_planner import BasicPlanner
from semantic_kernel.planning.plan import Plan
from typing_extensions import Self


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
        from app.semantic_kernel.component.skills.search.search_local import \
            SearchLocal
        from app.semantic_kernel.component.skills.search.search_web import \
            SearchWeb

        self.skills.append(self.kernel.import_skill(SearchLocal(), "SearchLocal"))
        self.skills.append(self.kernel.import_skill(SearchWeb(), "SearchWeb"))
        self.skills.append(self.kernel.import_native_skill_from_directory(skill_dir, "math"))
        self.skills.append(self.kernel.import_native_skill_from_directory(skill_dir, "answer"))
        return self

    async def do_plan(self, input_query: str, prompt: str = None) -> Plan:
        params = dict(goal=input_query, kernel=self.kernel)
        if prompt:
            params.update(dict(prompt=prompt))
        plan: Plan = await self.planner.create_plan_async(**params)
        return plan

    async def do_execute(self, plan: Plan) -> Self:
        response = await self.planner.execute_plan_async(plan, self.kernel)
        return response
