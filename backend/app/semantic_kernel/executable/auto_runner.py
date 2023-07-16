import json
import os
from inspect import signature

import semantic_kernel as sk
from dotenv import load_dotenv
from omegaconf import DictConfig
from semantic_kernel.connectors.ai.open_ai import (  # AzureTextCompletion,; AzureTextEmbedding,; ; OpenAITextCompletion,   # noqa
    OpenAIChatCompletion, OpenAITextEmbedding)
from semantic_kernel.orchestration.context_variables import ContextVariables
from semantic_kernel.orchestration.sk_function_base import SKFunctionBase
from semantic_kernel.planning.basic_planner import PROMPT, BasicPlanner
from semantic_kernel.planning.plan import Plan
from typing_extensions import Self

load_dotenv()


# NOTE: cf. python3.10/dist-packages/semantic_kernel/planning/basic_planner.py
class CustomPlanner(BasicPlanner):
    def __init__(self, max_tokens: int = 1000, temperature: float = 0.0) -> None:
        super().__init__()

        self.max_tokens = max_tokens
        self.temperature = temperature

    async def create_plan_async(
        self,
        goal: str,
        kernel: sk.Kernel,
        prompt: str = PROMPT,
    ) -> Plan:
        planner = kernel.create_semantic_function(
            prompt, max_tokens=self.max_tokens, temperature=self.temperature
        )

        available_functions_string = self._create_available_functions_string(kernel)

        context = ContextVariables()
        context["goal"] = goal
        context["available_functions"] = available_functions_string
        generated_plan = await planner.invoke_async(variables=context)
        return Plan(prompt=prompt, goal=goal, plan=generated_plan)

    async def execute_plan_async(self, plan: Plan, kernel: sk.Kernel) -> str:
        if not plan.generated_plan.result:
            raise Exception(plan.generated_plan)
        generated_plan = json.loads(plan.generated_plan.result)

        context = ContextVariables()
        context["input"] = generated_plan["input"]
        subtasks = generated_plan["subtasks"]

        result = ""
        for subtask in subtasks:
            skill_name, function_name = subtask["function"].split(".")
            sk_function = kernel.skills.get_function(skill_name, function_name)

            args = subtask.get("args", None)
            if args:
                for key, value in args.items():
                    context[key] = value
            output = await sk_function.invoke_async(variables=context)
            context["input"] = result = output.result
        return result


class AutoRunner(object):
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


async def _main(params: DictConfig):
    skill_dir = "./app/semantic_kernel/component/skills/"
    prompt = """今日の言語モデルに関するニュースを調べて100文字以内にまとめます
"""

    # NOTE: using BasicPlanner
    runner = AutoRunner(skill_dir=skill_dir)
    print("-" * 50)
    plan: Plan = await runner.do_plan(input_query=prompt)
    print(f"generated_plan: {json.loads(plan.generated_plan.result)}")
    print("-" * 25)
    response = await runner.do_execute(plan)
    print(response)

    # NOTE: using CustomPlanner (just customized `temperature`)
    runner = AutoRunner(planner=CustomPlanner(), skill_dir=skill_dir)
    print("-" * 50)
    plan: Plan = await runner.do_plan(input_query=prompt)
    print(f"generated_plan: {json.loads(plan.generated_plan.result)}")
    response = await runner.do_execute(plan)
    print(response)


# @from_config(params_file="conf/app.yml", root_key="/train")
# def config(cfg: DictConfig):
def config():
    cfg = DictConfig(dict(is_experiment=True, do_share=False))
    return cfg


def main(
    do_share: bool = None,
):
    s = signature(main)
    kwargs = {}
    for k in list(s.parameters):
        v = locals()[k]
        if v is not None:
            kwargs[k] = v

    params = config()  # use as default
    params.update(kwargs)

    import asyncio

    return asyncio.run(_main(params))


if __name__ == "__main__":
    import typer

    typer.run(main)
