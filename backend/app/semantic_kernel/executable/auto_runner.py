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
from semantic_kernel.planning.basic_planner import BasicPlanner
from semantic_kernel.planning.plan import Plan
from typing_extensions import Self

load_dotenv()


class AutoRunner(object):
    def __init__(self, skill_dir=None) -> None:
        self.planner: BasicPlanner = BasicPlanner()
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

    async def run(self, plan: Plan) -> Self:
        response = await self.planner.execute_plan_async(plan, self.kernel)
        return response

    async def run_custom(self, plan: Plan) -> Self:
        if not plan.generated_plan.result:
            raise Exception(plan.generated_plan)
        generated_plan = json.loads(plan.generated_plan.result)
        print(f"{generated_plan=}")

        context = ContextVariables()
        context["input"] = generated_plan["input"]
        subtasks = generated_plan["subtasks"]

        result = ""
        for subtask in subtasks:
            skill_name, function_name = subtask["function"].split(".")
            sk_function = self.kernel.skills.get_function(skill_name, function_name)

            args = subtask.get("args", None)
            if args:
                for key, value in args.items():
                    context[key] = value
            output = await sk_function.invoke_async(variables=context)
            context["input"] = result = output.result
        return result


async def _main(params: DictConfig):
    runner = AutoRunner(skill_dir="./app/semantic_kernel/component/skills/")
    prompt = """今日の言語モデルに関するニュースを調べて100文字以内にまとめます
"""

    plan: Plan = await runner.do_plan(input_query=prompt)
    print("-" * 50)
    print("runner.run()")
    response = await runner.run(plan)
    print(f"{response=}")
    print("-" * 50)
    print("runner.run_custom()")
    result = await runner.run_custom(plan)
    print(f"{result=}")


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
