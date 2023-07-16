import json
from inspect import signature

from dotenv import load_dotenv
from omegaconf import DictConfig

from app.semantic_kernel.component.planner import CustomPlanner
from app.semantic_kernel.component.runner import SimpleRunner

load_dotenv()


async def _main(params: DictConfig):
    skill_dir = "./app/semantic_kernel/component/skills/"
    prompt = """今日の言語モデルに関するニュースを調べて100文字以内にまとめます
"""

    # NOTE: using BasicPlanner
    runner = SimpleRunner(skill_dir=skill_dir)
    print("-" * 50)
    plan = await runner.do_plan(input_query=prompt)
    print(f"generated_plan: {json.loads(plan.generated_plan.result)}")
    print("-" * 25)
    response = await runner.do_execute(plan)
    print(response)

    # NOTE: using CustomPlanner (just customized `temperature`)
    runner = SimpleRunner(planner=CustomPlanner(), skill_dir=skill_dir)
    print("-" * 50)
    plan = await runner.do_plan(input_query=prompt)
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
