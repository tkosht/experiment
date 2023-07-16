import json

import semantic_kernel as sk
from semantic_kernel.orchestration.context_variables import ContextVariables
from semantic_kernel.planning.basic_planner import PROMPT, BasicPlanner
from semantic_kernel.planning.plan import Plan


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
