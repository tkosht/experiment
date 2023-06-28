import os
import random
import re
import time
from threading import BoundedSemaphore, Lock, Thread

import openai
from dotenv import load_dotenv
from omegaconf import DictConfig
from tqdm import tqdm
from typing_extensions import Self

from app.base.component.logger import Logger
from app.llm_toolmaker.bbh import get_task, get_wrapper

g_logger = Logger(logger_name="app")
load_dotenv()
openai.api_key = os.environ["OPENAI_API_KEY"]

# cf. https://github.com/ctlllll/LLM-ToolMaker.git


def pickup_func(wrapper: str):
    func = re.findall(r"```python\n(.*?)\n```", wrapper, re.DOTALL)[0]
    return func


option_map = {
    1: "(A)",
    2: "(B)",
    3: "(C)",
    4: "(D)",
    5: "(E)",
    6: "(F)",
    7: "(G)",
    8: "(H)",
    9: "(I)",
    10: "(J)",
    "A": "(A)",
    "B": "(B)",
    "C": "(C)",
    "D": "(D)",
    "E": "(E)",
    "F": "(F)",
    "G": "(G)",
    "H": "(H)",
    "I": "(I)",
    "J": "(J)",
}


def get_option(ans):
    assert isinstance(ans, str)
    if ans in option_map:
        return option_map[ans]
    return ans


class LlmToolUser(object):
    def __init__(
        self,
        wrapper: str,
        task_name: str = "example_task",
        model_name: str = "gpt-4",
        temperature: float = 0.3,
        max_tokens: int = 2048,
        n_retry: int = 3,
    ) -> None:
        self.wrapper: str = wrapper
        self.task_name: str = task_name
        self.model_name: str = model_name
        self.temperature: float = temperature
        self.max_tokens: int = max_tokens
        self.n_retry: int = n_retry

        assert self.model_name[: len("gpt")] == "gpt"

    def _params(self, messages: list[dict]):
        params = {
            "model": self.model_name,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": messages,
        }
        return params

    def _llm(self, **params):
        response = openai.ChatCompletion.create(**params)["choices"][0]["message"][
            "content"
        ]
        return response

    def generate(self, prompt):
        params = self._params(messages=[{"role": "user", "content": prompt}])
        for n_retry in range(self.n_retry):
            try:
                return self._llm(**params)
            except Exception as e:
                if "Rate limit" in " ".join(e.args):
                    sleep_seconds = 15 + 2**n_retry + random.random()
                    errmsg = re.sub(r"org-\w+", "org-" + ("x" * 24), f"{e}")
                    g_logger.warning(f"{errmsg} ... try to retry [{sleep_seconds=}]")
                    time.sleep(sleep_seconds)
                else:
                    g_logger.warning(f"{e} ... try to retry")
        raise Exception("Failed to generate")

    def is_option_selection(self, ans: str, sample: dict):
        return (
            "Options:" in sample["question"]
            and ans not in option_map.keys()
            and ans not in option_map.values()
        )

    def make_answer_from_sample(self, task: str, sample: dict):
        prompt = self.wrapper + "\n\nQuestion: " + sample["question"]
        ans = self.make_answer(prompt=prompt)

        if self.is_option_selection(ans, sample):
            options = (
                re.findall(r"Options:(.*)", sample["question"], re.DOTALL)[0]
                .strip()
                .split("\n")
            )
            for option in options:
                if ans in option:
                    ans = option.split(" ")[0]
                    break

        if task == "schedule_meeting":
            if ans is None:
                ans = "No time slot works."
            elif isinstance(ans, list) or isinstance(ans, tuple):
                ans = f"{ans[0]} - {ans[1]}"

        return get_option(ans)

    def make_answer(self, prompt: str):
        caller = self.generate(prompt)
        func_call = pickup_func(caller)
        func_def = pickup_func(self.wrapper)

        exec_code = func_def + "\n" + func_call
        _ = exec(exec_code, globals())  # output: printed texts
        answer_variable = re.findall(r"(ans.*?) =", func_call, re.DOTALL)[-1]
        ans = globals()[answer_variable]

        return ans


class LlmToolEvaluator(object):
    def __init__(
        self,
        wrapper: str,
        task_name: str = "example_task",
        model_name: str = "gpt-3.5-turbo",
        max_threads: int = 8,
    ) -> None:
        self.wrapper: str = wrapper
        self.task_name: str = task_name
        self.model_name: str = model_name

        self.max_threads: int = max_threads
        self.pool = BoundedSemaphore(self.max_threads)
        self.lock = Lock()

        self.tool_user = LlmToolUser(
            wrapper=wrapper, model_name=model_name, task_name=task_name
        )

        self.n_totals: int = 0
        self.n_corrects: int = 0

    def run(self, sample: dict):
        with self.pool:
            try:
                ans = self.tool_user.make_answer_from_sample(
                    task=self.task_name, sample=sample
                )
            except Exception as e:
                ans = f"Error: {e}"
            with self.lock:
                self.n_totals += 1
                if str(ans) == str(sample["answer"]):
                    self.n_corrects += 1
                else:
                    g_logger.info(f"incorrect: {ans=} / {sample['answer']=}")
                acc = self.n_corrects / self.n_totals
                g_logger.info(f"Thread Accuracy: {acc:.4f}")

    def eval(self, testset: list) -> Self:
        threads = []
        for sample in tqdm(testset, desc="creating threads"):
            thr = Thread(target=self.run, args=(sample,))
            threads.append(thr)
            thr.start()
            # self.run(sample)  # for debugging
            # break

        thr_bar = tqdm(threads, desc="waiting threads: ")
        for thr in thr_bar:
            thr.join()

        acc = self.n_corrects / self.n_totals
        g_logger.info(f"Last Accuracy: {acc:.4f}")
        return self


def _main(params: DictConfig):
    task_name = params.task_name
    trainset, validsdet, testset = get_task(task=task_name)
    wrapper = get_wrapper(task=task_name, tooldir="./llm_tools")

    lte = LlmToolEvaluator(task_name=task_name, wrapper=wrapper)
    lte.eval(testset=testset[:50])


# @from_config(params_file="conf/app.yml", root_key="/train")
# def config(cfg: DictConfig):
def config():
    cfg = DictConfig(dict(task_name="word_sorting"))
    return cfg


def main(
    task_name: str = None,
):
    s = signature(main)
    kwargs = {}
    for k in list(s.parameters):
        v = locals()[k]
        if v is not None:
            kwargs[k] = v

    params = config()  # use as default
    params.update(kwargs)
    return _main(params)


if __name__ == "__main__":
    from inspect import signature

    import typer

    typer.run(main)
