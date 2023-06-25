import json
import re

import requests


# cf. https://github.com/ctlllll/LLM-ToolMaker.git
def get_task(task: str):
    url: str = (
        f"https://raw.githubusercontent.com/ctlllll/LLM-ToolMaker/main/bbh/{task}.json"
    )
    res = requests.get(url)
    data = json.loads(res.text)

    # For dyck languages task, we need remove the spaces in the inputs to avoid unnecessary issues with tokenization
    if task == "dyck_languages":
        for example in data["examples"]:
            desc, input = example["input"].split("Input: ")
            input = input.replace(" ", "")
            example["input"] = f"{desc}Input: {input}"
            example["target"] = example["target"].replace(" ", "")

    train = []
    val = []
    test = []
    for index in range(len(data["examples"])):
        sample = {
            "question": data["examples"][index]["input"],
            "answer": data["examples"][index]["target"],
        }
        if index < 5:
            train.append(sample)
        elif index < 10:
            val.append(sample)
        else:
            test.append(sample)
    return train, val, test


def get_tool(task, tooldir="./tools"):
    message = json.load(open(f"{tooldir}/{task}.json"))
    wrapper = message[-1]["content"]
    func = re.findall(r"```python\n(.*?)\n```", wrapper, re.DOTALL)[0]
    return wrapper, func


def get_wrapper(task, tooldir="./tools") -> str:
    message = json.load(open(f"{tooldir}/{task}.json"))
    wrapper = message[-1]["content"]
    return wrapper
