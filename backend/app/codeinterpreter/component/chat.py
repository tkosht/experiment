"""
このコードは `Dominic Bäumer` 氏 のプロジェクト(https://github.com/shroominic/codeinterpreter-api.git)を参考に作成しました。

オリジナルのライセンス:
- MIT License
"""

import asyncio
import json
import re
from typing import List, Union

from codeboxapi import CodeBox  # type: ignore
from langchain.agents import AgentOutputParser
from langchain.base_language import BaseLanguageModel
from langchain.chat_models.base import BaseChatModel
from langchain.output_parsers.json import parse_json_markdown
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.schema import (
    AgentAction,
    AgentFinish,
    AIMessage,
    BaseChatMessageHistory,
    HumanMessage,
    OutputParserException,
    SystemMessage,
)
from langchain.schema.messages import BaseMessage, messages_from_dict, messages_to_dict

code_interpreter_system_message = SystemMessage(
    content="""
Assistant is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics.
As a language model, Assistant is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.
Assistant is constantly learning and improving, and its capabilities are constantly evolving.
It is able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. Additionally, Assistant is able to generate its own text based on the input it receives,
allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.

This version of Assistant is called "Code Interpreter" and capable of using a python code interpreter (sandboxed jupyter kernel) to run code.
The human also maybe thinks this code interpreter is for writing code but it is more for data science, data analysis, and data visualization, file manipulation, and other things that can be done using a jupyter kernel/ipython runtime.
Tell the human if they use the code interpreter incorrectly.
Already installed packages are: (numpy pandas matplotlib seaborn scikit-learn yfinance scipy statsmodels sympy bokeh plotly dash networkx).
If you encounter an error, try again and fix the code.
"""  # noqa: E501
)

remove_dl_link_prompt = ChatPromptTemplate(
    input_variables=["input_response"],
    messages=[
        SystemMessage(
            content="The user will send you a response and you need "
            "to remove the download link from it.\n"
            "Reformat the remaining message so no whitespace "
            "or half sentences are still there.\n"
            "If the response does not contain a download link, "
            "return the response as is.\n"
        ),
        HumanMessage(
            content="The dataset has been successfully converted to CSV format. "
            "You can download the converted file [here](sandbox:/Iris.csv)."
        ),  # noqa: E501
        AIMessage(content="The dataset has been successfully converted to CSV format."),
        HumanMessagePromptTemplate.from_template("{input_response}"),
    ],
)

determine_modifications_prompt = PromptTemplate(
    input_variables=["code"],
    template="The user will input some code and you need to determine "
    "if the code makes any changes to the file system. \n"
    "With changes it means creating new files or modifying exsisting ones.\n"
    "Format your answer as JSON inside a codeblock with a "
    "list of filenames that are modified by the code.\n"
    "If the code does not make any changes to the file system, "
    "return an empty list.\n\n"
    "Determine modifications:\n"
    "```python\n"
    "import matplotlib.pyplot as plt\n"
    "import numpy as np\n\n"
    "t = np.arange(0.0, 4.0*np.pi, 0.1)\n\n"
    "s = np.sin(t)\n\n"
    "fig, ax = plt.subplots()\n\n"
    "ax.plot(t, s)\n\n"
    'ax.set(xlabel="time (s)", ylabel="sin(t)",\n'
    '   title="Simple Sin Wave")\n'
    "ax.grid()\n\n"
    'plt.savefig("sin_wave.png")\n'
    "```\n\n"
    "Answer:\n"
    "```json\n"
    "{{\n"
    '  "modifications": ["sin_wave.png"]\n'
    "}}\n"
    "```\n\n"
    "Determine modifications:\n"
    "```python\n"
    "import matplotlib.pyplot as plt\n"
    "import numpy as np\n\n"
    "x = np.linspace(0, 10, 100)\n"
    "y = x**2\n\n"
    "plt.figure(figsize=(8, 6))\n"
    "plt.plot(x, y)\n"
    'plt.title("Simple Quadratic Function")\n'
    'plt.xlabel("x")\n'
    'plt.ylabel("y = x^2")\n'
    "plt.grid(True)\n"
    "plt.show()\n"
    "```\n\n"
    "Answer:\n"
    "```json\n"
    "{{\n"
    '  "modifications": []\n'
    "}}\n"
    "```\n\n"
    "Determine modifications:\n"
    "```python\n"
    "{code}\n"
    "```\n\n"
    "Answer:\n"
    "```json\n",
)


def extract_python_code(
    text: str,
    llm: BaseLanguageModel,
    retry: int = 2,
):
    pass


# TODO: This is probably not efficient, but it works for now.
class CodeBoxChatMessageHistory(BaseChatMessageHistory):
    """
    Chat message history that stores history inside the codebox.
    """

    def __init__(self, codebox: CodeBox):
        self.codebox = codebox

        if "history.json" not in [f.name for f in self.codebox.list_files()]:
            name, content = "history.json", b"{}"
            if (loop := asyncio.get_event_loop()).is_running():
                loop.create_task(self.codebox.aupload(name, content))
            else:
                self.codebox.upload(name, content)

    @property
    def messages(self) -> List[BaseMessage]:  # type: ignore
        """Retrieve the messages from the codebox"""
        msgs = (
            messages_from_dict(json.loads(file_content.decode("utf-8")))
            if (
                file_content := (
                    loop.run_until_complete(self.codebox.adownload("history.json"))
                    if (loop := asyncio.get_event_loop()).is_running()
                    else self.codebox.download("history.json")
                ).content
            )
            else []
        )
        return msgs

    def add_message(self, message: BaseMessage) -> None:
        """Append the message to the record in the local file"""
        print("Current messages: ", self.messages)
        messages = messages_to_dict(self.messages)
        print("Adding message: ", message)
        messages.append(messages_to_dict([message])[0])
        name, content = "history.json", json.dumps(messages).encode("utf-8")
        if (loop := asyncio.get_event_loop()).is_running():
            loop.create_task(self.codebox.aupload(name, content))
        else:
            self.codebox.upload(name, content)
        print("New messages: ", self.messages)

    def clear(self) -> None:
        """Clear session memory from the local file"""
        print("Clearing history CLEARING HISTORY")
        code = "import os; os.remove('history.json')"
        if (loop := asyncio.get_event_loop()).is_running():
            loop.create_task(self.codebox.arun(code))
        else:
            self.codebox.run(code)


class CodeAgentOutputParser(AgentOutputParser):
    ai_prefix: str = "AI"

    def get_format_instructions(self) -> str:
        from langchain.agents.conversational.prompt import FORMAT_INSTRUCTIONS

        return FORMAT_INSTRUCTIONS

    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        if f"{self.ai_prefix}:" in text:
            return AgentFinish(
                {"output": text.split(f"{self.ai_prefix}:")[-1].strip()}, text
            )
        regex = r"Action: (.*?)[\n]*Action Input: (.*)"
        match = re.search(regex, text)
        if not match:
            raise OutputParserException(f"Could not parse LLM output: `{text}`")
        action = match.group(1)
        action_input = match.group(2)
        return AgentAction(action.strip(), action_input.strip(" ").strip('"'), text)

    @property
    def _type(self) -> str:
        return "conversational"


class CodeChatAgentOutputParser(AgentOutputParser):
    def get_format_instructions(self) -> str:
        from langchain.agents.conversational_chat.prompt import FORMAT_INSTRUCTIONS

        return FORMAT_INSTRUCTIONS

    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        raise NotImplementedError

    async def aparse(
        self, text: str, llm: BaseChatModel
    ) -> Union[AgentAction, AgentFinish]:
        try:
            response = parse_json_markdown(text)
            action, action_input = response["action"], response["action_input"]
            if action == "Final Answer":
                return AgentFinish({"output": action_input}, text)
            else:
                return AgentAction(action, action_input, text)
        except Exception:
            if '"action": "python"' in text:
                # extract python code from text with prompt
                text = extract_python_code(text, llm=llm) or ""
                match = re.search(r"```python\n(.*?)```", text)
                if match:
                    code = match.group(1).replace("\\n", "; ")
                    return AgentAction("python", code, text)
            raise OutputParserException(f"Could not parse LLM output: `{text}`")

    @property
    def _type(self) -> str:
        return "conversational_chat"
