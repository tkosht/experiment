import json
import re
from typing import Union

from dotenv import load_dotenv
from langchain.agents import AgentOutputParser, AgentType, Tool, load_tools
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)
from langchain.schema import AgentAction, AgentFinish

from app.langchain.component.agent.agent_executor import CustomAgentExecutor
from app.langchain.component.agent.initialize import initialize_agent
from app.langchain.component.agent.prompt import FORMAT_INSTRUCTIONS, PREFIX, SUFFIX
from app.langchain.component.tools.custom_python import CustomPythonREPL
from app.langchain.component.tools.custom_shell import CustomShellTool

# from langchain.agents.chat.prompt import FORMAT_INSTRUCTIONS, PREFIX, SUFFIX
# from app.langchain.component.llms.redpajama import RedPajamaLLM

FINAL_ANSWER_ACTION = "Final Answer:"


# NOTE: cf. https://github.com/hwchase17/langchain/blob/master/langchain/agents/chat/output_parser.py
# Copyright (c) Harrison Chase
# Copyright (c) 2023 Takehito Oshita
class CustomOutputParser(AgentOutputParser):
    def get_format_instructions(self) -> str:
        return FORMAT_INSTRUCTIONS

    def parse(self, text: str) -> Union[list[AgentAction], AgentFinish]:
        if FINAL_ANSWER_ACTION in text or "最終回答:" in text:
            return AgentFinish(
                {"output": text.split(FINAL_ANSWER_ACTION)[-1].strip()}, text
            )
        try:
            actions = self._parse_action(text)
            return actions

        except Exception as e:
            # TODO: in this case, may clear the previous agent_executor.intermediate_steps
            print("-" * 80, f"{e.__repr__()} / {str(text)=}", "-" * 80, "", sep="\n")
            return AgentAction(
                "error_analyzing_tool",
                f"Please analyze this parsing error ({e.__repr__()}) "
                "for your Answer (Must Check: $JSON_BLOB format) "
                f"step-by-step with this your answer: '{text}'",
                text,
            )

    def _parse_action(self, text: str):
        _text = re.sub(r"```\w+\n", "```\n", text)
        parsed = _text.split("```")
        if len(parsed) < 3:
            print("=" * 100)
            print("[DEBUG]")
            print(f"{text=}")
            print("=" * 100)
            if "success" in text:
                return AgentFinish({"output": text}, text)
            raise Exception("Invalid Answer Format: missing '```'")

        actions = []
        for idx in range(1, len(parsed), 2):
            action = parsed[idx]
            action = action.strip()
            if not action:
                continue
            try:
                response = json.loads(action)
            except Exception:
                response = dict(action="python_repl", action_input=action)

            agent_action = AgentAction(
                response["action"], response["action_input"], text
            )
            actions.append(agent_action)
        return actions


def build_agent(
    model_name="gpt-3.5-turbo", temperature: float = 0, max_iterations: int = 15
) -> CustomAgentExecutor:
    load_dotenv()
    llm = ChatOpenAI(temperature=temperature, model_name=model_name)
    # llm = RedPajamaLLM()

    shell_tool = CustomShellTool()

    python_repl = CustomPythonREPL()
    python_tool = Tool(
        name="python_repl",
        description="A Python shell. "
        "Use this to execute python commands."
        "Input(`action_input`) should be a valid python command with python code grammar. "
        "If you want to see the output of a value, "
        "you should print it out with `print(...)`. "
        "NOTICE that this python commands should be excluded notebook way",
        func=python_repl.run,
    )

    memory = ConversationBufferMemory(return_messages=True)

    def exec_llm(msg: str):
        system_template = (
            "SYSTEM: Thougt step-by-step precisely, and exact summary at last"
        )
        human_template = "HUMAN: {input}"
        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(system_template),
                MessagesPlaceholder(variable_name="history"),
                HumanMessagePromptTemplate.from_template(human_template),
            ]
        )

        conversation = ConversationChain(memory=memory, prompt=prompt, llm=llm)
        guess = conversation.predict(input=msg)
        return guess

    trans_tool = Tool(
        name="trans_tool",
        description="A translation LLM. Use this to translate in japanese, "
        "Input should be a short string or summary which you have to know exactly "
        "and which with `Question` content and your `Thought` content in an Input sentence/statement. "
        "NEVER input the url only",
        func=exec_llm,
    )
    summary_tool = Tool(
        name="summary_tool",
        description="A summarization LLM. Use this to summarize the result of the tools "
        "like 'wikipedia' or 'serpapi', 'google-search', "
        "but NEVER use this tool for parsing contents like HTML or XML"
        "Input should be a short string or summary which you have to know exactly "
        "and which with `Question` content and your `Thought` content in an Input sentence/statement. "
        "NEVER input the url only",
        func=exec_llm,
    )
    error_analyzation_tool = Tool(
        name="error_analyzing_tool",
        description="An error analyzation LLM. Use this to analyze to fix the error results of the tools "
        "like 'python_reple' or 'terminal, "
        "if invalid format error, advise the $JSON_BLOB format, surely. "
        "Especially, if being thought as execution is impossible, you advise to just an answer using Aciton: with $JSON_BLOB. "  # noqa
        "Input should be a short string or summary which you have to know exactly "
        "and which with `Question` content and your `Thought` content in an Input sentence/statement. "
        "NEVER input the url only",
        func=exec_llm,
    )

    # def fake(msg: str):
    #     return msg

    # no_action = Tool(
    #     name="no_tools",
    #     description="No Action/No need to use any tools. Use this to respond your answer which you THOUGHT directly "
    #                 "Input should be a string.",
    #     func=fake
    # )

    # tools = load_tools(["serpapi", "llm-math", "wikipedia", "requests_all"], llm=llm)   # , "terminal"
    # tools += [python_tool, shell_tool, trans_tool, summary_tool, error_analyzation_tool, no_action]
    tools = load_tools(
        ["google-search", "llm-math", "wikipedia"], llm=llm
    )  # , "terminal"
    tools += [python_tool, shell_tool, trans_tool, summary_tool, error_analyzation_tool]

    kwargs = dict(memory=memory, return_intermediate_steps=True)

    agent_executor: CustomAgentExecutor = initialize_agent(
        tools,
        llm,
        agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        agent_kwargs=dict(
            prefix=PREFIX,
            suffix=SUFFIX,
            format_instructions=FORMAT_INSTRUCTIONS,
            output_parser=CustomOutputParser(),
            max_iterations=max_iterations,
            max_execution_time=None,
            early_stopping_method="force",
            handle_parsing_errors=False,
        ),
        **kwargs,
    )
    assert agent_executor.return_intermediate_steps
    return agent_executor
