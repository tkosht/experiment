import json
from typing import Union

from dotenv import load_dotenv
from langchain.agents import AgentOutputParser, AgentType, Tool, load_tools
# from langchain.agents.chat.prompt import FORMAT_INSTRUCTIONS, PREFIX, SUFFIX
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import (ChatPromptTemplate, HumanMessagePromptTemplate,
                               MessagesPlaceholder,
                               SystemMessagePromptTemplate)
from langchain.schema import AgentAction, AgentFinish
from langchain.tools import ShellTool
from langchain.utilities import PythonREPL

from app.langchain.component.agent_executor import CustomAgentExecutor
from app.langchain.component.agents.initialize import initialize_agent
from app.langchain.component.agents.prompt import (FORMAT_INSTRUCTIONS, PREFIX,
                                                   SUFFIX)

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
            if "Action:" not in text:
                raise Exception("Invalid Answer Format: Not Found 'Action:'")
            parsed = text.split("```")
            if len(parsed) < 3:
                raise Exception("Invalid Answer Format: missing '```'")

            actions = []
            for idx in range(1, len(parsed), 2):
                action = parsed[idx]
                response = json.loads(action.strip())
                agent_action = AgentAction(response["action"], response["action_input"], text)
                actions.append(agent_action)
            return actions

        except Exception as e:
            # TODO: in this case, may clear the previous agent_executor.intermediate_steps
            print("-" * 80, f"{e.__repr__()} / {str(text)=}", "-" * 80, "", sep="\n")
            return AgentAction("error_analyzing_tool",
                               f"Please analyze this parsing error ({e.__repr__()}) for your Answer (HINT: $JSON_BLOB) "
                               f"step-by-step with this your answer: '{text}'", text)


def build_agent(model_name="gpt-3.5-turbo", temperature: float = 0, max_iterations: int = 15) -> CustomAgentExecutor:
    load_dotenv()
    llm = ChatOpenAI(temperature=temperature, model_name=model_name)

    shell_tool = ShellTool()
    shell_tool.description += f"args {shell_tool.args}".replace("{", "{{").replace("}", "}}")

    python_repl = PythonREPL()
    python_tool = Tool(
        name="python_repl",
        description="A Python shell. "
                    "Use this to execute python commands for requesting raw HTML, "
                    "or parsing any texts, "
                    "for also executing shell commands, and so on. "
                    "Input should be a valid python command. "
                    "If you want to see the output of a value, "
                    "you should print it out with `print(...)`. "
                    "NOTICE that this python shell is not notebook"
                    ,
        func=python_repl.run
    )

    memory = ConversationBufferMemory(return_messages=True)

    def exec_llm(msg: str):
        system_template = "SYSTEM: Thougt step-by-step precisely, and exact summary at last"
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
        func=exec_llm
    )
    summary_tool = Tool(
        name="summary_tool",
        description="A summarization LLM. Use this to summarize the result of the tools "
                    "like 'wikipedia' or 'serpapi', 'google-search', "
                    "but NEVER use this tool for parsing contents like HTML or XML"
                    "Input should be a short string or summary which you have to know exactly "
                    "and which with `Question` content and your `Thought` content in an Input sentence/statement. "
                    "NEVER input the url only",
        func=exec_llm
    )
    error_analyzation_tool = Tool(
        name="error_analyzing_tool",
        description="An error analyzation LLM. Use this to analyze to fix the error results of the tools "
                    "like 'wikipedia' or 'serpapi', 'google-search', "
                    "but NEVER use this tool for parsing contents like HTML or XML"
                    "Input should be a short string or summary which you have to know exactly "
                    "and which with `Question` content and your `Thought` content in an Input sentence/statement. "
                    "NEVER input the url only",
        func=exec_llm
    )
    # def fake(msg: str):
    #     return msg
    #
    # no_tools = Tool(
    #     name="no_tools",
    #     description="Use this to respond your answer which you THOUGHT"
    #                 "Input should be a short string or summary which you have to know exactly "
    #                 "and which with `Question` content and your `Thought` content in an Input sentence/statement.",
    #     func=fake
    # )
    # tools = load_tools(["serpapi", "llm-math", "wikipedia", "requests"], llm=llm)   # , "terminal"
    tools = load_tools(["google-search", "llm-math", "wikipedia"], llm=llm)   # , "terminal"
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
