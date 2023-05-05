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
from langchain.utilities import PythonREPL

from app.langchain.component.agent_executor import CustomAgentExecutor
from app.langchain.component.agents.initialize import initialize_agent
from app.langchain.component.agents.prompt import (FORMAT_INSTRUCTIONS, PREFIX,
                                                   SUFFIX)

FINAL_ANSWER_ACTION = "Final Answer:"


# NOTE: cf. /usr/local/lib/python3.10/dist-packages/langchain/agents/chat/output_parser.py
class CustomOutputParser(AgentOutputParser):
    def get_format_instructions(self) -> str:
        return FORMAT_INSTRUCTIONS

    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        if FINAL_ANSWER_ACTION in text or "最終回答:" in text:
            return AgentFinish(
                {"output": text.split(FINAL_ANSWER_ACTION)[-1].strip()}, text
            )
        try:
            action = text.split("```")[1]
            response = json.loads(action.strip())
            return AgentAction(response["action"], response["action_input"], text)

        except Exception as e:
            print(f"{e=} / {text=}")
            return AgentFinish({"output": text.strip()}, text)


def build_agent():
    load_dotenv()
    # llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    # llm = ChatOpenAI(temperature=0, model_name="gpt-4-0314")
    llm = ChatOpenAI(temperature=0, model_name="gpt-4")

    python_repl = PythonREPL()

    python_tool = Tool(
        name="python_repl",
        description="A Python shell. Use this to execute python commands. "
                    "Input should be a valid python command. "
                    "If you want to see the output of a value, "
                    "you should print it out with `print(...)`.",
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

    def fake(msg: str):
        return msg

    trans_tool = Tool(
        name="trans_tool",
        description="A translation LLM. Use this to translate in japanese, "
                    "Input should be a short string or summary which you have to know exactly "
                    "and which with `Question` content and your `Thought` content in a sentence/statement.",
        func=exec_llm
    )
    summary_tool = Tool(
        name="summary_tool",
        description="A summarization LLM. Use this to summarize the result of tools like 'wikipedia' or 'serpapi', "
                    "or to parse result of using tools like 'requests'. "
                    "Input should be a short string or summary which you have to know exactly "
                    "and which with `Question` content and your `Thought` content in a sentence/statement.",
        func=exec_llm
    )
    no_tools = Tool(
        name="no_tools",
        description="Use this to respond your answer which you THOUGHT"
                    "Input should be a short string or summary which you have to know exactly "
                    "and which with `Question` content and your `Thought` content in a sentence/statement.",
        func=fake
    )
    # tools = load_tools(["serpapi", "llm-math", "wikipedia", "requests"], llm=llm)   # , "terminal"
    # tools = load_tools(["serpapi", "llm-math", "wikipedia"], llm=llm)   # , "terminal"
    tools = load_tools(["google-search", "llm-math", "wikipedia", "requests"], llm=llm)   # , "terminal"
    tools += [python_tool, trans_tool, summary_tool, no_tools]

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
        ),
        **kwargs,
    )
    assert agent_executor.return_intermediate_steps
    return agent_executor
