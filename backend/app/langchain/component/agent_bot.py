import base64
import time
from io import BytesIO

import PIL.Image
# from gradio.components.image_editor import EditorValue
from langchain.memory import ConversationBufferMemory
from langchain.schema import AgentAction
from langchain_openai import ChatOpenAI
from langchain.schema.messages import HumanMessage

from app.langchain.component.agent.agent_builder import build_agent, init
from app.langchain.component.agent.agent_executor import (  # AgentExecutor,
    CustomAgentExecutor,
)
from app.langchain.component.callbacks.simple import TextCallbackHandler


def _update_text(log_text: str, cb: TextCallbackHandler):
    return "\n".join(cb.log_texts)


def encode_image_from_pil(pil_image):
    buffered = BytesIO()
    pil_image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


class SimpleBot(object):
    def __init__(self) -> None:
        self.memory = ConversationBufferMemory(return_messages=True)
        self.intermediate_steps: list[tuple[AgentAction, str]] = []
        self._callback = TextCallbackHandler(targets=["CustomAgentExecutor"])

    def gr_chat(
        self,
        history: list[str],
        model_name: str,
        temperature_percent: int,
        max_iterations: int,
        context: str,  # deprecated ?
        img: PIL.Image = None,
    ):
        query = history[-1][0]
        answer = self.run(query, model_name, temperature_percent, max_iterations, img)
        assert answer
        history[-1][1] = answer  # may be changed url to href

        ctx = "\n".join([str(step[0]) for step in self.intermediate_steps])
        return history, ctx

    def run(
        self,
        query: str,
        model_name: str,
        temperature_percent: int,
        max_iterations: int,
        img: PIL.Image = None,
    ):
        # TODO: implement using `img`

        self.model_name = model_name
        self.temperature_percent = temperature_percent
        self.max_iterations = max_iterations

        if img is not None:
            init()
            base64_image = encode_image_from_pil(img)
            chat = ChatOpenAI(model="gpt-4-turbo", max_tokens=1024)
            msg = chat.invoke(
                [
                    HumanMessage(
                        content=[
                            {"type": "text", "text": query},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                            },
                        ]
                    )
                ]
            )
            return msg

        agent_executor: CustomAgentExecutor = build_agent(
            model_name,
            temperature=temperature_percent / 100,
            max_iterations=max_iterations,
            memory=self.memory,
        )
        query_org = query
        intermediate_steps = []

        max_retries = 20
        for _ in range(max_retries):
            try:
                answer = agent_executor.run(
                    input=query,
                    intermediate_steps=intermediate_steps,
                    callbacks=[self._callback],
                )
                break
            except Exception as e:
                print("-" * 80)
                print(f"# {e.__repr__()} / in run()")
                print("-" * 80)
                answer = f"Error Occured: '{e}' / couldn't be fixed."
                query = f"Observation: \nERROR: {str(e)}\n\nwith fix this Error in other way, "
                f"\n\nHUMAN: {query_org}\nThougt:"
                if "This model's maximum context length is" in str(e):
                    n_context = 1
                    if len(agent_executor.intermediate_steps) <= n_context:
                        # NOTE: already 'n_context' intermediate_steps, but context length error. so, restart newly
                        query = query_org
                        agent_executor.intermediate_steps = []
                        agent_executor.memory.clear()
                    else:
                        agent_executor.intermediate_steps = (
                            agent_executor.intermediate_steps[-n_context:]
                        )
                        # agent_executor.memory.chat_memory.messages = agent_executor.memory.chat_memory.messages[:1]
                else:
                    # NOTE: retry newly
                    query = query_org
                    agent_executor.intermediate_steps = []

            finally:
                intermediate_steps = agent_executor.intermediate_steps
                print("<" * 80)
                print("Next:")
                print(f"{query=}")
                print("-" * 20)
                print(f"{len(agent_executor.intermediate_steps)=}")
                print(">" * 80)
            print("waiting retrying ... (a little sleeping)")
            time.sleep(1)

        return answer

    def gr_clear_context(self, context: str):
        self.intermediate_steps.clear()
        return "\n".join(self.intermediate_steps)

    def gr_update_text(self, log_text: str):
        log = _update_text(log_text, self._callback)
        return log

    def gr_clear_text(self):
        self._callback.log_texts.clear()
        log = _update_text("", self._callback)
        return log
