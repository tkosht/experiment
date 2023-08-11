"""Example Streamlit chat UI that exposes a Feedback button and link to LangSmith traces."""

import streamlit as st
from dotenv import load_dotenv
from langchain.callbacks.tracers.run_collector import RunCollectorCallbackHandler
from langchain.schema.runnable import RunnableConfig

from app.codeinterpreter.component.session import (
    CodeInterpreter,
    CodeInterpreterResponse,
)
from app.langchain.component.chain.stream import get_chain, get_openai_type

load_dotenv()


user_avatar = "🙂"
ai_avatar = "🐳"

st.set_page_config(
    page_title="Chat LangSmith",
    page_icon=ai_avatar,
    layout="wide",
    initial_sidebar_state="collapsed",
)

f"# {ai_avatar}️ Whale Chat"


def _init_session_state(key: str, init_value):
    if key not in st.session_state:
        st.session_state[key] = init_value


# Initialize State(
_init_session_state("messages", init_value=[])
_init_session_state("images", init_value=[])


def init_codeinterpreter():
    st.session_state["codeinterpreter"] = cdp = CodeInterpreter(local=True)
    cdp.start()


def term_codeinterpreter():
    cdp: CodeInterpreter = st.session_state["codeinterpreter"]
    cdp.stop()


if "codeinterpreter" not in st.session_state:
    init_codeinterpreter()

# setup slidebar
st.sidebar.markdown(
    """
# Menu
"""
)
if st.sidebar.button("Clear message history"):
    print("Clearing message history")
    st.session_state.messages = []
    term_codeinterpreter()
    init_codeinterpreter()

_DEFAULT_SYSTEM_PROMPT = (
    "You are a cool and smart whale with the smartest AI brain. "
    "You love programming, coding, mathematics, Japanese, and friendship! "
    "You MUST always answer in Japanese. "
)

system_prompt = st.sidebar.text_area(
    "Custom Instructions",
    _DEFAULT_SYSTEM_PROMPT,
    help="Custom instructions to provide the language model to determine style, personality, etc.",
)
system_prompt = system_prompt.strip().replace("{", "{{").replace("}", "}}")
chain, memory = get_chain(system_prompt)


# chat messages from history
ai_idx = 0
for msg in st.session_state.messages:
    streamlit_type = get_openai_type(msg)
    avatar = None
    avatar = user_avatar if streamlit_type == "user" else avatar
    avatar = ai_avatar if streamlit_type == "assistant" else avatar
    with st.chat_message(streamlit_type, avatar=avatar):
        st.markdown(msg.content)
        if streamlit_type == "assistant":
            imgs = st.session_state.images[ai_idx]
            for img in imgs:
                st.image(image=img)
            ai_idx += 1
    memory.chat_memory.add_message(msg)

latest_avatar_user = st.empty()
latest_avatar_ai = st.empty()

with st.container():
    with st.form(key="my_form", clear_on_submit=True):
        col1, col2 = st.columns([0.96, 0.04])
        with col1:
            message_example = "Plot the bitcoin chart of 2023 YTD"
            prompt = st.text_area(
                label=f"Message: ex) {message_example}", key="input", value=""
            )
        with col2:
            with st.empty():
                st.markdown("<br>" * 3, unsafe_allow_html=True)
            submit_button = st.form_submit_button(label="🫧")
        # default_message = "Plot the bitcoin chart of 2023 YTD"
        # prompt = st.text_area(
        #     label=f"Message: ex) {default_message}", key="input", value=""
        # )
        # submit_button = st.form_submit_button(label="🫧")


def parse_response(response: CodeInterpreterResponse):
    text = ""
    img = None
    imgs = []

    try:
        text: str = response.content

        imgs = []
        for fl in response.files:
            img = fl.get_image()
            if img.mode not in ("RGB", "L"):  # L is for greyscale images
                img = img.convert("RGB")
            imgs.append(img)
    except Exception as e:
        print(e)
        if not text:
            text = str(e)

    return text, imgs


run_collector = RunCollectorCallbackHandler()
runnable_config = RunnableConfig(
    callbacks=[run_collector],
    tags=["Streamlit Chat"],
)

if submit_button and prompt:
    with latest_avatar_user:
        with st.chat_message("user", avatar=user_avatar):
            st.write(f"{prompt}")

    # chat streaming
    with latest_avatar_ai:
        with st.chat_message("assistant", avatar=ai_avatar):
            _msg_area_ai = st.empty()
            _image_area_ai = st.empty()

        # with _msg_area_ai:
        #     st.markdown("(少々お待ち下さい・・)")

        with _msg_area_ai:
            text: str = ""
            with st.spinner("少々お待ち下さい・・"):
                cdp: CodeInterpreter = st.session_state["codeinterpreter"]
                response = cdp.generate_response_sync(user_msg=prompt)
                _text, imgs = parse_response(response)

            # NOTE: just in japanese
            msg = f"""以下を日本語にしてください。
            ```
            {_text}
            ```
            """

            # streaming output
            for chunk in chain.stream({"input": msg}, config=runnable_config):
                text += chunk.content
                st.markdown(text + "▌")
            st.markdown(text)

            # display
            with st.empty():
                st.markdown(text)

        with _image_area_ai:
            for img in imgs:
                st.image(image=img)

    memory.save_context({"input": prompt}, {"output": text})
    st.session_state.messages = memory.buffer
    st.session_state.images.append(imgs)
