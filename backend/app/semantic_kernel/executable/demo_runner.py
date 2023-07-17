import asyncio

import streamlit as st
from dotenv import load_dotenv

from streamlit_extras.colored_header import colored_header
from streamlit_extras.add_vertical_space import add_vertical_space


load_dotenv()

st.set_page_config(page_title="Runner Chat")

with st.sidebar:
    st.title("💬Chat with Runner App")
    query_example = """今日の言語モデルに関するニュースを調べて情報源を含めて正確にわかりやすくまとめてくれますか？"""
    label_example = """example:"""
    st.text_input(label_example, value=f"{query_example}", disabled=True)
    add_vertical_space(5)
    st.write("by [tkosht](https://github.com/tkosht)")


if "bot_chat" not in st.session_state:
    st.session_state["bot_chat"] = []

if "user_chat" not in st.session_state:
    st.session_state["user_chat"] = []

# Layout of input/response containers
input_container = st.container()
colored_header(label="", description="", color_name="blue-30")
response_container = st.container()


# User input
def get_text():
    label = """メッセージを入力します"""
    input_text = st.text_input(label, value="")
    return input_text


with input_container:
    user_input = get_text()


# Response output
async def generate_response(prompt: str):
    from app.semantic_kernel.component.planner import CustomPlanner
    from app.semantic_kernel.component.runner import SimpleRunner

    skill_dir = "./app/semantic_kernel/component/skills/"
    runner = SimpleRunner(planner=CustomPlanner(), skill_dir=skill_dir)
    response = await runner.do_run(prompt)
    # response = "Hello"
    return response


def load_logo(logo_file: str):
    from PIL import Image

    img = Image.open(logo_file)
    img = img.resize((32, 32))

    return img


def chat_message(message: str, logo_file: str):
    logo = load_logo(logo_file=logo_file)
    col_rates = [0.05, 0.95]
    col1, col2 = st.columns(col_rates)
    with col1:
        st.image(logo)
    with col2:
        st.write(message)


async def do_chat():
    with response_container:
        if user_input:
            response = await generate_response(user_input)
            st.session_state.user_chat.append(user_input)
            st.session_state.bot_chat.append(response)

        if st.session_state["bot_chat"]:
            for i in range(len(st.session_state["bot_chat"])):
                # user
                user_message = st.session_state["user_chat"][i]
                chat_message(user_message, logo_file="data/logo.jpeg")

                # bot
                bot_message = st.session_state["bot_chat"][i]
                chat_message(bot_message, logo_file="data/logo_bot.jpeg")


asyncio.run(do_chat())
