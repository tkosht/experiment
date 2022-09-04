from __future__ import annotations

import os

from app.component.slack.slack_event_data import SlackEventData
from dotenv import load_dotenv
from slack_sdk import WebClient

load_dotenv(".env")


class SlackClient(object):
    def __init__(self) -> None:
        self.client = WebClient(token=os.environ["SLACK_API_TOKEN"])

    def test(self):
        response = self.client.api_test()
        print(response)

    def send(
        self, message: str, slack_event: SlackEventData, blocks: list[dict] = None
    ):
        response = self.client.chat_postMessage(
            channel=slack_event.channel,
            thread_ts=slack_event.thread_ts,
            text=message,
            blocks=blocks,
        )
        print(response)
        return response

    def add_reaction(self, slack_event: SlackEventData, reaction: str = "thumbsup"):
        response = self.client.reactions_add(
            channel=slack_event.channel, name=reaction, timestamp=slack_event.ts
        )
        print(response)
        return response

    def remove_reaction(self, slack_event: SlackEventData, reaction: str = "thumbsup"):
        response = self.client.reactions_remove(
            channel=slack_event.channel, name=reaction, timestamp=slack_event.ts
        )
        print(response)
        return response


if __name__ == "__main__":
    import json

    slc = SlackClient()
    with open("data/sample.json", "r") as frj:
        event_data: dict = json.load(frj)

    slack_event = SlackEventData(event_data)

    print(f"{slack_event.channel=}")
    print(f"{slack_event.thread_ts=}")

    message = "Hello world!"
    slc.send(message, slack_event)

    # image_url = "http://placekitten.com/500/500"      # ketten
    image_url: str = "https://448b-124-25-245-158.jp.ngrok.io/image?image_id=01G8E74VC29GV0D8G2NV6NM615"
    msg: str = "An incredibly amazing wordcloud!"
    blocks = [
        {
            "type": "image",
            "title": {
                "type": "plain_text",
                "text": "[test] Please enjoy this wordcloud",
            },
            "block_id": "image4",
            "image_url": image_url,
            "alt_text": msg,
        }
    ]
    slc.send(msg, slack_event, blocks)
