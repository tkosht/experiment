from __future__ import annotations

import datetime
import re
import threading
import traceback as tb

import ulid
from app.component.params import add_args
from app.component.slack.slack_client import SlackClient
from app.component.slack.slack_event_data import SlackEventData
from app.domain.localdb_provider import LocalDbProvider
from app.domain.slack_block_builder import build_block_image
from app.domain.wordcloud_maker import WordCloudMaker


def event_handler(event_data: dict):
    slack_event = SlackEventData(event_data)
    handler = EventHandler(slack_event)
    handler.event_action()
    return event_data


class EventHandler(object):
    lock = threading.Lock()
    board_processing = {}

    def __init__(self, slack_event: SlackEventData) -> None:
        self.slack_event = slack_event

    @add_args(
        params_file="conf/app.yml", root_key="/slack/server/localdb", as_default=False
    )
    def store_db(self, slack_event: SlackEventData, do_store: bool):
        if not do_store:
            return

        # NOTE: DB に、イベントデータを記録
        ldp = None
        try:
            ldp = LocalDbProvider()
            ldp.connect()
            ldp.insert(slack_event)
        except Exception as e:
            print(e, tb.format_exc())

        finally:
            if ldp is not None:
                ldp.close()
        return

    def is_processing(self) -> bool:
        slack_event = self.slack_event
        event_key: str = slack_event.client_msg_id
        assert event_key is not None
        with self.lock:
            if event_key not in self.board_processing:
                self.board_processing[event_key] = datetime.datetime.now()
                return False
        return True

    def event_action(self):
        slack_event = self.slack_event

        if self.is_processing():
            print(
                "Suppressed: Possibly resent message: "
                f"{slack_event.type=} {slack_event.subtype=} {slack_event.user=} {slack_event.is_bot=}"
            )
            return

        print(f"{slack_event.type=}")
        self.store_db(slack_event)

        if slack_event.type == "message":
            self.event_action_message()
        elif slack_event.type == "reaction_added":
            self.event_action_reaction_added()
        elif slack_event.type == "reaction_removed":
            self.event_action_reaction_removed()
        else:
            # NOTE: Do Nothing
            print(f"Unknown {slack_event.type=}")
            pass
        return

    def _extract_urls(self, text: str) -> list[str]:
        urls = []
        for line in text.split("¥n"):
            _urls = re.findall(r"https?://(?:[-\w.]|/|(?:%[\da-fA-F]{2}))+", line)
            urls.extend(_urls)
        return urls

    @add_args(
        params_file="conf/app.yml",
        root_key="/slack/server/action/hello",
        as_default=False,
    )
    def find_hello_message(self, patterns: list[tuple(str, str)]) -> str:
        text = self.slack_event.text.strip().lower()
        for ptn, msg in patterns:
            if re.search(ptn.lower(), text):
                return msg
        return None

    def event_action_message(self):
        slack_event = self.slack_event
        # subtype があるメッセージは、何もしない
        if slack_event.subtype:
            print(
                f"Suppressed: {slack_event.type=} {slack_event.subtype=} {slack_event.user=} {slack_event.is_bot=}"
            )
            return

        # bot メッセージは、何もしない
        if slack_event.is_bot:
            print(
                f"Suppressed: {slack_event.type=} {slack_event.subtype=} {slack_event.user=} {slack_event.is_bot=}"
            )
            return

        slc = SlackClient()

        hello_message: str = self.find_hello_message()
        if hello_message is not None:
            slc.send(message=hello_message, slack_event=slack_event)
            return

        text: str = slack_event.text
        print(f"{text=}")

        urls = self._extract_urls(text)
        if not urls:
            # if urls is not contained
            return

        image_id: str = str(ulid.new())
        url = urls[0]
        wcm = WordCloudMaker(url)
        print(f"extracted {url=}")
        wcm.make(file_id=image_id)
        print("WordCloudMaker Done.")

        # NOTE: replies to slack thread
        msg: str = "An incredibly amazing wordcloud!"
        image_block = build_block_image(image_id=image_id, msg=msg)

        slc.send(msg, slack_event, blocks=[image_block])

        return

    def event_action_reaction_added(self):
        reaction = self.slack_event.event["reaction"]
        print(f"added: {reaction=}")
        slc = SlackClient()
        slc.add_reaction(self.slack_event, reaction)
        return

    def event_action_reaction_removed(self):
        reaction = self.slack_event.event["reaction"]
        print(f"removed: {reaction=}")
        slc = SlackClient()
        slc.remove_reaction(self.slack_event, reaction)
        return
