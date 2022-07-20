from __future__ import annotations

import re
import traceback as tb

from app.component.slack_event_data import SlackEventData
from app.domain.localdb_provider import LocalDbProvider
from app.domain.wordcloud_maker import WordCloudMaker

import ulid


def event_handler(event_data: dict):
    slack_event = SlackEventData(event_data)
    event_action(slack_event)
    return


def event_action(slack_event: SlackEventData):
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

    print(f"{slack_event.type=}")

    if slack_event.type == "message":
        event_action_message(slack_event)
    elif slack_event.type == "reaction_added":
        event_action_reaction_add(slack_event)
    else:
        # NOTE: Do Nothing
        pass
    return


def _extract_urls(text: str) -> list[str]:
    urls = []
    for line in text.split("¥n"):
        _urls = re.findall('https?://(?:[-\w.]|/|(?:%[\da-fA-F]{2}))+', line)
        urls.extend(_urls)
    return urls


def event_action_message(slack_event: SlackEventData):
    # subtype があるメッセージは、何もしない
    if slack_event.subtype:
        print(f"Suppressed: {slack_event.type=} {slack_event.subtype=} {slack_event.user=} {slack_event.is_bot=}")
        return

    # bot メッセージは、何もしない
    if slack_event.is_bot:
        print(f"Suppressed: {slack_event.type=} {slack_event.subtype=} {slack_event.user=} {slack_event.is_bot=}")
        return

    text: str = slack_event.text
    print(f"{text=}")

    urls = _extract_urls(text)
    if not urls:
        # if urls is not contained
        return

    file_id: str = str(ulid.new())
    url = urls[0]
    wcm = WordCloudMaker(url)
    print(f"extracted {url=}")
    wcm.make(file_id)
    print("WordCloudMaker Done.")
    return


def event_action_reaction_add(slack_event: SlackEventData):
    reaction = slack_event.event["reaction"]
    print(f"{reaction=}")
    return
