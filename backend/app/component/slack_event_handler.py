import traceback as tb

from app.component.slack_event_data import SlackEventData
from app.domain.localdb_provider import LocalDbProvider


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


def event_action_message(slack_event: SlackEventData):
    # subtype があるメッセージは、何もしない
    if slack_event.subtype:
        print(f"{slack_event.subtype=}")
        return

    # bot メッセージは、何もしない
    if slack_event.is_bot:
        print(f"{slack_event.type=} {slack_event.subtype=}")
        return

    text: str = slack_event.text
    print(f"{text=}")

    return


def event_action_reaction_add(slack_event: SlackEventData):
    reaction = slack_event.event["reaction"]
    print(f"{reaction=}")
    return
