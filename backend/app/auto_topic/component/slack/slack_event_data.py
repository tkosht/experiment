from __future__ import annotations

import json

from app.component.ulid import build_ulid


class SlackEventData(object):
    def __init__(self, event_data: dict) -> None:
        # inner data
        self.event_id: str = build_ulid(prefix="Ev")
        self.event_data: str = event_data

        # meta data
        self.slack_event_id: str = event_data["event_id"]
        self.authorizations: dict = event_data["authorizations"]
        self.team_id: str = event_data["team_id"]
        self.api_app_id: str = event_data["api_app_id"]
        self.event_context: str = event_data["event_context"]
        self.event_time: int = event_data["event_time"]
        self.event: str = event_data["event"]

        # event data
        self.client_msg_id: str = self.event.get("client_msg_id")
        if self.client_msg_id is None:
            if "message" in self.event:
                self.client_msg_id: str = self.event["message"]["client_msg_id"]
            else:
                self.client_msg_id: str = "DMY" + self.slack_event_id
        self.type: str = self.event["type"]
        self.subtype: str = self.event.get("subtype")
        self.text: str = self.event.get("text", "[no text found]")
        self.user: str = self.event.get("user", "[undefined user]")
        if self.type == "message":
            self.channel: str = self.event["channel"]
            self.ts: str = self.event["ts"]
        elif self.type in ["reaction_added", "reaction_removed"]:
            self.channel: str = self.event["item"]["channel"]
            self.ts: str = self.event["item"]["ts"]
        self.thread_ts: str = self.event.get("thread_ts", self.ts)

        print("Information:", f"{self.slack_event_id=}", f"{self.client_msg_id=}", f"{self.event_id}")

    @property
    def is_bot(self):
        # return self.authorizations[0]["is_bot"] # is NG
        return bool(self.event.get("bot_profile", False))

    def to_json(self) -> str:
        return json.dumps(self.event_data)
