from __future__ import annotations

from app.component.slack_event_data import SlackEventData
from app.domain.db_interface import DbIf
from app.infra.localdb import LocalDb


class LocalDbProvider(object):
    def __init__(self) -> None:
        self.db = LocalDb()
        assert isinstance(self.db, DbIf)

    def connect(self) -> LocalDbProvider:
        self.db.connect()
        return self

    def close(self) -> LocalDbProvider:
        self.db.close()
        return self

    def insert(self, slack_event: SlackEventData) -> LocalDbProvider:
        event_id: str = slack_event.event_id
        event_type: str = slack_event.type
        event_data: str = slack_event.to_json()

        self.db.insert(event_id, event_type, event_data)
        return self
