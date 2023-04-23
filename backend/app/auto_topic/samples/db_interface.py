from __future__ import annotations


class DbIf(object):
    def connect(self) -> DbIf:
        raise NotImplementedError("connect()")

    def close(self) -> DbIf:
        # NOTE: 冪等性を担保すること
        raise NotImplementedError("close()")

    def create_tables(self) -> DbIf:
        raise NotImplementedError("create_tables()")

    def drop_tables(self) -> DbIf:
        raise NotImplementedError("drop_tables()")

    def insert(self, event_id: str, event_type: str, event_data: str) -> DbIf:
        # NOTE:
        # - event_id: ULID
        # - event_type: "message", "reaction_added", ...
        # - event_data: json string
        raise NotImplementedError("insert()")
