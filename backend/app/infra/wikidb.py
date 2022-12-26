from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from app.component.ulid import build_ulid
from app.domain.db_interface import DbIf

g_log = print


@dataclass
class WikiRecord(DbIf):
    document_id: str
    article: str = None
    section: str = None
    paragraph: str = None
    paragraph_id: str = None

    def __post_init__(self):
        self.paragraph_id = build_ulid(prefix="Prg")

    def from_dict(self, d: dict):
        self.article = d["article"]
        self.section = d["section"]
        self.paragraph = d["paragraph"]
        return self

    def to_record(self):
        return (
            self.document_id,
            self.paragraph_id,
            self.article,
            self.section,
            self.paragraph,
        )

    def assert_available(self):
        assert self.document_id is not None
        assert self.article is not None
        assert self.section is not None
        assert self.paragraph is not None
        assert self.paragraph_id is not None


class WikiDb(DbIf):
    def __init__(self, mode: str = "train", db_file: str = "data/wiki.db") -> None:
        self.mode = mode  # train|valid|test
        self.db_file = db_file

        self.table_name = f"{self.mode}_data"
        self.cnn: sqlite3.Connection = None
        self.csr: sqlite3.Cursor = None

    def connect(self):
        self.cnn = sqlite3.connect(self.db_file)
        self.csr = self.cnn.cursor()

    def close(self):
        if self.csr is not None:
            self.csr.close()
            self.csr = None
        if self.cnn is not None:
            self.cnn.close()
            self.cnn = None

    def execute(self, sql: str, values: list = []):
        self.csr.execute(sql, values)
        self.cnn.commit()

    def execute_many(self, sql: str, values: list = []):
        self.csr.executemany(sql, values)
        self.cnn.commit()

    def create_tables(self):
        g_log(f"CREATE TABLE {self.table_name}", "Start")
        self.execute(
            f"""CREATE TABLE IF NOT EXISTS {self.table_name} (
            document_id STRING NOT NULL,
            paragraph_id STRING NOT NULL,
            article STRING,
            section STRING,
            paragraph TEXT,
            created_at DATETIME NOT NULL,
            updated_at DATETIME,
            PRIMARY KEY (document_id, paragraph_id)
        )
        """
        )
        g_log(f"CREATE TABLE {self.table_name}", "End")

    def insert_sql(self) -> str:
        return f"""INSERT INTO {self.table_name} (
                document_id, paragraph_id, article, section, paragraph,
                created_at
            ) VALUES (?, ?, ?, ?, ?, datetime('now', 'localtime'))
            """

    def insert(
        self,
        document_id: str,
        paragraph_id: str,
        article: str,
        section: str,
        paragraph: str,
    ):
        values = [document_id, paragraph_id, article, section, paragraph]
        sql: str = self.insert_sql()
        self.execute(sql, values)

    def insert_many(self, values: list[tuple[str, str, str, str, str]]):
        # values[n] == (document_id, paragraph_id, article, section, paragraph)
        sql: str = self.insert_sql()
        self.execute_many(sql, values)

    def select(self, n_limits=-1):
        sql = f"""SELECT
                document_id, paragraph_id, article, section, paragraph, created_at
                FROM {self.table_name}
            """
        if n_limits > 0:
            sql += f" LIMIT {n_limits}"
        records = self.execute(sql)
        records = list(records)
        return records
