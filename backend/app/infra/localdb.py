import sqlite3

from app.domain.db_interface import DbIf

g_log = print


class LocalDb(DbIf):
    def __init__(self, db_file="data/local.db") -> None:
        self.db_file = db_file
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
        g_log("CREATE TABLE slack_events", "Start")
        self.execute(
            """CREATE TABLE IF NOT EXISTS slack_events (
            event_id STRING NOT NULL,
            event_type STRING,
            event_data TEXT,
            status STRING,
            png BLOB,
            created_at DATETIME NOT NULL,
            updated_at DATETIME,
            PRIMARY KEY (event_id)
        )
        """
        )
        g_log("CREATE TABLE slack_events", "End")

    def insert(self, event_id: str, event_type: str, event_data: str):
        values = [event_id, event_type, event_data, "UNPROCESSED"]
        self.execute(
            """INSERT INTO slack_events (
            event_id, event_type, event_data, status,
            created_at
        ) VALUES (?, ?, ?, ?, datetime('now', 'localtime'))
        """,
            values,
        )
