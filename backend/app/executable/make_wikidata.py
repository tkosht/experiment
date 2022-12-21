from __future__ import annotations

import datetime
import traceback as tb

import tensorflow_datasets as tfds

from app.infra.wikidb import WikiDb, WikiRecord


def now() -> str:
    return datetime.datetime.now().strftime("%Y/%m/%d %T")


class WikiDbMaker(object):
    def __init__(self, mode: str) -> None:
        self.mode = mode  # train|valid|test

        self.n_intervals = 10000
        self.db = WikiDb(mode)
        self._init_db()

    def _init_db(self) -> WikiDbMaker:
        self.db.connect()
        self.db.create_tables()
        return self

    def execute(self, ds):
        try:
            self._populate(ds)
        except Exception as e:
            print(e, tb.format_exc())
        finally:
            self.db.close()
        return self

    def _populate(self, ds):
        records = []
        for rec in ds.as_numpy_iterator():
            lines = rec["text"].decode().split("\n")
            parsed = self.parse(lines)

            recs = [WikiRecord().from_dict(prg).to_record() for prg in parsed]
            records.extend(recs)
            if len(records) >= self.n_intervals:
                self.db.insert_many(records)
                records = []
        if len(records) > 0:
            self.db.insert_many(records)
        return self

    def parse(self, lines: list[str]) -> WikiDbMaker:
        parsed = []
        status = "none"
        record: dict[str, str] = dict(
            article="_NO_ARTICLE_",
            section="_NO_SECTION_",
            paragraph="_NO_PARAGRAPH_",
        )

        for line in lines:
            if line.strip() == "":
                continue
            if line == "_START_ARTICLE_":
                status = "article"
                continue
            if line == "_START_SECTION_":
                # end of paragraph
                if status == "paragraph":
                    parsed.append(record)

                # initialize record
                assert record["article"]
                record = dict(
                    article=record["article"],
                    section="_NO_SECTION_",
                    paragraph="_NO_PARAGRAPH_",
                )

                # update status
                status = "section"
                continue
            if line == "_START_PARAGRAPH_":
                status = "paragraph"
                continue

            assert status != "none"
            if status == "paragraph":
                line = line.replace("_NEWLINE_", "\n")
            record[status] = line

        assert status == "paragraph"
        parsed.append(record)
        return parsed


if __name__ == "__main__":
    (ds_train, ds_valid, ds_test), ds_info = tfds.load(
        name="wiki40b/ja", split=["train", "validation", "test"], with_info=True
    )

    batchs = [
        ("train", ds_train),
        ("valid", ds_valid),
        ("test", ds_test),
    ]

    for mode, ds in batchs:
        print(now(), "dataset:", mode, len(ds))
        maker = WikiDbMaker(mode)
        maker.execute(ds)
