import joblib
import typer
from tqdm import tqdm

from app.component.logger import Logger
from app.component.models.model import TextSequences
from app.component.models.pipeline import Pipeline
from app.component.models.vectorizer import VectorizerWord2vec
from app.domain.models.tokenizer import TokenizerWord
from app.infra.wikidb import WikiDb, WikiRecord

g_logger = Logger(logger_name="train_wordvector")


def log_info(*args, **kwargs):
    g_logger.info(*args, **kwargs)


def main(
    n_limit: int = -1,
    mode: str = "train",
    batch_size: int = 10000,
    pipe_file: str = "data/pipe_wikivec.gz",
):
    log_info("Start", "train_wordvector")

    # pickup wiki data
    wdb = WikiDb(mode=mode)
    log_info("Start", "Select WikiDb")
    records = wdb.select(n_limit=n_limit)
    log_info("End", "Select WikiDb")

    log_info("Start", "Make WikiRecord")
    X: TextSequences = [WikiRecord(*rec).paragraph.splitlines() for rec in records]
    log_info("End", "Make WikiRecord")

    log_info("Start", "Create Pipeline")
    pipe_vectorizer = Pipeline(
        steps=[
            (
                TokenizerWord(
                    use_stoppoes=False, filterpos=["名詞", "動詞"], use_orgform=True
                ),
                None,
            ),
            (VectorizerWord2vec(min_count=1), None),
        ]
    )
    log_info("End", "Create Pipeline")

    log_info("Start", "Fit Wiki data")
    n = len(X)
    for bch_idx, offset in enumerate(tqdm(range(0, n, batch_size))):
        log_info("Processing ...", f"{bch_idx=}")
        bch = X[offset : offset + batch_size]
        pipe_vectorizer.fit(bch)
    log_info("End", "Fit Wiki data")

    log_info("Start", "Dump Pipeline for Wiki vectorizer")
    joblib.dump(pipe_vectorizer, pipe_file, compress=("gzip", 3))
    log_info("End", "Dump Pipeline for Wiki vectorizer")

    log_info("End", "train_wordvector")


if __name__ == "__main__":
    typer.run(main)
