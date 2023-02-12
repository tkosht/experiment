import joblib
import typer
from tqdm import tqdm

from app.component.models.model import TextSequences
from app.component.models.pipeline import Pipeline
from app.component.models.vectorizer import VectorizerWord2vec
from app.component.simple_logger import log_info
from app.domain.models.tokenizer import TokenizerWord
from app.infra.wikidb import WikiDb, WikiRecord


def main(
    n_limit: int = -1, mode: str = "train", pipe_file: str = "data/pipe_wikivec.gz"
):
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
    bs = 128
    for bch_idx, offset in enumerate(tqdm(range(0, n, bs))):
        log_info("Processing ...", f"{bch_idx=}")
        bch = X[offset : offset + bs]
        pipe_vectorizer.fit(bch)
    # y = pipe_vectorizer(X)
    log_info("End", "Fit Wiki data")

    log_info("Start", "Dump Pipeline for Wiki vectorizer")
    joblib.dump(pipe_vectorizer, pipe_file, compress=("gzip", 3))
    log_info("End", "Dump Pipeline for Wiki vectorizer")


if __name__ == "__main__":
    typer.run(main)
