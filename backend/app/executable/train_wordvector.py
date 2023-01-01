import joblib
import typer

from app.component.models.model import TextSequences
from app.component.models.pipeline import Pipeline
from app.component.models.vectorizer import VectorizerWord2vec
from app.domain.models.tokenizer import TokenizerWord
from app.infra.wikidb import WikiDb, WikiRecord


def main(
    n_limit: int = -1, mode: str = "train", pipe_file: str = "data/pipe_wikivec.gz"
):
    # pickup wiki data
    wdb = WikiDb(mode=mode)
    records = wdb.select(n_limit=n_limit)
    X: TextSequences = [WikiRecord(*rec).paragraph.splitlines() for rec in records]

    pipe_vectorizer = Pipeline(
        steps=[(TokenizerWord(), None), (VectorizerWord2vec(), None)]
    )

    pipe_vectorizer.fit(X)
    # y = pipe_vectorizer(X)

    joblib.dump(pipe_vectorizer, pipe_file, compress=("gzip", 3))


if __name__ == "__main__":
    typer.run(main)
