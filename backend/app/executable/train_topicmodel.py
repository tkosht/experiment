import joblib
import typer

from app.component.models.model import TextSequences
from app.component.models.pipeline import Pipeline
from app.component.models.vectorizer import VectorizerBoW
from app.domain.models.tokenizer import TokenizerWord
from app.domain.models.topic_model import TopicModel
from app.infra.wikidb import WikiDb, WikiRecord


def main(n_limit: int = -1, pipe_file: str = "data/pipe_topic.gz"):
    pipe_topic = Pipeline(
        steps=[
            (TokenizerWord(use_stoppoes=True), None),
            (VectorizerBoW(), None),
            (TopicModel(n_topic=50), None),
        ]
    )

    # pickup wiki data to analyze topic
    wdb = WikiDb(mode="test")
    records = wdb.select(n_limit=n_limit)
    X: TextSequences = [WikiRecord(*rec).paragraph.splitlines() for rec in records]
    pipe_topic.fit(X)

    joblib.dump(pipe_topic, pipe_file, compress=("gzip", 3))


if __name__ == "__main__":
    typer.run(main)
