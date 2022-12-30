import joblib

from app.component.models.model import TextSequences
from app.component.models.pipeline import Pipeline
from app.component.models.vectorizer import VectorizerWord2vec
from app.domain.models.tokenizer import TokenizerWord
from app.infra.wikidb import WikiDb, WikiRecord

if __name__ == "__main__":
    # pickup wiki data
    wdb = WikiDb(mode="train")
    # records = wdb.select(n_limits=)
    records = wdb.select(n_limits=64)
    X: TextSequences = [WikiRecord(*rec).paragraph.split("¥n") for rec in records]

    pipe = Pipeline(steps=[(TokenizerWord(), None), (VectorizerWord2vec(), None)])

    pipe.fit(X)
    y = pipe(X)

    pipe_file: str = "data/pipe_wikivec.gz"
    joblib.dump(pipe, pipe_file, compress=("gzip", 3))
