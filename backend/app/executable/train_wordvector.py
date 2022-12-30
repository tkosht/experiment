from app.component.models.pipeline import Pipeline
from app.domain.models.tokenizer import TokenizerWord

if __name__ == "__main__":
    pipe = Pipeline(steps=[(TokenizerWord(), None)])
    X = [["すもももももももものうち"]]
    y = pipe(X)
    print(y)
