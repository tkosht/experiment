from app.component.models.pipeline import Pipeline
from app.domain.models.tokenizer import JpTokenizerJanome

if __name__ == "__main__":
    pipe = Pipeline(steps=[JpTokenizerJanome(), None])
    pass
