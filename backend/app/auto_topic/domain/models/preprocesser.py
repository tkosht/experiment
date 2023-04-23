from app.auto_topic.component.models.model import Preprocesser, TextSequences


class Splitter(Preprocesser):
    def __init__(self, sep=" "):
        self.sep = sep

    def transform(self, X: list, **kwargs) -> TextSequences:
        return [x.split(self.sep) for x in X]


# class TagDocMaker(Preprocesser):
#     def transform(self, X, **kwargs):
#         return [
#             TaggedDocument(words=sentences, tags=[n]) for n, sentences in enumerate(X)
#         ]
#
# class Doc2Vectorizer(Preprocesser):
#     def __init__(self, n_components, window=7, min_count=1, workers=3):
#         super().__init__()
#         self.model = None
#         self.vector_size = n_components
#         self.window = window
#         self.min_count = min_count
#         self.workers = workers
#
#     def transform(self, X, **kwargs):
#         embedded = []
#         for tagdoc in X:
#             v = self.model.infer_vector(tagdoc.words)
#             embedded.append(v)
#         return embedded
#
#     def fit(self, X, y, **kwargs):
#         assert isinstance(X, list)
#         assert isinstance(X[0], TaggedDocument)
#         params = dict(
#             vector_size=self.vector_size,
#             window=self.window,
#             min_count=self.min_count,
#             workers=self.workers,
#             dm=0,
#             dbow_words=0,
#             negative=0,
#             hs=1,
#         )
#         self.model = Doc2Vec(X, **params)
#         return self
