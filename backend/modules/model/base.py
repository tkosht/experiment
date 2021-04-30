class Transer(object):
    def __init__(self):
        pass

    def fit(self, X, y, **params):
        return self

    def transform(self, X, **params):
        return X


class Estimator(object):
    def fit(self, X, y, **params):
        return self

    def predict(self, X, **params):
        return X
