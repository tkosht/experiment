from inspect import signature

import joblib
from typing_extensions import Self


class TrainerBase(object):
    def do_train(self):
        raise NotImplementedError("do_train()")

    def do_eval(self):
        raise NotImplementedError("do_eval()")

    def __getstate__(self):
        s = signature(self.__init__)
        state = {}
        for k in list(s.parameters):
            state[k] = getattr(self, k)
        return state

    def load(self, load_file: str) -> Self:
        state = joblib.load(load_file)
        self.__init__(**state)
        return self

    def save(self, save_file: str) -> Self:
        s = signature(self.__init__)
        state = {}
        for k in list(s.parameters):
            state[k] = getattr(self, k)
        joblib.dump(state, save_file, compress=("gzip", 3))
        return self
