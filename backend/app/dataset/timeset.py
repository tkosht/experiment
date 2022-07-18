import pandas
from typing import Tuple


class TimeSeriesDataset(object):
    def __init__(self, data_file="data/data.tsv", freq="D"):
        self.data_df = None
        self.data_file = data_file
        if data_file is not None:
            self._load()._aggregate()

    def _load(self):
        data_df = pandas.read_csv(
            self.data_file, sep="\t", header=0, index_col="ds", parse_dates=True
        )
        self._check_df(data_df)
        self.data_df = data_df
        return self

    def _check_df(self, df: pandas.DataFrame):
        assert "ds" in df.columns or df.index.name == "ds"
        assert "y" in df.columns
        return self

    def from_df(self, df: pandas.DataFrame):
        self._check_df(df)
        if "ds" in df.columns:
            df = df.copy()
            df.index = df.ds
            df.drop("ds", axis=1, inplace=True)
        self.data_df = df
        return self

    def _aggregate(self):
        self.data_df = self.data_df.resample(
            self.freq, label="left", closed="left"
        ).sum()
        return self

    def split(self, predict_date: str) -> Tuple[pandas.DataFrame, pandas.DataFrame]:
        train_df = self.data_df[:predict_date].iloc[:-1]
        test_df = self.data_df[predict_date:]
        train_df.reset_index(inplace=True)
        test_df.reset_index(inplace=True)
        assert "ds" in train_df.columns
        assert "y" in train_df.columns
        assert "ds" in test_df.columns
        assert "y" in test_df.columns
        return train_df, test_df
