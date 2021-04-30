import pandas
import datetime
from fbprophet import Prophet
from .base import Transer, Estimator
from ..util.params import add_args


class PreprocessProphet(Transer):
    def __init__(self):
        pass

    def fit(self, train_df: pandas.DataFrame, y: pandas.DataFrame, **params):
        return self

    def transform(self, train_predict_df, **params):
        return train_predict_df


class EstimatorProphet(Estimator):
    def __init__(
        self, holidays_df: pandas.DataFrame = None, exogs=[], model: Prophet = None
    ):
        self.model = model
        if model is None:
            self.model = self._create_prophet_model(holidays_df)
        self.exogs = exogs
        for _exg in exogs:
            self.model.add_regressor(_exg, prior_scale=0.5, mode="multiplicative")
        self.trained_df = None
        self.trained_date = None

    @staticmethod
    @add_args(params_file="conf/prophet.yml", root_key="/model/init", as_default=False)
    def _create_prophet_model(holidays_df, **params) -> Prophet:
        return Prophet(holidays=holidays_df, **params)

    @add_args(params_file="conf/prophet.yml", root_key="/model/fit", as_default=False)
    def fit(self, X: pandas.DataFrame, y: pandas.DataFrame, **params):
        assert "ds" in X.columns  # date series
        assert "y" in X.columns  # actual values
        assert set(self.exogs).issubset(set(X.columns))
        train_df = X.copy()
        if "cap" not in train_df.columns:
            mn = train_df.y.mean()
            sd = train_df.y.std()
            cap = mn + 3 * sd
            train_df["cap"] = cap
        self.model.fit(train_df, **params)
        self.trained_df = train_df
        self.trained_date = train_df.ds.iloc[-1]
        if isinstance(self.trained_date, str):
            self.trained_date = datetime.datetime.strptime(
                self.trained_date, "%Y-%m-%d"
            )
        return self

    def predict(self, predict_df, **params) -> pandas.DataFrame:
        """
        predict after trained_date by 1 year later
        """
        predict_by = self.trained_date + datetime.timedelta(days=365)
        if "predict_by" in params:
            predict_by = params["predict_by"]
        freq = "D"
        if "freq" in params:
            freq = params["freq"]

        futures = self.make_futures(predict_df, predict_by, freq)
        recent = 10
        if "cap" in predict_df.columns:
            futures["cap"] = predict_df.cap
            cap = predict_df.cap.tail(recent).mean()
            futures.cap.fillna(cap, inplace=True)
        else:
            futures["cap"] = self.trained_df.cap.tail(recent).mean()
        forecast_df = self.model.predict(futures)
        return forecast_df

    def make_futures(
        self, predict_df: pandas.DataFrame, predict_by: str = "2020-12-31", freq="D"
    ):
        if isinstance(predict_by, str):
            predict_by = datetime.datetime.strptime(predict_by, "%Y-%m-%d")
        predict_df = predict_df.copy()
        predict_date = self.trained_date + datetime.timedelta(1)
        future_date = (
            pandas.date_range(predict_date, predict_by, freq=freq, name="ds")
            .to_frame()
            .reset_index(drop=True)
        )

        df = pandas.concat([self.trained_df, predict_df], axis=0)
        df.ds = pandas.to_datetime(df.ds)
        data_df = future_date.merge(df, how="left", on="ds")
        data_df.reset_index(drop=True, inplace=True)
        data_df = data_df.interpolate(method="polynomial", order=5)
        data_df = data_df.fillna(0)  # fill far futures

        cols = ["ds"]
        if len(self.exogs) > 0:
            cols.extend(self.exogs)  # add exog columns for regressor
        futures = data_df[cols]
        return futures
