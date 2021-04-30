import pandas
from sklearn.pipeline import Pipeline
from ..dataset.timeset import TimeSeriesDataset
from ..model.prophet import PreprocessProphet, EstimatorProphet
from ..util.plotter import save_plot, save_plot_components


class ApiController(object):
    def __init__(self):
        self.exogs = []
        self.holidays_df = None
        self.steps = [
            ("preprocess", PreprocessProphet()),
            ("model", EstimatorProphet(self.holidays_df, self.exogs, model=None)),
        ]
        self.pipe = Pipeline(steps=self.steps)

        return

    @staticmethod
    def setup_for_prophet(df: pandas.DataFrame) -> pandas.DataFrame:
        df = df.copy()
        df.index.name = "ds"
        cols = list(df.columns)
        cols[-1] = "y"
        df.columns = cols
        return df

    def predict_trained(self, df: pandas.DataFrame) -> pandas.DataFrame:
        _freq = "D"
        df = self.setup_for_prophet(df)
        timeset = TimeSeriesDataset(data_file=None, freq=_freq)
        timeset.from_df(df)
        train_rate = 0.7
        n_train = int(train_rate * len(df))
        predict_date = df.index[n_train]
        # train_df, test_df = timeset.split(predict_date.strftime("%Y-%m-%d"))
        train_df, test_df = timeset.split(predict_date)

        # for train
        self.pipe.fit(train_df, y=None)
        # predict_by = test_df.index[-1]
        params = dict(freq=_freq)
        # params = dict(predict_by=predict_by, freq=_freq)
        forecast_df = self.pipe.predict(test_df, **params)

        # for debugging
        m = self.pipe[-1].model
        save_plot(
            f"img/forecasted_simple_{m.growth}.png", forecast_df, train_df, test_df
        )
        save_plot_components(
            f"img/components_simple_{m.growth}.png",
            m,
            forecast_df.drop(columns=["weekly"]),
        )

        return forecast_df
