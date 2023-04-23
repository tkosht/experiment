import datetime
import pathlib
import pandas
from dateutil.relativedelta import relativedelta
from matplotlib import pyplot


def make_plot_block(verif, start_date, end_date, ax=None):
    df = verif.loc[start_date:end_date, :]
    df.loc[:, "yhat"].plot(lw=2, ax=ax, color="r", ls="-", label="forecasts")
    ax.fill_between(
        df.index,
        df.loc[:, "yhat_lower"],
        df.loc[:, "yhat_upper"],
        color="coral",
        alpha=0.3,
    )
    df.loc[:, "y"].plot(lw=2, ax=ax, color="steelblue", ls="-", label="observations")
    ax.grid(ls=":")
    ax.legend(fontsize=15)
    [lbl.set_fontsize(13) for lbl in ax.xaxis.get_ticklabels()]
    [lbl.set_fontsize(13) for lbl in ax.yaxis.get_ticklabels()]
    ax.set_ylabel("cyclists number", fontsize=15)
    ax.set_xlabel("", fontsize=15)
    ax.set_title(f"{start_date} to {end_date}", fontsize=18)


def make_verif(forecast, data_train, data_test):
    forecast = forecast.copy()
    data_train = data_train.copy()
    data_test = data_test.copy()

    forecast.index = pandas.to_datetime(forecast.ds)
    data_train.index = pandas.to_datetime(data_train.ds)
    data_test.index = pandas.to_datetime(data_test.ds)
    data = pandas.concat([data_train, data_test], axis=0)
    forecast.loc[:, "y"] = data.loc[:, "y"]
    forecast.reset_index(drop=True, inplace=True)
    # for aggregated data like weekly data
    s = forecast.ds.iloc[0]
    e = forecast.ds.iloc[-1]
    x = pandas.date_range(s, e, freq="D", name="ds").to_frame().reset_index(drop=True)
    f = x.merge(forecast, on="ds")
    f.dropna(inplace=True)
    f.index = f.ds
    forecast = f
    return forecast


def save_plot(img_file, forecast_df, train_df, test_df, plot_freq="6M"):
    verif = make_verif(forecast_df, train_df, test_df)
    predict_date = test_df.ds.iloc[0].to_pydatetime()

    s = verif.ds.iloc[0]
    e = verif.ds.iloc[-1]
    intervals = pandas.date_range(s, e, freq=plot_freq) + datetime.timedelta()
    if intervals[-1] < e:
        date_end = intervals[-1] + relativedelta(months=6)
        intervals = pandas.Series(intervals).append(pandas.Series(date_end))

    n = len(intervals) - 1
    _, axes = pyplot.subplots(nrows=n, figsize=(14, 16), sharey=True)
    if n == 1:
        axes = [axes]

    for idx, (s, e) in enumerate(zip(intervals[:-1], intervals[1:])):
        ax = axes[idx]
        s = s.strftime("%Y-%m-%d")
        e = e.strftime("%Y-%m-%d")
        make_plot_block(verif, s, e, ax=ax)
        if s <= predict_date.strftime("%Y-%m-%d") <= e:
            ax.axvline(x=predict_date)
    pyplot.tight_layout()

    pathlib.Path("img").mkdir(parents=True, exist_ok=True)
    pyplot.savefig(img_file)


def save_plot_components(img_file, m, forecast_df):
    m.plot_components(forecast_df)
    pyplot.savefig(img_file)
