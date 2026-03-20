import pandas as pd
from sktime.forecasting.base import ForecastingHorizon


def create_fh(duration: str) -> ForecastingHorizon:
    duration_ts = pd.Timedelta(duration)
    steps = list(range(1, int(duration_ts / pd.Timedelta("15min")) + 1))
    return ForecastingHorizon(values=steps)
