from sktime.performance_metrics.forecasting import (
    MeanAbsoluteError,
    MeanAbsolutePercentageError,
    MeanAbsoluteScaledError,
    MeanSquaredError,
)
from sktime.performance_metrics.forecasting._base import BaseForecastingErrorMetric


class RootMeanSquaredError(MeanSquaredError):
    def __init__(
        self,
        multioutput="uniform_average",
        multilevel="uniform_average",
        by_index=False,
    ):
        super().__init__(
            multioutput=multioutput,
            multilevel=multilevel,
            by_index=by_index,
        )
        self.square_root = True


class MeanAbsoluteScaledError96(MeanAbsoluteScaledError):
    def __init__(
        self,
        multioutput="uniform_average",
        multilevel="uniform_average",
        by_index=False,
    ):
        super().__init__(
            multioutput=multioutput,
            multilevel=multilevel,
            by_index=by_index,
        )
        self.sp = 96


class MeanAbsoluteScaledError672(MeanAbsoluteScaledError):
    def __init__(
        self,
        multioutput="uniform_average",
        multilevel="uniform_average",
        by_index=False,
    ):
        super().__init__(
            multioutput=multioutput,
            multilevel=multilevel,
            by_index=by_index,
        )
        self.sp = 672


metrics: dict[str, BaseForecastingErrorMetric] = {
    "MAE": MeanAbsoluteError(),
    "MAPE": MeanAbsolutePercentageError(),
    "RMSE": RootMeanSquaredError(),
    "MSE": MeanSquaredError(),
    "MASE": MeanAbsoluteScaledError(),
    "MASE96": MeanAbsoluteScaledError96(),
    "MASE672": MeanAbsoluteScaledError672(),
}
