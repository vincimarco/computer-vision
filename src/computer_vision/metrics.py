from sktime.performance_metrics.forecasting import (
    MeanAbsoluteError,
    MeanAbsolutePercentageError,
    MeanSquaredError,
)


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


metrics = {
    "MAE": MeanAbsoluteError(),
    "MAPE": MeanAbsolutePercentageError(),
    "RMSE": RootMeanSquaredError(),
    "MSE": MeanSquaredError(),
}
