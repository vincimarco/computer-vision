import pathlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import collections.abc

    import keras
import tqdm.keras
from holidays import country_holidays
from sktime.forecasting.compose import ForecastingPipeline, make_reduction
from sktime.forecasting.darts import DartsXGBModel
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.neuralforecast import NeuralForecastLSTM, NeuralForecastTCN
from sktime.regression.deep_learning import CNNRegressor
from sktime.transformations.compose import (
    ColumnEnsembleTransformer,
    Id,
    TransformerPipeline,
)
from sktime.transformations.series.date import DateTimeFeatures
from sktime.transformations.series.holiday import HolidayFeatures
from sktime.transformations.series.impute import Imputer

from computer_vision.model.cnn3d import CNN3D
from computer_vision.transformer.cyclical_encoding import CyclicalEncodingTransformer


def create_forecaster(forecaster_name: str, params: dict) -> ForecastingPipeline:
    X_transformers, y_transformers = _create_transformers()

    if forecaster_name == "cnn3d":
        forecaster = create_cnn3d_forecaster(**params)

    elif forecaster_name == "naive":
        forecaster = create_naive_forecaster(**params)

    elif forecaster_name == "darts_xgb":
        forecaster = create_darts_xgb_forecaster(**params)

    elif forecaster_name == "lstm":
        forecaster = create_lstm_forecaster(**params)

    elif forecaster_name == "tcn":
        forecaster = create_tcn_forecaster(**params)

    elif forecaster_name == "cnn":
        forecaster = create_cnn_forecaster(**params)

    else:
        raise ValueError(f"Unknown forecaster: {forecaster_name}")

    pipeline = X_transformers ** (y_transformers * forecaster)

    return pipeline


def load_forecaster(path: pathlib.Path) -> ForecastingPipeline:
    forecaster = ForecastingPipeline.load_from_path(path)
    return forecaster


def create_cnn3d_forecaster(
    epochs: int,
    batch_size: int,
    random_state: int,
    loss: "str | keras.Metric",
    metrics: "collections.abc.Sequence[str | keras.Metric] | None",
    optimizer: "str | keras.optimizers.Optimizer",
    kernel_width: str,
    dropout_rate: float,
    sample_weights_function: "str | None",
    decay_rate: float,
    window_size: str,
) -> CNN3D:
    return CNN3D(
        epochs=epochs,
        batch_size=batch_size,
        random_state=random_state,
        loss=loss,
        metrics=metrics,
        optimizer=optimizer,
        kernel_width=kernel_width,
        dropout_rate=dropout_rate,
        sample_weights_function=sample_weights_function,
        decay_rate=decay_rate,
        window_size=window_size,
    )


def create_naive_forecaster(sp: int) -> NaiveForecaster:
    return NaiveForecaster(strategy="last", sp=sp)


def create_darts_xgb_forecaster(
    lags: int,
    lags_past_covariates: int,
    output_chunk_length: int,
    random_state: int,
    multi_models: bool,
) -> DartsXGBModel:
    return DartsXGBModel(
        lags=lags,
        output_chunk_length=output_chunk_length,
        random_state=random_state,
        multi_models=multi_models,
        lags_past_covariates=lags_past_covariates,
    )


def create_lstm_forecaster(
    random_seed: int,
    loss: "str | keras.Metric",
    broadcasting: bool,
    optimizer: "str | keras.optimizers.Optimizer",
    max_steps: int,
    local_scaler_type: str,
) -> NeuralForecastLSTM:

    import neuralforecast.losses.pytorch as nflosses
    import torch.optim

    losses = {
        "mse": nflosses.MSE(),
        "huber": nflosses.HuberLoss(),
    }

    optimizers = {
        "adam": torch.optim.Adam,
        "sgd": torch.optim.SGD,
    }

    return NeuralForecastLSTM(
        freq="15min",
        local_scaler_type=local_scaler_type,
        futr_exog_list=[
            "year",
            "month_of_year__sin",
            "month_of_year__cos",
            "day_of_week__sin",
            "day_of_week__cos",
            "hour_of_day__sin",
            "hour_of_day__cos",
            "is_holiday",
        ],
        verbose_fit=True,
        verbose_predict=True,
        loss=losses[loss] if isinstance(loss, str) else loss,
        # optimizer=optimizers[optimizer] if isinstance(optimizer, str) else optimizer,
        random_seed=random_seed,
        broadcasting=broadcasting,
        max_steps=max_steps,
    )


def create_tcn_forecaster(
    random_seed: int,
    loss: "str | keras.Metric",
    broadcasting: bool,
    optimizer: "str | keras.optimizers.Optimizer",
    max_steps: int,
    local_scaler_type: str,
) -> NeuralForecastTCN:
    import neuralforecast.losses.pytorch as nflosses
    import torch.optim

    losses = {
        "mse": nflosses.MSE(),
        "huber": nflosses.HuberLoss(),
    }

    optimizers = {
        "adam": torch.optim.Adam,
        "sgd": torch.optim.SGD,
    }

    return NeuralForecastTCN(
        freq="15min",
        local_scaler_type=local_scaler_type,
        futr_exog_list=[
            "year",
            "month_of_year__sin",
            "month_of_year__cos",
            "day_of_week__sin",
            "day_of_week__cos",
            "hour_of_day__sin",
            "hour_of_day__cos",
            "is_holiday",
        ],
        verbose_fit=True,
        verbose_predict=True,
        loss=losses[loss] if isinstance(loss, str) else loss,
        random_seed=random_seed,
        broadcasting=broadcasting,
        max_steps=max_steps,
    )


def create_cnn_forecaster(
    epochs: int,
    batch_size: int,
    kernel_size: int,
    avg_pool_size: int,
    n_conv_layers: int,
    random_state: int,
    loss: str,
    strategy: str,
    window_length: int,
    pooling: str,
):
    import keras

    return make_reduction(
        CNNRegressor(
            n_epochs=epochs,
            batch_size=batch_size,
            kernel_size=kernel_size,
            avg_pool_size=avg_pool_size,
            n_conv_layers=n_conv_layers,
            random_state=random_state,
            loss=loss,
            verbose=True,
            metrics=["mae"],
            callbacks=[
                tqdm.keras.TqdmCallback(verbose=1),
                keras.callbacks.EarlyStopping(
                    monitor="loss",
                    patience=5,
                    restore_best_weights=True,
                ),
            ],
        ),
        strategy=strategy,
        window_length=window_length,
        pooling=pooling,
        windows_identical=False,
    )


def _create_transformers() -> tuple[TransformerPipeline, TransformerPipeline]:
    _dtfeats_transformer = DateTimeFeatures()
    _holiday_transformer = HolidayFeatures(
        calendar=country_holidays("UY", years=[2019, 2020]),
        return_dummies=False,
        return_indicator=True,
        keep_original_columns=True,
    )
    _column_transformer = ColumnEnsembleTransformer(
        [
            ("id", Id(), ["year"]),
            (
                "cyclical",
                CyclicalEncodingTransformer(),
                ["month_of_year", "day_of_week", "hour_of_day"],
            ),
        ],
        feature_names_out="original",
    )

    X_transformers = _dtfeats_transformer * _column_transformer * _holiday_transformer

    y_transformers = Imputer()

    return X_transformers, y_transformers
