from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    import keras  # noqa: TC004
import contextlib

import numpy as np
import pandas as pd
from sktime.forecasting.base import BaseForecaster


class CNN3D(BaseForecaster):
    """Custom global forecasting model based on 3D CNN. todo: write docstring.

    todo: describe your custom forecaster here

    Parameters
    ----------
    epochs : int, default=200
        Number of training epochs.
    batch_size : int, default=32
        Batch size for training.
    random_state : int, default=42
        Random seed for reproducibility.
    loss : str | keras.Metric, default="mse"
        Loss function for training.
    metrics : Sequence[str | keras.Metric] | None, default=None
        Metrics to compute during training. Defaults to ["mae", "mape"].
    optimizer : str | keras.optimizers.Optimizer, default="adam"
        Optimizer for training.
    kernel_width : str, default="1h"
        Width of convolutional kernel as timedelta string.
    dropout_rate : float, default=0.2
        Dropout rate for regularization.
    sample_weights_function : str | None, default=None
        Method for computing sample weights ("exponential" or None).
    decay_rate : float, default=0.01
        Decay rate for exponential sample weights.
    window_size : str, default="168h"
        Size of history window as timedelta string.
    """

    _tags = {
        # tags and full specifications are available in the tag API reference
        # https://www.sktime.net/en/stable/api_reference/tags.html
        # to list all valid tags with description, use sktime.registry.all_tags
        #   all_tags(estimator_types="forecaster", as_dataframe=True)
        #
        # behavioural tags: internal type
        # -------------------------------
        #
        # y_inner_mtype, X_inner_mtype control which format X/y appears in
        # in the inner functions _fit, _predict, etc
        "y_inner_mtype": "pd-multiindex",
        "X_inner_mtype": "pd-multiindex",
        # valid values: str and list of str
        # if str, must be a valid mtype str, in sktime.datatypes.MTYPE_REGISTER
        #   of scitype Series, Panel (panel data) or Hierarchical (hierarchical series)
        #   in that case, all inputs are converted to that one type
        # if list of str, must be a list of valid str specifiers
        #   in that case, X/y are passed through without conversion if on the list
        #   if not on the list, converted to the first entry of the same scitype
        #
        # scitype:y controls whether internal y can be univariate/multivariate
        # if multivariate is not valid, applies vectorization over variables
        "scitype:y": "univariate",
        "capability:exogenous": True,
        "X-y-must-have-same-index": True,
        "enforce_index_type": pd.DatetimeIndex,
        "requires-fh-in-fit": True,
        # X-y-must-have-same-index = can estimator handle different X/y index?
        "X-y-must-have-same-index": True,
        # valid values: boolean True (yes), False (no)
        # if True, raises exception if X.index is not contained in y.index
        #
        # enforce_index_type = index type that needs to be enforced in X/y
        "enforce_index_type": pd.DatetimeIndex,
        # valid values: pd.Index subtype, or list of pd.Index subtype
        # if not None, raises exception if X.index, y.index level -1 is not of that type
        #
        # capability:global_forecasting = does forecaster support global forecasting?
        "capability:global_forecasting": False,
        # valid values: boolean True (yes), False (no)
        #
        # ownership and contribution tags
        # -------------------------------
        #
        # author = author(s) of th estimator
        # an author is anyone with significant contribution to the code at some point
        "authors": ["author1", "author2"],
        # valid values: str or list of str, should be GitHub handles
        # this should follow best scientific contribution practices
        # scope is the code, not the methodology (method is per paper citation)
        # if interfacing a 3rd party estimator, ensure to give credit to the
        # authors of the interfaced estimator
        #
        # maintainer = current maintainer(s) of the estimator
        # per algorithm maintainer role, see governance document
        # this is an "owner" type role, with rights and maintenance duties
        # for 3rd party interfaces, the scope is the sktime class only
        "maintainers": ["maintainer1", "maintainer2"],
        # valid values: str or list of str, should be GitHub handles
        # remove tag if maintained by sktime core team
    }

    # todo: add any hyper-parameters and components to constructor
    def __init__(
        self,
        epochs: int = 200,
        batch_size: int = 32,
        random_state: int = 42,
        loss: "str | keras.Metric" = "mse",
        metrics: "Sequence[str | keras.Metric] | None" = None,
        optimizer: "str | keras.optimizers.Optimizer" = "adam",
        kernel_width: str = "1h",
        dropout_rate: float = 0.2,
        sample_weights_function: "str | None" = None,
        decay_rate: float = 0.01,
        window_size: str = "168h",
    ):
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.loss = loss
        self.metrics = metrics
        self.optimizer = optimizer
        self.kernel_width = kernel_width
        self.dropout_rate = dropout_rate
        self.sample_weights_function = sample_weights_function
        self.decay_rate = decay_rate
        self.window_size = window_size

        # leave this as is
        super().__init__()

        # todo: optional, parameter checking logic (if applicable) should happen here
        # if writes derived values to self, should *not* overwrite self.parama etc
        # instead, write to self._parama, self._newparam (starting with _)
        self._metrics = metrics if metrics is not None else ["mae", "mape"]
        self._window_size = pd.Timedelta(window_size)

    # todo: implement this, mandatory
    def _fit(self, y, X, fh):
        """Fit forecaster to training data.

        private _fit containing the core logic, called from fit

        Writes to self:
            Sets fitted model attributes ending in "_".

        Parameters
        ----------
        y : guaranteed to be of a type in self.get_tag("y_inner_mtype")
            Time series to which to fit the forecaster.
            if self.get_tag("scitype:y")=="univariate":
                guaranteed to have a single column/variable
            if self.get_tag("scitype:y")=="multivariate":
                guaranteed to have 2 or more columns
            if self.get_tag("scitype:y")=="both": no restrictions apply
        fh : guaranteed to be ForecastingHorizon or None, optional (default=None)
            The forecasting horizon with the steps ahead to to predict.
            Required (non-optional) here if self.get_tag("requires-fh-in-fit")==True
            Otherwise, if not passed in _fit, guaranteed to be passed in _predict
        X : optional (default=None)
            guaranteed to be of a type in self.get_tag("X_inner_mtype")
            Exogeneous time series to fit to.

        Returns
        -------
        self : reference to self
        """

        # IMPORTANT: avoid side effects to y, X, fh

        from copy import deepcopy  # noqa: PLC0415

        self._data_freq = self._estimate_data_freq(y)

        (
            past_endos,
            past_exos,
            future_endos,
            future_exos,
        ) = self._time_series_to_tabular()

        endo_input_size = past_endos.shape[1]
        exo_input_size = past_exos.shape[2]
        self._n_exo_features = exo_input_size
        output_size = len(fh.to_numpy())

        self.model_ = self._build_model(
            input_size=endo_input_size,
            exo_input_size=exo_input_size,
            output_size=output_size,
            data_freq=self._data_freq,
        )

        self.model_.compile(
            optimizer=deepcopy(self.optimizer),
            loss=self.loss,
            metrics=self._metrics,
        )

        self.model_.summary()

        sample_weights = None
        if self.sample_weights_function == "exponential":
            sample_weights = self._calculate_sample_weights(len(past_endos))

        self._fit_model(
            past_endos, past_exos, future_exos, future_endos, sample_weights
        )

        return self

    def _predict(self, fh, X):
        """Forecast time series at future horizon."""
        group_names = self._y.index.get_level_values(0).unique()
        future_index = self._fh.to_absolute(self.cutoff).to_pandas()
        all_predictions = [
            self._predict_single_group(group, X) for group in group_names
        ]
        return self._build_prediction_dataframe(
            all_predictions, group_names, future_index
        )

    def _predict_single_group(self, group, X):
        """Predict for a single group."""
        past_endo, past_exo, future_exo = self._extract_group_data(group, X)
        y_pred_group = self.model_.predict([past_endo, past_exo, future_exo])
        return y_pred_group.flatten()

    def _extract_group_data(self, group, X):
        """Extract data for a single group."""
        import numpy as np  # noqa: PLC0415

        window_size = int(self._window_size // self._data_freq)
        future_horizon = len(self._fh)

        y_g = self._y.xs(group)
        past_endo = (
            y_g.iloc[-window_size:].values.astype(np.float32).reshape(1, window_size, 1)
        )

        past_exo = None
        if self._X is not None:
            X_g = self._X.xs(group)
            past_exo = (
                X_g.iloc[-window_size:]
                .values.astype(np.float32)
                .reshape(1, window_size, self._n_exo_features)
            )

        future_exo = None
        if X is not None:
            X_future_g = X.xs(group)
            future_exo = (
                X_future_g.iloc[:future_horizon]
                .values.astype(np.float32)
                .reshape(1, future_horizon, self._n_exo_features)
            )

        return past_endo, past_exo, future_exo

    def _build_prediction_dataframe(self, predictions, group_names, future_index):
        """Build multiindex DataFrame from predictions.

        Parameters
        ----------
        predictions : list of np.ndarray
            Flattened predictions for each group
        group_names : Index
            Group names from multiindex level 0
        future_index : Index
            Future time index

        Returns
        -------
        pd.DataFrame
            Predictions with multiindex (group, time)
        """

        window_size = self._window_size // self._data_freq

        past_endo = np.array(list(self._y.iloc[-window_size:]))
        print(f"Past endogenous values: {past_endo}")
        past_endo = past_endo.reshape(
            1, window_size, 1
        )  # Select the last window_size points, reshape, and wrap in numpy array

        past_index = self._y.iloc[-window_size:].index
        with contextlib.suppress(AttributeError):
            past_index = past_index.to_timestamp()

        past_exo = self._extract_temporal_features(past_index)
        past_exo = np.array([past_exo]).reshape(1, window_size, self._n_exo_features)

        future_index = self._fh.to_absolute(self.cutoff).to_pandas()
        future_index_ts = future_index
        with contextlib.suppress(AttributeError):
            future_index_ts = future_index.to_timestamp()
        future_exo = self._extract_temporal_features(future_index_ts)
        future_exo = np.array([future_exo]).reshape(
            1, len(self._fh.to_numpy()), self._n_exo_features
        )

        # Predict using the trained model
        y_pred = self.model_.predict([past_endo, past_exo, future_exo])

        # Correct y_pred index based on self._fh
        y_pred = pd.Series(y_pred.flatten(), index=future_index)

        return y_pred

        # implement here

    def _update(self, y, X=None, update_params=True):
        """Update model with new data."""
        if not update_params:
            return self

        (past_endos, past_exos, future_endos, future_exos) = (
            self._time_series_to_tabular()
        )

        sample_weights = None
        if self.sample_weights_function == "exponential":
            sample_weights = self._calculate_sample_weights(len(past_endos))

        self._fit_model(
            past_endos, past_exos, future_exos, future_endos, sample_weights
        )

        return self

    # todo: implement this if this is an estimator contributed to sktime
    #   or to run local automated unit and integration testing of estimator
    #   method should return default parameters, so that a test instance can be created
    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.
            There are currently no reserved values for forecasters.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """

        # todo: set the testing parameters for the estimators
        # Testing parameters can be dictionary or list of dictionaries.
        # Testing parameter choice should cover internal cases well.
        #   for "simple" extension, ignore the parameter_set argument.
        #
        # this method can, if required, use:
        #   class properties (e.g., inherited); parent class test case
        #   imported objects such as estimators from sktime or sklearn
        # important: all such imports should be *inside get_test_params*, not at the top
        #            since imports are used only at testing time
        #
        # A good parameter set should primarily satisfy two criteria,
        #   1. Chosen set of parameters should have a low testing time,
        #      ideally in the magnitude of few seconds for the entire test suite.
        #       This is vital for the cases where default values result in
        #       "big" models which not only increases test time but also
        #       run into the risk of test workers crashing.
        #   2. There should be a minimum two such parameter sets with different
        #      sets of values to ensure a wide range of code coverage is provided.
        #
        # example 1: specify params as dictionary
        # any number of params can be specified
        # params = {"est": value0, "parama": value1, "paramb": value2}
        #
        # example 2: specify params as list of dictionary
        # note: Only first dictionary will be used by create_test_instance
        params1 = {
            "epochs": 2,
            "batch_size": 32,
            "random_state": 42,
            "loss": "mse",
            "optimizer": "adam",
            "kernel_width": 3,
            "dropout_rate": 0.2,
            "window_size": "168h",
        }
        params2 = {
            "epochs": 5,
            "batch_size": 16,
            "random_state": 123,
            "loss": "mse",
            "optimizer": "adam",
            "kernel_width": 5,
            "dropout_rate": 0.3,
            "window_size": "24h",
        }
        return [params1, params2]

    @property
    def history(self):
        """Return training history of the model."""
        return self.history_.history

    # ---------------------------------------------------------------------------- #
    #                              FEATURE EXTRACTION                              #
    # ---------------------------------------------------------------------------- #

    def _extract_temporal_features(self, index: "pd.Index") -> "pd.DataFrame":
        """Extract temporal features from a pandas DatetimeIndex.

        Parameters
        ----------
        index : pd.Index
            DatetimeIndex from which to extract temporal features.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the extracted temporal features.
        """

        # Get the first level (id) unique value
        first_id = self._y.index.get_level_values(0).unique()[0]
        # Get the index for this id (timestamp level)
        idx = self._y.loc[first_id].index

        features = pd.DataFrame(
            {
                "hour": idx.hour,  # ty:ignore[unresolved-attribute]
                "dayofweek": idx.dayofweek,  # ty:ignore[unresolved-attribute]
                "month": idx.month,  # ty:ignore[unresolved-attribute]
            },
            index=idx,
        )

        # Min-max scaling
        features["hour"] = features["hour"] / 23
        features["dayofweek"] = features["dayofweek"] / 6
        features["month"] = (features["month"] - 1) / 11

        # Cyclical encoding
        features["hour_sin"] = np.sin(2 * np.pi * features["hour"])
        features["hour_cos"] = np.cos(2 * np.pi * features["hour"])
        features["dayofweek_sin"] = np.sin(2 * np.pi * features["dayofweek"])
        features["dayofweek_cos"] = np.cos(2 * np.pi * features["dayofweek"])
        features["month_sin"] = np.sin(2 * np.pi * features["month"])
        features["month_cos"] = np.cos(2 * np.pi * features["month"])

        # Keep only cyclical encoded features
        features = features[
            [
                "hour_sin",
                "hour_cos",
                "dayofweek_sin",
                "dayofweek_cos",
                "month_sin",
                "month_cos",
            ]
        ]

        self._n_exo_features = features.shape[1]

        return features

    # ---------------------------------------------------------------------------- #
    #                                     MODEL                                    #
    # ---------------------------------------------------------------------------- #

    def _build_model(self, input_size: int, output_size: int) -> "keras.Model":
        """Builds the Keras model."""

        import keras  # noqa: PLC0415

        endo_past_input = keras.layers.Input(shape=(input_size, 1))
        exo_past_input = keras.layers.Input(shape=(input_size, exo_input_size))
        exo_future_input = keras.layers.Input(shape=(output_size, exo_input_size))

        # Use input_size for all reshape targets to ensure consistent concatenation
        reshape_shape = self._get_reshape_target(input_size)

        past_exo_mask = self._build_exo_mask_layer(
            exo_past_input, input_size, reshape_shape, "past"
        )
        endo_mask = self._build_endo_mask_layer(
            endo_past_input, input_size, reshape_shape
        )
        # Use input_size for all masks to ensure concatenation compatibility
        # Dense layer will adapt to different input sizes (output_size vs input_size)
        future_exo_mask = self._build_exo_mask_layer(
            exo_future_input, input_size, reshape_shape, "future"
        )

        combined_mask = keras.layers.concatenate(
            [past_exo_mask, endo_mask, future_exo_mask], axis=3
        )
        cnn_3d = self._build_cnn3d_block(combined_mask, input_size, data_freq)

        # lstm_output = self._build_lstm_block(endo_past_input, exo_past_input)
        # combined = keras.layers.concatenate([cnn_3d, lstm_output])
        output = self._build_output_block(cnn_3d, output_size)

        return keras.Model(
            inputs=[endo_past_input, exo_past_input, exo_future_input],
            outputs=output,
        )

    def _get_reshape_target(self, dim_size: int) -> tuple:
        """Compute target reshape dimensions, handling edge cases."""
        # Use (7, dim_size//7, 1, 1) if divisible by 7, else (1, dim_size, 1, 1)
        if dim_size >= 7 and dim_size % 7 == 0:
            return (7, dim_size // 7, 1, 1)
        else:
            return (1, dim_size, 1, 1)

    def _build_exo_mask_layer(
        self,
        exo_input: "keras.Layer",
        input_size: int,
        reshape_target: tuple,
        name: str,
    ) -> "keras.Layer":
        """Build exogenous mask layer."""
        import keras  # noqa: PLC0415

        flattened = keras.layers.Flatten(name=f"{name}_exo_flatten")(exo_input)
        dropout1 = keras.layers.Dropout(self.dropout_rate, name=f"{name}_exo_dropout1")(
            flattened
        )
        dense1 = keras.layers.Dense(500, activation="relu", name=f"{name}_exo_dense1")(
            dropout1
        )
        dense2 = keras.layers.Dense(250, activation="relu", name=f"{name}_exo_dense2")(
            dense1
        )
        dense3 = keras.layers.Dense(input_size, name=f"{name}_exo_dense3")(dense2)
        dropout2 = keras.layers.Dropout(self.dropout_rate, name=f"{name}_exo_dropout2")(
            dense3
        )
        return keras.layers.Reshape(reshape_target, name=f"{name}_exo_reshape")(
            dropout2
        )

    def _build_endo_mask_layer(
        self,
        endo_input: "keras.Layer",
        input_size: int,
        reshape_target: tuple,
    ) -> "keras.Layer":
        """Build endogenous mask layer."""
        import keras  # noqa: PLC0415

        reshaped = keras.layers.Reshape(reshape_target)(endo_input)
        return keras.layers.BatchNormalization()(reshaped)

    def _build_cnn3d_block(
        self,
        combined_input: "keras.Layer",
        input_size: int,
        data_freq: "pd.Timedelta",
    ) -> "keras.Layer":
        """Build 3D CNN block."""
        import keras  # noqa: PLC0415

        kernel_w = int((self._kernel_width // data_freq) * 2 + 1)
        conv = keras.layers.Conv3D(
            filters=32,
            kernel_size=(combined_input.shape[1], kernel_w, combined_input.shape[3]),
            activation="relu",
            padding="valid",
        )(combined_input)
        flat = keras.layers.Flatten()(conv)
        dense1 = keras.layers.Dense(256, activation="relu")(flat)
        dense2 = keras.layers.Dense(128, activation="relu")(dense1)
        return keras.layers.Dense(64, activation="relu")(dense2)

    def _build_lstm_block(
        self,
        endo_input: "keras.Layer",
        exo_input: "keras.Layer",
    ) -> "keras.Layer":
        """Build LSTM block."""
        import keras  # noqa: PLC0415

        combined = keras.layers.Concatenate(axis=-1)([endo_input, exo_input])
        # Slice to use only the first 80% of the window (or at least 1 timestep)
        slice_end = max(1, int(combined.shape[1] * 0.8))
        sliced = combined[:, :slice_end, :]
        lstm = keras.layers.LSTM(64, return_sequences=False)(sliced)
        dense1 = keras.layers.Dense(256, activation="relu")(lstm)
        dense2 = keras.layers.Dense(128, activation="relu")(dense1)
        return keras.layers.Dense(64, activation="relu")(dense2)

    def _build_output_block(
        self,
        combined_input: "keras.Layer",
        output_size: int,
    ) -> "keras.Layer":
        """Build output block."""
        import keras  # noqa: PLC0415

        dense1 = keras.layers.Dense(256, activation="relu")(combined_input)
        dense2 = keras.layers.Dense(128, activation="relu")(dense1)
        dense3 = keras.layers.Dense(64, activation="relu")(dense2)
        return keras.layers.Dense(output_size)(dense3)

    def _fit_model(
        self, past_endos, past_exos, future_exos, future_endos, sample_weights
    ):
        """Train the model with callbacks and validation."""
        import pathlib  # noqa: PLC0415
        from tempfile import TemporaryDirectory  # noqa: PLC0415

        import keras  # noqa: PLC0415
        import tqdm.keras  # noqa: PLC0415

        self._validate_training_data(past_endos, past_exos, future_exos, future_endos)
        self._setup_gradient_clipping()

        with TemporaryDirectory() as tmpdir:
            weights_file = pathlib.Path(tmpdir) / "model.weights.h5"
            self.history_ = self.model_.fit(
                [past_endos, past_exos, future_exos],
                future_endos,
                epochs=self.epochs,
                batch_size=self.batch_size,
                verbose=0,
                sample_weight=sample_weights,
                callbacks=[
                    tqdm.keras.TqdmCallback(verbose=1),
                    keras.callbacks.ModelCheckpoint(
                        filepath=weights_file,
                        monitor="loss",
                        save_best_only=True,
                        save_weights_only=True,
                    ),
                    keras.callbacks.EarlyStopping(
                        monitor="loss",
                        patience=5,
                        restore_best_weights=True,
                    ),
                    keras.callbacks.TerminateOnNaN(),
                ],
            )
            self.model_.load_weights(weights_file)

    def _validate_training_data(self, past_endos, past_exos, future_exos, future_endos):
        """Validate training data for NaN and Inf values."""

        for data, name in [
            (past_endos, "past_endos"),
            (past_exos, "past_exos"),
            (future_exos, "future_exos"),
            (future_endos, "future_endos"),
        ]:
            if np.isnan(data).any():
                raise ValueError(f"NaN values found in {name}")
            if np.isinf(data).any():
                raise ValueError(f"Inf values found in {name}")

    def _setup_gradient_clipping(self):
        """Setup gradient clipping for the optimizer."""
        import keras  # noqa: PLC0415

        if isinstance(self.optimizer, str):
            optimizer = keras.optimizers.get(self.optimizer)
        else:
            optimizer = self.optimizer

        if hasattr(optimizer, "clipvalue"):
            optimizer.clipvalue = 1.0
        if hasattr(optimizer, "clipnorm"):
            optimizer.clipnorm = 1.0

        self.model_.compile(optimizer=optimizer, loss=self.loss, metrics=self._metrics)

    def _time_series_to_tabular(
        self,
    ) -> tuple["np.ndarray", "np.ndarray", "np.ndarray", "np.ndarray"]:

        import tqdm  # noqa: PLC0415
        from sktime.split.slidingwindow import SlidingWindowSplitter  # noqa: PLC0415

        window_length = int(self._window_size // self._data_freq)
        print(window_length)

        step_length = int(pd.Timedelta("1h") // self._data_freq)
        print(step_length)

        cv = SlidingWindowSplitter(
            window_length=window_length, fh=self._fh, step_length=step_length
        )

        cv_len = cv.get_n_splits(self._y)
        y_groups = list(self._y.groupby(level=0))
        group_labels = [label for label, _ in y_groups]

        past_endos, past_exos = [], []
        future_endos, future_exos = [], []

        for train_idx, test_idx in tqdm.tqdm(
            cv.split(self._y),
            total=cv_len,
            desc="Transforming time series to tabular format",
        ):
            past_endo = self._y.iloc[train_idx]
            future_endo = self._y.iloc[test_idx]
            past_exo = self._X.iloc[train_idx] if self._X is not None else None
            future_exo = self._X.iloc[test_idx] if self._X is not None else None

            self._extract_group_samples(
                group_labels,
                past_endo,
                future_endo,
                past_exo,
                future_exo,
                past_endos,
                past_exos,
                future_endos,
                future_exos,
            )

        return self._stack_and_validate_arrays(
            past_endos, past_exos, future_endos, future_exos, window_length
        )

    def _extract_group_samples(
        self,
        group_labels,
        past_endo,
        future_endo,
        past_exo,
        future_exo,
        past_endos,
        past_exos,
        future_endos,
        future_exos,
    ):
        """Extract samples for each group from multiindex data."""
        import numpy as np  # noqa: PLC0415

        for group_label in group_labels:
            past_endo_group = past_endo.loc[group_label]
            future_endo_group = future_endo.loc[group_label]

            past_endos.append(past_endo_group.values.astype(np.float32))
            future_endos.append(future_endo_group.values.astype(np.float32).flatten())

            if past_exo is not None:
                past_exos.append(past_exo.loc[group_label].values.astype(np.float32))
            else:
                past_exos.append(None)

            if future_exo is not None:
                future_exos.append(
                    future_exo.loc[group_label].values.astype(np.float32)
                )
            else:
                future_exos.append(None)

    def _stack_and_validate_arrays(
        self,
        past_endos,
        past_exos,
        future_endos,
        future_exos,
        window_length,
    ):
        """Stack arrays and validate data integrity."""
        import numpy as np  # noqa: PLC0415

        past_endos = np.array(past_endos, dtype=np.float32).reshape(
            -1, window_length, 1
        )
        future_endos = np.array(future_endos, dtype=np.float32)

        past_exos = self._pad_exogenous_arrays(past_exos)
        future_exos = self._pad_exogenous_arrays(future_exos)

        self._validate_array_integrity(past_endos, past_exos, future_endos, future_exos)

        return past_endos, past_exos, future_endos, future_exos

    def _pad_exogenous_arrays(self, exo_arrays):
        """Pad exogenous arrays with zeros for consistency."""
        import numpy as np  # noqa: PLC0415

        if not any(x is not None for x in exo_arrays):
            return None

        padded = []
        template = next(x for x in exo_arrays if x is not None)

        for x in exo_arrays:
            if x is not None:
                padded.append(x)
            else:
                padded.append(np.zeros_like(template))

        return np.array(padded, dtype=np.float32)

    def _validate_array_integrity(
        self, past_endos, past_exos, future_endos, future_exos
    ):
        """Validate array integrity for NaN values."""
        import numpy as np  # noqa: PLC0415

        for data, name in [
            (past_endos, "past_endos"),
            (future_endos, "future_endos"),
            (past_exos, "past_exos"),
            (future_exos, "future_exos"),
        ]:
            if data is not None and np.isnan(data).any():
                raise ValueError(f"NaN values detected in {name}")

    def _calculate_sample_weights(self, n_samples: int) -> "np.ndarray":
        """Calculate exponential decay sample weights."""
        import numpy as np  # noqa: PLC0415

        weights = np.exp(self.decay_rate * np.arange(n_samples))
        return weights / weights.max()

    def _estimate_data_freq(self, y: pd.DataFrame) -> "pd.Timedelta":
        """Estimate the frequency of the time series data.

        This method infers the frequency of the time series data based on the index
        of the target variable `y`. It uses pandas' `infer_freq` function to determine
        the frequency string and then converts it to a `pd.Timedelta` object for
        arithmetic operations.

        For MultiIndex data (e.g., (id, timestamp)), frequency is extracted from
        the timestamp level of the first entity.

        Parameters
        ----------
        y : pd.DataFrame
            The target time series data for which to estimate the frequency.


        Returns
        -------
        pd.Timedelta
            The inferred frequency of the time series data as a timedelta object.
        """
        # Get the first level (id) unique value
        first_id = self._y.index.get_level_values(0).unique()[0]
        # Get the index for this id (timestamp level)
        idx = self._y.loc[first_id].index

        freq_str = pd.infer_freq(idx)
        data_freq = pd.tseries.frequencies.to_offset(freq_str)
        # Convert offset to timedelta for arithmetic operations
        data_freq_timedelta = pd.Timedelta(data_freq)

        return data_freq_timedelta
