# todo: uncomment the following line, enter authors' GitHub IDs
# __author__ = [authorGitHubID, anotherAuthorGitHubID]
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    import keras  # noqa: TC004
    import numpy as np
    import pandas as pd  # noqa: TC004
    from sktime.forecasting.base import ForecastingHorizon

import contextlib

from sktime.forecasting.base import BaseForecaster


class CNN3D(BaseForecaster):
    """Custom forecaster. todo: write docstring.

    todo: describe your custom forecaster here

    Parameters
    ----------
    parama : int
        descriptive explanation of parama
    paramb : string, optional (default='default')
        descriptive explanation of paramb
    paramc : boolean, optional (default=MyOtherEstimator(foo=42))
        descriptive explanation of paramc
    and so on
    """

    # todo: fill out estimator tags here
    #  tags are inherited from parent class if they are not set
    # todo: define the forecaster scitype by setting the tags
    #  the "forecaster scitype" is determined by the tags
    #   scitype:y - the expected input scitype of y - univariate or multivariate or both
    # tag values are "safe defaults" which can usually be left as-is
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
        "y_inner_mtype": "pd.Series",
        "X_inner_mtype": "pd.DataFrame",
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
        # valid values: "univariate", "multivariate", "both"
        #   "univariate": inner _fit, _predict, etc, receive only univariate series
        #   "multivariate": inner methods receive only series with 2 or more variables
        #   "both": inner methods can see series with any number of variables
        #
        # capability tags: properties of the estimator
        # --------------------------------------------
        #
        # capability:exogenous = does estimator use exogeneous X nontrivially?
        "capability:exogenous": True,
        # valid values: boolean False (ignores X), True (uses X in non-trivial manner)
        # CAVEAT: if tag is set to False, inner methods always see X=None
        #
        # requires-fh-in-fit = is forecasting horizon always required in fit?
        "requires-fh-in-fit": True,
        # valid values: boolean True (yes), False (no)
        # if True, raises exception in fit if fh has not been passed
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
        # Keras stuff
        epochs: int = 200,
        batch_size: int = 32,
        random_state: int = 42,
        loss: "str | keras.Metric" = "mse",
        metrics: "Sequence[str | keras.Metric] | None" = None,
        optimizer: "str | keras.optimizers.Optimizer" = "adam",
        # Inner model
        kernel_width: int = 3,
        dropout_rate: float = 0.2,
        sample_weights_function: "str | None" = None,
        decay_rate: float = 0.01,
        window_size: str = "168h",
    ):
        # IMPORTANT: the self.params should never be overwritten or mutated from now on
        # for handling defaults etc, write to other attributes, e.g., self._parama
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
        import pandas as pd  # noqa: PLC0415

        self._metrics = metrics if metrics is not None else ["mae", "mape"]
        self._window_size = pd.Timedelta(window_size)

    # todo: implement this, mandatory
    def _fit(
        self,
        y: "pd.Series",
        X: "pd.DataFrame",
        fh: "ForecastingHorizon",
    ):
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

        (
            past_endos,
            past_exos,
            future_endos,
            future_exos,
        ) = self._time_series_to_tabular()

        input_size = past_endos.shape[1]
        output_size = len(fh.to_numpy())
        self.model_ = self._build_model(input_size=input_size, output_size=output_size)

        self.model_.compile(
            optimizer=deepcopy(self.optimizer), loss=self.loss, metrics=self._metrics
        )

        self.model_.summary()

        sample_weights = None
        if self.sample_weights_function == "exponential":
            sample_weights = self._calculate_sample_weights(len(past_endos))

        self._fit_model(
            past_endos, past_exos, future_exos, future_endos, sample_weights
        )

        return self

    # todo: implement this, mandatory
    def _predict(self, fh, X):
        """Forecast time series at future horizon.

        private _predict containing the core logic, called from predict

        State required:
            Requires state to be "fitted".

        Accesses in self:
            Fitted model attributes ending in "_"
            self.cutoff

        Parameters
        ----------
        fh : guaranteed to be ForecastingHorizon or None, optional (default=None)
            The forecasting horizon with the steps ahead to predict.
            If not passed in _fit, guaranteed to be passed here
        X : pd.DataFrame, optional (default=None)
            Exogenous time series

        Returns
        -------
        y_pred : sktime time series object
            should be of the same type as seen in _fit, as in "y_inner_mtype" tag
            Point predictions
        """

        import numpy as np  # noqa: PLC0415
        import pandas as pd  # noqa: PLC0415

        self._y: pd.Series
        self._fh: ForecastingHorizon

        series_freq = pd.infer_freq(self._y.index)
        window_size = pd.Timedelta(series_freq) // self._window_size

        past_endo = np.array(list(self._y.iloc[-window_size:]))
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
        if not update_params:
            return self

        (
            past_endos,
            past_exos,
            future_endos,
            future_exos,
        ) = self._time_series_to_tabular()

        self.model_.summary()

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
        # params = [{"est": value1, "parama": value2},
        #           {"est": value3, "parama": value4}]
        #
        # return params

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
        import numpy as np  # noqa: PLC0415
        import pandas as pd  # noqa: PLC0415

        datetime_index = pd.DatetimeIndex(index)

        features = pd.DataFrame(
            {
                "hour": datetime_index.hour,  # ty:ignore[unresolved-attribute]
                "dayofweek": datetime_index.dayofweek,  # ty:ignore[unresolved-attribute]
                "month": datetime_index.month,  # ty:ignore[unresolved-attribute]
            },
            index=index,
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

        endo_past_input: keras.KerasTensor = keras.layers.Input(shape=(input_size, 1))
        exo_past_input: keras.Layer = keras.layers.Input(
            shape=(input_size, self._n_exo_features)
        )
        exo_future_input: keras.Layer = keras.layers.Input(
            shape=(output_size, self._n_exo_features)
        )

        def exo_mask_layer(exo_input: keras.Layer, name: str = "exo") -> keras.Layer:
            """Applies a series of dense and dropout layers to exogenous input,
            followed by reshaping.

            Args:
            exo_input (keras.Layer): Input layer representing exogenous features.
                Shape (168, n)
            name (str): Name for the layer ("past" or "future").

            Returns:
            keras.Layer: Output layer with shape (7, 24, 1, 1)

            Architecture:
            - Flattens the input.
            - Applies dropout.
            - Passes through three dense layers (500, 250, 168 units).
            - Applies dropout again.
            - Reshapes the output to (7, 24, 1, 1).
            """
            flattened = keras.layers.Flatten(name=f"{name}_exo_flatten")(exo_input)

            input_dropout = keras.layers.Dropout(
                rate=self.dropout_rate, name=f"{name}_exo_dropout1"
            )(flattened)

            first_dense = keras.layers.Dense(
                units=500, activation="relu", name=f"{name}_exo_dense1"
            )(input_dropout)
            second_dense = keras.layers.Dense(
                units=250, activation="relu", name=f"{name}_exo_dense2"
            )(first_dense)
            third_dense = keras.layers.Dense(
                units=input_size, name=f"{name}_exo_dense3"
            )(second_dense)

            output_dropout = keras.layers.Dropout(
                rate=self.dropout_rate, name=f"{name}_exo_dropout2"
            )(third_dense)

            reshaped_output = keras.layers.Reshape(
                (7, input_size // 7, 1, 1), name=f"{name}_exo_reshape"
            )(output_dropout)

            return reshaped_output

        def energy_mask_layer(endo_past_input: keras.Layer) -> keras.Layer:
            """Reshapes and batch-normalizes the past endogenous data.

            Args:
                endo_past_input (keras.Layer): Input layer
                    representing past endogenous data.
                    Shape (168, 1)

            Returns:
                keras.Layer: The normalized tensor after reshaping to (7, 24, 1, 1)
                    and applying batch normalization.
            """
            reshaped = keras.layers.Reshape((7, input_size // 7, 1, 1))(endo_past_input)
            normalized = keras.layers.BatchNormalization()(reshaped)

            return normalized

        def cnn_3d_model(
            combined_input: keras.Layer,
        ) -> keras.Layer:
            """
            Builds the CNN-3D model.

            Args:
                combined_input (keras.Layer): Combined reshaped layer.
                    Shape (7, 24, 3)
            Returns:
                keras.Layer: Output layer. Shape (output_size,)
            """
            conv_1 = keras.layers.Conv3D(
                filters=32,
                kernel_size=(
                    combined_input.shape[1],
                    self.kernel_width,
                    combined_input.shape[3],
                ),
                activation="relu",
                padding="valid",
            )(combined_input)

            flat = keras.layers.Flatten()(conv_1)

            dense_1 = keras.layers.Dense(units=256, activation="relu")(flat)
            dense_2 = keras.layers.Dense(units=128, activation="relu")(dense_1)
            final_dense = keras.layers.Dense(units=64, activation="relu")(dense_2)

            return final_dense

        past_exo_mask = exo_mask_layer(exo_past_input, name="past")
        endo_mask = energy_mask_layer(endo_past_input)
        future_exo_mask = exo_mask_layer(exo_future_input, name="future")

        combined_mask = keras.layers.concatenate(
            [past_exo_mask, endo_mask, future_exo_mask], axis=3
        )
        cnn_3d = cnn_3d_model(combined_mask)

        flattened = keras.layers.Flatten()(cnn_3d)

        dense_1 = keras.layers.Dense(units=256, activation="relu")(flattened)
        dense_2 = keras.layers.Dense(units=128, activation="relu")(dense_1)
        final_dense = keras.layers.Dense(units=64, activation="relu")(dense_2)

        output = keras.layers.Dense(units=output_size)(final_dense)

        model = keras.Model(
            inputs=[endo_past_input, exo_past_input, exo_future_input], outputs=output
        )

        return model

    def _fit_model(
        self, past_endos, past_exos, future_exos, future_endos, sample_weights
    ):
        import pathlib  # noqa: PLC0415q
        from tempfile import TemporaryDirectory  # noqa: PLC0415

        import keras  # noqa: PLC0415
        import tqdm.keras  # noqa: PLC0415

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
                ],
            )
            self.model_.load_weights(weights_file)

    def _time_series_to_tabular(
        self,
    ) -> tuple["np.ndarray", "np.ndarray", "np.ndarray", "np.ndarray"]:
        import contextlib  # noqa: PLC0415

        import numpy as np  # noqa: PLC0415
        import pandas as pd  # noqa: PLC0415
        import tqdm  # noqa: PLC0415
        from sktime.split.slidingwindow import SlidingWindowSplitter  # noqa: PLC0415

        data_freq = pd.Timedelta(pd.infer_freq(self._y.index))

        window_length = self._window_size // data_freq
        print(window_length)

        step_length = pd.Timedelta("1h") // data_freq
        print(step_length)

        cv = SlidingWindowSplitter(
            window_length=window_length, fh=self._fh, step_length=step_length
        )

        cv_len = cv.get_n_splits(self._y)

        # Prepare data for CNN model
        past_endos, past_exos = [], []
        future_endos, future_exos = [], []
        for train_idx, test_idx in tqdm.tqdm(
            cv.split(self._y),
            total=cv_len,
            desc="Transforming time series to tabular format",
        ):
            past_endo = self._y.iloc[train_idx]
            future_endo = self._y.iloc[test_idx]

            past_index = self._y.iloc[train_idx].index
            with contextlib.suppress(AttributeError):
                past_index = past_index.to_timestamp()
            past_exo = self._extract_temporal_features(past_index)

            future_index = self._y.iloc[test_idx].index
            with contextlib.suppress(AttributeError):
                future_index = future_index.to_timestamp()
            future_exo = self._extract_temporal_features(future_index)

            past_endos.append(past_endo)
            future_endos.append(future_endo)
            past_exos.append(past_exo)
            future_exos.append(future_exo)

        # Features
        past_endos = np.array(past_endos)
        past_exos = np.array(past_exos)
        future_exos = np.array(future_exos)

        # Targets
        future_endos = np.array(future_endos)

        return (
            past_endos,
            past_exos,
            future_endos,
            future_exos,
        )

    def _calculate_sample_weights(self, n_samples: int) -> "np.ndarray":
        """Calculate sample weights using exponential decay.

        Weights are higher for more recent samples, decaying exponentially
        towards older samples based on their position in the time series.

        Parameters
        ----------
        n_samples : int
            Number of training samples.

        Returns
        -------
        np.ndarray
            Array of sample weights with shape matching the number of training samples.
        """
        import numpy as np  # noqa: PLC0415

        # Create exponential decay weights: more recent samples have higher weight
        weights = np.exp(self.decay_rate * np.arange(n_samples))

        # Normalize to [0, 1] range
        weights = weights / weights.max()

        return weights

    @property
    def _data_freq(self) -> "pd.Timedelta":
        import pandas as pd  # noqa: PLC0415

        return pd.Timedelta(pd.infer_freq(self._y.index))
