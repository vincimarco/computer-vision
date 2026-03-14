# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Extension template for transformers, SIMPLE version.

For series-to-series transformations, that transform a time series to another
time series, e.g., smoothing, deseasonalization, exponentiation.

For transformations that transform a time series to a feature vector,
e.g., summary statistics, word counts, see transformer_supersimple_features.py

For advanced cases, e.g., transforming panels, hierarchical data, inverse transform,
see extension templates in transformer.py or transformer_simple.py

Contains only bare minimum of implementation requirements for a functional transformer.
Covers only the case of series-to-series transformation.
Assumes pd.DataFrame used internally, and no hierarchical functionality.
Also assumes *no composition*, i.e., no transformer or other estimator components.
For advanced cases (inverse transform, composition, etc),
    see extension templates in transformer.py or transformer_simple.py

Purpose of this implementation template:
    quick implementation of new estimators following the template
    NOT a concrete class to import! This is NOT a base class or concrete class!
    This is to be used as a "fill-in" coding template.

How to use this implementation template to implement a new estimator:
- make a copy of the template in a suitable location, give it a descriptive name.
- work through all the "todo" comments below
- fill in code for mandatory methods, and optionally for optional methods
- do not write to reserved variables: is_fitted, _is_fitted, _X, _y,
    _converter_store_X, transformers_, _tags, _tags_dynamic
- you can add more private methods, but do not override BaseEstimator's private methods
    an easy way to be safe is to prefix your methods with "_custom"
- change docstrings for functions and the file
- ensure interface compatibility by sktime.utils.estimator_checks.check_estimator
- once complete: use as a local library, or contribute to sktime via PR
- more details:
  https://www.sktime.net/en/stable/developer_guide/add_estimators.html

Mandatory methods to implement:
    fitting         - _fit(self, X, y=None)
    transformation  - _transform(self, X, y=None)

Testing - required for sktime test framework and check_estimator usage:
    get default parameters for test instance(s) - get_test_params()
"""
# todo: write an informative docstring for the file or module, remove the above
# todo: add an appropriate copyright notice for your estimator
#       estimators contributed to sktime should have the copyright notice at the top
#       estimators of your own do not need to have permissive or BSD-3 copyright

# todo: uncomment the following line, enter authors' GitHub IDs
# __author__ = [authorGitHubID, anotherAuthorGitHubID]

# todo: add any necessary sktime external imports here

import numpy as np
import pandas as pd
from sktime.transformations.base import BaseTransformer

# todo: add any necessary sktime internal imports here


class CyclicalEncodingTransformer(BaseTransformer):
    """Custom transformer. todo: write docstring.

    todo: describe your custom transformer here
        fill in sections appropriately
        docstring must be numpydoc compliant

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

    # todo: fill in univariate-only tag
    _tags = {
        # capability:multivariate controls whether internal X can be multivariate
        # if False (only univariate), always applies vectorization over variables
        "capability:multivariate": False,
        # valid values: False = inner _fit, _transform receive only univariate series
        #   True = uni- and multivariate series are passed to inner methods
        #
        # specify one or multiple authors and maintainers, only for sktime contribution
        "authors": ["author1", "author2"],  # authors, GitHub handles
        "maintainers": ["maintainer1", "maintainer2"],  # maintainers, GitHub handles
        # author = significant contribution to code at some point
        #     if interfacing a 3rd party estimator, ensure to give credit to the
        #     authors of the interfaced estimator
        # maintainer = algorithm maintainer role, "owner" of the sktime class
        #     for 3rd party interfaces, the scope is the sktime class only
        # remove maintainer tag if maintained by sktime core team
        #
        # do not change these:
        # (look at advanced templates if you think these should change)
        "scitype:transform-input": "Series",
        "scitype:transform-output": "Series",
        "scitype:instancewise": True,
        "scitype:transform-labels": "None",
        "X_inner_mtype": "pd.DataFrame",
        "fit_is_empty": False,
        "capability:inverse_transform": False,
        "capability:unequal_length": True,
        "capability:missing_values": False,
    }

    # todo: add any hyper-parameters and components to constructor
    def __init__(self, max=None):
        # todo: write any hyper-parameters to self
        self.max = max
        # IMPORTANT: the self.params should never be overwritten or mutated from now on
        # for handling defaults etc, write to other attributes, e.g., self._parama

        # leave this as is
        super().__init__()

        # todo: optional, parameter checking logic (if applicable) should happen here
        # if writes derived values to self, should *not* overwrite self.parama etc
        # instead, write to self._parama, self._newparam (starting with _)

    # todo: implement this, mandatory (except in special case below)
    def _fit(self, X, y=None):
        """Fit transformer to X and y.

        private _fit containing the core logic, called from fit

        Parameters
        ----------
        X : pd.DataFrame
            if self.get_tag("capability:multivariate")==False:
                guaranteed to have a single column
            if self.get_tag("capability:multivariate")==True: no restrictions apply
        y : None, present only for interface compatibility

        Returns
        -------
        self: reference to self
        """
        # any model parameters should be written to attributes ending in "_"
        #  attributes set by the constructor must not be overwritten
        #
        # todo:
        # insert logic here
        # self.fitted_model_param_ = sthsth
        #
        return self

        # IMPORTANT: avoid side effects to X

        # Note: when interfacing a model that has fit, with parameters
        #   that are not data (X, y) or data-like
        #   but model parameters, *don't* add as arguments to fit, but treat as follows:
        #   1. pass to constructor,  2. write to self in constructor,
        #   3. read from self in _fit,  4. pass to interfaced_model.fit in _fit

    # todo: implement this, mandatory
    def _transform(self, X, y=None):
        """Transform X and return a transformed version using cyclical encoding.

        Parameters
        ----------
        X : pd.DataFrame
            Single column DataFrame to be transformed.
        y : None, present only for interface compatibility

        Returns
        -------
        Xt : pd.DataFrame
            Transformed DataFrame with sine and cosine encoded columns.
        """
        # Assume X has a single column
        max = self.max if self.max is not None else X.iloc[:, 0].max() + 1

        X_sin: pd.DataFrame = np.sin(2 * np.pi * X / max)
        X_cos: pd.DataFrame = np.cos(2 * np.pi * X / max)

        Xt = pd.concat([X_sin, X_cos], axis="columns")
        Xt.columns = ["sin", "cos"]
        return Xt
