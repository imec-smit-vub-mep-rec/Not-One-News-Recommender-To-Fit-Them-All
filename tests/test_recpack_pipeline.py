"""Focused tests for RecPack pipeline algorithm wiring."""

import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.evaluation import recpack_pipeline as rp


class _MatrixWrapper:
    def __init__(self, values):
        self.values = values


class _FakeMetric:
    def __init__(self, _k):
        self.value = 0.0

    def calculate(self, _y_true, _y_pred):
        self.value = 1.0


class _FakeMultVAE:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.fit_calls = []
        _FakeMultVAE.instances.append(self)

    def fit(self, *args, **kwargs):
        self.fit_calls.append((args, kwargs))

    def predict(self, X):
        arr = np.full(X.shape, 0.1, dtype=np.float64)
        return csr_matrix(arr)


def _install_fake_metrics_module(monkeypatch):
    metrics_mod = ModuleType("recpack.metrics")
    metrics_mod.NDCGK = _FakeMetric
    metrics_mod.RecallK = _FakeMetric
    metrics_mod.PrecisionK = _FakeMetric
    metrics_mod.CoverageK = _FakeMetric
    monkeypatch.setitem(sys.modules, "recpack.metrics", metrics_mod)


def test_get_available_algorithms_includes_multvae(monkeypatch):
    """Registry should expose MultVAE when RecPack classes are available."""
    monkeypatch.setattr(rp, "HAS_RECPACK", True)
    monkeypatch.setattr(rp, "Popularity", object)
    monkeypatch.setattr(rp, "ItemKNN", object)
    monkeypatch.setattr(rp, "EASE", object)
    monkeypatch.setattr(rp, "MultVAE", _FakeMultVAE)

    available = rp.get_available_algorithms(include_content_based=False)
    assert "MultVAE" in available
    assert available["MultVAE"] is _FakeMultVAE


def test_run_evaluation_passes_multvae_params_and_validation_tuple(monkeypatch):
    """run_evaluation should pass params to ctor and validation_data to fit."""
    _FakeMultVAE.instances = []
    _install_fake_metrics_module(monkeypatch)

    train = csr_matrix(np.array([[1, 1, 0], [0, 1, 1]], dtype=np.float64))
    test_in = csr_matrix(np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float64))
    test_out = csr_matrix(np.array([[0, 1, 0], [0, 0, 1]], dtype=np.float64))
    val_in = csr_matrix(np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float64))
    val_out = csr_matrix(np.array([[0, 1, 0], [0, 0, 1]], dtype=np.float64))

    class _Scenario:
        def __init__(self, validation=True, seed=42):
            self.validation = validation
            self.seed = seed

        def split(self, _interaction_matrix):
            self.full_training_data = _MatrixWrapper(train)
            self.test_data = (_MatrixWrapper(test_in), _MatrixWrapper(test_out))
            self.validation_data = (_MatrixWrapper(val_in), _MatrixWrapper(val_out))

    monkeypatch.setattr(rp, "HAS_RECPACK", True)
    monkeypatch.setattr(rp, "LastItemPrediction", _Scenario)
    monkeypatch.setattr(
        rp,
        "get_available_algorithms",
        lambda include_content_based=True: {"MultVAE": _FakeMultVAE},
    )

    results = rp.run_evaluation(
        interaction_matrix=_MatrixWrapper(train),
        algorithms=["MultVAE"],
        algorithm_params={"MultVAE": {"batch_size": 256, "predict_topK": 3}},
        k_values=[10, 20],
        seed=123,
    )

    assert isinstance(results, pd.DataFrame)
    assert not results.empty
    assert _FakeMultVAE.instances, "Expected MultVAE instance to be created"

    instance = _FakeMultVAE.instances[0]
    assert instance.kwargs["batch_size"] == 256
    assert instance.kwargs["predict_topK"] == 3
    assert instance.kwargs["stop_early"] is True
    assert instance.kwargs["seed"] == 123

    fit_args, fit_kwargs = instance.fit_calls[0]
    assert fit_args
    assert "validation_data" in fit_kwargs
    validation_data = fit_kwargs["validation_data"]
    assert isinstance(validation_data, tuple)
    assert len(validation_data) == 2
    assert validation_data[1].nnz > 0


def test_multvae_uses_fallback_validation_when_scenario_has_none(monkeypatch):
    """If scenario lacks validation_data, fallback split should be generated."""
    _FakeMultVAE.instances = []
    _install_fake_metrics_module(monkeypatch)

    train = csr_matrix(np.array([[1, 1, 1], [1, 1, 0]], dtype=np.float64))
    test_in = csr_matrix(np.array([[1, 0, 0], [1, 0, 0]], dtype=np.float64))
    test_out = csr_matrix(np.array([[0, 1, 0], [0, 1, 0]], dtype=np.float64))

    class _ScenarioNoValidation:
        def __init__(self, validation=True, seed=42):
            self.validation = validation
            self.seed = seed

        def split(self, _interaction_matrix):
            self.full_training_data = _MatrixWrapper(train)
            self.test_data = (_MatrixWrapper(test_in), _MatrixWrapper(test_out))

    monkeypatch.setattr(rp, "HAS_RECPACK", True)
    monkeypatch.setattr(rp, "LastItemPrediction", _ScenarioNoValidation)
    monkeypatch.setattr(
        rp,
        "get_available_algorithms",
        lambda include_content_based=True: {"MultVAE": _FakeMultVAE},
    )

    results = rp.run_evaluation(
        interaction_matrix=_MatrixWrapper(train),
        algorithms=["MultVAE"],
        k_values=[10],
        seed=42,
    )

    assert isinstance(results, pd.DataFrame)
    assert not results.empty
    fit_args, fit_kwargs = _FakeMultVAE.instances[0].fit_calls[0]
    assert fit_args
    assert "validation_data" in fit_kwargs
    fallback_val_in, fallback_val_out = fit_kwargs["validation_data"]
    assert fallback_val_in.shape == train.shape
    assert fallback_val_out.nnz > 0
