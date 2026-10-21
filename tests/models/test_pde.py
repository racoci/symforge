"""Automated test suite for Model 1 (PDE) - Primitive & Hierarchical Depth Estimator."""

from __future__ import annotations
import time
import pytest
import torch

from symforge.core.curriculum import CurriculumLevel
from symforge.models.pde import (
    AUDIO_VOCABULARY,
    DOMAIN_VOCABULARIES,
    PDE,
    PDEDataset,
    PrimitiveDepthEstimator,
    SDF_VOCABULARY,
    compute_pde_metrics,
    generate_pde_dataset,
    predict_primitives_with_depth,
    train_pde,
)


class TestPrimitiveDepthEstimatorArchitecture:
    """Validate PDE neural architecture, tensor shape invariants, and CPU latency."""

    def test_tensor_shape_invariants_batched(self):
        """Invariant: input (B, D_in) -> output (B, |Sigma|, D_max + 1)."""
        in_dim = 128
        batch_size = 4
        max_depth = 5
        model = PrimitiveDepthEstimator(
            in_features=in_dim,
            domain="audio",
            max_depth=max_depth,
            hidden_dim=128,
        )

        x = torch.randn(batch_size, in_dim)
        logits = model(x)

        expected_primitives = len(AUDIO_VOCABULARY)
        expected_depth_bins = max_depth + 1
        assert logits.shape == (batch_size, expected_primitives, expected_depth_bins)

    def test_tensor_shape_invariants_unbatched_1d(self):
        """Invariant: input (D_in,) -> output (1, |Sigma|, D_max + 1)."""
        in_dim = 64
        model = PrimitiveDepthEstimator(
            in_features=in_dim,
            domain="sdf",
            max_depth=4,
            hidden_dim=64,
        )

        x_1d = torch.randn(in_dim)
        logits = model(x_1d)

        expected_primitives = len(SDF_VOCABULARY)
        expected_depth_bins = 5
        assert logits.shape == (1, expected_primitives, expected_depth_bins)

    def test_probability_bounds_and_predict_proba(self):
        """Invariant: probabilities from predict_proba are strictly bounded in [0, 1]."""
        in_dim = 64
        model = PrimitiveDepthEstimator(in_features=in_dim, domain="audio")
        model.eval()
        x = torch.randn(2, in_dim)

        probs = model.predict_proba(x)
        assert torch.all(probs >= 0.0)
        assert torch.all(probs <= 1.0)

        # Symmetrical consistency between predict_proba and forward(return_probs=True)
        probs_fwd = model(x, return_probs=True)
        assert torch.allclose(probs, probs_fwd, atol=1e-6)

    def test_evaluation_latency_under_2ms_cpu(self):
        """Invariant: single-sample forward pass evaluates in < 2 ms on CPU."""
        in_dim = 512
        model = PrimitiveDepthEstimator(
            in_features=in_dim,
            domain="audio",
            hidden_dim=256,
        )
        model.eval()
        x = torch.randn(1, in_dim)

        # Warmup iterations
        for _ in range(5):
            _ = model(x)

        # Measure 50 consecutive forward passes
        t0 = time.perf_counter()
        num_runs = 50
        with torch.no_grad():
            for _ in range(num_runs):
                _ = model(x)
        total_time = time.perf_counter() - t0
        avg_ms = (total_time / num_runs) * 1000.0

        assert avg_ms < 2.0, f"Average inference time {avg_ms:.3f} ms exceeded 2.0 ms threshold"

    def test_domain_vocabularies_and_custom_vocabulary(self):
        """Verify standard domain vocabularies and custom vocabulary support."""
        assert len(AUDIO_VOCABULARY) == 11
        assert len(SDF_VOCABULARY) == 12
        assert "Sine" in AUDIO_VOCABULARY
        assert "SmoothUnion" in SDF_VOCABULARY

        custom_vocab = ["Alpha", "Beta", "Gamma"]
        model = PrimitiveDepthEstimator(in_features=32, vocabulary=custom_vocab, max_depth=3)
        assert model.vocabulary == ("Alpha", "Beta", "Gamma")
        assert model.num_primitives == 3
        assert model.depth_bins == 4

    def test_invalid_arguments_raise_value_error(self):
        """Verify invalid configurations trigger informative ValueErrors."""
        with pytest.raises(ValueError, match="in_features must be positive"):
            PrimitiveDepthEstimator(in_features=-1, domain="audio")

        with pytest.raises(ValueError, match="max_depth must be non-negative"):
            PrimitiveDepthEstimator(in_features=16, domain="audio", max_depth=-2)

        with pytest.raises(ValueError, match="Unknown domain"):
            PrimitiveDepthEstimator(in_features=16, domain="unsupported_domain")

        with pytest.raises(ValueError, match="Either 'domain' or 'vocabulary'"):
            PrimitiveDepthEstimator(in_features=16)


class TestPDEPredictionAndMetrics:
    """Validate prediction extraction, thresholding, and metric computation."""

    def test_predict_primitives_with_depth_mock(self):
        """Verify predict_primitives_with_depth extracts and sorts expected (primitive, depth) pairs."""
        vocab = ["Sine", "LowPass", "Mix"]
        model = PrimitiveDepthEstimator(in_features=16, vocabulary=vocab, max_depth=3)
        model.eval()

        # Artificially set head bias to trigger specific cells
        with torch.no_grad():
            model.head.weight.zero_()
            model.head.bias.fill_(-10.0)  # Very negative logits -> probs ~ 0
            # Set Sine at depth 1 (cell: Sine=0, depth=1 -> flat index = 0 * 4 + 1 = 1)
            model.head.bias[1] = 10.0
            # Set LowPass at depth 0 (cell: LowPass=1, depth=0 -> flat index = 1 * 4 + 0 = 4)
            model.head.bias[4] = 10.0

        x = torch.zeros(1, 16)
        pairs = predict_primitives_with_depth(model, x, threshold=0.5)

        # Expected: LowPass at depth 0, Sine at depth 1 (sorted by depth then name)
        assert pairs == [("LowPass", 0), ("Sine", 1)]

    def test_compute_pde_metrics_exact_agreement(self):
        """Verify compute_pde_metrics returns 1.0 on exact ground truth agreement."""
        y_true = torch.zeros(2, 3, 4)
        y_true[0, 0, 1] = 1.0  # Sample 0: primitive 0 at depth 1
        y_true[1, 1, 0] = 1.0  # Sample 1: primitive 1 at depth 0

        # Exact predictions (logits)
        y_pred = torch.full((2, 3, 4), -10.0)
        y_pred[0, 0, 1] = 10.0
        y_pred[1, 1, 0] = 10.0

        metrics = compute_pde_metrics(y_true, y_pred, threshold=0.5)
        assert metrics["f1"] == pytest.approx(1.0, abs=1e-5)
        assert metrics["precision"] == pytest.approx(1.0, abs=1e-5)
        assert metrics["recall"] == pytest.approx(1.0, abs=1e-5)
        assert metrics["primitive_f1"] == pytest.approx(1.0, abs=1e-5)


class TestPDEDatasetGeneration:
    """Validate procedural dataset generation across Audio and SDF domains."""

    def test_generate_pde_dataset_audio_shapes_and_invariants(self):
        """Verify audio dataset generation produces valid PDEDataset with expected shapes and binary values."""
        num_samples = 6
        levels = [CurriculumLevel.L1, CurriculumLevel.L2]
        dataset = generate_pde_dataset(
            domain="audio",
            num_samples=num_samples,
            levels=levels,
            seed=42,
            max_depth=5,
            duration=0.1,
        )

        assert isinstance(dataset, PDEDataset)
        assert len(dataset) == num_samples
        assert dataset.features.shape[0] == num_samples
        assert dataset.features.ndim == 2
        assert dataset.labels.shape == (num_samples, len(AUDIO_VOCABULARY), 6)

        # Check binary presence values {0, 1}
        unique_vals = torch.unique(dataset.labels).tolist()
        for v in unique_vals:
            assert v in (0.0, 1.0)

        # Each sample must contain at least 1 primitive
        sample_counts = dataset.labels.sum(dim=(1, 2))
        assert torch.all(sample_counts >= 1.0)

    def test_generate_pde_dataset_sdf_shapes_and_invariants(self):
        """Verify SDF dataset generation produces valid PDEDataset with expected shapes and binary values."""
        num_samples = 6
        levels = [CurriculumLevel.L1, CurriculumLevel.L2]
        dataset = generate_pde_dataset(
            domain="sdf",
            num_samples=num_samples,
            levels=levels,
            seed=101,
            max_depth=5,
            grid_resolution=4,
        )

        assert isinstance(dataset, PDEDataset)
        assert len(dataset) == num_samples
        assert dataset.features.shape[0] == num_samples
        assert dataset.features.ndim == 2
        assert dataset.labels.shape == (num_samples, len(SDF_VOCABULARY), 6)

        unique_vals = torch.unique(dataset.labels).tolist()
        for v in unique_vals:
            assert v in (0.0, 1.0)

        sample_counts = dataset.labels.sum(dim=(1, 2))
        assert torch.all(sample_counts >= 1.0)

    def test_generate_pde_dataset_invalid_parameters(self):
        """Verify error checking for domain, sample count, and levels."""
        with pytest.raises(ValueError, match="Domain 'invalid' must be one of"):
            generate_pde_dataset(domain="invalid", num_samples=10, levels=[CurriculumLevel.L1])

        with pytest.raises(ValueError, match="num_samples must be positive"):
            generate_pde_dataset(domain="audio", num_samples=0, levels=[CurriculumLevel.L1])

        with pytest.raises(ValueError, match="levels sequence cannot be empty"):
            generate_pde_dataset(domain="audio", num_samples=5, levels=[])


class TestPDETrainingAndConvergenceGate:
    """Validate training loop convergence and the >= 90% F1-score Pass-to-Complete Gate."""

    def test_training_loss_decreases_monotonically_audio(self):
        """Verify training loop executes and reduces loss on Audio synthetic data."""
        dataset = generate_pde_dataset(
            domain="audio",
            num_samples=30,
            levels=[CurriculumLevel.L1, CurriculumLevel.L2],
            seed=42,
            duration=0.2,
        )

        model = PrimitiveDepthEstimator(
            in_features=dataset.features.shape[1],
            domain="audio",
            hidden_dim=128,
        )

        res = train_pde(
            model=model,
            dataset=dataset,
            num_epochs=15,
            lr=2e-3,
            batch_size=16,
            pos_weight=3.0,
        )

        assert res["epochs"] == 15
        assert len(res["history"]) == 15
        # Loss at the end should be lower than initial loss
        assert res["history"][-1] < res["history"][0]

    def test_training_loss_decreases_monotonically_sdf(self):
        """Verify training loop executes and reduces loss on SDF synthetic data."""
        dataset = generate_pde_dataset(
            domain="sdf",
            num_samples=30,
            levels=[CurriculumLevel.L1, CurriculumLevel.L2],
            seed=84,
            grid_resolution=4,
        )

        model = PrimitiveDepthEstimator(
            in_features=dataset.features.shape[1],
            domain="sdf",
            hidden_dim=128,
        )

        res = train_pde(
            model=model,
            dataset=dataset,
            num_epochs=15,
            lr=2e-3,
            batch_size=16,
            pos_weight=3.0,
        )

        assert res["epochs"] == 15
        assert len(res["history"]) == 15
        assert res["history"][-1] < res["history"][0]

    def test_pass_to_complete_gate_sdf_level_2_3_f1_score(self):
        """Pass-to-Complete Gate: Model achieves >= 90% F1-score on synthetic Level 2/3 SDF primitive identification."""
        dataset = generate_pde_dataset(
            domain="sdf",
            num_samples=60,
            levels=[CurriculumLevel.L2, CurriculumLevel.L3],
            seed=2026,
            grid_resolution=5,
        )

        model = PrimitiveDepthEstimator(
            in_features=dataset.features.shape[1],
            domain="sdf",
            hidden_dim=256,
        )

        _ = train_pde(
            model=model,
            dataset=dataset,
            num_epochs=40,
            lr=1.5e-3,
            batch_size=20,
            pos_weight=3.0,
        )

        model.eval()
        with torch.no_grad():
            preds = model(dataset.features)
            metrics = compute_pde_metrics(dataset.labels, preds, threshold=0.5)

        # Invariant: Primitive identification F1-score >= 90%
        prim_f1 = metrics["primitive_f1"]
        assert prim_f1 >= 0.90, (
            f"Pass-to-Complete Gate failed: SDF Level 2/3 primitive F1 {prim_f1:.3f} < 0.90 "
            f"(Precision: {metrics['primitive_precision']:.3f}, Recall: {metrics['primitive_recall']:.3f})"
        )

    def test_pass_to_complete_gate_audio_level_2_3_f1_score(self):
        """Pass-to-Complete Gate: Model achieves >= 90% F1-score on synthetic Level 2/3 Audio primitive identification."""
        dataset = generate_pde_dataset(
            domain="audio",
            num_samples=60,
            levels=[CurriculumLevel.L2, CurriculumLevel.L3],
            seed=2026,
            duration=0.3,
        )

        model = PrimitiveDepthEstimator(
            in_features=dataset.features.shape[1],
            domain="audio",
            hidden_dim=256,
        )

        _ = train_pde(
            model=model,
            dataset=dataset,
            num_epochs=40,
            lr=1.5e-3,
            batch_size=20,
            pos_weight=3.0,
        )

        model.eval()
        with torch.no_grad():
            preds = model(dataset.features)
            metrics = compute_pde_metrics(dataset.labels, preds, threshold=0.5)

        # Invariant: Primitive identification F1-score >= 90%
        prim_f1 = metrics["primitive_f1"]
        assert prim_f1 >= 0.90, (
            f"Pass-to-Complete Gate failed: Audio Level 2/3 primitive F1 {prim_f1:.3f} < 0.90 "
            f"(Precision: {metrics['primitive_precision']:.3f}, Recall: {metrics['primitive_recall']:.3f})"
        )
