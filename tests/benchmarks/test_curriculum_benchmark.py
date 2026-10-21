"""Automated test suite for Task 3.4: Complexity-Ordered Curriculum Benchmark (Audio vs SDF)."""

from __future__ import annotations
import logging
from typing import List
import pytest
import torch

from symforge.benchmarks.curriculum_benchmark import (
    CrossDomainComparisonResult,
    DomainBenchmarkResult,
    LevelBenchmarkSummary,
    Stage1Metrics,
    Stage2Metrics,
    Stage3Metrics,
    TaskEvaluationResult,
    compare_domains_benchmark,
    evaluate_single_task,
    generate_markdown_report,
    run_curriculum_benchmark,
)
from symforge.core.ast import (
    ADSRNode,
    BoxNode,
    FreeParam,
    GainNode,
    LowPassNode,
    MixNode,
    SineNode,
    SphereNode,
    TranslateNode,
    UnionNode,
)
from symforge.core.curriculum import CurriculumLevel, CurriculumSampler
from symforge.models.pde import (
    PrimitiveDepthEstimator,
    generate_pde_dataset,
    train_pde,
)

logger = logging.getLogger(__name__)


class TestSingleTaskEvaluation:
    """Validate single task evaluation across Audio and SDF on L1, L2, L3."""

    def test_evaluate_single_task_audio_level_1_sine(self):
        """Invariant: Single L1 audio sine task evaluates through all 3 stages."""
        f_param = FreeParam(id="f", min_val=20.0, max_val=2000.0, default_val=440.0)
        phi_param = FreeParam(id="phi", min_val=0.0, max_val=6.28, default_val=0.0)
        target = SineNode(freq=f_param, phase=phi_param)

        result = evaluate_single_task(
            target_ast=target,
            domain="audio",
            pde_model=None,
            fpo_steps=15,
            curriculum_level=CurriculumLevel.L1,
        )

        assert isinstance(result, TaskEvaluationResult)
        assert result.domain == "audio"
        assert result.curriculum_level == CurriculumLevel.L1
        assert result.stage1.cell_f1 == 1.0
        assert ("Sine", 0) in result.stage1.predicted_primitives
        assert result.stage2.assembly_success is True
        assert result.stage2.tree_validity is True
        assert result.stage2.has_holes is False
        assert result.stage3.final_loss <= result.stage3.initial_loss
        assert result.stage3.loss_reduction >= 0.0
        assert result.total_latency_ms > 0.0
        assert result.is_success is True

    def test_evaluate_single_task_audio_level_2_lowpass(self):
        """Invariant: Single L2 audio modifier task evaluates through all 3 stages."""
        f_param = FreeParam(id="f", min_val=100.0, max_val=1000.0, default_val=300.0)
        phi_param = FreeParam(id="phi", min_val=0.0, max_val=6.28, default_val=0.0)
        fc_param = FreeParam(id="fc", min_val=200.0, max_val=4000.0, default_val=800.0)
        q_param = FreeParam(id="q", min_val=0.5, max_val=3.0, default_val=1.0)
        target = LowPassNode(
            cutoff=fc_param,
            resonance=q_param,
            input_node=SineNode(freq=f_param, phase=phi_param),
        )

        result = evaluate_single_task(
            target_ast=target,
            domain="audio",
            pde_model=None,
            fpo_steps=15,
            curriculum_level=CurriculumLevel.L2,
        )

        assert result.domain == "audio"
        assert result.curriculum_level == CurriculumLevel.L2
        assert result.stage2.assembly_success is True
        assert result.stage2.tree_validity is True
        assert result.stage3.step_count > 0

    def test_evaluate_single_task_audio_level_3_mix(self):
        """Invariant: Single L3 audio binary mix task evaluates through all 3 stages."""
        sampler = CurriculumSampler(domain="audio", seed=42)
        target = sampler.sample_level(CurriculumLevel.L3)

        result = evaluate_single_task(
            target_ast=target,
            domain="audio",
            pde_model=None,
            fpo_steps=15,
            curriculum_level=CurriculumLevel.L3,
        )

        assert result.domain == "audio"
        assert result.curriculum_level == CurriculumLevel.L3
        assert result.stage2.assembly_success is True
        assert result.stage2.tree_validity is True
        assert result.stage3.loss_reduction >= 0.0

    def test_evaluate_single_task_sdf_level_1_sphere(self):
        """Invariant: Single L1 SDF sphere task evaluates through all 3 stages."""
        r_param = FreeParam(id="r", min_val=0.1, max_val=3.0, default_val=1.2)
        target = SphereNode(radius=r_param)

        result = evaluate_single_task(
            target_ast=target,
            domain="sdf",
            pde_model=None,
            fpo_steps=15,
            curriculum_level=CurriculumLevel.L1,
        )

        assert isinstance(result, TaskEvaluationResult)
        assert result.domain == "sdf"
        assert result.curriculum_level == CurriculumLevel.L1
        assert result.stage1.cell_f1 == 1.0
        assert ("Sphere", 0) in result.stage1.predicted_primitives
        assert result.stage2.assembly_success is True
        assert result.stage2.tree_validity is True
        assert result.stage3.final_loss < 0.15 or result.stage3.loss_reduction >= 0.70
        assert result.is_success is True

    def test_evaluate_single_task_sdf_level_2_translate(self):
        """Invariant: Single L2 SDF translated box task evaluates through all 3 stages."""
        w_param = FreeParam(id="w", min_val=0.2, max_val=2.0, default_val=0.8)
        h_param = FreeParam(id="h", min_val=0.2, max_val=2.0, default_val=0.8)
        d_param = FreeParam(id="d", min_val=0.2, max_val=2.0, default_val=0.8)
        dx_param = FreeParam(id="dx", min_val=-1.0, max_val=1.0, default_val=0.3)
        dy_param = FreeParam(id="dy", min_val=-1.0, max_val=1.0, default_val=-0.2)
        dz_param = FreeParam(id="dz", min_val=-1.0, max_val=1.0, default_val=0.0)

        box = BoxNode(width=w_param, height=h_param, depth=d_param)
        target = TranslateNode(dx=dx_param, dy=dy_param, dz=dz_param, child=box)

        result = evaluate_single_task(
            target_ast=target,
            domain="sdf",
            pde_model=None,
            fpo_steps=15,
            curriculum_level=CurriculumLevel.L2,
        )

        assert result.domain == "sdf"
        assert result.curriculum_level == CurriculumLevel.L2
        assert result.stage2.assembly_success is True
        assert result.stage2.tree_validity is True
        assert result.stage3.loss_reduction >= 0.0

    def test_evaluate_single_task_sdf_level_3_union(self):
        """Invariant: Single L3 SDF CSG union task evaluates through all 3 stages."""
        sampler = CurriculumSampler(domain="sdf", seed=77)
        target = sampler.sample_level(CurriculumLevel.L3)

        result = evaluate_single_task(
            target_ast=target,
            domain="sdf",
            pde_model=None,
            fpo_steps=15,
            curriculum_level=CurriculumLevel.L3,
        )

        assert result.domain == "sdf"
        assert result.curriculum_level == CurriculumLevel.L3
        assert result.stage2.assembly_success is True
        assert result.stage2.tree_validity is True
        assert result.stage3.step_count > 0

    def test_evaluate_single_task_invalid_domain_raises(self):
        """Invariant: Invalid domain raises ValueError."""
        target = SphereNode(radius=FreeParam(id="r", min_val=0.5, max_val=2.0, default_val=1.0))
        with pytest.raises(ValueError, match="Unknown domain"):
            evaluate_single_task(target_ast=target, domain="unsupported_domain")


class TestCurriculumBenchmarkDomainRuns:
    """Validate run_curriculum_benchmark for single domain executions."""

    def test_run_curriculum_benchmark_audio_oracle(self):
        """Invariant: Benchmark runs across L1, L2, L3 in Audio and compiles valid summary."""
        levels = [CurriculumLevel.L1, CurriculumLevel.L2, CurriculumLevel.L3]
        result = run_curriculum_benchmark(
            domain="audio",
            levels=levels,
            samples_per_level=2,
            seed=42,
            train_pde_first=False,
            fpo_steps=10,
        )

        assert isinstance(result, DomainBenchmarkResult)
        assert result.domain == "audio"
        assert len(result.levels) == 3
        assert len(result.all_task_results) == 6  # 3 levels * 2 samples
        for lvl in levels:
            assert lvl in result.level_summaries
            summary = result.level_summaries[lvl]
            assert summary.num_samples == 2
            assert summary.stage2_assembly_success_rate == 1.0
            assert summary.stage2_tree_validity_rate == 1.0
            assert summary.stage3_loss_reduction_mean >= 0.0
        assert 0.0 <= result.overall_success_rate <= 1.0
        assert result.total_duration_seconds > 0.0

    def test_run_curriculum_benchmark_sdf_oracle(self):
        """Invariant: Benchmark runs across L1, L2, L3 in SDF and compiles valid summary."""
        levels = [CurriculumLevel.L1, CurriculumLevel.L2, CurriculumLevel.L3]
        result = run_curriculum_benchmark(
            domain="sdf",
            levels=levels,
            samples_per_level=2,
            seed=101,
            train_pde_first=False,
            fpo_steps=10,
        )

        assert isinstance(result, DomainBenchmarkResult)
        assert result.domain == "sdf"
        assert len(result.levels) == 3
        assert len(result.all_task_results) == 6
        for lvl in levels:
            assert lvl in result.level_summaries
            summary = result.level_summaries[lvl]
            assert summary.num_samples == 2
            assert summary.stage2_assembly_success_rate == 1.0
            assert summary.stage2_tree_validity_rate == 1.0
        assert 0.0 <= result.overall_success_rate <= 1.0

    def test_run_curriculum_benchmark_invalid_params_raise(self):
        """Invariant: Non-positive samples or invalid domain raises ValueError."""
        with pytest.raises(ValueError, match="Unknown domain"):
            run_curriculum_benchmark(domain="quantum_domain")

        with pytest.raises(ValueError, match="samples_per_level must be positive"):
            run_curriculum_benchmark(domain="audio", samples_per_level=0)


class TestNeuralPDEIntegration:
    """Validate Stage 1 neural evaluation with a trained PrimitiveDepthEstimator."""

    def test_evaluate_with_trained_pde_model_audio(self):
        """Invariant: Pipeline evaluates using neural PDE predictions on synthetic audio."""
        levels = [CurriculumLevel.L1, CurriculumLevel.L2]
        ds = generate_pde_dataset(
            domain="audio",
            num_samples=16,
            levels=levels,
            seed=42,
            duration=0.2,
        )
        pde_model = PrimitiveDepthEstimator(
            in_features=ds.features.shape[1],
            domain="audio",
            hidden_dim=64,
        )
        train_pde(model=pde_model, dataset=ds, num_epochs=6, batch_size=8)

        target = SineNode(freq=FreeParam(id="f", min_val=100.0, max_val=1000.0, default_val=440.0))
        result = evaluate_single_task(
            target_ast=target,
            domain="audio",
            pde_model=pde_model,
            fpo_steps=10,
        )

        assert result.stage1.latency_ms > 0.0
        assert len(result.stage1.predicted_primitives) > 0
        assert result.stage2.assembly_success is True


class TestCrossDomainComparisonAndReporting:
    """Validate parallel comparison between Audio and SDF, learning curves, and markdown reporting."""

    def test_compare_domains_benchmark_pass_to_complete_gate(self):
        """Pass-to-Complete Gate: Runs end-to-end benchmark across L1->L3 for Audio vs SDF

        and generates comparative learning curves (loss and F1 vs curriculum depth).
        """
        levels = [CurriculumLevel.L1, CurriculumLevel.L2, CurriculumLevel.L3]
        comp_res = compare_domains_benchmark(
            levels=levels,
            samples_per_level=2,
            seed=42,
            fpo_steps=10,
            train_pde_first=False,
        )

        assert isinstance(comp_res, CrossDomainComparisonResult)
        assert comp_res.audio_result.domain == "audio"
        assert comp_res.sdf_result.domain == "sdf"

        # Check learning curves structure
        curves = comp_res.learning_curves
        assert "audio" in curves and "sdf" in curves
        for dom_key in ("audio", "sdf"):
            assert len(curves[dom_key]["final_loss"]) == 3
            assert len(curves[dom_key]["loss_reduction"]) == 3
            assert len(curves[dom_key]["cell_f1"]) == 3
            assert len(curves[dom_key]["end_to_end_success"]) == 3
            assert len(curves[dom_key]["total_latency_ms"]) == 3

        # Check comparison summary metrics
        summary = comp_res.comparison_summary
        assert summary["harder_domain_overall"] in ("audio", "sdf", "comparable")
        assert 0.0 <= summary["audio_overall_success"] <= 1.0
        assert 0.0 <= summary["sdf_overall_success"] <= 1.0
        assert "learning_difficulty_gap_success" in summary

    def test_generate_markdown_report_structure(self):
        """Invariant: generate_markdown_report produces complete report with dual diagrams."""
        levels = [CurriculumLevel.L1, CurriculumLevel.L2, CurriculumLevel.L3]
        comp_res = compare_domains_benchmark(
            levels=levels,
            samples_per_level=2,
            seed=99,
            fpo_steps=10,
            train_pde_first=False,
        )

        report = generate_markdown_report(comp_res)
        assert isinstance(report, str)

        # Invariant checks on markdown structure
        assert "# SymForge Curriculum Benchmark" in report
        assert "## 1. Executive Summary & Difficulty Verdict" in report
        assert "## 2. Curriculum Progression Breakdown by Level" in report
        assert "## 3. Comparative Learning Curves" in report
        assert "## 4. Empirical Stage-by-Stage Analysis" in report

        # Dual diagram requirement: text/ASCII table AND Mermaid graph block
        assert "```text" in report
        assert "CURRICULUM LEARNING CURVES" in report
        assert "```mermaid" in report
        assert "graph TD" in report
        assert "subgraph AudioProgression" in report
        assert "subgraph SDFProgression" in report
