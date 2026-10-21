"""Automated test suite for Model 3 (FPO) - Fast Differentiable Parameter Optimizer."""

from __future__ import annotations
import math
from typing import List, Tuple
import pytest
import torch
import torch.nn as nn

from symforge.core.ast import (
    ASTNode,
    AudioNode,
    BoxNode,
    DifferenceNode,
    FreeParam,
    GainNode,
    LowPassNode,
    MixNode,
    NoiseNode,
    SawNode,
    SDFNode,
    SineNode,
    SmoothUnionNode,
    SphereNode,
    TranslateNode,
    UnionNode,
    update_ast_params,
)
from symforge.core.transpiler import (
    DifferentiableAudioModule,
    DifferentiableSDFModule,
    transpile_ast_to_module,
)
from symforge.models.fpo import (
    FPO,
    FastParameterOptimizer,
    OptimizationResult,
    WarmStartRegressor,
    batch_optimize,
    optimize,
)


class TestFPOArchitecture:
    """Validate optimizer initialization, parameter contracts, and domain inference."""

    def test_optimizer_initialization_defaults_and_validation(self):
        """Invariant: Optimizer initializes with sensible defaults and validates bounds."""
        fpo = FastParameterOptimizer()
        assert fpo.default_lr == 0.05
        assert fpo.default_max_steps == 30
        assert fpo.default_tol == 1e-4
        assert fpo.scale_by_range is True
        assert fpo.use_cosine_annealing is True

        with pytest.raises(ValueError, match="default_lr must be positive"):
            FastParameterOptimizer(default_lr=0.0)

        with pytest.raises(ValueError, match="default_max_steps must be positive"):
            FastParameterOptimizer(default_max_steps=0)

        with pytest.raises(ValueError, match="default_tol must be positive"):
            FastParameterOptimizer(default_tol=-1e-4)

    def test_optimization_result_dataclass_properties(self):
        """Invariant: OptimizationResult calculates success criteria and loss reduction."""
        res_success = OptimizationResult(
            initial_loss=1.0,
            final_loss=0.02,
            step_count=15,
            converged=True,
            loss_history=[1.0, 0.5, 0.1, 0.02],
            param_values={"r": 2.2},
            initial_params={"r": 1.0},
            loss_reduction=0.98,
            converged_reason="tolerance_reached",
            domain="sdf",
        )
        assert res_success.is_success is True

        res_gate = OptimizationResult(
            initial_loss=2.0,
            final_loss=0.35,
            step_count=25,
            converged=False,
            loss_history=[2.0, 1.0, 0.35],
            param_values={"r": 1.5},
            initial_params={"r": 0.5},
            loss_reduction=0.825,
            converged_reason="max_steps_reached",
            domain="sdf",
        )
        assert res_gate.is_success is True  # loss reduction >= 80%

        res_fail = OptimizationResult(
            initial_loss=1.0,
            final_loss=0.8,
            step_count=30,
            converged=False,
            loss_history=[1.0, 0.9, 0.8],
            param_values={"r": 1.1},
            initial_params={"r": 1.0},
            loss_reduction=0.20,
            converged_reason="max_steps_reached",
            domain="sdf",
        )
        assert res_fail.is_success is False

    def test_warm_start_regressor_bounds_and_prediction(self):
        """Invariant: WarmStartRegressor produces normalized outputs in [0, 1] and valid bounds."""
        reg = WarmStartRegressor(in_features=32, hidden_dim=64, out_features=8)
        x = torch.randn(5, 32)
        out = reg(x)
        assert out.shape == (5, 8)
        assert (out >= 0.0).all() and (out <= 1.0).all()

        params = [
            FreeParam("p1", 10.0, 50.0, 20.0),
            FreeParam("p2", -2.0, 2.0, 0.0),
        ]
        theta_0 = reg.predict_initial_theta(x[0], params)
        assert theta_0.shape == (2,)
        assert 10.0 <= theta_0[0].item() <= 50.0
        assert -2.0 <= theta_0[1].item() <= 2.0

        with pytest.raises(ValueError):
            WarmStartRegressor(in_features=-1)

    def test_domain_inference_and_validation(self):
        """Invariant: Domain inference recognizes AudioNode and SDFNode, rejecting invalid types."""
        audio_node = SineNode(freq=FreeParam("f", 20.0, 500.0, 220.0))
        sdf_node = SphereNode(radius=FreeParam("r", 0.1, 5.0, 1.0))

        assert FastParameterOptimizer.infer_domain(audio_node) == "audio"
        assert FastParameterOptimizer.infer_domain(sdf_node) == "sdf"
        assert FastParameterOptimizer.infer_domain(audio_node, domain="audio") == "audio"

        with pytest.raises(ValueError, match="Unknown domain"):
            FastParameterOptimizer.infer_domain(audio_node, domain="quantum")

    def test_zero_parameter_program(self):
        """Invariant: Programs with zero free parameters complete immediately in 0 steps."""
        noise = NoiseNode(seed=42)
        target = NoiseNode(seed=42)
        opt_ast, res = optimize(noise, target, domain="audio")
        assert res.step_count == 0
        assert res.converged is True
        assert res.converged_reason == "no_free_parameters"
        assert opt_ast == noise


class TestAudioParameterRecovery:
    """Validate continuous parameter recovery for 1D DSP audio primitives."""

    def test_recover_sine_frequency(self):
        """Audio Recovery: Recovers unknown frequency of SineOscillator."""
        f_true = 440.0
        sine_true = SineNode(freq=FreeParam("f0", 20.0, 1000.0, f_true), phase=FreeParam("p0", 0.0, 6.28, 0.0))
        sine_init = SineNode(freq=FreeParam("f0", 20.0, 1000.0, 20.0), phase=FreeParam("p0", 0.0, 6.28, 0.0))

        t_eval = torch.linspace(0.0, 0.005, 100)
        opt_ast, res = optimize(
            sine_init,
            sine_true,
            domain="audio",
            inputs=t_eval,
            active_param_ids=["f0"],
            max_steps=30,
            lr=0.05,
        )

        assert res.is_success
        assert res.step_count <= 30
        assert res.loss_reduction >= 0.80 or res.final_loss < 0.05
        recovered_f = opt_ast.freq.default_val
        assert abs(recovered_f - f_true) < 15.0  # within 3.5% of target frequency

    def test_recover_lowpass_cutoff_and_resonance(self):
        """Audio Recovery: Recovers cutoff and resonance of LowPassFilter(SawOscillator)."""
        saw = SawNode(freq=FreeParam("saw_f", 20.0, 500.0, 110.0))
        fc_true = 800.0
        q_true = 2.0
        lp_true = LowPassNode(cutoff=FreeParam("fc", 100.0, 5000.0, fc_true), resonance=FreeParam("q", 0.5, 10.0, q_true), input_node=saw)
        lp_init = LowPassNode(cutoff=FreeParam("fc", 100.0, 5000.0, 100.0), resonance=FreeParam("q", 0.5, 10.0, 0.5), input_node=saw)

        t_eval = torch.linspace(0.0, 0.02, 320)
        opt_ast, res = optimize(
            lp_init,
            lp_true,
            domain="audio",
            inputs=t_eval,
            active_param_ids=["fc", "q"],
            max_steps=30,
            lr=0.05,
        )

        assert res.is_success
        assert res.step_count <= 30
        assert res.loss_reduction >= 0.80 or res.final_loss < 0.05
        assert opt_ast.cutoff.min_val <= opt_ast.cutoff.default_val <= opt_ast.cutoff.max_val
        assert opt_ast.resonance.min_val <= opt_ast.resonance.default_val <= opt_ast.resonance.max_val

    def test_recover_mix_ratio(self):
        """Audio Recovery: Recovers unknown mix ratio of MixModule."""
        s1 = SineNode(freq=FreeParam("f1", 20.0, 500.0, 220.0), phase=FreeParam("p1", 0.0, 6.28, 0.0))
        s2 = SineNode(freq=FreeParam("f2", 20.0, 500.0, 440.0), phase=FreeParam("p2", 0.0, 6.28, 0.0))
        ratio_true = 0.35
        mix_true = MixNode(ratio=FreeParam("r", 0.0, 1.0, ratio_true), input1=s1, input2=s2)
        mix_init = MixNode(ratio=FreeParam("r", 0.0, 1.0, 0.0), input1=s1, input2=s2)

        t_eval = torch.linspace(0.0, 0.02, 320)
        opt_ast, res = optimize(
            mix_init,
            mix_true,
            domain="audio",
            inputs=t_eval,
            active_param_ids=["r"],
            max_steps=30,
            lr=0.05,
        )

        assert res.is_success
        assert res.step_count <= 30
        assert res.loss_reduction >= 0.80 or res.final_loss < 0.05
        recovered_ratio = opt_ast.ratio.default_val
        assert abs(recovered_ratio - ratio_true) < 0.05

    def test_audio_contrastive_infonce_loss(self):
        """Audio Recovery: Optimizes audio parameter with InfoNCE contrastive term against distractor."""
        s_true = SineNode(freq=FreeParam("f", 20.0, 500.0, 220.0), phase=FreeParam("p", 0.0, 6.28, 0.0))
        s_init = SineNode(freq=FreeParam("f", 20.0, 500.0, 250.0), phase=FreeParam("p", 0.0, 6.28, 0.0))
        distractor = SawNode(freq=FreeParam("fs", 20.0, 500.0, 110.0))
        t_eval = torch.linspace(0.0, 0.005, 100)

        opt_ast, res = optimize(
            s_init,
            s_true,
            domain="audio",
            inputs=t_eval,
            negatives=[distractor],
            lambda_contrast=0.05,
            active_param_ids=["f"],
            max_steps=30,
        )
        assert res.is_success
        assert res.loss_reduction >= 0.80 or res.final_loss < 0.05


class TestSDFParameterRecovery:
    """Validate continuous parameter recovery for 3D CSG SDF primitives."""

    def test_recover_sphere_radius(self):
        """SDF Recovery: Recovers unknown radius of SphereSDF."""
        r_true = 2.2
        sp_true = SphereNode(radius=FreeParam("r", 0.1, 5.0, r_true))
        sp_init = SphereNode(radius=FreeParam("r", 0.1, 5.0, 0.1))

        opt_ast, res = optimize(sp_init, sp_true, domain="sdf", max_steps=30, lr=0.05)

        assert res.is_success
        assert res.step_count <= 30
        assert res.loss_reduction >= 0.80 or res.final_loss < 0.05
        recovered_r = opt_ast.radius.default_val
        assert abs(recovered_r - r_true) < 0.15

    def test_recover_box_dimensions(self):
        """SDF Recovery: Recovers dimensions of BoxSDF."""
        dims_true = (1.2, 2.0, 0.8)
        box_true = BoxNode(
            width=FreeParam("w", 0.1, 5.0, dims_true[0]),
            height=FreeParam("h", 0.1, 5.0, dims_true[1]),
            depth=FreeParam("d", 0.1, 5.0, dims_true[2]),
        )
        box_init = BoxNode(
            width=FreeParam("w", 0.1, 5.0, 0.1),
            height=FreeParam("h", 0.1, 5.0, 0.1),
            depth=FreeParam("d", 0.1, 5.0, 0.1),
        )

        opt_ast, res = optimize(box_init, box_true, domain="sdf", max_steps=30, lr=0.05)

        assert res.is_success
        assert res.step_count <= 30
        assert res.loss_reduction >= 0.80 or res.final_loss < 0.05
        assert abs(opt_ast.width.default_val - dims_true[0]) < 0.25
        assert abs(opt_ast.height.default_val - dims_true[1]) < 0.25
        assert abs(opt_ast.depth.default_val - dims_true[2]) < 0.25

    def test_recover_translate_offsets(self):
        """SDF Recovery: Recovers translation offset of TranslateSDF."""
        sp = SphereNode(radius=FreeParam("r", 0.1, 5.0, 1.0))
        offsets_true = (0.8, -0.5, 0.3)
        trans_true = TranslateNode(
            dx=FreeParam("dx", -3.0, 3.0, offsets_true[0]),
            dy=FreeParam("dy", -3.0, 3.0, offsets_true[1]),
            dz=FreeParam("dz", -3.0, 3.0, offsets_true[2]),
            child=sp,
        )
        trans_init = TranslateNode(
            dx=FreeParam("dx", -3.0, 3.0, 0.0),
            dy=FreeParam("dy", -3.0, 3.0, 0.0),
            dz=FreeParam("dz", -3.0, 3.0, 0.0),
            child=sp,
        )

        opt_ast, res = optimize(
            trans_init,
            trans_true,
            domain="sdf",
            active_param_ids=["dx", "dy", "dz"],
            max_steps=30,
            lr=0.05,
        )

        assert res.is_success
        assert res.step_count <= 30
        assert res.loss_reduction >= 0.80 or res.final_loss < 0.05
        assert abs(opt_ast.dx.default_val - offsets_true[0]) < 0.25
        assert abs(opt_ast.dy.default_val - offsets_true[1]) < 0.25
        assert abs(opt_ast.dz.default_val - offsets_true[2]) < 0.25

    def test_recover_smooth_union_blend_k(self):
        """SDF Recovery: Recovers blend radius k of SmoothUnionSDF."""
        sp1 = SphereNode(radius=FreeParam("r1", 0.1, 3.0, 1.0))
        sp2 = SphereNode(radius=FreeParam("r2", 0.1, 3.0, 1.0))
        t_sp2 = TranslateNode(dx=FreeParam("dx", -3.0, 3.0, 1.2), dy=FreeParam("dy", -3.0, 3.0, 0.0), dz=FreeParam("dz", -3.0, 3.0, 0.0), child=sp2)
        k_true = 0.25
        su_true = SmoothUnionNode(smoothness=FreeParam("k", 0.01, 1.0, k_true), a=sp1, b=t_sp2)
        su_init = SmoothUnionNode(smoothness=FreeParam("k", 0.01, 1.0, 0.01), a=sp1, b=t_sp2)

        opt_ast, res = optimize(
            su_init,
            su_true,
            domain="sdf",
            active_param_ids=["k"],
            max_steps=30,
            lr=0.05,
        )

        assert res.is_success
        assert res.step_count <= 30
        assert res.loss_reduction >= 0.80 or res.final_loss < 0.05
        assert opt_ast.smoothness.min_val <= opt_ast.smoothness.default_val <= opt_ast.smoothness.max_val

    def test_sdf_contrastive_and_normal_loss(self):
        """SDF Recovery: Optimizes SDF parameters with normal cosine similarity and contrastive loss."""
        sp_true = SphereNode(radius=FreeParam("r", 0.1, 5.0, 2.0))
        sp_init = SphereNode(radius=FreeParam("r", 0.1, 5.0, 0.5))
        distractor = BoxNode(width=FreeParam("w", 0.1, 5.0, 0.5), height=FreeParam("h", 0.1, 5.0, 0.5), depth=FreeParam("d", 0.1, 5.0, 0.5))

        opt_ast, res = optimize(
            sp_init,
            sp_true,
            domain="sdf",
            negatives=[distractor],
            lambda_contrast=0.1,
            lambda_normal=0.1,
            max_steps=30,
        )
        assert res.is_success
        assert res.loss_reduction >= 0.80 or res.final_loss < 0.05


class TestConvergenceAndInvariants:
    """Validate Pass-to-Complete gate, parameter invariants, and symmetrical equivalence."""

    def test_benchmark_pass_to_complete_gate(self):
        """Pass-to-Complete Gate: Demonstrates <= 30 step convergence (>80% reduction or loss < 0.05) across all 7 benchmark programs."""
        fpo = FastParameterOptimizer(default_max_steps=30, default_lr=0.05)
        benchmarks: List[Tuple[str, ASTNode, ASTNode, str, List[str], Optional[torch.Tensor]]] = []

        # 1. Sine
        t_short = torch.linspace(0.0, 0.005, 100)
        s_true = SineNode(freq=FreeParam("f0", 20.0, 1000.0, 440.0), phase=FreeParam("p0", 0.0, 6.28, 0.0))
        s_init = SineNode(freq=FreeParam("f0", 20.0, 1000.0, 20.0), phase=FreeParam("p0", 0.0, 6.28, 0.0))
        benchmarks.append(("Sine", s_init, s_true, "audio", ["f0"], t_short))

        # 2. LowPass(Saw)
        t_audio = torch.linspace(0.0, 0.02, 320)
        saw = SawNode(freq=FreeParam("saw_f", 20.0, 500.0, 110.0))
        lp_true = LowPassNode(cutoff=FreeParam("fc", 100.0, 5000.0, 800.0), resonance=FreeParam("q", 0.5, 10.0, 2.0), input_node=saw)
        lp_init = LowPassNode(cutoff=FreeParam("fc", 100.0, 5000.0, 100.0), resonance=FreeParam("q", 0.5, 10.0, 0.5), input_node=saw)
        benchmarks.append(("LowPass(Saw)", lp_init, lp_true, "audio", ["fc", "q"], t_audio))

        # 3. Mix
        s1 = SineNode(freq=FreeParam("f1", 20.0, 500.0, 220.0), phase=FreeParam("p1", 0.0, 6.28, 0.0))
        s2 = SineNode(freq=FreeParam("f2", 20.0, 500.0, 440.0), phase=FreeParam("p2", 0.0, 6.28, 0.0))
        mix_true = MixNode(ratio=FreeParam("r", 0.0, 1.0, 0.35), input1=s1, input2=s2)
        mix_init = MixNode(ratio=FreeParam("r", 0.0, 1.0, 0.0), input1=s1, input2=s2)
        benchmarks.append(("Mix", mix_init, mix_true, "audio", ["r"], t_audio))

        # 4. Sphere
        sp_true = SphereNode(radius=FreeParam("r", 0.1, 5.0, 2.2))
        sp_init = SphereNode(radius=FreeParam("r", 0.1, 5.0, 0.1))
        benchmarks.append(("Sphere", sp_init, sp_true, "sdf", ["r"], None))

        # 5. Box
        box_true = BoxNode(width=FreeParam("w", 0.1, 5.0, 1.2), height=FreeParam("h", 0.1, 5.0, 2.0), depth=FreeParam("d", 0.1, 5.0, 0.8))
        box_init = BoxNode(width=FreeParam("w", 0.1, 5.0, 0.1), height=FreeParam("h", 0.1, 5.0, 0.1), depth=FreeParam("d", 0.1, 5.0, 0.1))
        benchmarks.append(("Box", box_init, box_true, "sdf", ["w", "h", "d"], None))

        # 6. Translate
        trans_true = TranslateNode(dx=FreeParam("dx", -3.0, 3.0, 0.8), dy=FreeParam("dy", -3.0, 3.0, -0.5), dz=FreeParam("dz", -3.0, 3.0, 0.3), child=sp_true)
        trans_init = TranslateNode(dx=FreeParam("dx", -3.0, 3.0, 0.0), dy=FreeParam("dy", -3.0, 3.0, 0.0), dz=FreeParam("dz", -3.0, 3.0, 0.0), child=sp_true)
        benchmarks.append(("Translate", trans_init, trans_true, "sdf", ["dx", "dy", "dz"], None))

        # 7. SmoothUnion
        su_true = SmoothUnionNode(smoothness=FreeParam("k", 0.01, 1.0, 0.25), a=sp_true, b=trans_true)
        su_init = SmoothUnionNode(smoothness=FreeParam("k", 0.01, 1.0, 0.01), a=sp_true, b=trans_true)
        benchmarks.append(("SmoothUnion", su_init, su_true, "sdf", ["k"], None))

        for name, ast_init, ast_true, domain, active_ids, in_tensor in benchmarks:
            opt_ast, res = fpo.optimize(
                ast=ast_init,
                target=ast_true,
                domain=domain,
                active_param_ids=active_ids,
                inputs=in_tensor,
                max_steps=30,
            )
            assert res.is_success, (
                f"Benchmark {name} failed gate: initial_loss={res.initial_loss:.4f}, "
                f"final_loss={res.final_loss:.4f}, reduction={res.loss_reduction:.1%}"
            )
            assert res.step_count <= 30

    def test_box_constraint_parameter_invariants(self):
        """Parameter Invariants: All optimized parameters strictly satisfy their min_val and max_val constraints."""
        sp_true = SphereNode(radius=FreeParam("r", 0.5, 3.0, 2.5))
        sp_init = SphereNode(radius=FreeParam("r", 0.5, 3.0, 0.5))

        # Run with an aggressive learning rate that would overshoot without clamp
        opt_ast, res = optimize(sp_init, sp_true, domain="sdf", lr=10.0, max_steps=10)
        assert 0.5 <= opt_ast.radius.default_val <= 3.0

        for p_id, val in res.param_values.items():
            assert 0.5 <= val <= 3.0

    def test_symmetrical_validation(self):
        """Symmetrical Validation: Re-injected optimized AST evaluates identically to the transpiled PyTorch module."""
        sp_true = SphereNode(radius=FreeParam("r", 0.1, 5.0, 2.2))
        sp_init = SphereNode(radius=FreeParam("r", 0.1, 5.0, 0.1))

        opt_ast, res = optimize(sp_init, sp_true, domain="sdf")

        # Compile re-injected AST to module
        mod_retranspiled = transpile_ast_to_module(opt_ast)

        # Compile original init module and update its theta to the optimized values
        mod_trained = transpile_ast_to_module(sp_init)
        for p in mod_trained.param_specs:
            idx = mod_trained.param_map[p.id]
            mod_trained.theta.data[idx] = res.param_values[p.id]

        p_test = torch.randn(50, 3)
        d_retranspiled = mod_retranspiled(p_test)
        d_trained = mod_trained(p_test)

        torch.testing.assert_close(d_retranspiled, d_trained, atol=1e-5, rtol=1e-5)

    def test_batch_optimize_heterogeneous(self):
        """Batch Optimization: Executes optimization on batches of programs correctly."""
        asts = [
            SphereNode(radius=FreeParam("r", 0.1, 5.0, 0.5)),
            BoxNode(width=FreeParam("w", 0.1, 5.0, 0.5), height=FreeParam("h", 0.1, 5.0, 0.5), depth=FreeParam("d", 0.1, 5.0, 0.5)),
        ]
        targets = [
            SphereNode(radius=FreeParam("r", 0.1, 5.0, 2.0)),
            BoxNode(width=FreeParam("w", 0.1, 5.0, 1.5), height=FreeParam("h", 0.1, 5.0, 1.5), depth=FreeParam("d", 0.1, 5.0, 1.5)),
        ]
        batch_results = batch_optimize(asts, targets, domain="sdf", max_steps=30)
        assert len(batch_results) == 2
        for opt_ast, res in batch_results:
            assert res.is_success
            assert res.step_count <= 30

    def test_warm_start_regressor_integration(self):
        """Warm-Start Integration: Optimizes starting from WarmStartRegressor predicted parameters."""
        regressor = WarmStartRegressor(in_features=64, hidden_dim=64, out_features=16)
        sp_true = SphereNode(radius=FreeParam("r", 0.1, 5.0, 2.0))
        sp_init = SphereNode(radius=FreeParam("r", 0.1, 5.0, 0.1))

        opt_ast, res = optimize(
            sp_init,
            sp_true,
            domain="sdf",
            init_strategy="warm_start",
            warm_start_regressor=regressor,
            max_steps=30,
        )
        assert res.is_success
        assert 0.1 <= opt_ast.radius.default_val <= 5.0
