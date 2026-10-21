"""Fast Differentiable Parameter Optimizer (Model 3 - FPO) for Neurosymbolic Synthesis."""

from __future__ import annotations
import copy
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from symforge.audio.stft import MultiScaleSTFTExtractor
from symforge.core.ast import (
    ASTNode,
    AudioNode,
    FreeParam,
    SDFNode,
    update_ast_params,
)
from symforge.core.transpiler import (
    DifferentiableAudioModule,
    DifferentiableProgram,
    DifferentiableSDFModule,
    transpile_ast_to_module,
)
from symforge.eval.loss import compute_infonce_loss

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Detailed telemetry and convergence metrics for a parameter optimization run."""

    initial_loss: float
    final_loss: float
    step_count: int
    converged: bool
    loss_history: List[float] = field(default_factory=list)
    param_values: Dict[str, float] = field(default_factory=dict)
    initial_params: Dict[str, float] = field(default_factory=dict)
    loss_reduction: float = 0.0
    converged_reason: str = "max_steps_reached"
    domain: str = "generic"

    @property
    def is_success(self) -> bool:
        """Return True if the optimization achieved target loss reduction or low absolute loss."""
        return self.converged or self.loss_reduction >= 0.80 or self.final_loss < 0.05


class WarmStartRegressor(nn.Module):
    """Lightweight neural regressor predicting continuous parameter warm-start values Theta_0.

    Maps target observation features (e.g. STFT spectral profiles or SDF probe distances)
    to normalized parameter offsets in [0, 1] relative to each parameter's [min_val, max_val].
    """

    def __init__(
        self,
        in_features: int = 64,
        hidden_dim: int = 128,
        out_features: int = 16,
    ) -> None:
        super().__init__()
        if in_features <= 0 or hidden_dim <= 0 or out_features <= 0:
            raise ValueError(
                f"Features and hidden dimensions must be positive integers, got "
                f"in_features={in_features}, hidden_dim={hidden_dim}, out_features={out_features}"
            )
        self.in_features = int(in_features)
        self.hidden_dim = int(hidden_dim)
        self.out_features = int(out_features)

        self.net = nn.Sequential(
            nn.Linear(self.in_features, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.out_features),
            nn.Sigmoid(),  # Maps to normalized [0, 1] parameter space
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict normalized parameter vector in [0, 1]."""
        if x.ndim == 1:
            x = x.unsqueeze(0)
            return self.net(x).squeeze(0)
        return self.net(x)

    def predict_initial_theta(
        self,
        features: torch.Tensor,
        param_specs: Sequence[FreeParam],
    ) -> torch.Tensor:
        """Predict unnormalized Theta_0 bounded by param_specs [min_val, max_val]."""
        with torch.no_grad():
            u = self.forward(features)
            if u.ndim == 2:
                u = u[0]
            theta_0: List[float] = []
            for i, p in enumerate(param_specs):
                idx = i % self.out_features
                norm_val = float(torch.clamp(u[idx], 0.0, 1.0).item())
                val = p.min_val + norm_val * (p.max_val - p.min_val)
                theta_0.append(val)
            return torch.tensor(theta_0, dtype=torch.float32)


class FastParameterOptimizer:
    """Fast Differentiable Parameter Optimizer (Model 3 - FPO).

    Given an assembled AST G (from Model 2 - TGC) and reference target observation x
    (audio waveform or SDF distance/surface samples), optimizes the continuous parameters
    Theta in R^K via fast autograd gradient refinement under box constraints.
    """

    def __init__(
        self,
        default_lr: float = 0.05,
        default_max_steps: int = 30,
        default_tol: float = 1e-4,
        scale_by_range: bool = True,
        use_cosine_annealing: bool = True,
    ) -> None:
        """Initialize the Fast Parameter Optimizer.

        Args:
            default_lr: Base learning rate for continuous parameter updates.
            default_max_steps: Maximum gradient refinement steps (default 30).
            default_tol: Convergence tolerance for absolute loss or relative loss delta.
            scale_by_range: If True, scales updates by (max_val - min_val) per parameter.
            use_cosine_annealing: If True, applies cosine annealing to learning rate.
        """
        if default_lr <= 0.0:
            raise ValueError(f"default_lr must be positive, got {default_lr}")
        if default_max_steps <= 0:
            raise ValueError(f"default_max_steps must be positive, got {default_max_steps}")
        if default_tol <= 0.0:
            raise ValueError(f"default_tol must be positive, got {default_tol}")

        self.default_lr = float(default_lr)
        self.default_max_steps = int(default_max_steps)
        self.default_tol = float(default_tol)
        self.scale_by_range = bool(scale_by_range)
        self.use_cosine_annealing = bool(use_cosine_annealing)

    @staticmethod
    def infer_domain(ast: ASTNode, domain: str = "auto") -> str:
        """Infer canonical domain ('audio' or 'sdf') from AST type or user specification."""
        dom = domain.strip().lower()
        if dom == "auto":
            if isinstance(ast, AudioNode):
                return "audio"
            if isinstance(ast, SDFNode):
                return "sdf"
            raise ValueError(
                f"Cannot infer domain for AST node type: {type(ast).__name__}. "
                f"Specify domain='audio' or domain='sdf' explicitly."
            )
        if dom in ("audio", "sdf"):
            return dom
        raise ValueError(f"Unknown domain: {domain}. Expected 'audio', 'sdf', or 'auto'.")

    def optimize(
        self,
        ast: ASTNode,
        target: Union[ASTNode, nn.Module, torch.Tensor, Callable[[torch.Tensor], torch.Tensor]],
        domain: str = "auto",
        max_steps: Optional[int] = None,
        lr: Optional[float] = None,
        tol: Optional[float] = None,
        active_param_ids: Optional[Sequence[str]] = None,
        init_strategy: str = "midpoint",
        inputs: Optional[torch.Tensor] = None,
        surface_points: Optional[torch.Tensor] = None,
        negatives: Optional[Union[torch.Tensor, Sequence[Any]]] = None,
        loss_type: str = "auto",
        warm_start_regressor: Optional[WarmStartRegressor] = None,
        observation_features: Optional[torch.Tensor] = None,
        lambda_rec: float = 1.0,
        lambda_spectral: float = 0.0,
        lambda_surface: float = 0.0,
        lambda_normal: float = 0.0,
        lambda_contrast: float = 0.0,
        temperature: float = 0.07,
    ) -> Tuple[ASTNode, OptimizationResult]:
        """Execute fast differentiable parameter optimization for AST against reference target.

        Args:
            ast: Input structural AST G to optimize continuous parameters for.
            target: Ground-truth reference target (ASTNode, PyTorch module, waveform/distance tensor, or callable).
            domain: Domain specification ('audio', 'sdf', or 'auto').
            max_steps: Maximum gradient steps (defaults to self.default_max_steps).
            lr: Base learning rate (defaults to self.default_lr).
            tol: Convergence tolerance (defaults to self.default_tol).
            active_param_ids: Optional subset of parameter IDs to optimize, keeping others fixed.
            init_strategy: Parameter initialization strategy ('midpoint', 'default', 'warm_start').
            inputs: Domain input coordinates (time t for audio, 3D points p for SDF).
            surface_points: Optional 3D surface points (|d(p)| = 0) for SDF surface loss.
            negatives: Optional negative distractor tensors or modules for InfoNCE loss.
            loss_type: Loss objective type ('auto', 'l2', 'l1', 'spectral', 'surface', 'contrastive').
            warm_start_regressor: Optional neural regressor to predict initial Theta_0.
            observation_features: Optional feature tensor passed to warm_start_regressor.
            lambda_rec: Weight for L1/L2 reconstruction loss.
            lambda_spectral: Weight for multi-scale spectral loss (audio).
            lambda_surface: Weight for surface point distance loss (SDF).
            lambda_normal: Weight for normal alignment cosine loss (SDF).
            lambda_contrast: Weight for InfoNCE contrastive perceptual loss.
            temperature: Softmax temperature parameter for InfoNCE contrastive loss.

        Returns:
            Tuple of (optimized_ast, optimization_result).
        """
        dom = self.infer_domain(ast, domain)
        steps_limit = int(max_steps if max_steps is not None else self.default_max_steps)
        base_lr = float(lr if lr is not None else self.default_lr)
        tolerance = float(tol if tol is not None else self.default_tol)

        # Step 1: Transpile AST G to differentiable PyTorch module
        module = transpile_ast_to_module(ast)
        param_specs = module.param_specs

        # Handle zero-parameter degenerate programs
        if not param_specs:
            inputs_eval = self._resolve_default_inputs(dom, inputs, target)
            init_l = self._compute_loss(
                module=module,
                target=target,
                inputs=inputs_eval,
                domain=dom,
                loss_type=loss_type,
                surface_points=surface_points,
                negatives=negatives,
                lambda_rec=lambda_rec,
                lambda_spectral=lambda_spectral,
                lambda_surface=lambda_surface,
                lambda_normal=lambda_normal,
                lambda_contrast=lambda_contrast,
                temperature=temperature,
            ).item()
            return ast, OptimizationResult(
                initial_loss=init_l,
                final_loss=init_l,
                step_count=0,
                converged=True,
                loss_history=[init_l],
                param_values={},
                initial_params={},
                loss_reduction=0.0,
                converged_reason="no_free_parameters",
                domain=dom,
            )

        mins = torch.tensor([p.min_val for p in param_specs], dtype=torch.float32)
        maxs = torch.tensor([p.max_val for p in param_specs], dtype=torch.float32)
        ranges = torch.clamp(maxs - mins, min=1e-6)

        # Step 2: Initialize parameters Theta_0
        initial_params: Dict[str, float] = {}
        active_set = set(active_param_ids) if active_param_ids is not None else None

        init_vals: List[float] = []
        if init_strategy == "warm_start" and warm_start_regressor is not None:
            feat = observation_features
            if feat is None:
                feat = self._extract_default_features(target, dom)
            theta_0 = warm_start_regressor.predict_initial_theta(feat, param_specs)
            for i, p in enumerate(param_specs):
                if active_set is None or p.id in active_set:
                    init_vals.append(float(theta_0[i].item()))
                else:
                    init_vals.append(p.default_val)
        elif init_strategy == "midpoint":
            for p in param_specs:
                if active_set is None or p.id in active_set:
                    init_vals.append(0.5 * (p.min_val + p.max_val))
                else:
                    init_vals.append(p.default_val)
        elif init_strategy == "default":
            for p in param_specs:
                init_vals.append(p.default_val)
        else:
            raise ValueError(
                f"Unknown init_strategy: {init_strategy}. Expected 'midpoint', 'default', or 'warm_start'."
            )

        with torch.no_grad():
            module.theta.copy_(torch.clamp(torch.tensor(init_vals, dtype=torch.float32), mins, maxs))

        for p in param_specs:
            idx = module.param_map[p.id]
            initial_params[p.id] = float(module.theta[idx].item())

        # Resolve inputs
        inputs_eval = self._resolve_default_inputs(dom, inputs, target)

        # Resolve active parameter mask
        if active_set is not None:
            mask = torch.tensor(
                [1.0 if p.id in active_set else 0.0 for p in param_specs],
                dtype=torch.float32,
            )
        else:
            mask = torch.ones(len(param_specs), dtype=torch.float32)

        # Step 3: Fast Differentiable Refinement Loop
        m = torch.zeros_like(module.theta)
        v = torch.zeros_like(module.theta)
        beta1, beta2 = 0.9, 0.999
        eps_opt = 1e-8

        # Evaluate initial loss
        with torch.no_grad():
            init_loss_tensor = self._compute_loss(
                module=module,
                target=target,
                inputs=inputs_eval,
                domain=dom,
                loss_type=loss_type,
                surface_points=surface_points,
                negatives=negatives,
                lambda_rec=lambda_rec,
                lambda_spectral=lambda_spectral,
                lambda_surface=lambda_surface,
                lambda_normal=lambda_normal,
                lambda_contrast=lambda_contrast,
                temperature=temperature,
            )
            initial_loss = float(init_loss_tensor.item())

        loss_history: List[float] = [initial_loss]
        best_loss = initial_loss
        best_theta = module.theta.detach().clone()
        converged = False
        converged_reason = "max_steps_reached"
        effective_step = 0

        # Safety loop cap enforced
        safety_max = min(steps_limit, 1000)
        for step in range(1, safety_max + 1):
            effective_step = step
            module.zero_grad()

            loss_val = self._compute_loss(
                module=module,
                target=target,
                inputs=inputs_eval,
                domain=dom,
                loss_type=loss_type,
                surface_points=surface_points,
                negatives=negatives,
                lambda_rec=lambda_rec,
                lambda_spectral=lambda_spectral,
                lambda_surface=lambda_surface,
                lambda_normal=lambda_normal,
                lambda_contrast=lambda_contrast,
                temperature=temperature,
            )

            current_loss = float(loss_val.item())
            loss_history.append(current_loss)

            if current_loss < best_loss:
                best_loss = current_loss
                best_theta = module.theta.detach().clone()

            # Check early convergence criteria
            loss_reduction_so_far = (initial_loss - current_loss) / (initial_loss + 1e-8)
            if current_loss < tolerance:
                converged = True
                converged_reason = "tolerance_reached"
                break
            if current_loss < 0.05 and loss_reduction_so_far >= 0.80 and step >= 10:
                converged = True
                converged_reason = "gate_threshold_achieved"
                break

            loss_val.backward()

            with torch.no_grad():
                grad = module.theta.grad
                if grad is None:
                    grad = torch.zeros_like(module.theta)
                else:
                    grad = grad * mask

                # Gradient norm stopping criterion
                if torch.norm(grad).item() < tolerance * 1e-2 and step > 3:
                    converged = True
                    converged_reason = "grad_norm_converged"
                    break

                # Adam biased first and second moment updates
                m = beta1 * m + (1.0 - beta1) * grad
                v = beta2 * v + (1.0 - beta2) * (grad ** 2)

                m_hat = m / (1.0 - beta1 ** step)
                v_hat = v / (1.0 - beta2 ** step)

                # Cosine annealing learning rate schedule
                if self.use_cosine_annealing:
                    lr_t = base_lr * 0.5 * (1.0 + math.cos(math.pi * step / safety_max))
                else:
                    lr_t = base_lr

                step_direction = m_hat / (torch.sqrt(v_hat) + eps_opt)
                if self.scale_by_range:
                    delta = lr_t * ranges * step_direction
                else:
                    delta = lr_t * step_direction

                module.theta.data -= delta

                # Box constraint enforcement: projection clamp
                module.theta.data.clamp_(mins, maxs)

        # Restore best parameters encountered during optimization
        with torch.no_grad():
            module.theta.copy_(best_theta)

        final_loss = best_loss

        if len(loss_history) <= effective_step:
            loss_history.append(final_loss)

        loss_reduction = (initial_loss - final_loss) / (initial_loss + 1e-8)
        if final_loss < tolerance or final_loss < 0.05 or loss_reduction >= 0.80:
            converged = True

        # Step 4: Parameter Re-injection into cloned AST
        final_param_values: Dict[str, float] = {}
        for p in param_specs:
            idx = module.param_map[p.id]
            final_param_values[p.id] = float(module.theta[idx].item())

        optimized_ast = update_ast_params(ast, final_param_values)

        result = OptimizationResult(
            initial_loss=initial_loss,
            final_loss=final_loss,
            step_count=effective_step,
            converged=converged,
            loss_history=loss_history,
            param_values=final_param_values,
            initial_params=initial_params,
            loss_reduction=loss_reduction,
            converged_reason=converged_reason,
            domain=dom,
        )

        return optimized_ast, result

    def batch_optimize(
        self,
        asts: Sequence[ASTNode],
        targets: Sequence[Any],
        domain: str = "auto",
        max_steps: Optional[int] = None,
        lr: Optional[float] = None,
        **kwargs: Any,
    ) -> List[Tuple[ASTNode, OptimizationResult]]:
        """Batch optimization over a sequence of ASTs and corresponding targets.

        Args:
            asts: Sequence of ASTNode graphs.
            targets: Sequence of reference targets.
            domain: Domain specification.
            max_steps: Maximum gradient steps.
            lr: Base learning rate.
            **kwargs: Extra arguments forwarded to optimize().

        Returns:
            List of (optimized_ast, OptimizationResult) tuples.
        """
        if len(asts) != len(targets):
            raise ValueError(
                f"Number of ASTs ({len(asts)}) must match number of targets ({len(targets)})"
            )

        results: List[Tuple[ASTNode, OptimizationResult]] = []
        for i, (ast_item, target_item) in enumerate(zip(asts, targets)):
            opt_ast, res = self.optimize(
                ast=ast_item,
                target=target_item,
                domain=domain,
                max_steps=max_steps,
                lr=lr,
                **kwargs,
            )
            results.append((opt_ast, res))
        return results

    def _resolve_default_inputs(
        self,
        domain: str,
        inputs: Optional[torch.Tensor],
        target: Any,
    ) -> torch.Tensor:
        """Resolve coordinate tensor inputs (t for audio, p for SDF) with sensible defaults."""
        if inputs is not None:
            return inputs

        if domain == "audio":
            # If target is raw 1D audio waveform tensor, use its length at 16kHz
            if isinstance(target, torch.Tensor) and target.ndim == 1:
                t_len = target.shape[0]
                sr = 16000.0
                return torch.linspace(0.0, (t_len - 1) / sr, t_len, dtype=torch.float32)
            # Default audio evaluation window: 20 ms at 16kHz (320 samples)
            return torch.linspace(0.0, 0.02, 320, dtype=torch.float32)

        if domain == "sdf":
            # Deterministic spatial probe grid in [-1.5, 1.5]^3 for fast, reproducible gradients
            g = torch.Generator().manual_seed(42)
            return torch.randn(500, 3, generator=g, dtype=torch.float32) * 1.5

        raise ValueError(f"Unsupported domain: {domain}")

    def _extract_default_features(self, target: Any, domain: str) -> torch.Tensor:
        """Extract a fixed-size 1D feature tensor for warm-start regressor input."""
        if isinstance(target, torch.Tensor):
            flat = target.flatten()
            if flat.shape[0] >= 64:
                return flat[:64]
            return F.pad(flat, (0, 64 - flat.shape[0]))

        if domain == "audio":
            t = torch.linspace(0.0, 0.02, 320)
            if isinstance(target, ASTNode):
                mod = transpile_ast_to_module(target)
                y = mod(t)
            elif callable(target):
                y = target(t)
            else:
                y = torch.zeros(320)
            return y[:64]

        # SDF domain
        p = torch.randn(64, 3)
        if isinstance(target, ASTNode):
            mod = transpile_ast_to_module(target)
            d = mod(p)
        elif callable(target):
            d = target(p)
        else:
            d = torch.zeros(64)
        return d[:64]

    def _compute_loss(
        self,
        module: DifferentiableProgram,
        target: Any,
        inputs: torch.Tensor,
        domain: str,
        loss_type: str,
        surface_points: Optional[torch.Tensor],
        negatives: Optional[Any],
        lambda_rec: float,
        lambda_spectral: float,
        lambda_surface: float,
        lambda_normal: float,
        lambda_contrast: float,
        temperature: float,
    ) -> torch.Tensor:
        """Compute the composite loss objective for audio or SDF."""
        # 1. Forward pass on synthesized module
        pred = module(inputs)

        # 2. Forward pass or extraction of ground truth target
        if isinstance(target, ASTNode):
            target_mod = transpile_ast_to_module(target)
            target_out = target_mod(inputs).detach()
        elif isinstance(target, nn.Module) or callable(target):
            target_out = target(inputs).detach()
        elif isinstance(target, torch.Tensor):
            target_out = target.detach()
            if target_out.shape != pred.shape and target_out.ndim == 1:
                # Interpolate if dimension mismatch
                target_out = F.interpolate(
                    target_out.unsqueeze(0).unsqueeze(0),
                    size=pred.shape[-1],
                    mode="linear",
                    align_corners=False,
                ).squeeze(0).squeeze(0)
        else:
            raise TypeError(f"Unsupported target type: {type(target)}")

        # 3. Base reconstruction loss
        if loss_type in ("l1", "mae"):
            rec_loss = F.l1_loss(pred, target_out)
        else:
            rec_loss = F.mse_loss(pred, target_out)

        total_loss = lambda_rec * rec_loss

        # 4. Domain-specific loss enhancements
        if domain == "audio":
            def _ensure_stft_len(x_tensor: torch.Tensor) -> torch.Tensor:
                if x_tensor.shape[-1] < 512:
                    return F.pad(x_tensor, (0, 512 - x_tensor.shape[-1]))
                return x_tensor

            if lambda_spectral > 0.0 or loss_type == "spectral":
                stft_extractor = MultiScaleSTFTExtractor(fft_sizes=(128, 256, 512))
                z_pred = stft_extractor(_ensure_stft_len(pred))
                z_target = stft_extractor(_ensure_stft_len(target_out))
                spectral_loss = 1.0 - F.cosine_similarity(z_pred, z_target, dim=-1).mean()
                weight = lambda_spectral if lambda_spectral > 0.0 else 1.0
                total_loss = total_loss + weight * spectral_loss

            if lambda_contrast > 0.0 and negatives is not None:
                stft_extractor = MultiScaleSTFTExtractor(fft_sizes=(128, 256, 512))
                z_pred = stft_extractor(_ensure_stft_len(pred))
                z_target = stft_extractor(_ensure_stft_len(target_out))
                if isinstance(negatives, (list, tuple)):
                    neg_embs = []
                    for n in negatives:
                        if isinstance(n, ASTNode):
                            n_wave = transpile_ast_to_module(n)(inputs).detach()
                            neg_embs.append(stft_extractor(_ensure_stft_len(n_wave)))
                        elif isinstance(n, torch.Tensor):
                            neg_embs.append(stft_extractor(_ensure_stft_len(n)))
                        elif callable(n):
                            neg_embs.append(stft_extractor(_ensure_stft_len(n(inputs).detach())))
                    z_negs = torch.cat(neg_embs, dim=0) if neg_embs else torch.zeros_like(z_target)
                elif isinstance(negatives, torch.Tensor):
                    z_negs = stft_extractor(_ensure_stft_len(negatives))
                else:
                    z_negs = torch.zeros_like(z_target)
                infonce = compute_infonce_loss(
                    query=z_pred,
                    positive=z_target,
                    negatives=z_negs,
                    temperature=temperature,
                    normalize=True,
                )
                total_loss = total_loss + lambda_contrast * infonce

        elif domain == "sdf":
            # Surface point distance: |d(p_surface)|
            if (lambda_surface > 0.0 or loss_type == "surface") and surface_points is not None:
                d_surf = module(surface_points)
                surf_loss = torch.abs(d_surf).mean()
                weight_surf = lambda_surface if lambda_surface > 0.0 else 1.0
                total_loss = total_loss + weight_surf * surf_loss

            # Normal cosine similarity alignment
            if lambda_normal > 0.0:
                normals_pred = self._compute_sdf_normals(module, inputs[:64])
                if isinstance(target, (ASTNode, nn.Module)) or callable(target):
                    t_eval = transpile_ast_to_module(target) if isinstance(target, ASTNode) else target
                    normals_target = self._compute_sdf_normals(t_eval, inputs[:64])
                    norm_sim = (normals_pred * normals_target).sum(dim=-1).mean()
                    total_loss = total_loss + lambda_normal * (1.0 - norm_sim)

            # InfoNCE contrastive term
            if lambda_contrast > 0.0 and negatives is not None:
                if isinstance(negatives, torch.Tensor):
                    z_negs = negatives
                else:
                    neg_tensors = []
                    for n in negatives:
                        if isinstance(n, ASTNode):
                            neg_tensors.append(transpile_ast_to_module(n)(inputs).detach())
                        elif callable(n):
                            neg_tensors.append(n(inputs).detach())
                        elif isinstance(n, torch.Tensor):
                            neg_tensors.append(n.detach())
                    z_negs = torch.stack(neg_tensors, dim=0) if neg_tensors else torch.zeros_like(target_out).unsqueeze(0)
                infonce = compute_infonce_loss(
                    query=pred,
                    positive=target_out,
                    negatives=z_negs,
                    temperature=temperature,
                    normalize=True,
                )
                total_loss = total_loss + lambda_contrast * infonce

        return total_loss

    def _compute_sdf_normals(
        self,
        sdf_func: Callable[[torch.Tensor], torch.Tensor],
        p: torch.Tensor,
        delta: float = 1e-3,
    ) -> torch.Tensor:
        """Compute surface normals via central finite differences."""
        dev = p.device
        dtype = p.dtype
        e_x = torch.tensor([delta, 0.0, 0.0], device=dev, dtype=dtype)
        e_y = torch.tensor([0.0, delta, 0.0], device=dev, dtype=dtype)
        e_z = torch.tensor([0.0, 0.0, delta], device=dev, dtype=dtype)

        dx = (sdf_func(p + e_x) - sdf_func(p - e_x)) / (2.0 * delta)
        dy = (sdf_func(p + e_y) - sdf_func(p - e_y)) / (2.0 * delta)
        dz = (sdf_func(p + e_z) - sdf_func(p - e_z)) / (2.0 * delta)

        grad = torch.stack([dx, dy, dz], dim=-1)
        norm = torch.clamp(torch.linalg.vector_norm(grad, dim=-1, keepdim=True), min=1e-8)
        return grad / norm


# Global singleton optimizer for convenience functions
_GLOBAL_FPO = FastParameterOptimizer()


def optimize(
    ast: ASTNode,
    target: Union[ASTNode, nn.Module, torch.Tensor, Callable[[torch.Tensor], torch.Tensor]],
    domain: str = "auto",
    max_steps: int = 30,
    lr: float = 0.05,
    tol: float = 1e-4,
    **kwargs: Any,
) -> Tuple[ASTNode, OptimizationResult]:
    """Convenience functional API for single AST parameter optimization."""
    return _GLOBAL_FPO.optimize(
        ast=ast,
        target=target,
        domain=domain,
        max_steps=max_steps,
        lr=lr,
        tol=tol,
        **kwargs,
    )


def batch_optimize(
    asts: Sequence[ASTNode],
    targets: Sequence[Any],
    domain: str = "auto",
    max_steps: int = 30,
    lr: float = 0.05,
    **kwargs: Any,
) -> List[Tuple[ASTNode, OptimizationResult]]:
    """Convenience functional API for batch AST parameter optimization."""
    return _GLOBAL_FPO.batch_optimize(
        asts=asts,
        targets=targets,
        domain=domain,
        max_steps=max_steps,
        lr=lr,
        **kwargs,
    )


# Alias FPO for concise architectural naming
FPO = FastParameterOptimizer
