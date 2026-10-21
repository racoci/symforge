"""Primitive and Hierarchical Depth Estimator (Model 1 - PDE) for Neurosymbolic Synthesis."""

from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

from symforge.core.ast import ASTNode
from symforge.core.curriculum import CurriculumLevel, CurriculumSampler
from symforge.core.transpiler import transpile_ast_to_module
from symforge.data.harvester import ProductionTupleHarvester

logger = logging.getLogger(__name__)

# Decoupled Domain Registries
AUDIO_VOCABULARY: Tuple[str, ...] = (
    "Sine",
    "Saw",
    "Square",
    "Noise",
    "ADSR",
    "ExpDecay",
    "LFO",
    "FM",
    "LowPass",
    "Gain",
    "Mix",
)

SDF_VOCABULARY: Tuple[str, ...] = (
    "Sphere",
    "Box",
    "Cylinder",
    "Torus",
    "Plane",
    "Translate",
    "Rotate",
    "Scale",
    "Union",
    "Difference",
    "Intersection",
    "SmoothUnion",
)

DOMAIN_VOCABULARIES: Dict[str, Tuple[str, ...]] = {
    "audio": AUDIO_VOCABULARY,
    "sdf": SDF_VOCABULARY,
}


def _strip_common_suffixes(raw: str) -> str:
    """Strip known class suffixes to normalize primitive tag names."""
    for suffix in ("Node", "Filter", "Oscillator", "Envelope", "SDF"):
        if raw.endswith(suffix) and len(raw) > len(suffix):
            return raw[: -len(suffix)]
    return raw


def _normalize_primitive_name(
    tag_or_production: str,
    vocabulary: Sequence[str],
    free_params: Optional[Sequence[Dict[str, Any]]] = None,
) -> Optional[str]:
    """Normalize and match a raw harvested AST tag or production string to a vocabulary primitive."""
    raw = tag_or_production.split("(")[0].strip()
    raw = _strip_common_suffixes(raw)

    # Check if a Union node represents a SmoothUnion based on parameters
    if raw == "Union" and free_params and "SmoothUnion" in vocabulary:
        for p in free_params:
            p_name = str(p.get("name", "")).lower()
            val = float(p.get("default_val", 0.0))
            if p_name in ("k", "smoothness", "smooth") and val > 0.05:
                return "SmoothUnion"

    # Case-insensitive search across target vocabulary
    for v in vocabulary:
        if raw.lower() == v.lower():
            return v
    return None


class PDEDataset(TensorDataset):
    """PyTorch dataset bundling observation feature vectors X and multi-depth label matrices Y."""

    def __init__(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        vocabulary: Sequence[str],
        domain: str = "generic",
    ) -> None:
        """Initialize the PDE dataset with validated feature and label dimensions.

        Args:
            features: 2D tensor of shape (N, D_feat).
            labels: 3D tensor of shape (N, |Sigma|, D_max + 1).
            vocabulary: Sequence of primitive names corresponding to label dim 1.
            domain: Domain identifier ('audio', 'sdf', or custom).
        """
        if features.ndim != 2:
            raise ValueError(f"Expected 2D features tensor (N, D), got shape {features.shape}")
        if labels.ndim != 3:
            raise ValueError(
                f"Expected 3D labels tensor (N, |Sigma|, D_max+1), got shape {labels.shape}"
            )
        if features.shape[0] != labels.shape[0]:
            raise ValueError(
                f"Feature count {features.shape[0]} does not match label count {labels.shape[0]}"
            )

        super().__init__(features, labels)
        self.vocabulary = tuple(str(v) for v in vocabulary)
        self.domain = str(domain)

    @property
    def features(self) -> torch.Tensor:
        """Return the observation feature tensor (N, D_feat)."""
        return self.tensors[0]

    @property
    def labels(self) -> torch.Tensor:
        """Return the target presence label tensor (N, |Sigma|, D_max + 1)."""
        return self.tensors[1]


class PrimitiveDepthEstimator(nn.Module):
    """Neural Primitive and Hierarchical Depth Estimator (Model 1 - PDE).

    Maps an observation feature vector x in R^{D_in} to a 2D prediction matrix
    C in [0, 1]^{|Sigma| x (D_max + 1)} predicting presence of primitive p at AST depth l.
    Evaluates in < 2 ms on modern CPU.
    """

    def __init__(
        self,
        in_features: int,
        domain: Optional[str] = None,
        vocabulary: Optional[Sequence[str]] = None,
        max_depth: int = 5,
        hidden_dim: int = 256,
        dropout: float = 0.05,
    ) -> None:
        """Initialize PDE neural architecture.

        Args:
            in_features: Input feature embedding dimension D_in.
            domain: Domain name ('audio' or 'sdf') to load standard vocabulary.
            vocabulary: Custom list of primitive names. Overrides domain vocabulary.
            max_depth: Maximum hierarchical AST depth D_max (default: 5).
            hidden_dim: Width of hidden layers (default: 256).
            dropout: Dropout probability (default: 0.05).
        """
        super().__init__()
        if in_features <= 0:
            raise ValueError(f"in_features must be positive, got {in_features}")
        if max_depth < 0:
            raise ValueError(f"max_depth must be non-negative, got {max_depth}")

        if vocabulary is not None:
            self.vocabulary = tuple(str(v) for v in vocabulary)
            self.domain = domain or "custom"
        elif domain is not None:
            dom_key = domain.lower()
            if dom_key not in DOMAIN_VOCABULARIES:
                raise ValueError(
                    f"Unknown domain '{domain}'. Valid options: {list(DOMAIN_VOCABULARIES.keys())}"
                )
            self.vocabulary = DOMAIN_VOCABULARIES[dom_key]
            self.domain = dom_key
        else:
            raise ValueError("Either 'domain' or 'vocabulary' must be provided.")

        self.in_features = int(in_features)
        self.max_depth = int(max_depth)
        self.num_primitives = len(self.vocabulary)
        self.depth_bins = self.max_depth + 1
        self.hidden_dim = int(hidden_dim)

        self.vocab_to_idx = {name: i for i, name in enumerate(self.vocabulary)}
        out_features = self.num_primitives * self.depth_bins

        # Lightweight MLP with LayerNorm, GELU, and residual skip connection
        self.fc1 = nn.Linear(self.in_features, self.hidden_dim)
        self.ln1 = nn.LayerNorm(self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.ln2 = nn.LayerNorm(self.hidden_dim)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(self.hidden_dim, out_features)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor, return_probs: bool = False) -> torch.Tensor:
        """Forward pass computing predicted logits or probabilities.

        Args:
            x: Input feature tensor of shape (D_in,) or (B, D_in).
            return_probs: If True, applies sigmoid to return probabilities in [0, 1].

        Returns:
            Tensor of shape (B, |Sigma|, D_max + 1) containing logits or probabilities.
        """
        if x.ndim == 1:
            x = x.unsqueeze(0)
        elif x.ndim != 2:
            raise ValueError(f"Expected 1D or 2D tensor, got {x.shape}")

        h = self.drop(self.act(self.ln1(self.fc1(x))))
        h = h + self.drop(self.act(self.ln2(self.fc2(h))))
        logits = self.head(h).view(-1, self.num_primitives, self.depth_bins)

        if return_probs:
            return torch.sigmoid(logits)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Compute predicted probability matrix C in [0, 1]^{B x |Sigma| x (D_max + 1)}."""
        return self.forward(x, return_probs=True)


PDE = PrimitiveDepthEstimator


def predict_primitives_with_depth(
    model: PrimitiveDepthEstimator,
    x: torch.Tensor,
    threshold: float = 0.5,
) -> List[Tuple[str, int]]:
    """Threshold model predictions to return list of (primitive_name, depth_level) pairs.

    Args:
        model: Trained PrimitiveDepthEstimator instance.
        x: Single observation feature tensor of shape (D_in,) or (1, D_in).
        threshold: Confidence threshold for presence detection (default: 0.5).

    Returns:
        Sorted list of (primitive_name, depth_level) tuples.
    """
    model.eval()
    with torch.no_grad():
        probs = model.predict_proba(x)[0]
        mask = probs >= float(threshold)

        results: List[Tuple[str, int]] = []
        for p_idx in range(model.num_primitives):
            prim_name = model.vocabulary[p_idx]
            for depth in range(model.depth_bins):
                if mask[p_idx, depth]:
                    results.append((prim_name, depth))

        results.sort(key=lambda item: (item[1], item[0]))
        return results


def _extract_audio_embedding(
    ast: ASTNode,
    extractor: nn.Module,
    sample_rate: int,
    duration: float,
) -> torch.Tensor:
    """Render 1D waveform and extract multi-scale STFT feature vector."""
    module = transpile_ast_to_module(ast)
    t = torch.linspace(0.0, duration, int(sample_rate * duration))
    with torch.no_grad():
        sig = module(t)
        emb = extractor(sig)
    return emb.detach().cpu()


def _extract_sdf_embedding(
    ast: ASTNode,
    extractor: nn.Module,
) -> torch.Tensor:
    """Render 3D CSG distance field and extract surface raymarch feature vector."""
    module = transpile_ast_to_module(ast)
    with torch.no_grad():
        emb = extractor(module)
    return emb.detach().cpu()


def generate_pde_dataset(
    domain: str,
    num_samples: int,
    levels: Sequence[CurriculumLevel],
    seed: int = 42,
    max_depth: int = 5,
    feature_extractor: Optional[nn.Module] = None,
    sample_rate: int = 16000,
    duration: float = 0.5,
    grid_resolution: int = 5,
) -> PDEDataset:
    """Build feature tensors X and depth-stratified label tensors Y using CurriculumSampler.

    Args:
        domain: Domain specification ('audio' or 'sdf').
        num_samples: Total number of procedural samples to generate.
        levels: List of CurriculumLevel instances to sample from.
        seed: Random seed for deterministic generation.
        max_depth: Maximum tree depth D_max for stratification matrix (default: 5).
        feature_extractor: Optional pre-configured feature extractor module.
        sample_rate: Audio sampling rate in Hz (default: 16000).
        duration: Audio sample duration in seconds (default: 0.5).
        grid_resolution: SDF probe grid resolution (default: 5).

    Returns:
        PDEDataset containing features X and ground truth label matrices Y.
    """
    dom = domain.lower()
    if dom not in DOMAIN_VOCABULARIES:
        raise ValueError(f"Domain '{domain}' must be one of {list(DOMAIN_VOCABULARIES.keys())}")
    if num_samples <= 0:
        raise ValueError(f"num_samples must be positive, got {num_samples}")
    if not levels:
        raise ValueError("levels sequence cannot be empty")

    vocabulary = DOMAIN_VOCABULARIES[dom]
    vocab_map = {name: i for i, name in enumerate(vocabulary)}

    # Lazy default feature extractor instantiation
    if feature_extractor is None:
        if dom == "audio":
            from symforge.audio.stft import MultiScaleSTFTExtractor

            feature_extractor = MultiScaleSTFTExtractor(fft_sizes=(256, 512))
        else:
            from symforge.sdf.surface import SurfaceRaymarchFeatureExtractor

            feature_extractor = SurfaceRaymarchFeatureExtractor(
                grid_resolution=grid_resolution,
                num_ray_views=4,
            )

    sampler = CurriculumSampler(domain=dom, seed=seed)
    harvester = ProductionTupleHarvester()

    features_list: List[torch.Tensor] = []
    labels_list: List[torch.Tensor] = []

    for i in range(num_samples):
        target_lvl = levels[i % len(levels)]
        ast = sampler.sample_level(target_lvl)

        if dom == "audio":
            emb = _extract_audio_embedding(ast, feature_extractor, sample_rate, duration)
        else:
            emb = _extract_sdf_embedding(ast, feature_extractor)
        features_list.append(emb)

        y = torch.zeros((len(vocabulary), max_depth + 1), dtype=torch.float32)
        tuples = harvester.harvest(ast)
        for t in tuples:
            p_name = _normalize_primitive_name(t.production, vocabulary, t.free_params)
            if p_name and p_name in vocab_map:
                p_idx = vocab_map[p_name]
                depth_idx = min(t.depth_from_root, max_depth)
                y[p_idx, depth_idx] = 1.0

        labels_list.append(y.unsqueeze(0))

    x_tensor = torch.cat(features_list, dim=0)
    y_tensor = torch.cat(labels_list, dim=0)
    return PDEDataset(x_tensor, y_tensor, vocabulary=vocabulary, domain=dom)


def train_pde(
    model: PrimitiveDepthEstimator,
    dataset: Union[PDEDataset, Tuple[torch.Tensor, torch.Tensor], Dataset],
    num_epochs: int = 30,
    lr: float = 1e-3,
    batch_size: int = 32,
    weight_decay: float = 1e-4,
    pos_weight: Optional[float] = 3.0,
    device: Optional[Union[str, torch.device]] = None,
) -> Dict[str, Any]:
    """Train PrimitiveDepthEstimator using weighted Binary Cross-Entropy loss on CPU.

    Args:
        model: PrimitiveDepthEstimator neural network.
        dataset: Training dataset (PDEDataset, tuple (X, Y), or PyTorch Dataset).
        num_epochs: Number of complete training epochs (default: 30).
        lr: AdamW learning rate (default: 1e-3).
        batch_size: Mini-batch size (default: 32).
        weight_decay: L2 regularization factor (default: 1e-4).
        pos_weight: Loss weight multiplier for positive presence labels (default: 3.0).
        device: Computation device (defaults to CPU).

    Returns:
        Dictionary containing final loss, training loss history, and epoch count.
    """
    dev = torch.device(device if device is not None else "cpu")
    model.to(dev)

    if isinstance(dataset, tuple):
        ds: Dataset = TensorDataset(dataset[0], dataset[1])
    else:
        ds = dataset

    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    pw_tensor = (
        torch.tensor([float(pos_weight)], device=dev, dtype=torch.float32)
        if pos_weight is not None
        else None
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw_tensor)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    loss_history: List[float] = []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        batch_count = 0

        for bx, by in loader:
            bx, by = bx.to(dev), by.to(dev)
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batch_count += 1

        avg_loss = total_loss / max(1, batch_count)
        loss_history.append(avg_loss)

    return {
        "loss": loss_history[-1] if loss_history else 0.0,
        "history": loss_history,
        "epochs": num_epochs,
    }


def compute_pde_metrics(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute evaluation metrics (F1, Precision, Recall) for depth-stratified predictions.

    Args:
        y_true: Ground truth binary label tensor (N, |Sigma|, D_max + 1).
        y_pred: Predicted logits or probability tensor of same shape as y_true.
        threshold: Decision threshold for classification (default: 0.5).

    Returns:
        Dictionary with cell-level (p, l) and marginal primitive-level precision, recall, and F1.
    """
    if y_pred.min() < 0.0 or y_pred.max() > 1.0:
        probs = torch.sigmoid(y_pred)
    else:
        probs = y_pred

    pred_bin = (probs >= float(threshold)).float()
    true_bin = (y_true >= 0.5).float()

    # Exact cell-level (p, l) precision, recall, F1
    tp = (pred_bin * true_bin).sum().item()
    fp = (pred_bin * (1.0 - true_bin)).sum().item()
    fn = ((1.0 - pred_bin) * true_bin).sum().item()
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2.0 * precision * recall / (precision + recall + 1e-8)

    # Marginal primitive-level presence (p)
    pred_prim = (pred_bin.max(dim=-1).values > 0.5).float()
    true_prim = (true_bin.max(dim=-1).values > 0.5).float()
    tp_p = (pred_prim * true_prim).sum().item()
    fp_p = (pred_prim * (1.0 - true_prim)).sum().item()
    fn_p = ((1.0 - pred_prim) * true_prim).sum().item()
    prim_prec = tp_p / (tp_p + fp_p + 1e-8)
    prim_rec = tp_p / (tp_p + fn_p + 1e-8)
    prim_f1 = 2.0 * prim_prec * prim_rec / (prim_prec + prim_rec + 1e-8)

    return {
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "primitive_f1": float(prim_f1),
        "primitive_precision": float(prim_prec),
        "primitive_recall": float(prim_rec),
    }
