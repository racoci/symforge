"""Supervised Production Tuple Harvester for PCFG and PHOG Grammar Learning."""

from __future__ import annotations
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, TextIO, Type, Union

from symforge.core.ast import (
    ADSRNode,
    ASTNode,
    AudioHoleNode,
    AudioNode,
    BoxNode,
    CylinderNode,
    DifferenceNode,
    FMNode,
    GainNode,
    HoleNode,
    IntersectionNode,
    LowPassNode,
    MixNode,
    NoiseNode,
    RotateNode,
    SawNode,
    ScaleNode,
    SDFHoleNode,
    SDFNode,
    SineNode,
    SphereNode,
    SquareNode,
    TorusNode,
    TranslateNode,
    UnionNode,
)


@dataclass
class ProductionTuple:
    """Canonical production decision tuple for probabilistic grammar learning."""

    nonterminal: str
    production: str
    production_id: str
    depth_from_root: int
    depth_to_leaves: int
    parent_context: List[str]
    grandparent_context: Optional[str]
    left_sibling_context: Optional[str]
    ancestor_chain: List[str]
    expected_type: str
    num_free_params: int
    free_params: List[Dict[str, Any]]
    children_types: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert tuple to a JSON-serializable dictionary."""
        return {
            "nonterminal": self.nonterminal,
            "production": self.production,
            "production_id": self.production_id,
            "depth_from_root": self.depth_from_root,
            "depth_to_leaves": self.depth_to_leaves,
            "parent_context": list(self.parent_context),
            "grandparent_context": self.grandparent_context,
            "left_sibling_context": self.left_sibling_context,
            "ancestor_chain": list(self.ancestor_chain),
            "expected_type": self.expected_type,
            "num_free_params": self.num_free_params,
            "free_params": [dict(p) for p in self.free_params],
            "children_types": list(self.children_types),
        }


class ProductionTupleHarvester:
    """Traverses SymForge AST derivations and extracts supervised production tuples."""

    _NODE_REGISTRY: Dict[Type[ASTNode], Dict[str, Any]] = {
        # SDF Domain
        SphereNode: {
            "nonterminal": "SDF",
            "production": "Sphere(radius: FreeParam)",
            "production_id": "sdf_sphere",
            "expected_type": "SDF",
            "children_types": [],
            "tag": "Sphere",
        },
        BoxNode: {
            "nonterminal": "SDF",
            "production": "Box(width: FreeParam, height: FreeParam, depth: FreeParam)",
            "production_id": "sdf_box",
            "expected_type": "SDF",
            "children_types": [],
            "tag": "Box",
        },
        CylinderNode: {
            "nonterminal": "SDF",
            "production": "Cylinder(radius: FreeParam, height: FreeParam)",
            "production_id": "sdf_cylinder",
            "expected_type": "SDF",
            "children_types": [],
            "tag": "Cylinder",
        },
        TorusNode: {
            "nonterminal": "SDF",
            "production": "Torus(major_radius: FreeParam, minor_radius: FreeParam)",
            "production_id": "sdf_torus",
            "expected_type": "SDF",
            "children_types": [],
            "tag": "Torus",
        },
        TranslateNode: {
            "nonterminal": "SDF",
            "production": "Translate(dx: FreeParam, dy: FreeParam, dz: FreeParam, child: SDF)",
            "production_id": "sdf_translate",
            "expected_type": "SDF",
            "children_types": ["SDF"],
            "tag": "Translate",
        },
        RotateNode: {
            "nonterminal": "SDF",
            "production": "Rotate(rx: FreeParam, ry: FreeParam, rz: FreeParam, child: SDF)",
            "production_id": "sdf_rotate",
            "expected_type": "SDF",
            "children_types": ["SDF"],
            "tag": "Rotate",
        },
        ScaleNode: {
            "nonterminal": "SDF",
            "production": "Scale(s: FreeParam, child: SDF)",
            "production_id": "sdf_scale",
            "expected_type": "SDF",
            "children_types": ["SDF"],
            "tag": "Scale",
        },
        UnionNode: {
            "nonterminal": "SDF",
            "production": "Union(smoothness: FreeParam, a: SDF, b: SDF)",
            "production_id": "sdf_union",
            "expected_type": "SDF",
            "children_types": ["SDF", "SDF"],
            "tag": "Union",
        },
        IntersectionNode: {
            "nonterminal": "SDF",
            "production": "Intersection(a: SDF, b: SDF)",
            "production_id": "sdf_intersection",
            "expected_type": "SDF",
            "children_types": ["SDF", "SDF"],
            "tag": "Intersection",
        },
        DifferenceNode: {
            "nonterminal": "SDF",
            "production": "Difference(SDF, SDF)",
            "production_id": "sdf_difference",
            "expected_type": "SDF",
            "children_types": ["SDF", "SDF"],
            "tag": "Difference",
        },
        # Audio Domain
        SineNode: {
            "nonterminal": "Audio",
            "production": "Sine(freq: FreeParam, phase: FreeParam)",
            "production_id": "audio_sine",
            "expected_type": "Audio",
            "children_types": [],
            "tag": "Sine",
        },
        SawNode: {
            "nonterminal": "Audio",
            "production": "Saw(freq: FreeParam)",
            "production_id": "audio_saw",
            "expected_type": "Audio",
            "children_types": [],
            "tag": "Saw",
        },
        SquareNode: {
            "nonterminal": "Audio",
            "production": "Square(freq: FreeParam, duty: FreeParam)",
            "production_id": "audio_square",
            "expected_type": "Audio",
            "children_types": [],
            "tag": "Square",
        },
        NoiseNode: {
            "nonterminal": "Audio",
            "production": "Noise(seed: int)",
            "production_id": "audio_noise",
            "expected_type": "Audio",
            "children_types": [],
            "tag": "Noise",
        },
        ADSRNode: {
            "nonterminal": "Audio",
            "production": "ADSR(attack: FreeParam, decay: FreeParam, sustain: FreeParam, release: FreeParam, gate: Audio)",
            "production_id": "audio_adsr",
            "expected_type": "Audio",
            "children_types": ["Audio"],
            "tag": "ADSR",
        },
        LowPassNode: {
            "nonterminal": "Audio",
            "production": "LowPass(cutoff: FreeParam, resonance: FreeParam, input: Audio)",
            "production_id": "audio_lowpass",
            "expected_type": "Audio",
            "children_types": ["Audio"],
            "tag": "LowPass",
        },
        GainNode: {
            "nonterminal": "Audio",
            "production": "Gain(gain: FreeParam, input: Audio)",
            "production_id": "audio_gain",
            "expected_type": "Audio",
            "children_types": ["Audio"],
            "tag": "Gain",
        },
        MixNode: {
            "nonterminal": "Audio",
            "production": "Mix(ratio: FreeParam, input1: Audio, input2: Audio)",
            "production_id": "audio_mix",
            "expected_type": "Audio",
            "children_types": ["Audio", "Audio"],
            "tag": "Mix",
        },
        FMNode: {
            "nonterminal": "Audio",
            "production": "FM(carrier: FreeParam, mod_freq: FreeParam, mod_index: FreeParam)",
            "production_id": "audio_fm",
            "expected_type": "Audio",
            "children_types": [],
            "tag": "FM",
        },
        # Partial Program Holes
        AudioHoleNode: {
            "nonterminal": "Audio",
            "production": "AudioHole(id: str)",
            "production_id": "audio_hole",
            "expected_type": "Audio",
            "children_types": [],
            "tag": "AudioHole",
        },
        SDFHoleNode: {
            "nonterminal": "SDF",
            "production": "SDFHole(id: str)",
            "production_id": "sdf_hole",
            "expected_type": "SDF",
            "children_types": [],
            "tag": "SDFHole",
        },
    }

    def _get_node_info(self, node: ASTNode) -> Dict[str, Any]:
        node_cls = type(node)
        if node_cls in self._NODE_REGISTRY:
            return self._NODE_REGISTRY[node_cls]

        # Fallback for subclassing
        for registered_cls, info in self._NODE_REGISTRY.items():
            if isinstance(node, registered_cls):
                return info

        # Generic fallback
        tag = node_cls.__name__.replace("Node", "")
        domain = "Audio" if isinstance(node, AudioNode) else "SDF"
        return {
            "nonterminal": domain,
            "production": f"{tag}(...)",
            "production_id": f"{domain.lower()}_{tag.lower()}",
            "expected_type": domain,
            "children_types": [domain for _ in node.children],
            "tag": tag,
        }

    def _compute_depth_to_leaves(self, node: ASTNode) -> int:
        if not node.children:
            return 0
        return 1 + max(self._compute_depth_to_leaves(child) for child in node.children)

    def harvest(self, root: ASTNode) -> List[ProductionTuple]:
        """Traverse a single AST and extract canonical production decision tuples."""
        tuples: List[ProductionTuple] = []

        def _traverse(
            node: ASTNode,
            depth_from_root: int,
            ancestor_chain: List[str],
            parent_context: List[str],
            grandparent_context: Optional[str],
            left_sibling_context: Optional[str],
        ) -> None:
            info = self._get_node_info(node)
            depth_to_leaves = self._compute_depth_to_leaves(node)
            free_params = [p.to_dict() for p in node._free_param_fields()]

            prod_tuple = ProductionTuple(
                nonterminal=info["nonterminal"],
                production=info["production"],
                production_id=info["production_id"],
                depth_from_root=depth_from_root,
                depth_to_leaves=depth_to_leaves,
                parent_context=parent_context,
                grandparent_context=grandparent_context,
                left_sibling_context=left_sibling_context,
                ancestor_chain=ancestor_chain,
                expected_type=info["expected_type"],
                num_free_params=len(free_params),
                free_params=free_params,
                children_types=info["children_types"],
            )
            tuples.append(prod_tuple)

            child_ancestor_chain = ancestor_chain + [info["tag"]]
            new_parent_context = [info["tag"]]
            new_grandparent_context = ancestor_chain[-1] if ancestor_chain else None

            for idx, child in enumerate(node.children):
                left_sibling = (
                    self._get_node_info(node.children[idx - 1])["tag"]
                    if idx > 0
                    else None
                )
                _traverse(
                    child,
                    depth_from_root=depth_from_root + 1,
                    ancestor_chain=child_ancestor_chain,
                    parent_context=new_parent_context,
                    grandparent_context=new_grandparent_context,
                    left_sibling_context=left_sibling,
                )

        _traverse(
            root,
            depth_from_root=0,
            ancestor_chain=["Root"],
            parent_context=["Root"],
            grandparent_context=None,
            left_sibling_context=None,
        )
        return tuples

    def harvest_batch(self, nodes: List[ASTNode]) -> List[ProductionTuple]:
        """Harvest production tuples across a collection of ASTs."""
        all_tuples: List[ProductionTuple] = []
        for node in nodes:
            all_tuples.extend(self.harvest(node))
        return all_tuples

    def to_dict_list(self, tuples: List[ProductionTuple]) -> List[Dict[str, Any]]:
        """Convert a list of ProductionTuples into serializable dictionaries."""
        return [t.to_dict() for t in tuples]

    def export_jsonl(
        self, tuples: List[ProductionTuple], path_or_buf: Union[str, Path, TextIO]
    ) -> None:
        """Export production tuples in line-delimited JSON (JSONL) format."""
        if isinstance(path_or_buf, (str, Path)):
            with open(path_or_buf, "w", encoding="utf-8") as f:
                for t in tuples:
                    f.write(json.dumps(t.to_dict()) + "\n")
        else:
            for t in tuples:
                path_or_buf.write(json.dumps(t.to_dict()) + "\n")

    def compute_production_frequencies(
        self, tuples: List[ProductionTuple]
    ) -> Dict[str, Dict[str, Any]]:
        """Compute marginal production frequencies and maximum-likelihood probabilities per nonterminal."""
        freqs: Dict[str, Dict[str, Any]] = {}
        for t in tuples:
            nt = t.nonterminal
            if nt not in freqs:
                freqs[nt] = {"counts": {}, "probabilities": {}, "total": 0}
            counts = freqs[nt]["counts"]
            counts[t.production_id] = counts.get(t.production_id, 0) + 1
            freqs[nt]["total"] += 1

        for nt, data in freqs.items():
            total = data["total"]
            if total > 0:
                data["probabilities"] = {
                    pid: count / total for pid, count in data["counts"].items()
                }
        return freqs

    def compute_context_cooccurrence(
        self, tuples: List[ProductionTuple]
    ) -> Dict[str, Dict[str, int]]:
        """Compute co-occurrence statistics between parent/sibling contexts and productions."""
        cooc: Dict[str, Dict[str, int]] = {}
        for t in tuples:
            parent = t.parent_context[0] if t.parent_context else "Root"
            ctx_key = f"parent:{parent}"
            if ctx_key not in cooc:
                cooc[ctx_key] = {}
            cooc[ctx_key][t.production_id] = cooc[ctx_key].get(t.production_id, 0) + 1

            if t.left_sibling_context is not None:
                sibling_key = f"parent:{parent}+sibling:{t.left_sibling_context}"
                if sibling_key not in cooc:
                    cooc[sibling_key] = {}
                cooc[sibling_key][t.production_id] = (
                    cooc[sibling_key].get(t.production_id, 0) + 1
                )
        return cooc
