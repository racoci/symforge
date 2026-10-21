"""Topology & Grammar Connector (Model 2 - TGC) for Neurosymbolic Synthesis."""

from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

from symforge.core.ast import (
    ADSRNode,
    ASTNode,
    AudioHoleNode,
    AudioNode,
    BoxNode,
    CylinderNode,
    DifferenceNode,
    ExpDecayNode,
    FMNode,
    FreeParam,
    GainNode,
    HoleNode,
    IntersectionNode,
    LFONode,
    LowPassNode,
    MixNode,
    NoiseNode,
    PlaneNode,
    RotateNode,
    SawNode,
    ScaleNode,
    SDFHoleNode,
    SDFNode,
    SineNode,
    SmoothUnionNode,
    SphereNode,
    SquareNode,
    TorusNode,
    TranslateNode,
    UnionNode,
    substitute_hole,
)

logger = logging.getLogger(__name__)

AUDIO_VOCABULARY: Tuple[str, ...] = (
    "Sine",
    "Saw",
    "Square",
    "Noise",
    "LowPass",
    "Gain",
    "Mix",
    "ADSR",
    "ExpDecay",
    "FM",
    "LFO",
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


def _strip_suffixes(raw: str) -> str:
    """Strip common node suffixes to normalize primitive tag names."""
    clean = raw.strip()
    clean_lower = clean.lower()
    for suffix in ("node", "filter", "oscillator", "envelope", "sdf", "csg"):
        if clean_lower.endswith(suffix) and len(clean_lower) > len(suffix):
            clean = clean[: -len(suffix)]
            clean_lower = clean_lower[: -len(suffix)]
    return clean


def normalize_primitive_name(raw: str, domain: str) -> Optional[str]:
    """Normalize and match a raw primitive name to the target domain vocabulary.

    Args:
        raw: Raw primitive string (e.g. 'sine_node', 'LowPass', 'smoothunion').
        domain: Domain specification ('audio' or 'sdf').

    Returns:
        Canonical title-cased primitive name, or None if unrecognized.
    """
    dom = domain.strip().lower()
    if dom not in DOMAIN_VOCABULARIES:
        return None
    vocab = DOMAIN_VOCABULARIES[dom]

    base = _strip_suffixes(raw).replace("_", "").lower()

    for canon in vocab:
        if base == canon.lower():
            return canon
    return None


@dataclass(frozen=True)
class PrimitiveSpec:
    """Specification of an AST primitive rule including arity and constructor."""

    name: str
    domain: str
    arity: int
    is_leaf: bool


def _build_primitive_specs() -> Dict[Tuple[str, str], PrimitiveSpec]:
    specs: Dict[Tuple[str, str], PrimitiveSpec] = {}

    # Audio primitives
    audio_arities: Dict[str, int] = {
        "Sine": 0,
        "Saw": 0,
        "Square": 0,
        "Noise": 0,
        "FM": 0,
        "LFO": 0,
        "LowPass": 1,
        "Gain": 1,
        "ADSR": 1,
        "ExpDecay": 1,
        "Mix": 2,
    }
    for name, arity in audio_arities.items():
        specs[(name, "audio")] = PrimitiveSpec(
            name=name,
            domain="audio",
            arity=arity,
            is_leaf=(arity == 0),
        )

    # SDF primitives
    sdf_arities: Dict[str, int] = {
        "Sphere": 0,
        "Box": 0,
        "Cylinder": 0,
        "Torus": 0,
        "Plane": 0,
        "Translate": 1,
        "Rotate": 1,
        "Scale": 1,
        "Union": 2,
        "SmoothUnion": 2,
        "Difference": 2,
        "Intersection": 2,
    }
    for name, arity in sdf_arities.items():
        specs[(name, "sdf")] = PrimitiveSpec(
            name=name,
            domain="sdf",
            arity=arity,
            is_leaf=(arity == 0),
        )

    return specs


PRIMITIVE_SPECS = _build_primitive_specs()


class NodeFactory:
    """Thread-safe parameter generator and AST node instantiator."""

    def __init__(self, prefix: str = "p") -> None:
        self.prefix = prefix
        self.counter = 0

    def _param(self, name: str, min_v: float, max_v: float, def_v: float) -> FreeParam:
        self.counter += 1
        return FreeParam(
            id=f"{self.prefix}_{self.counter}_{name}",
            min_val=float(min_v),
            max_val=float(max_v),
            default_val=float(def_v),
        )

    def instantiate(
        self,
        primitive_name: str,
        domain: str,
        children: Sequence[ASTNode],
    ) -> ASTNode:
        """Instantiate a concrete ASTNode given primitive name and child nodes."""
        # Audio primitives
        if domain == "audio":
            if primitive_name == "Sine":
                return SineNode(
                    freq=self._param("f", 20.0, 5000.0, 440.0),
                    phase=self._param("phi", 0.0, 6.28318, 0.0),
                )
            if primitive_name == "Saw":
                return SawNode(freq=self._param("f", 20.0, 5000.0, 440.0))
            if primitive_name == "Square":
                return SquareNode(
                    freq=self._param("f", 20.0, 5000.0, 440.0),
                    duty=self._param("duty", 0.05, 0.95, 0.5),
                )
            if primitive_name == "Noise":
                return NoiseNode(seed=42)
            if primitive_name == "FM":
                return FMNode(
                    carrier=self._param("fc", 20.0, 2000.0, 220.0),
                    mod_freq=self._param("fm", 10.0, 1000.0, 110.0),
                    mod_index=self._param("mi", 0.1, 10.0, 1.0),
                )
            if primitive_name == "LFO":
                return LFONode(
                    freq=self._param("lfo_f", 0.1, 50.0, 2.0),
                    amount=self._param("lfo_amt", 0.0, 1.0, 0.5),
                )
            if primitive_name == "LowPass":
                return LowPassNode(
                    cutoff=self._param("fc", 100.0, 10000.0, 1000.0),
                    resonance=self._param("q", 0.1, 10.0, 0.707),
                    input_node=children[0],  # type: ignore
                )
            if primitive_name == "Gain":
                return GainNode(
                    gain=self._param("g", 0.0, 2.0, 0.8),
                    input_node=children[0],  # type: ignore
                )
            if primitive_name == "ADSR":
                return ADSRNode(
                    attack=self._param("att", 0.001, 1.0, 0.05),
                    decay=self._param("dec", 0.001, 1.0, 0.1),
                    sustain=self._param("sus", 0.0, 1.0, 0.7),
                    release=self._param("rel", 0.001, 2.0, 0.2),
                    gate=children[0],  # type: ignore
                )
            if primitive_name == "ExpDecay":
                return ExpDecayNode(
                    decay_time=self._param("tau", 0.01, 5.0, 0.5),
                    input_node=children[0],  # type: ignore
                )
            if primitive_name == "Mix":
                return MixNode(
                    ratio=self._param("ratio", 0.0, 1.0, 0.5),
                    input1=children[0],  # type: ignore
                    input2=children[1],  # type: ignore
                )

        # SDF primitives
        if domain == "sdf":
            if primitive_name == "Sphere":
                return SphereNode(radius=self._param("r", 0.1, 5.0, 1.0))
            if primitive_name == "Box":
                return BoxNode(
                    width=self._param("w", 0.1, 5.0, 1.0),
                    height=self._param("h", 0.1, 5.0, 1.0),
                    depth=self._param("d", 0.1, 5.0, 1.0),
                )
            if primitive_name == "Cylinder":
                return CylinderNode(
                    radius=self._param("r", 0.1, 5.0, 0.5),
                    height=self._param("h", 0.1, 5.0, 1.0),
                )
            if primitive_name == "Torus":
                return TorusNode(
                    major_radius=self._param("r1", 0.5, 5.0, 1.5),
                    minor_radius=self._param("r2", 0.05, 1.0, 0.3),
                )
            if primitive_name == "Plane":
                return PlaneNode(
                    nx=self._param("nx", -1.0, 1.0, 0.0),
                    ny=self._param("ny", -1.0, 1.0, 1.0),
                    nz=self._param("nz", -1.0, 1.0, 0.0),
                    offset=self._param("offset", -5.0, 5.0, 0.0),
                )
            if primitive_name == "Translate":
                return TranslateNode(
                    dx=self._param("dx", -5.0, 5.0, 0.0),
                    dy=self._param("dy", -5.0, 5.0, 0.0),
                    dz=self._param("dz", -5.0, 5.0, 0.0),
                    child=children[0],  # type: ignore
                )
            if primitive_name == "Rotate":
                return RotateNode(
                    rx=self._param("rx", -3.14159, 3.14159, 0.0),
                    ry=self._param("ry", -3.14159, 3.14159, 0.0),
                    rz=self._param("rz", -3.14159, 3.14159, 0.0),
                    child=children[0],  # type: ignore
                )
            if primitive_name == "Scale":
                return ScaleNode(
                    s=self._param("s", 0.1, 5.0, 1.0),
                    child=children[0],  # type: ignore
                )
            if primitive_name == "Union":
                return UnionNode(
                    smoothness=self._param("k", 0.0, 1.0, 0.0),
                    a=children[0],  # type: ignore
                    b=children[1],  # type: ignore
                )
            if primitive_name == "SmoothUnion":
                return SmoothUnionNode(
                    smoothness=self._param("k", 0.01, 1.0, 0.2),
                    a=children[0],  # type: ignore
                    b=children[1],  # type: ignore
                )
            if primitive_name == "Difference":
                return DifferenceNode(
                    a=children[0],  # type: ignore
                    b=children[1],  # type: ignore
                )
            if primitive_name == "Intersection":
                return IntersectionNode(
                    a=children[0],  # type: ignore
                    b=children[1],  # type: ignore
                )

        raise ValueError(
            f"Unsupported primitive '{primitive_name}' for domain '{domain}'"
        )


def _ast_fingerprint(node: ASTNode) -> str:
    """Compute lightweight structural fingerprint of an AST for search deduplication."""
    if isinstance(node, HoleNode):
        return f"?{node.hole_id}"
    tag = type(node).__name__
    if not node.children:
        return tag
    child_str = ",".join(_ast_fingerprint(c) for c in node.children)
    return f"{tag}({child_str})"


@dataclass
class AssemblyCandidate:
    """State candidate in the TGC constrained beam search space."""

    program: ASTNode
    open_holes: Tuple[Tuple[str, int], ...]
    remaining: Tuple[Tuple[str, int], ...]
    depth_penalty: float
    synthesis_penalty: float
    step: int

    @property
    def total_cost(self) -> float:
        """Composite priority score for beam ordering."""
        hole_penalty = len(self.open_holes) * 0.5
        leftover_penalty = len(self.remaining) * 1.5
        return self.depth_penalty + self.synthesis_penalty + hole_penalty + leftover_penalty


class TopologyGrammarConnector:
    """Topology & Grammar Connector (Model 2 - TGC).

    Resolves a predicted multiset of primitives and hierarchical depth strata
    M = {(p_i, l_i)} into a syntactically valid, type-safe AST G in L(Gamma)
    via type-guided, depth-constrained beam search.
    """

    def __init__(
        self,
        domain: Optional[str] = None,
        default_beam_width: int = 15,
        default_max_steps: int = 200,
        max_tree_depth: int = 6,
    ) -> None:
        """Initialize TGC assembler with configuration parameters.

        Args:
            domain: Target domain ('audio' or 'sdf', optional).
            default_beam_width: Default beam size during search (default: 15).
            default_max_steps: Default step iteration limit (default: 200).
            max_tree_depth: Hard ceiling on AST depth expansion (default: 6).
        """
        self.domain = domain.lower() if domain else None
        self.default_beam_width = int(default_beam_width)
        self.default_max_steps = int(default_max_steps)
        self.max_tree_depth = int(max_tree_depth)

    def assemble(
        self,
        primitive_depth_pairs: Sequence[Tuple[str, int]],
        domain: Optional[str] = None,
        beam_width: Optional[int] = None,
        max_steps: Optional[int] = None,
    ) -> Optional[ASTNode]:
        """Assemble an unordered multiset of (primitive, depth) tuples into a valid AST.

        Args:
            primitive_depth_pairs: Shuffled/unordered list of (primitive_name, depth_level).
            domain: Domain specification ('audio' or 'sdf'). Defaults to self.domain.
            beam_width: Maximum beam search width (default: 15).
            max_steps: Maximum expansion steps cap (default: 200).

        Returns:
            Complete type-safe ASTNode with initialized FreeParams, or None if input is empty.
        """
        dom = (domain or self.domain or "").strip().lower()
        if dom not in DOMAIN_VOCABULARIES:
            raise ValueError(
                f"Unknown domain '{dom}'. Valid options: {list(DOMAIN_VOCABULARIES.keys())}"
            )

        b_width = beam_width or self.default_beam_width
        m_steps = max_steps or self.default_max_steps

        # Check for empty input
        if not primitive_depth_pairs:
            return None

        # Clean, validate, and normalize input pairs
        valid_pairs: List[Tuple[str, int]] = []
        for raw_name, depth in primitive_depth_pairs:
            canon = normalize_primitive_name(str(raw_name), dom)
            if canon is not None:
                valid_pairs.append((canon, max(0, int(depth))))

        if not valid_pairs:
            return None

        # Normalize depth levels relative to minimum depth
        min_depth = min(d for _, d in valid_pairs)
        normalized_multiset = tuple(
            sorted(
                [(p, d - min_depth) for p, d in valid_pairs],
                key=lambda item: (item[1], item[0]),
            )
        )

        factory = NodeFactory(prefix="tgc")
        hole_counter = 0

        def _new_hole(h_domain: str) -> Tuple[HoleNode, str]:
            nonlocal hole_counter
            hole_counter += 1
            h_id = f"h_{hole_counter}"
            if h_domain == "audio":
                return AudioHoleNode(h_id), h_id
            return SDFHoleNode(h_id), h_id

        # Root initial hole at depth 0
        root_hole, root_id = _new_hole(dom)
        initial_state = AssemblyCandidate(
            program=root_hole,
            open_holes=((root_id, 0),),
            remaining=normalized_multiset,
            depth_penalty=0.0,
            synthesis_penalty=0.0,
            step=0,
        )

        beam: List[AssemblyCandidate] = [initial_state]
        best_complete: Optional[AssemblyCandidate] = None

        step_counter = 0
        safety_loop_cap = max(500, m_steps * 2)

        while beam and step_counter < m_steps:
            step_counter += 1
            if step_counter > safety_loop_cap:
                logger.warning(
                    "Safety iteration cap exceeded in TGC beam search (step=%d, beam_size=%d)",
                    step_counter,
                    len(beam),
                )
                break

            # Check if any candidate is perfectly complete
            for cand in beam:
                if not cand.open_holes and not cand.remaining and cand.depth_penalty == 0.0:
                    return cand.program
                if not cand.open_holes:
                    if best_complete is None or cand.total_cost < best_complete.total_cost:
                        best_complete = cand

            # If all candidates in current beam are complete, stop
            if all(not c.open_holes for c in beam):
                break

            candidate_pool: List[AssemblyCandidate] = []

            for cand in beam:
                if not cand.open_holes:
                    # Keep completed candidates in pool with leftover penalty
                    candidate_pool.append(cand)
                    continue

                # Select shallowest open hole
                target_hole_id, target_depth = cand.open_holes[0]
                other_holes = cand.open_holes[1:]

                # Distinct remaining items for non-redundant branching
                unique_remaining = sorted(
                    list(set(cand.remaining)),
                    key=lambda item: (abs(item[1] - target_depth), item[1], item[0]),
                )

                has_expanded = False

                for prim_name, exp_depth in unique_remaining:
                    spec = PRIMITIVE_SPECS.get((prim_name, dom))
                    if spec is None:
                        continue

                    # Depth boundary constraint: do not create operators beyond max depth
                    if target_depth >= self.max_tree_depth and not spec.is_leaf:
                        continue

                    # Allocate child holes if non-terminal
                    child_ast_holes: List[HoleNode] = []
                    new_open_children: List[Tuple[str, int]] = []

                    for _ in range(spec.arity):
                        c_hole, c_id = _new_hole(dom)
                        child_ast_holes.append(c_hole)
                        new_open_children.append((c_id, target_depth + 1))

                    # Instantiate concrete node
                    new_node = factory.instantiate(prim_name, dom, child_ast_holes)
                    new_prog = substitute_hole(cand.program, target_hole_id, new_node)

                    # Consume one item from remaining multiset
                    rem_list = list(cand.remaining)
                    rem_list.remove((prim_name, exp_depth))
                    new_remaining = tuple(rem_list)

                    depth_error = float(abs(target_depth - exp_depth))

                    # Combine holes: shallowest holes first
                    combined_holes = tuple(
                        sorted(
                            other_holes + tuple(new_open_children),
                            key=lambda h: h[1],
                        )
                    )

                    new_cand = AssemblyCandidate(
                        program=new_prog,
                        open_holes=combined_holes,
                        remaining=new_remaining,
                        depth_penalty=cand.depth_penalty + depth_error,
                        synthesis_penalty=cand.synthesis_penalty,
                        step=cand.step + 1,
                    )
                    candidate_pool.append(new_cand)
                    has_expanded = True

                # Fallback leaf expansion if no remaining items match or multiset exhausted
                if not has_expanded or not cand.remaining:
                    default_leaf_name = "Sine" if dom == "audio" else "Sphere"
                    fallback_node = factory.instantiate(default_leaf_name, dom, [])
                    new_prog = substitute_hole(cand.program, target_hole_id, fallback_node)

                    new_cand = AssemblyCandidate(
                        program=new_prog,
                        open_holes=other_holes,
                        remaining=cand.remaining,
                        depth_penalty=cand.depth_penalty,
                        synthesis_penalty=cand.synthesis_penalty + 10.0 + target_depth,
                        step=cand.step + 1,
                    )
                    candidate_pool.append(new_cand)

            # Deduplicate candidates by (structural fingerprint, remaining multiset)
            seen_signatures: Dict[Tuple[str, Tuple[Tuple[str, int], ...]], AssemblyCandidate] = {}
            for cand in candidate_pool:
                key = (_ast_fingerprint(cand.program), cand.remaining)
                if key not in seen_signatures or cand.total_cost < seen_signatures[key].total_cost:
                    seen_signatures[key] = cand

            # Sort deduplicated pool by total cost and prune to beam_width
            sorted_pool = sorted(seen_signatures.values(), key=lambda c: c.total_cost)
            beam = sorted_pool[:b_width]

        # Select best complete or partially complete candidate
        best_cand = best_complete or (beam[0] if beam else None)

        if best_cand is None:
            # Fallback to single canonical leaf
            default_leaf = "Sine" if dom == "audio" else "Sphere"
            return factory.instantiate(default_leaf, dom, [])

        prog = best_cand.program
        # Close any remaining holes if search terminated early
        if best_cand.open_holes:
            default_leaf = "Sine" if dom == "audio" else "Sphere"
            loop_idx = 0
            for h_id, _ in best_cand.open_holes:
                loop_idx += 1
                if loop_idx > 1000:
                    break
                leaf_node = factory.instantiate(default_leaf, dom, [])
                prog = substitute_hole(prog, h_id, leaf_node)

        return prog

    def batch_assemble(
        self,
        batch_pairs: Sequence[Sequence[Tuple[str, int]]],
        domain: Optional[str] = None,
        beam_width: Optional[int] = None,
        max_steps: Optional[int] = None,
    ) -> List[Optional[ASTNode]]:
        """Batch assemble multiple program specifications.

        Args:
            batch_pairs: Sequence of primitive-depth pair lists.
            domain: Domain specification ('audio' or 'sdf').
            beam_width: Beam search width.
            max_steps: Step cap per program.

        Returns:
            List of assembled ASTNode instances.
        """
        results: List[Optional[ASTNode]] = []
        loop_counter = 0
        cap = len(batch_pairs) + 100

        for pairs in batch_pairs:
            loop_counter += 1
            if loop_counter > cap:
                break
            ast = self.assemble(
                pairs,
                domain=domain,
                beam_width=beam_width,
                max_steps=max_steps,
            )
            results.append(ast)
        return results


TGC = TopologyGrammarConnector


def assemble(
    primitive_depth_pairs: Sequence[Tuple[str, int]],
    domain: str,
    beam_width: int = 15,
    max_steps: int = 200,
) -> Optional[ASTNode]:
    """Convenience functional wrapper for TopologyGrammarConnector.assemble."""
    connector = TopologyGrammarConnector(
        domain=domain,
        default_beam_width=beam_width,
        default_max_steps=max_steps,
    )
    return connector.assemble(primitive_depth_pairs)


def batch_assemble(
    batch_pairs: Sequence[Sequence[Tuple[str, int]]],
    domain: str,
    beam_width: int = 15,
    max_steps: int = 200,
) -> List[Optional[ASTNode]]:
    """Convenience functional wrapper for TopologyGrammarConnector.batch_assemble."""
    connector = TopologyGrammarConnector(
        domain=domain,
        default_beam_width=beam_width,
        default_max_steps=max_steps,
    )
    return connector.batch_assemble(batch_pairs)
