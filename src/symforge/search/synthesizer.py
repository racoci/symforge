"""Type-guided beam and A* program synthesizer with typed hole expansion."""

from __future__ import annotations
import heapq
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple
import torch

from symforge.core.ast import (
    ASTNode,
    AudioNode,
    SDFNode,
    HoleNode,
    AudioHoleNode,
    SDFHoleNode,
    SineNode,
    SawNode,
    SquareNode,
    NoiseNode,
    ADSRNode,
    LowPassNode,
    GainNode,
    MixNode,
    FMNode,
    SphereNode,
    BoxNode,
    CylinderNode,
    TorusNode,
    TranslateNode,
    RotateNode,
    ScaleNode,
    UnionNode,
    IntersectionNode,
    DifferenceNode,
    FreeParam,
    find_holes,
    has_holes,
    substitute_hole,
    update_ast_params,
)
from symforge.core.transpiler import transpile_ast_to_module
from symforge.eval.cost import HardwareCostModel, compute_program_cost


@dataclass
class SearchConfig:
    """Configuration parameters for beam and A* program synthesis."""

    beam_width: int = 15
    max_depth: int = 5
    max_cost_budget: float = 60.0
    max_expansions: int = 200
    cost_weight: float = 0.01
    tolerance: float = 1e-3
    optimize_parameters: bool = False
    param_opt_steps: int = 15
    param_opt_lr: float = 0.05


@dataclass
class SynthesisCandidate:
    """State candidate in the program synthesis search space."""

    program: ASTNode
    cost: float
    score: float
    priority: float
    depth: int
    is_complete: bool

    def __lt__(self, other: SynthesisCandidate) -> bool:
        return self.priority < other.priority


@dataclass
class SynthesisResult:
    """Final outcome of the program synthesis run."""

    success: bool
    best_program: Optional[ASTNode]
    best_score: float
    best_cost: float
    total_expansions: int
    iterations: int
    all_candidates_explored: int
    elapsed_seconds: float = 0.0


class ProgramScorer(Protocol):
    """Protocol for scoring candidate programs against target conditions."""

    def score(self, program: ASTNode) -> float:
        """Return non-negative loss/distance (0.0 = perfect match)."""
        ...


logger = logging.getLogger(__name__)


class ReferenceASTScorer:
    """Evaluates candidates against a target reference AST via forward execution."""

    def __init__(
        self,
        target: ASTNode,
        domain: str = "audio",
        sample_count: int = 100,
    ):
        self.target = target
        self.domain = domain
        self.target_module = transpile_ast_to_module(target)

        if domain == "audio":
            self.eval_inputs = torch.linspace(0.0, 0.02, sample_count)
        else:
            torch.manual_seed(42)
            self.eval_inputs = torch.randn(sample_count, 3)

        with torch.no_grad():
            self.target_output = self.target_module(self.eval_inputs)

    def score(self, program: ASTNode) -> float:
        if has_holes(program):
            # Admissible optimistic heuristic: penalty proportional to remaining holes
            holes = find_holes(program)
            return 1.0 + 0.5 * len(holes)

        try:
            cand_module = transpile_ast_to_module(program)
            with torch.no_grad():
                cand_output = cand_module(self.eval_inputs)
                mse = torch.nn.functional.mse_loss(cand_output, self.target_output).item()
                return float(mse)
        except Exception as exc:
            logger.warning(
                "Evaluation failed for candidate %s (size=%d, depth=%d) with exception: %s",
                type(program).__name__,
                program.size,
                program.tree_depth,
                exc,
            )
            return 1e6


class CustomLossScorer:
    """Allows user-defined evaluation function for candidate programs."""

    def __init__(self, scoring_fn: Callable[[ASTNode], float]):
        self.scoring_fn = scoring_fn

    def score(self, program: ASTNode) -> float:
        return float(self.scoring_fn(program))


class TypedGrammarExpander:
    """Generates type-safe production rule expansions for placeholder holes."""

    def __init__(self):
        self.hole_counter = 0

    def _next_hole_id(self, prefix: str = "h") -> str:
        self.hole_counter += 1
        return f"{prefix}_{self.hole_counter}"

    def get_expansions(self, hole: HoleNode) -> List[ASTNode]:
        """Generate all single-step type-compatible replacements for a hole."""
        if isinstance(hole, AudioHoleNode):
            return self._audio_expansions()
        elif isinstance(hole, SDFHoleNode):
            return self._sdf_expansions()
        else:
            raise TypeError(f"Unknown or untyped hole: {type(hole).__name__}")

    def _audio_expansions(self) -> List[ASTNode]:
        h1 = self._next_hole_id("audio")
        h2 = self._next_hole_id("audio")
        return [
            # Terminal primitives
            SineNode(freq=FreeParam("freq", 20.0, 5000.0, 440.0)),
            SawNode(freq=FreeParam("freq", 20.0, 5000.0, 440.0)),
            SquareNode(freq=FreeParam("freq", 20.0, 5000.0, 440.0)),
            NoiseNode(seed=42),
            FMNode(
                carrier=FreeParam("fc", 20.0, 2000.0, 220.0),
                mod_freq=FreeParam("fm", 10.0, 1000.0, 110.0),
                mod_index=FreeParam("mi", 0.1, 10.0, 1.0),
            ),
            # Non-terminal combinators with typed child holes
            LowPassNode(
                cutoff=FreeParam("cutoff", 100.0, 10000.0, 1000.0),
                resonance=FreeParam("res", 0.1, 10.0, 0.707),
                input_node=AudioHoleNode(h1),
            ),
            GainNode(
                gain=FreeParam("gain", 0.0, 2.0, 1.0),
                input_node=AudioHoleNode(h1),
            ),
            MixNode(
                ratio=FreeParam("ratio", 0.0, 1.0, 0.5),
                input1=AudioHoleNode(h1),
                input2=AudioHoleNode(h2),
            ),
        ]

    def _sdf_expansions(self) -> List[ASTNode]:
        h1 = self._next_hole_id("sdf")
        h2 = self._next_hole_id("sdf")
        return [
            # Terminal primitives
            SphereNode(radius=FreeParam("r", 0.1, 5.0, 1.0)),
            BoxNode(
                width=FreeParam("w", 0.1, 5.0, 1.0),
                height=FreeParam("h", 0.1, 5.0, 1.0),
                depth=FreeParam("d", 0.1, 5.0, 1.0),
            ),
            CylinderNode(
                radius=FreeParam("r", 0.1, 5.0, 0.5),
                height=FreeParam("h", 0.1, 5.0, 1.0),
            ),
            TorusNode(
                major_radius=FreeParam("r1", 0.5, 5.0, 1.5),
                minor_radius=FreeParam("r2", 0.1, 1.0, 0.3),
            ),
            # Non-terminal transforms & booleans with typed child holes
            TranslateNode(
                dx=FreeParam("dx", -5.0, 5.0, 1.0),
                dy=FreeParam("dy", -5.0, 5.0, 0.0),
                dz=FreeParam("dz", -5.0, 5.0, 0.0),
                child=SDFHoleNode(h1),
            ),
            RotateNode(
                rx=FreeParam("rx", 0.0, 6.28, 0.0),
                ry=FreeParam("ry", 0.0, 6.28, 0.0),
                rz=FreeParam("rz", 0.0, 6.28, 0.0),
                child=SDFHoleNode(h1),
            ),
            ScaleNode(
                s=FreeParam("s", 0.1, 5.0, 2.0),
                child=SDFHoleNode(h1),
            ),
            UnionNode(
                smoothness=FreeParam("k", 0.0, 1.0, 0.1),
                a=SDFHoleNode(h1),
                b=SDFHoleNode(h2),
            ),
            IntersectionNode(
                a=SDFHoleNode(h1),
                b=SDFHoleNode(h2),
            ),
            DifferenceNode(
                a=SDFHoleNode(h1),
                b=SDFHoleNode(h2),
            ),
        ]


class TypeGuidedBeamSynthesizer:
    """Type-guided beam search synthesizer with cost-budget pruning."""

    def __init__(
        self,
        cost_model: Optional[HardwareCostModel] = None,
        expander: Optional[TypedGrammarExpander] = None,
    ):
        self.cost_model = cost_model or HardwareCostModel()
        self.expander = expander or TypedGrammarExpander()

    def synthesize(
        self,
        initial_hole: HoleNode,
        scorer: ProgramScorer,
        config: Optional[SearchConfig] = None,
    ) -> SynthesisResult:
        start_time = time.time()
        cfg = config or SearchConfig()

        initial_cost = self.cost_model.compute_cost(initial_hole)
        initial_score = scorer.score(initial_hole)
        initial_priority = initial_score + cfg.cost_weight * initial_cost

        beam: List[SynthesisCandidate] = [
            SynthesisCandidate(
                program=initial_hole,
                cost=initial_cost,
                score=initial_score,
                priority=initial_priority,
                depth=1,
                is_complete=False,
            )
        ]

        best_candidate = beam[0]
        total_expansions = 0
        iterations = 0
        all_candidates_explored = 1

        # Hard bounded iteration loop
        while beam and total_expansions < cfg.max_expansions:
            iterations += 1

            # Check if any candidate satisfies target tolerance
            for cand in beam:
                if cand.is_complete and cand.score <= cfg.tolerance:
                    return SynthesisResult(
                        success=True,
                        best_program=cand.program,
                        best_score=cand.score,
                        best_cost=cand.cost,
                        total_expansions=total_expansions,
                        iterations=iterations,
                        all_candidates_explored=all_candidates_explored,
                        elapsed_seconds=time.time() - start_time,
                    )
                if cand.is_complete and (best_candidate is None or cand.score < best_candidate.score):
                    best_candidate = cand

            next_candidates: List[SynthesisCandidate] = []

            for cand in beam:
                if cand.is_complete:
                    next_candidates.append(cand)
                    continue

                holes = find_holes(cand.program)
                if not holes:
                    cand.is_complete = True
                    next_candidates.append(cand)
                    continue

                target_hole = holes[0]
                expansions = self.expander.get_expansions(target_hole)

                for exp in expansions:
                    total_expansions += 1
                    all_candidates_explored += 1

                    new_prog = substitute_hole(cand.program, target_hole.hole_id, exp)
                    new_depth = new_prog.tree_depth
                    if new_depth > cfg.max_depth:
                        continue

                    cost = self.cost_model.compute_cost(new_prog)
                    if cost > cfg.max_cost_budget:
                        continue  # Cost budget pruning

                    new_has_holes = has_holes(new_prog)
                    score = scorer.score(new_prog)
                    priority = score + cfg.cost_weight * cost

                    new_cand = SynthesisCandidate(
                        program=new_prog,
                        cost=cost,
                        score=score,
                        priority=priority,
                        depth=new_depth,
                        is_complete=not new_has_holes,
                    )
                    next_candidates.append(new_cand)

                    if new_cand.is_complete and new_cand.score <= cfg.tolerance:
                        return SynthesisResult(
                            success=True,
                            best_program=new_cand.program,
                            best_score=new_cand.score,
                            best_cost=new_cand.cost,
                            total_expansions=total_expansions,
                            iterations=iterations,
                            all_candidates_explored=all_candidates_explored,
                            elapsed_seconds=time.time() - start_time,
                        )

                    if new_cand.is_complete and (best_candidate is None or new_cand.score < best_candidate.score):
                        best_candidate = new_cand

                    if total_expansions >= cfg.max_expansions:
                        break

                if total_expansions >= cfg.max_expansions:
                    break

            # Prune to beam width
            next_candidates.sort(key=lambda c: c.priority)
            beam = next_candidates[: cfg.beam_width]

        return SynthesisResult(
            success=best_candidate is not None and best_candidate.is_complete and best_candidate.score <= cfg.tolerance,
            best_program=best_candidate.program if best_candidate and best_candidate.is_complete else None,
            best_score=best_candidate.score if best_candidate else 1e6,
            best_cost=best_candidate.cost if best_candidate else 1e6,
            total_expansions=total_expansions,
            iterations=iterations,
            all_candidates_explored=all_candidates_explored,
            elapsed_seconds=time.time() - start_time,
        )


class AStarSynthesizer:
    """Type-guided A* search synthesizer ordered by f = g(cost) + h(loss)."""

    def __init__(
        self,
        cost_model: Optional[HardwareCostModel] = None,
        expander: Optional[TypedGrammarExpander] = None,
    ):
        self.cost_model = cost_model or HardwareCostModel()
        self.expander = expander or TypedGrammarExpander()

    def synthesize(
        self,
        initial_hole: HoleNode,
        scorer: ProgramScorer,
        config: Optional[SearchConfig] = None,
    ) -> SynthesisResult:
        start_time = time.time()
        cfg = config or SearchConfig()

        initial_cost = self.cost_model.compute_cost(initial_hole)
        initial_score = scorer.score(initial_hole)
        initial_priority = initial_score + cfg.cost_weight * initial_cost

        # Min-heap ordered by priority: (priority, tie_breaker_id, candidate)
        heap: List[Tuple[float, int, SynthesisCandidate]] = []
        counter = 0

        initial_cand = SynthesisCandidate(
            program=initial_hole,
            cost=initial_cost,
            score=initial_score,
            priority=initial_priority,
            depth=1,
            is_complete=False,
        )
        heapq.heappush(heap, (initial_priority, counter, initial_cand))

        best_candidate = initial_cand
        total_expansions = 0
        iterations = 0
        all_candidates_explored = 1

        while heap and total_expansions < cfg.max_expansions:
            iterations += 1
            _, _, curr = heapq.heappop(heap)

            if curr.is_complete and curr.score <= cfg.tolerance:
                return SynthesisResult(
                    success=True,
                    best_program=curr.program,
                    best_score=curr.score,
                    best_cost=curr.cost,
                    total_expansions=total_expansions,
                    iterations=iterations,
                    all_candidates_explored=all_candidates_explored,
                    elapsed_seconds=time.time() - start_time,
                )

            if curr.is_complete and curr.score < best_candidate.score:
                best_candidate = curr

            holes = find_holes(curr.program)
            if not holes:
                continue

            target_hole = holes[0]
            expansions = self.expander.get_expansions(target_hole)

            for exp in expansions:
                total_expansions += 1
                all_candidates_explored += 1

                new_prog = substitute_hole(curr.program, target_hole.hole_id, exp)
                new_depth = new_prog.tree_depth
                if new_depth > cfg.max_depth:
                    continue

                cost = self.cost_model.compute_cost(new_prog)
                if cost > cfg.max_cost_budget:
                    continue  # Cost budget pruning

                new_has_holes = has_holes(new_prog)
                score = scorer.score(new_prog)
                priority = score + cfg.cost_weight * cost

                new_cand = SynthesisCandidate(
                    program=new_prog,
                    cost=cost,
                    score=score,
                    priority=priority,
                    depth=new_depth,
                    is_complete=not new_has_holes,
                )

                if new_cand.is_complete and new_cand.score <= cfg.tolerance:
                    return SynthesisResult(
                        success=True,
                        best_program=new_cand.program,
                        best_score=new_cand.score,
                        best_cost=new_cand.cost,
                        total_expansions=total_expansions,
                        iterations=iterations,
                        all_candidates_explored=all_candidates_explored,
                        elapsed_seconds=time.time() - start_time,
                    )

                if new_cand.is_complete and (best_candidate is None or new_cand.score < best_candidate.score):
                    best_candidate = new_cand

                counter += 1
                heapq.heappush(heap, (priority, counter, new_cand))

                if total_expansions >= cfg.max_expansions:
                    break

        return SynthesisResult(
            success=best_candidate is not None and best_candidate.is_complete and best_candidate.score <= cfg.tolerance,
            best_program=best_candidate.program if best_candidate and best_candidate.is_complete else None,
            best_score=best_candidate.score if best_candidate else 1e6,
            best_cost=best_candidate.cost if best_candidate else 1e6,
            total_expansions=total_expansions,
            iterations=iterations,
            all_candidates_explored=all_candidates_explored,
            elapsed_seconds=time.time() - start_time,
        )
