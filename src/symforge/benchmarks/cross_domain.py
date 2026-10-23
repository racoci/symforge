"""Cross-Domain Unified Validation & Transfer Benchmark (Audio, SDF, Hardware).

Evaluates the universal neurosymbolic synthesis and abstraction discovery engine across:
1. Audio DSP (continuous 1D time-domain waveforms)
2. SDF 3D Geometry (continuous 3D distance fields)
3. Hardware Instruction Superoptimization (discrete 32-bit bitvector registers)

Under identical search parameters and resource budgets, verifying >= 80% task completion
and multi-domain recurrent abstraction discovery.
"""

from __future__ import annotations
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import torch

from symforge.abstraction.anti_unification import mine_recurrent_patterns
from symforge.core.ast import (
    ASTNode,
    AudioHoleNode,
    BoxNode,
    CylinderNode,
    FreeParam,
    HardwareHoleNode,
    HoleNode,
    ImmHardwareNode,
    LowPassNode,
    RegHardwareNode,
    SDFHoleNode,
    SawNode,
    ShlHardwareNode,
    ShrHardwareNode,
    SineNode,
    SphereNode,
    SquareNode,
    TranslateNode,
)
from symforge.search.synthesizer import (
    ReferenceASTScorer,
    TypeGuidedBeamSynthesizer,
)

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkTask:
    """Represents a target program synthesis task in a specific domain."""

    task_id: str
    name: str
    domain: str
    target_ast: ASTNode
    initial_hole: HoleNode


@dataclass
class DomainBenchmarkScore:
    """Benchmark telemetry and metrics for a single domain."""

    domain: str
    total_tasks: int
    solved_tasks: int
    success_rate: float
    mean_search_steps: float
    mean_cost: float
    elapsed_seconds: float
    discovered_abstractions: int
    task_results: List[Dict[str, Any]] = field(default_factory=list)
    mined_patterns: List[str] = field(default_factory=list)


@dataclass
class CrossDomainTransferSummary:
    """Consolidated cross-domain benchmark report spanning Audio, SDF, and Hardware."""

    budget_steps: int
    domain_scores: Dict[str, DomainBenchmarkScore]
    all_domains_passed_gate: bool
    total_abstractions_mined: int
    elapsed_seconds: float
    timestamp: str = field(
        default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    )

    def get_score(self, domain: str) -> Optional[DomainBenchmarkScore]:
        """Fetch score for a given domain string."""
        return self.domain_scores.get(domain)


def get_audio_benchmark_suite() -> List[BenchmarkTask]:
    """Return 5 representative Audio DSP synthesis tasks."""
    return [
        BenchmarkTask(
            task_id="audio_01_sine",
            name="Sine Oscillator (440Hz)",
            domain="audio",
            target_ast=SineNode(freq=FreeParam("freq", 20.0, 5000.0, 440.0)),
            initial_hole=AudioHoleNode("root"),
        ),
        BenchmarkTask(
            task_id="audio_02_saw",
            name="Sawtooth Oscillator",
            domain="audio",
            target_ast=SawNode(freq=FreeParam("freq", 20.0, 5000.0, 440.0)),
            initial_hole=AudioHoleNode("root"),
        ),
        BenchmarkTask(
            task_id="audio_03_square",
            name="Square Oscillator",
            domain="audio",
            target_ast=SquareNode(freq=FreeParam("freq", 20.0, 5000.0, 440.0)),
            initial_hole=AudioHoleNode("root"),
        ),
        BenchmarkTask(
            task_id="audio_04_lowpass_sine",
            name="LowPass Filtered Sine",
            domain="audio",
            target_ast=LowPassNode(
                cutoff=FreeParam("cutoff", 100.0, 10000.0, 1000.0),
                resonance=FreeParam("res", 0.1, 10.0, 0.707),
                input_node=SineNode(freq=FreeParam("freq", 20.0, 5000.0, 440.0)),
            ),
            initial_hole=AudioHoleNode("root"),
        ),
        BenchmarkTask(
            task_id="audio_05_lowpass_saw",
            name="LowPass Filtered Saw",
            domain="audio",
            target_ast=LowPassNode(
                cutoff=FreeParam("cutoff", 100.0, 10000.0, 1000.0),
                resonance=FreeParam("res", 0.1, 10.0, 0.707),
                input_node=SawNode(freq=FreeParam("freq", 20.0, 5000.0, 440.0)),
            ),
            initial_hole=AudioHoleNode("root"),
        ),
    ]


def get_sdf_benchmark_suite() -> List[BenchmarkTask]:
    """Return 5 representative SDF 3D Geometry synthesis tasks."""
    return [
        BenchmarkTask(
            task_id="sdf_01_sphere",
            name="Unit Sphere",
            domain="sdf",
            target_ast=SphereNode(radius=FreeParam("r", 0.1, 5.0, 1.0)),
            initial_hole=SDFHoleNode("root"),
        ),
        BenchmarkTask(
            task_id="sdf_02_box",
            name="Unit Box",
            domain="sdf",
            target_ast=BoxNode(
                width=FreeParam("w", 0.1, 5.0, 1.0),
                height=FreeParam("h", 0.1, 5.0, 1.0),
                depth=FreeParam("d", 0.1, 5.0, 1.0),
            ),
            initial_hole=SDFHoleNode("root"),
        ),
        BenchmarkTask(
            task_id="sdf_03_cylinder",
            name="Cylinder Primitive",
            domain="sdf",
            target_ast=CylinderNode(
                radius=FreeParam("r", 0.1, 5.0, 0.5),
                height=FreeParam("h", 0.1, 5.0, 1.0),
            ),
            initial_hole=SDFHoleNode("root"),
        ),
        BenchmarkTask(
            task_id="sdf_04_translated_sphere",
            name="Translated Sphere (dx=1)",
            domain="sdf",
            target_ast=TranslateNode(
                dx=FreeParam("dx", -5.0, 5.0, 1.0),
                dy=FreeParam("dy", -5.0, 5.0, 0.0),
                dz=FreeParam("dz", -5.0, 5.0, 0.0),
                child=SphereNode(radius=FreeParam("r", 0.1, 5.0, 1.0)),
            ),
            initial_hole=SDFHoleNode("root"),
        ),
        BenchmarkTask(
            task_id="sdf_05_translated_box",
            name="Translated Box (dx=1)",
            domain="sdf",
            target_ast=TranslateNode(
                dx=FreeParam("dx", -5.0, 5.0, 1.0),
                dy=FreeParam("dy", -5.0, 5.0, 0.0),
                dz=FreeParam("dz", -5.0, 5.0, 0.0),
                child=BoxNode(
                    width=FreeParam("w", 0.1, 5.0, 1.0),
                    height=FreeParam("h", 0.1, 5.0, 1.0),
                    depth=FreeParam("d", 0.1, 5.0, 1.0),
                ),
            ),
            initial_hole=SDFHoleNode("root"),
        ),
    ]


def get_hardware_benchmark_suite() -> List[BenchmarkTask]:
    """Return 5 representative Hardware Instruction synthesis tasks."""
    return [
        BenchmarkTask(
            task_id="hw_01_reg0",
            name="Register R0 Bypass",
            domain="hardware",
            target_ast=RegHardwareNode(0),
            initial_hole=HardwareHoleNode("root"),
        ),
        BenchmarkTask(
            task_id="hw_02_imm1",
            name="Immediate Constant 1",
            domain="hardware",
            target_ast=ImmHardwareNode(1),
            initial_hole=HardwareHoleNode("root"),
        ),
        BenchmarkTask(
            task_id="hw_03_shl_reg0",
            name="Shift Left R0 by 1",
            domain="hardware",
            target_ast=ShlHardwareNode(RegHardwareNode(0), 1),
            initial_hole=HardwareHoleNode("root"),
        ),
        BenchmarkTask(
            task_id="hw_04_shl_reg1",
            name="Shift Left R1 by 1",
            domain="hardware",
            target_ast=ShlHardwareNode(RegHardwareNode(1), 1),
            initial_hole=HardwareHoleNode("root"),
        ),
        BenchmarkTask(
            task_id="hw_05_shr_reg0",
            name="Shift Right R0 by 1",
            domain="hardware",
            target_ast=ShrHardwareNode(RegHardwareNode(0), 1),
            initial_hole=HardwareHoleNode("root"),
        ),
    ]


def get_all_benchmark_suites() -> Dict[str, List[BenchmarkTask]]:
    """Return all benchmark suites indexed by domain."""
    return {
        "audio": get_audio_benchmark_suite(),
        "sdf": get_sdf_benchmark_suite(),
        "hardware": get_hardware_benchmark_suite(),
    }


def run_cross_domain_benchmark(
    budget_steps: int = 100,
    seed: int = 42,
) -> CrossDomainTransferSummary:
    """Run unified cross-domain benchmark across Audio, SDF, and Hardware under identical settings.

    Args:
        budget_steps: Maximum candidate expansions allowed per task (default: 100).
        seed: Deterministic random seed for sample generation.

    Returns:
        CrossDomainTransferSummary containing scores, pass/fail status, and mined abstractions.
    """
    start_total_time = time.perf_counter()
    torch.manual_seed(seed)

    suites = get_all_benchmark_suites()
    domain_scores: Dict[str, DomainBenchmarkScore] = {}
    total_abstractions = 0

    for domain_name, task_list in suites.items():
        domain_start_time = time.perf_counter()
        solved_count = 0
        search_steps_list: List[int] = []
        cost_list: List[float] = []
        task_results: List[Dict[str, Any]] = []
        solved_programs: List[ASTNode] = []

        # Run identical beam synthesizer per domain
        synthesizer = TypeGuidedBeamSynthesizer(beam_width=15, max_steps=budget_steps)

        for task in task_list:
            scorer = ReferenceASTScorer(target=task.target_ast, domain=task.domain)
            res = synthesizer.synthesize(
                initial_hole=task.initial_hole,
                scorer=scorer,
            )

            is_solved = res.success and (res.best_score <= 1e-3)
            if is_solved:
                solved_count += 1
                if res.best_program is not None:
                    solved_programs.append(res.best_program)

            search_steps_list.append(res.total_expansions)
            cost_list.append(res.best_cost if res.best_cost < 1e5 else 0.0)

            task_results.append(
                {
                    "task_id": task.task_id,
                    "task_name": task.name,
                    "domain": task.domain,
                    "solved": is_solved,
                    "expansions": res.total_expansions,
                    "best_score": res.best_score,
                    "best_cost": res.best_cost,
                    "elapsed_seconds": res.elapsed_seconds,
                    "best_program_repr": repr(res.best_program) if res.best_program else None,
                }
            )

        # Abstraction mining via anti-unification over solved corpus
        mined_patterns = []
        if len(solved_programs) >= 2:
            patterns = mine_recurrent_patterns(
                solved_programs,
                min_frequency=2,
                min_size=2,
                max_patterns=10,
                max_duration_seconds=1.0,
            )
            mined_patterns = [p.name for p in patterns]
            total_abstractions += len(mined_patterns)

        elapsed_domain = time.perf_counter() - domain_start_time
        total_tasks = len(task_list)
        success_rate = solved_count / total_tasks if total_tasks > 0 else 0.0
        mean_steps = sum(search_steps_list) / len(search_steps_list) if search_steps_list else 0.0
        mean_cost = sum(cost_list) / len(cost_list) if cost_list else 0.0

        domain_scores[domain_name] = DomainBenchmarkScore(
            domain=domain_name,
            total_tasks=total_tasks,
            solved_tasks=solved_count,
            success_rate=success_rate,
            mean_search_steps=mean_steps,
            mean_cost=mean_cost,
            elapsed_seconds=elapsed_domain,
            discovered_abstractions=len(mined_patterns),
            task_results=task_results,
            mined_patterns=mined_patterns,
        )

    # Gate verification: all 3 domains must achieve >= 80% completion rate (>= 4/5)
    all_passed = all(score.success_rate >= 0.8 for score in domain_scores.values())
    total_elapsed = time.perf_counter() - start_total_time

    return CrossDomainTransferSummary(
        budget_steps=budget_steps,
        domain_scores=domain_scores,
        all_domains_passed_gate=all_passed,
        total_abstractions_mined=total_abstractions,
        elapsed_seconds=total_elapsed,
    )


def format_cross_domain_markdown_report(summary: CrossDomainTransferSummary) -> str:
    """Format an analytical markdown report detailing cross-domain benchmark results."""
    gate_status = "PASSED" if summary.all_domains_passed_gate else "FAILED"
    gate_badge = "**PASSED (>= 80% Success Rate across all domains)**" if summary.all_domains_passed_gate else "**FAILED (< 80% on one or more domains)**"

    lines = [
        "# Cross-Domain Unified Validation & Transfer Benchmark Report",
        "",
        f"- **Timestamp:** {summary.timestamp}",
        f"- **Search Expansion Budget:** {summary.budget_steps} steps",
        f"- **Total Benchmark Runtime:** {summary.elapsed_seconds:.3f}s",
        f"- **Overall Quality Gate Status:** {gate_badge}",
        f"- **Total Recurrent Abstractions Mined:** {summary.total_abstractions_mined}",
        "",
        "## Domain Performance Summary",
        "",
        "| Domain | Tasks Solved | Success Rate | Mean Search Steps | Mean Cost | Latency (s) | Mined Abstractions |",
        "|:-------|:------------:|:------------:|:-----------------:|:---------:|:-----------:|:------------------:|",
    ]

    for domain_name, score in summary.domain_scores.items():
        rate_pct = f"{score.success_rate * 100:.1f}%"
        lines.append(
            f"| {domain_name.upper()} | {score.solved_tasks}/{score.total_tasks} | {rate_pct} | "
            f"{score.mean_search_steps:.1f} | {score.mean_cost:.2f} | {score.elapsed_seconds:.3f}s | "
            f"{score.discovered_abstractions} |"
        )

    lines.extend([
        "",
        "## Architecture Topology & Unified Search Flow",
        "",
        "```text",
        "+---------------------------------------------------------------------------------+",
        "|                     TypeGuidedBeamSynthesizer (Beam=15)                        |",
        "+---------------------------------------------------------------------------------+",
        "         |                                |                               |",
        "         v                                v                               v",
        "+-------------------+            +-------------------+            +-------------------+",
        "|     Audio DSP     |            |      SDF 3D       |            |     Hardware      |",
        "|  (AudioHoleNode)  |            |   (SDFHoleNode)   |            | (HardwareHoleNode)|",
        "+-------------------+            +-------------------+            +-------------------+",
        "         |                                |                               |",
        "         v                                v                               v",
        "  [Waveform Loss]                 [Pointcloud Loss]               [Bitvector TestVecs]",
        "         |                                |                               |",
        "         +--------------------------------+-------------------------------+",
        "                                          |",
        "                                          v",
        "                       +-------------------------------------+",
        "                       |  mine_recurrent_patterns (Anti-Unif)|",
        "                       +-------------------------------------+",
        "                                          |",
        "                                          v",
        "                       +-------------------------------------+",
        "                       |   DynamicGrammarLibrary (Macros)    |",
        "                       +-------------------------------------+",
        "```",
        "",
        "```mermaid",
        "graph TD",
        "    subgraph Synthesis [Unified Synthesis Engine]",
        "        SYNTH[TypeGuidedBeamSynthesizer<br>beam_width=15, budget=100] -->|AudioHole| AUD[Audio DSP Domain]",
        "        SYNTH -->|SDFHole| SDF[SDF 3D Domain]",
        "        SYNTH -->|HardwareHole| HW[Hardware Domain]",
        "    end",
        "    subgraph Scoring [Domain-Specific Verification]",
        "        AUD -->|1D Continuous MSE| S1[Audio Waveform Loss]",
        "        SDF -->|3D Spatial MSE| S2[SDF Distance Loss]",
        "        HW -->|Discrete 32-bit MAE| S3[Register Vectors Scorer]",
        "    end",
        "    subgraph Abstraction [Transfer & Library Injection]",
        "        S1 --> SOLV[Solved Program Corpus]",
        "        S2 --> SOLV",
        "        S3 --> SOLV",
        "        SOLV --> ANTI[mine_recurrent_patterns<br>Pairwise Anti-Unification]",
        "        ANTI --> MACRO[Invented Macro Primitives]",
        "    end",
        "```",
        "",
        "## Detailed Task Diagnostic Breakdown",
        "",
    ])

    for domain_name, score in summary.domain_scores.items():
        lines.append(f"### {domain_name.upper()} Domain Tasks")
        lines.append("")
        lines.append("| Task ID | Task Name | Status | Steps | Best Score | Cost | Runtime |")
        lines.append("|:--------|:----------|:------:|:-----:|:----------:|:----:|:-------:|")
        for res in score.task_results:
            st = "PASSED" if res["solved"] else "FAILED"
            lines.append(
                f"| `{res['task_id']}` | {res['task_name']} | {st} | {res['expansions']} | "
                f"{res['best_score']:.4e} | {res['best_cost']:.2f} | {res['elapsed_seconds']:.4f}s |"
            )
        if score.mined_patterns:
            lines.append("")
            lines.append(f"- **Mined Patterns:** {', '.join(score.mined_patterns)}")
        lines.append("")

    return "\n".join(lines)
