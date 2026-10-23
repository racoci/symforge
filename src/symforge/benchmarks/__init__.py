"""SymForge Benchmark Harness Package."""

from symforge.benchmarks.cross_domain import (
    BenchmarkTask,
    CrossDomainTransferSummary,
    DomainBenchmarkScore,
    format_cross_domain_markdown_report,
    get_all_benchmark_suites,
    get_audio_benchmark_suite,
    get_hardware_benchmark_suite,
    get_sdf_benchmark_suite,
    run_cross_domain_benchmark,
)
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

__all__ = [
    # Cross-domain validation benchmark
    "BenchmarkTask",
    "CrossDomainTransferSummary",
    "DomainBenchmarkScore",
    "format_cross_domain_markdown_report",
    "get_all_benchmark_suites",
    "get_audio_benchmark_suite",
    "get_hardware_benchmark_suite",
    "get_sdf_benchmark_suite",
    "run_cross_domain_benchmark",
    # Curriculum 3-tier benchmark
    "CrossDomainComparisonResult",
    "DomainBenchmarkResult",
    "LevelBenchmarkSummary",
    "Stage1Metrics",
    "Stage2Metrics",
    "Stage3Metrics",
    "TaskEvaluationResult",
    "compare_domains_benchmark",
    "evaluate_single_task",
    "generate_markdown_report",
    "run_curriculum_benchmark",
]
