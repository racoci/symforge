# Implementation Plan: Adaptive Curriculum with Abstraction Discovery and Remote Decoupled Fine-Tuning

**Track ID:** `curriculum_abstraction_remote_training_202505`  
**Spec Reference:** `conductor/tracks/curriculum_abstraction_remote_training_202505/spec.md`  
**Status:** Approved  
**Date:** 2025-05-18  

---

## Plan Structure & Methodology

This implementation plan strictly adheres to the **Conductor Workflow** and **GEMINI.md Software Engineering Principles**:
- **Test-Driven Development (TDD):** Every task specifies exact automated test commands and test file targets before implementation.
- **Optimal Bottleneck Mitigation:** Modules are decoupled via strict abstract protocols (`ComputeProvider`, `TaskGenerator`, `AbstractionEngine`), enabling parallel implementation and testing without inter-task blocking.
- **Mandatory Markdown Documentation:** Every single task links directly to its design specification document or documentation section.
- **Pass-to-Complete Gate:** Tasks are only marked `[x]` after test suites pass 100%.
- **Cost & Risk Mitigation Gating:** Phase 6 is explicitly dedicated to hardening the system against cloud failures, library bloat, and cost runaway.

---

## Phase 1: Local Symbolic Control Plane Foundation

Focus: Core typed AST and DAG representations, vectorized SDF evaluation, cost profiling, and discrete beam/A* search running purely on CPU.

- [ ] Task 1.1: Typed AST & Canonical DAG Representation
  - **Objective:** Implement immutable typed AST nodes (`Hole`, `PrimitiveNode`, `OperatorNode`) and canonical DAG representation with subexpression sharing.
  - **Bottleneck Mitigation:** Pure Python domain objects with zero external network or PyTorch dependencies; acts as the foundational type layer.
  - **Automated Test Definition:** `pytest -v tests/core/test_ast.py tests/core/test_dag.py`
    - Invariants: Symmetrical round-trip serialization `deserialize(serialize(p)) == p`, topological sort correctness, acyclicity guarantee.
  - **Documentation Artifact:** `docs/architecture/typed_ast_dag.md`
  - **Pass-to-Complete Gate:** 100% unit test pass rate with Hypothesis property tests for 1,000 generated valid ASTs.

- [ ] Task 1.2: SDF Vectorized Evaluator & Cost Profiler
  - **Objective:** Build vectorized NumPy evaluation for 3D SDF points and target hardware compiled execution cost model ($C_H$).
  - **Bottleneck Mitigation:** Standalone math module; accepts AST/DAG nodes and coordinates array $(N, 3)$, returning distances $(N,)$.
  - **Automated Test Definition:** `pytest -v tests/eval/test_sdf_evaluator.py tests/eval/test_cost_model.py`
    - Invariants: Analytical distance verification for Sphere, Box, Cylinder, Union, Difference; cost monotonicity ($C(G_1 \cup G_2) > C(G_1)$).
  - **Documentation Artifact:** `docs/architecture/sdf_evaluator_cost.md`
  - **Pass-to-Complete Gate:** All SDF primitives match analytical math within $\epsilon = 10^{-5}$; cost profiler returns non-negative latency estimates.

- [ ] Task 1.3: Beam & A* Guided Synthesizer with Hole Expansion
  - **Objective:** Implement type-safe hole replacement and prioritized beam/A* search exploring partial programs under cost budgets.
  - **Bottleneck Mitigation:** Uses pluggable heuristic scorer interface, allowing synthetic or uniform priors before neural models are attached.
  - **Automated Test Definition:** `pytest -v tests/search/test_synthesizer.py`
    - Invariants: Hole expansion only generates type-compatible rules; beam search prunes branches exceeding cost budget; discovers simple target shapes.
  - **Documentation Artifact:** `docs/architecture/synthesizer_search.md`
  - **Pass-to-Complete Gate:** Synthesizer solves target Sphere and Hollow Cylinder within 200 expansion steps.

- [ ] Task 1.4: Phase 1 Verification & Checkpoint (Refer to workflow.md)

---

## Phase 2: Decoupled Compute Provider & Remote Training Abstraction Layer

Focus: Abstracting heavy training, LoRA fine-tuning, and neural gradient updates away from the local workstation into a pluggable remote compute protocol.

- [ ] Task 2.1: Pluggable ComputeProvider Protocol & Data Contracts
  - **Objective:** Define the asynchronous `ComputeProvider` protocol and Pydantic schemas for `TrainingPayload`, `RemoteJobConfig`, `JobStatusResponse`, and `AdapterArtifact`.
  - **Bottleneck Mitigation:** Pure protocol definition; decouples local synthesis engine from any specific cloud provider SDK.
  - **Automated Test Definition:** `pytest -v tests/compute/test_contracts.py`
    - Invariants: Schema validation for valid and invalid payloads; contract enforcement on dummy implementation.
  - **Documentation Artifact:** `docs/architecture/compute_provider_protocol.md`
  - **Pass-to-Complete Gate:** 100% schema serialization round-trip tests pass.

- [ ] Task 2.2: MockComputeProvider & Local Zero-Cost Testbed
  - **Objective:** Implement `MockComputeProvider` simulating asynchronous training job submission, polling, and artifact generation in < 10ms with zero GPU usage and zero cost.
  - **Bottleneck Mitigation:** Provides a fully functional backend for local test suites and CI/CD pipelines without requiring network or cloud credentials.
  - **Automated Test Definition:** `pytest -v tests/compute/test_mock_provider.py`
    - Invariants: Deterministic mock job IDs; simulated state transitions (`PENDING` -> `RUNNING` -> `COMPLETED`); returns valid dummy LoRA weights.
  - **Documentation Artifact:** `docs/architecture/mock_compute_provider.md`
  - **Pass-to-Complete Gate:** Async submission and artifact fetch cycle completes in < 50ms with valid artifact checksum.

- [ ] Task 2.3: Remote Serverless GPU Providers (Modal & SageMaker) with Async Resilient Loops
  - **Objective:** Implement `ModalComputeProvider` and `SageMakerComputeProvider` using resilient async HTTP/Boto3 clients with hard timeouts, exponential backoff with jitter, and max poll caps.
  - **Bottleneck Mitigation:** Wrapped behind interface; all network calls have strict connection (10s) and read (30s) timeouts; resources severed in `finally` blocks.
  - **Automated Test Definition:** `pytest -v tests/compute/test_remote_providers.py`
    - Invariants: Mocked HTTP/Boto3 responses; test exponential backoff formula; verify hard timeout and cancellation on simulated network freeze.
  - **Documentation Artifact:** `docs/architecture/remote_compute_providers.md`
  - **Pass-to-Complete Gate:** Unit tests pass with simulated network failure scenarios and verified clean client severance.

- [ ] Task 2.4: Specialized LoRA Adapter Registry & Local CPU Inference Loader
  - **Objective:** Create the `LoRAAdapterRegistry` to manage domain-specific adapters (`primitives`, `booleans`, `blends`) and load lightweight PEFT tensors into the local CPU prior evaluator.
  - **Bottleneck Mitigation:** Adapters are modular and small (< 15 MB); base model remains frozen; inference runs on CPU in milliseconds.
  - **Automated Test Definition:** `pytest -v tests/adapters/test_lora_registry.py`
    - Invariants: Adapter registration, version tracking, checksum validation, hot-swapping between adapters during search.
  - **Documentation Artifact:** `docs/architecture/lora_adapter_registry.md`
  - **Pass-to-Complete Gate:** Registry loads and hot-swaps simulated adapter weights without memory accumulation.

- [ ] Task 2.5: Phase 2 Verification & Checkpoint (Refer to workflow.md)

---

## Phase 3: Adaptive Curriculum & Competency Frontier Engine

Focus: Dynamic task generation, tracking learner success across difficulty bands, and harvesting trajectories for experience replay.

- [ ] Task 3.1: Task Generator & Structural Complexity Metric
  - **Objective:** Implement parameterized synthetic task generation (target point clouds, surface points) with measurable complexity metrics (primitive count, nesting depth, symmetry).
  - **Bottleneck Mitigation:** Generates tasks on-demand in memory; no disk I/O bottlenecks.
  - **Automated Test Definition:** `pytest -v tests/curriculum/test_task_generator.py`
    - Invariants: Generated tasks have verifiable ground truth; complexity metric strictly maps to difficulty levels $c \in [1, 10]$.
  - **Documentation Artifact:** `docs/architecture/curriculum_task_generator.md`
  - **Pass-to-Complete Gate:** Generates 100 distinct tasks across 5 difficulty bands with validated SDF points.

- [ ] Task 3.2: Competency Frontier Tracker & ZPD Sampler
  - **Objective:** Implement the Competency Frontier engine tracking moving-average success rate $S(c)$ and sampling tasks from the Zone of Proximal Development ($0.2 \le S(c) \le 0.8$).
  - **Bottleneck Mitigation:** Pure statistics and sampling logic; decoupled from synthesis execution.
  - **Automated Test Definition:** `pytest -v tests/curriculum/test_frontier.py`
    - Invariants: Advances frontier when $S(c) > 0.8$; falls back if regression occurs; avoids sampling from unmasterable bands ($S(c) < 0.2$).
  - **Documentation Artifact:** `docs/architecture/competency_frontier.md`
  - **Pass-to-Complete Gate:** Simulated student agent with expanding capability induces predictable, monotonic frontier progression.

- [ ] Task 3.3: Trajectory Harvester & Prioritized Experience Replay Buffer
  - **Objective:** Collect step-by-step synthesis decisions (`hole_type`, `chosen_rule`, `prior`, `reward`, `cost`) into an in-memory replay buffer with prioritized sampling.
  - **Bottleneck Mitigation:** Bounded buffer capacity ($N \le 10,000$ transitions) to eliminate local memory bloat; serialized on-demand for training dispatch.
  - **Automated Test Definition:** `pytest -v tests/curriculum/test_replay_buffer.py`
    - Invariants: FIFO eviction on capacity overflow; priority weighting matches TD-error/reward; batch extraction preserves schema.
  - **Documentation Artifact:** `docs/architecture/replay_buffer.md`
  - **Pass-to-Complete Gate:** Buffer handles 20,000 rapid pushes, maintains max size $10,000$, and samples valid training batches.

- [ ] Task 3.4: Phase 3 Verification & Checkpoint (Refer to workflow.md)

---

## Phase 4: DreamCoder-Style Abstraction Discovery Engine

Focus: Subgraph anti-unification, Minimum Description Length (MDL) compression evaluation, and dynamic grammar expansion with newly invented macro-primitives.

- [ ] Task 4.1: Subgraph Anti-Unification & Structural Pattern Extractor
  - **Objective:** Implement anti-unification on pairs of AST/DAG programs to identify the Least General Generalization (LGG) and extract shared subtrees with parameterized holes.
  - **Bottleneck Mitigation:** Operates over discovered solution programs ($M \approx 20..50$) rather than entire search spaces, keeping CPU overhead under 2 seconds.
  - **Automated Test Definition:** `pytest -v tests/abstraction/test_anti_unification.py`
    - Invariants: Identifies exact shared structure between `Difference(Cylinder, Cylinder)` and `Difference(Box, Box)`; parameter extraction matches free variables.
  - **Documentation Artifact:** `docs/architecture/anti_unification.md`
  - **Pass-to-Complete Gate:** Correctly discovers candidate macro-primitive from 5 pairs of geometric programs.

- [ ] Task 4.2: Minimum Description Length (MDL) Compression Evaluator
  - **Objective:** Calculate compression utility $\Delta \text{MDL}(\alpha) = \text{Size}(\alpha) - \sum (\text{Size}(p) - \text{Size}(p[\alpha]))$ to filter candidate abstractions.
  - **Bottleneck Mitigation:** Fast discrete counting of AST node types and parameter bits; rejects unhelpful candidates immediately.
  - **Automated Test Definition:** `pytest -v tests/abstraction/test_mdl.py`
    - Invariants: Trivial or single-use abstractions return $\Delta \text{MDL} \ge 0$ (rejected); recurring subgraphs return $\Delta \text{MDL} < 0$ (accepted).
  - **Documentation Artifact:** `docs/architecture/mdl_compression.md`
  - **Pass-to-Complete Gate:** Evaluates 20 candidates, accepting only the top 3 compressing abstractions.

- [ ] Task 4.3: Dynamic Grammar Expander & Library Injection
  - **Objective:** Invert candidate abstractions into first-class DSL productions $\mathcal{R}$, register typed signatures, and update the grammar prior distribution $\mathcal{L}(\Gamma)$.
  - **Bottleneck Mitigation:** Grammar updates are append-only with versioned rollback capability; new productions inherit strict type rules.
  - **Automated Test Definition:** `pytest -v tests/abstraction/test_grammar_expander.py`
    - Invariants: Injected macro can be instantiated as a typed node in subsequent synthesis runs; expanding macro produces identical SDF field.
  - **Documentation Artifact:** `docs/architecture/grammar_expander.md`
  - **Pass-to-Complete Gate:** Synthesizer successfully uses newly invented macro-primitive in a subsequent synthesis task, reducing search steps by $> 40\%$.

- [ ] Task 4.4: Phase 4 Verification & Checkpoint (Refer to workflow.md)

---

## Phase 5: Meta-Controller, Training Regime Orchestrator & CLI

Focus: Orchestrating curriculum, search, abstraction cycles, remote training dispatches, financial cost gating, and CLI reporting.

- [ ] Task 5.1: Meta-Controller Regime Loop
  - **Objective:** Build the central orchestration loop managing the wake-sleep cycle: (1) Synthesize on curriculum tasks, (2) Harvest trajectories, (3) Discover abstractions (sleep-compression), (4) Dispatch asynchronous remote LoRA fine-tuning.
  - **Bottleneck Mitigation:** Asynchronous orchestration; double-buffered adapter swapping prevents local synthesis from blocking on remote training jobs.
  - **Automated Test Definition:** `pytest -v tests/orchestration/test_meta_controller.py`
    - Invariants: State machine transitions; non-blocking remote dispatch; hot-swaps adapter when remote job completes.
  - **Documentation Artifact:** `docs/architecture/meta_controller.md`
  - **Pass-to-Complete Gate:** 5 full wake-sleep iterations run with `MockComputeProvider` in < 2 seconds.

- [ ] Task 5.2: Financial Pre-Flight Cost Watchdog & Safety Gate
  - **Objective:** Implement `CostWatchdog` calculating projected job costs, enforcing hard session spending caps ($2.00 max), and intercepting unauthorized cloud dispatches.
  - **Bottleneck Mitigation:** Local zero-cost arithmetic check prior to every remote API invocation.
  - **Automated Test Definition:** `pytest -v tests/orchestration/test_cost_watchdog.py`
    - Invariants: Dispatches blocked when estimated cost exceeds remaining budget; warning logs emitted; falls back gracefully to mock provider.
  - **Documentation Artifact:** `docs/architecture/cost_watchdog.md`
  - **Pass-to-Complete Gate:** Budget exhaustion test verifies zero unauthorized external API calls are made.

- [ ] Task 5.3: CLI Interface & Rich Real-Time Progress Monitor
  - **Objective:** Expose commands (`symforge run`, `symforge train`, `symforge benchmark`) using Typer and Rich with real-time throughput, frontier depth, and ETA display.
  - **Bottleneck Mitigation:** Throttled console updates (max 2 updates/sec) to avoid terminal rendering overhead.
  - **Automated Test Definition:** `pytest -v tests/cli/test_cli.py`
    - Invariants: CLI flags (`--provider mock|modal|sagemaker`, `--max-budget`, `--dry-run`); clean exit codes.
  - **Documentation Artifact:** `docs/architecture/cli_interface.md`
  - **Pass-to-Complete Gate:** CLI runs 10-step synthesis simulation, formats output cleanly, and exits with code 0.

- [ ] Task 5.4: Phase 5 Verification & Checkpoint (Refer to workflow.md)

---

## Phase 6: Future Risk Mitigation, Hardening & Architectural Debt Resolution

Focus: Resolving architectural risks identified during planning—provider failure resilience, library bloat pruning, and offline degradation.

- [ ] Task 6.1: Resilient Offline Degraded Mode (Provider Failure Fallback)
  - **Objective:** Implement circuit breaker and local offline cache for trajectory payloads: if the remote provider is unreachable or times out, synthesis continues uninterrupted using the current model and caches trajectories locally.
  - **Bottleneck Mitigation:** Guarantees zero local crash when external networks or cloud providers degrade.
  - **Automated Test Definition:** `pytest -v tests/resilience/test_fallback.py`
    - Invariants: Network partition simulation; circuit opens after 3 consecutive failures; queued trajectories sync when connection restores.
  - **Documentation Artifact:** `docs/architecture/offline_fallback.md`
  - **Pass-to-Complete Gate:** System continues synthesizing without error during simulated total cloud network severance.

- [ ] Task 6.2: Grammar Bloat Mitigation & Combinatorial Pruning
  - **Objective:** Implement utility-based abstraction pruning with dependency tracking: if the DSL library exceeds 50 primitives, rarely used leaf primitives are safely pruned or inlined.
  - **Bottleneck Mitigation:** Prevents search branching factor explosion as the library grows over hundreds of curriculum cycles.
  - **Automated Test Definition:** `pytest -v tests/resilience/test_library_pruner.py`
    - Invariants: Dependent abstractions are never broken; inlining preserves exact semantics; beam search branching factor remains bounded.
  - **Documentation Artifact:** `docs/architecture/library_pruner.md`
  - **Pass-to-Complete Gate:** Pruner reduces library of 60 primitives down to top 40 most active without breaking any previous solution.

- [ ] Task 6.3: Phase 6 Verification & Checkpoint (Refer to workflow.md)

---

## Recursive Risk & Mitigation Analysis (The 6-Level Deep Analysis)

The following recursive risk analysis stress-tests the planned architecture to a depth of up to 6 levels, terminating only when residual risks reach Low Likelihood.

```text
+-------------------------------------------------------------------------------------------------------------------------+
|                                           RECURSIVE RISK & MITIGATION ARCHITECTURE                                      |
|                                                                                                                         |
| [Depth 1] Remote Training Latency & Async Staleness (High)                                                              |
|   `--> Mitigation: Double-buffered adapter hot-swapping                                                                 |
|         `--> [Depth 2] Off-policy trajectory staleness under new adapter (Medium)                                       |
|               `--> Mitigation: PPO-style importance sampling ratio clipping                                             |
|                     `--> [Depth 3] Extreme importance weight variance (Medium)                                          |
|                           `--> Mitigation: Dual-path replay buffer (PER for breakthroughs + clipped exploration)        |
|                                 `--> [Depth 4] Replay buffer memory explosion (Medium)                                  |
|                                       `--> Mitigation: Fixed capacity (10k) + canonical string serialization            |
|                                             `--> [Depth 5] Remote decompression latency on GPU (Low)                   |
|                                                   `--> Branch Terminated (Low Likelihood)                               |
+-------------------------------------------------------------------------------------------------------------------------+
| [Depth 1] Abstraction Discovery Library Bloat & Combinatorial Explosion (High)                                          |
|   `--> Mitigation: Strict MDL threshold + hard library cap with LRU eviction                                            |
|         `--> [Depth 2] Pruning an abstraction breaks downstream higher-order macros (High)                              |
|               `--> Mitigation: Dependency graph tracking; only leaf primitives evicted or auto-inlined                  |
|                     `--> [Depth 3] Deep inlining cascades blow up AST program size and hardware cost (Medium)           |
|                           `--> Mitigation: Inlining depth cap; lock non-inlinable macros from eviction                  |
|                                 `--> [Depth 4] Locked macros accumulate and bypass library cap (Medium)                 |
|                                       `--> Mitigation: Periodic E-graph equivalence saturation rewriting                |
|                                             `--> [Depth 5] E-graph explosion on associative/commutative rules (Medium)  |
|                                                   `--> Mitigation: Hard 5-iteration cap & 1000-node cap on E-graphs     |
|                                                         `--> [Depth 6] Suboptimal compression on edge cases (Low)       |
|                                                               `--> Branch Terminated (Depth 6 / Low Likelihood)         |
+-------------------------------------------------------------------------------------------------------------------------+
| [Depth 1] Financial Budget Runaway from Automated Remote Dispatches (High)                                              |
|   `--> Mitigation: Pre-flight CostWatchdog with hard session budget cap ($2.00)                                         |
|         `--> [Depth 2] Under-reporting provider costs due to hidden egress or cold starts (Medium)                      |
|               `--> Mitigation: 25% safety over-estimation buffer + minimum billing rounding                             |
|                     `--> [Depth 3] Over-estimation prematurely halts valid jobs (Low)                                   |
|                           `--> Branch Terminated (Low Likelihood)                                                       |
+-------------------------------------------------------------------------------------------------------------------------+
```

```mermaid
graph TD
    subgraph Branch1 [Branch 1: Remote Training Latency & Off-Policy Staleness]
        R1_1[Depth 1: Remote training latency blocks local search<br/>High Likelihood] --> M1_1[Mitigation: Double-buffered adapter hot-swapping]
        M1_1 --> R1_2[Depth 2: Off-policy trajectory staleness<br/>Medium Likelihood]
        R1_2 --> M1_2[Mitigation: Importance sampling ratio clipping]
        M1_2 --> R1_3[Depth 3: Extreme weight variance<br/>Medium Likelihood]
        R1_3 --> M1_3[Mitigation: Dual-path replay buffer PER + clipped exploration]
        M1_3 --> R1_4[Depth 4: Replay buffer local memory leak<br/>Medium Likelihood]
        R1_4 --> M1_4[Mitigation: 10k capacity FIFO + canonical string serialization]
        M1_4 --> R1_5[Depth 5: Remote decompression overhead<br/>Low Likelihood]
        R1_5 --> T1[Terminated: Low Likelihood]
    end

    subgraph Branch2 [Branch 2: Library Bloat & Combinatorial Explosion]
        R2_1[Depth 1: Library Bloat causes beam search explosion<br/>High Likelihood] --> M2_1[Mitigation: Strict MDL threshold + hard library cap with LRU eviction]
        M2_1 --> R2_2[Depth 2: Pruning breaks downstream dependent macros<br/>High Likelihood]
        R2_2 --> M2_2[Mitigation: Dependency graph tracking + safe leaf pruning/inlining]
        M2_2 --> R2_3[Depth 3: Inlining cascades blow up AST program size<br/>Medium Likelihood]
        R2_3 --> M2_3[Mitigation: Inlining depth cap + lock non-inlinable macros]
        M2_3 --> R2_4[Depth 4: Locked macros accumulate and leak capacity<br/>Medium Likelihood]
        R2_4 --> M2_4[Mitigation: Periodic E-graph equivalence saturation rewriting]
        M2_4 --> R2_5[Depth 5: E-graph explosion on commutative rules<br/>Medium Likelihood]
        R2_5 --> M2_5[Mitigation: Hard 5-iteration cap + 1000-node bound]
        M2_5 --> R2_6[Depth 6: Rare suboptimal compression<br/>Low Likelihood]
        R2_6 --> T2[Terminated: Depth 6 / Low Likelihood]
    end

    subgraph Branch3 [Branch 3: Financial Budget Runaway]
        R3_1[Depth 1: Meta-controller fires rapid expensive cloud jobs<br/>High Likelihood] --> M3_1[Mitigation: Pre-flight CostWatchdog with hard session cap $2.00]
        M3_1 --> R3_2[Depth 2: Under-reporting provider costs due to hidden fees<br/>Medium Likelihood]
        R3_2 --> M3_2[Mitigation: 25% safety over-estimation buffer + minimum billing rounding]
        M3_2 --> R3_3[Depth 3: Over-estimation prematurely halts training<br/>Low Likelihood]
        R3_3 --> T3[Terminated: Low Likelihood]
    end
```

### Detailed Recursive Risk Tree:

*   **[Depth 1] Risk:** Remote training latency causes local search to stall waiting for new LoRA weights or heuristics (High Likelihood)
    *   **Mitigation:** Implement Asynchronous Double-Buffering (Local search continues with the previous generation adapter $W_t$ while training of $W_{t+1}$ runs in the background. On completion, the new adapter is hot-swapped).
        *   **[Depth 2] Risk:** Off-policy staleness—trajectories collected with $W_t$ might be incompatible with or degraded by $W_{t+1}$ (Medium Likelihood)
            *   **Mitigation:** Implement Importance Sampling correction weights in the replay buffer and an adapter version tag on each trajectory.
                *   **[Depth 3] Risk:** Importance weights with extreme variance causing gradient instability during remote fine-tuning (Medium Likelihood)
                    *   **Mitigation:** Truncate importance weights (clipped importance ratio $w \in [0.8, 1.2]$, like PPO) and compute running variance.
                        *   **[Depth 4] Risk:** Aggressive clipping suppresses rare high-reward breakthrough trajectories from shaping the policy (Medium Likelihood)
                            *   **Mitigation:** Dual-path replay buffer—prioritized experience replay (PER) for breakthrough solutions with raw policy gradients, and clipped updates for standard exploratory steps.
                                *   **[Depth 5] Risk:** Replay buffer memory leaks and growing state serialization payload size sent over the network to Modal/SageMaker (Medium Likelihood)
                                    *   **Mitigation:** Cap replay buffer to $K=10,000$ transitions, store only canonical string representations of ASTs, and stream compressed gzip archives.
                                        *   **[Depth 6] Risk:** CPU decompression overhead during remote training batch preparation (Low Likelihood) -> *Branch Terminated (Depth 6 / Low Likelihood).*

*   **[Depth 1] Risk:** Abstraction discovery introduces too many macro-primitives, causing "Library Bloat" and combinatorial branching explosion in beam search (High Likelihood)
    *   **Mitigation:** Enforce strict MDL compression threshold ($\Delta \text{MDL} > \theta_{\text{min}}$) and a hard cap on library size with LRU/utility-based pruning of rarely used abstractions.
        *   **[Depth 2] Risk:** Pruning an abstraction that is a sub-dependency of another higher-order abstraction breaks downstream programs (High Likelihood)
            *   **Mitigation:** Dependency graph tracking for abstractions; only leaf primitives with zero dependents can be evicted, or primitives are auto-inlined into their parents before removal.
                *   **[Depth 3] Risk:** Auto-inlining cascades across deep macro hierarchies, re-inflating AST program sizes and violating execution cost bounds (Medium Likelihood)
                    *   **Mitigation:** Bounded inlining depth check; if inlining exceeds cost budget $\Delta C_H > \epsilon$, the abstraction is locked and cannot be evicted.
                        *   **[Depth 4] Risk:** Locked abstractions accumulate over long training runs, effectively defeating the eviction cap and causing memory leak (Medium Likelihood)
                            *   **Mitigation:** Periodic global E-graph / equivalence saturation rewriting to find unified canonical minimal representations across all active programs.
                                *   **[Depth 5] Risk:** Equivalence saturation graph explosion when rewrite rules have associative/commutative symmetry (Medium Likelihood)
                                    *   **Mitigation:** Apply strict iteration limit (5 iterations) and AST node cap (1,000 nodes per E-graph) with timeout per rewrite session.
                                        *   **[Depth 6] Risk:** Suboptimal simplification on complex expressions due to early timeout (Low Likelihood) -> *Branch Terminated (Depth 6 / Low Likelihood).*

*   **[Depth 1] Risk:** Financial cost runaway if meta-controller triggers remote training jobs too frequently (High Likelihood)
    *   **Mitigation:** Hard financial budget ceiling per session (e.g. $2.00 max) tracked by `CostWatchdog`, which halts remote dispatch and forces mock mode when threshold is reached.
        *   **[Depth 2] Risk:** Flawed cost tracking logic under-reports actual cloud provider bill due to hidden egress or cold-start billing minimums (Medium Likelihood)
            *   **Mitigation:** Conservative over-estimation model (adds 25% safety margin + minimum billing duration buffer, e.g. rounding 35s to 60s).
                *   **[Depth 3] Risk:** Overly conservative estimation causes premature throttling of valid training jobs when budget is still sufficient (Low Likelihood) -> *Branch Terminated (Low Likelihood).*
