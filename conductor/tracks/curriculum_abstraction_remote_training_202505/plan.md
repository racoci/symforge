# Implementation Plan: Adaptive Curriculum with Abstraction Discovery and Remote Decoupled Fine-Tuning

**Track ID:** `curriculum_abstraction_remote_training_202505`  
**Spec Reference:** `conductor/tracks/curriculum_abstraction_remote_training_202505/spec.md`  
**Status:** Approved  
**Date:** 2026-10-16  

---

## Plan Structure & Methodology

This implementation plan strictly adheres to the **Conductor Workflow** and **GEMINI.md Software Engineering Principles**:
- **Three Core Pillars & Extended Domains:**
  1. **Pillar 1: Audio Synthesizer (1D Procedural DSP & SFX)** (NSynth, TorchSynth, Dexed, VCTK).
  2. **Pillar 2: SDF & Material Synthesizer (3D Implicit Geometry & PBR Textures)** (CSGNet, ABC CAD, MatSynth, Shadertoy).
  3. **Pillar 3: Generalization (Universal Neurosymbolic Core & Hardware Optimization)** (DreamCoder, BHive, TVM).
- **Lean 4 Formal Intermediate Representation (IR):** Formal grammar, dependent types, and inductive type declarations in Lean 4 act as the symbolic front-end with verified structural invariants, transpiled via metaprogramming to differentiable execution graphs (PyTorch/JAX) with explicit free parameter vectors $\Theta$.
- **Unified Problem Formulation:** Resolving all procedural domains via identical mechanics: combining known primitives into higher-order abstractions to satisfy target conditions evaluated via differentiable contrastive perceptual loss (InfoNCE), driven by procedural synthetic tasks from a Probabilistic Context-Free Grammar (PCFG).
- **Zero Local Training Footprint:** Heavy training workloads are delegated to remote providers (Modal, SageMaker) or deterministic local mocks.
- **Pass-to-Complete Gate:** Tasks are only marked `[x]` after test suites pass 100%.

---

## Phase 1: Lean 4 Formal IR, Universal Symbolic Core & PCFG Generator

Focus: Lean 4 package initialization, inductive type grammars (`FreeParam`, `AudioNode`, `SDFNode`), Lean-to-PyTorch metaprogramming transpiler, PCFG synthetic task generator, generic contrastive perceptual loss interface, and discrete beam/A* search.

- [x] Task 1.1: Lean 4 Package Scaffolding & Inductive Grammar Definitions [edff5ec]
  - **Objective:** Initialize the Lean 4 package (`SymForge.IR`) using `lake init` and define inductive grammar types (`AudioNode`, `SDFNode`, `HardwareNode`) with attached `FreeParam` records.
  - **Bottleneck Mitigation:** Pure symbolic definitions in Lean 4 without external network dependencies; provides compile-time mathematical guarantees of well-formedness.
  - **Automated Test Definition:** `lake test` or `lake build`
    - Invariants: Lean 4 compiler succeeds with 0 errors; proof of non-emptiness and acyclic inductive properties.
  - **Documentation Artifact:** `docs/architecture/lean4_formal_ir.md`
  - **Pass-to-Complete Gate:** `lake build` compiles cleanly without warnings.

- [x] Task 1.2: Typed AST & Canonical DAG Representation with Python-Lean Bridge [f9306f6]
  - **Objective:** Implement immutable typed AST/DAG representation in Python and bidirectional FFI/IPC bridge to serialize/deserialize programs between Python and Lean 4.
  - **Bottleneck Mitigation:** JSON/IPC serializable data contract; isolated bridge process with explicit timeouts.
  - **Automated Test Definition:** `pytest -v tests/core/test_ast.py tests/core/test_lean_bridge.py`
    - Invariants: Round-trip identity `deserialize(serialize(p)) == p`; Python AST exactly reflects Lean 4 inductive type signatures.
  - **Documentation Artifact:** `docs/architecture/typed_ast_dag.md`
  - **Pass-to-Complete Gate:** 100% round-trip serialization pass across 1,000 generated programs.

- [x] Task 1.3: Lean 4 Metaprogramming Transpiler (AST -> Differentiable ML Graph) [7be0b74]
  - **Objective:** Write Lean 4 elaboration macros (`Lean.Elab`) that visit an AST instance and emit a differentiable PyTorch `nn.Module` / JAX computational graph, bundling all `FreeParam` into a single continuous tensor $\Theta \in \mathbb{R}^K$.
  - **Bottleneck Mitigation:** Pure code generation; execution occurs downstream in Python.
  - **Automated Test Definition:** `pytest -v tests/core/test_transpiler.py`
    - Invariants: Transpiled PyTorch module produces identical forward pass values to analytical Lean specification; autograd computes non-zero gradients for all $\theta \in \Theta$.
  - **Documentation Artifact:** `docs/architecture/lean4_transpiler.md`
  - **Pass-to-Complete Gate:** Transpiled PyTorch modules for audio and SDF primitives match reference outputs within $\epsilon = 10^{-5}$.

- [x] Task 1.4: PCFG & Context-Sensitive Synthetic Task Generator [2cff1c3]
  - **Objective:** Build the general probabilistic grammar engine (`PCFGTaskGenerator`) capable of sampling procedural target programs and synthesizing evaluation sets under contextual validity rules.
  - **Bottleneck Mitigation:** In-memory stochastic derivation generator with seeded determinism for testing.
  - **Automated Test Definition:** `pytest -v tests/core/test_pcfg_generator.py`
    - Invariants: Sampling adheres strictly to rule probability distributions; context validator prunes invalid structural combinations.
  - **Documentation Artifact:** `docs/architecture/pcfg_generator.md`
  - **Pass-to-Complete Gate:** Generates 500 valid synthetic derivations without infinite recursive loops or type violations.

- [x] Task 1.5: Differentiable Contrastive Perceptual Loss Engine
  - **Objective:** Implement the domain-agnostic `ContrastivePerceptualLoss` interface and InfoNCE calculation against positive targets and hard negative distractors, integrated with compiled hardware cost $C_H(G)$.
  - **Bottleneck Mitigation:** Vectorized NumPy / SciPy computation with PyTorch autograd bridge.
  - **Automated Test Definition:** `pytest -v tests/eval/test_contrastive_loss.py tests/eval/test_cost_model.py`
    - Invariants: Loss decreases monotonically as synthesized embedding aligns with positive target; cost model monotonicity ($C(G_1 \cup G_2) > C(G_1)$).
  - **Documentation Artifact:** `docs/architecture/contrastive_perceptual_loss.md`
  - **Pass-to-Complete Gate:** InfoNCE loss reaches near-zero for identical embeddings and penalizes orthogonal distractors with verified gradients.

- [x] Task 1.6: Type-Guided Beam & A* Synthesizer with Hole Expansion
  - **Objective:** Implement type-safe hole replacement and prioritized beam/A* search exploring partial programs under cost budgets.
  - **Bottleneck Mitigation:** Uses pluggable heuristic scorer interface, allowing uniform or heuristic priors before neural models are attached.
  - **Automated Test Definition:** `pytest -v tests/search/test_synthesizer.py`
    - Invariants: Hole expansion only generates type-compatible rules; beam search prunes branches exceeding cost budget; discovers simple target shapes/signals.
  - **Documentation Artifact:** `docs/architecture/synthesizer_search.md`
  - **Pass-to-Complete Gate:** Synthesizer solves baseline target problems within 200 expansion steps.

- [x] Task 1.7: Phase 1 Verification & Checkpoint (Refer to workflow.md)

---

## Phase 2: Paired Domain Primitives, AST Extraction & Filtered Data Pipelines (Audio 1D + SDF 3D) [checkpoint: c255f7e]

Focus: Implement procedural primitives, differentiable perceptual loss evaluators, Tree-sitter AST normalizers for real-world code (OpenSCAD, Dexed/Faust), and procedural PCFG generators with triple quality filtering (Physical, Perceptual Entropy, Semantic Deduplication) to mitigate the synthetic prior trap.

- [x] Task 2.1: Audio 1D DSP Primitives & Multi-scale STFT Extractor
  - **Objective:** Implement pure functional 1D DSP primitives (Oscillators, Envelopes, Modulators, Filters) and the multi-resolution STFT contrastive loss extractor.
  - **Automated Test Definition:** `pytest -v tests/audio/test_dsp_primitives.py tests/audio/test_audio_loss.py`
  - **Pass-to-Complete Gate:** All primitives pass analytical waveform tests; STFT discriminates Timbre differences.

- [x] Task 2.2: SDF 3D CSG Primitives & Surface/Raymarching Extractor
  - **Objective:** Implement 3D SDF primitives (Sphere, Box, Cylinder, etc.), spatial transforms, CSG booleans, and the $\varepsilon$-boundary surface contrastive geometric loss extractor.
  - **Automated Test Definition:** `pytest -v tests/sdf/test_sdf_primitives.py tests/sdf/test_sdf_loss.py`
  - **Pass-to-Complete Gate:** 100% test pass on shape distances; Loss completely separates disjoint geometric shapes.

- [x] Task 2.3: Procedural PCFG with Triple Quality Filtering & Complexity Stratification
  - **Objective:** Implement PCFG sampling with Triple Quality Filtering (Physical Validity, Perceptual Entropy, and Semantic DAG Deduplication) across 5 discrete complexity levels to prevent the self-mimicry trap.
  - **Automated Test Definition:** `pytest -v tests/core/test_pcfg_curriculum.py`
  - **Documentation Artifact:** `docs/architecture/pcfg_curriculum_filtering.md`
  - **Pass-to-Complete Gate:** Rejects trivial/silent programs; generates 100 non-degenerate derivations per complexity level.

- [x] Task 2.4: Domain Code Normalizer & Supervised Production Tuple Harvester
  - **Objective:** Build ingestion normalizers (OpenSCAD CST $\to$ AST, Dexed/Faust $\to$ AST) and the Supervised Production Extractor exporting tuples `(nonterminal, production, parent_context, expected_type, free_params)` for PCFG/PHOG training.
  - **Automated Test Definition:** `pytest -v tests/data/test_harvester.py tests/data/test_normalizer.py`
  - **Documentation Artifact:** `docs/architecture/data_normalizer_and_harvester.md`
  - **Pass-to-Complete Gate:** Parses sample OpenSCAD and Dexed JSON patches into valid Lean 4 ASTs; extracts valid production decision tuples.

- [x] Task 2.5: Phase 2 Verification & Baseline PCFG vs. PHOG Comparison (Refer to workflow.md) [c255f7e]

---

## Phase 3: The 3-Tier Neurosymbolic Pipeline (PDE $\to$ TGC $\to$ FPO) [checkpoint: 72f5465]

Focus: Deconstruct the monolithic synthesis problem into the 3-Tier Model Pipeline (Primitive+Depth Estimator, Topology Connector, Fast Parameter Optimizer) and benchmark Audio vs SDF learning dynamics.

- [x] Task 3.1: Model 1 (PDE) - Primitive & Hierarchical Depth Estimator [f02476c]
  - **Objective:** Implement the PDE neural architecture that maps an observation $x$ (audio or SDF) to a multiset of primitives and their AST depth levels $\{(p_i, \ell_i)\}$.
  - **Automated Test Definition:** `pytest -v tests/models/test_pde.py`
  - **Pass-to-Complete Gate:** Model trains to $\ge 90\%$ F1-score on synthetic Level 2/3 primitive identification.

- [x] Task 3.2: Model 2 (TGC) - Topology & Grammar Connector [daad682]
  - **Objective:** Implement the TGC module which acts as an informed structural assembler, taking the predicted multiset $\{(p_i, \ell_i)\}$ and resolving the valid topological DAG $G$ via constrained beam search.
  - **Automated Test Definition:** `pytest -v tests/models/test_tgc.py`
  - **Documentation Artifact:** `docs/architecture/tgc_model.md`
  - **Pass-to-Complete Gate:** Assembly solves valid ASTs from shuffled multiset predictions in $< 50$ ms.

- [x] Task 3.3: Model 3 (FPO) - Fast Differentiable Parameter Optimizer [fcfe3eb]
  - **Objective:** Implement the final continuous parameter estimator, predicting warm-start $\Theta_0$ and executing fast gradient backprop over InfoNCE loss via PyTorch Autograd.
  - **Automated Test Definition:** `pytest -v tests/models/test_fpo.py`
  - **Pass-to-Complete Gate:** Gradient descent converges in $\le 30$ steps for known geometries/audio signals.

- [x] Task 3.4: Complexity-Ordered Curriculum Benchmark (Audio vs SDF) [72f5465]
  - **Objective:** Run the full 3-Tier pipeline on Curriculum Levels 1 through 3 for both Audio and SDF simultaneously to benchmark which domain exhibits higher learning difficulty.
  - **Automated Test Definition:** `pytest -v tests/benchmarks/test_curriculum_benchmark.py`
  - **Pass-to-Complete Gate:** Benchmark script runs end-to-end and outputs comparative learning curves (loss vs curriculum depth).

- [x] Task 3.5: Phase 3 Verification & Checkpoint (Refer to workflow.md) [72f5465]

---

## Phase 4: Decoupled Compute Provider & LoRA Layer (Zero Local Training Footprint) [checkpoint: 2c93b16]

Focus: Abstracting heavy neural training, LoRA fine-tuning, and gradient updates away from the local workstation into a pluggable remote compute protocol.

- [x] Task 4.1: Pluggable ComputeProvider Protocol & Data Contracts [4bb59c7]
  - **Objective:** Define the asynchronous `ComputeProvider` protocol and Pydantic schemas for `TrainingPayload`, `RemoteJobConfig`, `JobStatusResponse`, and `AdapterArtifact`.
  - **Bottleneck Mitigation:** Pure protocol definition; decouples local synthesis engine from any specific cloud provider SDK.
  - **Automated Test Definition:** `pytest -v tests/compute/test_contracts.py`
    - Invariants: Schema validation for valid and invalid payloads; contract enforcement on dummy implementation.
  - **Documentation Artifact:** `docs/architecture/compute_provider_protocol.md`
  - **Pass-to-Complete Gate:** 100% schema serialization round-trip tests pass.

- [x] Task 4.2: MockComputeProvider & Local Zero-Cost Testbed [3debca1]
  - **Objective:** Implement `MockComputeProvider` simulating asynchronous training job submission, polling, and artifact generation in < 10ms with zero GPU usage and zero cost.
  - **Bottleneck Mitigation:** Provides a fully functional backend for local test suites and CI/CD pipelines without requiring network or cloud credentials.
  - **Automated Test Definition:** `pytest -v tests/compute/test_mock_provider.py`
    - Invariants: Deterministic mock job IDs; simulated state transitions (`PENDING` -> `RUNNING` -> `COMPLETED`); returns valid dummy LoRA weights.
  - **Documentation Artifact:** `docs/architecture/mock_compute_provider.md`
  - **Pass-to-Complete Gate:** Async submission and artifact fetch cycle completes in < 50ms with valid artifact checksum.

- [x] Task 4.3: Remote Serverless GPU Providers (Modal & SageMaker) with Async Resilient Loops [e38ed16]
  - **Objective:** Implement `ModalComputeProvider` and `SageMakerComputeProvider` using resilient async HTTP/Boto3 clients with hard timeouts, exponential backoff with jitter, and max poll caps.
  - **Bottleneck Mitigation:** Wrapped behind interface; all network calls have strict connection (10s) and read (30s) timeouts; resources severed in `finally` blocks.
  - **Automated Test Definition:** `pytest -v tests/compute/test_remote_providers.py`
    - Invariants: Mocked HTTP/Boto3 responses; test exponential backoff formula; verify hard timeout and cancellation on simulated network freeze.
  - **Documentation Artifact:** `docs/architecture/remote_compute_providers.md`
  - **Pass-to-Complete Gate:** Unit tests pass with simulated network failure scenarios and verified clean client severance.

- [x] Task 4.4: Modular LoRA Adapter Registry & Local CPU Inference Loader [ed0aa5d]
  - **Objective:** Create the `LoRAAdapterRegistry` to manage domain-specific adapters (`audio_modulators`, `audio_filters`, `sdf_csg`, `sdf_blends`) and load lightweight PEFT tensors into the local CPU prior evaluator.
  - **Bottleneck Mitigation:** Adapters are modular and small (< 15 MB); base model remains frozen; inference runs on CPU in milliseconds.
  - **Automated Test Definition:** `pytest -v tests/adapters/test_lora_registry.py`
    - Invariants: Adapter registration, version tracking, checksum validation, hot-swapping between adapters during search.
  - **Documentation Artifact:** `docs/architecture/lora_adapter_registry.md`
  - **Pass-to-Complete Gate:** Registry loads and hot-swaps simulated adapter weights without memory accumulation.

- [x] Task 4.5: Phase 4 Verification & Checkpoint (Refer to workflow.md) [2c93b16]

---

## Phase 5: DreamCoder Abstraction Discovery Engine & Adaptive Curriculum [checkpoint: a676c3b]

Focus: Subgraph anti-unification, Minimum Description Length (MDL) compression evaluation, dynamic Lean 4 grammar expansion, and competency frontier tracking.

- [x] Task 5.1: Subgraph Anti-Unification & Structural Pattern Extractor [c700004]
  - **Objective:** Implement anti-unification on pairs of AST/DAG programs to identify the Least General Generalization (LGG) and extract shared subtrees with parameterized holes.
  - **Bottleneck Mitigation:** Operates over discovered solution programs ($M \approx 20..50$) rather than entire search spaces, keeping CPU overhead under 2 seconds.
  - **Automated Test Definition:** `pytest -v tests/abstraction/test_anti_unification.py`
    - Invariants: Identifies exact shared structure between `Difference(Cylinder, Cylinder)` and `Difference(Box, Box)` as well as audio `Mix(Osc, Osc)`.
  - **Documentation Artifact:** `docs/architecture/anti_unification.md`
  - **Pass-to-Complete Gate:** Correctly discovers candidate macro-primitives from pairs of geometric and audio programs.

- [x] Task 5.2: Minimum Description Length (MDL) Compression Evaluator [6cc68b5]
  - **Objective:** Calculate compression utility $\Delta \text{MDL}(\alpha) = \text{Size}(\alpha) - \sum (\text{Size}(p) - \text{Size}(p[\alpha]))$ to filter candidate abstractions.
  - **Bottleneck Mitigation:** Fast discrete counting of AST node types and parameter bits; rejects unhelpful candidates immediately.
  - **Automated Test Definition:** `pytest -v tests/abstraction/test_mdl.py`
    - Invariants: Trivial or single-use abstractions return $\Delta \text{MDL} \ge 0$ (rejected); recurring subgraphs return $\Delta \text{MDL} < 0$ (accepted).
  - **Documentation Artifact:** `docs/architecture/mdl_compression.md`
  - **Pass-to-Complete Gate:** Evaluates 20 candidates, accepting only the top compressing abstractions.

- [x] Task 5.3: Dynamic Lean 4 Grammar Expander & Library Injection [db61ce8]
  - **Objective:** Invert candidate abstractions into first-class DSL productions $\mathcal{R}$, register typed signatures, and dynamically update the Lean 4 inductive environment and PCFG prior distribution.
  - **Bottleneck Mitigation:** Grammar updates are append-only with versioned rollback capability; new productions inherit strict Lean 4 type rules.
  - **Automated Test Definition:** `pytest -v tests/abstraction/test_grammar_expander.py`
    - Invariants: Injected macro can be instantiated as a typed node in subsequent synthesis runs; expanding macro produces identical output function.
  - **Documentation Artifact:** `docs/architecture/grammar_expander.md`
  - **Pass-to-Complete Gate:** Synthesizer successfully uses newly invented macro-primitive in a subsequent synthesis task, reducing search steps by $> 40\%$.

- [x] Task 5.4: Competency Frontier Tracker & ZPD Sampler [a676c3b]
  - **Objective:** Implement the Competency Frontier engine tracking moving-average success rate $S(c)$ and sampling tasks from the Zone of Proximal Development ($0.2 \le S(c) \le 0.8$) across both audio and SDF tasks.
  - **Bottleneck Mitigation:** Pure statistics and sampling logic; decoupled from synthesis execution.
  - **Automated Test Definition:** `pytest -v tests/curriculum/test_frontier.py`
    - Invariants: Advances frontier when $S(c) > 0.8$; falls back if regression occurs; avoids sampling from unmasterable bands ($S(c) < 0.2$).
  - **Documentation Artifact:** `docs/architecture/competency_frontier.md`
  - **Pass-to-Complete Gate:** Monotonic frontier progression demonstrated on simulated student agents.

- [x] Task 5.5: Phase 5 Verification & Checkpoint (Refer to workflow.md) [a676c3b]

---

## Phase 6: Meta-Controller, Cross-Domain Generalization & Risk Mitigation

Focus: Central orchestration loop, financial cost watchdog, cross-domain validation (including hardware instruction superoptimization), library bloat pruning, and offline degradation resilience.

- [x] Task 6.1: Meta-Controller Regime Loop [6aa21e0]
  - **Objective:** Build the central orchestration loop managing the wake-sleep cycle: (1) Synthesize on curriculum tasks, (2) Harvest trajectories, (3) Discover abstractions (sleep-compression), (4) Dispatch asynchronous remote LoRA fine-tuning.
  - **Bottleneck Mitigation:** Asynchronous orchestration; double-buffered adapter swapping prevents local synthesis from blocking on remote training jobs.
  - **Automated Test Definition:** `pytest -v tests/orchestration/test_meta_controller.py`
    - Invariants: State machine transitions; non-blocking remote dispatch; hot-swaps adapter when remote job completes.
  - **Documentation Artifact:** `docs/architecture/meta_controller.md`
  - **Pass-to-Complete Gate:** 5 full wake-sleep iterations run with `MockComputeProvider` in < 2 seconds.

- [ ] Task 6.2: Financial Pre-Flight Cost Watchdog & Safety Gate
  - **Objective:** Implement `CostWatchdog` calculating projected job costs, enforcing hard session spending caps ($2.00 max), and intercepting unauthorized cloud dispatches.
  - **Bottleneck Mitigation:** Local zero-cost arithmetic check prior to every remote API invocation.
  - **Automated Test Definition:** `pytest -v tests/orchestration/test_cost_watchdog.py`
    - Invariants: Dispatches blocked when estimated cost exceeds remaining budget; warning logs emitted; falls back gracefully to mock provider.
  - **Documentation Artifact:** `docs/architecture/cost_watchdog.md`
  - **Pass-to-Complete Gate:** Budget exhaustion test verifies zero unauthorized external API calls are made.

- [ ] Task 6.3: Cross-Domain Unified Validation & Transfer Benchmark (Audio, SDF, Hardware)
  - **Objective:** Validate that the exact same synthesis and abstraction discovery engine operates successfully across Audio DSP, SDF 3D Geometry, and Hardware Instruction Superoptimization (BHive benchmark) without domain-specific algorithmic hacks.
  - **Bottleneck Mitigation:** Paired benchmark execution under identical resource budgets.
  - **Automated Test Definition:** `pytest -v tests/benchmarks/test_cross_domain_transfer.py`
    - Invariants: Identical search algorithm code paths used; all domains report successful synthesis and macro extraction.
  - **Documentation Artifact:** `docs/architecture/cross_domain_validation.md`
  - **Pass-to-Complete Gate:** Audio, SDF, and Hardware test suites achieve $\ge 80\%$ task completion rate under matched compute budgets.

- [ ] Task 6.4: Library Bloat Pruning & Resilient Offline Fallback
  - **Objective:** Implement utility-based abstraction pruning (hard cap on library size with safe leaf inlining) and offline circuit breaker (local trajectory caching during cloud network partitions).
  - **Bottleneck Mitigation:** Preserves bounded search branching factor and 100% uptime when external networks are offline.
  - **Automated Test Definition:** `pytest -v tests/resilience/test_library_pruner.py tests/resilience/test_fallback.py`
    - Invariants: Dependent abstractions never broken; inlining preserves semantics; synthesis continues during network drops.
  - **Documentation Artifact:** `docs/architecture/library_pruner.md`
  - **Pass-to-Complete Gate:** Pruner reduces library from 60 to 40 primitives safely; offline mode queues 100 trajectories without error.

- [ ] Task 6.5: Phase 6 Verification & Checkpoint (Refer to workflow.md)

---

## Recursive Risk & Mitigation Analysis (The 6-Level Deep Analysis)

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
    subgraph Branch1 [Ramo 1: Latência Remota & Estagnação Off-Policy]
        R1_1[Prof 1: Latência do treino remoto bloqueia a busca<br/>Alta Probabilidade] --> M1_1[Mitigação: Double-buffering e hot-swap de adaptadores]
        M1_1 --> R1_2[Prof 2: Obsolescência off-policy das trajetórias<br/>Média Probabilidade]
        R1_2 --> M1_2[Mitigação: Importance sampling ratio clipping tipo PPO]
        M1_2 --> R1_3[Prof 3: Variância extrema nos pesos<br/>Média Probabilidade]
        R1_3 --> M1_3[Mitigação: Buffer de replay dual PER + exploração truncada]
        M1_3 --> R1_4[Prof 4: Vazamento de memória local no buffer<br/>Média Probabilidade]
        R1_4 --> M1_4[Mitigação: Capacidade fixa 10k + strings canônicas gzip]
        M1_4 --> R1_5[Prof 5: Custo de descompressão remota<br/>Baixa Probabilidade]
        R1_5 --> T1[Terminado: Baixa Probabilidade]
    end

    subgraph Branch2 [Ramo 2: Inchaço da Biblioteca & Explosão de Busca]
        R2_1[Prof 1: Library Bloat explode branching do Beam Search<br/>Alta Probabilidade] --> M2_1[Mitigação: Limiar estrito de MDL + teto de biblioteca LRU]
        M2_1 --> R2_2[Prof 2: Remoção quebra macros dependentes<br/>Alta Probabilidade]
        R2_2 --> M2_2[Mitigação: Grafo de dependência + inlining seguro de folhas]
        M2_2 --> R2_3[Prof 3: Inlining em cascata estoura tamanho da AST<br/>Média Probabilidade]
        R2_3 --> M2_3[Mitigação: Teto de inlining + trava de macros essenciais]
        M2_3 --> R2_4[Prof 4: Macros travadas acumulam e furam teto<br/>Média Probabilidade]
        R2_4 --> M2_4[Mitigação: Saturação periódica por E-graphs]
        M2_4 --> R2_5[Prof 5: Explosão de E-graphs em regras comutativas<br/>Média Probabilidade]
        R2_5 --> M2_5[Mitigação: Limite rígido de 5 iterações e 1000 nós]
        M2_5 --> R2_6[Prof 6: Compressão rara subótima<br/>Baixa Probabilidade]
        R2_6 --> T2[Terminado: Profundidade 6 / Baixa Probabilidade]
    end

    subgraph Branch3 [Ramo 3: Descontrole Financeiro em Nuvem]
        R3_1[Prof 1: Meta-controller dispara jobs caros repetidamente<br/>Alta Probabilidade] --> M3_1[Mitigação: CostWatchdog com teto rígido de $2.00 por sessão]
        M3_1 --> R3_2[Prof 2: Subestimação de cobrança por tarifas de borda<br/>Média Probabilidade]
        R3_2 --> M3_2[Mitigação: Margem conservadora de 25% + arredondamento mínimo]
        M3_2 --> R3_3[Prof 3: Superestimação barra jobs válidos<br/>Baixa Probabilidade]
        R3_3 --> T3[Terminado: Baixa Probabilidade]
    end
```
