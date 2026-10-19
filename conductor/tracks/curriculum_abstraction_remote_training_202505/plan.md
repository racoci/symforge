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

## Phase 2: Pillar 1 - Audio Synthesizer (1D Procedural DSP & SFX)

Focus: 1D digital signal processing primitives, multi-scale STFT spectral perceptual loss, audio PCFG grammar, TorchSynth/Dexed ingestion, and procedural timbre/SFX synthesis.

- [ ] Task 2.1: DSP Primitives Library
  - **Objective:** Implement pure functional 1D DSP primitives: Oscillators (`Sine`, `Saw`, `Square`, `Noise`), Envelopes (`ADSR`, `ExpDecay`), Modulators (`LFO`, `FM`), and Filters (`LowPass`, `Gain`, `Mix`).
  - **Bottleneck Mitigation:** Vectorized NumPy audio array generation over time vector $t \in [0, T]$ at 44.1kHz / 16kHz.
  - **Automated Test Definition:** `pytest -v tests/audio/test_dsp_primitives.py`
    - Invariants: Analytical waveform frequencies match within 0.1Hz; amplitude bounds remain strictly within $[-1.0, 1.0]$.
  - **Documentation Artifact:** `docs/architecture/audio_dsp_primitives.md`
  - **Pass-to-Complete Gate:** All primitives pass waveform assertion tests and generate clean audio arrays without NaN/Inf.

- [ ] Task 2.2: Multi-Scale STFT Feature Extractor & Contrastive Audio Loss
  - **Objective:** Implement multi-resolution Short-Time Fourier Transform (STFT) log-magnitude extractor and contrastive InfoNCE loss against spectral distractors.
  - **Bottleneck Mitigation:** SciPy FFT implementation optimized for sub-second batch evaluation.
  - **Automated Test Definition:** `pytest -v tests/audio/test_audio_loss.py`
    - Invariants: Invariant to phase shifts; sensitive to harmonic frequency deviations; discriminates timbre differences.
  - **Documentation Artifact:** `docs/architecture/audio_perceptual_loss.md`
  - **Pass-to-Complete Gate:** Differentiates between Sawtooth and Square waves of equal pitch with contrastive separation margin $> 2.0$.

- [ ] Task 2.3: Audio PCFG Domain Grammar & Procedural Timbre Generator
  - **Objective:** Instantiate the audio domain grammar with stability constraints (preventing unstable IIR filter poles and audio clipping) to generate synthetic audio synthesis tasks.
  - **Bottleneck Mitigation:** Pre-calculated sample cache for common synthetic targets.
  - **Automated Test Definition:** `pytest -v tests/audio/test_audio_pcfg.py`
    - Invariants: 100% of generated audio programs evaluate without clipping ($|s(t)| \le 1.0$) or numerical divergence.
  - **Documentation Artifact:** `docs/architecture/audio_pcfg_grammar.md`
  - **Pass-to-Complete Gate:** Procedurally generates 100 diverse acoustic benchmarks (leads, pads, percussion).

- [ ] Task 2.4: Audio Abstraction Discovery Benchmark (Dexed, TorchSynth & NSynth)
  - **Objective:** Run end-to-end synthesis on synthetic audio targets and demonstrate automatic abstraction of common audio subgraphs (e.g., FM Synthesis Pair, Percussive Envelope) evaluated against TorchSynth/Dexed patches.
  - **Bottleneck Mitigation:** Local discrete beam search over audio grammar in < 5 seconds per task.
  - **Automated Test Definition:** `pytest -v tests/audio/test_audio_synthesis_benchmark.py`
    - Invariants: Reconstructs FM synthesis sound and identifies candidate reusable macro.
  - **Documentation Artifact:** `docs/architecture/audio_benchmark.md`
  - **Pass-to-Complete Gate:** Successfully discovers and solves audio target, extracting a valid 2-operator macro primitive.

- [ ] Task 2.5: Phase 2 Verification & Checkpoint (Refer to workflow.md)

---

## Phase 3: Pillar 2 - SDF & Material Synthesizer (3D Implicit Geometry & Shaders)

Focus: 3D geometric analytical primitives, affine transforms, CSG boolean operators, surface sampling, procedural material/PBR texturing, and contrastive geometric loss.

- [ ] Task 3.1: 3D SDF Primitives, Transforms & PBR Materials Library
  - **Objective:** Implement vectorized 3D SDF primitives (`Sphere`, `Box`, `Cylinder`, `Torus`), Affine Transforms (`Translate`, `Rotate`, `Scale`), CSG Combinators (`Union`, `Difference`, `SmoothUnion`), and procedural material/PBR texturing nodes (color, roughness, bump/normal maps).
  - **Bottleneck Mitigation:** Vectorized NumPy coordinates evaluation; accepts $(N, 3)$ points and returns $(N,)$ distance array and $(N, 3)$ material properties.
  - **Automated Test Definition:** `pytest -v tests/sdf/test_sdf_primitives.py`
    - Invariants: Analytical distance verification matches exact mathematical formulations within $\epsilon = 10^{-5}$.
  - **Documentation Artifact:** `docs/architecture/sdf_primitives.md`
  - **Pass-to-Complete Gate:** 100% test pass across all primitive shapes and spatial transformations.

- [ ] Task 3.2: Surface Sampling, Raymarching & Contrastive Geometric Loss
  - **Objective:** Implement $\varepsilon$-boundary surface point sampling, normal computation via finite differences, and contrastive geometric loss against distractor geometries.
  - **Bottleneck Mitigation:** Fast stratified sampling near bounding boxes; no heavy GPU shaders required for CPU verification.
  - **Automated Test Definition:** `pytest -v tests/sdf/test_sdf_loss.py`
    - Invariants: Surface point error approaches zero for matching shapes; separates geometrically disjoint shapes under InfoNCE.
  - **Documentation Artifact:** `docs/architecture/sdf_perceptual_loss.md`
  - **Pass-to-Complete Gate:** Contrastive loss separates sphere from cube with margin $> 2.0$.

- [ ] Task 3.3: SDF PCFG Domain Grammar & Procedural 3D Shape Generator
  - **Objective:** Instantiate the geometric PCFG grammar with non-empty intersection and spatial containment constraints to generate valid synthetic 3D CAD benchmarks.
  - **Bottleneck Mitigation:** Bounding box overlap pre-filtering prevents generation of disjoint boolean operations.
  - **Automated Test Definition:** `pytest -v tests/sdf/test_sdf_pcfg.py`
    - Invariants: All generated SDF programs have non-empty surface boundaries within $[-2, 2]^3$.
  - **Documentation Artifact:** `docs/architecture/sdf_pcfg_grammar.md`
  - **Pass-to-Complete Gate:** Generates 100 non-degenerate 3D geometric tasks.

- [ ] Task 3.4: SDF Abstraction Discovery Benchmark (CSGNet, ABC CAD & MatSynth)
  - **Objective:** Synthesize composite shapes (e.g., Flanged Cylinder, Chamfered Hollow Box, Procedural Textures) and extract geometric macro-primitives.
  - **Bottleneck Mitigation:** Local beam search solving shapes in < 10 seconds.
  - **Automated Test Definition:** `pytest -v tests/sdf/test_sdf_synthesis_benchmark.py`
    - Invariants: Successfully synthesizes target hollow shape and extracts reusable CSG macro.
  - **Documentation Artifact:** `docs/architecture/sdf_benchmark.md`
  - **Pass-to-Complete Gate:** Solves hollow shape and extracts a valid compound primitive.

- [ ] Task 3.5: Phase 3 Verification & Checkpoint (Refer to workflow.md)

---

## Phase 4: Decoupled Compute Provider & LoRA Layer (Zero Local Training Footprint)

Focus: Abstracting heavy neural training, LoRA fine-tuning, and gradient updates away from the local workstation into a pluggable remote compute protocol.

- [ ] Task 4.1: Pluggable ComputeProvider Protocol & Data Contracts
  - **Objective:** Define the asynchronous `ComputeProvider` protocol and Pydantic schemas for `TrainingPayload`, `RemoteJobConfig`, `JobStatusResponse`, and `AdapterArtifact`.
  - **Bottleneck Mitigation:** Pure protocol definition; decouples local synthesis engine from any specific cloud provider SDK.
  - **Automated Test Definition:** `pytest -v tests/compute/test_contracts.py`
    - Invariants: Schema validation for valid and invalid payloads; contract enforcement on dummy implementation.
  - **Documentation Artifact:** `docs/architecture/compute_provider_protocol.md`
  - **Pass-to-Complete Gate:** 100% schema serialization round-trip tests pass.

- [ ] Task 4.2: MockComputeProvider & Local Zero-Cost Testbed
  - **Objective:** Implement `MockComputeProvider` simulating asynchronous training job submission, polling, and artifact generation in < 10ms with zero GPU usage and zero cost.
  - **Bottleneck Mitigation:** Provides a fully functional backend for local test suites and CI/CD pipelines without requiring network or cloud credentials.
  - **Automated Test Definition:** `pytest -v tests/compute/test_mock_provider.py`
    - Invariants: Deterministic mock job IDs; simulated state transitions (`PENDING` -> `RUNNING` -> `COMPLETED`); returns valid dummy LoRA weights.
  - **Documentation Artifact:** `docs/architecture/mock_compute_provider.md`
  - **Pass-to-Complete Gate:** Async submission and artifact fetch cycle completes in < 50ms with valid artifact checksum.

- [ ] Task 4.3: Remote Serverless GPU Providers (Modal & SageMaker) with Async Resilient Loops
  - **Objective:** Implement `ModalComputeProvider` and `SageMakerComputeProvider` using resilient async HTTP/Boto3 clients with hard timeouts, exponential backoff with jitter, and max poll caps.
  - **Bottleneck Mitigation:** Wrapped behind interface; all network calls have strict connection (10s) and read (30s) timeouts; resources severed in `finally` blocks.
  - **Automated Test Definition:** `pytest -v tests/compute/test_remote_providers.py`
    - Invariants: Mocked HTTP/Boto3 responses; test exponential backoff formula; verify hard timeout and cancellation on simulated network freeze.
  - **Documentation Artifact:** `docs/architecture/remote_compute_providers.md`
  - **Pass-to-Complete Gate:** Unit tests pass with simulated network failure scenarios and verified clean client severance.

- [ ] Task 4.4: Modular LoRA Adapter Registry & Local CPU Inference Loader
  - **Objective:** Create the `LoRAAdapterRegistry` to manage domain-specific adapters (`audio_modulators`, `audio_filters`, `sdf_csg`, `sdf_blends`) and load lightweight PEFT tensors into the local CPU prior evaluator.
  - **Bottleneck Mitigation:** Adapters are modular and small (< 15 MB); base model remains frozen; inference runs on CPU in milliseconds.
  - **Automated Test Definition:** `pytest -v tests/adapters/test_lora_registry.py`
    - Invariants: Adapter registration, version tracking, checksum validation, hot-swapping between adapters during search.
  - **Documentation Artifact:** `docs/architecture/lora_adapter_registry.md`
  - **Pass-to-Complete Gate:** Registry loads and hot-swaps simulated adapter weights without memory accumulation.

- [ ] Task 4.5: Phase 4 Verification & Checkpoint (Refer to workflow.md)

---

## Phase 5: DreamCoder Abstraction Discovery Engine & Adaptive Curriculum

Focus: Subgraph anti-unification, Minimum Description Length (MDL) compression evaluation, dynamic Lean 4 grammar expansion, and competency frontier tracking.

- [ ] Task 5.1: Subgraph Anti-Unification & Structural Pattern Extractor
  - **Objective:** Implement anti-unification on pairs of AST/DAG programs to identify the Least General Generalization (LGG) and extract shared subtrees with parameterized holes.
  - **Bottleneck Mitigation:** Operates over discovered solution programs ($M \approx 20..50$) rather than entire search spaces, keeping CPU overhead under 2 seconds.
  - **Automated Test Definition:** `pytest -v tests/abstraction/test_anti_unification.py`
    - Invariants: Identifies exact shared structure between `Difference(Cylinder, Cylinder)` and `Difference(Box, Box)` as well as audio `Mix(Osc, Osc)`.
  - **Documentation Artifact:** `docs/architecture/anti_unification.md`
  - **Pass-to-Complete Gate:** Correctly discovers candidate macro-primitives from pairs of geometric and audio programs.

- [ ] Task 5.2: Minimum Description Length (MDL) Compression Evaluator
  - **Objective:** Calculate compression utility $\Delta \text{MDL}(\alpha) = \text{Size}(\alpha) - \sum (\text{Size}(p) - \text{Size}(p[\alpha]))$ to filter candidate abstractions.
  - **Bottleneck Mitigation:** Fast discrete counting of AST node types and parameter bits; rejects unhelpful candidates immediately.
  - **Automated Test Definition:** `pytest -v tests/abstraction/test_mdl.py`
    - Invariants: Trivial or single-use abstractions return $\Delta \text{MDL} \ge 0$ (rejected); recurring subgraphs return $\Delta \text{MDL} < 0$ (accepted).
  - **Documentation Artifact:** `docs/architecture/mdl_compression.md`
  - **Pass-to-Complete Gate:** Evaluates 20 candidates, accepting only the top compressing abstractions.

- [ ] Task 5.3: Dynamic Lean 4 Grammar Expander & Library Injection
  - **Objective:** Invert candidate abstractions into first-class DSL productions $\mathcal{R}$, register typed signatures, and dynamically update the Lean 4 inductive environment and PCFG prior distribution.
  - **Bottleneck Mitigation:** Grammar updates are append-only with versioned rollback capability; new productions inherit strict Lean 4 type rules.
  - **Automated Test Definition:** `pytest -v tests/abstraction/test_grammar_expander.py`
    - Invariants: Injected macro can be instantiated as a typed node in subsequent synthesis runs; expanding macro produces identical output function.
  - **Documentation Artifact:** `docs/architecture/grammar_expander.md`
  - **Pass-to-Complete Gate:** Synthesizer successfully uses newly invented macro-primitive in a subsequent synthesis task, reducing search steps by $> 40\%$.

- [ ] Task 5.4: Competency Frontier Tracker & ZPD Sampler
  - **Objective:** Implement the Competency Frontier engine tracking moving-average success rate $S(c)$ and sampling tasks from the Zone of Proximal Development ($0.2 \le S(c) \le 0.8$) across both audio and SDF tasks.
  - **Bottleneck Mitigation:** Pure statistics and sampling logic; decoupled from synthesis execution.
  - **Automated Test Definition:** `pytest -v tests/curriculum/test_frontier.py`
    - Invariants: Advances frontier when $S(c) > 0.8$; falls back if regression occurs; avoids sampling from unmasterable bands ($S(c) < 0.2$).
  - **Documentation Artifact:** `docs/architecture/competency_frontier.md`
  - **Pass-to-Complete Gate:** Monotonic frontier progression demonstrated on simulated student agents.

- [ ] Task 5.5: Phase 5 Verification & Checkpoint (Refer to workflow.md)

---

## Phase 6: Meta-Controller, Cross-Domain Generalization & Risk Mitigation

Focus: Central orchestration loop, financial cost watchdog, cross-domain validation (including hardware instruction superoptimization), library bloat pruning, and offline degradation resilience.

- [ ] Task 6.1: Meta-Controller Regime Loop
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
