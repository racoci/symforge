# Type-Guided Beam & A* Synthesizer with Hole Expansion

## 1. Architectural Mission

The **Type-Guided Synthesizer Engine** (`symforge.search.synthesizer`) explores the infinite discrete combinatorial space of domain programs to discover computational graphs satisfying target perceptual and functional criteria under strict hardware cost budgets.

Key functional pillars:
1. **Typed Hole Expansion:** Partial programs contain typed placeholder holes (`AudioHoleNode`, `SDFHoleNode`). Expanding a hole generates only syntactically and semantically valid production rules for that specific domain type.
2. **Hardware Cost Budget Pruning:** Partial or complete candidate programs whose compiled hardware cost $C_H(G)$ exceeds the user-configured budget are immediately pruned from the search beam.
3. **Dual Search Strategies:** Provides both width-controlled **Beam Search** (`TypeGuidedBeamSynthesizer`) and optimal priority-ordered **A\* Search** (`AStarSynthesizer`) using $f = g(\text{cost}) + h(\text{loss})$.
4. **Pluggable Scorer Interface:** Decoupled heuristic scorer protocol (`ProgramScorer`) supporting reference AST comparison (`ReferenceASTScorer`), contrastive perceptual InfoNCE loss (`CustomLossScorer`), or learned heuristic estimators.

---

## 2. Search & Expansion Topology

```text
+-----------------------------------------------------------------------------------------------+
|                                TYPE-GUIDED SYNTHESIS PIPELINE                                 |
+-----------------------------------------------------------------------------------------------+
                                                |
                               +--------------------------------+
                               | Start Symbol: HoleNode("root") |
                               +----------------+---------------+
                                                |
                                                v
                              +----------------------------------+
                              | Prioritized Frontier / Beam      |
                              | Pop Best Candidate               |
                              +-----------------+----------------+
                                                |
                         +----------------------+----------------------+
                         | Complete Program?                           | Has Holes?
                         v                                             v
        +----------------------------------+         +----------------------------------+
        | Evaluate Loss against Target:    |         | Select First Hole h_0 in AST     |
        | score = Scorer.score(P)          |         +-----------------+----------------+
        +----------------+-----------------+                           |
                         |                                             v
        +----------------+-----------------+         +----------------------------------+
        | score <= tolerance?              |         | TypedGrammarExpander.expand(h_0) |
        +----------------+-----------------+         | Strictly Type-Safe Rules Only    |
                         |                           +-----------------+----------------+
                +--------+--------+                                    |
                | YES             | NO                                 v
                v                 v                  +----------------------------------+
        +---------------+ +---------------+          | Check Hardware Cost Budget:      |
        | Return Solved | | Keep in Beam/ |          | C_H(P') <= max_cost_budget?      |
        | Result AST    | | Continue      |          +-----------------+----------------+
        +---------------+ +---------------+                            |
                                                      +----------------+----------------+
                                                      | YES                             | NO
                                                      v                                 v
                                            +-------------------+             +-------------------+
                                            | Compute Priority  |             | Prune Branch      |
                                            | Push to Frontier  |             | (Budget Exceeded) |
                                            +-------------------+             +-------------------+
```

```mermaid
graph TD
    RootNode([Initial Typed Hole: AudioHole / SDFHole]) --> Frontier[Priority Frontier / Beam Queue]
    
    Frontier --> PopBest[Pop Highest Priority Candidate]
    PopBest --> CheckHoles{Contains Holes?}
    
    CheckHoles -->|No - Complete| EvalComplete[Evaluate Perceptual Loss / MSE]
    EvalComplete --> GoalCheck{Score <= Tolerance?}
    GoalCheck -->|Yes| Solved([Synthesis Success: Optimal Program Found])
    GoalCheck -->|No| RetainCandidate[Retain in Frontier Pool]
    
    CheckHoles -->|Yes - Partial| SelectHole[Select First Hole h_0]
    SelectHole --> Expander[Typed Grammar Expander]
    Expander --> TypeFilter[Generate Domain-Compatible Rules Only]
    
    TypeFilter --> BudgetCheck{C_H(P') <= Cost Budget?}
    BudgetCheck -->|No| Pruned([Pruned: Budget Exceeded])
    BudgetCheck -->|Yes| ScoreHeuristic[Score Candidate: f = score + lambda_cost * C_H]
    ScoreHeuristic --> PushQueue[Insert into Frontier / Beam Pool]
    PushQueue --> Frontier
```

---

## 3. Typed Hole Expansion Rules

### 3.1 Audio Domain
When expanding `AudioHoleNode(id)`:
- **Terminal Primitives:** `SineNode`, `SawNode`, `SquareNode`, `NoiseNode`, `FMNode`.
- **Non-Terminal Combinators:**
  - `LowPassNode(cutoff, resonance, input_node=AudioHoleNode(h1))`
  - `GainNode(gain, input_node=AudioHoleNode(h1))`
  - `MixNode(ratio, input1=AudioHoleNode(h1), input2=AudioHoleNode(h2))`

### 3.2 SDF 3D Geometric Domain
When expanding `SDFHoleNode(id)`:
- **Terminal Primitives:** `SphereNode`, `BoxNode`, `CylinderNode`, `TorusNode`.
- **Non-Terminal Transforms & Booleans:**
  - `TranslateNode(dx, dy, dz, child=SDFHoleNode(h1))`
  - `RotateNode(rx, ry, rz, child=SDFHoleNode(h1))`
  - `ScaleNode(s, child=SDFHoleNode(h1))`
  - `UnionNode(smoothness, a=SDFHoleNode(h1), b=SDFHoleNode(h2))`
  - `IntersectionNode(a=SDFHoleNode(h1), b=SDFHoleNode(h2))`
  - `DifferenceNode(a=SDFHoleNode(h1), b=SDFHoleNode(h2))`

---

## 4. Key Invariants & Guarantees

1. **Type Safety by Construction:** An `AudioHoleNode` can never be populated by an `SDFNode`, and an `SDFHoleNode` can never be populated by an `AudioNode`. Cross-domain structural errors are impossible.
2. **Admissible Heuristics & Cost Regularization:** Candidates are ranked by:
   $$\text{Priority}(P) = \text{Score}(P) + \lambda_{\text{cost}} C_H(P)$$
   Partial programs receive an optimistic distance heuristic plus lower-bound hardware cost.
3. **Monotonic Budget Pruning:** Any program violating $C_H(P) > \text{max\_cost\_budget}$ is immediately eliminated, constraining search memory and preventing exponential explosion.
4. **Target Convergence Gate:** Baseline targets (`SawNode`, `SphereNode`, `TranslateNode(SphereNode)`, `LowPassNode(SawNode)`) are solved within fewer than 50 expansions, well below the 200-expansion threshold.
