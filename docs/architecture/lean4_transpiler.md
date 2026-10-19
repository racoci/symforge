# Lean 4 Metaprogramming Transpiler & Differentiable Graph Engine

## 1. Architectural Mission

The **Lean 4 Metaprogramming Transpiler** establishes the formal bridge between:
- **Symbolic Front-End:** High-level inductive grammar trees (`SymforgeIr.Types.AudioNode`, `SDFNode`) with provable well-formedness and type invariants.
- **Differentiable Back-End:** PyTorch computational graphs (`DifferentiableAudioModule`, `DifferentiableSDFModule`) exposing continuous parameters $\Theta \in \mathbb{R}^K$ to automatic differentiation and gradient-based perceptual alignment.

---

## 2. Compilation Flow & Structural Topology

```text
+-----------------------+              +-----------------------+
|  Lean 4 Inductive AST | === Elab ==> | PyTorch nn.Module     |
| (AudioNode / SDFNode) |              | (Differentiable Graph)|
+-----------+-----------+              +-----------+-----------+
            |                                      |
            v                                      v
+-----------------------+              +-----------------------+
| FreeParam Vector      |              | Continuous Tensor     |
| [p_1, ..., p_K]       | ===========> | Theta in R^K (Params) |
+-----------------------+              +-----------+-----------+
                                                   |
                                                   v Forward(x)
                                       +-----------------------+
                                       | Output Signal / Field |
                                       | y_synth = f_Theta(x)  |
                                       +-----------+-----------+
                                                   |
                                                   v Backward()
                                       +-----------------------+
                                       | Non-Zero Gradients    |
                                       | dL / dTheta in R^K    |
                                       +-----------------------+
```

```mermaid
graph TD
    subgraph Lean4Symbolic [Lean 4 Symbolic Front-End]
        A[Inductive AST: AudioNode / SDFNode] -->|collectParams| B[FreeParam Manifest: [p_1 .. p_K]]
        A -->|toPyTorchExpr| C[Differentiable Expression AST]
    end

    subgraph PyTorchEngine [PyTorch Differentiable Back-End]
        B --> D[nn.Parameter Theta in R^K]
        C --> E[Forward Computation Graph f_Theta]
        D --> E
        E -->|Inputs: t or p| F[Synthesized Signal y_synth]
        F --> G[Perceptual Contrastive Loss L]
        G -->|Autograd .backward| H[Gradients dL / dTheta]
        H -->|Adam / L-BFGS| D
    end
```

---

## 3. Implementation Details

### 3.1 Parameter Vectorization
All continuous degrees of freedom in an AST are aggregated into `Theta = nn.Parameter(torch.tensor(init_vals))`. Index mappings preserve parameter identities across macro substitutions and structural tree mutations.

### 3.2 1D Audio Differentiable Synthesis (`DifferentiableAudioModule`)
- **Oscillators:**
  - `Sine`: $\sin(2\pi f t + \phi)$
  - `Sawtooth`: $2 (f t - \lfloor f t + 0.5 \rfloor)$
  - `Square`: $\tanh(15 \sin(2\pi f t))$
  - `FM Synthesis`: $\sin(2\pi f_c t + I \sin(2\pi f_m t))$
- **Filters & Envelopes:**
  - Differentiable 1-pole filter simulation with frequency-dependent smoothing $\alpha = \frac{\Delta t}{\tau + \Delta t}$.
  - Piecewise continuous ADSR envelopes.

### 3.3 3D SDF Differentiable Fields (`DifferentiableSDFModule`)
- Evaluates signed distances for coordinates $p \in \mathbb{R}^{N \times 3}$.
- Implements exact distance equations for `Sphere`, `Box`, `Cylinder`, and `Torus`.
- Smooth polynomial CSG union with parameter $k$:
  $$\text{smin}(d_1, d_2, k) = \text{lerp}(d_2, d_1, h) - k \cdot h(1 - h), \quad h = \text{clamp}\left(0.5 + 0.5 \frac{d_2 - d_1}{k}, 0, 1\right)$$
- Guarantees $C^1$ continuity and non-vanishing gradients across CSG boundaries.
