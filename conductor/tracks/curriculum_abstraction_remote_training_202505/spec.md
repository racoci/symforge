# Specification: Adaptive Curriculum with Abstraction Discovery and Remote Decoupled Fine-Tuning

**Track ID:** `curriculum_abstraction_remote_training_202505`  
**Type:** Feature / Architecture  
**Status:** Approved  
**Date:** 2026-10-16  

---

## 1. Executive Summary & Problem Statement

The goal of **SymForge** is cost-aware differentiable program synthesis over typed grammars with dynamic structural growth and subexpression sharing. To establish a universal framework, the project is divided into three concrete, mutually reinforcing pillars:

1. **Pillar 1: Audio Synthesizer (1D Procedural DSP):** Synthesizing acoustic timbres and sound effects from known oscillators, filters, envelopes, and modulators.
2. **Pillar 2: SDF Synthesizer (3D Implicit Geometry):** Synthesizing geometric CAD and shapes from analytical primitives, spatial transforms, and CSG booleans.
3. **Pillar 3: Generalization (Universal Neurosymbolic Core):** The shared domain-agnostic engine that solves both problems using the exact same paradigm:
   - **Unified Primitive Composition:** Combining known primitives into higher-order primitives and complex functions to satisfy procedural target conditions.
   - **Differentiable Contrastive Perceptual Loss:** Evaluating perceptual similarity via contrastive objectives (InfoNCE) over latent acoustic/geometric embeddings, overcoming the failure modes of raw $L_1/L_2$ sample matching.
   - **Procedural Synthetic Data Generation:** Generating infinite curriculum tasks via Probabilistic Context-Free Grammars (PCFG) and Context-Sensitive / Attribute Grammars, guaranteeing numerical stability and physical validity without human data annotation.
   - **DreamCoder-Style Abstraction Discovery:** Identifying recurrent subgraphs across solved tasks and promoting them to first-class macro-primitives via Minimum Description Length (MDL) compression.
   - **Decoupled Remote Compute Engine:** Preserving local machine responsiveness by abstracting heavy training and LoRA updates behind cloud APIs (Modal, SageMaker) or deterministic zero-cost mocks.

```text
+-------------------------------------------------------------------------------------------------+
|                                    LOCAL CONTROL PLANE (CPU-ONLY)                               |
|                                                                                                 |
|   +-----------------------+       +------------------------+       +------------------------+   |
|   |  Adaptive Curriculum  | ----> |  Typed Beam / A* Synth | ----> |  Abstraction Discovery |   |
|   |  (PCFG Task Generator)|       |  (Dynamic AST/DAG)     |       |  (DreamCoder / MDL)    |   |
|   +-----------------------+       +-----------+------------+       +-----------+------------+   |
|               ^                               |                                |                |
|               | Task Feedback                 v Evaluation & Loss              v New Primitives |
|   +-----------+-----------+       +-----------+------------+       +-----------+------------+   |
|   |    Meta-Controller    | <---  | Contrastive Perceptual |       |    DSL Grammar Library |   |
|   | (Pacing & Regimes)    |       | Loss Engine (InfoNCE)  | ----> | (Typed Hole Expansions)|   |
|   +-----------+-----------+       +-----------+------------+       +------------------------+   |
|               |                               |                                                 |
+---------------|-------------------------------|-------------------------------------------------+
                | Replay Trajectories           | Download Weights
                v                               v
+-------------------------------------------------------------------------------------------------+
|                                 REMOTE COMPUTE & DISPATCH PLANE                                 |
|                                                                                                 |
|                          +------------------------------------------+                           |
|                          |     ComputeProvider (Async Protocol)     |                           |
|                          +--------------------+---------------------+                           |
|                                               |                                                 |
|                 +-----------------------------+-----------------------------+                   |
|                 |                             |                             |                   |
|                 v                             v                             v                   |
|   +---------------------------+ +---------------------------+ +---------------------------+     |
|   |    MockComputeProvider    | |    ModalComputeProvider   | |  SageMakerComputeProvider |     |
|   |    (Local 0-Cost / CI)    | |   (Serverless GPU A10G)   | |   (AWS Cloud Cluster)     |     |
|   +---------------------------+ +---------------------------+ +---------------------------+     |
+-------------------------------------------------------------------------------------------------+
```

```mermaid
graph TD
    subgraph LocalControlPlane [Local Control Plane - CPU Only]
        AC[Adaptive Curriculum & PCFG Task Sampler] -->|Task Specifications| SYN[Typed AST / DAG Synthesizer<br/>Beam / A* Search Engine]
        SYN -->|Candidate Programs| EVL[Contrastive Perceptual Loss Engine<br/>InfoNCE / Hardware Cost Model]
        EVL -->|Solved Trajectories| HARV[Trajectory Harvester & Replay Buffer]
        EVL -->|Accepted Programs| ABD[Abstraction Discovery<br/>Anti-Unification & MDL Compression]
        ABD -->|Invented Macro Primitives| DSL[DSL Grammar & Production Rules]
        DSL -->|Extended Type Rules| SYN
        
        MC[Meta-Controller<br/>Training Regime & Pacing] -->|Curriculum Pacing| AC
        MC -->|Search Budget & Temperature| SYN
        HARV -->|Batch Trajectories| MC
    end

    subgraph ComputeDispatchLayer [Decoupled Dispatch Layer]
        MC -->|Async Training Job Request| PROV{ComputeProvider Protocol}
        PROV -->|CI/Local Dev| MOCK[MockComputeProvider<br/>Instant Deterministic Emulation]
        PROV -->|Serverless Cloud GPU| MODAL[ModalComputeProvider<br/>Remote A10G / T4 Container]
        PROV -->|Enterprise Cloud| SMR[SageMakerComputeProvider<br/>AWS Managed GPU Training]
    end

    subgraph WeightSync [Adapter Synchronization]
        MOCK -.->|Simulated Adapter| REG[Specialized LoRA Registry]
        MODAL -.->|Trained PEFT Weights| REG
        SMR -.->|Trained PEFT Weights| REG
        REG -->|Lightweight Local CPU Inference| SYN
    end
```

---

## 2. The Three Project Pillars

```text
+---------------------------------------------------------------------------------------------------+
|                              MAPA DE ISOMORFISMO: ÁUDIO vs. SDF                                   |
|                                                                                                   |
|  Conceito                 Sintetizador de Áudio (1D DSP)         Sintetizador de SDF (3D CSG)     |
|  -----------------------  -------------------------------------  -------------------------------- |
|  Entrada Espacial         t in R (Tempo)                         p = (x,y,z) in R^3 (Posição)     |
|  Saída Contínua           s(t) in [-1, 1] (Pressão Acústica)     d(p) in R (Distância à Superf.)  |
|  Operação Booleana/Mix    Mix(s1, s2, w) / RingMod(s1, s2)       Union / SmoothUnion(d1, d2, k)   |
|  Transformação de Domínio Shift Temporal / Pitch Shift           Translação / Rotação / Escala    |
|  Extrator Perceptual      Multi-scale STFT / Mel Filterbanks     Surface Sampling / Raymarching   |
|  Espaço de Contraste      Latente Espectral (Audio Latent)       Latente Geométrico (Shape Latent)|
+---------------------------------------------------------------------------------------------------+
```

```mermaid
graph LR
    subgraph AudioDomain [Pilar 1: Sintetizador de Áudio]
        A_IN[Tempo: t] --> A_PROG[DAG de DSP: Osc, Filter, Env]
        A_PROG --> A_OUT[Sinal s_t]
        A_OUT --> A_SPEC[Multi-scale STFT]
        A_SPEC --> A_LAT[Latente Acústico z_audio]
    end

    subgraph SDFDomain [Pilar 2: Sintetizador de SDFs]
        S_IN[Posição: x, y, z] --> S_PROG[DAG de CSG: Sphere, Box, Union]
        S_PROG --> S_OUT[Campo d_p]
        S_OUT --> S_REND[Amostragem de Superfície]
        S_REND --> S_LAT[Latente Geométrico z_sdf]
    end

    A_LAT --> LOSS_UNI[Perda Contrastiva Perceptual InfoNCE]
    S_LAT --> LOSS_UNI
```

### 2.1 Pilar 1: Sintetizador de Áudio (DSP Procedural 1D)
- **Representação:** Grafo computacional acíclico (DAG) mapeando tempo discreto $t \in [0, T]$ para sinal de áudio $s(t) \in [-1, 1]$.
- **Primitivas Conhecidas:**
  - *Fontes Sonoras:* `Sine(freq, phase)`, `Saw(freq)`, `Square(freq, pulse_width)`, `Noise(seed)`.
  - *Envelopes & Moduladores:* `ADSR(attack, decay, sustain, release)`, `ExpDecay(half_life)`, `LFO(freq, waveform)`, `FM(carrier, modulator, depth)`.
  - *Filtros & Misturas:* `LowPass(cutoff, resonance)`, `HighPass(cutoff, resonance)`, `Gain(amplitude)`, `Mix(sig1, sig2, ratio)`.
- **Perda Perceptual de Contraste:** Extrai características de magnitude espectral via STFT em múltiplas resoluções temporais (ex: 512, 1024, 2048 samples) e calcula distância contrastiva InfoNCE contra ruídos e distratores harmônicos.

### 2.2 Pilar 2: Sintetizador de SDFs (Geometria Implícita 3D)
- **Representação:** Grafo computacional acíclico (DAG) mapeando coordenadas espaciais $\mathbf{p} \in \mathbb{R}^3$ para distância com sinal $d(\mathbf{p}) \in \mathbb{R}$.
- **Primitivas Conhecidas:**
  - *Formas Analíticas:* `Sphere(radius)`, `Box(half_extents)`, `Cylinder(radius, height)`, `Torus(r_major, r_minor)`, `Plane(normal, offset)`.
  - *Transformações Afins:* `Translate(sdf, offset)`, `Rotate(sdf, angles)`, `Scale(sdf, factor)`.
  - *Operadores Booleanos & Suavização:* `Union(sdf1, sdf2)`, `Difference(sdf1, sdf2)`, `Intersection(sdf1, sdf2)`, `SmoothUnion(sdf1, sdf2, k)`.
- **Perda Perceptual de Contraste:** Amostragem estratificada na isosuperfície ($\varepsilon$-boundary) e silhuetas renderizadas por raymarching com contraste InfoNCE contra formas geometricamente dissimilares.

### 2.3 Pilar 3: Generalização (Universal Neurosymbolic Core)
- **Gramática Tipada e Hole Expansion:** Expansão progressiva de buracos tipados `Hole[T]` em ASTs com conversão canônica para DAGs (deduplicação de subexpressões puras).
- **Gerador Procedural por PCFG:** Amostragem dirigida por regras probabilísticas com restrições contextuais (impedindo clipping em áudio e interseções vazias em SDFs).
- **Descoberta de Abstrações (DreamCoder):** Identificação de subgrafos isomórficos por anti-unificação e compressão por Descrição Mínima (MDL), promovendo-os a novas produções na gramática.
- **Treinamento Desacoplado:** Protocolo assíncrono para delegar ajuste contínuo e PEFT/LoRA para provedores remotos, mantendo zero consumo de GPU local.

---

## 3. Formulação Matemática da Perda Contrastiva e PCFG

### 3.1 Gerador de Dados Sintéticos via PCFG com Contexto
Dado um conjunto de símbolos não-terminais $V$, terminais $\Sigma$, regras de produção $R$ e símbolo inicial $S$, a Probabilistic Context-Free Grammar com Restrições Contextuais define:

$$\mathbb{P}(A \to \alpha \mid \text{context}) = \frac{\exp(\mathbf{w}_{A \to \alpha} \cdot \mathbf{\phi}(\text{context}))}{\sum_{\beta \in \text{Valid}(A, \text{context})} \exp(\mathbf{w}_{A \to \beta} \cdot \mathbf{\phi}(\text{context}))}$$

Onde $\text{Valid}(A, \text{context})$ aplica filtros de estabilidade do domínio:
- Em áudio: Filtros com pólos estáveis ($|z| < 1$), ganho global normalizado ($|s(t)| \le 1.0$).
- Em SDFs: Bounding boxes sobrepostas para operadores booleanos (evitando nós mortos).

### 3.2 Perda Contrastiva Perceptual InfoNCE
Para qualquer programa candidato $(G, \theta)$ que gera função $\hat{f}$ e alvo $f^*$:
1. O extrator perceptual diferenciável $\Phi(\cdot)$ computa o vetor latente normalizado $\mathbf{z}_{\hat{f}} = \Phi(\hat{f}) / \|\Phi(\hat{f})\|_2$.
2. O objetivo contrastivo contra $K$ programas negativos $\{f^-_j\}_{j=1}^K$ com temperatura $\tau$ é dado por:

$$\mathcal{L}_{\text{contrast}}(\hat{f}, f^*, \{f^-_j\}) = -\log \frac{\exp\left(\frac{\mathbf{z}_{\hat{f}} \cdot \mathbf{z}_{f^*}}{\tau}\right)}{\exp\left(\frac{\mathbf{z}_{\hat{f}} \cdot \mathbf{z}_{f^*}}{\tau}\right) + \sum_{j=1}^{K} \exp\left(\frac{\mathbf{z}_{\hat{f}} \cdot \mathbf{z}_{f^-_j}}{\tau}\right)}$$

3. A perda global incorpora o modelo de custo computacional de hardware compilado $C_H(G)$:

$$\mathcal{J}(G, \theta) = \lambda_{\text{contrast}} \mathcal{L}_{\text{contrast}} + \lambda_{\text{rec}} \mathcal{L}_{\text{rec}}(\hat{f}, f^*) + \lambda_{\text{cost}} C_H(G)$$

---

## 4. Subprocessos e Interfaces do Sistema

### 4.1 Protocolo do Provedor de Computação Desacoplada
```python
class ComputeProvider(Protocol):
    async def submit_training_job(
        self, 
        payload: TrainingPayload, 
        config: RemoteJobConfig
    ) -> str: ...
    
    async def poll_job_status(
        self, 
        job_id: str
    ) -> JobStatusResponse: ...
    
    async def fetch_job_artifacts(
        self, 
        job_id: str, 
        destination_dir: Path
    ) -> AdapterArtifact: ...
    
    async def cancel_job(
        self, 
        job_id: str
    ) -> bool: ...
```

Provedores concretos suportados:
1. `MockComputeProvider`: Execução em < 10ms local, zero custo, ideal para testes TDD e CI/CD.
2. `ModalComputeProvider`: Serverless GPU sob demanda (NVIDIA A10G/T4).
3. `SageMakerComputeProvider`: Treinamento corporativo AWS SageMaker com persistência em S3.

### 4.2 Descoberta de Abstrações via Anti-Unificação & MDL
- **Anti-unificação:** Determina o Generalizador Menos Geral (LGG) entre soluções que atingiram limiar de perda perceptual.
- **Métrica de Descrição Mínima (MDL):**
$$\Delta \text{MDL}(\alpha) = \text{Size}(\alpha) - \sum_{p \in \mathcal{D}_{\text{solved}}} (\text{Size}(p) - \text{Size}(p[\alpha])) < 0$$
- Abstrações aceitas são registradas na gramática tipada com novas probabilidades PCFG e assinaturas formais.

---

## 5. Diretrizes de Qualidade e Critérios de Aceite

1. **Test-Driven Development (TDD):** Cobertura de testes unitários $\ge 80\%$, cobrindo nós do AST, canonicalização de DAGs, gerador PCFG, perda contrastiva e provedores remotos.
2. **Isolamento de Recursos:** Nenhum processo de treinamento pesado bloqueia a CPU local; execução sob limites estritos de memória e timeouts explícitos em todas as chamadas assíncronas.
3. **Validação Simétrica:** Serialização e desserialização de programas preservam $f(x) \equiv f^{-1}(f(x))$.
