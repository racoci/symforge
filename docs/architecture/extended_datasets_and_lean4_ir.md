# Catálogo Expandido de Datasets Procedurais e Arquitetura do Transpilador Lean 4

Este documento expande o catálogo de dados do **SymForge** para englobar uma visão generalizada de "busca por lógica procedural paramétrica", incluindo síntese de texturas, hardware (superotimização) e áudio avançado. Além disso, detalha a arquitetura do **Transpilador de Representação Intermediária (IR) baseado em Lean 4**, que formaliza a gramática, garante invariantes e exporta o grafo para motores diferenciáveis de Machine Learning (como PyTorch/JAX).

---

## 1. Catálogo Expandido: Problemas de Lógica Procedural Parametrizável

A premissa do SymForge pode ser aplicada a qualquer domínio onde existe um **conjunto fechado de primitivas (módulos) e parâmetros livres** cujo objetivo é reconstruir um alvo complexo avaliado por uma função de custo/perda.

### 1.1 Domínio Gráfico: Texturas, Shaders e Materiais Procedurais
Neste domínio, o objetivo é combinar nós de texturização (ruído Perlin, Voronoi, gradientes, reflexão, mapas normais) para reproduzir uma textura de referência.
*   **MatSynth (Substance Designer):**
    *   *Conteúdo:* Milhares de grafos nodais extraídos do Adobe Substance 3D Designer, mapeando grafos procedurais complexos para PBR (Physically Based Rendering) materials.
    *   *Primitivas:* Nós de Ruído, Blur, Blend, Color Ramp, Warp.
    *   *Objetivo:* Descobrir o grafo que gera texturas de madeira, metal ou pedra a partir de uma foto de referência.
*   **Shadertoy / GLSL Fragment Shaders (Dataset Shadertoy-100k):**
    *   *Conteúdo:* Código-fonte de shaders procedurais que renderizam arte 2D/3D no navegador usando pura matemática.
    *   *Primitivas:* Funções matemáticas (`sin`, `fract`, `mod`), Raymarching loops, combinações de cores.
    *   *Objetivo:* Otimização de código gráfico, conversão de shaders ineficientes para versões otimizadas ou síntese de efeitos visuais a partir de um prompt/imagem.

### 1.2 Domínio Acústico Avançado: SFX e Síntese de Voz (Vocoders)
*   **TorchSynth Dataset:**
    *   *Conteúdo:* Banco de dados criado especificamente para a pesquisa de áudio, onde sons de sintetizadores modulares em PyTorch são combinados com seus parâmetros originais.
    *   *Primitivas:* VCO, VCA, VCF, Noise, ADSR.
    *   *Objetivo:* Síntese inversa de patches inteiros para SFX e instrumentos eletrônicos.
*   **DDSP (Differentiable Digital Signal Processing) Corpora (ex: VCTK, URMP):**
    *   *Conteúdo:* Gravações de voz e instrumentos acústicos reais.
    *   *Primitivas:* Osciladores harmônicos, filtros de ruído formatado, reverberadores impulsivos.
    *   *Objetivo:* Encontrar a modulação temporal exata das frequências fundamentais (f0) e contornos de loudness que, combinados com um sintetizador aditivo, recriam voz ou instrumentos reais.

### 1.3 Domínio de Hardware e Otimização de Compiladores (Superotimização)
Onde as "primitivas" são instruções de CPU/GPU e os "parâmetros" são registradores e offsets. O alvo é obter a mesma equivalência semântica de um código não-otimizado, mas com latência e uso de registradores minimizados.
*   **BHive (Block Basic Block Dataset):**
    *   *Conteúdo:* Centenas de milhares de blocos básicos da arquitetura x86-64 medidos em hardware real.
    *   *Primitivas:* Instruções Assembly (`ADD`, `MUL`, `LEA`, `MOV`, `XOR`, SIMD intrinsics).
    *   *Objetivo:* Superotimização. Descobrir uma combinação menor/mais rápida de instruções que computa a mesma saída (equivalência provada simbolicamente).
*   **TVM Tensor Programs / Halide Schedules:**
    *   *Conteúdo:* Espaços de escalonamento (schedules) para otimizar multiplicadores de tensores e redes neurais.
    *   *Primitivas:* Operações de Loop Tiling, Unrolling, Vectorization, Caching.
    *   *Objetivo:* Descobrir o schedule ótimo que minimiza o tempo de execução numa GPU específica, preservando o resultado da matriz matemática.

### 1.4 Outros Domínios Análogos
*   **Química Computacional / Retrossíntese (USPTO Dataset):** Primitivas são reações químicas básicas; parâmetros são condições de temperatura/catalisadores. Objetivo: obter a molécula alvo (referência).
*   **Design de UI / Layout (RICO Dataset):** Primitivas são Flexboxes, Containers, botões e textos. Parâmetros são paddings e cores. Objetivo: recriar um screenshot de um app com uma árvore UI procedural.

---

## 2. A Arquitetura do Transpilador de IR com Lean 4

Para que o **SymForge** lide unificadamente com todos esses domínios (Áudio, SDFs, Shaders, Hardware), precisamos de uma Representação Intermediária (IR) rigorosa. **Lean 4** é uma linguagem funcional e um provador de teoremas baseado no Cálculo de Construções Indutivas. 

Ele é perfeito para isso porque permite definir a gramática (tipos) e declarar explicitamente os parâmetros livres, garantindo matematicamente a validade da estrutura (ex: um filtro precisa de uma entrada de áudio, um operador CSG precisa de duas geometrias).

### 2.1 Por que Lean 4?
1. **Tipos Dependentes e Garantias Físicas:** No Lean, podemos definir o tipo `Filter (cutoff : Float) (h : cutoff > 0)`, garantindo a estabilidade em tempo de compilação simbólica.
2. **Meta-programação Integrada:** Podemos escrever scripts dentro do próprio Lean que inspecionam o AST e geram código Python/PyTorch otimizado (Transpilação).
3. **Equivalência Simbólica:** Para o caso de *superotimização de hardware* e *CSG booleano*, o Lean pode provar que o programa A e o programa B produzem o mesmo valor, antes mesmo de enviar ao motor de ML.

### 2.2 Estrutura da Representação Intermediária (Lean 4 AST)

```lean
-- Exemplo de definição de IR Unificada em Lean 4 para o Domínio de Áudio

-- 1. Os "parâmetros livres" (que o ML terá que otimizar por gradiente)
structure FreeParam where
  id : String
  min : Float
  max : Float

-- 2. A Gramática Procedural (Módulos/Primitivas)
inductive AudioNode : Type where
  | sine (freq : FreeParam) : AudioNode
  | noise (seed : Int) : AudioNode
  | lowpass (cutoff : FreeParam) (res : FreeParam) (input : AudioNode) : AudioNode
  | mix (ratio : FreeParam) (input1 : AudioNode) (input2 : AudioNode) : AudioNode

-- 3. No caso de Geometria (SDF)
inductive SDFNode : Type where
  | sphere (radius : FreeParam) : SDFNode
  | box (width height depth : FreeParam) : SDFNode
  | union (smoothness : FreeParam) (a b : SDFNode) : SDFNode
```

### 2.3 O Pipeline de Transpilação e Integração com Machine Learning

Para que essa IR seja processada em velocidades altíssimas para otimização de gradientes, o Lean 4 atua como o **Front-end Simbólico**, enquanto o PyTorch/JAX/CUDA atua como o **Back-end de Execução**.

#### Etapa A: Frontend / Ingestão
1. Um parser converte os arquivos brutos (ex: patches do Dexed, código GLSL do Shadertoy, instruções x86) para a representação indutiva (AST) no Lean 4.
2. O Lean 4 valida os tipos, resolve referências de DAG (deduplicação) e identifica todos os `FreeParam` (parâmetros numéricos que a rede neural deve descobrir).

#### Etapa B: Transpilação (Lean 4 -> ML Backend)
1. Usando o poder de metaprogramação do Lean (`Lean.Elab`), o transpilador percorre a AST e emite um grafo de computação.
2. **Exportação:** Ele gera um código PyTorch/JAX equivalente ou um grafo ONNX.
   - *Exemplo:* Um nó `AudioNode.mix` em Lean vira `torch.lerp(input1, input2, weight)` em Python.
   - O Lean agrupa todos os `FreeParam` em um único vetor contínuo de pesos $\Theta \in \mathbb{R}^K$.

#### Etapa C: Otimização Contínua Diferenciável (Back-end)
1. O motor de Machine Learning (Python/C++) recebe o código gerado e o vetor $\Theta$.
2. Ele renderiza a saída da função parametrizada procedural $f_\Theta(x)$ contra os dados de referência (o dataset OOD, o som alvo, a forma 3D).
3. Calcula a **Perda Perceptual Baseada em Contraste (InfoNCE)** usando gradientes (Autograd) ou métodos Black-Box (CMA-ES, Evolution Strategies para funções não-diferenciáveis).
4. Otimiza os valores dos `FreeParam` e devolve a configuração ideal para o Lean 4.

### 2.4 Vantagens dessa Abordagem

1. **Separação Limpa:** O Lean 4 foca em manter as regras gramaticais e inventar novas abstrações tipadas. O PyTorch foca unicamente na otimização numérica massiva de matrizes.
2. **Discovery de Abstrações Robusto:** Quando o DreamCoder atua sobre a IR do Lean 4, as novas macro-primitivas inventadas (ex: uma sequência "Filtro + LFO + Mix") recebem imediatamente garantia de tipo e são incorporadas à linguagem.
3. **Escalabilidade Universal:** Se quisermos adicionar otimização de instruções Assembly no futuro, basta criar o arquivo `HardwareNode.lean`, compilar a semântica, e o motor de otimização de ML e busca em árvore funcionará idêntico ao de áudio ou SDF.

---

## 3. O Que Precisa Ser Feito (Roadmap de Implementação Lean 4)

Para incorporar o Lean 4 à arquitetura local, os seguintes passos técnicos (Tasks) devem ser incluídos no escopo do SymForge:

1. **Configurar o Ambiente Lean 4 (Local Control Plane):**
   - Integrar o `lake` (Lean Package Manager) ao repositório.
   - Criar o pacote `SymForge.IR` contendo os Tipos Indutivos base de SDF e DSP.

2. **Implementar Parsers de Ingestão (Lean):**
   - Criar importadores para converter DSLs clássicos (arquivos CSG, patches FM JSON) em árvores de tipos do Lean 4.

3. **Escrever a "Meta-Macro" Transpiladora em Lean:**
   - Usar as APIs de macros de Lean para visitar uma instância `AudioNode` ou `SDFNode` e emitir (como String ou JSON) a classe PyTorch `nn.Module` / `JAX` equivalente.
   - Extrair automaticamente o catálogo de tensores/parâmetros livres para otimização.

4. **Binding Python-Lean:**
   - Desenvolver o pipeline (por via de FFI - Foreign Function Interface, C-bindings, ou IPC/JSON) que permita ao orquestrador em Python submeter gramáticas parciais ao Lean para validação, e receber do Lean as expressões Python otimizáveis prontas.