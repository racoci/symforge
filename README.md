# Proposta de pesquisa: síntese simbólica diferenciável de ASTs e DAGs tipados, com expansão dinâmica e otimização por custo

Título provisório em inglês: Cost-Aware Differentiable Grammar-Guided Program Synthesis with Dynamic AST and DAG Construction

## 1. Resumo

Propõe-se desenvolver um framework de síntese de programas que combine busca simbólica, otimização diferenciável e aprendizado neural para descobrir programas executáveis a partir de exemplos de entrada e saída. O programa deve ser composto a partir de uma gramática conhecida, cujas produções, tipos, restrições semânticas e primitivas já estejam especificados. A estrutura do programa, entretanto, é desconhecida e pode crescer durante a busca.

O problema será investigado inicialmente no domínio das funções de distância assinada (Signed Distance Functions, SDFs), em que primitivas geométricas e operadores de composição geram funções implícitas avaliadas repetidamente por um renderizador conhecido. O objetivo é descobrir estruturas simbólicas capazes de reproduzir geometrias ou imagens observadas, minimizando simultaneamente o erro de reconstrução e o custo computacional.

A proposta combina três linhas existentes. A primeira é a síntese diferenciável sobre gramáticas, representada por dPads, que aprende distribuições sobre derivações gramaticais. A segunda é a busca simbólica guiada por relaxações neurais diferenciáveis, representada por NEAR. A terceira compreende métodos de seleção diferenciável de estruturas, execução de programas de comprimento variável e poda de componentes, como mecanismos encontrados em NLI, DARTS e regularização \\(L_0\\).&#x20;

[image](https://www.google.com/s2/favicons?domain=https://papers.neurips.cc\&sz=32)

NeurIPS Papers

+4



A contribuição pretendida não consiste simplesmente em combinar esses métodos, mas em investigar uma arquitetura que una expansão gramatical tipada, crescimento dinâmico de programas, compartilhamento de subgrafos, otimização dos parâmetros contínuos e avaliação explícita do custo de execução. A validade dos programas produzidos deverá ser preservada mesmo quando relaxações diferenciáveis forem utilizadas durante o treinamento.

## 2. Problema de pesquisa

Dado um conjunto de observações

\\[ \mathcal D=\\{(x_i,y_i)\\}\_{i=1}^{N}, \\]

pretende-se descobrir um programa \\(p\\) constituído por uma estrutura simbólica \\(G\\) e parâmetros contínuos \\(\theta\\):

\\[ p=(G,\theta). \\]

O programa deve satisfazer a gramática \\(\Gamma\\), respeitar as regras de tipos e as restrições semânticas do domínio e produzir saídas compatíveis com as observações.

Formalmente, o objetivo principal será:

\\[ \begin{aligned} \min\_{G,\theta}\quad & C_H(G,\theta)\\\ \text{sujeito a}\quad & L(\operatorname{Exec}(G,\theta),\mathcal D) \leq \varepsilon,\\\ & G\in\mathcal L(\Gamma),\\\ & \operatorname{Valid}(G,\theta)=\mathrm{true}. \end{aligned} \\]

Nesse problema, \\(\mathcal L(\Gamma)\\) é o conjunto de programas gerados pela gramática, \\(L\\) mede o erro funcional ou visual e \\(C_H\\) representa o custo de execução no ambiente-alvo \\(H\\).

O custo pode começar como uma contagem ponderada de operações e posteriormente incorporar a latência do código compilado. Essa distinção é necessária porque uma expressão com menos nós não é necessariamente mais rápida em hardware real.

O problema contém três decisões interdependentes:

- Decisão estrutural: quais produções aplicar e como conectar os operadores.
- Decisão paramétrica: quais valores atribuir aos parâmetros de cada nó.
- Decisão de otimização: quais estruturas equivalentes ou aproximadamente equivalentes devem ser preferidas segundo o custo.

A dificuldade central é que a primeira decisão é discreta, a segunda pode ser diferenciável e a terceira depende da execução efetiva do programa. Portanto, não se deve presumir que um único procedimento de backpropagation convencional seja suficiente para resolver todo o problema.

## 3. Hipótese central e questões de pesquisa

A hipótese central é que uma busca gramatical diferenciável com expansão dinâmica, refinamento paramétrico por gradientes e avaliação explícita de custo pode descobrir programas simbólicos mais eficientemente do que métodos isolados de busca discreta ou relaxação contínua.

Essa hipótese será investigada por cinco questões.

| Questão                      | Pergunta experimental                                                                                                |
| ---------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| RQ1 — Busca gramatical       | Distribuições aprendidas sobre produções reduzem o número de candidatos necessários para encontrar soluções válidas? |
| RQ2 — Otimização híbrida     | Combinar busca estrutural e gradientes para parâmetros melhora a relação entre fidelidade e custo?                   |
| RQ3 — Crescimento dinâmico   | A expansão progressiva supera superestruturas de tamanho fixo na síntese de programas de diferentes complexidades?   |
| RQ4 — DAGs                   | O compartilhamento explícito de subexpressões reduz o custo executável sem degradar a fidelidade?                    |
| RQ5 — Aprendizado amortizado | O treinamento com programas sintéticos melhora a síntese de estruturas inéditas e reduz o custo de busca por tarefa? |

As hipóteses serão consideradas sustentadas apenas se os resultados forem obtidos em dados de teste separados por estrutura dos programas geradores, e não somente por parâmetros aleatórios de programas que a rede já viu.

## 4. Estado da arte e posicionamento da proposta

Os trabalhos selecionados resolvem partes diferentes do problema. A proposta parte deles sem presumir que algum já tenha a arquitetura completa desejada.

| Trabalho                                                                                                                                                   | Contribuição aproveitada                                                                         | Lacuna em relação ao projeto                                                                                                                    |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| [dPads](https://github.com/RU-Automated-Reasoning-Group/dPads?utm_source=chatgpt.com)                                                                      | Distribuições diferenciáveis sobre derivações de gramáticas livres de contexto                   | Precisamos estender a abordagem a programas geométricos e investigar explicitamente DAGs com compartilhamento e crescimento dinâmico.           |
| [NEAR](https://github.com/trishullab/near?utm_source=chatgpt.com)                                                                                          | Busca sobre programas simbólicos com relaxações neurais como heurísticas                         | Precisamos adaptar o custo e a função objetivo para reconstrução geométrica e custo de execução.                                                |
| [NLI, ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/c9cde817d04811ba28e44071bd9f76a5-Abstract-Conference.html?utm_source=chatgpt.com) | Linguagem de operações simbólicas e execução diferenciável de sequências de comprimento variável | A gramática do nosso domínio é conhecida; precisamos preservar seus tipos, assinaturas e restrições, além de permitir DAGs arbitrários tipados. |
| [DreamCoder](https://people.csail.mit.edu/asolar/papers/EllisWNSMHCST21.pdf?utm_source=chatgpt.com)                                                        | Síntese amortizada, política neural de busca e aprendizado de abstrações reutilizáveis           | A otimização por gradientes e o custo executável devem ser integrados explicitamente ao objetivo.                                               |
| [DARTS](https://arxiv.org/abs/1806.09055?utm_source=chatgpt.com) e [DrNAS](https://github.com/xiangning-chen/DrNAS?utm_source=chatgpt.com)                 | Relaxação de escolhas discretas e aprendizado de distribuições sobre estruturas                  | Suas superestruturas convencionais não resolvem sozinhas o crescimento ilimitado, a tipagem semântica e a criação de subgrafos.                 |
| [CSGNet](https://github.com/Hippogriff/CSGNet?utm_source=chatgpt.com)                                                                                      | Recuperação de programas geométricos simbólicos a partir de formas                               | Constitui um baseline direto de síntese de geometria, mas não o sistema geral de otimização diferenciável e custo-consciente.                   |

dPads formula a busca por arquiteturas de programas como aprendizado de distribuições sobre derivações de uma gramática livre de contexto. Seus resultados demonstram a viabilidade da relaxação diferenciável de escolhas gramaticais, embora a busca continue dependente de uma representação de derivações e de um espaço de busca configurado.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://papers.neurips.cc\&sz=32)

NeurIPS Papers

+1



NEAR oferece uma complementaridade importante: transforma a síntese em busca sobre arquiteturas parciais e usa programas neurossimbólicos diferenciáveis para estimar o custo das possibilidades de conclusão. Isso fornece uma ponte entre gradientes e busca simbólica explícita. Sua noção de heurística aproximadamente admissível, contudo, não deve ser confundida com uma garantia universal de otimalidade quando a heurística aprendida não satisfaz as condições necessárias.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://arxiv.org\&sz=32)

arXiv

+1



O NLI é um antecedente recente para execução diferenciável de sequências de operações de comprimento variável. Diferentemente dele, esta proposta parte de uma biblioteca e gramática explicitamente declaradas, porque deseja otimizar a composição de componentes conhecidos, e não aprender uma linguagem nova do zero.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://proceedings.iclr.cc\&sz=32)

ICLR Proceedings



Já o DreamCoder demonstra que uma política de busca e uma biblioteca de abstrações podem ser aprendidas conjuntamente a partir de muitas tarefas de síntese. Isso justifica incluir aprendizado amortizado e, como extensão, descoberta de abstrações reutilizáveis no framework.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://people.csail.mit.edu\&sz=32)

people.csail.mit.edu

+1



### Lacuna específica

A lacuna a investigar é a combinação, num único processo de síntese:

1. Distribuições aprendidas sobre produções de uma gramática tipada.
2. Crescimento adaptativo das estruturas candidatas.
3. Refinamento contínuo dos parâmetros de programas simbólicos explícitos.
4. Construção de DAGs com compartilhamento seguro de subexpressões.
5. Função de custo que considere o programa executável final.
6. Treinamento amortizado em exemplos sintéticos para generalizar a tarefas inéditas.

Não afirmo que nenhuma variante dessa combinação tenha sido publicada. A contribuição científica deverá ser delimitada pela comparação experimental e por uma revisão sistemática dos sistemas mais próximos.

## 5. Representação da linguagem e dos programas

A primeira entrega técnica será uma representação intermediária que separe o conhecimento fixo do espaço de síntese.

### 5.1. Gramática tipada de SDFs

A gramática inicial será pequena o suficiente para permitir experimentos controlados, mas suficientemente expressiva para produzir formas diversas.

Uma versão ilustrativa é:

\\[ \begin{aligned} \mathrm{SDF} \\;::={}& \mathrm{Sphere}(\theta)\\\ &\mid \mathrm{Box}(\theta)\\\ &\mid \mathrm{Cylinder}(\theta)\\\ &\mid \mathrm{Union}(\mathrm{SDF},\mathrm{SDF})\\\ &\mid \mathrm{Difference}(\mathrm{SDF},\mathrm{SDF})\\\ &\mid \mathrm{Intersection}(\mathrm{SDF},\mathrm{SDF})\\\ &\mid \mathrm{Transform}(\mathrm{SDF},\theta)\\\ &\mid \mathrm{SmoothUnion}(\mathrm{SDF},\mathrm{SDF},\theta). \end{aligned} \\]

Os parâmetros \\(\theta\\) terão tipos, intervalos, unidades e restrições. Por exemplo, raios e dimensões devem ser positivos; transformações devem obedecer à convenção geométrica escolhida; operadores devem receber os tipos apropriados.

As implementações dessas primitivas serão congeladas durante os primeiros experimentos. O sintetizador poderá selecionar e compor as operações, mas não modificar arbitrariamente sua semântica.

Também será necessário diferenciar duas noções: a validade do programa dentro da DSL e a exatidão geométrica do campo produzido. Uma composição booleana pode ser válida sintaticamente e ainda assim não representar a distância euclidiana exata à superfície final. O executor e o avaliador devem conhecer essa diferença.

### 5.2. AST versus DAG

A representação inicial será uma AST, apropriada para gramáticas recursivas e programas sem compartilhamento. Posteriormente, ela será ampliada para DAGs, nos quais um mesmo subprograma puro pode alimentar múltiplos consumidores.

Por exemplo, uma transformação espacial pode ser compartilhada entre diferentes expressões, em vez de ser recalculada em cada ramo.

Esse compartilhamento exige regras explícitas de identidade dos nós, escopo de parâmetros, tipos e semântica de avaliação. Como os primeiros programas SDF são funções puras das coordenadas e dos parâmetros, o compartilhamento é mais simples do que em linguagens com estado. Ainda assim, parâmetros mutáveis, ruído aleatório e efeitos terão de ser tratados com cuidado quando o framework for generalizado.

Para manter os programas válidos, o DAG será construído incrementalmente, respeitando uma ordem topológica. Não recomendo inicialmente uma matriz global de adjacência irrestrita: ela criaria candidatos incompatíveis com tipos e assinaturas, além de exigir mecanismos adicionais de controle de ciclos.

### 5.3. Programa parcial e nó desconhecido

A síntese parte de programas incompletos, representados por buracos tipados:

\\[ G_0 = \mathrm{Hole}[\mathrm{SDF}]. \\]

Cada buraco identifica uma posição que precisa receber uma produção gramatical válida. Uma expansão pode substituir o buraco por uma primitiva completa ou por um operador com novos buracos em seus argumentos.

Por exemplo:

\\[ \mathrm{Hole}[\mathrm{SDF}] \\;\longrightarrow\\; \mathrm{Union} \bigl( \mathrm{Hole}[\mathrm{SDF}], \mathrm{Hole}[\mathrm{SDF}] \bigr). \\]

O processo continua até que os buracos sejam preenchidos, o programa satisfaça as condições de qualidade ou o orçamento de busca seja esgotado.

A busca será progressiva e adaptativa. Isso não significa que o sistema explorará uma quantidade matematicamente ilimitada de nós numa única execução: haverá limites de recursos e critérios de parada, mas a profundidade máxima não precisa ser um hiperparâmetro rígido compartilhado por todos os problemas.

## 6. Método proposto: síntese com expansão gramatical diferenciável e busca custo-consciente

O sistema terá quatro mecanismos centrais, integrados num ciclo comum.

### 6.1. Mecanismo A — Distribuições sobre produções gramaticais

A partir de dPads, será implementado um mecanismo que atribui parâmetros treináveis às produções permitidas em cada tipo de buraco.

Para um contexto gramatical \\(h\\), definimos:

\\[ q\_\phi(r\mid h)= \frac{ \exp(\alpha\_{\phi,h,r}/\tau) }{ \sum\_{r'\in\mathcal R(h)} \exp(\alpha\_{\phi,h,r'}/\tau) }, \\]

onde \\(\mathcal R(h)\\) contém apenas as regras compatíveis com o tipo e as restrições do contexto, \\(\alpha\_{\phi,h,r}\\) são os parâmetros aprendidos e \\(\tau\\) controla a suavidade da distribuição.

Durante o treinamento, a distribuição permite atribuir gradientes às preferências por regras. Durante a busca final, o sistema seleciona produções concretas e constrói programas simbólicos convencionais.

Eu não trataria a saída suave da distribuição como se fosse um programa válido. Se `Union` e `Difference` forem combinados por uma mistura numérica de suas saídas, essa mistura terá uma semântica diferente da escolha entre os dois operadores. Portanto, a relaxação servirá para orientar a busca e treinar os parâmetros da política, enquanto o candidato final continuará sendo uma derivação discreta da gramática.

Para evitar explosão combinatória, a expansão será incremental. Em cada estágio, o sistema poderá considerar apenas as regras mais promissoras segundo a distribuição aprendida, mas deverá preservar alguma exploração de alternativas menos prováveis para reduzir o risco de eliminar prematuramente uma solução boa.

### 6.2. Mecanismo B — Otimização diferenciável dos parâmetros de cada candidato

Para cada estrutura concreta \\(G\\), o sistema ajustará os parâmetros \\(\theta\\) por otimização contínua.

No primeiro benchmark, a entrada será um conjunto de consultas espaciais e os valores SDF de referência. A função de erro inicial será:

\\[ L\_{\mathrm{field}}(G,\theta)= \frac{1}{N} \sum\_{i=1}^{N} \rho\\!\left( f\_{G,\theta}(x_i)-d_i \right), \\]

onde \\(\rho\\) pode ser uma função robusta como Huber e \\(d_i\\) são as distâncias de referência.

Quando houver geometria de superfície, também poderão ser utilizadas perdas de superfície e de normais. Quando a única supervisão disponível for uma imagem, a perda será calculada sobre as observações renderizadas.

O otimizador deve ser configurável por tipo de parâmetro. Gradientes analíticos ou automáticos são apropriados para parâmetros contínuos diferenciáveis; parâmetros discretos ou não diferenciáveis precisam de mecanismos de busca ou estimadores específicos.

A infraestrutura de renderização permanecerá fixa, mas seu caminho de derivadas precisará ser validado. Métodos como DIST demonstram que o sphere tracing pode ser integrado a um processo de otimização diferenciável a partir de observações 2D.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://openaccess.thecvf.com\&sz=32)

The CVF Open Access

+1



### 6.3. Mecanismo C — Busca orientada por heurísticas neurais

A estrutura gramatical será explorada por busca sobre programas parciais. Para cada estado \\(u\\), o sistema terá de decidir quais expansões merecem receber mais orçamento computacional.

O NEAR oferece um modelo útil: uma relaxação neural pode estimar quanto custa completar um programa parcial, levando em conta tanto o custo estrutural quanto o erro do candidato final. Essa estimativa serve para ordenar os estados da busca, em lugar de exigir a enumeração exaustiva de todas as derivações.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://arxiv.org\&sz=32)

arXiv

+1



A proposta usará inicialmente uma variante de beam search guiada por essa estimativa, comparada posteriormente com A\* e busca em árvore de Monte Carlo.

A escolha depende da natureza da estimativa de custo. Quando houver um limite inferior comprovado para o custo restante, podemos explorar algoritmos com garantias formais. Se a heurística for apenas uma predição neural, ela será usada como prioridade de exploração, sem alegar otimalidade global.

### 6.4. Mecanismo D — Criação, poda e compartilhamento de subgrafos

O quarto mecanismo trata a diferença entre uma árvore e um grafo computacional.

A geração começa pela expansão de nós tipados. Em seguida, podem ocorrer três operações estruturais:

1. Extensão: criar novos nós e conectá-los a buracos existentes.
2. Poda: remover ramos que não contribuam suficientemente para a qualidade.
3. Compartilhamento: substituir subexpressões repetidas por referências ao mesmo nó, quando a semântica e os parâmetros permitirem.

Gates diferenciáveis podem ser usados para priorizar a remoção de componentes, inspirados em regularização \\(L_0\\) com variáveis aleatórias Hard Concrete. No entanto, a criação de novos nós continuará sendo uma decisão explícita de busca.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://arxiv.org\&sz=32)

arXiv

+1



O compartilhamento também será avaliado com cautela: ele reduz operações redundantes quando o subgrafo é realmente reutilizável, mas pode aumentar pressão sobre registradores, memória ou agendamento. Portanto, o benefício precisa ser medido no programa executável, e não inferido exclusivamente pela redução da quantidade de nós.

## 7. Função objetivo e treinamento conjunto

A formulação principal será uma otimização sob restrições:

\\[ \begin{aligned} \min\_{G,\theta}\quad & C_H(G,\theta)\\\ \text{sujeito a}\quad & L\_{\mathrm{geom}}\leq\varepsilon_g,\\\ & L\_{\mathrm{render}}\leq\varepsilon_r,\\\ & G\in\mathcal L(\Gamma). \end{aligned} \\]

Durante o treinamento, essa formulação será aproximada por uma função objetivo diferenciável ou por uma recompensa:

\\[ \mathcal J(G,\theta)= L\_{\mathrm{geom}} +\lambda_r L\_{\mathrm{render}} +\lambda_c C_H(G,\theta) +\lambda_s S(G), \\]

onde \\(S(G)\\) representa penalidades estruturais, como complexidade excessiva ou redundância.

O termo de fidelidade terá prioridade: programas que não satisfaçam o limite de erro não devem ser considerados soluções competitivas apenas por serem baratos. Para a avaliação final, apresentaremos fronteiras de Pareto entre qualidade, custo e tamanho simbólico, em vez de depender de um único conjunto de coeficientes.

O treinamento terá três níveis:

| Nível                     | Mecanismo de aprendizagem                                             |
| ------------------------- | --------------------------------------------------------------------- |
| Parâmetros contínuos      | Backpropagation e otimizadores numéricos                              |
| Preferências gramaticais  | Gradientes de distribuições sobre produções                           |
| Estrutura discreta e DAGs | Busca guiada, expansões e edições simbólicas; relaxações quando úteis |

O modelo também poderá aprender a reutilizar conhecimento entre tarefas. Ao receber muitos exemplos sintéticos, deve reduzir gradualmente o esforço necessário para encontrar composições eficazes em novas tarefas, como ocorre com o aprendizado amortizado de políticas de busca no DreamCoder.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://people.csail.mit.edu\&sz=32)

People CSail

+1



O ponto de pesquisa não será exigir que tudo seja diferenciável. Será determinar em quais partes do problema a diferenciação oferece ganho efetivo e em quais partes uma busca simbólica explícita continua sendo superior.

## 8. Geração de dados e protocolo de treinamento

O benchmark inicial será produzido por um gerador de programas conforme a gramática SDF.

Para cada programa gerado:

1. Serão amostrados uma estrutura e parâmetros válidos.
2. O executor de referência calculará as saídas.
3. Serão geradas consultas SDF, observações de superfície e imagens quando aplicáveis.
4. Será medida a complexidade estrutural e, quando possível, o custo do programa compilado.
5. Serão criadas tarefas inversas que ocultam o programa e fornecem somente suas observações.

O conjunto deve variar independentemente a profundidade da árvore, o número de primitivas, as classes de operadores, as distribuições dos parâmetros e o número de consultas de treinamento. Isso evita que a rede aprenda apenas correlações triviais entre a quantidade de nós e o formato da saída.

O protocolo terá quatro divisões:

- Treinamento: estruturas e parâmetros sorteados dentro da gramática conhecida.
- Teste paramétrico: estruturas vistas durante o treinamento, mas com novos parâmetros.
- Teste composicional: novas combinações de operadores e subestruturas.
- Teste de profundidade: programas mais complexos do que os predominantes no treinamento, respeitando o orçamento computacional.

A divisão será feita por famílias de programas geradores, não só por instâncias. Do contrário, versões parametrizadas da mesma estrutura podem aparecer nas duas partições e superestimar a generalização.

Como segunda fase, serão usados programas CSG e conjuntos de SDFs ou geometrias de referência para testar a transferência para formas menos artificiais. O CSGNet é um baseline particularmente importante porque já investiga a indução de programas de geometria a partir de formas e o uso de policy gradients quando não existem rótulos de programa.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://openaccess.thecvf.com\&sz=32)

The CVF Open Access

+1



## 9. Baselines e desenho experimental

A avaliação precisa separar o ganho decorrente da representação gramatical, do ganho decorrente dos gradientes e do ganho decorrente da política neural de busca.

### 9.1. Métodos de comparação

Busca enumerativa tipada

Enumerar programas por custo estrutural crescente e ajustar seus parâmetros. Esse é o baseline para medir o benefício de explorar o espaço de forma não exaustiva.

dPads adaptado

Aprender distribuições sobre regras gramaticais, sem acrescentar inicialmente compartilhamento de subgrafos ou uma política separada de edição.

NEAR adaptado

Usar busca simbólica com heurísticas neurais e otimização de parâmetros nos candidatos completos.

Relaxação de superestrutura

Usar uma topologia candidata limitada, inspirada em DARTS ou técnicas relacionadas, e comparar com a expansão dinâmica da proposta.

Método completo proposto

Busca gramatical diferenciável, refinamento paramétrico, crescimento dinâmico, compartilhamento seguro e seleção pelo custo executável.

Não é necessário reproduzir integralmente cada método original se a adaptação para SDFs exigir uma implementação nova. É necessário, porém, documentar quais componentes foram preservados e quais foram modificados, garantindo que a comparação seja justa.

### 9.2. Métricas

A avaliação contemplará cinco dimensões.

Correção simbólica: validade dos tipos, conformidade com a gramática e taxa de programas executáveis.

Fidelidade funcional: erro nos pontos SDF, erro de superfície, alinhamento de normais e consistência fora dos pontos usados no ajuste.

Fidelidade visual: erro de silhueta, profundidade, normais renderizadas e métricas de imagem em múltiplas vistas.

Eficiência: operações executadas por consulta, latência medida, custo total de busca e memória consumida.

Generalização: taxa de sucesso e qualidade em topologias inéditas, programas mais profundos e novas distribuições de parâmetros.

A métrica principal será a qualidade atingida para um orçamento fixo de busca. Em seguida, mediremos o menor custo executável alcançado sob um limite fixo de erro. O número de programas explorados e o tempo de ajuste de seus parâmetros serão reportados separadamente.

### 9.3. Estudos de ablação

Os seguintes componentes serão removidos individualmente para medir suas contribuições:

| Ablação                          | Questão                                                        |
| -------------------------------- | -------------------------------------------------------------- |
| Sem relaxação gramatical         | O gradiente sobre as produções reduz a busca?                  |
| Sem heurística neural            | A política de priorização acelera a descoberta?                |
| Sem refinamento paramétrico      | Quanto da qualidade depende de gradientes sobre os parâmetros? |
| Sem expansão dinâmica            | Uma superestrutura fixa limita a generalização?                |
| Sem compartilhamento DAG         | A reutilização reduz o custo real de execução?                 |
| Sem treinamento sintético        | O aprendizado amortizado reduz o orçamento por tarefa?         |
| Sem custo de execução na seleção | O método realmente aprende programas mais eficientes?          |

Também será testada a sensibilidade aos limites de profundidade, à temperatura das distribuições gramaticais e ao orçamento atribuído à exploração de candidatos pouco prováveis.

## 10. Riscos científicos e estratégias de mitigação

O principal risco da proposta é que as relaxações diferenciáveis melhorem a otimização de um programa suave ou híbrido sem melhorar a qualidade do programa discreto que finalmente será executado.

Para detectar isso, todos os métodos serão avaliados depois da discretização e da compilação, nunca somente pelo valor da perda durante o treinamento.

Outro risco é a discrepância entre gradientes e execução real. Operadores como mínimo e máximo têm pontos não diferenciáveis, enquanto o ray marching e as decisões de visibilidade podem introduzir descontinuidades. Usar suavizações pode ser útil para otimizar parâmetros, mas não garante que o programa final tenha a mesma geometria ou desempenho.

Como estratégia, começaremos com perdas funcionais em amostras SDF, depois introduziremos a perda de superfície e, por último, a avaliação por imagens renderizadas. Métodos de diferenciação de programas descontínuos, como Aδ, poderão ser avaliados como uma extensão específica do executor, e não como parte obrigatória da busca estrutural.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

yyuting/Adelta · GitHub

+1



A explosão combinatória será tratada com expansão local, poda, caching de avaliações, reutilização de subprogramas e limites de orçamento. Para DAGs, a criação de nós ocorrerá de forma tipada e acíclica por construção, em vez de depender exclusivamente de uma otimização irrestrita de uma matriz de adjacência.

Por fim, várias estruturas podem gerar saídas praticamente indistinguíveis. A avaliação privilegiará, por isso, equivalência funcional, qualidade e custo em vez de exigir igualdade exata entre a árvore descoberta e a árvore que gerou a observação.

## 11. Plano de trabalho

Proponho um plano inicial de 18 meses, dividido em fases com resultados verificáveis.

### M1–M3

Fase 1 — Linguagem e benchmark

Definir a IR tipada, a gramática SDF, o executor, a geração de dados e as métricas. Implementar um baseline enumerativo com otimização paramétrica.

### M4–M6

Fase 2 — Síntese diferenciável

Reproduzir os componentes necessários de dPads e testar distribuições sobre regras gramaticais em programas de profundidade limitada.

### M7–M9

Fase 3 — Busca híbrida

Integrar refinamento contínuo, busca guiada por heurísticas neurais e mecanismos de cache e expansão local.

### M10–M12

Fase 4 — DAGs e custo executável

Implementar compartilhamento de subgrafos, poda e avaliação do custo compilado; realizar as primeiras ablações.

### M13–M15

Fase 5 — Generalização e transferência

Treinar a política em múltiplas famílias de programas sintéticos e avaliar novas topologias e níveis de profundidade.

### M16–M18

Fase 6 — Consolidação

Comparar baselines, analisar resultados negativos, documentar a DSL, preparar o código reprodutível e redigir a publicação.

Os marcos são metas de execução, não garantias de resultado. O critério para avançar entre fases será a obtenção de uma implementação reproduzível e de evidências experimentais suficientes, mesmo que algumas hipóteses sejam refutadas.

## 12. Contribuições científicas esperadas

O projeto deverá produzir resultados em quatro níveis.

1\. Um algoritmo híbrido de síntese. Uma estratégia que combine probabilidades sobre regras gramaticais, ajuste de parâmetros e expansão dinâmica de programas. A contribuição será demonstrada pela comparação com os componentes isolados.

2\. Uma representação para síntese tipada de DAGs. Uma IR que permita construir programas incompletos, preservar restrições de tipos e compartilhar subexpressões sem perder a validade do programa executável.

3\. Um benchmark reproduzível. Um gerador de programas SDF com variação de estruturas e parâmetros, saídas de referência, métricas de custo e divisões que avaliem a generalização composicional.

4\. Um resultado empírico sobre otimização estrutural. Evidências sobre quando a relaxação diferenciável ajuda, quando a busca discreta é preferível e como o custo do programa executável altera as estruturas descobertas.

Essas contribuições são objetivos de pesquisa. A novidade não está garantida pelo uso conjunto dos componentes; será estabelecida demonstrando uma capacidade ou eficiência que os métodos de referência não alcançam nas mesmas condições.

## 13. Critérios de sucesso e decisão sobre a continuidade

Eu estabeleceria três critérios de decisão para evitar que o projeto se transforme numa implementação grande sem uma hipótese científica verificável.

- Critério de síntese: o sistema encontra programas válidos que atingem a qualidade requerida em estruturas inéditas.
- Critério de eficiência: sob o mesmo orçamento de busca, a combinação proposta melhora a taxa de sucesso ou o custo final dos programas em relação aos baselines.
- Critério de generalização: as melhorias permanecem quando a estrutura dos programas de teste difere da estrutura dos exemplos usados para treinamento.

Os resultados devem incluir também condições nas quais o método falha. Por exemplo, se dPads reduzir fortemente o número de expansões, mas conduzir a programas maiores depois da discretização, isso indicará que a distribuição gramatical precisa incorporar o custo do candidato executável. Se a expansão dinâmica for sistematicamente mais cara do que uma superestrutura pequena, a contribuição pode estar em definir quando utilizar cada estratégia, e não em substituir todas as arquiteturas fixas.

## 14. Extensão para um framework geral

Embora a primeira implementação seja voltada para SDFs, o núcleo não deve depender de geometria.

A generalização será feita pelo contrato de domínio:

- Uma gramática com produções tipadas.
- Uma biblioteca de primitivas executáveis.
- Um mecanismo de validação e execução.
- Um conjunto de observações de referência.
- Uma função de perda e um modelo de custo.

A gramática SDF será a primeira instância desse contrato. Mais tarde, uma DSL de DSP poderá substituir `SDF` por tipos de sinal, osciladores, filtros e operações de modulação, preservando o mecanismo de síntese.

A extensão para áudio seria uma avaliação forte de generalidade: exigiria lidar com sinais temporais, parâmetros estocásticos e topologias diferentes, em vez de apenas reproduzir um segundo conjunto de formas geométricas.

## Conclusão

A proposta mais promissora é investigar síntese simbólica com gramática conhecida e crescimento dinâmico, usando relaxações diferenciáveis para aprender a priorizar produções e otimização numérica para refinar os parâmetros, enquanto a validade estrutural permanece sob controle de um mecanismo simbólico explícito.

dPads é o ponto de partida para a distribuição sobre derivações; NEAR fornece a conexão com busca guiada; NLI oferece ideias para execução diferenciável de programas variáveis; e as técnicas de NAS e poda ajudam a estudar decisões estruturais adicionais. Nenhuma delas deve ser adotada sem adaptação ao domínio e avaliação depois da discretização.

O primeiro protótipo deve ser deliberadamente pequeno: gramática SDF tipada, geração sintética, busca enumerativa, ajuste por gradientes e comparação com dPads. Só depois de demonstrar essa base acrescentaria heurísticas neurais, DAGs com compartilhamento e otimização por latência compilada.

Esse recorte torna a pesquisa executável e cientificamente testável. Mais importante, permite que o resultado seja um framework reutilizável para descobrir programas em linguagens conhecidas — e não apenas uma rede especializada em reconstrução de formas.
