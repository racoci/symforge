# Bidirectional Lean 4 AST Parser & Pretty-Printer Architecture

**Document ID:** `ARCH-LEAN4-PARSER-202610`  
**Track:** `lean4_web_ide_interactive_widgets_202610`  
**Task Reference:** Task 1.4: Bidirectional Lean 4 AST Parser & Pretty-Printer (Code $\leftrightarrow$ AST)  
**Status:** Approved & Verified  
**Date:** 2026-10-26  

---

## 1. Executive Summary & Architectural Overview

The SymForge Bidirectional Lean 4 AST Parser & Pretty-Printer (`packages/web/src/core/parser/`) provides a client-side recursive descent compiler pipeline that bridges human-readable Lean 4 source definitions and immutable typed AST graphs (`ASTNode`). Operating entirely in the browser runtime without external language server binaries, it guarantees:

1. **Bidirectional Isomorphism:** Complete round-trip invariance where $\text{parse}(\text{format}(ast)) \equiv ast$ verified under structural equality (`areNodesEqual`).
2. **Exact Source Coordinate Span Tracking:** Every subexpression and continuous parameter (`FreeParam`) retains exact source coordinates (`SourceSpan`) with 1-indexed line/column boundaries and 0-indexed character offsets, enabling surgical in-editor selection bounding and 1-click injection in Phase 3.
3. **Resilient Grammar Resolution:** Support for multiple Lean 4 syntactic paradigms, including fully qualified constructor calls (`SDFNode.difference`), constructor dot notation (`.difference`), direct syntax (`difference`), explicit parameters (`FreeParam.mk`), anonymous constructor brackets (`⟨"id", min, max, default⟩`), record notation (`{ id := ... }`), and shorthand positional numeric literals.

---

## 2. Lexical Grammar & Tokenization Architecture

The lexer (`packages/web/src/core/parser/lexer.ts`) streams input UTF-8 text into typed tokens (`Keyword`, `Identifier`, `Number`, `String`, `Operator`, `Delimiter`, `EOF`). Whitespace, line comments (`--`), and arbitrarily nested block comments (`/- ... /- ... -/ -/`) are stripped deterministically while maintaining exact line, column, and character offsets.

```text
+---------------------------------------------------------------------------------------------------------+
|                                    LEAN 4 LEXICAL TOKENIZER PIPELINE                                    |
|                                                                                                         |
|   Source Stream "def myShape : SDFNode := .difference (SDFNode.box 1.0 2.0 3.0) (SDFNode.sphere 0.5)"   |
|                                             |                                                           |
|                                             v                                                           |
|                       +-------------------------------------------+                             |
|                       |   Character Cursor & Coordinate Tracker   |                             |
|                       |   (startLine, startCol, offsets 0..N)     |                             |
|                       +-------------------------------------------+                             |
|                                             |                                                           |
|             +-------------------------------+-------------------------------+                           |
|             |                               |                               |                           |
|             v                               v                               v                           |
|     [Comments & Blanks]             [Literal Scanning]              [Delimiters & Ops]                  |
|     -- Line comment                 Strings: "...", escapes         :=, :, ., ->                        |
|     /- /- Nested -/ -/              Numbers: -1.5, 42, 1e-4         (, ), [, ], {, }, ⟨, ⟩              |
|                                             |                                                           |
|                                             v                                                           |
|                        +-----------------------------------------+                                      |
|                        | Typed Token Stream with Exact SourceSpan|                                      |
|                        +-----------------------------------------+                                      |
+---------------------------------------------------------------------------------------------------------+
```

```mermaid
flowchart TD
    A[Lean 4 Source Code String] --> B[Character Stream Scanner]
    B --> C{Character Lookahead}
    C -->|Whitespace & Comments| D[Skip Whitespace & Nested Comments]
    C -->|Quotes| E[Scan String Literal with Escapes]
    C -->|Digits or Negative Sign| F[Scan Float, Int, or Sci-Notation]
    C -->|Alpha & Underscore| G[Scan Identifier or Keyword]
    C -->|Punctuation & Unicode| H[Scan Delimiter or Operator]
    D --> I[Advance Line, Col, and Offset Coordinates]
    E --> J[Emit Token: String]
    F --> J1[Emit Token: Number]
    G --> J2[Emit Token: Keyword / Identifier]
    H --> J3[Emit Token: Operator / Delimiter]
    I --> B
    J --> K[Typed Token Stream with SourceSpan]
    J1 --> K
    J2 --> K
    J3 --> K
    K --> L[Emit Token: EOF]
```

---

## 3. Recursive Descent Parser & Invariant Mapping

The parser (`packages/web/src/core/parser/leanParser.ts`) implements a recursive descent grammar with explicit backtracking prevention, bounded loops, and domain-agnostic constructor resolution.

```text
+---------------------------------------------------------------------------------------------------------+
|                                  RECURSIVE DESCENT COMPILER GRAMMAR                                     |
|                                                                                                         |
|  [Preamble]          ::= ( 'import' <ident> | 'open' <ident> | 'namespace' <ident> )*                   |
|  [Declaration]       ::= 'def' <ident> [ ':' <Type> ] ':=' [ <LetBindings> ] <Expression>              |
|  [LetBindings]       ::= ( 'let' <ident> [ ':' <Type> ] ':=' <ExpressionOrParam> [ 'in' ] )*            |
|  [Expression]        ::= <ConstructorCall> | '(' <Expression> ')' | <Atom>                              |
|  [ConstructorCall]   ::= ( <QualifiedName> | '.' <Ident> | <Ident> ) <Argument>*                       |
|  [Argument]          ::= <Atom> | '(' <ExpressionOrParam> ')' | '⟨' <ParamArgs> '⟩' | '{' <Fields> '}'  |
|  [FreeParam]         ::= 'FreeParam.mk' <id> <min> <max> <default>                                      |
|                        | '⟨' <id> [','] <min> [','] <max> [','] <default> '⟩'                           |
|                        | '{' 'id' ':=' <id> ',' 'minVal' ':=' <min> ... '}'                             |
|                        | <NumberLiteral> (Auto-inferred continuous bounds)                              |
+---------------------------------------------------------------------------------------------------------+
```

```mermaid
flowchart TD
    subgraph ParsePipeline [Parser Execution Pipeline]
        P1[Input Token Stream] --> P2[Skip Preamble: import, open, namespace]
        P2 --> P3{Header Match: 'def'?}
        P3 -->|Yes| P4[Extract declarationName, domainType, and ':=']
        P3 -->|No| P5[Default declarationName='main', infer domainType]
        P4 --> P6[Parse 'let' Environment Bindings]
        P5 --> P6
        P6 --> P7[Parse Root Expression]
        P7 --> P8{Constructor Domain}
        P8 -->|AudioNode| P9[Audio Constructor: sine, saw, square, lowpass, adsr, fm, mix]
        P8 -->|SDFNode| P10[SDF Constructor: sphere, box, cylinder, torus, translate, union]
        P8 -->|HardwareNode| P11[Hardware Constructor: reg, imm, add, sub, mul, shl, seq]
        P8 -->|UniversalNode| P12[Universal Container: audio, sdf, hardware, macroNode]
        P9 --> P13[Attach Unique Node ID & Register SourceSpan]
        P10 --> P13
        P11 --> P13
        P12 --> P13
        P13 --> P14[Return ParseResult: ast, declarationName, domainType, spans, errors]
    end
```

---

## 4. SourceSpan Tracking & Range-to-AST Mapping

Every AST node and continuous parameter parsed is assigned a unique identifier (`node_0`, `node_1`, ..., and `param.id`). Their exact coordinates are preserved in `ParseResult.spans` and `ParseResult.nodeSpanMap`, establishing a direct bidirectional map between Monaco editor selection ranges and the underlying syntax tree.

```text
+---------------------------------------------------------------------------------------------------------+
|                                    AST RANGE COORDINATE ENCLOSURE                                       |
|                                                                                                         |
|   Source Code:                                                                                          |
|   Line 1: def testShape : SDFNode :=                                                                    |
|   Line 2:   SDFNode.difference                                                                          |
|   Line 3:     (SDFNode.box (FreeParam.mk "w" 0.1 10.0 1.0) ...)   <-- Span: Line 3 Col 5 to Line 3 Col 61|
|   Line 4:     (SDFNode.sphere (FreeParam.mk "r" 0.1 5.0 0.5))     <-- Span: Line 4 Col 5 to Line 4 Col 52|
|                                                                                                         |
|   Root Span:                                                                                            |
|   startOffset: 0, endOffset: 165 (slices full expression)                                               |
|                                                                                                         |
|   Subexpression Span:                                                                                   |
|   source.slice(paramSpan.startOffset, paramSpan.endOffset) == '(FreeParam.mk "r" 0.1 5.0 0.5)'          |
+---------------------------------------------------------------------------------------------------------+
```

```mermaid
flowchart LR
    subgraph SourceText [Monaco Editor Document]
        Doc["Lean 4 Source File"]
        Line1["def testShape : SDFNode :="]
        Line2["  SDFNode.difference"]
        Line3["    (SDFNode.box ...)"]
        Line4["    (SDFNode.sphere ...)"]
    end

    subgraph SpanIndex [Coordinate Registry]
        RootSpan["root: [Line 1, Col 1 to Line 4, Col 52]"]
        BoxSpan["node_1: [Line 3, Col 5 to Line 3, Col 61]"]
        SphereSpan["node_2: [Line 4, Col 5 to Line 4, Col 52]"]
        ParamRSpan["param:r: [Line 4, Col 21 to Line 4, Col 51]"]
    end

    subgraph ASTNodes [Immutable Client AST]
        DiffNode["DifferenceSDFNode: node_0"]
        BoxNode["BoxSDFNode: node_1"]
        SphereNode["SphereSDFNode: node_2"]
        ParamR["FreeParam 'r': 0.5"]
    end

    Line2 --> RootSpan --> DiffNode
    Line3 --> BoxSpan --> BoxNode
    Line4 --> SphereSpan --> SphereNode
    Line4 --> ParamRSpan --> ParamR
```

---

## 5. Deterministic Lean 4 Pretty-Printer & Formatting Rules

The pretty-printer (`packages/web/src/core/parser/leanPrinter.ts`) renders typed AST nodes back into clean, idiomatic Lean 4 code. It supports:
- **Clean Indentation:** Nested indentation of subexpressions (default 2 spaces).
- **Parenthesization Invariants:** Non-atomic constructor arguments are strictly parenthesized `( ... )`, while root definitions remain unparenthesized.
- **Float Invariant Formatting:** Floating point numbers are always emitted with trailing decimals (e.g., `440.0` rather than `440`) to adhere to Lean 4 `Float` literal requirements.
- **Compact & Dot-Notation Modes:** Optional flags for single-line compression (`compact: true`) and constructor dot notation (`dotNotation: true`, emitting `.sphere` instead of `SDFNode.sphere`).

```text
+---------------------------------------------------------------------------------------------------------+
|                                    PRETTY-PRINTER TRANSFORMATION                                        |
|                                                                                                         |
|   ASTNode (SphereSDFNode)  ---------------------> formatLeanCode("myShape", "SDFNode", ast)            |
|                                                                                                         |
|   Emits Standard Indented Lean 4:                                                                       |
|   def myShape : SDFNode :=                                                                              |
|     SDFNode.sphere (FreeParam.mk "radius" 0.01 5.0 1.25)                                                |
|                                                                                                         |
|   Emits Compact Dot-Notation Lean 4:                                                                    |
|   def myShape : SDFNode := .sphere (FreeParam.mk "radius" 0.01 5.0 1.25)                                |
+---------------------------------------------------------------------------------------------------------+
```

```mermaid
flowchart TD
    AST[Typed AST Node] --> InspectDomain{Detect Domain Type}
    InspectDomain -->|AudioNode| FA[Format Audio Expression]
    InspectDomain -->|SDFNode| FS[Format SDF Expression]
    InspectDomain -->|HardwareNode| FH[Format Hardware Expression]
    InspectDomain -->|UniversalNode| FU[Format Universal Expression]

    FA --> FormatArgs[Format Sub-arguments & FreeParams]
    FS --> FormatArgs
    FH --> FormatArgs
    FU --> FormatArgs

    FormatArgs --> FloatCheck["formatFloat: Ensure .0 for Floats"]
    FloatCheck --> WrapCheck{Is Argument in Parent Call?}
    WrapCheck -->|Yes| WrapParens[Wrap in Parentheses]
    WrapCheck -->|No| NoParens[Leave Unparenthesized at Root]

    WrapParens --> IndentFormat{Check FormatOptions}
    NoParens --> IndentFormat
    IndentFormat -->|compact: true| SingleLine["def decl : Type := expr"]
    IndentFormat -->|compact: false| MultiLine["def decl : Type := \n  expr"]
```

---

## 6. Round-Trip Invariance & Symmetrical Validation

The bidirectional pipeline enforces **Symmetrical Cross-Validation** (as defined in `GEMINI.md`):

$$\text{parse}(\text{format}(ast)).ast \equiv ast$$

This invariant is verified by the **Pass-to-Complete Gate** in `packages/web/src/core/parser/leanParser.test.ts` across:
- **25 Canonical Audio Programs:** Covering primitive oscillators, FM synthesis, ADSR envelopes, lowpass filters, gain modulation, multi-branch mixes, and deep DSP graphs.
- **25 Canonical SDF Programs:** Covering primitive geometry (spheres, boxes, cylinders, tori), spatial transformations (translation, rotation, scale), and boolean CSG operations (unions, smooth unions, intersections, differences).
- **Hardware Instruction Programs:** Instruction sequences verifying register, immediate, arithmetic, shift, and sequential composition invariance.
- **Universal Container Programs:** Polymorphic wrapper nodes verifying round-trip equivalence.

```text
+---------------------------------------------------------------------------------------------------------+
|                                    SYMMETRICAL ROUND-TRIP INVARIANT                                     |
|                                                                                                         |
|       AST Node Instance [X] ----------------------- formatLeanCode --------------------> Lean 4 Code   |
|               ^                                                                               |         |
|               |                                                                               |         |
|         areNodesEqual? (Verified: true)                                                  parseLeanCode  |
|               |                                                                               |         |
|               +----------------------------- Result AST [X'] <--------------------------------+         |
+---------------------------------------------------------------------------------------------------------+
```

```mermaid
sequenceDiagram
    autonumber
    participant Client as Test Runner
    participant Printer as leanPrinter
    participant Lexer as lexer
    participant Parser as leanParser
    participant Validator as areNodesEqual

    Client->>Printer: formatLeanCode("test", "SDFNode", originalAST)
    Printer-->>Client: leanSourceCode
    Client->>Lexer: tokenize(leanSourceCode)
    Lexer-->>Parser: tokenStream + sourceSpans
    Parser->>Parser: parseExpression(tokenStream)
    Parser-->>Client: parseResult (parsedAST, spans, errors)
    Client->>Validator: areNodesEqual(parseResult.ast, originalAST)
    Validator-->>Client: true (Zero Discrepancy Gate)
```

---

## 7. Integration Contract with In-Context Scoped Variation Generator (Phase 3)

The AST Parser and SourceSpan registry directly unlock Phase 3's Neurosymbolic In-Context Scoped Variation Generator:
1. **Range Intersection:** When the user selects a code block in Monaco Editor (`startLine`, `startCol`, `endLine`, `endCol`), the range is queried against `ParseResult.spans`.
2. **Subtree Identification:** The smallest AST node whose span completely encloses the user selection is located.
3. **Puncturing & Sampling:** The selected subtree is punctured with a typed hole (`Hole[T]`), the contextual tuple $(\text{Parent}, \text{Grandparent}, \text{LeftSibling})$ is extracted, and the PCFG sampler in the background Web Worker generates complexity-stratified variations.
4. **1-Click Replacement:** When a variation candidate is selected, its pretty-printed Lean 4 code is spliced cleanly into the editor using `span.startOffset` and `span.endOffset`.
