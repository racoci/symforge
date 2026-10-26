import { describe, it, expect, vi } from 'vitest';
import {
  LEAN4_LANGUAGE_ID,
  LEAN4_THEME_NAME,
  LEAN4_KEYWORDS,
  LEAN4_TYPE_KEYWORDS,
  LEAN4_DOMAIN_OPERATORS,
  LEAN4_OPERATORS,
  leanLanguageConfiguration,
  leanMonarchLanguage,
  symforgeDarkTheme,
  registerLean4Language,
  tokenizeLean4,
} from './leanLanguage';

describe('Lean 4 Language Configuration & Monarch Definition', () => {
  it('defines correct language identifier and dark theme name', () => {
    expect(LEAN4_LANGUAGE_ID).toBe('lean4');
    expect(LEAN4_THEME_NAME).toBe('symforge-dark');
  });

  it('configures comment delimiters and nested block comment support', () => {
    expect(leanLanguageConfiguration.comments?.lineComment).toBe('--');
    expect(leanLanguageConfiguration.comments?.blockComment).toEqual(['/-', '-/']);
  });

  it('configures standard and mathematical unicode brackets', () => {
    const brackets = leanLanguageConfiguration.brackets;
    expect(brackets).toEqual([
      ['(', ')'],
      ['{', '}'],
      ['[', ']'],
      ['⟨', '⟩'],
      ['⟦', '⟧'],
    ]);
  });

  it('configures auto-closing and surrounding pairs', () => {
    const autoPairs = leanLanguageConfiguration.autoClosingPairs;
    expect(autoPairs).toBeDefined();
    expect(autoPairs?.some((p) => p.open === '⟨' && p.close === '⟩')).toBe(true);
    expect(autoPairs?.some((p) => p.open === '⟦' && p.close === '⟧')).toBe(true);

    const surroundingPairs = leanLanguageConfiguration.surroundingPairs;
    expect(surroundingPairs).toBeDefined();
    expect(surroundingPairs?.some((p) => p.open === '⟨' && p.close === '⟩')).toBe(true);
  });

  it('contains all required keywords in Monarch definition', () => {
    const requiredKeywords = [
      'def', 'theorem', 'lemma', 'axiom', 'inductive', 'structure', 'class', 'instance',
      'where', 'open', 'namespace', 'section', 'end', 'import', 'variable', 'variables',
      'universe', 'universes', 'example', 'match', 'with', 'if', 'then', 'else',
      'let', 'have', 'show', 'from', 'by', 'deriving', 'mutual', 'private',
      'protected', 'noncomputable',
    ];

    for (const kw of requiredKeywords) {
      expect(LEAN4_KEYWORDS).toContain(kw);
      expect((leanMonarchLanguage.keywords as string[]).includes(kw)).toBe(true);
    }
  });

  it('contains all required type keywords and builtins in Monarch definition', () => {
    const requiredTypes = [
      'Type', 'Prop', 'Sort', 'Nat', 'Int', 'Float', 'String', 'Bool',
      'List', 'Array', 'Option', 'UInt32', 'FreeParam', 'AudioNode',
      'SDFNode', 'HardwareNode', 'UniversalNode',
    ];

    for (const ty of requiredTypes) {
      expect(LEAN4_TYPE_KEYWORDS).toContain(ty);
      expect((leanMonarchLanguage.typeKeywords as string[]).includes(ty)).toBe(true);
    }
  });

  it('contains all required SymForge domain operators and constructors in Monarch definition', () => {
    const domainOps = [
      'sine', 'saw', 'square', 'noise', 'lowpass', 'gain', 'mix', 'adsr',
      'exp_decay', 'lfo', 'fm', 'sphere', 'box', 'cylinder', 'torus',
      'plane', 'translate', 'rotate', 'scale', 'union', 'difference',
      'intersection', 'smooth_union',
    ];

    for (const op of domainOps) {
      expect(LEAN4_DOMAIN_OPERATORS).toContain(op);
      expect((leanMonarchLanguage.domainOperators as string[]).includes(op)).toBe(true);
    }
  });

  it('contains required operators and mathematical unicode in Monarch definition', () => {
    const operators = [
      ':=', '→', '->', '=>', '∀', '∃', 'λ', 'fun', '∧', '∨', '¬', '≠',
      '≤', '<=', '≥', '>=', '=', '==', '<', '>', '+', '-', '*', '/', '%',
      '++', '∘', '•', '⟨', '⟩', '⟦', '⟧',
    ];

    for (const op of operators) {
      expect(LEAN4_OPERATORS).toContain(op);
      expect((leanMonarchLanguage.operators as string[]).includes(op)).toBe(true);
    }
  });

  it('configures symforge-dark theme with consistent syntax colors', () => {
    expect(symforgeDarkTheme.base).toBe('vs-dark');
    expect(symforgeDarkTheme.colors['editor.background']).toBe('#0f172a');
    expect(symforgeDarkTheme.colors['editor.foreground']).toBe('#f8fafc');

    const ruleTokens = symforgeDarkTheme.rules.map((r) => r.token);
    expect(ruleTokens).toContain('keyword');
    expect(ruleTokens).toContain('type');
    expect(ruleTokens).toContain('identifier');
    expect(ruleTokens).toContain('operator');
    expect(ruleTokens).toContain('delimiter');
    expect(ruleTokens).toContain('string');
    expect(ruleTokens).toContain('number');
    expect(ruleTokens).toContain('comment');
    expect(ruleTokens).toContain('attribute');
  });

  it('registers lean4 language, configuration, and theme with Monaco instance', () => {
    const registeredLangs: Array<{ id: string }> = [];
    const configs: Record<string, unknown> = {};
    const tokensProviders: Record<string, unknown> = {};
    const definedThemes: Record<string, unknown> = {};

    const mockMonaco = {
      languages: {
        getLanguages: vi.fn(() => registeredLangs),
        register: vi.fn((desc: { id: string }) => {
          registeredLangs.push(desc);
        }),
        setLanguageConfiguration: vi.fn((id: string, config: unknown) => {
          configs[id] = config;
        }),
        setMonarchTokensProvider: vi.fn((id: string, provider: unknown) => {
          tokensProviders[id] = provider;
        }),
      },
      editor: {
        defineTheme: vi.fn((name: string, theme: unknown) => {
          definedThemes[name] = theme;
        }),
      },
    };

    registerLean4Language(mockMonaco);

    expect(mockMonaco.languages.register).toHaveBeenCalledWith(
      expect.objectContaining({ id: 'lean4' }),
    );
    expect(mockMonaco.languages.setLanguageConfiguration).toHaveBeenCalledWith(
      'lean4',
      leanLanguageConfiguration,
    );
    expect(mockMonaco.languages.setMonarchTokensProvider).toHaveBeenCalledWith(
      'lean4',
      leanMonarchLanguage,
    );
    expect(mockMonaco.editor.defineTheme).toHaveBeenCalledWith(
      'symforge-dark',
      symforgeDarkTheme,
    );

    // Calling again does not re-register if already present
    registerLean4Language(mockMonaco);
    expect(mockMonaco.languages.register).toHaveBeenCalledTimes(1);
  });
});

describe('Lean 4 Monarch Tokenizer Rules', () => {
  it('tags standard keywords as keyword', () => {
    const sample = 'def theorem lemma axiom inductive structure class instance where open import by let have';
    const tokens = tokenizeLean4(sample);

    for (const token of tokens) {
      expect(token.tokenClass).toBe('keyword');
    }
  });

  it('tags type keywords as type', () => {
    const sample = 'Type Prop Sort Nat Int Float String Bool List Array Option UInt32 FreeParam AudioNode SDFNode HardwareNode UniversalNode';
    const tokens = tokenizeLean4(sample);

    for (const token of tokens) {
      expect(token.tokenClass).toBe('type');
    }
  });

  it('tags SymForge domain operators as keyword', () => {
    const sample = 'sphere box cylinder torus plane union intersection difference smooth_union sine saw square noise lowpass gain mix';
    const tokens = tokenizeLean4(sample);

    for (const token of tokens) {
      expect(token.tokenClass).toBe('keyword');
    }
  });

  it('tags mathematical unicode and multi-character operators as operator', () => {
    const sample = ':= → -> => ∀ ∃ λ fun ∧ ∨ ¬ ≠ ≤ <= ≥ >= = == < > + - * / % ++ ∘ •';
    const tokens = tokenizeLean4(sample);

    for (const token of tokens) {
      expect(token.tokenClass).toBe('operator');
    }
  });

  it('tags mathematical unicode and standard brackets as delimiter', () => {
    const sample = '⟨ ⟩ ⟦ ⟧ ( ) [ ] { } , : ; .';
    const tokens = tokenizeLean4(sample);

    for (const token of tokens) {
      expect(token.tokenClass).toBe('delimiter');
    }
  });

  it('properly tokenizes line comments', () => {
    const sample = 'def x := 1 -- this is a line comment\ndef y := 2';
    const tokens = tokenizeLean4(sample);

    const commentToken = tokens.find((t) => t.value.includes('this is a line comment'));
    expect(commentToken).toBeDefined();
    expect(commentToken?.tokenClass).toBe('comment');
    expect(commentToken?.value).toBe('-- this is a line comment');
  });

  it('properly tokenizes block comments and nested block comments', () => {
    const sample = 'def a := /- block comment -/ 42';
    const tokens = tokenizeLean4(sample);
    const commentTokens = tokens.filter((t) => t.tokenClass === 'comment');
    expect(commentTokens.length).toBeGreaterThan(0);
    expect(commentTokens.some((t) => t.value === '/-')).toBe(true);
    expect(commentTokens.some((t) => t.value === '-/')).toBe(true);

    // Nested block comments
    const nestedSample = 'def b := /- outer /- nested -/ outer -/ 100';
    const nestedTokens = tokenizeLean4(nestedSample);
    const numToken = nestedTokens.find((t) => t.value === '100');
    expect(numToken).toBeDefined();
    expect(numToken?.tokenClass).toBe('number');
  });

  it('properly tokenizes attributes @[default] and @[inline, simp]', () => {
    const sample = '@[default] def x : Nat := 0\n@[inline, simp] def y : Nat := 1';
    const tokens = tokenizeLean4(sample);

    const attributeTokens = tokens.filter((t) => t.tokenClass === 'attribute');
    expect(attributeTokens.length).toBeGreaterThan(0);
    expect(attributeTokens.some((t) => t.value === '@[')).toBe(true);
  });

  it('properly tokenizes string literals and escaped strings', () => {
    const sample = 'def name : String := "sphere_primitive"\ndef msg := "escaped \\"quote\\""';
    const tokens = tokenizeLean4(sample);

    const strTokens = tokens.filter((t) => t.tokenClass === 'string');
    expect(strTokens.length).toBeGreaterThan(0);
    expect(strTokens.some((t) => t.value.includes('sphere_primitive'))).toBe(true);
  });

  it('properly tokenizes numbers (floats, integers, hex, binary)', () => {
    const sample = '1.25 0.08 3.14 42 0xDEADBEEF 0b1010 1e-4';
    const tokens = tokenizeLean4(sample);

    expect(tokens.length).toBe(7);
    for (const token of tokens) {
      expect(token.tokenClass).toBe('number');
    }
  });

  it('tokenizes full SymForge 3D SDF Gyroid program without syntax breakdown', () => {
    const gyroidCode = `import SymforgeIr.Types

def myGyroidSDF : SDFNode :=
  SDFNode.intersection
    (SDFNode.sphere (FreeParam.mk "radius" 1.25 0.5 3.0))
    (SDFNode.gyroid
      (FreeParam.mk "scale" 3.14 1.0 10.0)
      (FreeParam.mk "thickness" 0.08 0.01 0.5))`;

    const tokens = tokenizeLean4(gyroidCode);
    expect(tokens.length).toBeGreaterThan(20);

    // Invariants:
    // 1. 'import' and 'def' are keyword
    const importToken = tokens.find((t) => t.value === 'import');
    expect(importToken?.tokenClass).toBe('keyword');
    const defToken = tokens.find((t) => t.value === 'def');
    expect(defToken?.tokenClass).toBe('keyword');

    // 2. 'SDFNode' and 'FreeParam' are type
    const sdfTokens = tokens.filter((t) => t.value === 'SDFNode');
    expect(sdfTokens.length).toBeGreaterThan(0);
    expect(sdfTokens.every((t) => t.tokenClass === 'type')).toBe(true);

    const paramTokens = tokens.filter((t) => t.value === 'FreeParam');
    expect(paramTokens.length).toBeGreaterThan(0);
    expect(paramTokens.every((t) => t.tokenClass === 'type')).toBe(true);

    // 3. 'intersection' and 'sphere' are domain operators (keyword)
    const intersectionToken = tokens.find((t) => t.value === 'intersection');
    expect(intersectionToken?.tokenClass).toBe('keyword');
    const sphereToken = tokens.find((t) => t.value === 'sphere');
    expect(sphereToken?.tokenClass).toBe('keyword');

    // 4. ':=' is operator
    const defOpToken = tokens.find((t) => t.value === ':=');
    expect(defOpToken?.tokenClass).toBe('operator');

    // 5. Strings: "radius", "scale", "thickness"
    const radiusToken = tokens.find((t) => t.value.includes('radius'));
    expect(radiusToken?.tokenClass).toBe('string');

    // 6. Numbers: 1.25, 0.5, 3.0, 3.14, 1.0, 10.0, 0.08, 0.01, 0.5
    const numTokens = tokens.filter((t) => t.tokenClass === 'number');
    expect(numTokens.length).toBe(9);
  });

  it('tokenizes full SymForge Audio DSP synthesis patch without syntax breakdown', () => {
    const audioCode = `import SymforgeIr.Types

def mySynthPatch : AudioNode :=
  AudioNode.mix 0.5
    (AudioNode.sine (FreeParam.mk "freq" 440.0 20.0 20000.0))
    (AudioNode.lowpass
      (AudioNode.saw (FreeParam.mk "sawFreq" 220.0 20.0 5000.0))
      (FreeParam.mk "cutoff" 800.0 100.0 12000.0))`;

    const tokens = tokenizeLean4(audioCode);
    expect(tokens.length).toBeGreaterThan(25);

    const audioNodeTokens = tokens.filter((t) => t.value === 'AudioNode');
    expect(audioNodeTokens.every((t) => t.tokenClass === 'type')).toBe(true);

    const mixToken = tokens.find((t) => t.value === 'mix');
    expect(mixToken?.tokenClass).toBe('keyword');

    const sineToken = tokens.find((t) => t.value === 'sine');
    expect(sineToken?.tokenClass).toBe('keyword');

    const lowpassToken = tokens.find((t) => t.value === 'lowpass');
    expect(lowpassToken?.tokenClass).toBe('keyword');
  });

  it('tokenizes mathematical theorem with unicode operators and proof structure', () => {
    const theoremCode = `theorem sound_equiv (x : AudioNode) : x → ∀ y, ∃ z, λ t, t ∧ ¬ y ≤ z := by
  ⟨x, y⟩`;

    const tokens = tokenizeLean4(theoremCode);

    const theoremToken = tokens.find((t) => t.value === 'theorem');
    expect(theoremToken?.tokenClass).toBe('keyword');

    const arrowToken = tokens.find((t) => t.value === '→');
    expect(arrowToken?.tokenClass).toBe('operator');

    const forallToken = tokens.find((t) => t.value === '∀');
    expect(forallToken?.tokenClass).toBe('operator');

    const existsToken = tokens.find((t) => t.value === '∃');
    expect(existsToken?.tokenClass).toBe('operator');

    const lambdaToken = tokens.find((t) => t.value === 'λ');
    expect(lambdaToken?.tokenClass).toBe('operator');

    const andToken = tokens.find((t) => t.value === '∧');
    expect(andToken?.tokenClass).toBe('operator');

    const notToken = tokens.find((t) => t.value === '¬');
    expect(notToken?.tokenClass).toBe('operator');

    const leToken = tokens.find((t) => t.value === '≤');
    expect(leToken?.tokenClass).toBe('operator');

    const openAngleToken = tokens.find((t) => t.value === '⟨');
    expect(openAngleToken?.tokenClass).toBe('delimiter');

    const closeAngleToken = tokens.find((t) => t.value === '⟩');
    expect(closeAngleToken?.tokenClass).toBe('delimiter');
  });
});
