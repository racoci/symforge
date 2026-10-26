import type { languages, editor } from 'monaco-editor';

export const LEAN4_LANGUAGE_ID = 'lean4';
export const LEAN4_THEME_NAME = 'symforge-dark';

export const LEAN4_KEYWORDS = [
  'def',
  'theorem',
  'lemma',
  'axiom',
  'inductive',
  'structure',
  'class',
  'instance',
  'where',
  'open',
  'namespace',
  'section',
  'end',
  'import',
  'variable',
  'variables',
  'universe',
  'universes',
  'example',
  'match',
  'with',
  'if',
  'then',
  'else',
  'let',
  'have',
  'show',
  'from',
  'by',
  'deriving',
  'mutual',
  'private',
  'protected',
  'noncomputable',
] as const;

export const LEAN4_TYPE_KEYWORDS = [
  'Type',
  'Prop',
  'Sort',
  'Nat',
  'Int',
  'Float',
  'String',
  'Bool',
  'List',
  'Array',
  'Option',
  'UInt32',
  'FreeParam',
  'AudioNode',
  'SDFNode',
  'HardwareNode',
  'UniversalNode',
] as const;

export const LEAN4_DOMAIN_OPERATORS = [
  'sine',
  'saw',
  'square',
  'noise',
  'lowpass',
  'gain',
  'mix',
  'adsr',
  'exp_decay',
  'lfo',
  'fm',
  'sphere',
  'box',
  'cylinder',
  'torus',
  'plane',
  'translate',
  'rotate',
  'scale',
  'union',
  'difference',
  'intersection',
  'smooth_union',
] as const;

export const LEAN4_OPERATORS = [
  ':=',
  '→',
  '->',
  '=>',
  '∀',
  '∃',
  'λ',
  'fun',
  '∧',
  '∨',
  '¬',
  '≠',
  '≤',
  '<=',
  '≥',
  '>=',
  '=',
  '==',
  '<',
  '>',
  '+',
  '-',
  '*',
  '/',
  '%',
  '++',
  '∘',
  '•',
  '⟨',
  '⟩',
  '⟦',
  '⟧',
] as const;

export type Lean4TokenClass =
  | 'keyword'
  | 'type'
  | 'identifier'
  | 'operator'
  | 'delimiter'
  | 'string'
  | 'number'
  | 'comment'
  | 'attribute';

export interface Lean4Token {
  value: string;
  tokenClass: Lean4TokenClass;
  start: number;
  end: number;
}

export const leanLanguageConfiguration: languages.LanguageConfiguration = {
  comments: {
    lineComment: '--',
    blockComment: ['/-', '-/'],
  },
  brackets: [
    ['(', ')'],
    ['{', '}'],
    ['[', ']'],
    ['⟨', '⟩'],
    ['⟦', '⟧'],
  ],
  autoClosingPairs: [
    { open: '(', close: ')' },
    { open: '{', close: '}' },
    { open: '[', close: ']' },
    { open: '⟨', close: '⟩' },
    { open: '⟦', close: '⟧' },
    { open: '"', close: '"', notIn: ['string'] },
    { open: '/-', close: '-/', notIn: ['string'] },
  ],
  surroundingPairs: [
    { open: '(', close: ')' },
    { open: '{', close: '}' },
    { open: '[', close: ']' },
    { open: '⟨', close: '⟩' },
    { open: '⟦', close: '⟧' },
    { open: '"', close: '"' },
  ],
  folding: {
    markers: {
      start: /^\s*(?:section|namespace)\b/,
      end: /^\s*end\b/,
    },
  },
};

export const leanMonarchLanguage: languages.IMonarchLanguage = {
  defaultToken: '',
  tokenPostfix: '.lean',

  keywords: [...LEAN4_KEYWORDS],
  typeKeywords: [...LEAN4_TYPE_KEYWORDS],
  domainOperators: [...LEAN4_DOMAIN_OPERATORS],
  operators: [...LEAN4_OPERATORS],

  brackets: [
    { open: '(', close: ')', token: 'delimiter' },
    { open: '{', close: '}', token: 'delimiter' },
    { open: '[', close: ']', token: 'delimiter' },
    { open: '⟨', close: '⟩', token: 'delimiter' },
    { open: '⟦', close: '⟧', token: 'delimiter' },
  ],

  tokenizer: {
    root: [
      // Line comments: -- ...
      [/--.*$/, 'comment'],

      // Block comments: /- ... -/
      [/\/-/, 'comment', '@comment'],

      // Attributes: @[default], @[inline, simp]
      [/@\[/, 'attribute', '@attribute'],

      // Strings
      [/"([^"\\]|\\.)*$/, 'string.invalid'],
      [/"/, 'string', '@string'],

      // Floating-point numbers, exponents, and integers
      [/\d*\.\d+(?:[eE][-+]?\d+)?/, 'number'],
      [/\d+(?:[eE][-+]?\d+)/, 'number'],

      // Hexadecimal and binary literals
      [/0[xX][0-9a-fA-F]+/, 'number'],
      [/0[bB][01]+/, 'number'],

      // Integers
      [/\d+/, 'number'],

      // Multi-character operators
      [/(:=|->|=>|\+\+|==|<=|>=)/, 'operator'],

      // Unicode mathematical operators
      [/[→∀∃λ∧∨¬≠≤≥∘•]/, 'operator'],

      // Mathematical Unicode bracket delimiters
      [/[⟨⟩⟦⟧]/, 'delimiter'],

      // ASCII delimiters & punctuation
      [/[()[\]{}]/, 'delimiter'],
      [/[:,;.]/, 'delimiter'],

      // Single-character mathematical operators
      [/[=><!~?&|+\-*/%^]/, 'operator'],

      // Identifiers, Keywords, Types, Domain Operators
      [
        /[a-zA-Z_][a-zA-Z0-9_']*/,
        {
          cases: {
            '@keywords': 'keyword',
            '@typeKeywords': 'type',
            '@domainOperators': 'keyword',
            '@operators': 'operator',
            '@default': 'identifier',
          },
        },
      ],

      // Whitespace
      [/[ \t\r\n]+/, 'white'],
    ],

    comment: [
      [/[^-/]+/, 'comment'],
      [/\/-/, 'comment', '@push'],
      [/-\//, 'comment', '@pop'],
      [/[-/]/, 'comment'],
    ],

    string: [
      [/[^\\"]+/, 'string'],
      [/\\(?:[abfnrtv\\"']|x[0-9A-Fa-f]{1,4}|u[0-9A-Fa-f]{4}|U[0-9A-Fa-f]{8})/, 'string'],
      [/\\./, 'string'],
      [/"/, 'string', '@pop'],
    ],

    attribute: [
      [/\]/, 'attribute', '@pop'],
      [/[a-zA-Z_][a-zA-Z0-9_.]*/, 'attribute'],
      [/[, \t\r\n]+/, 'attribute'],
      [/[\s\S]/, 'attribute'],
    ],
  },
};

export const symforgeDarkTheme: editor.IStandaloneThemeData = {
  base: 'vs-dark',
  inherit: true,
  rules: [
    { token: 'keyword', foreground: 'c586c0', fontStyle: 'bold' },
    { token: 'type', foreground: '4ec9b0', fontStyle: 'bold' },
    { token: 'identifier', foreground: '9cdcfe' },
    { token: 'operator', foreground: '38bdf8' },
    { token: 'delimiter', foreground: '818cf8' },
    { token: 'string', foreground: 'ce9178' },
    { token: 'number', foreground: 'b5cea8' },
    { token: 'comment', foreground: '6a9955', fontStyle: 'italic' },
    { token: 'attribute', foreground: 'fbbf24', fontStyle: 'bold' },
    { token: 'string.invalid', foreground: 'f43f5e' },
  ],
  colors: {
    'editor.background': '#0f172a',
    'editor.foreground': '#f8fafc',
    'editorCursor.foreground': '#38bdf8',
    'editor.lineHighlightBackground': '#1e293b55',
    'editorLineNumber.foreground': '#64748b',
    'editorLineNumber.activeForeground': '#38bdf8',
    'editor.selectionBackground': '#38bdf833',
    'editor.inactiveSelectionBackground': '#38bdf81a',
  },
};

export function registerLean4Language(monacoInstance: unknown): void {
  if (!monacoInstance || typeof monacoInstance !== 'object') return;
  const m = monacoInstance as {
    languages?: {
      getLanguages?: () => Array<{ id: string }>;
      register?: (desc: { id: string; extensions?: string[]; aliases?: string[] }) => void;
      setLanguageConfiguration?: (id: string, config: unknown) => void;
      setMonarchTokensProvider?: (id: string, provider: unknown) => void;
    };
    editor?: {
      defineTheme?: (name: string, theme: unknown) => void;
    };
  };

  if (!m.languages) return;

  const languages = m.languages.getLanguages?.() ?? [];
  const isRegistered = languages.some((lang) => lang.id === LEAN4_LANGUAGE_ID);

  if (!isRegistered && m.languages.register) {
    m.languages.register({
      id: LEAN4_LANGUAGE_ID,
      extensions: ['.lean'],
      aliases: ['Lean', 'Lean 4', 'lean4'],
    });
  }

  m.languages.setLanguageConfiguration?.(LEAN4_LANGUAGE_ID, leanLanguageConfiguration);
  m.languages.setMonarchTokensProvider?.(LEAN4_LANGUAGE_ID, leanMonarchLanguage);

  if (m.editor?.defineTheme) {
    m.editor.defineTheme(LEAN4_THEME_NAME, symforgeDarkTheme);
  }
}

const KEYWORD_SET = new Set<string>(LEAN4_KEYWORDS);
const TYPE_KEYWORD_SET = new Set<string>(LEAN4_TYPE_KEYWORDS);
const DOMAIN_OPERATOR_SET = new Set<string>(LEAN4_DOMAIN_OPERATORS);

const MULTI_CHAR_OPERATORS = [':=', '->', '=>', '++', '==', '<=', '>='];
const UNICODE_OPERATORS = new Set(['→', '∀', '∃', 'λ', '∧', '∨', '¬', '≠', '≤', '≥', '∘', '•']);
const UNICODE_DELIMITERS = new Set(['⟨', '⟩', '⟦', '⟧']);
const ASCII_DELIMITERS = new Set(['(', ')', '{', '}', '[', ']', ':', ',', ';', '.']);
const SINGLE_CHAR_OPERATORS = new Set(['=', '>', '<', '+', '-', '*', '/', '%', '^', '!', '~', '?', '&', '|']);

/**
 * Pure TypeScript tokenizer mirroring Lean 4 Monarch finite-state rules.
 * Runs deterministically in any JS environment (browser, worker, node, Vitest)
 * without external DOM or Monaco dependencies.
 */
export function tokenizeLean4(code: string): Lean4Token[] {
  const tokens: Lean4Token[] = [];
  let i = 0;
  const len = code.length;
  const stateStack: Array<'root' | 'comment' | 'attribute' | 'string'> = ['root'];

  while (i < len) {
    const currentState = stateStack[stateStack.length - 1];

    if (currentState === 'comment') {
      const remaining = code.slice(i);
      if (remaining.startsWith('/-')) {
        tokens.push({ value: '/-', tokenClass: 'comment', start: i, end: i + 2 });
        stateStack.push('comment');
        i += 2;
        continue;
      }
      if (remaining.startsWith('-/')) {
        tokens.push({ value: '-/', tokenClass: 'comment', start: i, end: i + 2 });
        stateStack.pop();
        i += 2;
        continue;
      }
      const nextSpecial = code.slice(i).search(/\/[-]|-\//);
      if (nextSpecial === -1) {
        tokens.push({ value: code.slice(i), tokenClass: 'comment', start: i, end: len });
        i = len;
      } else {
        const chunkLen = Math.max(1, nextSpecial);
        tokens.push({ value: code.slice(i, i + chunkLen), tokenClass: 'comment', start: i, end: i + chunkLen });
        i += chunkLen;
      }
      continue;
    }

    if (currentState === 'attribute') {
      if (code[i] === ']') {
        tokens.push({ value: ']', tokenClass: 'attribute', start: i, end: i + 1 });
        stateStack.pop();
        i += 1;
        continue;
      }
      const match = code.slice(i).match(/^[^\]]+/);
      if (match) {
        const val = match[0];
        tokens.push({ value: val, tokenClass: 'attribute', start: i, end: i + val.length });
        i += val.length;
        continue;
      }
      tokens.push({ value: code[i], tokenClass: 'attribute', start: i, end: i + 1 });
      i += 1;
      continue;
    }

    if (currentState === 'string') {
      if (code[i] === '"') {
        tokens.push({ value: '"', tokenClass: 'string', start: i, end: i + 1 });
        stateStack.pop();
        i += 1;
        continue;
      }
      if (code[i] === '\\' && i + 1 < len) {
        tokens.push({ value: code.slice(i, i + 2), tokenClass: 'string', start: i, end: i + 2 });
        i += 2;
        continue;
      }
      const nextQuote = code.indexOf('"', i);
      const nextSlash = code.indexOf('\\', i);
      let endIdx = len;
      if (nextQuote !== -1 && nextSlash !== -1) endIdx = Math.min(nextQuote, nextSlash);
      else if (nextQuote !== -1) endIdx = nextQuote;
      else if (nextSlash !== -1) endIdx = nextSlash;

      const val = code.slice(i, endIdx);
      tokens.push({ value: val, tokenClass: 'string', start: i, end: endIdx });
      i = endIdx;
      continue;
    }

    // State is 'root'
    const char = code[i];

    // Whitespace
    if (/\s/.test(char)) {
      i += 1;
      continue;
    }

    // Line comment: -- ...
    if (char === '-' && code[i + 1] === '-') {
      const eol = code.indexOf('\n', i);
      const commentEnd = eol === -1 ? len : eol;
      tokens.push({ value: code.slice(i, commentEnd), tokenClass: 'comment', start: i, end: commentEnd });
      i = commentEnd;
      continue;
    }

    // Block comment: /- ...
    if (char === '/' && code[i + 1] === '-') {
      tokens.push({ value: '/-', tokenClass: 'comment', start: i, end: i + 2 });
      stateStack.push('comment');
      i += 2;
      continue;
    }

    // Attribute start: @[ ...
    if (char === '@' && code[i + 1] === '[') {
      tokens.push({ value: '@[', tokenClass: 'attribute', start: i, end: i + 2 });
      stateStack.push('attribute');
      i += 2;
      continue;
    }

    // String literal start: "
    if (char === '"') {
      tokens.push({ value: '"', tokenClass: 'string', start: i, end: i + 1 });
      stateStack.push('string');
      i += 1;
      continue;
    }

    // Numbers: hex, binary, float, exponent, integer
    const numMatch = code.slice(i).match(/^(?:0[xX][0-9a-fA-F]+|0[bB][01]+|\d*\.\d+(?:[eE][-+]?\d+)?|\d+(?:[eE][-+]?\d+)?)/);
    if (numMatch && (char >= '0' && char <= '9' || (char === '.' && i + 1 < len && /\d/.test(code[i + 1])))) {
      const val = numMatch[0];
      tokens.push({ value: val, tokenClass: 'number', start: i, end: i + val.length });
      i += val.length;
      continue;
    }

    // Multi-character operators
    let matchedMulti = false;
    for (const op of MULTI_CHAR_OPERATORS) {
      if (code.slice(i, i + op.length) === op) {
        tokens.push({ value: op, tokenClass: 'operator', start: i, end: i + op.length });
        i += op.length;
        matchedMulti = true;
        break;
      }
    }
    if (matchedMulti) continue;

    // Unicode operators
    if (UNICODE_OPERATORS.has(char)) {
      tokens.push({ value: char, tokenClass: 'operator', start: i, end: i + 1 });
      i += 1;
      continue;
    }

    // Unicode delimiters: ⟨, ⟩, ⟦, ⟧
    if (UNICODE_DELIMITERS.has(char)) {
      tokens.push({ value: char, tokenClass: 'delimiter', start: i, end: i + 1 });
      i += 1;
      continue;
    }

    // ASCII delimiters: (, ), {, }, [, ], :, ,, ;, .
    if (ASCII_DELIMITERS.has(char)) {
      tokens.push({ value: char, tokenClass: 'delimiter', start: i, end: i + 1 });
      i += 1;
      continue;
    }

    // Single-char operators
    if (SINGLE_CHAR_OPERATORS.has(char)) {
      tokens.push({ value: char, tokenClass: 'operator', start: i, end: i + 1 });
      i += 1;
      continue;
    }

    // Identifiers, Keywords, TypeKeywords, DomainOperators
    const idMatch = code.slice(i).match(/^[a-zA-Z_][a-zA-Z0-9_']*/);
    if (idMatch) {
      const val = idMatch[0];
      let tokenClass: Lean4TokenClass = 'identifier';

      if (KEYWORD_SET.has(val)) {
        tokenClass = 'keyword';
      } else if (TYPE_KEYWORD_SET.has(val)) {
        tokenClass = 'type';
      } else if (DOMAIN_OPERATOR_SET.has(val)) {
        tokenClass = 'keyword';
      } else if (val === 'fun') {
        tokenClass = 'operator';
      }

      tokens.push({ value: val, tokenClass, start: i, end: i + val.length });
      i += val.length;
      continue;
    }

    // Fallback single character
    tokens.push({ value: char, tokenClass: 'delimiter', start: i, end: i + 1 });
    i += 1;
  }

  return tokens;
}
