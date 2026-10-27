/**
 * Lean 4 AST Lexical Tokenizer.
 *
 * Scans Lean 4 source strings into structured tokens with exact line, column,
 * and character offset coordinates. Strips comments (-- and /- ... -/) deterministically.
 */

import {
  createSourceSpan,
  type Token,
  type TokenType,
  type ParseError,
} from './types';

const LEAN_KEYWORDS = new Set([
  'def',
  'let',
  'in',
  'open',
  'import',
  'namespace',
  'end',
  'where',
  'inductive',
  'structure',
  'deriving',
  'theorem',
  'example',
  'by',
  'cases',
  'simp',
  'omega',
]);

const DELIMITERS = new Set(['(', ')', '[', ']', '{', '}', '⟨', '⟩', '⟦', '⟧', ',']);

/**
 * State container for lexical scanning.
 */
interface LexerState {
  source: string;
  cursor: number;
  line: number;
  col: number;
  length: number;
  tokens: Token[];
  errors: ParseError[];
}

function isDigit(ch: string): boolean {
  return ch >= '0' && ch <= '9';
}

function isAlpha(ch: string): boolean {
  return (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') || ch === '_';
}

function isIdentChar(ch: string): boolean {
  return isAlpha(ch) || isDigit(ch) || ch === "'";
}

function peekChar(state: LexerState, offset = 0): string {
  const idx = state.cursor + offset;
  return idx < state.length ? state.source[idx] : '';
}

function advance(state: LexerState): string {
  const ch = state.source[state.cursor];
  state.cursor++;
  if (ch === '\n') {
    state.line++;
    state.col = 1;
  } else {
    state.col++;
  }
  return ch;
}

function skipWhitespace(state: LexerState): void {
  const maxIterations = state.length * 2;
  let iter = 0;
  while (state.cursor < state.length && iter++ < maxIterations) {
    const ch = peekChar(state);
    if (ch === ' ' || ch === '\t' || ch === '\r' || ch === '\n') {
      advance(state);
      continue;
    }
    // Line comment: -- ...
    if (ch === '-' && peekChar(state, 1) === '-') {
      skipLineComment(state);
      continue;
    }
    // Block comment: /- ... -/ (supports nesting)
    if (ch === '/' && peekChar(state, 1) === '-') {
      skipBlockComment(state);
      continue;
    }
    break;
  }
}

function skipLineComment(state: LexerState): void {
  advance(state); // consume '-'
  advance(state); // consume '-'
  const maxIterations = state.length;
  let iter = 0;
  while (state.cursor < state.length && iter++ < maxIterations) {
    const ch = peekChar(state);
    if (ch === '\n') {
      advance(state);
      break;
    }
    advance(state);
  }
}

function skipBlockComment(state: LexerState): void {
  const startLine = state.line;
  const startCol = state.col;
  const startOffset = state.cursor;
  advance(state); // consume '/'
  advance(state); // consume '-'
  let depth = 1;
  const maxIterations = state.length * 2;
  let iter = 0;

  while (state.cursor < state.length && depth > 0 && iter++ < maxIterations) {
    const ch = peekChar(state);
    const next = peekChar(state, 1);
    if (ch === '/' && next === '-') {
      depth++;
      advance(state);
      advance(state);
      continue;
    }
    if (ch === '-' && next === '/') {
      depth--;
      advance(state);
      advance(state);
      continue;
    }
    advance(state);
  }

  if (depth > 0) {
    state.errors.push({
      message: 'Unterminated block comment',
      span: createSourceSpan(startLine, startCol, state.line, state.col, startOffset, state.cursor),
      line: startLine,
      col: startCol,
    });
  }
}

function readStringToken(state: LexerState): void {
  const startLine = state.line;
  const startCol = state.col;
  const startOffset = state.cursor;
  advance(state); // consume opening quote

  let content = '';
  let closed = false;
  const maxIterations = state.length;
  let iter = 0;

  while (state.cursor < state.length && iter++ < maxIterations) {
    const ch = peekChar(state);
    if (ch === '"') {
      advance(state);
      closed = true;
      break;
    }
    if (ch === '\\') {
      advance(state);
      const esc = advance(state);
      if (esc === 'n') content += '\n';
      else if (esc === 't') content += '\t';
      else if (esc === 'r') content += '\r';
      else content += esc;
      continue;
    }
    content += advance(state);
  }

  const span = createSourceSpan(startLine, startCol, state.line, state.col, startOffset, state.cursor);
  if (!closed) {
    state.errors.push({
      message: 'Unterminated string literal',
      span,
      line: startLine,
      col: startCol,
    });
  }
  state.tokens.push({ type: 'String', value: content, span });
}

function readNumberToken(state: LexerState): void {
  const startLine = state.line;
  const startCol = state.col;
  const startOffset = state.cursor;
  let text = '';

  if (peekChar(state) === '-') {
    text += advance(state);
  }

  const maxDigits = state.length;
  let iter = 0;
  while (state.cursor < state.length && isDigit(peekChar(state)) && iter++ < maxDigits) {
    text += advance(state);
  }

  // Fractional part
  if (peekChar(state) === '.' && isDigit(peekChar(state, 1))) {
    text += advance(state); // '.'
    while (state.cursor < state.length && isDigit(peekChar(state)) && iter++ < maxDigits) {
      text += advance(state);
    }
  }

  // Scientific notation exponent
  const nextExp = peekChar(state);
  if ((nextExp === 'e' || nextExp === 'E') && (isDigit(peekChar(state, 1)) || peekChar(state, 1) === '+' || peekChar(state, 1) === '-')) {
    text += advance(state);
    if (peekChar(state) === '+' || peekChar(state) === '-') {
      text += advance(state);
    }
    while (state.cursor < state.length && isDigit(peekChar(state)) && iter++ < maxDigits) {
      text += advance(state);
    }
  }

  const numVal = parseFloat(text);
  const span = createSourceSpan(startLine, startCol, state.line, state.col, startOffset, state.cursor);
  state.tokens.push({ type: 'Number', value: text, span, numberValue: numVal });
}

function readIdentOrKeyword(state: LexerState): void {
  const startLine = state.line;
  const startCol = state.col;
  const startOffset = state.cursor;
  let text = '';

  const maxLen = state.length;
  let iter = 0;
  while (state.cursor < state.length && isIdentChar(peekChar(state)) && iter++ < maxLen) {
    text += advance(state);
  }

  const span = createSourceSpan(startLine, startCol, state.line, state.col, startOffset, state.cursor);
  const type: TokenType = LEAN_KEYWORDS.has(text) ? 'Keyword' : 'Identifier';
  state.tokens.push({ type, value: text, span });
}

function isNegativeNumberStart(state: LexerState): boolean {
  if (peekChar(state) !== '-') return false;
  if (!isDigit(peekChar(state, 1))) return false;
  const prevToken = state.tokens[state.tokens.length - 1];
  if (!prevToken) return true;
  // If preceded by an identifier, number, or closing delimiter, '-' is subtraction operator
  if (prevToken.type === 'Identifier' || prevToken.type === 'Number') return false;
  if (prevToken.type === 'Delimiter' && (prevToken.value === ')' || prevToken.value === ']' || prevToken.value === '}')) {
    return false;
  }
  return true;
}

function readOperatorOrDelimiter(state: LexerState): void {
  const startLine = state.line;
  const startCol = state.col;
  const startOffset = state.cursor;
  const ch = peekChar(state);

  // Check delimiter
  if (DELIMITERS.has(ch)) {
    advance(state);
    const span = createSourceSpan(startLine, startCol, state.line, state.col, startOffset, state.cursor);
    state.tokens.push({ type: 'Delimiter', value: ch, span });
    return;
  }

  // Check multi-char operators: ':=', '->'
  if (ch === ':' && peekChar(state, 1) === '=') {
    advance(state);
    advance(state);
    const span = createSourceSpan(startLine, startCol, state.line, state.col, startOffset, state.cursor);
    state.tokens.push({ type: 'Operator', value: ':=', span });
    return;
  }
  if (ch === '-' && peekChar(state, 1) === '>') {
    advance(state);
    advance(state);
    const span = createSourceSpan(startLine, startCol, state.line, state.col, startOffset, state.cursor);
    state.tokens.push({ type: 'Operator', value: '->', span });
    return;
  }

  // Single-character operator
  advance(state);
  const span = createSourceSpan(startLine, startCol, state.line, state.col, startOffset, state.cursor);
  state.tokens.push({ type: 'Operator', value: ch, span });
}

/**
 * Tokenize a Lean 4 source string into typed tokens and parse errors.
 */
export function tokenize(source: string): { tokens: Token[]; errors: ParseError[] } {
  const state: LexerState = {
    source,
    cursor: 0,
    line: 1,
    col: 1,
    length: source.length,
    tokens: [],
    errors: [],
  };

  const maxTotalTokens = source.length + 10;
  let count = 0;

  while (state.cursor < state.length && count++ < maxTotalTokens) {
    skipWhitespace(state);
    if (state.cursor >= state.length) break;

    const ch = peekChar(state);

    if (ch === '"') {
      readStringToken(state);
    } else if (isNegativeNumberStart(state) || isDigit(ch)) {
      readNumberToken(state);
    } else if (isAlpha(ch)) {
      readIdentOrKeyword(state);
    } else {
      readOperatorOrDelimiter(state);
    }
  }

  const eofSpan = createSourceSpan(state.line, state.col, state.line, state.col, state.cursor, state.cursor);
  state.tokens.push({ type: 'EOF', value: '', span: eofSpan });

  return { tokens: state.tokens, errors: state.errors };
}
