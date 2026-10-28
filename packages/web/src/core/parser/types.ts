/**
 * Lean 4 AST Parser & Pretty-Printer Type Definitions.
 *
 * Defines source coordinate spans, lexical tokens, parse errors,
 * parse results, and format options for bidirectional Lean 4 <-> AST transformations.
 */

import type { ASTNode } from '../ast/types';

/**
 * Exact source code range coordinates (lines and cols 1-indexed, offsets 0-indexed).
 */
export interface SourceSpan {
  readonly startLine: number;
  readonly startCol: number;
  readonly endLine: number;
  readonly endCol: number;
  readonly startOffset: number;
  readonly endOffset: number;
}

/**
 * Diagnostic error produced during lexical analysis or AST parsing.
 */
export interface ParseError {
  readonly message: string;
  readonly span: SourceSpan;
  readonly line: number;
  readonly col: number;
}

/**
 * Comprehensive result of parsing a Lean 4 declaration or expression into an AST.
 */
export interface ParseResult {
  readonly declarationName: string;
  readonly domainType: string;
  readonly ast: ASTNode;
  readonly spans: Map<string, SourceSpan>;
  readonly errors: ParseError[];
  readonly nodeSpanMap?: WeakMap<object, SourceSpan>;
  readonly nodeIdMap?: WeakMap<object, string>;
}

/**
 * Supported token categories emitted by the Lean 4 lexer.
 */
export type TokenType =
  | 'Keyword'
  | 'Identifier'
  | 'Number'
  | 'String'
  | 'Operator'
  | 'Delimiter'
  | 'EOF';

/**
 * Individual lexical token with exact source coordinates.
 */
export interface Token {
  readonly type: TokenType;
  readonly value: string;
  readonly span: SourceSpan;
  readonly numberValue?: number;
}

/**
 * Configuration options for the Lean 4 pretty-printer.
 */
export interface FormatOptions {
  /** Number of spaces for each indentation level (default: 2) */
  readonly indent?: number;
  /** Whether to format into a single compact line (default: false) */
  readonly compact?: boolean;
  /** Whether to emit dot-notation constructors e.g. `.sphere` instead of `SDFNode.sphere` (default: false) */
  readonly dotNotation?: boolean;
}

/**
 * Helper to construct a frozen SourceSpan object.
 */
export function createSourceSpan(
  startLine: number,
  startCol: number,
  endLine: number,
  endCol: number,
  startOffset: number,
  endOffset: number
): SourceSpan {
  return Object.freeze({
    startLine,
    startCol,
    endLine,
    endCol,
    startOffset,
    endOffset,
  });
}

/**
 * Combine two spans to span the entire range from start of first to end of second.
 */
export function combineSpans(start: SourceSpan, end: SourceSpan): SourceSpan {
  return createSourceSpan(
    start.startLine,
    start.startCol,
    end.endLine,
    end.endCol,
    start.startOffset,
    end.endOffset
  );
}
