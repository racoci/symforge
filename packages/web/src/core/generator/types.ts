/**
 * In-Context Scoped Variation Generator - Type Definitions.
 *
 * Defines coordinate selection ranges, node traversal path elements,
 * match result structures, and query options for AST selection bounding.
 */

import type { ASTNode } from '../ast/types';
import type { SourceSpan } from '../parser/types';

/**
 * Normalized 2D/1D selection range in editor coordinates.
 * Lines and columns are 1-indexed (Monaco standard), offsets are 0-indexed character offsets.
 */
export interface SelectionRange {
  readonly startLine: number;
  readonly startCol: number;
  readonly endLine: number;
  readonly endCol: number;
  readonly startOffset: number;
  readonly endOffset: number;
}

/**
 * Structural element in an AST descent path.
 */
export interface NodePathElement {
  readonly key: string;
  readonly index?: number;
}

/**
 * Result of resolving a user's text selection to the minimal syntactically complete enclosing AST node.
 */
export interface NodeMatchResult {
  /** The minimal syntactically complete enclosing AST subtree */
  readonly targetNode: ASTNode;
  /** Unique identifier for the matched node */
  readonly nodeId: string;
  /** Hierarchical path from root (e.g. ["b"] or ["child", "a"]) */
  readonly nodePath: string[];
  /** Exact source coordinate span */
  readonly span: SourceSpan;
  /** Whether the selection boundary matches the node boundary exactly */
  readonly isExactMatch: boolean;
  /** Ratio of selection overlapping the node [0.0, 1.0] */
  readonly overlapRatio: number;
  /** Immediate parent AST node, or null if targetNode is root */
  readonly parentNode: ASTNode | null;
  /** Syntactic relation to parent (e.g. 'a', 'b', 'child', 'input_node', 'gate') */
  readonly parentRelation: string | null;
  /** Enclosing top-level Lean declaration name (e.g. 'myGyroidSDF') */
  readonly enclosingDeclaration: string;
}

/**
 * Internal metadata associated with an AST node during interval search.
 */
export interface ASTNodeMetadata {
  readonly node: ASTNode;
  readonly parentNode: ASTNode | null;
  readonly parentRelation: string | null;
  readonly nodePath: string[];
  readonly depth: number;
  readonly span: SourceSpan;
  readonly nodeId: string;
  readonly postOrderIdx: number;
}
