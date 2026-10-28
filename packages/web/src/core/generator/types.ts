/**
 * In-Context Scoped Variation Generator - Type Definitions.
 *
 * Defines coordinate selection ranges, node traversal path elements,
 * match result structures, domain types, structural contexts, and
 * hole puncturing results.
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
 * Target domain classification for AST subtrees.
 */
export type DomainType = 'sdf' | 'audio' | 'hardware' | 'universal';

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

/**
 * Ancestral structural context extracted at the hole puncturing location.
 * Encodes the structural tuple (Parent, Grandparent, LeftSibling, Depth) for PCFG derivation.
 */
export interface StructuralContext {
  /** Immediate parent operation (e.g. 'difference', 'lowpass', 'seq', 'root') */
  readonly parentOp: string | null;
  /** Grandparent operation (e.g. 'translate', 'mix', 'root', or null if root has no grandparent) */
  readonly grandparentOp: string | null;
  /** Kind of immediate left sibling if in multi-branch node (e.g. 'box' in difference(box, cylinder)), or null */
  readonly leftSiblingOp: string | null;
  /** 1-indexed depth of target node in the root AST (root is 1) */
  readonly targetDepth: number;
  /** Expected domain of the punctured hole */
  readonly expectedDomain: DomainType;
  /** Expected formal AST type string (e.g. 'SDFNode', 'AudioNode', 'HardwareNode', 'UniversalNode') */
  readonly expectedType: string;
  /** Syntactic relation to parent (e.g. 'a', 'b', 'child', 'input_node', 'gate', 'first', 'second') */
  readonly parentRelation: string | null;
  /** Depth alias for convenience and backward compatibility */
  readonly depth?: number;
}

/**
 * Result of puncturing an AST node with a typed HoleNode.
 */
export interface PunctureResult {
  /** Unique identifier assigned to the hole (e.g. 'hole_1') */
  readonly holeId: string;
  /** Typed hole node placed into the punctured AST */
  readonly holeNode: ASTNode;
  /** Functional immutable clone of the root AST with target replaced by hole */
  readonly puncturedRootAST: ASTNode;
  /** Original target AST node that was replaced */
  readonly originalTargetNode: ASTNode;
  /** Extracted structural context tuple */
  readonly context: StructuralContext;
  /** Exact source coordinate span of the target node */
  readonly targetSpan: SourceSpan;
}
