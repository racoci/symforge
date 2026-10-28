/**
 * Monaco Selection Bounding & AST Range Mapper.
 *
 * Resolves editor coordinate selections to the minimal syntactically complete
 * enclosing AST subtree with syntax boundary expansion and parent relation tracking.
 */

import type { ASTNode } from '../ast/types';
import type { SourceSpan } from '../parser/types';
import type {
  SelectionRange,
  NodeMatchResult,
  ASTNodeMetadata,
} from './types';

/**
 * Translates Monaco 1-indexed line and column coordinates to a 0-indexed character offset.
 */
export function lineColToOffset(code: string, line: number, col: number): number {
  if (line <= 1) {
    return Math.min(Math.max(col - 1, 0), code.length);
  }

  let currentLine = 1;
  let currentOffset = 0;

  while (currentOffset < code.length && currentLine < line) {
    if (code.charCodeAt(currentOffset) === 10) {
      currentLine++;
    }
    currentOffset++;
  }

  return Math.min(currentOffset + Math.max(col - 1, 0), code.length);
}

/**
 * Normalizes selection coordinates into an offset-bounded SelectionRange.
 */
export function computeSelectionRange(
  raw: {
    startLine: number;
    startCol: number;
    endLine: number;
    endCol: number;
    startOffset?: number;
    endOffset?: number;
    selectedText?: string;
  },
  code?: string
): SelectionRange {
  let startOffset = raw.startOffset ?? 0;
  let endOffset = raw.endOffset ?? 0;

  const needsOffsetComputation =
    code !== undefined &&
    (raw.startOffset === undefined ||
      raw.endOffset === undefined ||
      (startOffset === 0 && endOffset === 0 && (raw.startLine > 1 || raw.startCol > 1)));

  if (needsOffsetComputation && code) {
    startOffset = lineColToOffset(code, raw.startLine, raw.startCol);
    endOffset = lineColToOffset(code, raw.endLine, raw.endCol);
  }

  return {
    startLine: raw.startLine,
    startCol: raw.startCol,
    endLine: raw.endLine,
    endCol: raw.endCol,
    startOffset: Math.min(startOffset, endOffset),
    endOffset: Math.max(startOffset, endOffset),
  };
}

/**
 * Formats an ASTNode kind into a human-readable PascalCase display name.
 */
export function formatNodeDisplayName(node: ASTNode): string {
  if (!node || typeof node !== 'object' || !('kind' in node)) {
    return 'ASTNode';
  }
  const kind = String(node.kind);
  if (kind === 'lowpass') return 'LowPassNode';
  if (kind === 'adsr') return 'ADSRNode';
  if (kind === 'fm') return 'FMNode';
  if (kind === 'sdf') return 'SDFNode';
  const parts = kind.split('_');
  const capitalized = parts
    .map((p) => p.charAt(0).toUpperCase() + p.slice(1))
    .join('');
  return capitalized.endsWith('Node') ? capitalized : `${capitalized}Node`;
}

/**
 * Extracts child AST subtrees and their syntactic relations from any valid ASTNode.
 */
function getChildNodes(node: ASTNode): Array<{ child: ASTNode; relation: string; key: string }> {
  if (!node || typeof node !== 'object' || !('kind' in node)) {
    return [];
  }

  switch (node.kind) {
    // 1. Procedural Audio AST Nodes
    case 'adsr':
      return [{ child: node.gate, relation: 'gate', key: 'gate' }];
    case 'lowpass':
      return [{ child: node.input, relation: 'input_node', key: 'input' }];
    case 'gain':
      return [{ child: node.input, relation: 'input_node', key: 'input' }];
    case 'mix':
      return [
        { child: node.input1, relation: 'input1', key: 'input1' },
        { child: node.input2, relation: 'input2', key: 'input2' },
      ];

    // 2. 3D SDF AST Nodes
    case 'translate':
    case 'rotate':
    case 'scale':
      return [{ child: node.child, relation: 'child', key: 'child' }];
    case 'union':
    case 'smooth_union':
    case 'intersection':
    case 'difference':
      return [
        { child: node.a, relation: 'a', key: 'a' },
        { child: node.b, relation: 'b', key: 'b' },
      ];

    // 3. Hardware Nodes
    case 'add':
    case 'sub':
    case 'mul':
      return [
        { child: node.src1, relation: 'src1', key: 'src1' },
        { child: node.src2, relation: 'src2', key: 'src2' },
      ];
    case 'shl':
    case 'shr':
      return [{ child: node.src, relation: 'src', key: 'src' }];
    case 'seq':
      return [
        { child: node.first, relation: 'first', key: 'first' },
        { child: node.second, relation: 'second', key: 'second' },
      ];

    // 4. Universal Nodes
    case 'universal_audio':
    case 'universal_sdf':
    case 'universal_hardware':
      return [{ child: node.node, relation: 'node', key: 'node' }];
    case 'universal_macro':
      return node.children.map((c, idx) => ({
        child: c as unknown as ASTNode,
        relation: 'children',
        key: String(idx),
      }));

    // 5. Generic Macro Call Node
    case 'macro': {
      const res: Array<{ child: ASTNode; relation: string; key: string }> = [];
      if (node.arguments && typeof node.arguments === 'object') {
        for (const [k, v] of Object.entries(node.arguments)) {
          if (v && typeof v === 'object' && 'kind' in v) {
            res.push({ child: v as ASTNode, relation: k, key: k });
          }
        }
      }
      return res;
    }

    default:
      return [];
  }
}

/**
 * Resolves the source coordinate span of an AST node via span maps or heuristics.
 */
function resolveNodeSpan(
  node: ASTNode,
  relation: string | null,
  path: string[],
  postOrderIdx: number,
  spans: Map<string, SourceSpan>
): SourceSpan | undefined {
  // 1. WeakMap attached to spans by parser
  const nodeSpanMap = (spans as any)?.nodeSpanMap;
  if (nodeSpanMap && typeof nodeSpanMap.get === 'function') {
    const s = nodeSpanMap.get(node);
    if (s) return s;
  }

  // 2. Direct map lookup if object used as key
  if (typeof (spans as any).get === 'function') {
    const direct = (spans as any).get(node);
    if (direct) return direct;
  }

  // 3. Hole / Explicit IDs
  if ('holeId' in node && spans.has((node as any).holeId)) {
    return spans.get((node as any).holeId);
  }
  if ('id' in node && spans.has((node as any).id)) {
    return spans.get((node as any).id);
  }

  // 4. NodeId map lookup
  const nodeIdMap = (spans as any)?.nodeIdMap;
  if (nodeIdMap && typeof nodeIdMap.get === 'function') {
    const nid = nodeIdMap.get(node);
    if (nid && spans.has(nid)) return spans.get(nid);
  }

  // 5. Path or relation key lookup
  const pathKey = path.join('.');
  if (pathKey && spans.has(pathKey)) return spans.get(pathKey);
  if (relation && spans.has(relation)) return spans.get(relation);

  // 6. Post-order counter
  const postOrderKey = `node_${postOrderIdx}`;
  if (spans.has(postOrderKey)) return spans.get(postOrderKey);

  // 7. Node kind key
  if (spans.has(node.kind)) return spans.get(node.kind);

  // 8. Root key if at top level
  if (path.length === 0 && spans.has('root')) return spans.get('root');

  return undefined;
}

/**
 * Resolves the unique node identifier for a given AST node.
 */
function resolveNodeId(
  node: ASTNode,
  span: SourceSpan | undefined,
  path: string[],
  postOrderIdx: number,
  spans: Map<string, SourceSpan>
): string {
  const nodeIdMap = (spans as any)?.nodeIdMap;
  if (nodeIdMap && typeof nodeIdMap.get === 'function') {
    const id = nodeIdMap.get(node);
    if (id) return id;
  }

  if ('holeId' in node && typeof (node as any).holeId === 'string') {
    return (node as any).holeId;
  }
  if ('id' in node && typeof (node as any).id === 'string') {
    return (node as any).id;
  }

  if (span) {
    for (const [key, s] of spans.entries()) {
      if (s === span && key.startsWith('node_')) {
        return key;
      }
    }
  }

  if (path.length === 0) return 'root';
  return `node_${postOrderIdx}`;
}

/**
 * Traverses an AST tree depth-first and collects all nodes with metadata in post-order.
 */
function collectASTNodes(
  node: ASTNode,
  parentNode: ASTNode | null,
  parentRelation: string | null,
  path: string[],
  depth: number,
  spans: Map<string, SourceSpan>,
  counterRef: { count: number },
  resultList: ASTNodeMetadata[]
): void {
  const children = getChildNodes(node);

  for (const childInfo of children) {
    const nextRelation = childInfo.relation;
    const nextPath = [...path, nextRelation];
    collectASTNodes(
      childInfo.child,
      node,
      nextRelation,
      nextPath,
      depth + 1,
      spans,
      counterRef,
      resultList
    );
  }

  const postOrderIdx = counterRef.count++;
  const span = resolveNodeSpan(node, parentRelation, path, postOrderIdx, spans);
  const nodeId = resolveNodeId(node, span, path, postOrderIdx, spans);

  if (span) {
    resultList.push({
      node,
      parentNode,
      parentRelation,
      nodePath: path,
      depth,
      span,
      nodeId,
      postOrderIdx,
    });
  }
}

interface EvaluatedCandidate {
  readonly meta: ASTNodeMetadata;
  readonly nodeLen: number;
  readonly depth: number;
  readonly overlapRatio: number;
  readonly jaccard: number;
  readonly encloses: boolean;
  readonly isExactMatch: boolean;
}

/**
 * Calculates overlap metrics for a candidate node against the user selection.
 */
function evaluateCandidate(
  meta: ASTNodeMetadata,
  sStart: number,
  sEnd: number,
  trimmedStart: number,
  trimmedEnd: number
): EvaluatedCandidate | null {
  const span = meta.span;
  const nodeLen = Math.max(0, span.endOffset - span.startOffset);
  if (nodeLen <= 0) return null;

  const selectionLen = sEnd - sStart;
  const trimmedLen = trimmedEnd - trimmedStart;

  const rawIntersection = Math.max(0, Math.min(sEnd, span.endOffset) - Math.max(sStart, span.startOffset));
  const trimmedIntersection = Math.max(0, Math.min(trimmedEnd, span.endOffset) - Math.max(trimmedStart, span.startOffset));
  const effectiveIntersection = Math.max(rawIntersection, trimmedIntersection);

  const overlapRatio = selectionLen > 0 ? effectiveIntersection / selectionLen : 0;
  if (overlapRatio < 0.15) {
    return null;
  }

  const enclosesRaw = span.startOffset <= sStart && sEnd <= span.endOffset;
  const enclosesTrimmed = span.startOffset <= trimmedStart && trimmedEnd <= span.endOffset;
  const encloses = enclosesRaw || enclosesTrimmed;

  const rawUnion = selectionLen + nodeLen - rawIntersection;
  const rawJaccard = rawUnion > 0 ? rawIntersection / rawUnion : 0;

  const trimmedUnion = trimmedLen + nodeLen - trimmedIntersection;
  const trimmedJaccard = trimmedUnion > 0 ? trimmedIntersection / trimmedUnion : 0;
  const jaccard = Math.max(rawJaccard, trimmedJaccard);

  if (!encloses && jaccard < 0.7) {
    return null;
  }

  const isExactMatch =
    (sStart === span.startOffset && sEnd === span.endOffset) ||
    (trimmedStart === span.startOffset && trimmedEnd === span.endOffset);

  return {
    meta,
    nodeLen,
    depth: meta.depth,
    overlapRatio,
    jaccard,
    encloses,
    isExactMatch,
  };
}

/**
 * Discovers the minimal syntactically complete enclosing AST node corresponding
 * to a Monaco selection range, with syntax boundary expansion and relation tracking.
 */
export function findEnclosingASTNode(
  ast: ASTNode,
  spans: Map<string, SourceSpan>,
  selection: SelectionRange,
  code?: string
): NodeMatchResult | null {
  if (!ast || !spans || !selection) {
    return null;
  }

  const normalized = computeSelectionRange(selection, code);
  const sStart = normalized.startOffset;
  const sEnd = normalized.endOffset;
  const selectionLen = sEnd - sStart;

  if (selectionLen <= 0) {
    return null;
  }

  let trimmedStart = sStart;
  let trimmedEnd = sEnd;

  if (code) {
    const selectedText = code.slice(sStart, sEnd);
    if (selectedText.trim().length === 0) {
      return null;
    }
    const leadingWs = selectedText.length - selectedText.trimStart().length;
    const trailingWs = selectedText.length - selectedText.trimEnd().length;
    trimmedStart = sStart + leadingWs;
    trimmedEnd = sEnd - trailingWs;
    if (trimmedEnd <= trimmedStart) {
      return null;
    }
  }

  const candidateNodes: ASTNodeMetadata[] = [];
  collectASTNodes(ast, null, null, [], 0, spans, { count: 0 }, candidateNodes);

  if (candidateNodes.length === 0) {
    return null;
  }

  const qualifying: EvaluatedCandidate[] = [];
  for (const meta of candidateNodes) {
    const evaluated = evaluateCandidate(meta, sStart, sEnd, trimmedStart, trimmedEnd);
    if (evaluated) {
      qualifying.push(evaluated);
    }
  }

  if (qualifying.length === 0) {
    return null;
  }

  const enclosing = qualifying.filter((c) => c.encloses);
  let best: EvaluatedCandidate;

  if (enclosing.length > 0) {
    enclosing.sort((a, b) => {
      if (a.nodeLen !== b.nodeLen) return a.nodeLen - b.nodeLen;
      if (a.depth !== b.depth) return b.depth - a.depth;
      return b.jaccard - a.jaccard;
    });
    best = enclosing[0];
  } else {
    qualifying.sort((a, b) => {
      if (Math.abs(b.jaccard - a.jaccard) > 0.001) return b.jaccard - a.jaccard;
      if (a.depth !== b.depth) return b.depth - a.depth;
      return a.nodeLen - b.nodeLen;
    });
    best = qualifying[0];
  }

  let enclosingDeclaration = (spans as any)?.declarationName || '';
  if (!enclosingDeclaration && code) {
    const declMatch = code.match(/(?:def|theorem|example)\s+([A-Za-z0-9_'.]+)/);
    if (declMatch) {
      enclosingDeclaration = declMatch[1];
    }
  }
  if (!enclosingDeclaration) {
    enclosingDeclaration = 'anonymous';
  }

  return {
    targetNode: best.meta.node,
    nodeId: best.meta.nodeId,
    nodePath: best.meta.nodePath,
    span: best.meta.span,
    isExactMatch: best.isExactMatch,
    overlapRatio: Math.min(1.0, best.overlapRatio),
    parentNode: best.meta.parentNode,
    parentRelation: best.meta.parentRelation,
    enclosingDeclaration,
  };
}
