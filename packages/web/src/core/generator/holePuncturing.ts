/**
 * Type Resolution, Hole Puncturing (`Hole[T]`) & Structural Context Extractor.
 *
 * Implements functional immutable hole puncturing, domain type resolution,
 * ancestral structural context extraction (Parent, Grandparent, LeftSibling, Depth),
 * and round-trip hole substitution with strict domain invariance verification.
 */

import type {
  ASTNode,
  AudioNode,
  SDFNode,
  HardwareNode,
} from '../ast/types';
import {
  isAudioNode,
  isSDFNode,
  isHardwareNode,
  isUniversalNode,
  areNodesEqual,
  createADSR,
  createLowPass,
  createGain,
  createMix,
  createTranslate,
  createRotate,
  createScale,
  createUnion,
  createSmoothUnion,
  createIntersection,
  createDifference,
  createAdd,
  createSub,
  createMul,
  createShl,
  createShr,
  createSeq,
  createUniversalAudio,
  createUniversalSDF,
  createUniversalHardware,
  createUniversalMacro,
  createMacroCall,
  createAudioHole,
  createSDFHole,
  createHardwareHole,
  createHole,
} from '../ast/models';
import type {
  DomainType,
  NodeMatchResult,
  StructuralContext,
  PunctureResult,
} from './types';

/**
 * Child entry representing a recursive AST branch and its syntactic relation.
 */
export interface ChildEntry {
  readonly child: ASTNode;
  readonly relation: string;
  readonly key: string;
}

/**
 * Internal traversal context for locating a target node within an AST tree.
 */
interface TraversalContext {
  readonly targetDepth: number;
  readonly parentOp: string | null;
  readonly grandparentOp: string | null;
  readonly leftSiblingOp: string | null;
  readonly parentRelation: string | null;
  readonly targetNode: ASTNode;
}

/**
 * Resolves the domain type ('sdf' | 'audio' | 'hardware' | 'universal') of any ASTNode.
 */
export function resolveDomainType(node: ASTNode): DomainType {
  if (!node || typeof node !== 'object') {
    return 'universal';
  }

  // Explicit domain on hole or macro
  if ('domain' in node && typeof (node as any).domain === 'string') {
    const d = (node as any).domain.toLowerCase();
    if (d === 'sdf') return 'sdf';
    if (d === 'audio') return 'audio';
    if (d === 'hardware') return 'hardware';
    if (d === 'universal') return 'universal';
  }

  // Universal containers
  if (isUniversalNode(node)) {
    if (node.kind === 'universal_sdf') return 'sdf';
    if (node.kind === 'universal_audio') return 'audio';
    if (node.kind === 'universal_hardware') return 'hardware';
    return 'universal';
  }

  // Standard SDF nodes
  if (isSDFNode(node)) return 'sdf';

  // Standard Audio nodes
  if (isAudioNode(node)) return 'audio';

  // Standard Hardware nodes
  if (isHardwareNode(node)) return 'hardware';

  return 'universal';
}

/**
 * Formats domain type into standard expected type name (e.g. 'SDFNode', 'AudioNode').
 */
export function resolveExpectedType(domain: DomainType): string {
  switch (domain) {
    case 'sdf':
      return 'SDFNode';
    case 'audio':
      return 'AudioNode';
    case 'hardware':
      return 'HardwareNode';
    case 'universal':
    default:
      return 'UniversalNode';
  }
}

/**
 * Extracts child AST subtrees and their syntactic relations from any valid ASTNode.
 */
export function getChildEntries(node: ASTNode): ChildEntry[] {
  if (!node || typeof node !== 'object' || !('kind' in node)) {
    return [];
  }

  switch (node.kind) {
    // 1. Procedural Audio
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

    // 2. 3D SDF
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

    // 3. Hardware
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

    // 4. Universal Containers
    case 'universal_audio':
      return [{ child: node.node, relation: 'node', key: 'node' }];
    case 'universal_sdf':
      return [{ child: node.node, relation: 'node', key: 'node' }];
    case 'universal_hardware':
      return [{ child: node.node, relation: 'node', key: 'node' }];
    case 'universal_macro':
      return node.children.map((c, idx) => ({
        child: c as unknown as ASTNode,
        relation: `children[${idx}]`,
        key: String(idx),
      }));

    // 5. Generic Macro
    case 'macro': {
      const res: ChildEntry[] = [];
      if (node.arguments && typeof node.arguments === 'object') {
        const sortedKeys = Object.keys(node.arguments).sort();
        for (const k of sortedKeys) {
          const v = node.arguments[k];
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
 * Reconstructs a parent AST node with a specific child updated, preserving immutability.
 */
export function rebuildNodeWithUpdatedChild(
  parent: ASTNode,
  relationOrKey: string,
  updatedChild: ASTNode
): ASTNode {
  switch (parent.kind) {
    // Audio composite
    case 'adsr':
      return createADSR(parent.attack, parent.decay, parent.sustain, parent.release, updatedChild as AudioNode);
    case 'lowpass':
      return createLowPass(parent.cutoff, parent.resonance, updatedChild as AudioNode);
    case 'gain':
      return createGain(parent.gain, updatedChild as AudioNode);
    case 'mix':
      if (relationOrKey === 'input1') {
        return createMix(parent.ratio, updatedChild as AudioNode, parent.input2);
      }
      return createMix(parent.ratio, parent.input1, updatedChild as AudioNode);

    // SDF composite
    case 'translate':
      return createTranslate(parent.dx, parent.dy, parent.dz, updatedChild as SDFNode);
    case 'rotate':
      return createRotate(parent.rx, parent.ry, parent.rz, updatedChild as SDFNode);
    case 'scale':
      return createScale(parent.s, updatedChild as SDFNode);
    case 'union':
      if (relationOrKey === 'a') {
        return createUnion(parent.smoothness, updatedChild as SDFNode, parent.b);
      }
      return createUnion(parent.smoothness, parent.a, updatedChild as SDFNode);
    case 'smooth_union':
      if (relationOrKey === 'a') {
        return createSmoothUnion(parent.smoothness, updatedChild as SDFNode, parent.b);
      }
      return createSmoothUnion(parent.smoothness, parent.a, updatedChild as SDFNode);
    case 'intersection':
      if (relationOrKey === 'a') {
        return createIntersection(updatedChild as SDFNode, parent.b);
      }
      return createIntersection(parent.a, updatedChild as SDFNode);
    case 'difference':
      if (relationOrKey === 'a') {
        return createDifference(updatedChild as SDFNode, parent.b);
      }
      return createDifference(parent.a, updatedChild as SDFNode);

    // Hardware composite
    case 'add':
      if (relationOrKey === 'src1') {
        return createAdd(updatedChild as HardwareNode, parent.src2, parent.dst);
      }
      return createAdd(parent.src1, updatedChild as HardwareNode, parent.dst);
    case 'sub':
      if (relationOrKey === 'src1') {
        return createSub(updatedChild as HardwareNode, parent.src2, parent.dst);
      }
      return createSub(parent.src1, updatedChild as HardwareNode, parent.dst);
    case 'mul':
      if (relationOrKey === 'src1') {
        return createMul(updatedChild as HardwareNode, parent.src2, parent.dst);
      }
      return createMul(parent.src1, updatedChild as HardwareNode, parent.dst);
    case 'shl':
      return createShl(updatedChild as HardwareNode, parent.shift, parent.dst);
    case 'shr':
      return createShr(updatedChild as HardwareNode, parent.shift, parent.dst);
    case 'seq':
      if (relationOrKey === 'first') {
        return createSeq(updatedChild as HardwareNode, parent.second);
      }
      return createSeq(parent.first, updatedChild as HardwareNode);

    // Universal containers
    case 'universal_audio':
      return createUniversalAudio(updatedChild as AudioNode);
    case 'universal_sdf':
      return createUniversalSDF(updatedChild as SDFNode);
    case 'universal_hardware':
      return createUniversalHardware(updatedChild as HardwareNode);
    case 'universal_macro': {
      const idx = parseInt(relationOrKey.replace(/[^0-9]/g, ''), 10);
      const newChildren = [...parent.children];
      if (!isNaN(idx) && idx >= 0 && idx < newChildren.length) {
        newChildren[idx] = updatedChild as any;
      }
      return createUniversalMacro(parent.name, newChildren);
    }

    // Generic Macro
    case 'macro': {
      const newArgs = { ...parent.arguments, [relationOrKey]: updatedChild };
      return createMacroCall(parent.macroName, newArgs, parent.domain);
    }

    default:
      return parent;
  }
}

/**
 * Searches the AST tree depth-first for the target node and gathers ancestral context.
 */
function findTargetInTree(
  current: ASTNode,
  target: ASTNode,
  nodePath: string[],
  ancestorStack: Array<{ node: ASTNode; relation: string; leftSibling: ASTNode | null }>,
  currentDepth: number
): TraversalContext | null {
  const entries = getChildEntries(current);

  for (let i = 0; i < entries.length; i++) {
    const entry = entries[i];
    const leftSibling = i > 0 ? entries[i - 1].child : null;

    let isTarget = false;
    if (entry.child === target) {
      isTarget = true;
    } else if (nodePath.length > 0) {
      const expectedRelation = nodePath[currentDepth - 1];
      if (
        currentDepth === nodePath.length &&
        (entry.relation === expectedRelation || entry.key === expectedRelation) &&
        areNodesEqual(entry.child, target)
      ) {
        isTarget = true;
      }
    } else if (areNodesEqual(entry.child, target)) {
      isTarget = true;
    }

    if (isTarget) {
      const parentNode = current;
      const grandparent =
        ancestorStack.length > 0
          ? ancestorStack[ancestorStack.length - 1].node
          : null;

      const parentOp = parentNode.kind;
      const grandparentOp = grandparent !== null ? grandparent.kind : 'root';
      const leftSiblingOp = leftSibling !== null ? leftSibling.kind : null;
      const targetDepth = currentDepth + 1;
      const parentRelation = entry.relation;

      return {
        targetDepth,
        parentOp,
        grandparentOp,
        leftSiblingOp,
        parentRelation,
        targetNode: entry.child,
      };
    }

    // Recurse into child
    ancestorStack.push({ node: current, relation: entry.relation, leftSibling });
    const found = findTargetInTree(
      entry.child,
      target,
      nodePath,
      ancestorStack,
      currentDepth + 1
    );
    if (found) {
      return found;
    }
    ancestorStack.pop();
  }

  return null;
}

/**
 * Extracts the ancestral structural context (Parent, Grandparent, LeftSibling, Depth)
 * and expected domain/type for a matched AST node.
 */
export function extractStructuralContext(
  rootAST: ASTNode,
  matchResult: NodeMatchResult
): StructuralContext {
  const target = matchResult.targetNode;
  const isRootTarget =
    rootAST === target ||
    (matchResult.nodePath.length === 0 && areNodesEqual(rootAST, target));

  let traversal: TraversalContext | null = null;
  if (!isRootTarget) {
    traversal = findTargetInTree(rootAST, target, matchResult.nodePath, [], 1);
  }

  let parentOp: string | null;
  let grandparentOp: string | null;
  let leftSiblingOp: string | null;
  let targetDepth: number;
  let parentRelation: string | null;

  if (isRootTarget) {
    targetDepth = 1;
    parentOp = 'root';
    grandparentOp = null;
    leftSiblingOp = null;
    parentRelation = null;
  } else if (traversal) {
    targetDepth = traversal.targetDepth;
    parentOp = traversal.parentOp;
    grandparentOp = traversal.grandparentOp;
    leftSiblingOp = traversal.leftSiblingOp;
    parentRelation = traversal.parentRelation;
  } else if (matchResult.parentNode) {
    // Graceful fallback for synthetic or detached match results
    targetDepth = Math.max(2, matchResult.nodePath.length + 1);
    parentOp = matchResult.parentNode.kind;
    grandparentOp = 'root';
    leftSiblingOp = null;
    parentRelation = matchResult.parentRelation ?? null;
  } else {
    targetDepth = 1;
    parentOp = 'root';
    grandparentOp = null;
    leftSiblingOp = null;
    parentRelation = null;
  }

  // Domain resolution with parent fallback for untyped nodes
  let expectedDomain = resolveDomainType(target);
  if (expectedDomain === 'universal' && matchResult.parentNode) {
    const parentDomain = resolveDomainType(matchResult.parentNode);
    if (parentDomain !== 'universal') {
      expectedDomain = parentDomain;
    }
  }
  if (expectedDomain === 'universal' && rootAST) {
    const rootDomain = resolveDomainType(rootAST);
    if (rootDomain !== 'universal') {
      expectedDomain = rootDomain;
    }
  }

  const expectedType = resolveExpectedType(expectedDomain);

  return Object.freeze({
    parentOp,
    grandparentOp,
    leftSiblingOp,
    targetDepth,
    expectedDomain,
    expectedType,
    parentRelation,
    depth: targetDepth,
  });
}

/**
 * Replaces a subtree inside rootAST at the given path or target reference functionally and immutably.
 */
export function replaceSubtreeAtPath(
  root: ASTNode,
  path: string[],
  target: ASTNode,
  replacement: ASTNode
): ASTNode {
  // If target is root
  if (root === target || (path.length === 0 && areNodesEqual(root, target))) {
    return replacement;
  }

  // If path is provided and non-empty, follow path first
  if (path && path.length > 0) {
    const [currentSegment, ...remainingPath] = path;
    const entries = getChildEntries(root);
    const matchedEntry = entries.find(
      (e) => e.relation === currentSegment || e.key === currentSegment
    );
    if (matchedEntry) {
      const updatedChild =
        remainingPath.length === 0
          ? replacement
          : replaceSubtreeAtPath(matchedEntry.child, remainingPath, target, replacement);
      return rebuildNodeWithUpdatedChild(root, matchedEntry.relation, updatedChild);
    }
  }

  // Fallback: search children by reference or structural equality
  const entries = getChildEntries(root);
  for (const entry of entries) {
    if (entry.child === target || areNodesEqual(entry.child, target)) {
      return rebuildNodeWithUpdatedChild(root, entry.relation, replacement);
    }
    const replaced = replaceSubtreeAtPath(entry.child, [], target, replacement);
    if (replaced !== entry.child) {
      return rebuildNodeWithUpdatedChild(root, entry.relation, replaced);
    }
  }

  return root;
}

/**
 * Punctures a typed hole (`Hole[T]`) at the target AST node location,
 * returning a functional immutable clone of rootAST alongside structural context.
 */
export function punctureASTHole(
  rootAST: ASTNode,
  matchResult: NodeMatchResult,
  holeId?: string
): PunctureResult {
  const resolvedHoleId = holeId || 'hole_1';
  const context = extractStructuralContext(rootAST, matchResult);

  let holeNode: ASTNode;
  switch (context.expectedDomain) {
    case 'sdf':
      holeNode = createSDFHole(resolvedHoleId);
      break;
    case 'audio':
      holeNode = createAudioHole(resolvedHoleId);
      break;
    case 'hardware':
      holeNode = createHardwareHole(resolvedHoleId);
      break;
    case 'universal':
    default:
      holeNode = createHole(resolvedHoleId, context.expectedDomain);
      break;
  }

  const puncturedRootAST =
    context.targetDepth === 1
      ? holeNode
      : replaceSubtreeAtPath(rootAST, matchResult.nodePath, matchResult.targetNode, holeNode);

  return Object.freeze({
    holeId: resolvedHoleId,
    holeNode,
    puncturedRootAST,
    originalTargetNode: matchResult.targetNode,
    context,
    targetSpan: matchResult.span,
  });
}

/**
 * Inversely substitutes a typed HoleNode(holeId) with replacementNode,
 * validating domain type compatibility and preserving immutability.
 */
export function substituteASTHole(
  puncturedAST: ASTNode,
  holeId: string,
  replacementNode: ASTNode
): ASTNode {
  if (!puncturedAST || typeof puncturedAST !== 'object') {
    return puncturedAST;
  }

  // If root node itself is the hole
  if (puncturedAST.kind === 'hole' && (puncturedAST as any).holeId === holeId) {
    const holeDomain = (puncturedAST as any).domain;
    if (holeDomain && holeDomain !== 'universal') {
      const repDomain = resolveDomainType(replacementNode);
      if (repDomain !== holeDomain && repDomain !== 'universal') {
        throw new Error(
          `Type mismatch: cannot substitute ${repDomain} node into ${holeDomain} hole "${holeId}"`
        );
      }
    }
    return replacementNode;
  }

  // Traverse children
  const entries = getChildEntries(puncturedAST);
  let changed = false;
  let updatedParent = puncturedAST;

  for (const entry of entries) {
    if (entry.child.kind === 'hole' && (entry.child as any).holeId === holeId) {
      const holeDomain = (entry.child as any).domain;
      if (holeDomain && holeDomain !== 'universal') {
        const repDomain = resolveDomainType(replacementNode);
        if (repDomain !== holeDomain && repDomain !== 'universal') {
          throw new Error(
            `Type mismatch: cannot substitute ${repDomain} node into ${holeDomain} hole "${holeId}"`
          );
        }
      }
      updatedParent = rebuildNodeWithUpdatedChild(updatedParent, entry.relation, replacementNode);
      changed = true;
    } else {
      const substitutedChild = substituteASTHole(entry.child, holeId, replacementNode);
      if (substitutedChild !== entry.child) {
        updatedParent = rebuildNodeWithUpdatedChild(updatedParent, entry.relation, substitutedChild);
        changed = true;
      }
    }
  }

  return changed ? updatedParent : puncturedAST;
}
