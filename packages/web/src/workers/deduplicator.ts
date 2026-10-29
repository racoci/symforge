/**
 * Semantic Deduplication Engine for AST Variations.
 *
 * Identifies and prunes isomorphic or syntactically duplicate candidate ASTs,
 * computing canonical hashes, normalizing alpha-equivalent free parameter names,
 * and selecting the optimal variation based on derivation probability and syntactic conciseness.
 */

import type { ASTNode, AudioNode, FreeParam, HardwareNode, SDFNode } from '../core/ast/types';
import {
  areNodesEqual,
  createFreeParam,
  createSphere,
  createBox,
  createCylinder,
  createTorus,
  createPlane,
  createTranslate,
  createRotate,
  createScale,
  createUnion,
  createSmoothUnion,
  createIntersection,
  createDifference,
  createSine,
  createSaw,
  createSquare,
  createNoise,
  createADSR,
  createLowPass,
  createGain,
  createMix,
  createFM,
  createReg,
  createImm,
  createAdd,
  createSub,
  createMul,
  createShl,
  createShr,
  createSeq,
  createHole,
  createMacroCall,
  createUniversalSDF,
  createUniversalAudio,
  createUniversalHardware,
} from '../core/ast/models';
import { resolveExpectedType } from '../core/generator/holePuncturing';
import { formatLeanCode } from '../core/parser/leanPrinter';
import type { VariationCandidate } from '../core/pcfg/types';
import type { DeduplicationResult } from './types';

/**
 * Computes a fast 32-bit FNV-1a hash formatted as an 8-character hex string.
 */
export function computeCanonicalHash(str: string): string {
  let hash = 2166136261;
  const maxLen = Math.min(str.length, 100000); // Bounded iteration cap
  for (let i = 0; i < maxLen; i++) {
    hash ^= str.charCodeAt(i);
    hash = Math.imul(hash, 16777619);
  }
  return (hash >>> 0).toString(16).padStart(8, '0');
}

/**
 * Normalizes FreeParam IDs in pre-order traversal sequence (p0, p1, ...) and
 * rounds continuous numerical bounds to 6 decimal places to enable alpha-equivalence matching.
 */
class ASTCanonicalizer {
  private paramCounter = 0;

  private canonicalizeParam(param: FreeParam): FreeParam {
    const id = `p${this.paramCounter++}`;
    return createFreeParam(
      id,
      Number(param.minVal.toFixed(6)),
      Number(param.maxVal.toFixed(6)),
      Number(param.defaultVal.toFixed(6))
    );
  }

  public visit(node: ASTNode): ASTNode {
    switch (node.kind) {
      case 'sphere':
        return createSphere(this.canonicalizeParam(node.radius));
      case 'box':
        return createBox(
          this.canonicalizeParam(node.width),
          this.canonicalizeParam(node.height),
          this.canonicalizeParam(node.depth)
        );
      case 'cylinder':
        return createCylinder(
          this.canonicalizeParam(node.radius),
          this.canonicalizeParam(node.height)
        );
      case 'torus':
        return createTorus(
          this.canonicalizeParam(node.majorRadius),
          this.canonicalizeParam(node.minorRadius)
        );
      case 'plane': {
        const norm = node.normal
          ? {
              x: Number(node.normal.x.toFixed(6)),
              y: Number(node.normal.y.toFixed(6)),
              z: Number(node.normal.z.toFixed(6)),
            }
          : undefined;
        return createPlane(this.canonicalizeParam(node.offset), norm);
      }
      case 'translate':
        return createTranslate(
          this.canonicalizeParam(node.dx),
          this.canonicalizeParam(node.dy),
          this.canonicalizeParam(node.dz),
          this.visit(node.child) as SDFNode
        );
      case 'rotate':
        return createRotate(
          this.canonicalizeParam(node.rx),
          this.canonicalizeParam(node.ry),
          this.canonicalizeParam(node.rz),
          this.visit(node.child) as SDFNode
        );
      case 'scale':
        return createScale(this.canonicalizeParam(node.s), this.visit(node.child) as SDFNode);
      case 'union':
        return createUnion(
          this.canonicalizeParam(node.smoothness),
          this.visit(node.a) as SDFNode,
          this.visit(node.b) as SDFNode
        );
      case 'smooth_union':
        return createSmoothUnion(
          this.canonicalizeParam(node.smoothness),
          this.visit(node.a) as SDFNode,
          this.visit(node.b) as SDFNode
        );
      case 'intersection':
        return createIntersection(
          this.visit(node.a) as SDFNode,
          this.visit(node.b) as SDFNode
        );
      case 'difference':
        return createDifference(
          this.visit(node.a) as SDFNode,
          this.visit(node.b) as SDFNode
        );

      case 'sine':
        return createSine(
          this.canonicalizeParam(node.freq),
          this.canonicalizeParam(node.phase)
        );
      case 'saw':
        return createSaw(this.canonicalizeParam(node.freq));
      case 'square':
        return createSquare(
          this.canonicalizeParam(node.freq),
          this.canonicalizeParam(node.duty)
        );
      case 'noise':
        return createNoise(Math.floor(node.seed));
      case 'adsr':
        return createADSR(
          this.canonicalizeParam(node.attack),
          this.canonicalizeParam(node.decay),
          this.canonicalizeParam(node.sustain),
          this.canonicalizeParam(node.release),
          this.visit(node.gate) as AudioNode
        );
      case 'lowpass':
        return createLowPass(
          this.canonicalizeParam(node.cutoff),
          this.canonicalizeParam(node.resonance),
          this.visit(node.input) as AudioNode
        );
      case 'gain':
        return createGain(
          this.canonicalizeParam(node.gain),
          this.visit(node.input) as AudioNode
        );
      case 'mix':
        return createMix(
          this.canonicalizeParam(node.ratio),
          this.visit(node.input1) as AudioNode,
          this.visit(node.input2) as AudioNode
        );
      case 'fm':
        return createFM(
          this.canonicalizeParam(node.carrier),
          this.canonicalizeParam(node.modFreq),
          this.canonicalizeParam(node.modIndex)
        );

      case 'reg':
        return createReg(node.name);
      case 'imm':
        return createImm(node.val);
      case 'add':
        return createAdd(
          this.visit(node.src1) as HardwareNode,
          this.visit(node.src2) as HardwareNode,
          node.dst
        );
      case 'sub':
        return createSub(
          this.visit(node.src1) as HardwareNode,
          this.visit(node.src2) as HardwareNode,
          node.dst
        );
      case 'mul':
        return createMul(
          this.visit(node.src1) as HardwareNode,
          this.visit(node.src2) as HardwareNode,
          node.dst
        );
      case 'shl':
        return createShl(this.visit(node.src) as HardwareNode, node.shift, node.dst);
      case 'shr':
        return createShr(this.visit(node.src) as HardwareNode, node.shift, node.dst);
      case 'seq':
        return createSeq(
          this.visit(node.first) as HardwareNode,
          this.visit(node.second) as HardwareNode
        );

      case 'hole':
        return createHole(node.holeId, 'domain' in node ? (node as any).domain : undefined);
      case 'macro':
        return createMacroCall(node.macroName, node.arguments, node.domain);
      case 'universal_sdf':
        return createUniversalSDF(this.visit(node.node) as SDFNode);
      case 'universal_audio':
        return createUniversalAudio(this.visit(node.node) as AudioNode);
      case 'universal_hardware':
        return createUniversalHardware(this.visit(node.node) as HardwareNode);
      default:
        return node;
    }
  }
}

/**
 * Creates an alpha-equivalent canonicalized AST where all FreeParam IDs are normalized.
 */
export function canonicalizeAST(node: ASTNode): ASTNode {
  const canonicalizer = new ASTCanonicalizer();
  return canonicalizer.visit(node);
}

/**
 * Evaluates whether two variation candidates are isomorphic duplicates.
 */
interface EnrichedCandidate {
  candidate: VariationCandidate;
  canonicalAst: ASTNode;
  canonicalLean: string;
  canonicalHash: string;
}

export function areCandidatesEquivalent(a: EnrichedCandidate, b: EnrichedCandidate): boolean {
  // 1. Exact Lean code match
  if (a.candidate.leanCode.trim() === b.candidate.leanCode.trim()) {
    return true;
  }

  // 2. Exact AST structural match
  if (areNodesEqual(a.candidate.ast, b.candidate.ast)) {
    return true;
  }

  // 3. Alpha-equivalent canonical match
  if (a.canonicalHash === b.canonicalHash) {
    if (a.canonicalLean === b.canonicalLean) {
      return true;
    }
    if (areNodesEqual(a.canonicalAst, b.canonicalAst)) {
      return true;
    }
  }

  return false;
}

/**
 * Selection heuristic for duplicate resolution:
 * 1. Prioritize candidate with higher derivation probability.
 * 2. If probabilities are equal, prioritize more concise Lean 4 syntax.
 * 3. If syntax lengths are equal, prioritize fewer AST nodes.
 * 4. If node counts are equal, prioritize shallower tree depth.
 */
export function isBetterCandidate(
  candidate: VariationCandidate,
  currentBest: VariationCandidate
): boolean {
  // 1. Highest probability
  const probDiff = candidate.probability - currentBest.probability;
  if (Math.abs(probDiff) > 1e-9) {
    return probDiff > 0;
  }

  // 2. Most concise syntax (shorter Lean code)
  if (candidate.leanCode.length !== currentBest.leanCode.length) {
    return candidate.leanCode.length < currentBest.leanCode.length;
  }

  // 3. Lower node count
  if (candidate.nodeCount !== currentBest.nodeCount) {
    return candidate.nodeCount < currentBest.nodeCount;
  }

  // 4. Shallower tree depth
  if (candidate.depth !== currentBest.depth) {
    return candidate.depth < currentBest.depth;
  }

  return false;
}

/**
 * Filters out isomorphic duplicate candidates, retaining the one with highest probability
 * or most concise syntax while preserving overall curriculum stratum progression.
 *
 * @param candidates List of generated variation candidates.
 * @returns Deduplicated list of candidates and count of pruned duplicate items.
 */
export function deduplicateVariations(candidates: VariationCandidate[]): DeduplicationResult {
  if (!candidates || candidates.length === 0) {
    return { uniqueCandidates: [], deduplicatedCount: 0 };
  }

  const maxCandidates = Math.min(candidates.length, 1000); // Safety iteration cap
  const enrichedList: EnrichedCandidate[] = [];

  for (let i = 0; i < maxCandidates; i++) {
    const c = candidates[i];
    const canonicalAst = canonicalizeAST(c.ast);
    const domain = c.domain ?? 'sdf';
    const canonicalLean = formatLeanCode('', resolveExpectedType(domain), canonicalAst, {
      compact: true,
    }).trim();
    const canonicalHash = computeCanonicalHash(canonicalLean);

    enrichedList.push({
      candidate: c,
      canonicalAst,
      canonicalLean,
      canonicalHash,
    });
  }

  const uniqueEnriched: EnrichedCandidate[] = [];

  for (let i = 0; i < enrichedList.length; i++) {
    const item = enrichedList[i];
    let duplicateIndex = -1;

    for (let j = 0; j < uniqueEnriched.length; j++) {
      if (areCandidatesEquivalent(item, uniqueEnriched[j])) {
        duplicateIndex = j;
        break;
      }
    }

    if (duplicateIndex === -1) {
      uniqueEnriched.push(item);
    } else {
      const existing = uniqueEnriched[duplicateIndex];
      if (isBetterCandidate(item.candidate, existing.candidate)) {
        uniqueEnriched[duplicateIndex] = item;
      }
    }
  }

  const uniqueCandidates = uniqueEnriched.map((e) => e.candidate);
  const deduplicatedCount = candidates.length - uniqueCandidates.length;

  return {
    uniqueCandidates,
    deduplicatedCount,
  };
}
