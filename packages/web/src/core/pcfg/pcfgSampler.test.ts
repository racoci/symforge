/**
 * Automated Unit & Invariant Test Suite for Client-Side Contextual PCFG/PHOG Sampler.
 *
 * Verifies:
 * 1. Type preservation across SDF, Audio, and Hardware domains.
 * 2. Monotonic complexity stratification across L1..L5.
 * 3. Contextual PHOG conditioning distribution shifts.
 * 4. Deterministic PRNG seeding and reproducibility.
 * 5. Derivation safety bounds (hard depth and step caps).
 * 6. Pass-to-Complete Gate: 100 valid variations per domain with 0 type violations.
 */

import { describe, expect, it } from 'vitest';
import {
  isAudioNode,
  isHardwareNode,
  isSDFNode,
  getTreeDepth,
  getNodeCount,
  createBox,
  createSphere,
  createDifference,
  createFreeParam,
} from '../ast';
import type { StructuralContext, PunctureResult } from '../generator/types';
import {
  createPRNG,
  getGrammarRules,
  getConditionedRules,
  sampleLevel,
  sampleScopedVariations,
  sampleFreeParam,
} from './pcfgSampler';
import type { CurriculumLevel } from './types';

describe('Client-Side Contextual PCFG/PHOG Sampler', () => {
  const dummySpan = {
    startLine: 1,
    startCol: 1,
    endLine: 1,
    endCol: 20,
    startOffset: 0,
    endOffset: 20,
  };

  // ==========================================
  // 1. Type Preservation Tests
  // ==========================================
  describe('Type Preservation', () => {
    it('verifies 100% of generated SDF nodes satisfy isSDFNode across all levels', () => {
      const context: StructuralContext = {
        parentOp: 'root',
        grandparentOp: null,
        leftSiblingOp: null,
        targetDepth: 1,
        expectedDomain: 'sdf',
        expectedType: 'SDFNode',
        parentRelation: null,
      };

      const levels: CurriculumLevel[] = ['L1', 'L2', 'L3', 'L4', 'L5'];
      for (const level of levels) {
        for (let i = 0; i < 20; i += 1) {
          const rng = createPRNG(1000 + i);
          const node = sampleLevel(level, context, rng);
          expect(isSDFNode(node)).toBe(true);
        }
      }
    });

    it('verifies 100% of generated Audio nodes satisfy isAudioNode across all levels', () => {
      const context: StructuralContext = {
        parentOp: 'root',
        grandparentOp: null,
        leftSiblingOp: null,
        targetDepth: 1,
        expectedDomain: 'audio',
        expectedType: 'AudioNode',
        parentRelation: null,
      };

      const levels: CurriculumLevel[] = ['L1', 'L2', 'L3', 'L4', 'L5'];
      for (const level of levels) {
        for (let i = 0; i < 20; i += 1) {
          const rng = createPRNG(2000 + i);
          const node = sampleLevel(level, context, rng);
          expect(isAudioNode(node)).toBe(true);
        }
      }
    });

    it('verifies 100% of generated Hardware nodes satisfy isHardwareNode across all levels', () => {
      const context: StructuralContext = {
        parentOp: 'root',
        grandparentOp: null,
        leftSiblingOp: null,
        targetDepth: 1,
        expectedDomain: 'hardware',
        expectedType: 'HardwareNode',
        parentRelation: null,
      };

      const levels: CurriculumLevel[] = ['L1', 'L2', 'L3', 'L4', 'L5'];
      for (const level of levels) {
        for (let i = 0; i < 20; i += 1) {
          const rng = createPRNG(3000 + i);
          const node = sampleLevel(level, context, rng);
          expect(isHardwareNode(node)).toBe(true);
        }
      }
    });
  });

  // ==========================================
  // 2. Complexity Stratification Tests
  // ==========================================
  describe('Complexity Stratification (L1..L5)', () => {
    it('verifies individual structural invariants for each level', () => {
      const context: StructuralContext = {
        parentOp: 'root',
        grandparentOp: null,
        leftSiblingOp: null,
        targetDepth: 1,
        expectedDomain: 'sdf',
        expectedType: 'SDFNode',
        parentRelation: null,
      };

      for (let i = 0; i < 15; i += 1) {
        const rng = createPRNG(4000 + i);

        // L1: Atomic (depth 1, nodeCount 1)
        const l1 = sampleLevel('L1', context, rng);
        expect(getTreeDepth(l1)).toBe(1);
        expect(getNodeCount(l1)).toBe(1);

        // L2: Unary (depth 2, nodeCount 2)
        const l2 = sampleLevel('L2', context, rng);
        expect(getTreeDepth(l2)).toBe(2);
        expect(getNodeCount(l2)).toBe(2);

        // L3: Binary Combinator (depth 2..3, nodeCount 3..4)
        const l3 = sampleLevel('L3', context, rng);
        expect(getTreeDepth(l3)).toBeGreaterThanOrEqual(2);
        expect(getTreeDepth(l3)).toBeLessThanOrEqual(3);
        expect(getNodeCount(l3)).toBeGreaterThanOrEqual(3);
        expect(getNodeCount(l3)).toBeLessThanOrEqual(4);

        // L4: Hierarchical (depth 3..4, nodeCount 4..7)
        const l4 = sampleLevel('L4', context, rng);
        expect(getTreeDepth(l4)).toBeGreaterThanOrEqual(3);
        expect(getTreeDepth(l4)).toBeLessThanOrEqual(4);
        expect(getNodeCount(l4)).toBeGreaterThanOrEqual(4);

        // L5: Compound Structural Abstraction (depth >= 4, nodeCount >= 7)
        const l5 = sampleLevel('L5', context, rng);
        expect(getTreeDepth(l5)).toBeGreaterThanOrEqual(4);
        expect(getTreeDepth(l5)).toBeLessThanOrEqual(6);
        expect(getNodeCount(l5)).toBeGreaterThanOrEqual(7);
      }
    });

    it('verifies monotonic increase in mean depth and node count across L1..L5', () => {
      const context: StructuralContext = {
        parentOp: 'root',
        grandparentOp: null,
        leftSiblingOp: null,
        targetDepth: 1,
        expectedDomain: 'sdf',
        expectedType: 'SDFNode',
        parentRelation: null,
      };

      const sampleBatch = (level: CurriculumLevel, count = 25) => {
        let totalDepth = 0;
        let totalNodes = 0;
        for (let i = 0; i < count; i += 1) {
          const rng = createPRNG(5000 + i * 17);
          const node = sampleLevel(level, context, rng);
          totalDepth += getTreeDepth(node);
          totalNodes += getNodeCount(node);
        }
        return {
          meanDepth: totalDepth / count,
          meanNodes: totalNodes / count,
        };
      };

      const m1 = sampleBatch('L1');
      const m2 = sampleBatch('L2');
      const m3 = sampleBatch('L3');
      const m4 = sampleBatch('L4');
      const m5 = sampleBatch('L5');

      // Monotonic mean depth increase
      expect(m1.meanDepth).toBeLessThan(m2.meanDepth);
      expect(m2.meanDepth).toBeLessThan(m3.meanDepth);
      expect(m3.meanDepth).toBeLessThan(m4.meanDepth);
      expect(m4.meanDepth).toBeLessThan(m5.meanDepth);

      // Monotonic mean node count increase
      expect(m1.meanNodes).toBeLessThan(m2.meanNodes);
      expect(m2.meanNodes).toBeLessThan(m3.meanNodes);
      expect(m3.meanNodes).toBeLessThan(m4.meanNodes);
      expect(m4.meanNodes).toBeLessThan(m5.meanNodes);
    });
  });

  // ==========================================
  // 3. Contextual PHOG Conditioning Tests
  // ==========================================
  describe('Contextual PHOG Conditioning', () => {
    it('returns base grammar production rules for each domain', () => {
      const sdfRules = getGrammarRules('sdf');
      expect(sdfRules.length).toBeGreaterThan(0);
      expect(sdfRules.some((r) => r.constructorName === 'sphere')).toBe(true);

      const audioRules = getGrammarRules('audio');
      expect(audioRules.length).toBeGreaterThan(0);
      expect(audioRules.some((r) => r.constructorName === 'sine')).toBe(true);

      const hwRules = getGrammarRules('hardware');
      expect(hwRules.length).toBeGreaterThan(0);
      expect(hwRules.some((r) => r.constructorName === 'reg')).toBe(true);
    });

    it('shifts rule probabilities when conditioned on difference (cutting tool) vs union', () => {
      const diffContext: StructuralContext = {
        parentOp: 'difference',
        grandparentOp: 'root',
        leftSiblingOp: 'box',
        targetDepth: 2,
        expectedDomain: 'sdf',
        expectedType: 'SDFNode',
        parentRelation: 'b',
      };

      const unionContext: StructuralContext = {
        parentOp: 'union',
        grandparentOp: 'root',
        leftSiblingOp: 'box',
        targetDepth: 2,
        expectedDomain: 'sdf',
        expectedType: 'SDFNode',
        parentRelation: 'b',
      };

      const diffRules = getConditionedRules('sdf', diffContext);
      const unionRules = getConditionedRules('sdf', unionContext);

      const diffCylinder = diffRules.find((r) => r.constructorName === 'cylinder')!;
      const unionCylinder = unionRules.find((r) => r.constructorName === 'cylinder')!;

      // Cylinder is boosted 3x under difference right child (cutting tool)
      expect(diffCylinder.probability).toBeGreaterThan(unionCylinder.probability);
    });

    it('empirically produces higher cylinder frequency under difference(b) context', () => {
      const diffContext: StructuralContext = {
        parentOp: 'difference',
        grandparentOp: 'root',
        leftSiblingOp: 'box',
        targetDepth: 2,
        expectedDomain: 'sdf',
        expectedType: 'SDFNode',
        parentRelation: 'b',
      };

      const unionContext: StructuralContext = {
        parentOp: 'union',
        grandparentOp: 'root',
        leftSiblingOp: 'box',
        targetDepth: 2,
        expectedDomain: 'sdf',
        expectedType: 'SDFNode',
        parentRelation: 'b',
      };

      const sampleCount = 200;
      let diffCylinders = 0;
      let unionCylinders = 0;

      for (let i = 0; i < sampleCount; i += 1) {
        const rngDiff = createPRNG(7000 + i);
        const nodeDiff = sampleLevel('L1', diffContext, rngDiff);
        if (nodeDiff.kind === 'cylinder') diffCylinders += 1;

        const rngUnion = createPRNG(7000 + i);
        const nodeUnion = sampleLevel('L1', unionContext, rngUnion);
        if (nodeUnion.kind === 'cylinder') unionCylinders += 1;
      }

      // Difference(b) conditioning boosts cylinder significantly
      expect(diffCylinders).toBeGreaterThan(unionCylinders);
    });

    it('shifts audio probabilities when conditioned on lowpass filter', () => {
      const lowpassContext: StructuralContext = {
        parentOp: 'lowpass',
        grandparentOp: 'root',
        leftSiblingOp: null,
        targetDepth: 2,
        expectedDomain: 'audio',
        expectedType: 'AudioNode',
        parentRelation: 'input_node',
      };

      const baselineContext: StructuralContext = {
        parentOp: 'root',
        grandparentOp: null,
        leftSiblingOp: null,
        targetDepth: 1,
        expectedDomain: 'audio',
        expectedType: 'AudioNode',
        parentRelation: null,
      };

      const lpRules = getConditionedRules('audio', lowpassContext);
      const baseRules = getConditionedRules('audio', baselineContext);

      const lpSaw = lpRules.find((r) => r.constructorName === 'saw')!;
      const baseSaw = baseRules.find((r) => r.constructorName === 'saw')!;

      // Saw wave (rich harmonics) is boosted under lowpass filter input
      expect(lpSaw.probability).toBeGreaterThan(baseSaw.probability);
    });
  });

  // ==========================================
  // 4. Deterministic PRNG Tests
  // ==========================================
  describe('Deterministic PRNG', () => {
    it('generates identical AST trees with the same seed', () => {
      const context: StructuralContext = {
        parentOp: 'root',
        grandparentOp: null,
        leftSiblingOp: null,
        targetDepth: 1,
        expectedDomain: 'sdf',
        expectedType: 'SDFNode',
        parentRelation: null,
      };

      const seed = 98765;
      const rng1 = createPRNG(seed);
      const rng2 = createPRNG(seed);

      const ast1 = sampleLevel('L4', context, rng1);
      const ast2 = sampleLevel('L4', context, rng2);

      expect(JSON.stringify(ast1)).toEqual(JSON.stringify(ast2));
      expect(getTreeDepth(ast1)).toEqual(getTreeDepth(ast2));
      expect(getNodeCount(ast1)).toEqual(getNodeCount(ast2));
    });

    it('generates diverse AST trees with different seeds', () => {
      const context: StructuralContext = {
        parentOp: 'root',
        grandparentOp: null,
        leftSiblingOp: null,
        targetDepth: 1,
        expectedDomain: 'sdf',
        expectedType: 'SDFNode',
        parentRelation: null,
      };

      const rng1 = createPRNG(11111);
      const rng2 = createPRNG(99999);

      const ast1 = sampleLevel('L3', context, rng1);
      const ast2 = sampleLevel('L3', context, rng2);

      expect(JSON.stringify(ast1)).not.toEqual(JSON.stringify(ast2));
    });
  });

  // ==========================================
  // 5. Derivation Safety & Termination Tests
  // ==========================================
  describe('Derivation Safety Bounds', () => {
    it('guarantees termination without stack overflow even under aggressive sampling', () => {
      const context: StructuralContext = {
        parentOp: 'difference',
        grandparentOp: 'translate',
        leftSiblingOp: 'box',
        targetDepth: 5,
        expectedDomain: 'sdf',
        expectedType: 'SDFNode',
        parentRelation: 'b',
      };

      for (let i = 0; i < 50; i += 1) {
        const rng = createPRNG(12345 + i);
        const node = sampleLevel('L5', context, rng);
        expect(getTreeDepth(node)).toBeLessThanOrEqual(6);
        expect(getNodeCount(node)).toBeGreaterThanOrEqual(1);
      }
    });

    it('sampleFreeParam creates valid bounded parameters', () => {
      const rng = createPRNG(42);
      const p = sampleFreeParam('rad', 0.5, 4.0, rng, 2.0);
      expect(p.id.startsWith('rad_')).toBe(true);
      expect(p.minVal).toBe(0.5);
      expect(p.maxVal).toBe(4.0);
      expect(p.defaultVal).toBe(2.0);
    });

    it('handles extreme seed values (0, negative, large uint32)', () => {
      const context: StructuralContext = {
        parentOp: 'root',
        grandparentOp: null,
        leftSiblingOp: null,
        targetDepth: 1,
        expectedDomain: 'sdf',
        expectedType: 'SDFNode',
        parentRelation: null,
      };

      for (const seed of [0, -1, -99999, 4294967295, 2147483647]) {
        const rng = createPRNG(seed);
        const node = sampleLevel('L2', context, rng);
        expect(isSDFNode(node)).toBe(true);
        expect(getTreeDepth(node)).toBe(2);
      }
    });

    it('normalizes conditioned rule groups so sum of probabilities is 1.0', () => {
      const context: StructuralContext = {
        parentOp: 'difference',
        grandparentOp: 'root',
        leftSiblingOp: 'box',
        targetDepth: 2,
        expectedDomain: 'sdf',
        expectedType: 'SDFNode',
        parentRelation: 'b',
      };

      const rules = getConditionedRules('sdf', context);
      const terminals = rules.filter((r) => r.isTerminal);
      const termSum = terminals.reduce((acc, r) => acc + r.probability, 0);
      expect(termSum).toBeCloseTo(1.0, 1);

      const unaries = rules.filter((r) => r.nonTerminal === 'SDF_Unary');
      const unarySum = unaries.reduce((acc, r) => acc + r.probability, 0);
      expect(unarySum).toBeCloseTo(1.0, 1);

      const binaries = rules.filter((r) => r.nonTerminal === 'SDF_Binary');
      const binSum = binaries.reduce((acc, r) => acc + r.probability, 0);
      expect(binSum).toBeCloseTo(1.0, 1);
    });

    it('supports Universal domain with typed container wrapping', () => {
      const context: StructuralContext = {
        parentOp: 'root',
        grandparentOp: null,
        leftSiblingOp: null,
        targetDepth: 1,
        expectedDomain: 'universal',
        expectedType: 'UniversalNode',
        parentRelation: null,
      };

      const rng = createPRNG(99);
      const node = sampleLevel('L1', context, rng);
      expect(node.kind).toBe('universal_sdf');
    });
  });

  // ==========================================
  // 6. Scoped Variations Entry Point
  // ==========================================
  describe('sampleScopedVariations API', () => {
    it('derives candidates with unique IDs, labels, Lean code, and valid probabilities', () => {
      const root = createDifference(
        createBox(
          createFreeParam('b_w', 1, 10, 2),
          createFreeParam('b_h', 1, 10, 2),
          createFreeParam('b_d', 1, 10, 2)
        ),
        createSphere(createFreeParam('s_r', 0.5, 5, 1))
      );

      const fakePuncture: PunctureResult = {
        holeId: 'hole_cutout',
        holeNode: root.b,
        puncturedRootAST: root,
        originalTargetNode: root.b,
        context: {
          parentOp: 'difference',
          grandparentOp: 'root',
          leftSiblingOp: 'box',
          targetDepth: 2,
          expectedDomain: 'sdf',
          expectedType: 'SDFNode',
          parentRelation: 'b',
        },
        targetSpan: dummySpan,
      };

      const candidates = sampleScopedVariations(fakePuncture, {
        countPerLevel: 2,
        seed: 42,
      });

      // 2 candidates per level * 5 levels = 10 candidates
      expect(candidates.length).toBe(10);

      for (const cand of candidates) {
        expect(cand.id).toContain('hole_cutout');
        expect(cand.domain).toBe('sdf');
        expect(isSDFNode(cand.ast)).toBe(true);
        expect(cand.depth).toBeGreaterThanOrEqual(1);
        expect(cand.depth).toBeLessThanOrEqual(6);
        expect(cand.nodeCount).toBeGreaterThanOrEqual(1);
        expect(cand.probability).toBeGreaterThan(0);
        expect(cand.probability).toBeLessThanOrEqual(1);
        expect(cand.label).toMatch(/^\[L[1-5]\]/);
        expect(cand.leanCode.length).toBeGreaterThan(0);
        expect(cand.leanCode).toContain('SDFNode');
      }
    });

    it('respects custom levels and countPerLevel options', () => {
      const root = createSphere(createFreeParam('r', 0.5, 5, 1));
      const fakePuncture: PunctureResult = {
        holeId: 'hole_custom',
        holeNode: root,
        puncturedRootAST: root,
        originalTargetNode: root,
        context: {
          parentOp: 'root',
          grandparentOp: null,
          leftSiblingOp: null,
          targetDepth: 1,
          expectedDomain: 'audio',
          expectedType: 'AudioNode',
          parentRelation: null,
        },
        targetSpan: dummySpan,
      };

      const candidates = sampleScopedVariations(fakePuncture, {
        levels: ['L1', 'L3'],
        countPerLevel: 3,
        seed: 777,
      });

      // 2 levels * 3 per level = 6 candidates
      expect(candidates.length).toBe(6);
      expect(candidates.every((c) => c.complexityLevel === 'L1' || c.complexityLevel === 'L3')).toBe(true);
      expect(candidates.every((c) => isAudioNode(c.ast))).toBe(true);
    });
  });

  // ==========================================
  // 7. Pass-to-Complete Gate
  // ==========================================
  describe('Pass-to-Complete Gate: 100 Variations per Domain', () => {
    it('generates 100 valid variations (20 per level L1..L5) across SDF domain with 0 violations', () => {
      const context: StructuralContext = {
        parentOp: 'difference',
        grandparentOp: 'root',
        leftSiblingOp: 'box',
        targetDepth: 2,
        expectedDomain: 'sdf',
        expectedType: 'SDFNode',
        parentRelation: 'b',
      };

      const fakePuncture: PunctureResult = {
        holeId: 'gate_sdf',
        holeNode: createSphere(createFreeParam('r', 0.5, 5, 1)),
        puncturedRootAST: createSphere(createFreeParam('r', 0.5, 5, 1)),
        originalTargetNode: createSphere(createFreeParam('r', 0.5, 5, 1)),
        context,
        targetSpan: dummySpan,
      };

      const variations = sampleScopedVariations(fakePuncture, {
        countPerLevel: 20,
        seed: 123456,
      });

      expect(variations.length).toBe(100);

      for (const v of variations) {
        expect(isSDFNode(v.ast)).toBe(true);
        expect(v.depth).toBeGreaterThanOrEqual(1);
        expect(v.depth).toBeLessThanOrEqual(6);
        expect(v.nodeCount).toBeGreaterThanOrEqual(1);
        expect(v.probability).toBeGreaterThan(0);
        expect(v.probability).toBeLessThanOrEqual(1);
        expect(v.leanCode.length).toBeGreaterThan(0);
        expect(v.leanCode).toContain('SDFNode');
      }
    });

    it('generates 100 valid variations (20 per level L1..L5) across Audio domain with 0 violations', () => {
      const context: StructuralContext = {
        parentOp: 'lowpass',
        grandparentOp: 'root',
        leftSiblingOp: null,
        targetDepth: 2,
        expectedDomain: 'audio',
        expectedType: 'AudioNode',
        parentRelation: 'input_node',
      };

      const fakePuncture: PunctureResult = {
        holeId: 'gate_audio',
        holeNode: createSphere(createFreeParam('r', 0.5, 5, 1)),
        puncturedRootAST: createSphere(createFreeParam('r', 0.5, 5, 1)),
        originalTargetNode: createSphere(createFreeParam('r', 0.5, 5, 1)),
        context,
        targetSpan: dummySpan,
      };

      const variations = sampleScopedVariations(fakePuncture, {
        countPerLevel: 20,
        seed: 654321,
      });

      expect(variations.length).toBe(100);

      for (const v of variations) {
        expect(isAudioNode(v.ast)).toBe(true);
        expect(v.depth).toBeGreaterThanOrEqual(1);
        expect(v.depth).toBeLessThanOrEqual(6);
        expect(v.nodeCount).toBeGreaterThanOrEqual(1);
        expect(v.probability).toBeGreaterThan(0);
        expect(v.probability).toBeLessThanOrEqual(1);
        expect(v.leanCode.length).toBeGreaterThan(0);
        expect(v.leanCode).toContain('AudioNode');
      }
    });

    it('generates 100 valid variations (20 per level L1..L5) across Hardware domain with 0 violations', () => {
      const context: StructuralContext = {
        parentOp: 'seq',
        grandparentOp: 'root',
        leftSiblingOp: 'add',
        targetDepth: 2,
        expectedDomain: 'hardware',
        expectedType: 'HardwareNode',
        parentRelation: 'second',
      };

      const fakePuncture: PunctureResult = {
        holeId: 'gate_hw',
        holeNode: createSphere(createFreeParam('r', 0.5, 5, 1)),
        puncturedRootAST: createSphere(createFreeParam('r', 0.5, 5, 1)),
        originalTargetNode: createSphere(createFreeParam('r', 0.5, 5, 1)),
        context,
        targetSpan: dummySpan,
      };

      const variations = sampleScopedVariations(fakePuncture, {
        countPerLevel: 20,
        seed: 789101,
      });

      expect(variations.length).toBe(100);

      for (const v of variations) {
        expect(isHardwareNode(v.ast)).toBe(true);
        expect(v.depth).toBeGreaterThanOrEqual(1);
        expect(v.depth).toBeLessThanOrEqual(6);
        expect(v.nodeCount).toBeGreaterThanOrEqual(1);
        expect(v.probability).toBeGreaterThan(0);
        expect(v.probability).toBeLessThanOrEqual(1);
        expect(v.leanCode.length).toBeGreaterThan(0);
        expect(v.leanCode).toContain('HardwareNode');
      }
    });
  });
});
