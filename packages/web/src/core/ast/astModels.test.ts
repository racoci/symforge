/**
 * Comprehensive Unit and Property-Based Tests for SymForge Client-Side AST Engine.
 *
 * Validates:
 * 1. Structural invariants (node count, tree depth, unique parameter collection).
 * 2. Lean 4 JSON round-trip serialization identity across all node types.
 * 3. Bidirectional compatibility with Lean 4 camelCase and Python snake_case schemas.
 * 4. Immutable parameter cloning via `cloneWithParams`.
 * 5. Pass-to-Complete Gate: 500 random AST trees across L1-L4 complexity (Audio and SDF) with 100% roundtrip pass.
 */

import { describe, it, expect } from 'vitest';
import {
  createFreeParam,
  createSine,
  createSaw,
  createSquare,
  createNoise,
  createADSR,
  createLowPass,
  createGain,
  createMix,
  createFM,
  createAudioHole,
  createAudioMacro,
  createSphere,
  createBox,
  createCylinder,
  createTorus,
  createTranslate,
  createRotate,
  createScale,
  createUnion,
  createSmoothUnion,
  createIntersection,
  createDifference,
  createSDFHole,
  createSDFMacro,
  createReg,
  createImm,
  createAdd,
  createSub,
  createMul,
  createShl,
  createShr,
  createSeq,
  createHardwareHole,
  createUniversalAudio,
  createUniversalSDF,
  createUniversalHardware,
  createUniversalMacro,
  createHole,
  createMacroCall,
  collectFreeParams,
  getTreeDepth,
  getNodeCount,
  cloneWithParams,
  areNodesEqual,
  serializeToLeanJson,
  deserializeFromLeanJson,
  serializeFreeParam,
  deserializeFreeParam,
  type AudioNode,
  type SDFNode,
  type HardwareNode,
  type ASTNode,
} from './index';

describe('SymForge AST Core Models & Utilities', () => {
  describe('FreeParam primitives', () => {
    it('creates immutable free parameters with correct bounds', () => {
      const p = createFreeParam('cutoff', 20, 20000, 1000);
      expect(p.id).toBe('cutoff');
      expect(p.minVal).toBe(20);
      expect(p.maxVal).toBe(20000);
      expect(p.defaultVal).toBe(1000);
      expect(Object.isFrozen(p)).toBe(true);
    });

    it('serializes and deserializes FreeParam in Lean 4 camelCase format', () => {
      const p = createFreeParam('vol', 0, 1, 0.75);
      const json = serializeFreeParam(p);
      expect(json).toEqual({ id: 'vol', minVal: 0, maxVal: 1, defaultVal: 0.75 });
      const decoded = deserializeFreeParam(json);
      expect(decoded).toEqual(p);
    });

    it('deserializes FreeParam in Python snake_case format', () => {
      const pyJson = { id: 'freq_0', min_val: 100, max_val: 5000, default_val: 440 };
      const decoded = deserializeFreeParam(pyJson);
      expect(decoded.id).toBe('freq_0');
      expect(decoded.minVal).toBe(100);
      expect(decoded.maxVal).toBe(5000);
      expect(decoded.defaultVal).toBe(440);
    });
  });

  describe('Structural Invariants (Size, Depth, Param Collection)', () => {
    it('calculates size and depth accurately for procedural Audio trees', () => {
      const f0 = createFreeParam('f0', 20, 2000, 220);
      const saw = createSaw(f0);
      expect(getNodeCount(saw)).toBe(1);
      expect(getTreeDepth(saw)).toBe(1);
      expect(collectFreeParams(saw)).toHaveLength(1);

      const cutoff = createFreeParam('fc', 100, 10000, 1200);
      const res = createFreeParam('q', 0.1, 10, 0.707);
      const lp = createLowPass(cutoff, res, saw);
      expect(getNodeCount(lp)).toBe(2);
      expect(getTreeDepth(lp)).toBe(2);
      expect(collectFreeParams(lp)).toHaveLength(3);

      const ratio = createFreeParam('mixRatio', 0, 1, 0.5);
      const sine = createSine(createFreeParam('f_sine', 20, 2000, 440));
      const mix = createMix(ratio, lp, sine);
      expect(getNodeCount(mix)).toBe(4); // mix + lp + saw + sine
      expect(getTreeDepth(mix)).toBe(3); // mix -> lp -> saw
      expect(collectFreeParams(mix)).toHaveLength(6);
    });

    it('calculates size and depth accurately for 3D SDF CSG trees', () => {
      const r = createFreeParam('r', 0.1, 5, 1.0);
      const sphere = createSphere(r);
      expect(getNodeCount(sphere)).toBe(1);
      expect(getTreeDepth(sphere)).toBe(1);

      const w = createFreeParam('w', 0.1, 5, 0.8);
      const h = createFreeParam('h', 0.1, 5, 0.8);
      const d = createFreeParam('d', 0.1, 5, 0.8);
      const box = createBox(w, h, d);

      const sm = createFreeParam('smooth', 0, 1, 0.2);
      const union = createUnion(sm, sphere, box);
      expect(getNodeCount(union)).toBe(3);
      expect(getTreeDepth(union)).toBe(2);

      const dx = createFreeParam('dx', -5, 5, 1.5);
      const dy = createFreeParam('dy', -5, 5, 0);
      const dz = createFreeParam('dz', -5, 5, 0);
      const translated = createTranslate(dx, dy, dz, union);
      expect(getNodeCount(translated)).toBe(4);
      expect(getTreeDepth(translated)).toBe(3);
      expect(collectFreeParams(translated)).toHaveLength(8);
    });

    it('calculates size and depth for Hardware instruction sequences', () => {
      const r1 = createReg('r1');
      const imm = createImm(42);
      const add = createAdd(r1, imm, 'r0');
      const sub = createSub(r1, imm, 'r2');
      const seq = createSeq(add, sub);

      expect(getNodeCount(r1)).toBe(1);
      expect(getNodeCount(add)).toBe(3); // add + r1 + imm
      expect(getNodeCount(seq)).toBe(7); // seq + add(3) + sub(3)
      expect(getTreeDepth(seq)).toBe(3); // seq -> add -> r1
      expect(collectFreeParams(seq)).toHaveLength(0);
    });

    it('deduplicates shared FreeParam references during collection', () => {
      const sharedFreq = createFreeParam('shared_freq', 100, 1000, 440);
      const osc1 = createSaw(sharedFreq);
      const osc2 = createSine(sharedFreq);
      const mix = createMix(createFreeParam('ratio', 0, 1, 0.5), osc1, osc2);

      const params = collectFreeParams(mix);
      const ids = params.map((p) => p.id);
      expect(ids).toContain('shared_freq');
      expect(ids).toContain('ratio');
      // shared_freq should appear only once despite being used by both osc1 and osc2
      expect(ids.filter((id) => id === 'shared_freq')).toHaveLength(1);
    });
  });

  describe('cloneWithParams immutability', () => {
    it('clones an AST with updated parameter values without mutating original', () => {
      const r = createFreeParam('radius', 0.1, 5.0, 1.0);
      const sphere = createSphere(r);
      const cloned = cloneWithParams(sphere, { radius: 2.5 });

      expect(cloned.radius.defaultVal).toBe(2.5);
      expect(cloned.radius.id).toBe('radius');
      // Original remains untouched
      expect(sphere.radius.defaultVal).toBe(1.0);
    });

    it('clones complex nested compositions updating only targeted parameters', () => {
      const fc = createFreeParam('cutoff', 100, 10000, 1000);
      const q = createFreeParam('res', 0.1, 10, 0.707);
      const f0 = createFreeParam('f0', 20, 2000, 440);
      const tree = createLowPass(fc, q, createSaw(f0));

      const cloned = cloneWithParams(tree, { cutoff: 3500, non_existent: 999 });
      expect(cloned.cutoff.defaultVal).toBe(3500);
      expect(cloned.resonance.defaultVal).toBe(0.707);
      expect((cloned.input as any).freq.defaultVal).toBe(440);

      // Verify original remains pristine
      expect(tree.cutoff.defaultVal).toBe(1000);
    });
  });

  describe('areNodesEqual Deep Structural Equality', () => {
    it('recognizes identical AST structures as equal', () => {
      const a = createUnion(
        createFreeParam('sm', 0, 1, 0.1),
        createSphere(createFreeParam('r1', 0, 5, 1)),
        createBox(createFreeParam('w', 0, 5, 1), createFreeParam('h', 0, 5, 1), createFreeParam('d', 0, 5, 1))
      );
      const b = createUnion(
        createFreeParam('sm', 0, 1, 0.1),
        createSphere(createFreeParam('r1', 0, 5, 1)),
        createBox(createFreeParam('w', 0, 5, 1), createFreeParam('h', 0, 5, 1), createFreeParam('d', 0, 5, 1))
      );
      expect(areNodesEqual(a, b)).toBe(true);
    });

    it('detects parameter value differences', () => {
      const a = createSphere(createFreeParam('r', 0, 5, 1.0));
      const b = createSphere(createFreeParam('r', 0, 5, 1.5));
      expect(areNodesEqual(a, b)).toBe(false);
    });

    it('detects structural node kind differences', () => {
      const a = createSphere(createFreeParam('r', 0, 5, 1.0));
      const b = createSaw(createFreeParam('r', 0, 5, 1.0));
      expect(areNodesEqual(a, b)).toBe(false);
    });
  });

  describe('Lean 4 JSON Round-Trip Serialization Identity', () => {
    it('round-trips all Audio primitive nodes', () => {
      const nodes: AudioNode[] = [
        createSine(createFreeParam('freq', 20, 2000, 440), createFreeParam('phase', 0, 6.28, 0)),
        createSaw(createFreeParam('freq', 20, 2000, 220)),
        createSquare(createFreeParam('freq', 20, 2000, 110), createFreeParam('duty', 0.1, 0.9, 0.5)),
        createNoise(1337),
        createADSR(
          createFreeParam('a', 0.01, 2, 0.1),
          createFreeParam('d', 0.01, 2, 0.2),
          createFreeParam('s', 0, 1, 0.7),
          createFreeParam('r', 0.01, 5, 0.5),
          createSaw(createFreeParam('f', 20, 2000, 440))
        ),
        createLowPass(
          createFreeParam('fc', 100, 10000, 1200),
          createFreeParam('q', 0.1, 10, 0.707),
          createSaw(createFreeParam('f', 20, 2000, 440))
        ),
        createGain(createFreeParam('g', 0, 2, 0.8), createSaw(createFreeParam('f', 20, 2000, 440))),
        createMix(
          createFreeParam('ratio', 0, 1, 0.5),
          createSaw(createFreeParam('f1', 20, 2000, 220)),
          createSine(createFreeParam('f2', 20, 2000, 440))
        ),
        createFM(
          createFreeParam('c', 20, 2000, 440),
          createFreeParam('mf', 20, 2000, 220),
          createFreeParam('mi', 0, 10, 2.5)
        ),
        createAudioHole('hole_audio_0'),
        createAudioMacro('superSaw', { detune: createFreeParam('det', 0, 1, 0.2) }),
      ];

      for (const node of nodes) {
        const serialized = serializeToLeanJson(node);
        const deserialized = deserializeFromLeanJson(serialized, 'audio');
        expect(areNodesEqual(deserialized, node)).toBe(true);
      }
    });

    it('round-trips all SDF primitive nodes', () => {
      const nodes: SDFNode[] = [
        createSphere(createFreeParam('r', 0.1, 10, 1.5)),
        createBox(createFreeParam('w', 0.1, 10, 1), createFreeParam('h', 0.1, 10, 2), createFreeParam('d', 0.1, 10, 3)),
        createCylinder(createFreeParam('r', 0.1, 5, 0.5), createFreeParam('h', 0.1, 5, 2.0)),
        createTorus(createFreeParam('R', 0.5, 5, 2.0), createFreeParam('r', 0.1, 2, 0.5)),
        createTranslate(createFreeParam('dx', -5, 5, 1), createFreeParam('dy', -5, 5, 0), createFreeParam('dz', -5, 5, 0), createSphere(createFreeParam('r', 0.1, 5, 1))),
        createRotate(createFreeParam('rx', 0, 6.28, 0.5), createFreeParam('ry', 0, 6.28, 0), createFreeParam('rz', 0, 6.28, 0), createSphere(createFreeParam('r', 0.1, 5, 1))),
        createScale(createFreeParam('s', 0.1, 5, 1.5), createSphere(createFreeParam('r', 0.1, 5, 1))),
        createUnion(createFreeParam('sm', 0, 1, 0.2), createSphere(createFreeParam('r', 0.1, 5, 1)), createBox(createFreeParam('w', 0.1, 5, 1), createFreeParam('h', 0.1, 5, 1), createFreeParam('d', 0.1, 5, 1))),
        createSmoothUnion(createFreeParam('sm', 0, 1, 0.2), createSphere(createFreeParam('r', 0.1, 5, 1)), createBox(createFreeParam('w', 0.1, 5, 1), createFreeParam('h', 0.1, 5, 1), createFreeParam('d', 0.1, 5, 1))),
        createIntersection(createSphere(createFreeParam('r', 0.1, 5, 1)), createBox(createFreeParam('w', 0.1, 5, 1), createFreeParam('h', 0.1, 5, 1), createFreeParam('d', 0.1, 5, 1))),
        createDifference(createSphere(createFreeParam('r', 0.1, 5, 1)), createBox(createFreeParam('w', 0.1, 5, 1), createFreeParam('h', 0.1, 5, 1), createFreeParam('d', 0.1, 5, 1))),
        createSDFHole('hole_sdf_1'),
        createSDFMacro('gyroid', { scale: createFreeParam('s', 0.1, 5, 1.0) }),
      ];

      for (const node of nodes) {
        const serialized = serializeToLeanJson(node);
        const deserialized = deserializeFromLeanJson(serialized, 'sdf');
        expect(areNodesEqual(deserialized, node)).toBe(true);
      }
    });

    it('round-trips all Hardware primitive nodes', () => {
      const nodes: HardwareNode[] = [
        createReg('rax', 0),
        createImm(255),
        createAdd(createReg('r0'), createImm(1), 'r0'),
        createSub(createReg('r0'), createImm(1), 'r0'),
        createMul(createReg('r0'), createReg('r1'), 'r2'),
        createShl(createReg('r0'), 4, 'r0'),
        createShr(createReg('r0'), 2, 'r0'),
        createSeq(createAdd(createReg('r0'), createImm(1)), createSub(createReg('r0'), createImm(2))),
        createHardwareHole('hw_hole_0'),
      ];

      for (const node of nodes) {
        const serialized = serializeToLeanJson(node);
        const deserialized = deserializeFromLeanJson(serialized, 'hardware');
        expect(areNodesEqual(deserialized, node)).toBe(true);
      }
    });

    it('round-trips Universal nodes and generic Macro calls', () => {
      const uAudio = createUniversalAudio(createSaw(createFreeParam('f', 20, 2000, 440)));
      const uSdf = createUniversalSDF(createSphere(createFreeParam('r', 0.1, 5, 1.0)));
      const uHw = createUniversalHardware(createAdd(createReg('r1'), createImm(10)));
      const uMacro = createUniversalMacro('combined', [uAudio, uSdf, uHw]);

      expect(areNodesEqual(deserializeFromLeanJson(serializeToLeanJson(uAudio)), uAudio)).toBe(true);
      expect(areNodesEqual(deserializeFromLeanJson(serializeToLeanJson(uSdf)), uSdf)).toBe(true);
      expect(areNodesEqual(deserializeFromLeanJson(serializeToLeanJson(uHw)), uHw)).toBe(true);
      expect(areNodesEqual(deserializeFromLeanJson(serializeToLeanJson(uMacro)), uMacro)).toBe(true);

      const genericHole = createHole('any_hole_0');
      expect(areNodesEqual(deserializeFromLeanJson(serializeToLeanJson(genericHole)), genericHole)).toBe(true);

      const genericMacro = createMacroCall('customMacro', { p: createFreeParam('param', 0, 1, 0.5) }, 'generic');
      expect(areNodesEqual(deserializeFromLeanJson(serializeToLeanJson(genericMacro)), genericMacro)).toBe(true);
    });

    it('faithfully deserializes canonical Lean 4 CLI output for exampleAudioPatch', () => {
      // Direct raw JSON produced by Lean 4 CLI: symforge_cli example-audio
      const leanAudioJson = {
        lowpass: {
          cutoff: { defaultVal: 1200, id: 'fc', maxVal: 10000, minVal: 100 },
          input: {
            saw: {
              freq: { defaultVal: 440, id: 'f0', maxVal: 20000, minVal: 20 },
            },
          },
          resonance: { defaultVal: 0.707, id: 'q', maxVal: 10, minVal: 0.1 },
        },
      };

      const node = deserializeFromLeanJson(leanAudioJson, 'audio');
      expect(node.kind).toBe('lowpass');
      expect(getNodeCount(node)).toBe(2);
      expect(getTreeDepth(node)).toBe(2);
      expect(collectFreeParams(node)).toHaveLength(3);

      // Re-serializing should match Lean 4 schema
      const reSerialized = serializeToLeanJson(node);
      expect(reSerialized).toEqual(leanAudioJson);
    });

    it('faithfully deserializes canonical Lean 4 CLI output for exampleSDFPatch', () => {
      // Direct raw JSON produced by Lean 4 CLI: symforge_cli example-sdf
      const leanSdfJson = {
        union: {
          a: {
            sphere: {
              radius: { defaultVal: 1, id: 'radius', maxVal: 5, minVal: 0.01 },
            },
          },
          b: {
            box: {
              depth: { defaultVal: 0.8, id: 'd', maxVal: 5, minVal: 0.01 },
              height: { defaultVal: 0.8, id: 'h', maxVal: 5, minVal: 0.01 },
              width: { defaultVal: 0.8, id: 'w', maxVal: 5, minVal: 0.01 },
            },
          },
          smoothness: { defaultVal: 0.2, id: 'k_smooth', maxVal: 1, minVal: 0 },
        },
      };

      const node = deserializeFromLeanJson(leanSdfJson, 'sdf');
      expect(node.kind).toBe('union');
      expect(getNodeCount(node)).toBe(3);
      expect(getTreeDepth(node)).toBe(2);
      expect(collectFreeParams(node)).toHaveLength(5);

      // Re-serializing should match Lean 4 schema
      const reSerialized = serializeToLeanJson(node);
      expect(reSerialized).toEqual(leanSdfJson);
    });
  });

  describe('Pass-to-Complete Gate: Property-Based 500 Random AST Trees', () => {
    // Deterministic pseudo-random generator (LCG) to assure 100% reproducible test runs
    class DeterministicRNG {
      private state: number;
      constructor(seed = 123456789) {
        this.state = seed >>> 0;
      }
      next(): number {
        this.state = (1664525 * this.state + 1013904223) >>> 0;
        return this.state / 4294967296;
      }
      nextInt(min: number, max: number): number {
        return Math.floor(this.next() * (max - min + 1)) + min;
      }
      nextFloat(min: number, max: number): number {
        return min + this.next() * (max - min);
      }
    }

    const rng = new DeterministicRNG(424242);

    function randomFreeParam(prefix: string, index: number): ReturnType<typeof createFreeParam> {
      const min = Number(rng.nextFloat(0.01, 10).toFixed(2));
      const max = Number(rng.nextFloat(min + 1, min + 100).toFixed(2));
      const def = Number(rng.nextFloat(min, max).toFixed(2));
      return createFreeParam(`${prefix}_${index}`, min, max, def);
    }

    function generateRandomAudioNode(level: number, idPrefix: string): AudioNode {
      if (level <= 1) {
        // L1 Atomic
        const choices = ['sine', 'saw', 'square', 'noise', 'fm'];
        const chosen = choices[rng.nextInt(0, choices.length - 1)];
        switch (chosen) {
          case 'sine':
            return createSine(randomFreeParam(`${idPrefix}_f`, 1), randomFreeParam(`${idPrefix}_p`, 2));
          case 'saw':
            return createSaw(randomFreeParam(`${idPrefix}_f`, 1));
          case 'square':
            return createSquare(randomFreeParam(`${idPrefix}_f`, 1), randomFreeParam(`${idPrefix}_d`, 2));
          case 'noise':
            return createNoise(rng.nextInt(1, 10000));
          case 'fm':
          default:
            return createFM(
              randomFreeParam(`${idPrefix}_c`, 1),
              randomFreeParam(`${idPrefix}_mf`, 2),
              randomFreeParam(`${idPrefix}_mi`, 3)
            );
        }
      }

      if (level === 2) {
        // L2 Unary Transform
        const child = generateRandomAudioNode(1, `${idPrefix}_sub`);
        const op = rng.nextInt(0, 1);
        if (op === 0) {
          return createLowPass(randomFreeParam(`${idPrefix}_fc`, 1), randomFreeParam(`${idPrefix}_q`, 2), child);
        } else {
          return createGain(randomFreeParam(`${idPrefix}_g`, 1), child);
        }
      }

      if (level === 3) {
        // L3 Binary Combinator
        const child1 = generateRandomAudioNode(rng.nextInt(1, 2), `${idPrefix}_c1`);
        const child2 = generateRandomAudioNode(rng.nextInt(1, 2), `${idPrefix}_c2`);
        const op = rng.nextInt(0, 1);
        if (op === 0) {
          return createMix(randomFreeParam(`${idPrefix}_ratio`, 1), child1, child2);
        } else {
          return createADSR(
            randomFreeParam(`${idPrefix}_a`, 1),
            randomFreeParam(`${idPrefix}_d`, 2),
            randomFreeParam(`${idPrefix}_s`, 3),
            randomFreeParam(`${idPrefix}_r`, 4),
            child1
          );
        }
      }

      // L4 Hierarchical Compositions
      const child1 = generateRandomAudioNode(rng.nextInt(2, 3), `${idPrefix}_h1`);
      const child2 = generateRandomAudioNode(rng.nextInt(1, 2), `${idPrefix}_h2`);
      return createMix(randomFreeParam(`${idPrefix}_mix`, 1), child1, child2);
    }

    function generateRandomSDFNode(level: number, idPrefix: string): SDFNode {
      if (level <= 1) {
        // L1 Atomic
        const choices = ['sphere', 'box', 'cylinder', 'torus'];
        const chosen = choices[rng.nextInt(0, choices.length - 1)];
        switch (chosen) {
          case 'sphere':
            return createSphere(randomFreeParam(`${idPrefix}_r`, 1));
          case 'box':
            return createBox(
              randomFreeParam(`${idPrefix}_w`, 1),
              randomFreeParam(`${idPrefix}_h`, 2),
              randomFreeParam(`${idPrefix}_d`, 3)
            );
          case 'cylinder':
            return createCylinder(randomFreeParam(`${idPrefix}_r`, 1), randomFreeParam(`${idPrefix}_h`, 2));
          case 'torus':
          default:
            return createTorus(randomFreeParam(`${idPrefix}_R`, 1), randomFreeParam(`${idPrefix}_r`, 2));
        }
      }

      if (level === 2) {
        // L2 Unary Transform
        const child = generateRandomSDFNode(1, `${idPrefix}_sub`);
        const op = rng.nextInt(0, 2);
        if (op === 0) {
          return createTranslate(
            randomFreeParam(`${idPrefix}_dx`, 1),
            randomFreeParam(`${idPrefix}_dy`, 2),
            randomFreeParam(`${idPrefix}_dz`, 3),
            child
          );
        } else if (op === 1) {
          return createRotate(
            randomFreeParam(`${idPrefix}_rx`, 1),
            randomFreeParam(`${idPrefix}_ry`, 2),
            randomFreeParam(`${idPrefix}_rz`, 3),
            child
          );
        } else {
          return createScale(randomFreeParam(`${idPrefix}_s`, 1), child);
        }
      }

      if (level === 3) {
        // L3 Binary Combinator
        const child1 = generateRandomSDFNode(rng.nextInt(1, 2), `${idPrefix}_c1`);
        const child2 = generateRandomSDFNode(rng.nextInt(1, 2), `${idPrefix}_c2`);
        const op = rng.nextInt(0, 3);
        switch (op) {
          case 0:
            return createUnion(randomFreeParam(`${idPrefix}_sm`, 1), child1, child2);
          case 1:
            return createSmoothUnion(randomFreeParam(`${idPrefix}_sm`, 1), child1, child2);
          case 2:
            return createIntersection(child1, child2);
          case 3:
          default:
            return createDifference(child1, child2);
        }
      }

      // L4 Hierarchical Compositions
      const child1 = generateRandomSDFNode(rng.nextInt(2, 3), `${idPrefix}_h1`);
      const child2 = generateRandomSDFNode(rng.nextInt(1, 2), `${idPrefix}_h2`);
      const op = rng.nextInt(0, 1);
      if (op === 0) {
        return createUnion(randomFreeParam(`${idPrefix}_sm`, 1), child1, child2);
      } else {
        return createDifference(child1, child2);
      }
    }

    it('passes 100% roundtrip serialization identity across 500 random AST trees (L1-L4)', () => {
      const TOTAL_TREES = 500;
      let audioSuccessCount = 0;
      let sdfSuccessCount = 0;

      for (let i = 0; i < TOTAL_TREES; i++) {
        // Alternate between Audio and SDF trees
        const isAudio = i % 2 === 0;
        const level = (i % 4) + 1; // Stratified cycle through L1, L2, L3, L4
        const idPrefix = `p${i}_l${level}`;

        const tree: ASTNode = isAudio
          ? generateRandomAudioNode(level, idPrefix)
          : generateRandomSDFNode(level, idPrefix);

        // 1. Verify structural invariants
        const count = getNodeCount(tree);
        const depth = getTreeDepth(tree);
        expect(count).toBeGreaterThanOrEqual(1);
        expect(depth).toBeGreaterThanOrEqual(1);
        expect(depth).toBeLessThanOrEqual(count);

        // 2. Verify unique parameter collection
        const params = collectFreeParams(tree);
        const ids = params.map((p) => p.id);
        const uniqueIds = new Set(ids);
        expect(uniqueIds.size).toBe(ids.length);

        // 3. Verify serialization roundtrip identity
        const serialized = serializeToLeanJson(tree);
        const domain = isAudio ? 'audio' : 'sdf';
        const deserialized = deserializeFromLeanJson(serialized, domain);

        const areEqual = areNodesEqual(deserialized, tree);
        expect(areEqual).toBe(true);

        // 4. Verify cloneWithParams preserves structure while updating targeted parameters
        if (params.length > 0) {
          const targetParam = params[0];
          const updatedValue = targetParam.defaultVal + 0.5;
          const cloned = cloneWithParams(tree, { [targetParam.id]: updatedValue });
          expect(getNodeCount(cloned)).toBe(count);
          expect(getTreeDepth(cloned)).toBe(depth);
          const clonedParams = collectFreeParams(cloned);
          const matchingCloned = clonedParams.find((p) => p.id === targetParam.id);
          expect(matchingCloned?.defaultVal).toBe(updatedValue);
        }

        if (isAudio) audioSuccessCount++;
        else sdfSuccessCount++;
      }

      expect(audioSuccessCount + sdfSuccessCount).toBe(500);
      expect(audioSuccessCount).toBe(250);
      expect(sdfSuccessCount).toBe(250);
    });
  });
});
