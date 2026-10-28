import { describe, it, expect } from 'vitest';
import {
  sampleColorMap,
  sampleColorMap255,
  Rastrigin5DFunction,
  Rosenbrock4DFunction,
  Ackley5DFunction,
  MultiWaveRipple5DFunction,
  BENCHMARK_FUNCTIONS,
  createFunctionFromAST,
  evaluate2DSlice,
} from './functionEvaluator';
import type { ColorMapType } from './types';
import {
  createSphere,
  createBox,
  createSine,
  createFreeParam,
  createNoise,
} from '../../../core/ast/models';

describe('Hyperdimensional Function Evaluator', () => {
  // ==========================================
  // 1. Analytical Benchmark Functions Accuracy
  // ==========================================
  describe('Analytical Accuracy Against Invariant Formulas', () => {
    it('evaluates Rastrigin 5D accurately at global optimum and known test points', () => {
      // f(0,0,0,0,0) = 0
      const origin = new Float32Array([0, 0, 0, 0, 0]);
      expect(Rastrigin5DFunction.evaluate(origin)).toBeCloseTo(0.0, 6);

      // f(1,1,1,1,1) = 5 * (1^2 - 10*cos(2pi) + 10) = 5 * (1 - 10 + 10) = 5
      const ones = new Float32Array([1, 1, 1, 1, 1]);
      expect(Rastrigin5DFunction.evaluate(ones)).toBeCloseTo(5.0, 6);

      // f(1,0,0,0,0) = 1
      const single = new Float32Array([1, 0, 0, 0, 0]);
      expect(Rastrigin5DFunction.evaluate(single)).toBeCloseTo(1.0, 6);

      // General analytical comparison
      const testCoords = [0.5, -0.25, 1.2, -1.0, 0.75];
      let expected = 0.0;
      for (const x of testCoords) {
        expected += x * x - 10.0 * Math.cos(2.0 * Math.PI * x) + 10.0;
      }
      expect(Rastrigin5DFunction.evaluate(testCoords)).toBeCloseTo(expected, 6);
    });

    it('evaluates Rosenbrock 4D accurately at global optimum and known test points', () => {
      // Global optimum: f(1,1,1,1) = 0
      const ones = new Float32Array([1, 1, 1, 1]);
      expect(Rosenbrock4DFunction.evaluate(ones)).toBeCloseTo(0.0, 6);

      // Origin: f(0,0,0,0) = 3 * (100*(0 - 0)^2 + (1 - 0)^2) = 3
      const origin = new Float32Array([0, 0, 0, 0]);
      expect(Rosenbrock4DFunction.evaluate(origin)).toBeCloseTo(3.0, 6);

      // Point (1, 2, 3, 4):
      // i=0: 100*(2 - 1)^2 + (1 - 1)^2 = 100
      // i=1: 100*(3 - 4)^2 + (1 - 2)^2 = 100 + 1 = 101
      // i=2: 100*(4 - 9)^2 + (1 - 3)^2 = 100*25 + 4 = 2504
      // Total = 100 + 101 + 2504 = 2705
      const seq = new Float32Array([1, 2, 3, 4]);
      expect(Rosenbrock4DFunction.evaluate(seq)).toBeCloseTo(2705.0, 6);
    });

    it('evaluates Ackley 5D accurately at global optimum and known test points', () => {
      // Global optimum: f(0,0,0,0,0) = 0
      const origin = new Float32Array([0, 0, 0, 0, 0]);
      expect(Ackley5DFunction.evaluate(origin)).toBeCloseTo(0.0, 6);

      // Analytical check at (0.5, 0.5, 0.5, 0.5, 0.5)
      const half = [0.5, 0.5, 0.5, 0.5, 0.5];
      const sumSq = 5 * 0.25;
      const sumCos = 5 * Math.cos(Math.PI);
      const expected =
        -20.0 * Math.exp(-0.2 * Math.sqrt(sumSq / 5.0)) -
        Math.exp(sumCos / 5.0) +
        20.0 +
        Math.E;
      expect(Ackley5DFunction.evaluate(half)).toBeCloseTo(expected, 6);
    });

    it('evaluates Multi-Wave Ripple 5D accurately', () => {
      // Origin: sin(0) * exp(0) = 0
      const origin = new Float32Array([0, 0, 0, 0, 0]);
      expect(MultiWaveRipple5DFunction.evaluate(origin)).toBeCloseTo(0.0, 6);

      // Peak ripple: r = pi / 6 with omega = 3.0 => sin(pi / 2) = 1.0; orthogonal = 0 => 1.0
      const peak = new Float32Array([Math.PI / 6, 0, 0, 0, 0]);
      expect(MultiWaveRipple5DFunction.evaluate(peak)).toBeCloseTo(1.0, 6);

      // Radial symmetry: same radius gives same ripple
      const peakY = new Float32Array([0, Math.PI / 6, 0, 0, 0]);
      expect(MultiWaveRipple5DFunction.evaluate(peakY)).toBeCloseTo(1.0, 6);

      // Orthogonal exponential damping
      const damped = new Float32Array([Math.PI / 6, 0, 2, 0, 0]);
      const expectedDamped = 1.0 * Math.exp(-0.08 * (2 * 2));
      expect(MultiWaveRipple5DFunction.evaluate(damped)).toBeCloseTo(expectedDamped, 6);
    });

    it('registers all 4 benchmark functions in BENCHMARK_FUNCTIONS', () => {
      expect(BENCHMARK_FUNCTIONS.length).toBe(4);
      const ids = BENCHMARK_FUNCTIONS.map((f) => f.id);
      expect(ids).toContain('rastrigin-5d');
      expect(ids).toContain('rosenbrock-4d');
      expect(ids).toContain('ackley-5d');
      expect(ids).toContain('multi-wave-5d');
    });
  });

  // ==========================================
  // 2. Orthogonal Coordinate Pinning
  // ==========================================
  describe('Orthogonal Coordinate Pinning & Projection', () => {
    it('preserves unselected orthogonal coordinates and shifts slice values deterministically', () => {
      const res = 10;
      const sliceCoordsA = new Float32Array([0, 0, 0, 0, 0]);
      const sliceCoordsB = new Float32Array([0, 0, 2.0, 0, 0]); // x_2 shifted to 2.0

      // Evaluate (x_0, x_1) slice with x_2 = 0 vs x_2 = 2.0
      const resultA = evaluate2DSlice(Rastrigin5DFunction, 0, 1, sliceCoordsA, res, res);
      const resultB = evaluate2DSlice(Rastrigin5DFunction, 0, 1, sliceCoordsB, res, res);

      // In Rastrigin 5D, shifting x_2 from 0 to 2 adds:
      // (2^2 - 10*cos(4pi) + 10) - (0^2 - 10*cos(0) + 10) = (4 - 10 + 10) - 0 = 4.0
      const expectedDelta = 4.0;

      for (let i = 0; i < res * res; i++) {
        const delta = resultB.values[i] - resultA.values[i];
        expect(delta).toBeCloseTo(expectedDelta, 5);
      }

      // Verify sliceCoords returned in result accurately records pinned coordinates
      expect(resultB.sliceCoords[2]).toBeCloseTo(2.0, 6);
      expect(resultB.sliceCoords[3]).toBeCloseTo(0.0, 6);
      expect(resultB.sliceCoords[4]).toBeCloseTo(0.0, 6);
    });

    it('throws descriptive error on invalid axis selections', () => {
      const slice = new Float32Array([0, 0, 0, 0, 0]);
      // Same axis
      expect(() => evaluate2DSlice(Rastrigin5DFunction, 1, 1, slice, 10, 10)).toThrow(
        /Invalid axis pair/
      );
      // Out of bounds
      expect(() => evaluate2DSlice(Rastrigin5DFunction, -1, 2, slice, 10, 10)).toThrow(
        /Invalid axis pair/
      );
      expect(() => evaluate2DSlice(Rastrigin5DFunction, 0, 7, slice, 10, 10)).toThrow(
        /Invalid axis pair/
      );
    });

    it('correctly reuses provided output buffer without heap reallocation', () => {
      const preallocated = new Float32Array(50 * 50);
      const slice = new Float32Array([0, 0, 0, 0, 0]);

      const result = evaluate2DSlice(
        Rastrigin5DFunction,
        0,
        1,
        slice,
        50,
        50,
        preallocated
      );

      expect(result.values).toBe(preallocated);
      expect(result.values.length).toBe(2500);
      expect(result.minVal).toBeLessThanOrEqual(result.maxVal);
    });
  });

  // ==========================================
  // 3. Colormap Polynomial Approximations
  // ==========================================
  describe('Colormap Polynomial Sampling', () => {
    const colormaps: ColorMapType[] = ['viridis', 'turbo', 'magma', 'coolwarm'];

    it('generates valid bounded RGB channels across boundaries and midpoints for all colormaps', () => {
      for (const cm of colormaps) {
        for (const t of [0.0, 0.25, 0.5, 0.75, 1.0]) {
          const [r, g, b] = sampleColorMap(t, cm);
          expect(r).toBeGreaterThanOrEqual(0.0);
          expect(r).toBeLessThanOrEqual(1.0);
          expect(g).toBeGreaterThanOrEqual(0.0);
          expect(g).toBeLessThanOrEqual(1.0);
          expect(b).toBeGreaterThanOrEqual(0.0);
          expect(b).toBeLessThanOrEqual(1.0);
        }
      }
    });

    it('clamps out-of-bounds t values gracefully', () => {
      const [rMin, gMin, bMin] = sampleColorMap(-0.5, 'viridis');
      const [rZero, gZero, bZero] = sampleColorMap(0.0, 'viridis');
      expect(rMin).toBe(rZero);
      expect(gMin).toBe(gZero);
      expect(bMin).toBe(bZero);

      const [rMax, gMax, bMax] = sampleColorMap(1.5, 'viridis');
      const [rOne, gOne, bOne] = sampleColorMap(1.0, 'viridis');
      expect(rMax).toBe(rOne);
      expect(gMax).toBe(gOne);
      expect(bMax).toBe(bOne);
    });

    it('verifies colormap qualitative domain invariants', () => {
      // Viridis: low t is purple/blue (b > r), high t is yellow (r > 0.8, g > 0.8, b < 0.3)
      const viridis0 = sampleColorMap(0.0, 'viridis');
      expect(viridis0[2]).toBeGreaterThan(viridis0[0]);

      const viridis1 = sampleColorMap(1.0, 'viridis');
      expect(viridis1[0]).toBeGreaterThan(0.8);
      expect(viridis1[1]).toBeGreaterThan(0.8);
      expect(viridis1[2]).toBeLessThan(0.3);

      // Coolwarm: low t is cool blue (b > r), high t is warm red (r > b)
      const cw0 = sampleColorMap(0.0, 'coolwarm');
      expect(cw0[2]).toBeGreaterThan(cw0[0]);

      const cw1 = sampleColorMap(1.0, 'coolwarm');
      expect(cw1[0]).toBeGreaterThan(cw1[2]);

      // Magma: low t is near-black (< 0.05 intensity), high t is bright (> 0.7)
      const magma0 = sampleColorMap(0.0, 'magma');
      expect(magma0[0] + magma0[1] + magma0[2]).toBeLessThan(0.05);

      const magma1 = sampleColorMap(1.0, 'magma');
      expect(magma1[0]).toBeGreaterThan(0.8);
      expect(magma1[1]).toBeGreaterThan(0.8);
    });

    it('produces integer 8-bit RGB channels via sampleColorMap255', () => {
      const [r, g, b] = sampleColorMap255(0.5, 'turbo');
      expect(Number.isInteger(r)).toBe(true);
      expect(Number.isInteger(g)).toBe(true);
      expect(Number.isInteger(b)).toBe(true);
      expect(r).toBeGreaterThanOrEqual(0);
      expect(r).toBeLessThanOrEqual(255);
    });
  });

  // ==========================================
  // 4. AST-Derived Function Generation
  // ==========================================
  describe('AST-Derived Function Generator', () => {
    it('derives continuous function from SDF AST with FreeParams', () => {
      const radiusParam = createFreeParam('sphere_radius', 0.2, 4.0, 1.5);
      const sphereNode = createSphere(radiusParam);

      const fn = createFunctionFromAST(sphereNode, 'MySphereSDF');
      expect(fn.name).toBe('MySphereSDF');
      expect(fn.inputDim).toBeGreaterThanOrEqual(2);
      expect(fn.dimensions[0].id).toBe('sphere_radius');
      expect(fn.dimensions[0].defaultValue).toBe(1.5);

      // Evaluate function at default coordinates
      const coords = new Float32Array(fn.inputDim);
      for (let i = 0; i < fn.inputDim; i++) coords[i] = fn.dimensions[i].defaultValue;

      const val1 = fn.evaluate(coords);
      expect(typeof val1).toBe('number');
      expect(Number.isFinite(val1)).toBe(true);

      // Shifting radius modulates evaluated distance
      coords[0] = 3.0;
      const val2 = fn.evaluate(coords);
      expect(val2).not.toBe(val1);
    });

    it('derives continuous function from multi-parameter composite SDF', () => {
      const boxNode = createBox(
        createFreeParam('width', 0.5, 5.0, 2.0),
        createFreeParam('height', 0.5, 5.0, 1.0),
        createFreeParam('depth', 0.5, 5.0, 3.0)
      );

      const fn = createFunctionFromAST(boxNode);
      expect(fn.inputDim).toBe(3);
      expect(fn.dimensions.map((d) => d.id)).toEqual(['width', 'height', 'depth']);

      const sliceResult = evaluate2DSlice(
        fn,
        0,
        1,
        new Float32Array([2.0, 1.0, 3.0]),
        20,
        20
      );
      expect(sliceResult.values.length).toBe(400);
      expect(sliceResult.minVal).toBeLessThan(sliceResult.maxVal);
    });

    it('derives continuous function from Audio AST', () => {
      const freqParam = createFreeParam('freq', 100, 1000, 440);
      const phaseParam = createFreeParam('phase', 0, 6.28, 0);
      const sineNode = createSine(freqParam, phaseParam);

      const fn = createFunctionFromAST(sineNode);
      expect(fn.inputDim).toBe(2);
      expect(fn.dimensions[0].id).toBe('freq');
      expect(fn.dimensions[1].id).toBe('phase');

      const evalRes = fn.evaluate([440, 0]);
      expect(typeof evalRes).toBe('number');
      expect(Number.isFinite(evalRes)).toBe(true);
    });

    it('handles zero-parameter AST by generating canonical continuous coordinates', () => {
      const noiseNode = createNoise(12345);
      const fn = createFunctionFromAST(noiseNode);

      expect(fn.inputDim).toBe(2);
      expect(fn.dimensions[0].id).toBe('x_0');
      expect(fn.dimensions[1].id).toBe('x_1');

      const sliceRes = evaluate2DSlice(fn, 0, 1, new Float32Array([0, 0]), 15, 15);
      expect(sliceRes.values.length).toBe(225);
    });
  });

  // ==========================================
  // 5. Pass-to-Complete Gate: Latency Benchmark
  // ==========================================
  describe('Pass-to-Complete Gate: Latency Benchmark', () => {
    it('evaluates 100 x 100 (10,000 points) across 5 dimensions in < 15 ms', () => {
      const res = 100;
      const sliceCoords = new Float32Array([0.0, 0.0, 0.5, -0.5, 1.0]);
      const preallocatedBuffer = new Float32Array(res * res);

      // JIT warm-up pass
      evaluate2DSlice(
        Rastrigin5DFunction,
        0,
        1,
        sliceCoords,
        res,
        res,
        preallocatedBuffer
      );

      // Timed execution benchmark
      const start = performance.now();
      const result = evaluate2DSlice(
        Rastrigin5DFunction,
        0,
        1,
        sliceCoords,
        res,
        res,
        preallocatedBuffer
      );
      const elapsedMs = performance.now() - start;

      // Assert correctness of evaluated data
      expect(result.values.length).toBe(10000);
      expect(result.minVal).toBeLessThanOrEqual(result.maxVal);
      expect(Number.isFinite(result.minVal)).toBe(true);
      expect(Number.isFinite(result.maxVal)).toBe(true);

      // Strict latency gate threshold: < 15 ms
      expect(elapsedMs).toBeLessThan(15.0);
    });

    it('evaluates Ackley 5D 100 x 100 in < 15 ms with buffer allocation', () => {
      const res = 100;
      const sliceCoords = new Float32Array([0, 0, 0, 0, 0]);

      // Warm up
      evaluate2DSlice(Ackley5DFunction, 0, 1, sliceCoords, res, res);

      const start = performance.now();
      const result = evaluate2DSlice(Ackley5DFunction, 0, 1, sliceCoords, res, res);
      const elapsedMs = performance.now() - start;

      expect(result.values.length).toBe(10000);
      expect(elapsedMs).toBeLessThan(15.0);
    });
  });
});
