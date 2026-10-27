/**
 * FreeParam Reactive State Store Unit & Performance Latency Tests.
 *
 * Validates:
 * 1. Parameter extraction count on Audio and SDF ASTs.
 * 2. Parameter clamping, value preservation, and state mutations.
 * 3. Immutable AST cloning with modified parameter values via getUpdatedAST().
 * 4. Microsecond subscriber notification callbacks via subscribeParam.
 * 5. Latency benchmark measuring 1,000 parameter updates (< 1 ms mean latency gate).
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import {
  createFreeParamStore,
  useFreeParamStore,
} from './freeParamStore';
import {
  createFreeParam,
  createSphere,
  createBox,
  createIntersection,
  createSaw,
  createADSR,
  createLowPass,
} from '../ast/models';
import type { SphereSDFNode, LowPassAudioNode } from '../ast/types';

describe('FreeParam Reactive State Store (freeParamStore)', () => {
  let store: ReturnType<typeof createFreeParamStore>;

  beforeEach(() => {
    store = createFreeParamStore();
  });

  describe('1. AST Parameter Extraction', () => {
    it('initializes with empty state when AST is null', () => {
      expect(store.getState().ast).toBeNull();
      expect(store.getState().paramOrder).toEqual([]);
      expect(store.getState().params).toEqual({});
      expect(store.getState().values).toEqual({});
    });

    it('extracts FreeParams from 3D SDF geometry ASTs', () => {
      const sphere = createSphere(createFreeParam('radius', 0.1, 5.0, 1.25));
      const box = createBox(
        createFreeParam('width', 0.1, 10.0, 2.0),
        createFreeParam('height', 0.1, 10.0, 3.0),
        createFreeParam('depth', 0.1, 10.0, 1.0)
      );
      const csg = createIntersection(sphere, box);

      store.getState().setAST(csg, 'myCSG', 'SDFNode');

      const state = store.getState();
      expect(state.declarationName).toBe('myCSG');
      expect(state.domainType).toBe('SDFNode');
      expect(state.paramOrder).toEqual(['radius', 'width', 'height', 'depth']);
      expect(Object.keys(state.params)).toHaveLength(4);
      expect(state.values).toEqual({
        radius: 1.25,
        width: 2.0,
        height: 3.0,
        depth: 1.0,
      });
    });

    it('extracts FreeParams from Procedural Audio DSP ASTs', () => {
      const saw = createSaw(createFreeParam('f0', 20.0, 20000.0, 440.0));
      const adsr = createADSR(
        createFreeParam('atk', 0.001, 2.0, 0.05),
        createFreeParam('dec', 0.001, 2.0, 0.1),
        createFreeParam('sus', 0.0, 1.0, 0.7),
        createFreeParam('rel', 0.001, 5.0, 0.3),
        saw
      );
      const filter = createLowPass(
        createFreeParam('fc', 20.0, 20000.0, 1200.0),
        createFreeParam('q', 0.1, 10.0, 0.707),
        adsr
      );

      store.getState().setAST(filter, 'mySynth', 'AudioNode');

      const state = store.getState();
      expect(state.declarationName).toBe('mySynth');
      expect(state.domainType).toBe('AudioNode');
      expect(state.paramOrder).toEqual(['fc', 'q', 'atk', 'dec', 'sus', 'rel', 'f0']);
      expect(Object.keys(state.params)).toHaveLength(7);
      expect(state.values.fc).toBe(1200.0);
      expect(state.values.q).toBe(0.707);
      expect(state.values.f0).toBe(440.0);
    });

    it('preserves existing user-tweaked values when AST is re-set with matching parameter IDs', () => {
      const sphere1 = createSphere(createFreeParam('radius', 0.1, 5.0, 1.25));
      store.getState().setAST(sphere1, 'shape', 'SDFNode');

      // User tweaks radius
      store.getState().updateParamValue('radius', 3.8);
      expect(store.getState().values.radius).toBe(3.8);

      // Re-parse/re-set AST containing the same parameter ID plus a new one
      const sphere2 = createSphere(createFreeParam('radius', 0.1, 5.0, 1.25));
      const box = createBox(
        createFreeParam('width', 0.1, 10.0, 2.0),
        createFreeParam('height', 0.1, 10.0, 3.0),
        createFreeParam('depth', 0.1, 10.0, 1.0)
      );
      const combined = createIntersection(sphere2, box);
      store.getState().setAST(combined, 'shape', 'SDFNode');

      // radius must stay 3.8, new params must take default values
      expect(store.getState().values.radius).toBe(3.8);
      expect(store.getState().values.width).toBe(2.0);
    });

    it('clears state cleanly when AST is set to null', () => {
      const sphere = createSphere(createFreeParam('radius', 0.1, 5.0, 1.25));
      store.getState().setAST(sphere, 'shape', 'SDFNode');
      expect(store.getState().paramOrder).toHaveLength(1);

      store.getState().setAST(null);
      expect(store.getState().ast).toBeNull();
      expect(store.getState().paramOrder).toHaveLength(0);
      expect(store.getState().params).toEqual({});
      expect(store.getState().values).toEqual({});
    });
  });

  describe('2. Parameter Clamping & State Mutations', () => {
    beforeEach(() => {
      const sphere = createSphere(createFreeParam('radius', 0.5, 3.0, 1.25));
      store.getState().setAST(sphere, 'mySphere', 'SDFNode');
    });

    it('updates parameter value within allowed range', () => {
      store.getState().updateParamValue('radius', 2.0);
      expect(store.getState().values.radius).toBe(2.0);
    });

    it('clamps values exceeding maxVal to maxVal', () => {
      store.getState().updateParamValue('radius', 10.0);
      expect(store.getState().values.radius).toBe(3.0);
    });

    it('clamps values below minVal to minVal', () => {
      store.getState().updateParamValue('radius', -5.0);
      expect(store.getState().values.radius).toBe(0.5);
    });

    it('ignores invalid numeric inputs (NaN, Infinity)', () => {
      store.getState().updateParamValue('radius', 2.0);
      store.getState().updateParamValue('radius', NaN);
      expect(store.getState().values.radius).toBe(2.0);

      store.getState().updateParamValue('radius', Infinity);
      expect(store.getState().values.radius).toBe(2.0);
    });

    it('safely ignores updates to non-existent parameter IDs', () => {
      store.getState().updateParamValue('non_existent', 42);
      expect(store.getState().values.non_existent).toBeUndefined();
    });

    it('resets a single parameter to its defaultVal', () => {
      store.getState().updateParamValue('radius', 2.8);
      expect(store.getState().values.radius).toBe(2.8);

      store.getState().resetParam('radius');
      expect(store.getState().values.radius).toBe(1.25);
    });

    it('resets all parameters simultaneously via resetAllParams', () => {
      const box = createBox(
        createFreeParam('w', 1, 10, 2),
        createFreeParam('h', 1, 10, 4),
        createFreeParam('d', 1, 10, 6)
      );
      store.getState().setAST(box, 'boxDecl', 'SDFNode');

      store.getState().updateParamValue('w', 9);
      store.getState().updateParamValue('h', 8);
      store.getState().updateParamValue('d', 7);

      expect(store.getState().values).toEqual({ w: 9, h: 8, d: 7 });

      store.getState().resetAllParams();
      expect(store.getState().values).toEqual({ w: 2, h: 4, d: 6 });
    });
  });

  describe('3. Immutable AST Cloning via getUpdatedAST()', () => {
    it('returns null when AST is null', () => {
      expect(store.getState().getUpdatedAST()).toBeNull();
    });

    it('returns a fresh cloned AST with modified parameter values without mutating root AST', () => {
      const originalSphere = createSphere(createFreeParam('radius', 0.5, 3.0, 1.25));
      store.getState().setAST(originalSphere, 'sphereDecl', 'SDFNode');

      // Tweak parameter in store
      store.getState().updateParamValue('radius', 2.75);

      const updatedAST = store.getState().getUpdatedAST() as SphereSDFNode;
      expect(updatedAST).not.toBeNull();
      expect(updatedAST.kind).toBe('sphere');
      expect(updatedAST.radius.defaultVal).toBe(2.75);

      // Verify canonical root AST reference remains untouched
      const currentCanonical = store.getState().ast as SphereSDFNode;
      expect(currentCanonical.radius.defaultVal).toBe(1.25);
    });

    it('correctly updates nested complex audio graph parameters in cloned AST', () => {
      const saw = createSaw(createFreeParam('f0', 20.0, 20000.0, 440.0));
      const lp = createLowPass(
        createFreeParam('cutoff', 100.0, 10000.0, 1200.0),
        createFreeParam('res', 0.1, 10.0, 0.707),
        saw
      );
      store.getState().setAST(lp, 'patch', 'AudioNode');

      store.getState().updateParamValue('cutoff', 3500.0);
      store.getState().updateParamValue('f0', 220.0);

      const cloned = store.getState().getUpdatedAST() as LowPassAudioNode;
      expect(cloned.cutoff.defaultVal).toBe(3500.0);
      expect((cloned.input as any).freq.defaultVal).toBe(220.0);
      expect(cloned.resonance.defaultVal).toBe(0.707); // Untouched
    });
  });

  describe('4. Direct Real-Time Subscriber Notifications (subscribeParam)', () => {
    it('invokes subscriber callback when parameter is updated', () => {
      const sphere = createSphere(createFreeParam('radius', 0.5, 3.0, 1.25));
      store.getState().setAST(sphere);

      const callback = vi.fn();
      const unsubscribe = store.getState().subscribeParam('radius', callback);

      expect(callback).not.toHaveBeenCalled();

      store.getState().updateParamValue('radius', 2.1);
      expect(callback).toHaveBeenCalledTimes(1);
      expect(callback).toHaveBeenCalledWith(2.1);

      store.getState().updateParamValue('radius', 2.9);
      expect(callback).toHaveBeenCalledTimes(2);
      expect(callback).toHaveBeenCalledWith(2.9);

      // Unsubscribe and verify no further calls
      unsubscribe();
      store.getState().updateParamValue('radius', 1.0);
      expect(callback).toHaveBeenCalledTimes(2);
    });

    it('notifies subscribers with defaultVal when resetParam or resetAllParams is called', () => {
      const sphere = createSphere(createFreeParam('radius', 0.5, 3.0, 1.25));
      store.getState().setAST(sphere);

      const callback = vi.fn();
      store.getState().subscribeParam('radius', callback);

      store.getState().updateParamValue('radius', 2.5);
      expect(callback).toHaveBeenLastCalledWith(2.5);

      store.getState().resetParam('radius');
      expect(callback).toHaveBeenLastCalledWith(1.25);

      store.getState().updateParamValue('radius', 2.8);
      expect(callback).toHaveBeenLastCalledWith(2.8);

      store.getState().resetAllParams();
      expect(callback).toHaveBeenLastCalledWith(1.25);
    });

    it('isolates subscriber errors preventing crash of updateParamValue', () => {
      const sphere = createSphere(createFreeParam('radius', 0.5, 3.0, 1.25));
      store.getState().setAST(sphere);

      const faultyCallback = vi.fn(() => {
        throw new Error('Subscriber failure simulation');
      });
      const healthyCallback = vi.fn();

      store.getState().subscribeParam('radius', faultyCallback);
      store.getState().subscribeParam('radius', healthyCallback);

      expect(() => {
        store.getState().updateParamValue('radius', 1.8);
      }).not.toThrow();

      expect(faultyCallback).toHaveBeenCalledWith(1.8);
      expect(healthyCallback).toHaveBeenCalledWith(1.8);
      expect(store.getState().values.radius).toBe(1.8);
    });
  });

  describe('5. Pass-to-Complete Gate: 1,000 Parameter Updates Latency Benchmark', () => {
    it('executes 1,000 parameter updates with mean latency strictly < 1 ms (typically < 0.05 ms)', () => {
      const sphere = createSphere(createFreeParam('radius', 0.0, 100.0, 50.0));
      store.getState().setAST(sphere);

      const subscriberFn = vi.fn();
      const unsub = store.getState().subscribeParam('radius', subscriberFn);

      const iterations = 1000;
      const startTime = performance.now();

      for (let i = 0; i < iterations; i++) {
        const value = (i % 100) + 0.5;
        store.getState().updateParamValue('radius', value);
      }

      const totalElapsedMs = performance.now() - startTime;
      const meanLatencyMs = totalElapsedMs / iterations;

      unsub();

      expect(subscriberFn).toHaveBeenCalledTimes(iterations);
      expect(store.getState().values.radius).toBe(99.5);

      // Quality Gate: Strict requirement is < 1 ms mean latency
      expect(meanLatencyMs).toBeLessThan(1.0);

      // Verify microsecond-class performance
      console.log(
        `[Benchmark] 1,000 FreeParam updates total: ${totalElapsedMs.toFixed(2)} ms, mean latency: ${(meanLatencyMs * 1000).toFixed(2)} µs/op`
      );
    });
  });

  describe('6. Singleton useFreeParamStore Verification', () => {
    it('provides global singleton store instance', () => {
      expect(useFreeParamStore).toBeDefined();
      expect(typeof useFreeParamStore.getState).toBe('function');
      expect(typeof useFreeParamStore.getState().setAST).toBe('function');
    });
  });
});
