import { describe, it, expect, beforeEach, vi } from 'vitest';
import * as THREE from 'three';
import {
  UniformBindingManager,
  bindMaterial,
  syncAllUniforms,
  createUniformRecord,
} from './uniformBinding';
import { createFreeParamStore, useFreeParamStore } from '../../../core/store/freeParamStore';
import { createFreeParam, createSphere, createBox, createUnion } from '../../../core/ast/models';
import type { FreeParam } from '../../../core/ast/types';
import { getParamUniformName } from './glslCodegen';

describe('UniformBindingManager & Real-Time FreeParam Uniform Binding', () => {
  beforeEach(() => {
    useFreeParamStore.getState().setAST(null);
  });

  describe('createUniformRecord', () => {
    it('creates Three.js uniform map with defaults when no initial values are provided', () => {
      const p1 = createFreeParam('radius', 0.1, 10.0, 2.5);
      const p2 = createFreeParam('offset_x', -5.0, 5.0, 0.0);

      const uniforms = createUniformRecord([p1, p2]);

      expect(uniforms).toHaveProperty('u_param_radius');
      expect(uniforms).toHaveProperty('u_param_offset_x');
      expect(uniforms.u_param_radius.value).toBe(2.5);
      expect(uniforms.u_param_offset_x.value).toBe(0.0);
    });

    it('creates Three.js uniform map using custom initial values', () => {
      const p1 = createFreeParam('radius', 0.1, 10.0, 2.5);
      const p2 = createFreeParam('offset_x', -5.0, 5.0, 0.0);

      const customValues = { radius: 4.2, offset_x: -1.8 };
      const uniforms = createUniformRecord([p1, p2], customValues);

      expect(uniforms.u_param_radius.value).toBe(4.2);
      expect(uniforms.u_param_offset_x.value).toBe(-1.8);
    });

    it('correctly sanitizes special characters in parameter IDs', () => {
      const pSpecial = createFreeParam('scale-factor.1', 0.1, 5.0, 1.0);
      const uniforms = createUniformRecord([pSpecial]);

      const expectedName = getParamUniformName(pSpecial);
      expect(expectedName).toBe('u_param_scale_factor_1');
      expect(uniforms).toHaveProperty('u_param_scale_factor_1');
      expect(uniforms.u_param_scale_factor_1.value).toBe(1.0);
    });
  });

  describe('syncAllUniforms', () => {
    it('synchronously copies values to matching material uniforms without touching needsUpdate or material.version', () => {
      const p1 = createFreeParam('radius', 0.1, 10.0, 1.0);
      const p2 = createFreeParam('thickness', 0.01, 2.0, 0.2);

      const uniforms = createUniformRecord([p1, p2]);
      const material = new THREE.ShaderMaterial({
        vertexShader: 'void main() { gl_Position = vec4(0.0); }',
        fragmentShader: 'void main() { gl_FragColor = vec4(1.0); }',
        uniforms,
      });

      const initialVersion = material.version;
      const needsUpdateSpy = vi.spyOn(material, 'needsUpdate', 'set');

      const newValues = { radius: 3.75, thickness: 0.85 };
      syncAllUniforms(material, [p1, p2], newValues);

      expect(material.uniforms.u_param_radius.value).toBe(3.75);
      expect(material.uniforms.u_param_thickness.value).toBe(0.85);
      // Critical GPU invariant: never trigger shader re-compilation / re-link
      expect(needsUpdateSpy).not.toHaveBeenCalled();
      expect(material.version).toBe(initialVersion);
    });

    it('safely ignores null materials or non-existent uniforms', () => {
      const p1 = createFreeParam('missing', 0.1, 5.0, 1.0);
      expect(() => {
        syncAllUniforms(null as unknown as THREE.ShaderMaterial, [p1], { missing: 2.0 });
      }).not.toThrow();

      const emptyMaterial = new THREE.ShaderMaterial({ uniforms: {} });
      expect(() => {
        syncAllUniforms(emptyMaterial, [p1], { missing: 2.0 });
      }).not.toThrow();
    });
  });

  describe('bindMaterial & Event-Driven Subscriptions', () => {
    it('binds material uniforms to store updates with instant synchronization', () => {
      const p1 = createFreeParam('r', 0.1, 10.0, 1.0);
      const ast = createSphere(p1);

      const store = createFreeParamStore(ast);
      const uniforms = createUniformRecord([p1]);
      const material = new THREE.ShaderMaterial({ uniforms });
      const initialVersion = material.version;
      const needsUpdateSpy = vi.spyOn(material, 'needsUpdate', 'set');

      const unbind = bindMaterial(material, [p1], store);

      // Verify initial state
      expect(material.uniforms.u_param_r.value).toBe(1.0);

      // Update store
      store.getState().updateParamValue('r', 4.5);

      // Verify immediate uniform update on GPU uniform buffer
      expect(material.uniforms.u_param_r.value).toBe(4.5);
      expect(needsUpdateSpy).not.toHaveBeenCalled();
      expect(material.version).toBe(initialVersion);

      // Update again
      store.getState().updateParamValue('r', 8.2);
      expect(material.uniforms.u_param_r.value).toBe(8.2);
      expect(needsUpdateSpy).not.toHaveBeenCalled();
      expect(material.version).toBe(initialVersion);

      unbind();
    });

    it('safely handles multiple parameters bound simultaneously', () => {
      const w = createFreeParam('width', 0.1, 10.0, 1.0);
      const h = createFreeParam('height', 0.1, 10.0, 2.0);
      const d = createFreeParam('depth', 0.1, 10.0, 3.0);
      const ast = createBox(w, h, d);

      const store = createFreeParamStore(ast);
      const uniforms = createUniformRecord([w, h, d]);
      const material = new THREE.ShaderMaterial({ uniforms });
      const initialVersion = material.version;
      const needsUpdateSpy = vi.spyOn(material, 'needsUpdate', 'set');

      const unbind = bindMaterial(material, [w, h, d], store);

      store.getState().updateParamValue('width', 5.5);
      store.getState().updateParamValue('height', 7.5);
      store.getState().updateParamValue('depth', 9.5);

      expect(material.uniforms.u_param_width.value).toBe(5.5);
      expect(material.uniforms.u_param_height.value).toBe(7.5);
      expect(material.uniforms.u_param_depth.value).toBe(9.5);
      expect(needsUpdateSpy).not.toHaveBeenCalled();
      expect(material.version).toBe(initialVersion);

      unbind();
    });

    it('cleans up all subscriptions on unbind to prevent dangling callbacks and memory leaks', () => {
      const p1 = createFreeParam('radius', 0.1, 5.0, 1.0);
      const ast = createSphere(p1);

      const store = createFreeParamStore(ast);
      const uniforms = createUniformRecord([p1]);
      const material = new THREE.ShaderMaterial({ uniforms });

      const unbind = bindMaterial(material, [p1], store);
      store.getState().updateParamValue('radius', 2.0);
      expect(material.uniforms.u_param_radius.value).toBe(2.0);

      // Unbind
      unbind();

      // Mutation after unbind must not affect material uniform
      store.getState().updateParamValue('radius', 4.0);
      expect(store.getState().values.radius).toBe(4.0);
      expect(material.uniforms.u_param_radius.value).toBe(2.0); // Stays at 2.0

      // Calling unbind again is idempotent
      expect(() => unbind()).not.toThrow();
    });
  });

  describe('UniformBindingManager Class API', () => {
    it('tracks active disposers and disposes all on manager.dispose()', () => {
      const p1 = createFreeParam('p1', 0, 10, 1);
      const p2 = createFreeParam('p2', 0, 10, 2);
      const ast = createBox(p1, p2, createFreeParam('p3', 0, 10, 3));
      const store = createFreeParamStore(ast);

      const manager = new UniformBindingManager(store);

      const mat1 = new THREE.ShaderMaterial({ uniforms: manager.createUniformRecord([p1]) });
      const mat2 = new THREE.ShaderMaterial({ uniforms: manager.createUniformRecord([p2]) });

      manager.bindMaterial(mat1, [p1]);
      manager.bindMaterial(mat2, [p2]);

      store.getState().updateParamValue('p1', 5);
      store.getState().updateParamValue('p2', 8);

      expect(mat1.uniforms.u_param_p1.value).toBe(5);
      expect(mat2.uniforms.u_param_p2.value).toBe(8);

      // Dispose manager
      manager.dispose();

      store.getState().updateParamValue('p1', 9);
      store.getState().updateParamValue('p2', 10);

      // Values should remain at pre-dispose state
      expect(mat1.uniforms.u_param_p1.value).toBe(5);
      expect(mat2.uniforms.u_param_p2.value).toBe(8);
    });

    it('provides static delegate methods', () => {
      const p1 = createFreeParam('stat_param', 0, 10, 3);
      const uniforms = UniformBindingManager.createUniformRecord([p1], { stat_param: 7 });
      expect(uniforms.u_param_stat_param.value).toBe(7);

      const mat = new THREE.ShaderMaterial({ uniforms });
      UniformBindingManager.syncAllUniforms(mat, [p1], { stat_param: 9 });
      expect(mat.uniforms.u_param_stat_param.value).toBe(9);
    });
  });

  describe('Pass-to-Complete Gate: High-Frequency Multi-Parameter Benchmark', () => {
    it('updates 10 simultaneous parameters across 1,000 updates with mean latency < 1.0 ms and zero re-links', () => {
      const paramCount = 10;
      const totalRounds = 100; // 100 rounds * 10 params = 1,000 updates
      const totalUpdates = paramCount * totalRounds;

      const params: FreeParam[] = [];
      for (let i = 0; i < paramCount; i++) {
        params.push(createFreeParam(`param_${i}`, 0.0, 5000.0, 10.0 + i));
      }

      // Build composite AST node containing all 10 parameters
      let ast: any = createSphere(params[0]);
      for (let i = 1; i < 9; i += 2) {
        ast = createUnion(params[i], ast, createSphere(params[i + 1]));
      }
      ast = createUnion(createFreeParam('dummy_sm', 0, 1, 0), ast, createSphere(params[9]));

      const store = createFreeParamStore(ast);

      const uniforms = createUniformRecord(params);
      const material = new THREE.ShaderMaterial({
        vertexShader: 'void main() { gl_Position = vec4(0.0); }',
        fragmentShader: 'void main() { gl_FragColor = vec4(1.0); }',
        uniforms,
      });

      const initialVersion = material.version;
      const needsUpdateSpy = vi.spyOn(material, 'needsUpdate', 'set');

      const unbind = bindMaterial(material, params, store);

      // Benchmark execution timing
      const startTime = performance.now();

      for (let round = 0; round < totalRounds; round++) {
        for (let i = 0; i < paramCount; i++) {
          const newVal = (round + 1) * 10.0 + i;
          store.getState().updateParamValue(params[i].id, newVal);
        }
      }

      const endTime = performance.now();
      const totalElapsedMs = endTime - startTime;
      const meanLatencyMs = totalElapsedMs / totalUpdates;

      // Assertions
      // 1. Mean latency per uniform update is well below 1.0 ms (typically < 0.05 ms)
      expect(meanLatencyMs).toBeLessThan(1.0);

      // 2. Critical GPU compilation invariant: needsUpdate was never set, material.version unchanged
      expect(needsUpdateSpy).not.toHaveBeenCalled();
      expect(material.version).toBe(initialVersion);

      // 3. Final values in GPU uniform memory buffer match store state
      const finalRound = totalRounds - 1;
      for (let i = 0; i < paramCount; i++) {
        const expectedVal = (finalRound + 1) * 10.0 + i;
        const uName = getParamUniformName(params[i]);
        expect(material.uniforms[uName].value).toBe(expectedVal);
        expect(store.getState().values[params[i].id]).toBe(expectedVal);
      }

      // 4. Clean unbind
      unbind();
    });
  });
});
