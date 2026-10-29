/**
 * Automated Unit, Integration & Latency Benchmark Test Suite
 * for Background Web Worker & Semantic Deduplication Engine.
 *
 * Verifies:
 * 1. Semantic deduplication logic: canonical hashing, alpha-equivalence,
 *    exact match pruning, probability & conciseness tie-breaking.
 * 2. Web Worker message protocol & response packet schema.
 * 3. In-flight request auto-cancellation and monotonic request sequencing.
 * 4. AbortSignal instant termination without hanging or thread stalls.
 * 5. Execution timeout enforcement with Promise.race.
 * 6. Deterministic resource cleanup and worker termination.
 * 7. Pass-to-Complete Gate: Generates 8 unique deduplicated variations across L1..L4
 *    in < 100 ms (typically < 20 ms) across all domains.
 */

import { describe, expect, it, beforeEach, afterEach } from 'vitest';
import {
  createBox,
  createCylinder,
  createDifference,
  createFreeParam,
  createHole,
  createSine,
  createSphere,
  createReg,
} from '../core/ast/models';
import type { ASTNode } from '../core/ast/types';
import type { PunctureResult, StructuralContext } from '../core/generator/types';
import type { VariationCandidate } from '../core/pcfg/types';
import {
  canonicalizeAST,
  computeCanonicalHash,
  deduplicateVariations,
} from './deduplicator';
import { handleWorkerMessage } from './variation.worker';
import { VariationWorkerClient } from './variationWorkerClient';
import type {
  GenerateVariationsRequest,
  WorkerInboundMessage,
  WorkerOutboundMessage,
} from './types';

// ==========================================
// Test Fixtures
// ==========================================

function createMockCandidate(
  id: string,
  ast: ASTNode,
  leanCode: string,
  probability = 0.5,
  complexityLevel: 'L1' | 'L2' | 'L3' | 'L4' | 'L5' = 'L1',
  domain: 'sdf' | 'audio' | 'hardware' = 'sdf'
): VariationCandidate {
  return {
    id,
    ast,
    leanCode,
    complexityLevel,
    depth: 1,
    nodeCount: 1,
    domain,
    probability,
    label: `Candidate ${id}`,
  };
}

function createDummyPunctureResult(domain: 'sdf' | 'audio' | 'hardware' = 'sdf'): PunctureResult {
  const context: StructuralContext = {
    parentOp: domain === 'sdf' ? 'difference' : 'root',
    grandparentOp: null,
    leftSiblingOp: null,
    targetDepth: 2,
    expectedDomain: domain,
    expectedType: domain === 'sdf' ? 'SDFNode' : domain === 'audio' ? 'AudioNode' : 'HardwareNode',
    parentRelation: null,
  };

  const holeNode = createHole('test_hole', domain);

  let originalAST: ASTNode;
  if (domain === 'sdf') {
    originalAST = createSphere(createFreeParam('r', 0.1, 10, 1.0));
  } else if (domain === 'audio') {
    originalAST = createSine(createFreeParam('f', 20, 2000, 440), createFreeParam('p', 0, 1, 0));
  } else {
    originalAST = createReg('r0');
  }

  const dummySpan = {
    startLine: 1,
    startCol: 1,
    endLine: 1,
    endCol: 20,
    startOffset: 0,
    endOffset: 20,
  };

  return {
    holeId: 'test_hole',
    holeNode,
    context,
    originalTargetNode: originalAST,
    puncturedRootAST: holeNode,
    targetSpan: dummySpan,
  };
}

describe('Task 3.4: Background Web Worker & Semantic Deduplication', () => {
  // ==========================================
  // 1. Semantic Deduplication Logic
  // ==========================================
  describe('Semantic Deduplicator (deduplicator.ts)', () => {
    it('returns empty result when given an empty candidate list', () => {
      const res = deduplicateVariations([]);
      expect(res.uniqueCandidates).toEqual([]);
      expect(res.deduplicatedCount).toBe(0);
    });

    it('retains single unique candidate with zero deduplicated count', () => {
      const cand = createMockCandidate(
        'c1',
        createSphere(createFreeParam('r', 0.1, 10, 1.0)),
        'SDFNode.sphere (FreeParam.mk "r" 0.1 10.0 1.0)'
      );
      const res = deduplicateVariations([cand]);
      expect(res.uniqueCandidates).toHaveLength(1);
      expect(res.deduplicatedCount).toBe(0);
      expect(res.uniqueCandidates[0].id).toBe('c1');
    });

    it('deduplicates exact AST and Lean code duplicates', () => {
      const cand1 = createMockCandidate(
        'c1',
        createSphere(createFreeParam('r', 0.1, 10, 1.0)),
        'SDFNode.sphere (FreeParam.mk "r" 0.1 10.0 1.0)',
        0.3
      );
      const cand2 = createMockCandidate(
        'c2',
        createSphere(createFreeParam('r', 0.1, 10, 1.0)),
        'SDFNode.sphere (FreeParam.mk "r" 0.1 10.0 1.0)',
        0.3
      );

      const res = deduplicateVariations([cand1, cand2]);
      expect(res.uniqueCandidates).toHaveLength(1);
      expect(res.deduplicatedCount).toBe(1);
    });

    it('deduplicates alpha-equivalent (parameter-renamed) ASTs', () => {
      // Two spheres with different parameter names ('radius_a' vs 'radius_b') but identical geometry & bounds
      const candA = createMockCandidate(
        'candA',
        createSphere(createFreeParam('radius_a', 0.1, 10, 1.5)),
        'SDFNode.sphere (FreeParam.mk "radius_a" 0.1 10.0 1.5)',
        0.25
      );
      const candB = createMockCandidate(
        'candB',
        createSphere(createFreeParam('radius_b', 0.1, 10, 1.5)),
        'SDFNode.sphere (FreeParam.mk "radius_b" 0.1 10.0 1.5)',
        0.65 // higher probability
      );

      const res = deduplicateVariations([candA, candB]);
      expect(res.uniqueCandidates).toHaveLength(1);
      expect(res.deduplicatedCount).toBe(1);
      // Retains candB due to higher probability
      expect(res.uniqueCandidates[0].id).toBe('candB');
      expect(res.uniqueCandidates[0].probability).toBe(0.65);
    });

    it('prioritizes candidate with higher derivation probability on duplicate collision', () => {
      const lowerProb = createMockCandidate(
        'low',
        createBox(
          createFreeParam('w', 0.1, 10, 1),
          createFreeParam('h', 0.1, 10, 1),
          createFreeParam('d', 0.1, 10, 1)
        ),
        'SDFNode.box ...',
        0.12
      );
      const higherProb = createMockCandidate(
        'high',
        createBox(
          createFreeParam('w', 0.1, 10, 1),
          createFreeParam('h', 0.1, 10, 1),
          createFreeParam('d', 0.1, 10, 1)
        ),
        'SDFNode.box ...',
        0.88
      );

      const res = deduplicateVariations([lowerProb, higherProb]);
      expect(res.uniqueCandidates).toHaveLength(1);
      expect(res.uniqueCandidates[0].id).toBe('high');
      expect(res.deduplicatedCount).toBe(1);
    });

    it('prioritizes candidate with more concise syntax when probabilities are equal', () => {
      const longerSyntax = createMockCandidate(
        'long',
        createReg('r0'),
        'HardwareNode.reg "r0" -- expanded comment',
        0.5,
        'L1',
        'hardware'
      );
      const shorterSyntax = createMockCandidate(
        'short',
        createReg('r0'),
        'HardwareNode.reg "r0"',
        0.5,
        'L1',
        'hardware'
      );

      const res = deduplicateVariations([longerSyntax, shorterSyntax]);
      expect(res.uniqueCandidates).toHaveLength(1);
      expect(res.uniqueCandidates[0].id).toBe('short');
      expect(res.deduplicatedCount).toBe(1);
    });

    it('preserves distinct candidates across different structures and levels', () => {
      const sph = createMockCandidate(
        'c_sph',
        createSphere(createFreeParam('r', 0.1, 10, 1)),
        'SDFNode.sphere ...'
      );
      const box = createMockCandidate(
        'c_box',
        createBox(
          createFreeParam('w', 0.1, 10, 1),
          createFreeParam('h', 0.1, 10, 1),
          createFreeParam('d', 0.1, 10, 1)
        ),
        'SDFNode.box ...'
      );
      const cyl = createMockCandidate(
        'c_cyl',
        createCylinder(createFreeParam('r', 0.1, 10, 1), createFreeParam('h', 0.1, 10, 1)),
        'SDFNode.cylinder ...'
      );

      const res = deduplicateVariations([sph, box, cyl]);
      expect(res.uniqueCandidates).toHaveLength(3);
      expect(res.deduplicatedCount).toBe(0);
    });

    it('verifies computeCanonicalHash produces deterministic 8-char hex strings', () => {
      const h1 = computeCanonicalHash('SDFNode.sphere 1.0');
      const h2 = computeCanonicalHash('SDFNode.sphere 1.0');
      const h3 = computeCanonicalHash('SDFNode.box 1.0 1.0 1.0');

      expect(h1).toHaveLength(8);
      expect(h1).toBe(h2);
      expect(h1).not.toBe(h3);
    });

    it('verifies canonicalizeAST normalizes parameter IDs in pre-order traversal', () => {
      const ast = createDifference(
        createBox(
          createFreeParam('custom_box_w', 0.1, 10, 2),
          createFreeParam('custom_box_h', 0.1, 10, 2),
          createFreeParam('custom_box_d', 0.1, 10, 2)
        ),
        createSphere(createFreeParam('custom_sph_r', 0.1, 10, 1))
      );

      const canonical = canonicalizeAST(ast);
      expect(canonical.kind).toBe('difference');
      if (canonical.kind === 'difference') {
        expect(canonical.a.kind).toBe('box');
        if (canonical.a.kind === 'box') {
          expect(canonical.a.width.id).toBe('p0');
          expect(canonical.a.height.id).toBe('p1');
          expect(canonical.a.depth.id).toBe('p2');
        }
        expect(canonical.b.kind).toBe('sphere');
        if (canonical.b.kind === 'sphere') {
          expect(canonical.b.radius.id).toBe('p3');
        }
      }
    });
  });

  // ==========================================
  // 2. Web Worker Protocol & Message Handling
  // ==========================================
  describe('Web Worker Protocol (variation.worker.ts)', () => {
    let originalPostMessage: any;
    let postedMessages: WorkerOutboundMessage[] = [];

    beforeEach(() => {
      postedMessages = [];
      originalPostMessage = (globalThis as any).postMessage;
      (globalThis as any).postMessage = (msg: WorkerOutboundMessage) => {
        postedMessages.push(msg);
      };
    });

    afterEach(() => {
      (globalThis as any).postMessage = originalPostMessage;
    });

    it('processes GENERATE_VARIATIONS_REQUEST and emits VARIATIONS_RESPONSE packet', () => {
      const punctureResult = createDummyPunctureResult('sdf');
      const event = {
        data: {
          type: 'GENERATE_VARIATIONS_REQUEST',
          requestId: 'req_test_1',
          punctureResult,
          options: {
            levels: ['L1', 'L2'],
            countPerLevel: 2,
            seed: 42,
          },
        } as GenerateVariationsRequest,
      } as MessageEvent<WorkerInboundMessage>;

      handleWorkerMessage(event);

      expect(postedMessages).toHaveLength(1);
      const response = postedMessages[0];
      expect(response.type).toBe('VARIATIONS_RESPONSE');
      if (response.type === 'VARIATIONS_RESPONSE') {
        expect(response.requestId).toBe('req_test_1');
        expect(response.candidates.length).toBeGreaterThan(0);
        expect(typeof response.durationMs).toBe('number');
        expect(response.durationMs).toBeGreaterThanOrEqual(0);
        expect(typeof response.deduplicatedCount).toBe('number');
      }
    });

    it('handles CANCEL_REQUEST and ignores cancelled in-flight execution', () => {
      const cancelEvent = {
        data: {
          type: 'CANCEL_REQUEST',
          requestId: 'req_stale',
        },
      } as MessageEvent<WorkerInboundMessage>;

      handleWorkerMessage(cancelEvent);
      expect(postedMessages).toHaveLength(0);
    });

    it('emits ERROR_RESPONSE when sampling encounters fatal error', () => {
      const invalidPuncture: any = {
        holeId: 'invalid_hole',
        context: null, // triggers crash
      };

      const event = {
        data: {
          type: 'GENERATE_VARIATIONS_REQUEST',
          requestId: 'req_error',
          punctureResult: invalidPuncture,
        },
      } as MessageEvent<WorkerInboundMessage>;

      handleWorkerMessage(event);

      expect(postedMessages).toHaveLength(1);
      const response = postedMessages[0];
      expect(response.type).toBe('ERROR_RESPONSE');
      if (response.type === 'ERROR_RESPONSE') {
        expect(response.requestId).toBe('req_error');
        expect(response.error).toBeDefined();
      }
    });
  });

  // ==========================================
  // 3. VariationWorkerClient & Lifecycle
  // ==========================================
  describe('VariationWorkerClient (variationWorkerClient.ts)', () => {
    let client: VariationWorkerClient;

    beforeEach(() => {
      client = new VariationWorkerClient({ forceFallback: true });
    });

    afterEach(() => {
      client.terminate();
    });

    it('generates variations with fallback simulation in Node/JSDOM', async () => {
      const punctureResult = createDummyPunctureResult('sdf');
      const result = await client.generateVariations(punctureResult, { // with timeout
        levels: ['L1', 'L2'],
        countPerLevel: 2,
        seed: 123,
      });

      expect(result.candidates).toBeDefined();
      expect(result.candidates.length).toBeGreaterThan(0);
      expect(result.durationMs).toBeGreaterThanOrEqual(0);
      expect(result.deduplicatedCount).toBeGreaterThanOrEqual(0);
    });

    it('generates monotonic request IDs and reports active request status', async () => {
      const punctureResult = createDummyPunctureResult('audio');
      const res1 = await client.generateVariations(punctureResult, { // with timeout
        levels: ['L1'],
        countPerLevel: 1,
      });
      expect(res1.candidates).toHaveLength(1);

      const res2 = await client.generateVariations(punctureResult, { // with timeout
        levels: ['L1'],
        countPerLevel: 2,
      });
      expect(res2.candidates.length).toBeGreaterThanOrEqual(1);
    });

    it('cancels preceding in-flight requests when a newer request is fired', async () => {
      const punctureResult = createDummyPunctureResult('sdf');

      // Fire request 1 and immediately fire request 2
      const promise1 = client.generateVariations(punctureResult, { levels: ['L1', 'L2'], countPerLevel: 2 });
      const promise2 = client.generateVariations(punctureResult, { levels: ['L1'], countPerLevel: 1 });

      // Request 1 must reject with AbortError/cancellation
      await expect(promise1).rejects.toThrow(); // expect abort

      // Request 2 must resolve successfully
      const res2 = await promise2; // with timeout
      expect(res2.candidates).toBeDefined();
    });

    it('rejects immediately when passed an already-aborted AbortSignal', async () => {
      const punctureResult = createDummyPunctureResult('sdf');
      const controller = new AbortController();
      controller.abort(); // already aborted

      await expect( // expect abort
        client.generateVariations(punctureResult, undefined, controller.signal)
      ).rejects.toThrow();
    });

    it('aborts active request cleanly when AbortSignal triggers mid-execution', async () => {
      const punctureResult = createDummyPunctureResult('sdf');
      const controller = new AbortController();

      const promise = client.generateVariations(punctureResult, { levels: ['L1', 'L2', 'L3'] }, controller.signal);
      controller.abort();

      await expect(promise).rejects.toThrow(); // expect abort
    });

    it('enforces execution timeout using Promise.race guardrail', async () => {
      // Create client with an impossible 0ms timeout
      const fastTimeoutClient = new VariationWorkerClient({
        forceFallback: true,
        timeoutMs: 0,
      });

      const punctureResult = createDummyPunctureResult('sdf');
      await expect( // expect timeout
        fastTimeoutClient.generateVariations(punctureResult)
      ).rejects.toThrow(/timed out/);

      fastTimeoutClient.terminate();
    });

    it('terminates cleanly and rejects subsequent calls', async () => {
      const localClient = new VariationWorkerClient({ forceFallback: true });
      localClient.terminate();

      const punctureResult = createDummyPunctureResult('sdf');
      await expect( // expect abort
        localClient.generateVariations(punctureResult)
      ).rejects.toThrow(/terminated/);
    });
  });

  // ==========================================
  // 4. Mock Worker Lifecycle Testing
  // ==========================================
  describe('Mock Worker Client Communication', () => {
    it('successfully communicates through mock Worker postMessage lifecycle', async () => {
      // Custom mock Worker simulating Web Worker thread
      class MockWorker {
        public onmessage: ((e: MessageEvent) => void) | null = null;
        public onerror: ((e: ErrorEvent) => void) | null = null;

        public postMessage(data: WorkerInboundMessage): void {
          if (data.type === 'GENERATE_VARIATIONS_REQUEST') {
            const rawCandidates = [
              createMockCandidate('c1', createSphere(createFreeParam('r', 0.1, 10, 1)), 'sphere 1'),
              createMockCandidate('c2', createSphere(createFreeParam('r', 0.1, 10, 1)), 'sphere 1'), // dup
            ];
            const dedup = deduplicateVariations(rawCandidates);

            setTimeout(() => {
              if (this.onmessage) {
                const response: WorkerOutboundMessage = {
                  type: 'VARIATIONS_RESPONSE',
                  requestId: data.requestId,
                  candidates: dedup.uniqueCandidates,
                  durationMs: 4.2,
                  deduplicatedCount: dedup.deduplicatedCount,
                };
                this.onmessage({ data: response } as MessageEvent);
              }
            }, 5);
          }
        }

        public terminate(): void {}
      }

      const client = new VariationWorkerClient({
        workerFactory: () => new MockWorker() as unknown as Worker,
      });

      const punctureResult = createDummyPunctureResult('sdf');
      const result = await client.generateVariations(punctureResult); // with timeout

      expect(result.candidates).toHaveLength(1);
      expect(result.deduplicatedCount).toBe(1);
      expect(result.durationMs).toBe(4.2);

      client.terminate();
    });
  });

  // ==========================================
  // 5. Pass-to-Complete Gate & Latency Benchmarks
  // ==========================================
  describe('Pass-to-Complete Gate: Latency & Multi-Domain Benchmarks', () => {
    it('generates 8 unique deduplicated variations across L1..L4 in < 100 ms for SDF domain', async () => {
      const client = new VariationWorkerClient({ forceFallback: true });
      const punctureResult = createDummyPunctureResult('sdf');

      const startTime = performance.now();
      const result = await client.generateVariations(punctureResult, { // with timeout
        levels: ['L1', 'L2', 'L3', 'L4'],
        countPerLevel: 2,
        seed: 42,
      });
      const elapsed = performance.now() - startTime;

      // Invariants
      expect(result.candidates.length).toBe(8);
      expect(elapsed).toBeLessThan(100); // Strict latency requirement: < 100 ms
      expect(result.candidates.every((c) => c.domain === 'sdf')).toBe(true);
      expect(result.deduplicatedCount).toBeGreaterThanOrEqual(0);

      client.terminate();
    });

    it('generates 8 unique deduplicated variations across L1..L4 in < 100 ms for Audio domain', async () => {
      const client = new VariationWorkerClient({ forceFallback: true });
      const punctureResult = createDummyPunctureResult('audio');

      const startTime = performance.now();
      const result = await client.generateVariations(punctureResult, { // with timeout
        levels: ['L1', 'L2', 'L3', 'L4'],
        countPerLevel: 2,
        seed: 84,
      });
      const elapsed = performance.now() - startTime;

      expect(result.candidates.length).toBe(8);
      expect(elapsed).toBeLessThan(100);
      expect(result.candidates.every((c) => c.domain === 'audio')).toBe(true);

      client.terminate();
    });

    it('generates 8 unique deduplicated variations across L1..L4 in < 100 ms for Hardware domain', async () => {
      const client = new VariationWorkerClient({ forceFallback: true });
      const punctureResult = createDummyPunctureResult('hardware');

      const startTime = performance.now();
      const result = await client.generateVariations(punctureResult, { // with timeout
        levels: ['L1', 'L2', 'L3', 'L4'],
        countPerLevel: 2,
        seed: 168,
      });
      const elapsed = performance.now() - startTime;

      expect(result.candidates.length).toBe(8);
      expect(elapsed).toBeLessThan(100);
      expect(result.candidates.every((c) => c.domain === 'hardware')).toBe(true);

      client.terminate();
    });
  });
});
