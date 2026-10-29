/**
 * Background Web Worker for Non-Blocking Variation Generation & Semantic Deduplication.
 *
 * Runs stochastic PCFG derivation and AST semantic deduplication off the main thread.
 * Listens to incoming typed messages, tracks active request IDs, supports instant
 * cancellation of stale requests, and emits structured response packets.
 */

import { sampleScopedVariations } from '../core/pcfg/pcfgSampler';
import { deduplicateVariations } from './deduplicator';
import type {
  ErrorResponse,
  VariationsResponse,
  WorkerInboundMessage,
  WorkerOutboundMessage,
} from './types';

// Track the currently active in-flight request ID
let currentRequestId: string | null = null;

function postWorkerMessage(msg: WorkerOutboundMessage): void {
  if (typeof self !== 'undefined' && typeof (self as any).postMessage === 'function') {
    (self as unknown as { postMessage: (msg: WorkerOutboundMessage) => void }).postMessage(msg);
  }
}

/**
 * Message handler executing variation derivation, cancellation, and deduplication.
 */
export function handleWorkerMessage(event: MessageEvent<WorkerInboundMessage>): void {
  const data = event.data;
  if (!data || !data.type) return;

  if (data.type === 'CANCEL_REQUEST') {
    if (currentRequestId === data.requestId) {
      currentRequestId = null;
    }
    return;
  }

  if (data.type === 'GENERATE_VARIATIONS_REQUEST') {
    const { requestId, punctureResult, options } = data;
    // Set this request as the active in-flight request
    currentRequestId = requestId;
    const startTime = performance.now();

    try {
      // 1. Stochastic contextual PCFG derivation
      const rawCandidates = sampleScopedVariations(punctureResult, options);

      // Check for cancellation or newer request superseding this one
      if (currentRequestId !== requestId) {
        return;
      }

      // 2. Semantic deduplication & isomorphic pruning
      const { uniqueCandidates, deduplicatedCount } = deduplicateVariations(rawCandidates);

      // Final cancellation check before posting
      if (currentRequestId !== requestId) {
        return;
      }

      const durationMs = performance.now() - startTime;
      const response: VariationsResponse = {
        type: 'VARIATIONS_RESPONSE',
        requestId,
        candidates: uniqueCandidates,
        durationMs,
        deduplicatedCount,
      };

      postWorkerMessage(response);
    } catch (err: unknown) {
      if (currentRequestId !== requestId) {
        return;
      }
      const errorMsg = err instanceof Error ? err.message : String(err);
      console.error(
        `[VariationWorker] Error processing request ${requestId} for hole ${punctureResult?.holeId}:`,
        err
      );
      const response: ErrorResponse = {
        type: 'ERROR_RESPONSE',
        requestId,
        error: errorMsg,
      };
      postWorkerMessage(response);
    } finally {
      if (currentRequestId === requestId) {
        currentRequestId = null;
      }
    }
  }
}

// Bind event listener if in WorkerGlobalScope
if (typeof self !== 'undefined' && typeof (self as any).addEventListener === 'function') {
  self.addEventListener('message', handleWorkerMessage as EventListener);
}
