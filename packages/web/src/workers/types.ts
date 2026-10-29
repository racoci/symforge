/**
 * Background Web Worker - Type Definitions & Message Protocol.
 *
 * Defines the bidirectional postMessage protocol, request/response payloads,
 * candidate variation structures, and configuration options for off-thread
 * stochastic PCFG sampling and semantic deduplication.
 */

import type { PunctureResult } from '../core/generator/types';
import type { SamplerOptions, VariationCandidate } from '../core/pcfg/types';

/**
 * Message from Main Thread -> Web Worker: Request variation generation.
 */
export interface GenerateVariationsRequest {
  type: 'GENERATE_VARIATIONS_REQUEST';
  requestId: string;
  punctureResult: PunctureResult;
  options?: SamplerOptions;
}

/**
 * Message from Main Thread -> Web Worker: Cancel an active in-flight request.
 */
export interface CancelRequest {
  type: 'CANCEL_REQUEST';
  requestId: string;
}

/**
 * Union of inbound message types processed by the Web Worker.
 */
export type WorkerInboundMessage = GenerateVariationsRequest | CancelRequest;

/**
 * Message from Web Worker -> Main Thread: Successful variation generation & deduplication.
 */
export interface VariationsResponse {
  type: 'VARIATIONS_RESPONSE';
  requestId: string;
  candidates: VariationCandidate[];
  durationMs: number;
  deduplicatedCount: number;
}

/**
 * Message from Web Worker -> Main Thread: Fatal error during generation.
 */
export interface ErrorResponse {
  type: 'ERROR_RESPONSE';
  requestId: string;
  error: string;
}

/**
 * Union of outbound message types emitted by the Web Worker.
 */
export type WorkerOutboundMessage = VariationsResponse | ErrorResponse;

/**
 * Result returned to caller from `VariationWorkerClient.generateVariations`.
 */
export interface GenerateVariationsResult {
  candidates: VariationCandidate[];
  durationMs: number;
  deduplicatedCount: number;
}

/**
 * Result returned by the semantic deduplication engine.
 */
export interface DeduplicationResult {
  uniqueCandidates: VariationCandidate[];
  deduplicatedCount: number;
}

/**
 * Options for configuring the VariationWorkerClient.
 */
export interface VariationWorkerClientOptions {
  timeoutMs?: number;
  workerFactory?: () => Worker;
  forceFallback?: boolean;
}
