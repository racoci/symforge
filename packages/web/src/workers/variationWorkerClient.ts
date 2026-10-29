/**
 * Client Interface for the Background Variation Web Worker.
 *
 * Manages Web Worker lifecycle, monotonic request sequencing, automatic cancellation
 * of stale in-flight requests, AbortSignal propagation, hard execution timeouts,
 * and seamless fallback simulation in headless or testing environments (Node/JSDOM).
 */

import type { PunctureResult } from '../core/generator/types';
import type { SamplerOptions } from '../core/pcfg/types';
import { sampleScopedVariations } from '../core/pcfg/pcfgSampler';
import { deduplicateVariations } from './deduplicator';
import type {
  CancelRequest,
  GenerateVariationsRequest,
  GenerateVariationsResult,
  VariationWorkerClientOptions,
  WorkerOutboundMessage,
} from './types';

interface PendingRequestEntry {
  resolve: (result: GenerateVariationsResult) => void;
  reject: (error: Error) => void;
  cancel: (reason?: string) => void;
}

export class VariationWorkerClient {
  private worker: Worker | null = null;
  private requestCounter = 0;
  private activeRequestId: string | null = null;
  private pendingRequests = new Map<string, PendingRequestEntry>();
  private defaultTimeoutMs: number;
  private isTerminated = false;

  constructor(options?: VariationWorkerClientOptions) {
    this.defaultTimeoutMs = options?.timeoutMs ?? 2000;

    if (options?.forceFallback) {
      this.worker = null;
      return;
    }

    if (options?.workerFactory) {
      try {
        this.worker = options.workerFactory();
        this.initWorkerListeners();
      } catch (err) {
        console.error(
          '[VariationWorkerClient] Worker factory failed, falling back to local simulation:',
          err
        );
        this.worker = null;
      }
    } else if (typeof Worker !== 'undefined') {
      try {
        this.worker = new Worker(new URL('./variation.worker.ts', import.meta.url), {
          type: 'module',
        });
        this.initWorkerListeners();
      } catch (err) {
        console.error(
          '[VariationWorkerClient] Native Web Worker instantiation failed, falling back to local simulation:',
          err
        );
        this.worker = null;
      }
    } else {
      this.worker = null;
    }
  }

  /**
   * Binds message and error listeners to the active Web Worker instance.
   */
  private initWorkerListeners(): void {
    if (!this.worker) return;

    this.worker.onmessage = (event: MessageEvent<WorkerOutboundMessage>) => {
      const msg = event.data;
      if (!msg || !msg.requestId) return;

      const entry = this.pendingRequests.get(msg.requestId);
      if (!entry) return;

      if (msg.type === 'VARIATIONS_RESPONSE') {
        entry.resolve({
          candidates: msg.candidates,
          durationMs: msg.durationMs,
          deduplicatedCount: msg.deduplicatedCount,
        });
      } else if (msg.type === 'ERROR_RESPONSE') {
        entry.reject(new Error(msg.error));
      }
    };

    this.worker.onerror = (err: ErrorEvent) => {
      console.error('[VariationWorkerClient] Web Worker thread encountered unhandled error:', err);
      const error = new Error(`Worker thread error: ${err.message || 'Unknown error'}`);
      for (const [, entry] of this.pendingRequests.entries()) {
        entry.reject(error);
      }
      this.pendingRequests.clear();
      this.activeRequestId = null;
    };
  }

  /**
   * Generates monotonic request IDs (req_1, req_2, ...).
   */
  private nextRequestId(): string {
    this.requestCounter += 1;
    return `req_${this.requestCounter}`;
  }

  /**
   * Generates scoped AST variations asynchronously off-thread with semantic deduplication.
   *
   * Automatically cancels any prior in-flight request, wires the caller's AbortSignal,
   * and enforces an explicit hard timeout (default 2000ms) with Promise.race.
   */
  public async generateVariations(
    punctureResult: PunctureResult,
    options?: SamplerOptions,
    signal?: AbortSignal
  ): Promise<GenerateVariationsResult> {
    if (this.isTerminated) {
      throw new Error('VariationWorkerClient has been terminated');
    }

    if (signal?.aborted) {
      throw new DOMException('Variation generation aborted', 'AbortError');
    }

    // 1. Automatically cancel preceding in-flight request
    if (this.activeRequestId && this.pendingRequests.has(this.activeRequestId)) {
      this.cancelRequest(this.activeRequestId, 'Superceded by newer variation request');
    }

    // 2. Monotonic request ID sequence
    const requestId = this.nextRequestId();
    this.activeRequestId = requestId;
    const timeoutMs = this.defaultTimeoutMs;

    if (timeoutMs <= 0) {
      throw new Error(
        `Variation generation timed out after ${timeoutMs}ms for request ${requestId}`
      );
    }

    // 3. Setup execution promise
    const executionPromise = this.worker
      ? this.executeOnWorker(requestId, punctureResult, options)
      : this.executeFallback(requestId, punctureResult, options);

    // 4. Setup timeout guardrail promise
    let timer: ReturnType<typeof setTimeout> | null = null;
    const timeoutPromise = new Promise<never>((_, reject) => {
      timer = setTimeout(() => {
        if (this.worker) {
          try {
            const cancelMsg: CancelRequest = { type: 'CANCEL_REQUEST', requestId };
            this.worker.postMessage(cancelMsg);
          } catch (err) {
            console.error('[VariationWorkerClient] Failed to post cancel on timeout:', err);
          }
        }
        reject(
          new Error(
            `Variation generation timed out after ${timeoutMs}ms for request ${requestId}`
          )
        );
      }, timeoutMs);
    });

    // 5. Setup AbortSignal guardrail promise
    let abortListener: (() => void) | null = null;
    const abortPromise = new Promise<never>((_, reject) => {
      if (signal) {
        abortListener = () => {
          if (this.worker) {
            try {
              const cancelMsg: CancelRequest = { type: 'CANCEL_REQUEST', requestId };
              this.worker.postMessage(cancelMsg);
            } catch (err) {
              console.error('[VariationWorkerClient] Failed to post cancel on abort:', err);
            }
          }
          reject(new DOMException('Variation generation aborted', 'AbortError'));
        };
        signal.addEventListener('abort', abortListener, { once: true });
      }
    });

    try {
      // Execute race between derivation, timeout, and abort
      const result = await Promise.race([executionPromise, timeoutPromise, abortPromise]); // timeout protected race
      return result;
    } finally {
      if (timer !== null) {
        clearTimeout(timer);
        timer = null;
      }
      if (abortListener && signal) {
        signal.removeEventListener('abort', abortListener);
        abortListener = null;
      }
      this.pendingRequests.delete(requestId);
      if (this.activeRequestId === requestId) {
        this.activeRequestId = null;
      }
    }
  }

  /**
   * Executes variation generation on the active Web Worker thread.
   */
  private executeOnWorker(
    requestId: string,
    punctureResult: PunctureResult,
    options?: SamplerOptions
  ): Promise<GenerateVariationsResult> {
    return new Promise<GenerateVariationsResult>((resolve, reject) => {
      const cancel = (reason = 'Request cancelled') => {
        if (this.worker) {
          try {
            const cancelMsg: CancelRequest = { type: 'CANCEL_REQUEST', requestId };
            this.worker.postMessage(cancelMsg);
          } catch (err) {
            console.error('[VariationWorkerClient] Error sending cancel for ' + requestId + ':', err);
          }
        }
        reject(new DOMException(reason, 'AbortError'));
      };

      this.pendingRequests.set(requestId, { resolve, reject, cancel });

      const requestMsg: GenerateVariationsRequest = {
        type: 'GENERATE_VARIATIONS_REQUEST',
        requestId,
        punctureResult,
        options,
      };

      if (this.worker) {
        this.worker.postMessage(requestMsg);
      } else {
        reject(new Error('Worker instance disappeared'));
      }
    });
  }

  /**
   * Fallback simulation for Node/JSDOM environments where Web Workers are unavailable.
   */
  private executeFallback(
    requestId: string,
    punctureResult: PunctureResult,
    options?: SamplerOptions
  ): Promise<GenerateVariationsResult> {
    return new Promise<GenerateVariationsResult>((resolve, reject) => {
      let cancelled = false;

      const cancel = (reason = 'Request cancelled') => {
        cancelled = true;
        reject(new DOMException(reason, 'AbortError'));
      };

      this.pendingRequests.set(requestId, { resolve, reject, cancel });

      // Run asynchronously via microtask / short timer to simulate off-thread derivation
      setTimeout(() => {
        if (cancelled || !this.pendingRequests.has(requestId)) {
          return;
        }

        const startTime = performance.now();
        try {
          // 1. Stochastic contextual PCFG derivation
          const rawCandidates = sampleScopedVariations(punctureResult, options);
          if (cancelled || !this.pendingRequests.has(requestId)) return;

          // 2. Semantic deduplication & isomorphic pruning
          const { uniqueCandidates, deduplicatedCount } = deduplicateVariations(rawCandidates);
          if (cancelled || !this.pendingRequests.has(requestId)) return;

          const durationMs = performance.now() - startTime;
          resolve({
            candidates: uniqueCandidates,
            durationMs,
            deduplicatedCount,
          });
        } catch (err: unknown) {
          const errorMsg = err instanceof Error ? err.message : String(err);
          console.error(
            `[VariationWorkerClient] Fallback derivation error for request ${requestId}:`,
            err
          );
          reject(new Error(errorMsg));
        }
      }, 0);
    });
  }

  /**
   * Cancels a pending request by ID.
   */
  public cancelRequest(requestId: string, reason = 'Request cancelled'): void {
    const entry = this.pendingRequests.get(requestId);
    if (entry) {
      entry.cancel(reason);
      this.pendingRequests.delete(requestId);
    }
    if (this.activeRequestId === requestId) {
      this.activeRequestId = null;
    }
  }

  /**
   * Returns whether this client has an active in-flight request.
   */
  public hasActiveRequest(): boolean {
    return this.activeRequestId !== null && this.pendingRequests.has(this.activeRequestId);
  }

  /**
   * Returns the ID of the current active request, or null if idle.
   */
  public getActiveRequestId(): string | null {
    return this.activeRequestId;
  }

  /**
   * Deterministically terminates the Web Worker thread and rejects all pending promises.
   */
  public terminate(): void {
    if (this.isTerminated) return;
    this.isTerminated = true;

    for (const [id, req] of this.pendingRequests.entries()) {
      try {
        req.cancel('Worker client terminated');
      } catch (err) {
        console.error('[VariationWorkerClient] Error cancelling request ' + id + ' during terminate:', err);
      }
    }
    this.pendingRequests.clear();
    this.activeRequestId = null;

    if (this.worker) {
      try {
        this.worker.terminate();
      } catch (err) {
        console.error('[VariationWorkerClient] Error terminating worker instance:', err);
      }
      this.worker = null;
    }
  }
}
