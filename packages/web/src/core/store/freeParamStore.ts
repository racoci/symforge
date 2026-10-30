/**
 * FreeParam Reactive State Store for SymForge Web IDE.
 *
 * Provides a high-performance Zustand reactive state store that extracts
 * continuous FreeParam variables from the active AST, coordinates continuous
 * slider/numeric adjustments, executes parameter clamping, and enables
 * microsecond-latency parameter change subscriptions for WebGL/WebAudio viewports.
 */

import { create, type StoreApi, type UseBoundStore } from 'zustand';
import type { ASTNode, FreeParam } from '../ast/types';
import {
  collectFreeParams,
  cloneWithParams,
  isAudioNode,
  isSDFNode,
  isHardwareNode,
} from '../ast/models';

export type ParamSubscriber = (val: number) => void;

export interface FreeParamState {
  /** The root canonical AST node currently active in the IDE */
  readonly ast: ASTNode | null;
  /** Active Lean 4 declaration identifier (e.g., 'myGyroidSDF') */
  readonly declarationName: string;
  /** Symbolic domain type string (e.g., 'SDFNode', 'AudioNode', 'HardwareNode') */
  readonly domainType: string;
  /** Map of FreeParam definitions indexed by parameter ID */
  readonly params: Record<string, FreeParam>;
  /** Ordered list of parameter IDs preserving AST traversal order */
  readonly paramOrder: string[];
  /** Active numeric values for each parameter */
  readonly values: Record<string, number>;

  /**
   * Updates the active AST, re-extracting all FreeParams.
   * Preserves existing user-tweaked values when IDs match, otherwise initializes with defaultVal.
   */
  setAST: (ast: ASTNode | null, declarationName?: string, domainType?: string) => void;

  /**
   * Synchronously updates and clamps a parameter value to [minVal, maxVal].
   * Notifies registered subscribers in < 1 ms.
   */
  updateParamValue: (id: string, value: number) => void;

  /**
   * Reverts a specific parameter to its defaultVal.
   */
  resetParam: (id: string) => void;

  /**
   * Reverts all active parameter values to their respective defaultVal.
   */
  resetAllParams: () => void;

  /**
   * Returns a clone of the AST with user-modified parameter values applied.
   * Leaves the root canonical AST reference unmutated.
   */
  getUpdatedAST: () => ASTNode | null;

  /**
   * Subscribes a callback to receive real-time updates for a specific parameter ID
   * without requiring React tree re-renders.
   * Returns an unsubscribe function.
   */
  subscribeParam: (id: string, callback: ParamSubscriber) => () => void;
}

export type FreeParamStore = UseBoundStore<StoreApi<FreeParamState>>;

/**
 * Infer the domain category from AST node properties.
 */
function inferDomainType(ast: ASTNode): string {
  if (isAudioNode(ast)) return 'AudioNode';
  if (isSDFNode(ast)) return 'SDFNode';
  if (isHardwareNode(ast)) return 'HardwareNode';
  return 'UniversalNode';
}

/**
 * Creates an isolated instance of the FreeParam Reactive State Store.
 */
export function createFreeParamStore(initialAst: ASTNode | null = null): FreeParamStore {
  const subscribers = new Map<string, Set<ParamSubscriber>>();

  const notifySubscribers = (id: string, val: number) => {
    const subs = subscribers.get(id);
    if (!subs) return;
    for (const callback of subs) {
      try {
        callback(val);
      } catch (err) {
        console.error('[FreeParamStore] Error in subscriber for parameter ' + id, {
          paramId: id,
          paramValue: val,
          error: err,
          timestamp: Date.now(),
        });
      }
    }
  };

  return create<FreeParamState>((set, get) => {
    const initialParams: Record<string, FreeParam> = {};
    const initialParamOrder: string[] = [];
    const initialValues: Record<string, number> = {};

    if (initialAst) {
      const extracted = collectFreeParams(initialAst);
      for (const p of extracted) {
        initialParams[p.id] = p;
        initialParamOrder.push(p.id);
        initialValues[p.id] = p.defaultVal;
      }
    }

    return {
      ast: initialAst,
      declarationName: '',
      domainType: initialAst ? inferDomainType(initialAst) : '',
      params: initialParams,
      paramOrder: initialParamOrder,
      values: initialValues,

      setAST: (ast: ASTNode | null, declarationName?: string, domainType?: string) => {
        if (!ast) {
          set({
            ast: null,
            declarationName: declarationName ?? '',
            domainType: domainType ?? '',
            params: {},
            paramOrder: [],
            values: {},
          });
          return;
        }

        const extracted = collectFreeParams(ast);
        const currentValues = get().values;
        const nextParams: Record<string, FreeParam> = {};
        const nextParamOrder: string[] = [];
        const nextValues: Record<string, number> = {};

        for (const p of extracted) {
          nextParams[p.id] = p;
          nextParamOrder.push(p.id);

          if (Object.prototype.hasOwnProperty.call(currentValues, p.id) && typeof currentValues[p.id] === 'number') {
            const min = Math.min(p.minVal, p.maxVal);
            const max = Math.max(p.minVal, p.maxVal);
            nextValues[p.id] = Math.min(Math.max(currentValues[p.id], min), max);
          } else {
            nextValues[p.id] = p.defaultVal;
          }
        }

        const resolvedDeclarationName =
          declarationName !== undefined ? declarationName : (get().declarationName || 'anonymous');
        const resolvedDomainType =
          domainType !== undefined ? domainType : (get().domainType || inferDomainType(ast));

        set({
          ast,
          declarationName: resolvedDeclarationName,
          domainType: resolvedDomainType,
          params: nextParams,
          paramOrder: nextParamOrder,
          values: nextValues,
        });
      },

      updateParamValue: (id: string, value: number) => {
        if (Number.isNaN(value) || !Number.isFinite(value)) return;
        const param = get().params[id];
        if (!param) return;

        const min = Math.min(param.minVal, param.maxVal);
        const max = Math.max(param.minVal, param.maxVal);
        const clamped = Math.min(Math.max(value, min), max);

        set((state) => ({
          values: {
            ...state.values,
            [id]: clamped,
          },
        }));

        notifySubscribers(id, clamped);
      },

      resetParam: (id: string) => {
        const param = get().params[id];
        if (!param) return;
        const defaultVal = param.defaultVal;

        set((state) => ({
          values: {
            ...state.values,
            [id]: defaultVal,
          },
        }));

        notifySubscribers(id, defaultVal);
      },

      resetAllParams: () => {
        const { params } = get();
        const nextValues: Record<string, number> = {};

        for (const [id, param] of Object.entries(params)) {
          nextValues[id] = param.defaultVal;
        }

        set({ values: nextValues });

        for (const [id, val] of Object.entries(nextValues)) {
          notifySubscribers(id, val);
        }
      },

      getUpdatedAST: (): ASTNode | null => {
        const { ast, values } = get();
        if (!ast) return null;
        return cloneWithParams(ast, values);
      },

      subscribeParam: (id: string, callback: ParamSubscriber): (() => void) => {
        if (!subscribers.has(id)) {
          subscribers.set(id, new Set());
        }
        const setOfSubs = subscribers.get(id)!;
        setOfSubs.add(callback);

        return () => {
          const s = subscribers.get(id);
          if (s) {
            s.delete(callback);
            if (s.size === 0) {
              subscribers.delete(id);
            }
          }
        };
      },
    };
  });
}

/**
 * Singleton reactive store for active Lean 4 IDE FreeParam state.
 */
export const useFreeParamStore = createFreeParamStore();

if (typeof window !== 'undefined') {
  (window as any).__freeParamStore = useFreeParamStore;
}
