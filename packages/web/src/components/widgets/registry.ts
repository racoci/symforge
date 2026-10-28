import type { ASTNode } from '../../core/ast/types';
import {
  isSDFNode,
  isAudioNode,
  isHardwareNode,
  isUniversalNode,
} from '../../core/ast/models';
import type {
  DomainWidgetDefinition,
  WidgetRegistryInterface,
} from './types';
import { SDFRaymarchWidget } from './sdf/SDFRaymarchWidget';
import { AudioDSPWidget } from './audio';
import { HyperdimPlotterWidget } from './hyperdim';
import { HardwareTimelineWidget } from './DefaultDomainWidgets';
import { GenericASTWidget } from './GenericASTWidget';

type WidgetChangeListener = (widgets: DomainWidgetDefinition[]) => void;

/**
 * Creates canonical default widget definitions for SymForge.
 */
export function createDefaultWidgetDefinitions(): DomainWidgetDefinition[] {
  return [
    {
      id: 'sdf-raymarch',
      name: '3D WebGL SDF Raymarching',
      domain: 'sdf',
      description: '3D WebGL raymarching canvas for Signed Distance Fields with Phong illumination.',
      canHandle: (ast: ASTNode): boolean => isSDFNode(ast),
      component: SDFRaymarchWidget,
      priority: 100,
    },
    {
      id: 'audio-dsp',
      name: 'WebAudio DSP Synthesis & FFT',
      domain: 'audio',
      description: 'Real-time procedural WebAudio synthesis graph with live FFT frequency waterfall.',
      canHandle: (ast: ASTNode): boolean => isAudioNode(ast),
      component: AudioDSPWidget,
      priority: 100,
    },
    {
      id: 'hardware-timeline',
      name: 'Hardware Pipeline Visualizer',
      domain: 'hardware',
      description: 'Register dependency DAG and hardware pipeline execution timeline.',
      canHandle: (ast: ASTNode): boolean => isHardwareNode(ast),
      component: HardwareTimelineWidget,
      priority: 100,
    },
    {
      id: 'hyperdim-plotter',
      name: 'Generalized Hyperdim Plotter',
      domain: 'plotter',
      description: 'Multidimensional coordinate slicer with 2D contour heatmaps and 3D heightmaps.',
      canHandle: (ast: ASTNode): boolean =>
        isUniversalNode(ast) || (ast as { domain?: string })?.domain === 'plotter',
      component: HyperdimPlotterWidget,
      priority: 80,
    },
    {
      id: 'generic-ast-tree',
      name: 'Generic AST Tree Inspector',
      domain: 'generic',
      description: 'Interactive visual syntax tree inspector displaying node kinds, parameters, and hierarchy.',
      canHandle: (_ast: ASTNode): boolean => true,
      component: GenericASTWidget,
      priority: 0,
    },
  ];
}

/**
 * Domain Widget Registry singleton engine.
 *
 * Manages the extensible lifecycle of domain visualizer widgets, priority-ordered
 * type-based dispatch resolution, and reactive change notification subscriptions.
 */
export class DomainWidgetRegistry implements WidgetRegistryInterface {
  private readonly widgets: Map<string, DomainWidgetDefinition> = new Map();
  private readonly listeners: Set<WidgetChangeListener> = new Set();

  public constructor() {
    this.reset();
  }

  /**
   * Resets registry state to the default pre-registered widgets.
   */
  public reset(): void {
    this.widgets.clear();
    for (const def of createDefaultWidgetDefinitions()) {
      this.widgets.set(def.id, def);
    }
    this.notifyListeners();
  }

  /**
   * Registers a domain widget definition. Replaces existing definition if ID collides.
   */
  public registerWidget(definition: DomainWidgetDefinition): void {
    if (!definition || typeof definition.id !== 'string') {
      throw new Error('[DomainWidgetRegistry] Invalid widget definition: id is required.');
    }
    this.widgets.set(definition.id, Object.freeze({ ...definition }));
    this.notifyListeners();
  }

  /**
   * Unregisters a widget definition by ID.
   */
  public unregisterWidget(widgetId: string): boolean {
    const deleted = this.widgets.delete(widgetId);
    if (deleted) {
      this.notifyListeners();
    }
    return deleted;
  }

  /**
   * Resolves the highest-priority widget capable of rendering the provided AST node.
   */
  public resolveWidget(ast: ASTNode | null | undefined): DomainWidgetDefinition | null {
    if (!ast) return null;
    const candidates = this.getWidgetsForAST(ast);
    return candidates.length > 0 ? candidates[0] : null;
  }

  /**
   * Returns all registered widgets matching the provided AST node, sorted descending by priority.
   */
  public getWidgetsForAST(ast: ASTNode | null | undefined): DomainWidgetDefinition[] {
    if (!ast) return [];
    const matched: DomainWidgetDefinition[] = [];
    for (const def of this.widgets.values()) {
      try {
        if (def.canHandle(ast)) {
          matched.push(def);
        }
      } catch (err) {
        console.error('[DomainWidgetRegistry] canHandle check threw for widget ' + def.id, {
          error: err,
          widgetId: def.id,
          astKind: ast.kind,
          timestamp: Date.now(),
        });
      }
    }

    return matched.sort((a, b) => b.priority - a.priority);
  }

  /**
   * Returns all currently registered widgets, sorted descending by priority.
   */
  public getAllWidgets(): DomainWidgetDefinition[] {
    return Array.from(this.widgets.values()).sort((a, b) => b.priority - a.priority);
  }

  /**
   * Subscribes a listener to registry modifications.
   */
  public onWidgetsChanged(listener: WidgetChangeListener): () => void {
    this.listeners.add(listener);
    return () => {
      this.listeners.delete(listener);
    };
  }

  private notifyListeners(): void {
    if (this.listeners.size === 0) return;
    const all = this.getAllWidgets();
    for (const listener of this.listeners) {
      try {
        listener(all);
      } catch (err) {
        console.error('[DomainWidgetRegistry] Exception in widget change listener:', err);
      }
    }
  }
}

/**
 * Shared singleton instance of the Domain Widget Registry.
 */
export const widgetRegistry: DomainWidgetRegistry = new DomainWidgetRegistry();
