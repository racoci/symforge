import type React from 'react';
import type { ASTNode } from '../../core/ast/types';

/**
 * Common properties supplied to all domain visualizer widgets.
 */
export interface DomainWidgetProps<T = ASTNode> {
  /**
   * The parsed, typed Abstract Syntax Tree for the current declaration.
   */
  readonly ast: T;

  /**
   * Domain identifier (e.g., 'sdf', 'audio', 'hardware', 'universal', 'generic').
   */
  readonly domain: string;

  /**
   * Lean 4 declaration identifier (e.g., 'myGyroidSDF', 'mainAudio').
   */
  readonly declarationName: string;

  /**
   * Optional callback triggered when a continuous parameter value is modified within the widget.
   */
  readonly onParameterChange?: (id: string, value: number) => void;
}

/**
 * Specification and metadata definition for registering an extensible domain widget.
 */
export interface DomainWidgetDefinition {
  /**
   * Unique identifier for the widget (e.g., 'sdf-raymarch', 'audio-dsp', 'generic-ast-tree').
   */
  readonly id: string;

  /**
   * Human-readable display name.
   */
  readonly name: string;

  /**
   * Domain classification (e.g., 'sdf', 'audio', 'hardware', 'plotter', 'generic').
   */
  readonly domain: string;

  /**
   * Descriptive documentation of the widget's rendering capabilities.
   */
  readonly description: string;

  /**
   * Predicate determining if this widget can visualize the provided AST node.
   */
  readonly canHandle: (ast: ASTNode) => boolean;

  /**
   * React component responsible for rendering the interactive viewport.
   */
  readonly component: React.ComponentType<DomainWidgetProps<any>>;

  /**
   * Priority score used for automated resolution (higher priority takes precedence).
   * Default domain widgets use 100, generic fallback uses 0.
   */
  readonly priority: number;
}

/**
 * Interface contract for the Domain Widget Registry.
 */
export interface WidgetRegistryInterface {
  /**
   * Registers a new domain widget definition. Replaces existing widget with the same ID.
   */
  registerWidget(definition: DomainWidgetDefinition): void;

  /**
   * Unregisters a widget by ID. Returns true if removed, false if not found.
   */
  unregisterWidget(widgetId: string): boolean;

  /**
   * Resolves the highest-priority widget capable of handling the provided AST.
   */
  resolveWidget(ast: ASTNode | null | undefined): DomainWidgetDefinition | null;

  /**
   * Returns all registered widgets capable of handling the provided AST, sorted by priority descending.
   */
  getWidgetsForAST(ast: ASTNode | null | undefined): DomainWidgetDefinition[];

  /**
   * Returns all currently registered widgets, sorted by priority descending.
   */
  getAllWidgets(): DomainWidgetDefinition[];

  /**
   * Subscribes a listener to registry modifications. Returns an unsubscribe cleanup callback.
   */
  onWidgetsChanged(listener: (widgets: DomainWidgetDefinition[]) => void): () => void;

  /**
   * Resets the registry to the default pre-registered widgets.
   */
  reset(): void;
}
