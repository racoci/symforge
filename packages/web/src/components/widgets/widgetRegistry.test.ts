import { describe, it, expect, beforeEach, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import React from 'react';
import {
  DomainWidgetRegistry,
  widgetRegistry,
} from './registry';
import type { DomainWidgetDefinition, DomainWidgetProps } from './types';
import {
  createSphere,
  createDifference,
  createSine,
  createLowPass,
  createAdd,
  createReg,
  createImm,
  createFreeParam,
} from '../../core/ast/models';
import type { ASTNode } from '../../core/ast/types';
import { ViewportShell } from './ViewportShell';
import { WidgetErrorBoundary } from './WidgetErrorBoundary';
import { useFreeParamStore } from '../../core/store/freeParamStore';

describe('DomainWidgetRegistry & Type-Based Dispatch', () => {
  let registry: DomainWidgetRegistry;

  beforeEach(() => {
    registry = new DomainWidgetRegistry();
    widgetRegistry.reset();
  });

  describe('Registration, Unregistration & Priority Ordering', () => {
    it('initializes with default domain widgets in priority descending order', () => {
      const all = registry.getAllWidgets();
      expect(all.length).toBeGreaterThanOrEqual(4);

      const ids = all.map((w) => w.id);
      expect(ids).toContain('sdf-raymarch');
      expect(ids).toContain('audio-dsp');
      expect(ids).toContain('hardware-timeline');
      expect(ids).toContain('generic-ast-tree');

      // Verify descending priority order
      for (let i = 0; i < all.length - 1; i++) {
        expect(all[i].priority).toBeGreaterThanOrEqual(all[i + 1].priority);
      }
      expect(all[all.length - 1].id).toBe('generic-ast-tree');
      expect(all[all.length - 1].priority).toBe(0);
    });

    it('registers custom widgets and sorts by priority', () => {
      const mockComponent: React.FC<DomainWidgetProps<any>> = () =>
        React.createElement('div', null, 'Custom Widget');
      const customWidget: DomainWidgetDefinition = {
        id: 'quantum-circuit-vis',
        name: 'Quantum Circuit Visualizer',
        domain: 'quantum',
        description: 'Bloch sphere and quantum gate timeline',
        canHandle: (ast: ASTNode) => (ast as any).kind === 'quantum_circuit',
        component: mockComponent,
        priority: 150,
      };

      registry.registerWidget(customWidget);
      const all = registry.getAllWidgets();

      expect(all[0].id).toBe('quantum-circuit-vis');
      expect(all[0].priority).toBe(150);

      const fakeNode = { kind: 'quantum_circuit' } as unknown as ASTNode;
      const resolved = registry.resolveWidget(fakeNode);
      expect(resolved?.id).toBe('quantum-circuit-vis');
    });

    it('unregisters widgets by id and notifies subscribers', () => {
      const listener = vi.fn();
      const unsubscribe = registry.onWidgetsChanged(listener);

      const removed = registry.unregisterWidget('audio-dsp');
      expect(removed).toBe(true);
      expect(registry.getAllWidgets().map((w) => w.id)).not.toContain('audio-dsp');
      expect(listener).toHaveBeenCalled();

      // Non-existent id returns false
      const nonExistent = registry.unregisterWidget('non-existent-widget');
      expect(nonExistent).toBe(false);

      unsubscribe();
    });

    it('resets registry back to default widgets', () => {
      registry.unregisterWidget('sdf-raymarch');
      expect(registry.getAllWidgets().some((w) => w.id === 'sdf-raymarch')).toBe(false);

      registry.reset();
      expect(registry.getAllWidgets().some((w) => w.id === 'sdf-raymarch')).toBe(true);
    });
  });

  describe('Pass-to-Complete Gate: 4 Distinct Domain AST Resolutions', () => {
    it('resolves SphereNode to sdf-raymarch', () => {
      const sphereNode = createSphere(createFreeParam('radius', 0.1, 5.0, 1.0));
      const resolved = registry.resolveWidget(sphereNode);

      expect(resolved).not.toBeNull();
      expect(resolved?.id).toBe('sdf-raymarch');
      expect(resolved?.domain).toBe('sdf');
      expect(resolved?.name).toBe('3D WebGL SDF Raymarching');
    });

    it('resolves DifferenceNode to sdf-raymarch', () => {
      const sphere1 = createSphere(createFreeParam('r1', 0.1, 5.0, 1.5));
      const sphere2 = createSphere(createFreeParam('r2', 0.1, 5.0, 0.8));
      const diffNode = createDifference(sphere1, sphere2);

      const resolved = registry.resolveWidget(diffNode);
      expect(resolved).not.toBeNull();
      expect(resolved?.id).toBe('sdf-raymarch');
      expect(resolved?.domain).toBe('sdf');
    });

    it('resolves SineNode to audio-dsp', () => {
      const sineNode = createSine(createFreeParam('freq', 20, 20000, 440));
      const resolved = registry.resolveWidget(sineNode);

      expect(resolved).not.toBeNull();
      expect(resolved?.id).toBe('audio-dsp');
      expect(resolved?.domain).toBe('audio');
      expect(resolved?.name).toBe('WebAudio DSP Synthesis & FFT');
    });

    it('resolves LowPassNode to audio-dsp', () => {
      const sine = createSine(createFreeParam('freq', 20, 20000, 440));
      const lowpassNode = createLowPass(
        createFreeParam('cutoff', 20, 20000, 1000),
        createFreeParam('res', 0.1, 10, 0.707),
        sine
      );

      const resolved = registry.resolveWidget(lowpassNode);
      expect(resolved).not.toBeNull();
      expect(resolved?.id).toBe('audio-dsp');
    });

    it('resolves AddHardwareNode to hardware-timeline', () => {
      const reg = createReg('r0');
      const imm = createImm(42);
      const addNode = createAdd(reg, imm);

      const resolved = registry.resolveWidget(addNode);
      expect(resolved).not.toBeNull();
      expect(resolved?.id).toBe('hardware-timeline');
      expect(resolved?.domain).toBe('hardware');
      expect(resolved?.name).toBe('Hardware Pipeline Visualizer');
    });

    it('resolves RegHardwareNode to hardware-timeline', () => {
      const regNode = createReg('r1');
      const resolved = registry.resolveWidget(regNode);

      expect(resolved).not.toBeNull();
      expect(resolved?.id).toBe('hardware-timeline');
    });

    it('resolves unknown/custom node to generic-ast-tree fallback', () => {
      const unknownNode = {
        kind: 'unsupported_custom_node',
        customField: 123,
      } as unknown as ASTNode;

      const resolved = registry.resolveWidget(unknownNode);
      expect(resolved).not.toBeNull();
      expect(resolved?.id).toBe('generic-ast-tree');
      expect(resolved?.domain).toBe('generic');
      expect(resolved?.name).toBe('Generic AST Tree Inspector');
    });

    it('resolves null or undefined AST to null', () => {
      expect(registry.resolveWidget(null)).toBeNull();
      expect(registry.resolveWidget(undefined)).toBeNull();
    });
  });

  describe('Candidate Widgets for AST (getWidgetsForAST)', () => {
    it('returns both primary domain widget and generic fallback for domain AST', () => {
      const sphere = createSphere(createFreeParam('r', 0.5, 2.0, 1.0));
      const candidates = registry.getWidgetsForAST(sphere);

      expect(candidates.length).toBeGreaterThanOrEqual(2);
      expect(candidates[0].id).toBe('sdf-raymarch');
      expect(candidates[candidates.length - 1].id).toBe('generic-ast-tree');
    });

    it('returns empty array for null/undefined AST', () => {
      expect(registry.getWidgetsForAST(null)).toEqual([]);
      expect(registry.getWidgetsForAST(undefined)).toEqual([]);
    });
  });
});

describe('ViewportShell Integration & Manual Override', () => {
  beforeEach(() => {
    widgetRegistry.reset();
  });

  it('renders resolved SDF widget and allows manual override to generic-ast-tree', () => {
    const sphere = createSphere(createFreeParam('radius', 0.5, 3.0, 1.25));
    useFreeParamStore.getState().setAST(sphere, 'mySphere', 'SDFNode');

    render(React.createElement(ViewportShell, { activeDomainTab: 'sdf' }));

    // Default SDF widget renders
    expect(screen.getByText('3D WebGL SDF Raymarching')).toBeInTheDocument();
    expect(screen.getByTestId('viewport-domain-badge')).toHaveTextContent('SDF');

    // Switch widget to generic-ast-tree via dropdown
    const select = screen.getByTestId('widget-switcher') as HTMLSelectElement;
    fireEvent.change(select, { target: { value: 'generic-ast-tree' } });

    // Generic AST widget renders
    expect(screen.getByText(/AST Tree Inspector:/i)).toBeInTheDocument();
    expect(screen.getByTestId('generic-ast-widget')).toBeInTheDocument();

    // Click Reset button to return to Auto
    const resetBtn = screen.getByTestId('viewport-reset-btn');
    fireEvent.click(resetBtn);

    // Auto SDF widget restored
    expect(screen.getByText('3D WebGL SDF Raymarching')).toBeInTheDocument();
  });

  it('toggles fullscreen mode in ViewportShell', () => {
    render(React.createElement(ViewportShell, { activeDomainTab: 'sdf' }));

    const container = screen.getByTestId('viewport-shell-container');
    const fullscreenBtn = screen.getByTestId('viewport-fullscreen-btn');

    expect(container).not.toHaveClass('viewport-panel-fullscreen');
    fireEvent.click(fullscreenBtn);
    expect(container).toHaveClass('viewport-panel-fullscreen');

    fireEvent.click(fullscreenBtn);
    expect(container).not.toHaveClass('viewport-panel-fullscreen');
  });
});

describe('WidgetErrorBoundary Fault Isolation', () => {
  const ThrowingWidget: React.FC = () => {
    throw new Error('Simulated WebGL Context Lost');
  };

  it('isolates runtime rendering exceptions without crashing the parent layout', () => {
    const spy = vi.spyOn(console, 'error').mockImplementation(() => {});

    render(
      React.createElement(
        WidgetErrorBoundary,
        null,
        React.createElement(ThrowingWidget, null)
      )
    );

    expect(screen.getByTestId('widget-error-boundary-fallback')).toBeInTheDocument();
    expect(screen.getByText('Widget Runtime Exception Isolated')).toBeInTheDocument();
    expect(screen.getByText('Simulated WebGL Context Lost')).toBeInTheDocument();

    spy.mockRestore();
  });

  it('recovers when retry button is clicked after error condition clears', () => {
    const spy = vi.spyOn(console, 'error').mockImplementation(() => {});
    let throwError = true;

    const FlakyWidget: React.FC = () => {
      if (throwError) {
        throw new Error('Transient shader compilation crash');
      }
      return React.createElement('div', null, 'Restored Visual Canvas');
    };

    const { rerender } = render(
      React.createElement(
        WidgetErrorBoundary,
        null,
        React.createElement(FlakyWidget, null)
      )
    );

    expect(screen.getByTestId('widget-error-boundary-fallback')).toBeInTheDocument();

    // Clear error condition and click retry
    throwError = false;
    const retryBtn = screen.getByTestId('widget-error-retry-btn');
    fireEvent.click(retryBtn);

    rerender(
      React.createElement(
        WidgetErrorBoundary,
        null,
        React.createElement(FlakyWidget, null)
      )
    );

    expect(screen.getByText('Restored Visual Canvas')).toBeInTheDocument();
    spy.mockRestore();
  });
});
