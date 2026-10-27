import React, { useState, useEffect, useMemo, useCallback } from 'react';
import { useFreeParamStore } from '../../core/store/freeParamStore';
import {
  isSDFNode,
  isAudioNode,
  isHardwareNode,
  isUniversalNode,
} from '../../core/ast/models';
import type { ASTNode } from '../../core/ast/types';
import { widgetRegistry } from './registry';
import type { DomainWidgetDefinition } from './types';
import { WidgetErrorBoundary } from './WidgetErrorBoundary';

export interface ViewportShellProps {
  readonly activeDomainTab?: 'sdf' | 'audio' | 'plotter' | 'hardware';
  readonly onDomainTabClick?: (domain: 'sdf' | 'audio' | 'plotter' | 'hardware') => void;
  readonly dataTestId?: string;
}

export const ViewportShell: React.FC<ViewportShellProps> = ({
  activeDomainTab = 'sdf',
  onDomainTabClick,
  dataTestId = 'widget-viewport',
}) => {
  // Subscribe to reactive Zustand state store
  const ast = useFreeParamStore((state) => state.ast);
  const declarationName = useFreeParamStore((state) => state.declarationName);
  const domainType = useFreeParamStore((state) => state.domainType);
  const updateParamValue = useFreeParamStore((state) => state.updateParamValue);

  // Local viewport control states
  const [selectedWidgetId, setSelectedWidgetId] = useState<string>('auto');
  const [isFullscreen, setIsFullscreen] = useState<boolean>(false);
  const [resetNonce, setResetNonce] = useState<number>(0);

  // Registry change subscription
  const [allRegisteredWidgets, setAllRegisteredWidgets] = useState<DomainWidgetDefinition[]>(() =>
    widgetRegistry.getAllWidgets()
  );

  useEffect(() => {
    const unsubscribe = widgetRegistry.onWidgetsChanged((updated) => {
      setAllRegisteredWidgets(updated);
    });
    return unsubscribe;
  }, []);

  // Compute available widgets for current AST
  const availableWidgets = useMemo(() => {
    return widgetRegistry.getWidgetsForAST(ast);
  }, [ast, allRegisteredWidgets]);

  // Compute auto-resolved default widget
  const autoResolvedWidget = useMemo(() => {
    return widgetRegistry.resolveWidget(ast);
  }, [ast, allRegisteredWidgets]);

  // Determine active widget definition to render
  const activeWidgetDefinition = useMemo((): DomainWidgetDefinition | null => {
    if (selectedWidgetId !== 'auto') {
      const manual = allRegisteredWidgets.find((w) => w.id === selectedWidgetId);
      if (manual) return manual;
    }

    // If a domain tab is active and no conflicting typed AST overrides it, match domain tab
    if (activeDomainTab === 'plotter') {
      const plotterWidget = allRegisteredWidgets.find((w) => w.id === 'hyperdim-plotter');
      if (plotterWidget) return plotterWidget;
    } else if (activeDomainTab === 'audio' && (!ast || isAudioNode(ast))) {
      const audioWidget = allRegisteredWidgets.find((w) => w.id === 'audio-dsp');
      if (audioWidget) return audioWidget;
    } else if (activeDomainTab === 'hardware' && (!ast || isHardwareNode(ast))) {
      const hwWidget = allRegisteredWidgets.find((w) => w.id === 'hardware-timeline');
      if (hwWidget) return hwWidget;
    } else if (activeDomainTab === 'sdf' && (!ast || isSDFNode(ast))) {
      const sdfWidget = allRegisteredWidgets.find((w) => w.id === 'sdf-raymarch');
      if (sdfWidget) return sdfWidget;
    }

    return autoResolvedWidget;
  }, [selectedWidgetId, allRegisteredWidgets, autoResolvedWidget, activeDomainTab, ast]);

  // Derive domain badge
  const domainBadge = useMemo(() => {
    if (ast) {
      if (isSDFNode(ast)) return 'SDF';
      if (isAudioNode(ast)) return 'Audio';
      if (isHardwareNode(ast)) return 'Hardware';
      if (isUniversalNode(ast)) return 'Universal';
      return 'Generic';
    }
    return activeDomainTab.toUpperCase();
  }, [ast, activeDomainTab]);

  const handleReset = useCallback(() => {
    setSelectedWidgetId('auto');
    setResetNonce((n) => n + 1);
  }, []);

  const handleToggleFullscreen = useCallback(() => {
    setIsFullscreen((prev) => !prev);
  }, []);

  const handleFallbackToGeneric = useCallback(() => {
    setSelectedWidgetId('generic-ast-tree');
  }, []);

  return (
    <section
      className={`viewport-panel ${isFullscreen ? 'viewport-panel-fullscreen' : ''}`}
      aria-label="Domain Viewport Panel"
      data-testid="viewport-shell-container"
    >
      {/* Viewport Header Bar */}
      <div className="panel-header viewport-shell-header">
        <div className="viewport-header-left" style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <span>Viewport: Interactive Domain Widget</span>
          <span
            className="brand-badge"
            style={{
              textTransform: 'uppercase',
              fontSize: '0.625rem',
              letterSpacing: '0.04em',
            }}
            data-testid="viewport-domain-badge"
          >
            {domainBadge}
          </span>
        </div>

        {/* Center: Domain Presets (Tabs) */}
        {onDomainTabClick && (
          <div className="domain-tags" role="tablist" style={{ margin: 0 }}>
            <button
              type="button"
              className={`btn ${activeDomainTab === 'sdf' ? 'btn-primary' : ''}`}
              style={{ padding: '2px 8px', fontSize: '0.75rem' }}
              onClick={() => onDomainTabClick('sdf')}
              data-testid="domain-tab-sdf"
            >
              3D SDF
            </button>
            <button
              type="button"
              className={`btn ${activeDomainTab === 'audio' ? 'btn-primary' : ''}`}
              style={{ padding: '2px 8px', fontSize: '0.75rem' }}
              onClick={() => onDomainTabClick('audio')}
              data-testid="domain-tab-audio"
            >
              WebAudio
            </button>
            <button
              type="button"
              className={`btn ${activeDomainTab === 'plotter' ? 'btn-primary' : ''}`}
              style={{ padding: '2px 8px', fontSize: '0.75rem' }}
              onClick={() => onDomainTabClick('plotter')}
              data-testid="domain-tab-plotter"
            >
              Hyperdim
            </button>
            <button
              type="button"
              className={`btn ${activeDomainTab === 'hardware' ? 'btn-primary' : ''}`}
              style={{ padding: '2px 8px', fontSize: '0.75rem' }}
              onClick={() => onDomainTabClick('hardware')}
              data-testid="domain-tab-hardware"
            >
              Hardware
            </button>
          </div>
        )}

        {/* Right: Controls & Widget Switcher */}
        <div className="viewport-header-right" style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          {/* Widget Switcher Dropdown */}
          <div className="widget-switcher-wrapper" style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
            <label htmlFor="widget-select" style={{ fontSize: '0.6875rem', color: 'var(--text-muted)' }}>
              Widget:
            </label>
            <select
              id="widget-select"
              className="widget-select-dropdown"
              value={selectedWidgetId}
              onChange={(e) => setSelectedWidgetId(e.target.value)}
              data-testid="widget-switcher"
              style={{
                backgroundColor: 'var(--bg-surface)',
                color: 'var(--text-primary)',
                border: '1px solid var(--border-color)',
                borderRadius: '4px',
                padding: '2px 6px',
                fontSize: '0.75rem',
                outline: 'none',
              }}
            >
              <option value="auto">
                Auto ({autoResolvedWidget ? autoResolvedWidget.name : 'None'})
              </option>
              {allRegisteredWidgets.map((w) => (
                <option key={w.id} value={w.id}>
                  [{w.domain.toUpperCase()}] {w.name} {availableWidgets.some((aw) => aw.id === w.id) ? '✓' : ''}
                </option>
              ))}
            </select>
          </div>

          {/* Reset Controls Button */}
          <button
            type="button"
            className="btn"
            style={{ padding: '2px 6px', fontSize: '0.75rem' }}
            onClick={handleReset}
            title="Reset Viewport Controls"
            data-testid="viewport-reset-btn"
          >
            ↺ Reset
          </button>

          {/* Fullscreen Toggle */}
          <button
            type="button"
            className="btn"
            style={{ padding: '2px 6px', fontSize: '0.75rem' }}
            onClick={handleToggleFullscreen}
            title={isFullscreen ? 'Exit Fullscreen' : 'Enter Fullscreen'}
            data-testid="viewport-fullscreen-btn"
          >
            {isFullscreen ? '⤡' : '⛶'}
          </button>
        </div>
      </div>

      {/* Viewport Body with Error Boundary Isolation */}
      <div
        className="viewport-body"
        style={{ flex: 1, display: 'flex', flexDirection: 'column', position: 'relative', overflow: 'hidden' }}
        data-testid={dataTestId}
      >
        <WidgetErrorBoundary
          resetKey={`${activeWidgetDefinition?.id}_${declarationName}_${resetNonce}`}
          onReset={handleReset}
          onFallbackToGeneric={handleFallbackToGeneric}
        >
          {activeWidgetDefinition ? (
            React.createElement(activeWidgetDefinition.component, {
              ast: ast as ASTNode,
              domain: domainType || activeWidgetDefinition.domain,
              declarationName,
              onParameterChange: updateParamValue,
            })
          ) : (
            <div className="widget-stage" data-testid="viewport-waiting-state">
              <div className="stage-card">
                <div className="stage-icon">⏳</div>
                <h2 className="stage-title">Waiting for Valid AST</h2>
                <p className="stage-description">
                  Enter or synchronize Lean 4 code defining a valid domain expression (e.g. <code>SDFNode</code>, <code>AudioNode</code>, or <code>HardwareNode</code>).
                </p>
              </div>
            </div>
          )}
        </WidgetErrorBoundary>
      </div>

      {/* Sub-bar / Viewport Status Footer */}
      <div
        className="viewport-subbar"
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          padding: '2px 12px',
          fontSize: '0.6875rem',
          color: 'var(--text-muted)',
          backgroundColor: 'var(--bg-primary)',
          borderTop: '1px solid var(--border-color)',
        }}
      >
        <span>Active Widget: {activeWidgetDefinition?.name || 'None'}</span>
        <span>Mode: {selectedWidgetId === 'auto' ? 'Auto-Dispatched' : 'Manual Override'}</span>
      </div>
    </section>
  );
};
