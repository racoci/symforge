/**
 * ParameterInspector Component for SymForge Web IDE.
 *
 * Provides a dark-themed reactive parameter inspection drawer featuring
 * search filtering, real-time continuous sliders, direct precision numeric inputs,
 * domain badges, and individual/global parameter resets.
 */

import React, { useState, useMemo } from 'react';
import { useFreeParamStore } from '../../core/store/freeParamStore';

export interface ParameterInspectorProps {
  className?: string;
  isCollapsed?: boolean;
  onToggleCollapse?: () => void;
  dataTestId?: string;
}

function getDomainBadge(domainType: string): string {
  if (domainType) {
    const lower = domainType.toLowerCase();
    if (lower.includes('sdf')) return 'SDF';
    if (lower.includes('audio')) return 'Audio';
    if (lower.includes('hardware')) return 'HW';
    return domainType;
  }
  return 'Param';
}

function getSliderStep(min: number, max: number): number {
  const range = Math.abs(max - min);
  if (range <= 0) return 0.01;
  if (range <= 2) return 0.001;
  if (range <= 20) return 0.01;
  if (range <= 200) return 0.1;
  return 1;
}

export const ParameterInspector: React.FC<ParameterInspectorProps> = ({
  className = '',
  isCollapsed = false,
  onToggleCollapse,
  dataTestId = 'parameter-inspector',
}) => {
  const [searchQuery, setSearchQuery] = useState('');

  const params = useFreeParamStore((state) => state.params);
  const paramOrder = useFreeParamStore((state) => state.paramOrder);
  const values = useFreeParamStore((state) => state.values);
  const domainType = useFreeParamStore((state) => state.domainType);
  const declarationName = useFreeParamStore((state) => state.declarationName);
  const updateParamValue = useFreeParamStore((state) => state.updateParamValue);
  const resetParam = useFreeParamStore((state) => state.resetParam);
  const resetAllParams = useFreeParamStore((state) => state.resetAllParams);

  const filteredParamIds = useMemo(() => {
    const q = searchQuery.toLowerCase().trim();
    if (!q) return paramOrder;
    return paramOrder.filter((id) => id.toLowerCase().includes(q));
  }, [paramOrder, searchQuery]);

  const domainBadgeText = useMemo(() => getDomainBadge(domainType), [domainType]);

  const hasModifiedParams = useMemo(() => {
    for (const id of paramOrder) {
      const p = params[id];
      if (!p) continue;
      const val = values[id];
      if (val !== undefined && Math.abs(val - p.defaultVal) > 1e-6) {
        return true;
      }
    }
    return false;
  }, [paramOrder, params, values]);

  if (isCollapsed) {
    return (
      <aside
        className={`param-inspector param-inspector-collapsed ${className}`}
        data-testid={`${dataTestId}-collapsed`}
      >
        <button
          type="button"
          className="btn-toggle-collapse"
          onClick={onToggleCollapse}
          title="Expand Parameter Inspector"
          data-testid="expand-inspector-btn"
        >
          <span>🎚️ Parameters</span>
          <span className="badge">{paramOrder.length}</span>
        </button>
      </aside>
    );
  }

  return (
    <aside
      className={`param-inspector ${className}`}
      data-testid={dataTestId}
      aria-label="Parameter Inspector"
    >
      {/* Inspector Header */}
      <div className="inspector-header">
        <div className="inspector-title-group">
          <span className="inspector-icon" aria-hidden="true">🎚️</span>
          <h3 className="inspector-title">FreeParams</h3>
          <span className="param-count-badge" data-testid="param-count-badge">
            {paramOrder.length}
          </span>
          {declarationName && (
            <span className="inspector-decl-badge" title={declarationName}>
              {declarationName}
            </span>
          )}
        </div>

        <div className="inspector-header-actions">
          <button
            type="button"
            className="btn btn-sm btn-reset-all"
            onClick={resetAllParams}
            disabled={!hasModifiedParams || paramOrder.length === 0}
            data-testid="reset-all-params-btn"
            title="Reset all parameters to default values"
          >
            Reset All
          </button>
          {onToggleCollapse && (
            <button
              type="button"
              className="btn-icon-collapse"
              onClick={onToggleCollapse}
              title="Collapse Inspector"
              data-testid="collapse-inspector-btn"
              aria-label="Collapse Inspector"
            >
              ✕
            </button>
          )}
        </div>
      </div>

      {/* Search Filter Bar */}
      <div className="inspector-search-container">
        <div className="search-input-wrapper">
          <span className="search-icon" aria-hidden="true">🔍</span>
          <input
            type="text"
            className="inspector-search-input"
            placeholder="Search parameters by ID..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            data-testid="param-search-input"
          />
          {searchQuery && (
            <button
              type="button"
              className="btn-clear-search"
              onClick={() => setSearchQuery('')}
              aria-label="Clear search"
              data-testid="clear-search-btn"
            >
              ×
            </button>
          )}
        </div>
      </div>

      {/* Parameter Cards List */}
      <div className="inspector-body" data-testid="inspector-params-list">
        {paramOrder.length === 0 ? (
          <div className="inspector-empty-state" data-testid="param-empty-state">
            <div className="empty-state-icon">⚡</div>
            <p className="empty-state-text">No continuous FreeParams in declaration.</p>
            <p className="empty-state-hint">
              Use <code>FreeParam.mk &quot;id&quot; min max def</code> to declare tunable parameters.
            </p>
          </div>
        ) : filteredParamIds.length === 0 ? (
          <div className="inspector-empty-state" data-testid="param-no-matches-state">
            <p className="empty-state-text">No parameters matching &quot;{searchQuery}&quot;</p>
            <button
              type="button"
              className="btn btn-sm"
              onClick={() => setSearchQuery('')}
            >
              Clear Filter
            </button>
          </div>
        ) : (
          <div className="param-cards-stack">
            {filteredParamIds.map((id) => {
              const param = params[id];
              if (!param) return null;

              const val = values[id] ?? param.defaultVal;
              const step = getSliderStep(param.minVal, param.maxVal);
              const isDefault = Math.abs(val - param.defaultVal) < 1e-6;

              return (
                <div
                  key={id}
                  className={`param-card ${!isDefault ? 'param-card-modified' : ''}`}
                  data-testid={`param-card-${id}`}
                >
                  {/* Parameter Meta Header */}
                  <div className="param-card-header">
                    <div className="param-meta-left">
                      <span className="param-id" data-testid={`param-id-${id}`}>
                        {id}
                      </span>
                      <span className="param-domain-badge" data-testid={`param-domain-${id}`}>
                        {domainBadgeText}
                      </span>
                    </div>

                    <div className="param-meta-right">
                      {/* Direct Numeric Input for Precision */}
                      <input
                        type="number"
                        className="param-num-input"
                        min={param.minVal}
                        max={param.maxVal}
                        step="any"
                        value={Number.isFinite(val) ? Number(val.toFixed(4)) : val}
                        onChange={(e) => {
                          const parsed = parseFloat(e.target.value);
                          if (!Number.isNaN(parsed)) {
                            updateParamValue(id, parsed);
                          }
                        }}
                        data-testid={`param-input-${id}`}
                        aria-label={`Numeric value for ${id}`}
                      />

                      {/* Reset Single Parameter */}
                      <button
                        type="button"
                        className="btn-param-reset"
                        onClick={() => resetParam(id)}
                        disabled={isDefault}
                        title={`Reset to default (${param.defaultVal})`}
                        data-testid={`param-reset-${id}`}
                        aria-label={`Reset ${id} to default`}
                      >
                        ↺
                      </button>
                    </div>
                  </div>

                  {/* Range Slider Row */}
                  <div className="param-slider-row">
                    <span className="param-limit-label" title={`Min: ${param.minVal}`}>
                      {param.minVal}
                    </span>
                    <input
                      type="range"
                      className="param-slider"
                      min={param.minVal}
                      max={param.maxVal}
                      step={step}
                      value={val}
                      onChange={(e) => updateParamValue(id, parseFloat(e.target.value))}
                      data-testid={`param-slider-${id}`}
                      aria-label={`Slider for ${id}`}
                    />
                    <span className="param-limit-label" title={`Max: ${param.maxVal}`}>
                      {param.maxVal}
                    </span>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </aside>
  );
};

export default ParameterInspector;
