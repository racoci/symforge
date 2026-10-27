import React from 'react';
import type { DomainWidgetProps } from './types';
import { collectFreeParams } from '../../core/ast/models';

/**
 * Default placeholder stage component for the 3D WebGL Raymarching Widget (Task 2.2).
 */
export const SDFRaymarchWidget: React.FC<DomainWidgetProps> = ({
  ast,
  declarationName,
}) => {
  const params = React.useMemo(() => (ast ? collectFreeParams(ast) : []), [ast]);

  return (
    <div className="widget-stage" data-testid="sdf-raymarch-widget">
      <div className="stage-card">
        <div className="stage-icon" aria-hidden="true">🔮</div>
        <h2 className="stage-title">3D WebGL SDF Raymarching</h2>
        <p className="stage-description">
          Interactive 3D WebGL raymarching canvas with Phong illumination and real-time FreeParam uniform modulation for <code>{declarationName || 'SDFNode'}</code>.
        </p>
        <div className="domain-tags">
          <span className="domain-tag">Raymarching 60 FPS</span>
          <span className="domain-tag">Three.js Canvas</span>
          <span className="domain-tag">CSG Invariants</span>
        </div>
        {params.length > 0 && (
          <div style={{ marginTop: '16px', width: '100%', textAlign: 'left' }}>
            <span style={{ fontSize: '0.6875rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
              Bound Parameters ({params.length})
            </span>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '6px', marginTop: '6px' }}>
              {params.slice(0, 3).map((p) => (
                <div
                  key={p.id}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    fontSize: '0.75rem',
                    background: 'var(--bg-surface)',
                    padding: '4px 8px',
                    borderRadius: '4px',
                    border: '1px solid var(--border-color)',
                  }}
                >
                  <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--accent-cyan)' }}>{p.id}</span>
                  <span style={{ color: 'var(--text-secondary)' }}>
                    [{p.minVal}..{p.maxVal}] (def: {p.defaultVal})
                  </span>
                </div>
              ))}
              {params.length > 3 && (
                <span style={{ fontSize: '0.6875rem', color: 'var(--text-muted)', textAlign: 'center' }}>
                  +{params.length - 3} more in Inspector
                </span>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

/**
 * Default placeholder stage component for the WebAudio DSP Synthesis Widget (Task 2.4).
 */
export const AudioDSPWidget: React.FC<DomainWidgetProps> = ({
  declarationName,
}) => {
  return (
    <div className="widget-stage" data-testid="audio-dsp-widget">
      <div className="stage-card">
        <div className="stage-icon" aria-hidden="true">🎛️</div>
        <h2 className="stage-title">WebAudio DSP Synthesis & FFT</h2>
        <p className="stage-description">
          Real-time procedural WebAudio synthesis graph with live FFT frequency waterfall and oscilloscope for <code>{declarationName || 'AudioNode'}</code>.
        </p>
        <div className="domain-tags">
          <span className="domain-tag">WebAudio Graph</span>
          <span className="domain-tag">2048-pt FFT</span>
          <span className="domain-tag">Zero-Latency DSP</span>
        </div>
      </div>
    </div>
  );
};

/**
 * Default placeholder stage component for the Hardware Pipeline Visualizer (Task 2.6).
 */
export const HardwareTimelineWidget: React.FC<DomainWidgetProps> = ({
  declarationName,
}) => {
  return (
    <div className="widget-stage" data-testid="hardware-timeline-widget">
      <div className="stage-card">
        <div className="stage-icon" aria-hidden="true">⚡</div>
        <h2 className="stage-title">Hardware Pipeline Visualizer</h2>
        <p className="stage-description">
          Directed acyclic register dependency graph, instruction cycle execution timeline, and hazard inspector for <code>{declarationName || 'HardwareNode'}</code>.
        </p>
        <div className="domain-tags">
          <span className="domain-tag">Dependency DAG</span>
          <span className="domain-tag">Cycle Timeline</span>
          <span className="domain-tag">Register Allocation</span>
        </div>
      </div>
    </div>
  );
};

/**
 * Default placeholder stage component for the Generalized Hyperdimensional Plotter (Task 2.5).
 */
export const HyperdimPlotterWidget: React.FC<DomainWidgetProps> = ({
  declarationName,
}) => {
  return (
    <div className="widget-stage" data-testid="hyperdim-plotter-widget">
      <div className="stage-card">
        <div className="stage-icon" aria-hidden="true">📈</div>
        <h2 className="stage-title">Generalized Hyperdim Plotter</h2>
        <p className="stage-description">
          Multidimensional coordinate slicer with 2D contour heatmaps and 3D heightmaps for <code>{declarationName || 'R^n -> R^m'}</code>.
        </p>
        <div className="domain-tags">
          <span className="domain-tag">Coordinate Slicer</span>
          <span className="domain-tag">Contour Heatmap</span>
          <span className="domain-tag">Heightmap Mesh</span>
        </div>
      </div>
    </div>
  );
};
