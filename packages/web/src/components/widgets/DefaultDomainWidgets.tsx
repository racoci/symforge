import React from 'react';
import type { DomainWidgetProps } from './types';
export { SDFRaymarchWidget } from './sdf/SDFRaymarchWidget';

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
