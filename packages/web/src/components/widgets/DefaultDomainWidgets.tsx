import React from 'react';
import type { DomainWidgetProps } from './types';
export { SDFRaymarchWidget } from './sdf/SDFRaymarchWidget';
export { AudioDSPWidget } from './audio/AudioDSPWidget';
export { HyperdimPlotterWidget } from './hyperdim';

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
