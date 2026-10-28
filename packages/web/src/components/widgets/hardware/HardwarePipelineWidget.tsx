import React, { useState, useEffect, useMemo, useRef, useCallback } from 'react';
import type { DomainWidgetProps } from '../types';
import type { HardwareNode } from '../../../core/ast/types';
import { isHardwareNode } from '../../../core/ast/models';
import type {
  HardwareViewMode,
  IssueWidth,
  FlatInstruction,
  ScheduledInstruction,
  DependencyEdge,
} from './types';
import {
  flattenHardwareAST,
  schedulePipeline,
  HARDWARE_BENCHMARK_PRESETS,
} from './hardwarePipeline';

interface DAGNodeLayout {
  readonly instruction: FlatInstruction;
  readonly scheduled: ScheduledInstruction;
  readonly level: number;
  readonly x: number;
  readonly y: number;
  readonly width: number;
  readonly height: number;
}

/**
 * Computes 2D topological layout coordinates for Dependency DAG nodes.
 */
function computeDAGLayout(
  instructions: FlatInstruction[],
  scheduled: ScheduledInstruction[],
  edges: DependencyEdge[]
): {
  nodes: DAGNodeLayout[];
  width: number;
  height: number;
} {
  const schedMap = new Map<string, ScheduledInstruction>(
    scheduled.map((s) => [s.instruction.id, s])
  );

  // Compute topological level (longest path from roots)
  const levels = new Map<string, number>();
  for (const inst of instructions) {
    levels.set(inst.id, 0);
  }

  for (const edge of edges) {
    const fromLvl = levels.get(edge.fromId) ?? 0;
    const toLvl = levels.get(edge.toId) ?? 0;
    if (toLvl <= fromLvl) {
      levels.set(edge.toId, fromLvl + 1);
    }
  }

  // Group by level
  const levelGroups = new Map<number, FlatInstruction[]>();
  for (const inst of instructions) {
    const lvl = levels.get(inst.id) ?? 0;
    const group = levelGroups.get(lvl) ?? [];
    group.push(inst);
    levelGroups.set(lvl, group);
  }

  const nodeWidth = 140;
  const nodeHeight = 52;
  const xSpacing = 180;
  const ySpacing = 72;
  const paddingX = 40;
  const paddingY = 40;

  const nodeLayouts: DAGNodeLayout[] = [];
  let maxLevel = 0;
  let maxNodesInLevel = 1;

  for (const [lvl, group] of levelGroups.entries()) {
    if (lvl > maxLevel) maxLevel = lvl;
    if (group.length > maxNodesInLevel) maxNodesInLevel = group.length;

    for (let idx = 0; idx < group.length; idx++) {
      const inst = group[idx];
      const si = schedMap.get(inst.id)!;
      nodeLayouts.push({
        instruction: inst,
        scheduled: si,
        level: lvl,
        x: paddingX + lvl * xSpacing,
        y: paddingY + idx * ySpacing,
        width: nodeWidth,
        height: nodeHeight,
      });
    }
  }

  const width = Math.max(paddingX * 2 + (maxLevel + 1) * xSpacing, 600);
  const height = Math.max(paddingY * 2 + maxNodesInLevel * ySpacing, 280);

  return { nodes: nodeLayouts, width, height };
}

/**
 * Interactive Hardware Pipeline Visualizer and Register Dependency Timeline widget.
 */
export const HardwarePipelineWidget: React.FC<DomainWidgetProps<HardwareNode>> = ({
  ast,
  declarationName,
}) => {
  const [selectedPresetId, setSelectedPresetId] = useState<string>('custom');
  const [customAst, setCustomAst] = useState<HardwareNode | null>(null);
  const [viewMode, setViewMode] = useState<HardwareViewMode>('split');
  const [issueWidth, setIssueWidth] = useState<IssueWidth>(1);
  const [currentCycle, setCurrentCycle] = useState<number>(0);
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const playTimerRef = useRef<number | null>(null);

  // Determine active AST (from custom user declaration or selected preset)
  const activeAst = useMemo<HardwareNode>(() => {
    if (customAst) return customAst;
    if (ast && isHardwareNode(ast)) return ast;
    return HARDWARE_BENCHMARK_PRESETS[0].createAST();
  }, [ast, customAst]);

  // Extract flat instructions and run pipeline simulation
  const { instructions, scheduleResult, dagLayout } = useMemo(() => {
    const flat = flattenHardwareAST(activeAst);
    const schedule = schedulePipeline(flat, issueWidth);
    const layout = computeDAGLayout(flat, schedule.instructions, schedule.edges);
    return {
      instructions: flat,
      scheduleResult: schedule,
      dagLayout: layout,
    };
  }, [activeAst, issueWidth]);

  const totalCycles = scheduleResult.totalCycles;

  // Auto-play timer for clock cycle scrubber
  useEffect(() => {
    if (isPlaying) {
      playTimerRef.current = window.setInterval(() => {
        setCurrentCycle((prev) => (prev >= totalCycles ? 0 : prev + 1));
      }, 500);
    } else if (playTimerRef.current !== null) {
      clearInterval(playTimerRef.current);
      playTimerRef.current = null;
    }

    return () => {
      if (playTimerRef.current !== null) {
        clearInterval(playTimerRef.current);
        playTimerRef.current = null;
      }
    };
  }, [isPlaying, totalCycles]);

  // Scrubber controls
  const handlePlayPause = useCallback(() => {
    setIsPlaying((prev) => !prev);
  }, []);

  const handleStepBack = useCallback(() => {
    setIsPlaying(false);
    setCurrentCycle((prev) => Math.max(0, prev - 1));
  }, []);

  const handleStepForward = useCallback(() => {
    setIsPlaying(false);
    setCurrentCycle((prev) => Math.min(totalCycles, prev + 1));
  }, [totalCycles]);

  const handleSliderChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    setIsPlaying(false);
    setCurrentCycle(Number(e.target.value));
  }, []);

  const handlePresetSelect = useCallback((e: React.ChangeEvent<HTMLSelectElement>) => {
    const val = e.target.value;
    setSelectedPresetId(val);
    setIsPlaying(false);
    setCurrentCycle(0);
    const preset = HARDWARE_BENCHMARK_PRESETS.find((p) => p.id === val);
    if (preset) {
      setCustomAst(preset.createAST());
    } else {
      setCustomAst(null);
    }
  }, []);

  // Node position map for rendering SVG dependency edges
  const nodePositionMap = useMemo(() => {
    const map = new Map<string, DAGNodeLayout>();
    for (const node of dagLayout.nodes) {
      map.set(node.instruction.id, node);
    }
    return map;
  }, [dagLayout.nodes]);

  // Render fallback if AST is invalid
  if (!activeAst) {
    return (
      <div className="widget-stage" data-testid="hardware-timeline-fallback">
        <div className="stage-card">
          <div className="stage-icon" aria-hidden="true">⚡</div>
          <h2 className="stage-title">Hardware Pipeline Visualizer</h2>
          <p className="stage-description">
            No valid <code>HardwareNode</code> found in declaration <code>{declarationName || 'unknown'}</code>.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="hardware-container" data-testid="hardware-pipeline-widget">
      {/* 1. Control Toolbar */}
      <div className="hardware-toolbar" data-testid="hardware-toolbar">
        {/* Preset Selector */}
        <div className="hardware-toolbar-group">
          <span className="hardware-label">Preset:</span>
          <select
            className="hardware-select"
            value={selectedPresetId}
            onChange={handlePresetSelect}
            data-testid="hardware-preset-select"
          >
            <option value="custom">Current AST ({declarationName || 'HardwareNode'})</option>
            {HARDWARE_BENCHMARK_PRESETS.map((preset) => (
              <option key={preset.id} value={preset.id}>
                {preset.name}
              </option>
            ))}
          </select>
        </div>

        {/* View Mode Switcher */}
        <div className="hardware-toolbar-group">
          <span className="hardware-label">View:</span>
          <button
            type="button"
            className={`hardware-btn ${viewMode === 'dag' ? 'active' : ''}`}
            onClick={() => setViewMode('dag')}
            data-testid="hardware-mode-dag"
          >
            Dependency DAG
          </button>
          <button
            type="button"
            className={`hardware-btn ${viewMode === 'gantt' ? 'active' : ''}`}
            onClick={() => setViewMode('gantt')}
            data-testid="hardware-mode-gantt"
          >
            Gantt Timeline
          </button>
          <button
            type="button"
            className={`hardware-btn ${viewMode === 'liverange' ? 'active' : ''}`}
            onClick={() => setViewMode('liverange')}
            data-testid="hardware-mode-liverange"
          >
            Live Ranges
          </button>
          <button
            type="button"
            className={`hardware-btn ${viewMode === 'split' ? 'active' : ''}`}
            onClick={() => setViewMode('split')}
            data-testid="hardware-mode-split"
          >
            Split View
          </button>
        </div>

        {/* Issue Width Toggle */}
        <div className="hardware-toolbar-group" data-testid="hardware-issue-width-toggle">
          <span className="hardware-label">Issue Width:</span>
          <button
            type="button"
            className={`hardware-btn ${issueWidth === 1 ? 'active' : ''}`}
            onClick={() => setIssueWidth(1)}
            data-testid="hardware-issue-1way"
          >
            1-Way In-Order
          </button>
          <button
            type="button"
            className={`hardware-btn ${issueWidth === 2 ? 'active' : ''}`}
            onClick={() => setIssueWidth(2)}
            data-testid="hardware-issue-2way"
          >
            2-Way Superscalar
          </button>
        </div>

        {/* Clock Cycle Scrubber Controls */}
        <div className="hardware-toolbar-group hardware-scrubber-group">
          <button
            type="button"
            className="hardware-btn hardware-btn-icon"
            onClick={handleStepBack}
            title="Step Backward (|◀)"
            data-testid="hardware-step-back-btn"
          >
            |◀
          </button>
          <button
            type="button"
            className={`hardware-btn hardware-btn-icon ${isPlaying ? 'active' : ''}`}
            onClick={handlePlayPause}
            title={isPlaying ? 'Pause' : 'Play'}
            data-testid="hardware-play-pause-btn"
          >
            {isPlaying ? '⏸' : '▶'}
          </button>
          <button
            type="button"
            className="hardware-btn hardware-btn-icon"
            onClick={handleStepForward}
            title="Step Forward (▶|)"
            data-testid="hardware-step-fwd-btn"
          >
            ▶|
          </button>
          <input
            type="range"
            min={0}
            max={totalCycles}
            value={currentCycle}
            onChange={handleSliderChange}
            className="hardware-slider"
            data-testid="hardware-cycle-slider"
          />
          <span className="hardware-cycle-readout" data-testid="hardware-cycle-readout">
            Cycle: <strong>{currentCycle}</strong> / {totalCycles}
          </span>
        </div>
      </div>

      {/* 2. KPI Summary Banner */}
      <div className="hardware-kpi-banner">
        <div className="hardware-kpi-item" data-testid="hardware-kpi-cycles">
          <span className="hardware-kpi-label">Total Cycles</span>
          <span className="hardware-kpi-val">{totalCycles}</span>
        </div>
        <div className="hardware-kpi-item" data-testid="hardware-kpi-instructions">
          <span className="hardware-kpi-label">Instructions</span>
          <span className="hardware-kpi-val">{instructions.length}</span>
        </div>
        <div className="hardware-kpi-item" data-testid="hardware-kpi-cpi">
          <span className="hardware-kpi-label">CPI</span>
          <span className="hardware-kpi-val">{scheduleResult.cpi.toFixed(2)}</span>
        </div>
        <div className="hardware-kpi-item" data-testid="hardware-kpi-stalls">
          <span className="hardware-kpi-label">Total Stalls</span>
          <span className="hardware-kpi-val hardware-val-warn">{scheduleResult.totalStalls}</span>
        </div>
        <div className="hardware-kpi-item">
          <span className="hardware-kpi-label">Graph Topology</span>
          <span className="hardware-kpi-val hardware-val-ok">
            {scheduleResult.isAcyclic ? 'Acyclic DAG' : 'Cyclic Graph'}
          </span>
        </div>
      </div>

      {/* 3. Main Viewport Stage */}
      <div className="hardware-viewport-stage">
        {/* VIEW: Dependency DAG */}
        {(viewMode === 'dag' || viewMode === 'split') && (
          <div
            className={`hardware-panel hardware-dag-panel ${viewMode === 'split' ? 'split-half' : 'full-width'}`}
            data-testid="hardware-dag-panel"
          >
            <div className="hardware-panel-header">
              <span className="hardware-panel-title">Register Dependency DAG</span>
              <div className="hardware-dag-legend">
                <span className="legend-item legend-raw">■ RAW (True)</span>
                <span className="legend-item legend-war">■ WAR (Anti)</span>
                <span className="legend-item legend-waw">■ WAW (Output)</span>
              </div>
            </div>
            <div className="hardware-svg-scroll">
              <svg
                width={dagLayout.width}
                height={dagLayout.height}
                className="hardware-dag-svg"
                data-testid="hardware-dag-svg"
              >
                <defs>
                  <marker
                    id="arrow-raw"
                    viewBox="0 0 10 10"
                    refX="10"
                    refY="5"
                    markerWidth="6"
                    markerHeight="6"
                    orient="auto-start-reverse"
                  >
                    <path d="M 0 0 L 10 5 L 0 10 z" fill="#06b6d4" />
                  </marker>
                  <marker
                    id="arrow-war"
                    viewBox="0 0 10 10"
                    refX="10"
                    refY="5"
                    markerWidth="6"
                    markerHeight="6"
                    orient="auto-start-reverse"
                  >
                    <path d="M 0 0 L 10 5 L 0 10 z" fill="#f59e0b" />
                  </marker>
                  <marker
                    id="arrow-waw"
                    viewBox="0 0 10 10"
                    refX="10"
                    refY="5"
                    markerWidth="6"
                    markerHeight="6"
                    orient="auto-start-reverse"
                  >
                    <path d="M 0 0 L 10 5 L 0 10 z" fill="#a855f7" />
                  </marker>
                </defs>

                {/* Render Dependency Edges */}
                {scheduleResult.edges.map((edge, idx) => {
                  const from = nodePositionMap.get(edge.fromId);
                  const to = nodePositionMap.get(edge.toId);
                  if (!from || !to) return null;

                  const x1 = from.x + from.width;
                  const y1 = from.y + from.height / 2;
                  const x2 = to.x;
                  const y2 = to.y + to.height / 2;
                  const dx = Math.max(30, (x2 - x1) / 2);
                  const pathD = `M ${x1} ${y1} C ${x1 + dx} ${y1}, ${x2 - dx} ${y2}, ${x2} ${y2}`;

                  const strokeColor =
                    edge.type === 'RAW' ? '#06b6d4' : edge.type === 'WAR' ? '#f59e0b' : '#a855f7';
                  const markerId =
                    edge.type === 'RAW' ? 'arrow-raw' : edge.type === 'WAR' ? 'arrow-war' : 'arrow-waw';

                  return (
                    <g key={`edge-${edge.fromId}-${edge.toId}-${edge.reg}-${idx}`}>
                      <path
                        d={pathD}
                        fill="none"
                        stroke={strokeColor}
                        strokeWidth="1.8"
                        strokeDasharray={edge.type === 'WAR' ? '4 2' : undefined}
                        markerEnd={`url(#${markerId})`}
                      />
                      <rect
                        x={(x1 + x2) / 2 - 12}
                        y={(y1 + y2) / 2 - 8}
                        width="24"
                        height="16"
                        rx="4"
                        fill="var(--bg-secondary, #1e293b)"
                        stroke={strokeColor}
                        strokeWidth="1"
                      />
                      <text
                        x={(x1 + x2) / 2}
                        y={(y1 + y2) / 2 + 4}
                        fill={strokeColor}
                        fontSize="10"
                        fontWeight="600"
                        textAnchor="middle"
                      >
                        {edge.reg}
                      </text>
                    </g>
                  );
                })}

                {/* Render Instruction Nodes */}
                {dagLayout.nodes.map((layout) => {
                  const { instruction, scheduled, x, y, width, height } = layout;
                  const isCurrentExec =
                    currentCycle >= scheduled.executeStartCycle &&
                    currentCycle <= scheduled.executeEndCycle;
                  const isCurrentFetch = currentCycle === scheduled.fetchCycle;
                  const isCurrentDecode = currentCycle === scheduled.decodeCycle;
                  const isCurrentWB = currentCycle === scheduled.writebackCycle;

                  let borderColor = 'var(--border-color, #334155)';
                  let borderGlow = 'none';

                  if (isCurrentExec) {
                    borderColor = '#10b981';
                    borderGlow = '0 0 10px rgba(16, 185, 129, 0.6)';
                  } else if (isCurrentDecode) {
                    borderColor = '#6366f1';
                  } else if (isCurrentFetch) {
                    borderColor = '#38bdf8';
                  } else if (isCurrentWB) {
                    borderColor = '#f59e0b';
                  }

                  const opText = `${instruction.op.toUpperCase()} ${instruction.dst}`;
                  const argsText = [
                    instruction.src1,
                    instruction.src2,
                    instruction.immVal !== undefined ? `#${instruction.immVal}` : null,
                  ]
                    .filter(Boolean)
                    .join(', ');

                  return (
                    <g
                      key={instruction.id}
                      className="dag-node-group"
                      data-testid={`dag-node-${instruction.id}`}
                    >
                      <rect
                        x={x}
                        y={y}
                        width={width}
                        height={height}
                        rx="8"
                        fill="var(--bg-surface, #0f172a)"
                        stroke={borderColor}
                        strokeWidth={isCurrentExec ? 2.5 : 1.5}
                        style={{ filter: borderGlow }}
                      />
                      <text
                        x={x + 10}
                        y={y + 18}
                        fill="#94a3b8"
                        fontSize="11"
                        fontWeight="700"
                      >
                        {instruction.id}
                      </text>
                      <rect
                        x={x + width - 30}
                        y={y + 6}
                        width="22"
                        height="14"
                        rx="4"
                        fill="rgba(56, 189, 248, 0.15)"
                      />
                      <text
                        x={x + width - 19}
                        y={y + 17}
                        fill="#38bdf8"
                        fontSize="9"
                        fontWeight="600"
                        textAnchor="middle"
                      >
                        {instruction.latency}c
                      </text>
                      <text
                        x={x + 10}
                        y={y + 36}
                        fill="#f8fafc"
                        fontSize="12"
                        fontWeight="600"
                      >
                        {opText}
                      </text>
                      {argsText && (
                        <text
                          x={x + 10}
                          y={y + 46}
                          fill="#64748b"
                          fontSize="9"
                        >
                          {argsText}
                        </text>
                      )}
                    </g>
                  );
                })}
              </svg>
            </div>
          </div>
        )}

        {/* VIEW: Gantt Chart and/or Live Ranges */}
        {(viewMode === 'gantt' || viewMode === 'liverange' || viewMode === 'split') && (
          <div
            className={`hardware-panel hardware-timeline-panel ${viewMode === 'split' ? 'split-half' : 'full-width'}`}
          >
            {/* Gantt Chart Section */}
            {(viewMode === 'gantt' || viewMode === 'split') && (
              <div className="hardware-gantt-section" data-testid="hardware-gantt-panel">
                <div className="hardware-panel-header">
                  <span className="hardware-panel-title">Pipeline Execution Gantt Chart</span>
                  <div className="hardware-gantt-legend">
                    <span className="legend-cell stage-fetch">F: Fetch</span>
                    <span className="legend-cell stage-decode">D: Decode</span>
                    <span className="legend-cell stage-exec">E: Execute</span>
                    <span className="legend-cell stage-writeback">W: Writeback</span>
                    <span className="legend-cell stage-stall">· Stall</span>
                  </div>
                </div>
                <div className="hardware-table-scroll">
                  <table className="hardware-gantt-table" data-testid="hardware-gantt-table">
                    <thead>
                      <tr>
                        <th className="th-inst">Instruction</th>
                        {Array.from({ length: totalCycles }, (_, c) => (
                          <th
                            key={`th-cycle-${c}`}
                            className={`th-cycle ${c === currentCycle ? 'th-active-cycle' : ''}`}
                          >
                            C{c}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {scheduleResult.instructions.map((si) => (
                        <tr key={`gantt-row-${si.instruction.id}`}>
                          <td className="td-inst">
                            <span className="gantt-inst-id">{si.instruction.id}:</span>{' '}
                            <span className="gantt-inst-op">
                              {si.instruction.op} {si.instruction.dst}
                            </span>
                          </td>
                          {Array.from({ length: totalCycles }, (_, c) => {
                            let cellContent: React.ReactNode = null;
                            let cellClass = 'stage-cell-empty';

                            if (c === si.fetchCycle) {
                              cellContent = 'F';
                              cellClass = 'stage-cell stage-fetch';
                            } else if (c === si.decodeCycle) {
                              cellContent = 'D';
                              cellClass = 'stage-cell stage-decode';
                            } else if (c > si.decodeCycle && c < si.executeStartCycle) {
                              cellContent = '·';
                              cellClass = 'stage-cell stage-stall';
                            } else if (c >= si.executeStartCycle && c <= si.executeEndCycle) {
                              const eStageNum = si.instruction.latency > 1 ? c - si.executeStartCycle + 1 : '';
                              cellContent = `E${eStageNum}`;
                              cellClass = 'stage-cell stage-exec';
                            } else if (c === si.writebackCycle) {
                              cellContent = 'W';
                              cellClass = 'stage-cell stage-writeback';
                            }

                            return (
                              <td
                                key={`td-cycle-${si.instruction.id}-${c}`}
                                className={`td-stage ${c === currentCycle ? 'td-active-col' : ''}`}
                              >
                                {cellContent && <span className={cellClass}>{cellContent}</span>}
                              </td>
                            );
                          })}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}

            {/* Register Live Ranges Section */}
            {(viewMode === 'liverange' || viewMode === 'split') && (
              <div
                className="hardware-liverange-section"
                data-testid="hardware-liverange-panel"
              >
                <div className="hardware-panel-header">
                  <span className="hardware-panel-title">Register Live Ranges</span>
                </div>
                <div className="hardware-liverange-list">
                  {Object.entries(scheduleResult.registerLiveRanges).map(([reg, range]) => {
                    const spanCycles = range.endCycle - range.startCycle + 1;
                    const leftPct = totalCycles > 0 ? (range.startCycle / totalCycles) * 100 : 0;
                    const widthPct = totalCycles > 0 ? Math.max(2, (spanCycles / totalCycles) * 100) : 100;
                    const isLiveNow = currentCycle >= range.startCycle && currentCycle <= range.endCycle;

                    return (
                      <div key={`liverange-${reg}`} className="liverange-row">
                        <div className="liverange-label">
                          <span className="liverange-reg">{reg}</span>
                          <span className="liverange-span">
                            C{range.startCycle}..C{range.endCycle} ({spanCycles}c)
                          </span>
                        </div>
                        <div className="liverange-track">
                          <div
                            className={`liverange-bar ${isLiveNow ? 'active-range' : ''}`}
                            style={{
                              left: `${leftPct}%`,
                              width: `${widthPct}%`,
                            }}
                          >
                            <span className="liverange-bar-text">{reg}</span>
                          </div>
                          {/* Current cycle needle */}
                          {totalCycles > 0 && (
                            <div
                              className="liverange-cursor"
                              style={{ left: `${(currentCycle / totalCycles) * 100}%` }}
                            />
                          )}
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};
