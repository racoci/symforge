import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, fireEvent, act } from '@testing-library/react';
import { HardwarePipelineWidget } from './HardwarePipelineWidget';
import { HARDWARE_BENCHMARK_PRESETS } from './hardwarePipeline';
import { createAdd, createReg, createImm } from '../../../core/ast/models';

describe('HardwarePipelineWidget Component', () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.clearAllTimers();
    vi.useRealTimers();
  });

  it('renders correctly with default AST in split view mode', () => {
    const ast = HARDWARE_BENCHMARK_PRESETS[0].createAST();
    render(
      <HardwarePipelineWidget
        ast={ast}
        domain="hardware"
        declarationName="myArithmeticChain"
      />
    );

    expect(screen.getByTestId('hardware-pipeline-widget')).toBeInTheDocument();
    expect(screen.getByTestId('hardware-toolbar')).toBeInTheDocument();
    expect(screen.getByTestId('hardware-kpi-cycles')).toBeInTheDocument();
    expect(screen.getByTestId('hardware-kpi-instructions')).toBeInTheDocument();
    expect(screen.getByTestId('hardware-kpi-cpi')).toBeInTheDocument();
    expect(screen.getByTestId('hardware-kpi-stalls')).toBeInTheDocument();

    // In split mode, both DAG and timeline panels are rendered
    expect(screen.getByTestId('hardware-dag-panel')).toBeInTheDocument();
    expect(screen.getByTestId('hardware-gantt-panel')).toBeInTheDocument();
    expect(screen.getByTestId('hardware-liverange-panel')).toBeInTheDocument();
  });

  it('switches view modes via toolbar buttons', () => {
    const ast = HARDWARE_BENCHMARK_PRESETS[0].createAST();
    render(
      <HardwarePipelineWidget
        ast={ast}
        domain="hardware"
        declarationName="myPipeline"
      />
    );

    // Switch to DAG view only
    fireEvent.click(screen.getByTestId('hardware-mode-dag'));
    expect(screen.getByTestId('hardware-dag-panel')).toBeInTheDocument();
    expect(screen.queryByTestId('hardware-gantt-panel')).not.toBeInTheDocument();

    // Switch to Gantt view only
    fireEvent.click(screen.getByTestId('hardware-mode-gantt'));
    expect(screen.queryByTestId('hardware-dag-panel')).not.toBeInTheDocument();
    expect(screen.getByTestId('hardware-gantt-panel')).toBeInTheDocument();

    // Switch to Live Range view only
    fireEvent.click(screen.getByTestId('hardware-mode-liverange'));
    expect(screen.queryByTestId('hardware-dag-panel')).not.toBeInTheDocument();
    expect(screen.getByTestId('hardware-liverange-panel')).toBeInTheDocument();

    // Switch back to Split view
    fireEvent.click(screen.getByTestId('hardware-mode-split'));
    expect(screen.getByTestId('hardware-dag-panel')).toBeInTheDocument();
    expect(screen.getByTestId('hardware-gantt-panel')).toBeInTheDocument();
  });

  it('toggles issue width between 1-way and 2-way superscalar', () => {
    const ast = HARDWARE_BENCHMARK_PRESETS[1].createAST();
    render(
      <HardwarePipelineWidget
        ast={ast}
        domain="hardware"
        declarationName="parallelOps"
      />
    );

    const btn1Way = screen.getByTestId('hardware-issue-1way');
    const btn2Way = screen.getByTestId('hardware-issue-2way');

    expect(btn1Way).toHaveClass('active');
    expect(btn2Way).not.toHaveClass('active');

    // Toggle to 2-Way
    fireEvent.click(btn2Way);
    expect(btn2Way).toHaveClass('active');
    expect(btn1Way).not.toHaveClass('active');

    // Toggle back to 1-Way
    fireEvent.click(btn1Way);
    expect(btn1Way).toHaveClass('active');
  });

  it('controls clock cycle scrubber via slider, step buttons, and auto-play', () => {
    const ast = HARDWARE_BENCHMARK_PRESETS[2].createAST(); // MAC benchmark
    render(
      <HardwarePipelineWidget
        ast={ast}
        domain="hardware"
        declarationName="macBenchmark"
      />
    );

    const readout = screen.getByTestId('hardware-cycle-readout');
    expect(readout).toHaveTextContent('Cycle: 0 / 7');

    // Step Forward
    fireEvent.click(screen.getByTestId('hardware-step-fwd-btn'));
    expect(readout).toHaveTextContent('Cycle: 1 / 7');

    // Step Forward again
    fireEvent.click(screen.getByTestId('hardware-step-fwd-btn'));
    expect(readout).toHaveTextContent('Cycle: 2 / 7');

    // Step Backward
    fireEvent.click(screen.getByTestId('hardware-step-back-btn'));
    expect(readout).toHaveTextContent('Cycle: 1 / 7');

    // Slider scrubbing
    const slider = screen.getByTestId('hardware-cycle-slider');
    fireEvent.change(slider, { target: { value: '5' } });
    expect(readout).toHaveTextContent('Cycle: 5 / 7');

    // Auto-Play toggle
    const playBtn = screen.getByTestId('hardware-play-pause-btn');
    expect(playBtn).toHaveTextContent('▶');

    fireEvent.click(playBtn);
    expect(playBtn).toHaveTextContent('⏸');

    // Advance timer by 500ms
    act(() => {
      vi.advanceTimersByTime(500);
    });
    expect(readout).toHaveTextContent('Cycle: 6 / 7');

    // Pause
    fireEvent.click(playBtn);
    expect(playBtn).toHaveTextContent('▶');
  });

  it('switches benchmark presets from dropdown', () => {
    const defaultAst = createAdd(createReg('r0'), createImm(1), 'r1');
    render(
      <HardwarePipelineWidget
        ast={defaultAst}
        domain="hardware"
        declarationName="simpleAdd"
      />
    );

    const presetSelect = screen.getByTestId('hardware-preset-select');
    expect(presetSelect).toBeInTheDocument();

    // Select Benchmark 3 (MAC with 2 stalls)
    fireEvent.change(presetSelect, {
      target: { value: 'benchmark-3-mac-mul-stall' },
    });

    const stallsKpi = screen.getByTestId('hardware-kpi-stalls');
    expect(stallsKpi).toHaveTextContent('2');

    const cyclesKpi = screen.getByTestId('hardware-kpi-cycles');
    expect(cyclesKpi).toHaveTextContent('7');
  });

  it('renders SVG DAG elements including nodes and colored hazard markers', () => {
    const ast = HARDWARE_BENCHMARK_PRESETS[4].createAST(); // Benchmark 5: RAW, WAR, WAW
    render(
      <HardwarePipelineWidget
        ast={ast}
        domain="hardware"
        declarationName="hazardSeq"
      />
    );

    const svg = screen.getByTestId('hardware-dag-svg');
    expect(svg).toBeInTheDocument();

    // Check SVG markers
    expect(svg.querySelector('#arrow-raw')).toBeInTheDocument();
    expect(svg.querySelector('#arrow-war')).toBeInTheDocument();
    expect(svg.querySelector('#arrow-waw')).toBeInTheDocument();

    // Check instruction node rendering
    expect(screen.getByTestId('dag-node-I0')).toBeInTheDocument();
    expect(screen.getByTestId('dag-node-I1')).toBeInTheDocument();
  });

  it('renders Gantt chart table with stage cells', () => {
    const ast = HARDWARE_BENCHMARK_PRESETS[0].createAST();
    render(
      <HardwarePipelineWidget
        ast={ast}
        domain="hardware"
        declarationName="arithmeticGantt"
      />
    );

    const ganttTable = screen.getByTestId('hardware-gantt-table');
    expect(ganttTable).toBeInTheDocument();

    const fetchCells = ganttTable.querySelectorAll('.stage-fetch');
    expect(fetchCells.length).toBeGreaterThanOrEqual(1);

    const execCells = ganttTable.querySelectorAll('.stage-exec');
    expect(execCells.length).toBeGreaterThanOrEqual(1);
  });
});
