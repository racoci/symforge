import { describe, it, expect } from 'vitest';
import {
  flattenHardwareAST,
  buildDependencyDAG,
  schedulePipeline,
  HARDWARE_BENCHMARK_PRESETS,
  OP_LATENCIES,
} from './hardwarePipeline';
import {
  createReg,
  createImm,
  createAdd,
  createSub,
  createMul,
  createShl,
  createShr,
  createSeq,
} from '../../../core/ast/models';
import type { FlatInstruction } from './types';

describe('Hardware Pipeline Analysis Engine (Task 2.6)', () => {
  describe('AST Flattening and Opcode Identification', () => {
    it('flattens simple register and immediate instructions', () => {
      const regNode = createReg('r1');
      const immNode = createImm(42);
      const seqNode = createSeq(regNode, immNode);

      const flat = flattenHardwareAST(seqNode);
      expect(flat).toHaveLength(2);

      expect(flat[0].op).toBe('reg');
      expect(flat[0].src1).toBe('r1');
      expect(flat[0].dst).toBe('r1');
      expect(flat[0].latency).toBe(OP_LATENCIES.reg);

      expect(flat[1].op).toBe('imm');
      expect(flat[1].immVal).toBe(42);
      expect(flat[1].latency).toBe(OP_LATENCIES.imm);
    });

    it('flattens ALU, multiplication, and shift operations with correct latencies', () => {
      const addNode = createAdd(createReg('r0'), createReg('r1'), 'r2');
      const subNode = createSub(createReg('r2'), createReg('r3'), 'r4');
      const mulNode = createMul(createReg('r4'), createReg('r5'), 'r6');
      const shlNode = createShl(createReg('r6'), 2, 'r7');
      const shrNode = createShr(createReg('r7'), 1, 'r8');

      const program = createSeq(
        addNode,
        createSeq(subNode, createSeq(mulNode, createSeq(shlNode, shrNode)))
      );

      const flat = flattenHardwareAST(program);
      expect(flat).toHaveLength(5);

      expect(flat[0].op).toBe('add');
      expect(flat[0].latency).toBe(1);
      expect(flat[0].dst).toBe('r2');
      expect(flat[0].src1).toBe('r0');
      expect(flat[0].src2).toBe('r1');

      expect(flat[1].op).toBe('sub');
      expect(flat[1].latency).toBe(1);

      expect(flat[2].op).toBe('mul');
      expect(flat[2].latency).toBe(3);

      expect(flat[3].op).toBe('shl');
      expect(flat[3].latency).toBe(1);
      expect(flat[3].immVal).toBe(2);

      expect(flat[4].op).toBe('shr');
      expect(flat[4].latency).toBe(1);
      expect(flat[4].immVal).toBe(1);
    });

    it('flattens nested compound expressions in post-order', () => {
      const nested = createAdd(
        createMul(createReg('r0'), createReg('r1'), 't0'),
        createReg('r2'),
        'out'
      );

      const flat = flattenHardwareAST(nested);
      expect(flat).toHaveLength(2);
      expect(flat[0].op).toBe('mul');
      expect(flat[0].dst).toBe('t0');
      expect(flat[1].op).toBe('add');
      expect(flat[1].src1).toBe('t0');
      expect(flat[1].src2).toBe('r2');
      expect(flat[1].dst).toBe('out');
    });
  });

  describe('Dependency Detection (RAW, WAR, WAW)', () => {
    it('detects RAW (Read-After-Write, True) dependency', () => {
      const instructions: FlatInstruction[] = [
        { id: 'I0', op: 'add', dst: 'r1', src1: 'r2', src2: 'r3', latency: 1, originalIndex: 0 },
        { id: 'I1', op: 'sub', dst: 'r4', src1: 'r1', src2: 'r5', latency: 1, originalIndex: 1 },
      ];

      const { edges, isAcyclic } = buildDependencyDAG(instructions);
      expect(isAcyclic).toBe(true);

      const rawEdge = edges.find((e) => e.type === 'RAW');
      expect(rawEdge).toBeDefined();
      expect(rawEdge).toEqual({
        fromId: 'I0',
        toId: 'I1',
        reg: 'r1',
        type: 'RAW',
      });
    });

    it('detects WAR (Write-After-Read, Anti) dependency', () => {
      const instructions: FlatInstruction[] = [
        { id: 'I0', op: 'add', dst: 'r2', src1: 'r0', src2: 'r1', latency: 1, originalIndex: 0 },
        { id: 'I1', op: 'sub', dst: 'r0', src1: 'r3', src2: 'r4', latency: 1, originalIndex: 1 },
      ];

      const { edges, isAcyclic } = buildDependencyDAG(instructions);
      expect(isAcyclic).toBe(true);

      const warEdge = edges.find((e) => e.type === 'WAR');
      expect(warEdge).toBeDefined();
      expect(warEdge).toEqual({
        fromId: 'I0',
        toId: 'I1',
        reg: 'r0',
        type: 'WAR',
      });
    });

    it('detects WAW (Write-After-Write, Output) dependency', () => {
      const instructions: FlatInstruction[] = [
        { id: 'I0', op: 'add', dst: 'r0', src1: 'r1', src2: 'r2', latency: 1, originalIndex: 0 },
        { id: 'I1', op: 'mul', dst: 'r0', src1: 'r3', src2: 'r4', latency: 3, originalIndex: 1 },
      ];

      const { edges, isAcyclic } = buildDependencyDAG(instructions);
      expect(isAcyclic).toBe(true);

      const wawEdge = edges.find((e) => e.type === 'WAW');
      expect(wawEdge).toBeDefined();
      expect(wawEdge).toEqual({
        fromId: 'I0',
        toId: 'I1',
        reg: 'r0',
        type: 'WAW',
      });
    });

    it('correctly identifies simultaneous RAW, WAR, and WAW in register reuse pattern', () => {
      // Benchmark 5 pattern
      const ast = HARDWARE_BENCHMARK_PRESETS.find(
        (p) => p.id === 'benchmark-5-register-pressure-reuse'
      )!.createAST();
      const instructions = flattenHardwareAST(ast);
      const { edges, isAcyclic } = buildDependencyDAG(instructions);

      expect(isAcyclic).toBe(true);

      const rawEdges = edges.filter((e) => e.type === 'RAW');
      const warEdges = edges.filter((e) => e.type === 'WAR');
      const wawEdges = edges.filter((e) => e.type === 'WAW');

      expect(rawEdges.length).toBeGreaterThanOrEqual(2);
      expect(warEdges.length).toBeGreaterThanOrEqual(1);
      expect(wawEdges.length).toBeGreaterThanOrEqual(1);

      // I0 writes r0, I1 reads r0 -> RAW
      expect(rawEdges).toContainEqual({ fromId: 'I0', toId: 'I1', reg: 'r0', type: 'RAW' });
      // I1 reads r0, I2 writes r0 -> WAR
      expect(warEdges).toContainEqual({ fromId: 'I1', toId: 'I2', reg: 'r0', type: 'WAR' });
      // I0 writes r0, I2 writes r0 -> WAW
      expect(wawEdges).toContainEqual({ fromId: 'I0', toId: 'I2', reg: 'r0', type: 'WAW' });
    });
  });

  describe('Pipeline Simulation & Hazard Scheduling', () => {
    it('verifies multiplication introduces exactly 2 stall cycles on subsequent dependent ADD', () => {
      // Benchmark 3: MAC stall
      // I0: mul r1, r2 -> r0 (latency 3)
      // I1: add r0, r4 -> r3 (latency 1)
      const ast = HARDWARE_BENCHMARK_PRESETS.find(
        (p) => p.id === 'benchmark-3-mac-mul-stall'
      )!.createAST();
      const instructions = flattenHardwareAST(ast);
      const schedule = schedulePipeline(instructions, 1);

      expect(schedule.instructions).toHaveLength(2);

      const i0 = schedule.instructions[0];
      const i1 = schedule.instructions[1];

      // I0 (MUL, latency 3):
      // Fetch: 0, Decode: 1, Exec: 2..4, WB: 5
      expect(i0.fetchCycle).toBe(0);
      expect(i0.decodeCycle).toBe(1);
      expect(i0.executeStartCycle).toBe(2);
      expect(i0.executeEndCycle).toBe(4);
      expect(i0.writebackCycle).toBe(5);
      expect(i0.stallCycles).toBe(0);

      // I1 (ADD dependent on I0):
      // Fetch: 1, Decode: 2, Stalls: 2 cycles (cycles 3 and 4), Exec: 5, WB: 6
      expect(i1.fetchCycle).toBe(1);
      expect(i1.decodeCycle).toBe(2);
      expect(i1.stallCycles).toBe(2); // MUST BE EXACTLY 2 STALL CYCLES
      expect(i1.executeStartCycle).toBe(5);
      expect(i1.executeEndCycle).toBe(5);
      expect(i1.writebackCycle).toBe(6);

      expect(schedule.totalCycles).toBe(7);
      expect(schedule.totalStalls).toBe(2);
      expect(schedule.cpi).toBe(3.5);
    });

    it('verifies independent ALU instructions experience zero stalls with data forwarding', () => {
      const ast = HARDWARE_BENCHMARK_PRESETS.find(
        (p) => p.id === 'benchmark-1-arithmetic-chain'
      )!.createAST();
      const instructions = flattenHardwareAST(ast);
      const schedule = schedulePipeline(instructions, 1);

      expect(schedule.instructions).toHaveLength(4);
      expect(schedule.totalStalls).toBe(0);
      expect(schedule.totalCycles).toBe(7);
      expect(schedule.cpi).toBe(1.75);
    });

    it('verifies interleaved instructions hide multiplication latency', () => {
      // Benchmark 6
      const ast = HARDWARE_BENCHMARK_PRESETS.find(
        (p) => p.id === 'benchmark-6-interleaved-independent'
      )!.createAST();
      const instructions = flattenHardwareAST(ast);
      const schedule = schedulePipeline(instructions, 1);

      expect(schedule.instructions).toHaveLength(4);
      // Even though I3 depends on I0 (MUL), 2 intervening instructions (I1, I2) completely hide the latency
      expect(schedule.totalStalls).toBe(0);
      expect(schedule.totalCycles).toBe(7);
    });

    it('handles 2-way superscalar dual-issue scheduling', () => {
      const ast = HARDWARE_BENCHMARK_PRESETS.find(
        (p) => p.id === 'benchmark-2-multi-register-parallel'
      )!.createAST();
      const instructions = flattenHardwareAST(ast);
      const schedule2Way = schedulePipeline(instructions, 2);

      expect(schedule2Way.instructions).toHaveLength(6);
      // In 2-way superscalar, independent instructions can issue in parallel pairs
      expect(schedule2Way.totalCycles).toBeLessThan(9);
      expect(schedule2Way.cpi).toBeLessThan(1.5);
    });
  });

  describe('Register Live Range Calculation', () => {
    it('calculates lifetime intervals for all defined and referenced registers', () => {
      const ast = HARDWARE_BENCHMARK_PRESETS.find(
        (p) => p.id === 'benchmark-3-mac-mul-stall'
      )!.createAST();
      const instructions = flattenHardwareAST(ast);
      const schedule = schedulePipeline(instructions, 1);

      const ranges = schedule.registerLiveRanges;
      expect(ranges).toBeDefined();

      // Registers involved: r1, r2, r0, r4, r3
      expect(ranges.r1).toBeDefined();
      expect(ranges.r2).toBeDefined();
      expect(ranges.r0).toBeDefined();
      expect(ranges.r4).toBeDefined();
      expect(ranges.r3).toBeDefined();

      // Input registers start at cycle 0
      expect(ranges.r1.startCycle).toBe(0);
      expect(ranges.r2.startCycle).toBe(0);
      expect(ranges.r4.startCycle).toBe(0);

      // r0 is defined by I0 (MUL) and consumed by I1 (ADD)
      expect(ranges.r0.startCycle).toBeLessThanOrEqual(ranges.r0.endCycle);
      expect(ranges.r0.endCycle).toBeGreaterThanOrEqual(schedule.instructions[1].executeStartCycle);
    });
  });

  describe('Pass-to-Complete Gate: 10 Canonical Benchmark Sequences', () => {
    it('evaluates all 10 canonical benchmark instruction sequences', () => {
      expect(HARDWARE_BENCHMARK_PRESETS).toHaveLength(10);

      for (let i = 0; i < HARDWARE_BENCHMARK_PRESETS.length; i++) {
        const preset = HARDWARE_BENCHMARK_PRESETS[i];
        const ast = preset.createAST();
        const instructions = flattenHardwareAST(ast);

        expect(instructions.length).toBeGreaterThan(0);

        const schedule = schedulePipeline(instructions, 1);

        // Core Invariant 1: Dependency Graph MUST be acyclic (DAG)
        expect(schedule.isAcyclic).toBe(true);

        // Core Invariant 2: Total cycles must equal analytical expectation
        expect(schedule.totalCycles).toBe(preset.expectedCycles1Way);

        // Core Invariant 3: Stall count must equal analytical expectation
        expect(schedule.totalStalls).toBe(preset.expectedStalls1Way);

        // Invariant 4: CPI must be positive and non-zero
        expect(schedule.cpi).toBeGreaterThan(0);

        // Invariant 5: Live ranges computed for all referenced registers
        const regKeys = Object.keys(schedule.registerLiveRanges);
        expect(regKeys.length).toBeGreaterThan(0);
        for (const reg of regKeys) {
          const range = schedule.registerLiveRanges[reg];
          expect(range.startCycle).toBeLessThanOrEqual(range.endCycle);
        }
      }
    });
  });
});
