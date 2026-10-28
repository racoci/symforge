import type {
  HardwareNode,
  RegHardwareNode,
  ImmHardwareNode,
} from '../../../core/ast/types';
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
import type {
  FlatInstruction,
  InstructionOp,
  HazardType,
  DependencyEdge,
  ScheduledInstruction,
  PipelineScheduleResult,
  HardwareBenchmarkPreset,
  IssueWidth,
} from './types';

/**
 * Execution latencies by instruction operation opcode.
 * - ALU operations (add, sub): 1 cycle.
 * - Bitwise shifts (shl, shr): 1 cycle.
 * - Multiplication (mul): 3 cycles.
 * - Immediate / Register loads: 1 cycle.
 */
export const OP_LATENCIES: Record<InstructionOp, number> = {
  add: 1,
  sub: 1,
  shl: 1,
  shr: 1,
  mul: 3,
  reg: 1,
  imm: 1,
  seq: 1,
  nop: 1,
};

/**
 * Creates an immediate node with an explicit destination register.
 */
export function createImmOp(val: number, dst = 'r0'): ImmHardwareNode & { dst: string } {
  return Object.freeze({ kind: 'imm', val: Math.trunc(val), dst });
}

/**
 * Creates a register load node with an explicit destination register.
 */
export function createRegOp(name: string, dst?: string): RegHardwareNode & { dst?: string } {
  return Object.freeze({ kind: 'reg', name, dst: dst ?? name });
}

interface TraversalContext {
  tempCounter: number;
  instructions: Array<Omit<FlatInstruction, 'id' | 'originalIndex'>>;
}

function extractOperand(
  operand: HardwareNode,
  ctx: TraversalContext
): { reg?: string; immVal?: number } {
  if (operand.kind === 'reg') {
    return { reg: operand.name };
  }
  if (operand.kind === 'imm') {
    return { immVal: operand.val };
  }
  // Compound nested node: flatten child first
  flattenNode(operand, ctx);
  const lastEmitted = ctx.instructions[ctx.instructions.length - 1];
  return { reg: lastEmitted.dst };
}

function flattenNode(node: HardwareNode, ctx: TraversalContext): void {
  if (node.kind === 'seq') {
    flattenNode(node.first, ctx);
    flattenNode(node.second, ctx);
    return;
  }

  if (node.kind === 'reg') {
    const dst = (node as { dst?: string }).dst ?? node.name ?? 'r0';
    ctx.instructions.push({
      op: 'reg',
      dst,
      src1: node.name,
      latency: OP_LATENCIES.reg,
    });
    return;
  }

  if (node.kind === 'imm') {
    const dst = (node as { dst?: string }).dst ?? 'r0';
    ctx.instructions.push({
      op: 'imm',
      dst,
      immVal: node.val,
      latency: OP_LATENCIES.imm,
    });
    return;
  }

  if (node.kind === 'add' || node.kind === 'sub' || node.kind === 'mul') {
    const op1 = extractOperand(node.src1, ctx);
    const op2 = extractOperand(node.src2, ctx);
    const dst = node.dst ?? `t${ctx.tempCounter++}`;
    ctx.instructions.push({
      op: node.kind,
      dst,
      src1: op1.reg,
      src2: op2.reg,
      immVal: op1.immVal ?? op2.immVal,
      latency: OP_LATENCIES[node.kind],
    });
    return;
  }

  if (node.kind === 'shl' || node.kind === 'shr') {
    const op = extractOperand(node.src, ctx);
    const dst = node.dst ?? `t${ctx.tempCounter++}`;
    ctx.instructions.push({
      op: node.kind,
      dst,
      src1: op.reg,
      immVal: node.shift,
      latency: OP_LATENCIES[node.kind],
    });
    return;
  }

  // Hole, Macro or unknown node treated as NOP
  ctx.instructions.push({
    op: 'nop',
    dst: 'r0',
    latency: OP_LATENCIES.nop,
  });
}

/**
 * Recursively flattens a HardwareNode AST into an in-order sequence of FlatInstruction records.
 */
export function flattenHardwareAST(node: HardwareNode): FlatInstruction[] {
  const ctx: TraversalContext = { tempCounter: 0, instructions: [] };
  flattenNode(node, ctx);

  return ctx.instructions.map((inst, index) => ({
    id: `I${index}`,
    op: inst.op,
    dst: inst.dst,
    src1: inst.src1,
    src2: inst.src2,
    immVal: inst.immVal,
    latency: inst.latency,
    originalIndex: index,
  }));
}

/**
 * Checks whether a directed graph is acyclic using Kahn's topological sorting algorithm.
 */
function checkAcyclic(nodes: FlatInstruction[], edges: DependencyEdge[]): boolean {
  if (nodes.length <= 1) return true;

  const inDegree = new Map<string, number>();
  const adj = new Map<string, string[]>();

  for (const node of nodes) {
    inDegree.set(node.id, 0);
    adj.set(node.id, []);
  }

  for (const edge of edges) {
    const current = inDegree.get(edge.toId) ?? 0;
    inDegree.set(edge.toId, current + 1);
    adj.get(edge.fromId)?.push(edge.toId);
  }

  const queue: string[] = [];
  for (const [id, deg] of inDegree.entries()) {
    if (deg === 0) queue.push(id);
  }

  let visited = 0;
  const loopCap = nodes.length * 2;
  let iterations = 0;

  while (queue.length > 0 && iterations < loopCap) {
    iterations++;
    const currentId = queue.shift()!;
    visited++;
    const neighbors = adj.get(currentId) ?? [];
    for (const nextId of neighbors) {
      const nextDeg = (inDegree.get(nextId) ?? 1) - 1;
      inDegree.set(nextId, nextDeg);
      if (nextDeg === 0) {
        queue.push(nextId);
      }
    }
  }

  return visited === nodes.length;
}

/**
 * Detects all register data dependencies (RAW, WAR, WAW) across an instruction sequence
 * and verifies that the resulting dependency graph is an Acyclic Directed Graph (DAG).
 */
export function buildDependencyDAG(instructions: FlatInstruction[]): {
  nodes: FlatInstruction[];
  edges: DependencyEdge[];
  isAcyclic: boolean;
} {
  const edges: DependencyEdge[] = [];
  const edgeSet = new Set<string>();

  const addEdge = (fromId: string, toId: string, reg: string, type: HazardType): void => {
    if (fromId === toId) return;
    const key = `${fromId}->${toId}:${reg}:${type}`;
    if (!edgeSet.has(key)) {
      edgeSet.add(key);
      edges.push({ fromId, toId, reg, type });
    }
  };

  for (let j = 0; j < instructions.length; j++) {
    const instJ = instructions[j];
    const readsJ: string[] = [];
    if (instJ.src1 && instJ.src1.trim() !== '') readsJ.push(instJ.src1);
    if (instJ.src2 && instJ.src2.trim() !== '') readsJ.push(instJ.src2);
    const writeJ = instJ.dst && instJ.dst.trim() !== '' ? instJ.dst : null;

    // 1. RAW (True Dependency): Instruction J reads register written by instruction I (I < J)
    for (const reg of readsJ) {
      for (let i = j - 1; i >= 0; i--) {
        const instI = instructions[i];
        if (instI.dst === reg) {
          addEdge(instI.id, instJ.id, reg, 'RAW');
          break; // Most recent write provides the live value
        }
      }
    }

    // 2. WAR & WAW: Instruction J writes register writeJ
    if (writeJ) {
      // Find the most recent write to writeJ before J (for WAW and boundary of WAR)
      let prevWriteIndex = -1;
      for (let i = j - 1; i >= 0; i--) {
        if (instructions[i].dst === writeJ) {
          prevWriteIndex = i;
          addEdge(instructions[i].id, instJ.id, writeJ, 'WAW');
          break;
        }
      }

      // WAR (Anti-dependency): Any instruction after prevWriteIndex that read writeJ
      const startSearch = Math.max(0, prevWriteIndex);
      for (let i = startSearch; i < j; i++) {
        const instI = instructions[i];
        const readsI: string[] = [];
        if (instI.src1) readsI.push(instI.src1);
        if (instI.src2) readsI.push(instI.src2);
        if (readsI.includes(writeJ)) {
          addEdge(instI.id, instJ.id, writeJ, 'WAR');
        }
      }
    }
  }

  const isAcyclic = checkAcyclic(instructions, edges);

  return { nodes: instructions, edges, isAcyclic };
}

/**
 * Computes register lifetime intervals [startCycle, endCycle] for all registers.
 */
function computeRegisterLiveRanges(
  scheduledInstructions: ScheduledInstruction[]
): Record<string, { startCycle: number; endCycle: number }> {
  const registerLiveRanges: Record<string, { startCycle: number; endCycle: number }> = {};
  const allRegisters = new Set<string>();

  for (const item of scheduledInstructions) {
    if (item.instruction.dst) allRegisters.add(item.instruction.dst);
    if (item.instruction.src1) allRegisters.add(item.instruction.src1);
    if (item.instruction.src2) allRegisters.add(item.instruction.src2);
  }

  for (const reg of allRegisters) {
    const writers = scheduledInstructions.filter((si) => si.instruction.dst === reg);
    const readers = scheduledInstructions.filter(
      (si) => si.instruction.src1 === reg || si.instruction.src2 === reg
    );

    let startCycle = 0;
    if (writers.length > 0) {
      const firstWriter = writers[0];
      const readBeforeWrite = readers.some(
        (r) => r.instruction.originalIndex < firstWriter.instruction.originalIndex
      );
      if (!readBeforeWrite) {
        startCycle = firstWriter.executeEndCycle;
      }
    }

    let endCycle = startCycle;
    if (readers.length > 0) {
      const lastReader = readers[readers.length - 1];
      endCycle = Math.max(endCycle, lastReader.executeStartCycle);
    }
    if (writers.length > 0) {
      const lastWriter = writers[writers.length - 1];
      endCycle = Math.max(endCycle, lastWriter.writebackCycle);
    }

    registerLiveRanges[reg] = {
      startCycle: Math.min(startCycle, endCycle),
      endCycle,
    };
  }

  return registerLiveRanges;
}

/**
 * Simulates in-order pipelined CPU execution (Fetch, Decode, Execute, Writeback)
 * under 1-way scalar or 2-way superscalar issue width constraints.
 */
export function schedulePipeline(
  instructions: FlatInstruction[],
  issueWidth: IssueWidth = 1
): PipelineScheduleResult {
  if (instructions.length === 0) {
    return {
      instructions: [],
      totalCycles: 0,
      cpi: 0,
      totalStalls: 0,
      edges: [],
      registerLiveRanges: {},
      isAcyclic: true,
    };
  }

  const { edges, isAcyclic } = buildDependencyDAG(instructions);
  const scheduledMap = new Map<string, ScheduledInstruction>();
  const scheduledList: ScheduledInstruction[] = [];

  // Track functional unit execute occupancy per cycle for superscalar structural hazards
  const execSlotsOccupied = new Map<number, number>();

  for (let i = 0; i < instructions.length; i++) {
    const inst = instructions[i];
    const prev = i > 0 ? scheduledList[i - 1] : null;

    // RAW producers that this instruction depends on
    const rawProducerEdges = edges.filter((e) => e.toId === inst.id && e.type === 'RAW');
    const rawProducers = rawProducerEdges
      .map((e) => scheduledMap.get(e.fromId))
      .filter((si): si is ScheduledInstruction => si !== undefined);

    let fetchCycle: number;
    let decodeCycle: number;

    if (issueWidth === 1) {
      // 1-Way In-Order Scalar
      if (!prev) {
        fetchCycle = 0;
        decodeCycle = 1;
      } else {
        decodeCycle = Math.max(prev.decodeCycle + 1, prev.executeStartCycle);
        fetchCycle = Math.max(prev.fetchCycle + 1, decodeCycle - 1);
      }
    } else {
      // 2-Way Superscalar
      if (!prev) {
        fetchCycle = 0;
        decodeCycle = 1;
      } else {
        const canDualIssue = i % 2 === 1 && !rawProducers.some((p) => p.instruction.id === prev.instruction.id);
        if (canDualIssue) {
          fetchCycle = prev.fetchCycle;
          decodeCycle = prev.decodeCycle;
        } else {
          decodeCycle = Math.max(prev.decodeCycle + 1, Math.floor(prev.executeStartCycle));
          fetchCycle = Math.max(Math.floor(i / 2), decodeCycle - 1);
        }
      }
    }

    // Earliest cycle operand results are available from execution stage forwarding
    const operandReadyCycle = rawProducers.reduce(
      (max, p) => Math.max(max, p.executeEndCycle + 1),
      0
    );

    const nominalExecStart = decodeCycle + 1;
    let executeStartCycle = Math.max(nominalExecStart, operandReadyCycle);

    // Enforce in-order execution with respect to previous instruction
    if (prev) {
      if (issueWidth === 1) {
        executeStartCycle = Math.max(executeStartCycle, prev.executeStartCycle + 1);
      } else {
        executeStartCycle = Math.max(executeStartCycle, prev.executeStartCycle);
      }
    }

    // Enforce structural issue width execution limit
    while ((execSlotsOccupied.get(executeStartCycle) ?? 0) >= issueWidth) {
      executeStartCycle++;
    }
    execSlotsOccupied.set(executeStartCycle, (execSlotsOccupied.get(executeStartCycle) ?? 0) + 1);

    const stallCycles = Math.max(0, executeStartCycle - nominalExecStart);
    const executeEndCycle = executeStartCycle + inst.latency - 1;

    let writebackCycle = executeEndCycle + 1;
    // Enforce WAW dependency: instruction must write back after the previous write to the same register
    const wawProducerEdges = edges.filter((e) => e.toId === inst.id && e.type === 'WAW');
    for (const edge of wawProducerEdges) {
      const prod = scheduledMap.get(edge.fromId);
      if (prod) {
        writebackCycle = Math.max(writebackCycle, prod.writebackCycle + 1);
      }
    }

    const scheduled: ScheduledInstruction = {
      instruction: inst,
      fetchCycle,
      decodeCycle,
      executeStartCycle,
      executeEndCycle,
      writebackCycle,
      stallCycles,
    };

    scheduledMap.set(inst.id, scheduled);
    scheduledList.push(scheduled);
  }

  const totalCycles = Math.max(...scheduledList.map((si) => si.writebackCycle)) + 1;
  const totalStalls = scheduledList.reduce((sum, si) => sum + si.stallCycles, 0);
  const cpi = Number((totalCycles / scheduledList.length).toFixed(2));
  const registerLiveRanges = computeRegisterLiveRanges(scheduledList);

  return {
    instructions: scheduledList,
    totalCycles,
    cpi,
    totalStalls,
    edges,
    registerLiveRanges,
    isAcyclic,
  };
}

/**
 * 10 Canonical Benchmark Instruction Sequences for verification and interactive exploration.
 */
export const HARDWARE_BENCHMARK_PRESETS: HardwareBenchmarkPreset[] = [
  {
    id: 'benchmark-1-arithmetic-chain',
    name: '1. Simple Arithmetic Chain',
    description: 'Cascade of dependent ALU additions and subtractions exhibiting consecutive RAW data forwarding.',
    expectedCycles1Way: 7,
    expectedStalls1Way: 0,
    createAST: () =>
      createSeq(
        createImmOp(10, 'r0'),
        createSeq(
          createAdd(createReg('r0'), createImm(5), 'r1'),
          createSeq(
            createAdd(createReg('r1'), createImm(3), 'r2'),
            createSub(createReg('r2'), createImm(1), 'r3')
          )
        )
      ),
  },
  {
    id: 'benchmark-2-multi-register-parallel',
    name: '2. Multi-Register Parallel Operations',
    description: 'Two completely independent arithmetic register tracks suitable for dual-issue superscalar concurrency.',
    expectedCycles1Way: 9,
    expectedStalls1Way: 0,
    createAST: () =>
      createSeq(
        createImmOp(1, 'r0'),
        createSeq(
          createImmOp(2, 'r1'),
          createSeq(
            createAdd(createReg('r0'), createReg('r1'), 'r2'),
            createSeq(
              createImmOp(3, 'r3'),
              createSeq(
                createImmOp(4, 'r4'),
                createAdd(createReg('r3'), createReg('r4'), 'r5')
              )
            )
          )
        )
      ),
  },
  {
    id: 'benchmark-3-mac-mul-stall',
    name: '3. Multiplier-Accumulator (MAC Stall)',
    description: 'Multiply-Accumulate sequence where a 3-cycle multiplication introduces 2 stall cycles on subsequent dependent ADD.',
    expectedCycles1Way: 7,
    expectedStalls1Way: 2,
    createAST: () =>
      createSeq(
        createMul(createReg('r1'), createReg('r2'), 'r0'),
        createAdd(createReg('r0'), createReg('r4'), 'r3')
      ),
  },
  {
    id: 'benchmark-4-shift-and-add',
    name: '4. Bitwise Shift and Add',
    description: 'Combined logical shifts and addition demonstrating split and recombine register paths.',
    expectedCycles1Way: 6,
    expectedStalls1Way: 0,
    createAST: () =>
      createSeq(
        createShl(createReg('r0'), 2, 'r1'),
        createSeq(
          createShr(createReg('r0'), 1, 'r2'),
          createAdd(createReg('r1'), createReg('r2'), 'r3')
        )
      ),
  },
  {
    id: 'benchmark-5-register-pressure-reuse',
    name: '5. Register Pressure & Reuse (WAW & WAR)',
    description: 'Register reuse pattern generating explicit Read-After-Write, Write-After-Read, and Write-After-Write hazards.',
    expectedCycles1Way: 9,
    expectedStalls1Way: 0,
    createAST: () =>
      createSeq(
        createAdd(createReg('r1'), createReg('r2'), 'r0'),
        createSeq(
          createAdd(createReg('r0'), createReg('r4'), 'r3'),
          createSeq(
            createSub(createReg('r5'), createReg('r6'), 'r0'),
            createMul(createReg('r0'), createReg('r8'), 'r7')
          )
        )
      ),
  },
  {
    id: 'benchmark-6-interleaved-independent',
    name: '6. Interleaved Independent Instructions',
    description: 'Compiler instruction scheduling hiding 3-cycle multiplication latency via interleaved independent ALU operations.',
    expectedCycles1Way: 7,
    expectedStalls1Way: 0,
    createAST: () =>
      createSeq(
        createMul(createReg('r1'), createReg('r2'), 'r0'),
        createSeq(
          createAdd(createReg('r6'), createReg('r7'), 'r5'),
          createSeq(
            createSub(createReg('r9'), createReg('r10'), 'r8'),
            createAdd(createReg('r0'), createReg('r4'), 'r3')
          )
        )
      ),
  },
  {
    id: 'benchmark-7-long-raw-chain',
    name: '7. Long RAW Dependency Chain',
    description: 'Eight-stage linear dependency pipeline cascade verifying induction through deep forwarding chains.',
    expectedCycles1Way: 11,
    expectedStalls1Way: 0,
    createAST: () =>
      createSeq(
        createAdd(createReg('r0'), createImm(1), 'r1'),
        createSeq(
          createAdd(createReg('r1'), createImm(1), 'r2'),
          createSeq(
            createAdd(createReg('r2'), createImm(1), 'r3'),
            createSeq(
              createAdd(createReg('r3'), createImm(1), 'r4'),
              createSeq(
                createAdd(createReg('r4'), createImm(1), 'r5'),
                createSeq(
                  createAdd(createReg('r5'), createImm(1), 'r6'),
                  createSeq(
                    createAdd(createReg('r6'), createImm(1), 'r7'),
                    createAdd(createReg('r7'), createImm(1), 'r8')
                  )
                )
              )
            )
          )
        )
      ),
  },
  {
    id: 'benchmark-8-register-renaming',
    name: '8. Register Renaming Benchmark',
    description: 'Highlights false name dependencies (WAR/WAW) that can be eliminated through register renaming.',
    expectedCycles1Way: 9,
    expectedStalls1Way: 2,
    createAST: () =>
      createSeq(
        createAdd(createReg('r1'), createReg('r2'), 'r0'),
        createSeq(
          createAdd(createReg('r0'), createReg('r4'), 'r3'),
          createSeq(
            createMul(createReg('r5'), createReg('r6'), 'r0'),
            createAdd(createReg('r0'), createReg('r8'), 'r7')
          )
        )
      ),
  },
  {
    id: 'benchmark-9-diamond-topology-dag',
    name: '9. Diamond Topology DAG',
    description: 'Symmetric diamond dependency graph branching from r0 and re-converging at an accumulator.',
    expectedCycles1Way: 7,
    expectedStalls1Way: 0,
    createAST: () =>
      createSeq(
        createAdd(createReg('r1'), createReg('r2'), 'r0'),
        createSeq(
          createShl(createReg('r0'), 1, 'r3'),
          createSeq(
            createShr(createReg('r0'), 1, 'r4'),
            createAdd(createReg('r3'), createReg('r4'), 'r5')
          )
        )
      ),
  },
  {
    id: 'benchmark-10-deep-complex-sequence',
    name: '10. Deep Complex Instruction Sequence',
    description: 'Full-featured 12-instruction sequence combining arithmetic, multiplication, shifts, register reuse, and joins.',
    expectedCycles1Way: 16,
    expectedStalls1Way: 1,
    createAST: () =>
      createSeq(
        createImmOp(10, 'r0'),
        createSeq(
          createImmOp(20, 'r1'),
          createSeq(
            createMul(createReg('r0'), createReg('r1'), 'r2'),
            createSeq(
              createAdd(createReg('r4'), createReg('r5'), 'r3'),
              createSeq(
                createShl(createReg('r3'), 2, 'r6'),
                createSeq(
                  createAdd(createReg('r2'), createReg('r6'), 'r7'),
                  createSeq(
                    createSub(createReg('r7'), createReg('r1'), 'r8'),
                    createSeq(
                      createShr(createReg('r8'), 1, 'r0'),
                      createSeq(
                        createMul(createReg('r0'), createReg('r3'), 'r9'),
                        createSeq(
                          createAdd(createReg('r11'), createReg('r12'), 'r10'),
                          createSeq(
                            createAdd(createReg('r9'), createReg('r10'), 'r13'),
                            createSub(createReg('r13'), createReg('r7'), 'r14')
                          )
                        )
                      )
                    )
                  )
                )
              )
            )
          )
        )
      ),
  },
];
