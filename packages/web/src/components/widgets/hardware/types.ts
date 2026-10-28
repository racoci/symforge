import type { HardwareNode } from '../../../core/ast/types';

/**
 * Microarchitectural instruction operation codes supported by the pipeline engine.
 */
export type InstructionOp =
  | 'reg'
  | 'imm'
  | 'add'
  | 'sub'
  | 'mul'
  | 'shl'
  | 'shr'
  | 'seq'
  | 'nop';

/**
 * Normalized flat assembly instruction representation extracted from HardwareNode AST.
 */
export interface FlatInstruction {
  readonly id: string;
  readonly op: InstructionOp;
  readonly dst: string;
  readonly src1?: string;
  readonly src2?: string;
  readonly immVal?: number;
  readonly latency: number;
  readonly originalIndex: number;
}

/**
 * Pipeline hazard classification types.
 * - RAW: Read-After-Write (True dependency)
 * - WAR: Write-After-Read (Anti-dependency)
 * - WAW: Write-After-Write (Output dependency)
 */
export type HazardType = 'RAW' | 'WAR' | 'WAW';

/**
 * Directed dependency edge between two instructions in the execution DAG.
 */
export interface DependencyEdge {
  readonly fromId: string;
  readonly toId: string;
  readonly reg: string;
  readonly type: HazardType;
}

/**
 * Scheduled instruction with calculated clock cycle execution timings.
 */
export interface ScheduledInstruction {
  readonly instruction: FlatInstruction;
  readonly fetchCycle: number;
  readonly decodeCycle: number;
  readonly executeStartCycle: number;
  readonly executeEndCycle: number;
  readonly writebackCycle: number;
  readonly stallCycles: number;
}

/**
 * Complete result of pipeline scheduling simulation.
 */
export interface PipelineScheduleResult {
  readonly instructions: ScheduledInstruction[];
  readonly totalCycles: number;
  readonly cpi: number;
  readonly totalStalls: number;
  readonly edges: DependencyEdge[];
  readonly registerLiveRanges: Record<string, { startCycle: number; endCycle: number }>;
  readonly isAcyclic: boolean;
}

/**
 * Available interactive viewport layout modes.
 */
export type HardwareViewMode = 'dag' | 'gantt' | 'liverange' | 'split';

/**
 * Microarchitecture issue width mode (1-way scalar vs 2-way superscalar).
 */
export type IssueWidth = 1 | 2;

/**
 * Canonical benchmark sequence definition.
 */
export interface HardwareBenchmarkPreset {
  readonly id: string;
  readonly name: string;
  readonly description: string;
  readonly expectedCycles1Way: number;
  readonly expectedStalls1Way: number;
  readonly createAST: () => HardwareNode;
}
