/**
 * Core Abstract Syntax Tree (AST) Type Definitions for SymForge Client Engine.
 *
 * Symmetrically mirrors the Lean 4 formal inductive specifications in:
 * `packages/lean/symforge_ir/SymforgeIr/Types.lean`
 */

/**
 * Continuous free parameter subject to gradient optimization or user modulation.
 */
export interface FreeParam {
  readonly id: string;
  readonly minVal: number;
  readonly maxVal: number;
  readonly defaultVal: number;
}

// ==========================================
// 1. Procedural Audio AST Nodes
// ==========================================

export interface SineAudioNode {
  readonly kind: 'sine';
  readonly freq: FreeParam;
  readonly phase: FreeParam;
}

export interface SawAudioNode {
  readonly kind: 'saw';
  readonly freq: FreeParam;
}

export interface SquareAudioNode {
  readonly kind: 'square';
  readonly freq: FreeParam;
  readonly duty: FreeParam;
}

export interface NoiseAudioNode {
  readonly kind: 'noise';
  readonly seed: number;
}

export interface ADSRAudioNode {
  readonly kind: 'adsr';
  readonly attack: FreeParam;
  readonly decay: FreeParam;
  readonly sustain: FreeParam;
  readonly release: FreeParam;
  readonly gate: AudioNode;
}

export interface LowPassAudioNode {
  readonly kind: 'lowpass';
  readonly cutoff: FreeParam;
  readonly resonance: FreeParam;
  readonly input: AudioNode;
}

export interface GainAudioNode {
  readonly kind: 'gain';
  readonly gain: FreeParam;
  readonly input: AudioNode;
}

export interface MixAudioNode {
  readonly kind: 'mix';
  readonly ratio: FreeParam;
  readonly input1: AudioNode;
  readonly input2: AudioNode;
}

export interface FMAudioNode {
  readonly kind: 'fm';
  readonly carrier: FreeParam;
  readonly modFreq: FreeParam;
  readonly modIndex: FreeParam;
}

export interface HoleAudioNode {
  readonly kind: 'hole';
  readonly holeId: string;
  readonly domain?: 'audio';
}

export interface MacroAudioNode {
  readonly kind: 'macro';
  readonly macroName: string;
  readonly arguments: Record<string, any>;
  readonly domain: 'audio' | string;
}

export type AudioNode =
  | SineAudioNode
  | SawAudioNode
  | SquareAudioNode
  | NoiseAudioNode
  | ADSRAudioNode
  | LowPassAudioNode
  | GainAudioNode
  | MixAudioNode
  | FMAudioNode
  | HoleAudioNode
  | MacroAudioNode;

// ==========================================
// 2. 3D Signed Distance Field (SDF) AST Nodes
// ==========================================

export interface SphereSDFNode {
  readonly kind: 'sphere';
  readonly radius: FreeParam;
}

export interface BoxSDFNode {
  readonly kind: 'box';
  readonly width: FreeParam;
  readonly height: FreeParam;
  readonly depth: FreeParam;
}

export interface CylinderSDFNode {
  readonly kind: 'cylinder';
  readonly radius: FreeParam;
  readonly height: FreeParam;
}

export interface TorusSDFNode {
  readonly kind: 'torus';
  readonly majorRadius: FreeParam;
  readonly minorRadius: FreeParam;
}

export interface PlaneSDFNode {
  readonly kind: 'plane';
  readonly offset: FreeParam;
  readonly normal?: { readonly x: number; readonly y: number; readonly z: number };
}

export interface TranslateSDFNode {
  readonly kind: 'translate';
  readonly dx: FreeParam;
  readonly dy: FreeParam;
  readonly dz: FreeParam;
  readonly child: SDFNode;
}

export interface RotateSDFNode {
  readonly kind: 'rotate';
  readonly rx: FreeParam;
  readonly ry: FreeParam;
  readonly rz: FreeParam;
  readonly child: SDFNode;
}

export interface ScaleSDFNode {
  readonly kind: 'scale';
  readonly s: FreeParam;
  readonly child: SDFNode;
}

export interface UnionSDFNode {
  readonly kind: 'union';
  readonly smoothness: FreeParam;
  readonly a: SDFNode;
  readonly b: SDFNode;
}

export interface SmoothUnionSDFNode {
  readonly kind: 'smooth_union';
  readonly smoothness: FreeParam;
  readonly a: SDFNode;
  readonly b: SDFNode;
}

export interface IntersectionSDFNode {
  readonly kind: 'intersection';
  readonly a: SDFNode;
  readonly b: SDFNode;
}

export interface DifferenceSDFNode {
  readonly kind: 'difference';
  readonly a: SDFNode;
  readonly b: SDFNode;
}

export interface HoleSDFNode {
  readonly kind: 'hole';
  readonly holeId: string;
  readonly domain?: 'sdf';
}

export interface MacroSDFNode {
  readonly kind: 'macro';
  readonly macroName: string;
  readonly arguments: Record<string, any>;
  readonly domain: 'sdf' | string;
}

export type SDFNode =
  | SphereSDFNode
  | BoxSDFNode
  | CylinderSDFNode
  | TorusSDFNode
  | PlaneSDFNode
  | TranslateSDFNode
  | RotateSDFNode
  | ScaleSDFNode
  | UnionSDFNode
  | SmoothUnionSDFNode
  | IntersectionSDFNode
  | DifferenceSDFNode
  | HoleSDFNode
  | MacroSDFNode;

// ==========================================
// 3. Hardware Superoptimization AST Nodes
// ==========================================

export interface RegHardwareNode {
  readonly kind: 'reg';
  readonly name: string;
  readonly regIdx?: number;
}

export interface ImmHardwareNode {
  readonly kind: 'imm';
  readonly val: number;
}

export interface AddHardwareNode {
  readonly kind: 'add';
  readonly dst?: string;
  readonly src1: HardwareNode;
  readonly src2: HardwareNode;
}

export interface SubHardwareNode {
  readonly kind: 'sub';
  readonly dst?: string;
  readonly src1: HardwareNode;
  readonly src2: HardwareNode;
}

export interface MulHardwareNode {
  readonly kind: 'mul';
  readonly dst?: string;
  readonly src1: HardwareNode;
  readonly src2: HardwareNode;
}

export interface ShlHardwareNode {
  readonly kind: 'shl';
  readonly dst?: string;
  readonly src: HardwareNode;
  readonly shift: number;
}

export interface ShrHardwareNode {
  readonly kind: 'shr';
  readonly dst?: string;
  readonly src: HardwareNode;
  readonly shift: number;
}

export interface SeqHardwareNode {
  readonly kind: 'seq';
  readonly first: HardwareNode;
  readonly second: HardwareNode;
}

export interface HoleHardwareNode {
  readonly kind: 'hole';
  readonly holeId: string;
  readonly domain?: 'hardware';
}

export interface MacroHardwareNode {
  readonly kind: 'macro';
  readonly macroName: string;
  readonly arguments: Record<string, any>;
  readonly domain: 'hardware' | string;
}

export type HardwareNode =
  | RegHardwareNode
  | ImmHardwareNode
  | AddHardwareNode
  | SubHardwareNode
  | MulHardwareNode
  | ShlHardwareNode
  | ShrHardwareNode
  | SeqHardwareNode
  | HoleHardwareNode
  | MacroHardwareNode;

// ==========================================
// 4. Universal AST Nodes & Generic Containers
// ==========================================

export interface UniversalAudioNode {
  readonly kind: 'universal_audio';
  readonly node: AudioNode;
}

export interface UniversalSDFNode {
  readonly kind: 'universal_sdf';
  readonly node: SDFNode;
}

export interface UniversalHardwareNode {
  readonly kind: 'universal_hardware';
  readonly node: HardwareNode;
}

export interface UniversalMacroNode {
  readonly kind: 'universal_macro';
  readonly name: string;
  readonly children: readonly UniversalNode[];
}

export type UniversalNode =
  | UniversalAudioNode
  | UniversalSDFNode
  | UniversalHardwareNode
  | UniversalMacroNode;

export interface MacroCallNode {
  readonly kind: 'macro';
  readonly macroName: string;
  readonly arguments: Record<string, any>;
  readonly domain: string;
}

export interface HoleNode {
  readonly kind: 'hole';
  readonly holeId: string;
  readonly domain?: string;
}

/**
 * Top-level union representing any valid SymForge client AST node.
 */
export type ASTNode =
  | AudioNode
  | SDFNode
  | HardwareNode
  | UniversalNode
  | MacroCallNode
  | HoleNode;
