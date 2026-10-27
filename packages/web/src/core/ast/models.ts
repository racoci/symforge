/**
 * Client-Side SymForge AST Core Models, Constructors, and Structural Utilities.
 *
 * Implements immutable functional data structures with structural equality,
 * parameter extraction, cloning, and type guards.
 */

import type {
  FreeParam,
  AudioNode,
  SineAudioNode,
  SawAudioNode,
  SquareAudioNode,
  NoiseAudioNode,
  ADSRAudioNode,
  LowPassAudioNode,
  GainAudioNode,
  MixAudioNode,
  FMAudioNode,
  HoleAudioNode,
  MacroAudioNode,
  SDFNode,
  SphereSDFNode,
  BoxSDFNode,
  CylinderSDFNode,
  TorusSDFNode,
  TranslateSDFNode,
  RotateSDFNode,
  ScaleSDFNode,
  UnionSDFNode,
  SmoothUnionSDFNode,
  IntersectionSDFNode,
  DifferenceSDFNode,
  HoleSDFNode,
  MacroSDFNode,
  HardwareNode,
  RegHardwareNode,
  ImmHardwareNode,
  AddHardwareNode,
  SubHardwareNode,
  MulHardwareNode,
  ShlHardwareNode,
  ShrHardwareNode,
  SeqHardwareNode,
  HoleHardwareNode,
  MacroHardwareNode,
  UniversalNode,
  UniversalAudioNode,
  UniversalSDFNode,
  UniversalHardwareNode,
  UniversalMacroNode,
  MacroCallNode,
  HoleNode,
  ASTNode,
} from './types';

// ==========================================
// 1. Factory Helper Functions
// ==========================================

export function createFreeParam(
  id: string,
  minVal: number,
  maxVal: number,
  defaultVal: number
): FreeParam {
  return Object.freeze({
    id,
    minVal: Number(minVal),
    maxVal: Number(maxVal),
    defaultVal: Number(defaultVal),
  });
}

// Audio Factories
export function createSine(freq: FreeParam, phase?: FreeParam): SineAudioNode {
  const resolvedPhase = phase ?? createFreeParam(`${freq.id}_phase`, 0.0, 6.2831853, 0.0);
  return Object.freeze({ kind: 'sine', freq, phase: resolvedPhase });
}

export function createSaw(freq: FreeParam): SawAudioNode {
  return Object.freeze({ kind: 'saw', freq });
}

export function createSquare(freq: FreeParam, duty?: FreeParam): SquareAudioNode {
  const resolvedDuty = duty ?? createFreeParam(`${freq.id}_duty`, 0.01, 0.99, 0.5);
  return Object.freeze({ kind: 'square', freq, duty: resolvedDuty });
}

export function createNoise(seed = 42): NoiseAudioNode {
  return Object.freeze({ kind: 'noise', seed: Math.floor(seed) });
}

export function createADSR(
  attack: FreeParam,
  decay: FreeParam,
  sustain: FreeParam,
  release: FreeParam,
  gate: AudioNode
): ADSRAudioNode {
  return Object.freeze({ kind: 'adsr', attack, decay, sustain, release, gate });
}

export function createLowPass(
  cutoff: FreeParam,
  resonance: FreeParam,
  input: AudioNode
): LowPassAudioNode {
  return Object.freeze({ kind: 'lowpass', cutoff, resonance, input });
}

export function createGain(gain: FreeParam, input: AudioNode): GainAudioNode {
  return Object.freeze({ kind: 'gain', gain, input });
}

export function createMix(
  ratio: FreeParam,
  input1: AudioNode,
  input2: AudioNode
): MixAudioNode {
  return Object.freeze({ kind: 'mix', ratio, input1, input2 });
}

export function createFM(
  carrier: FreeParam,
  modFreq: FreeParam,
  modIndex: FreeParam
): FMAudioNode {
  return Object.freeze({ kind: 'fm', carrier, modFreq, modIndex });
}

export function createAudioHole(holeId: string): HoleAudioNode {
  return Object.freeze({ kind: 'hole', holeId, domain: 'audio' });
}

export function createAudioMacro(
  macroName: string,
  args: Record<string, any>
): MacroAudioNode {
  return Object.freeze({ kind: 'macro', macroName, arguments: Object.freeze({ ...args }), domain: 'audio' });
}

// SDF Factories
export function createSphere(radius: FreeParam): SphereSDFNode {
  return Object.freeze({ kind: 'sphere', radius });
}

export function createBox(
  width: FreeParam,
  height: FreeParam,
  depth: FreeParam
): BoxSDFNode {
  return Object.freeze({ kind: 'box', width, height, depth });
}

export function createCylinder(radius: FreeParam, height: FreeParam): CylinderSDFNode {
  return Object.freeze({ kind: 'cylinder', radius, height });
}

export function createTorus(majorRadius: FreeParam, minorRadius: FreeParam): TorusSDFNode {
  return Object.freeze({ kind: 'torus', majorRadius, minorRadius });
}

export function createTranslate(
  dx: FreeParam,
  dy: FreeParam,
  dz: FreeParam,
  child: SDFNode
): TranslateSDFNode {
  return Object.freeze({ kind: 'translate', dx, dy, dz, child });
}

export function createRotate(
  rx: FreeParam,
  ry: FreeParam,
  rz: FreeParam,
  child: SDFNode
): RotateSDFNode {
  return Object.freeze({ kind: 'rotate', rx, ry, rz, child });
}

export function createScale(s: FreeParam, child: SDFNode): ScaleSDFNode {
  return Object.freeze({ kind: 'scale', s, child });
}

export function createUnion(
  smoothness: FreeParam,
  a: SDFNode,
  b: SDFNode
): UnionSDFNode {
  return Object.freeze({ kind: 'union', smoothness, a, b });
}

export function createSmoothUnion(
  smoothness: FreeParam,
  a: SDFNode,
  b: SDFNode
): SmoothUnionSDFNode {
  return Object.freeze({ kind: 'smooth_union', smoothness, a, b });
}

export function createIntersection(a: SDFNode, b: SDFNode): IntersectionSDFNode {
  return Object.freeze({ kind: 'intersection', a, b });
}

export function createDifference(a: SDFNode, b: SDFNode): DifferenceSDFNode {
  return Object.freeze({ kind: 'difference', a, b });
}

export function createSDFHole(holeId: string): HoleSDFNode {
  return Object.freeze({ kind: 'hole', holeId, domain: 'sdf' });
}

export function createSDFMacro(
  macroName: string,
  args: Record<string, any>
): MacroSDFNode {
  return Object.freeze({ kind: 'macro', macroName, arguments: Object.freeze({ ...args }), domain: 'sdf' });
}

// Hardware Factories
export function createReg(name: string, regIdx?: number): RegHardwareNode {
  return Object.freeze({ kind: 'reg', name, regIdx });
}

export function createImm(val: number): ImmHardwareNode {
  return Object.freeze({ kind: 'imm', val: Math.trunc(val) });
}

export function createAdd(
  src1: HardwareNode,
  src2: HardwareNode,
  dst = 'out'
): AddHardwareNode {
  return Object.freeze({ kind: 'add', dst, src1, src2 });
}

export function createSub(
  src1: HardwareNode,
  src2: HardwareNode,
  dst = 'out'
): SubHardwareNode {
  return Object.freeze({ kind: 'sub', dst, src1, src2 });
}

export function createMul(
  src1: HardwareNode,
  src2: HardwareNode,
  dst = 'out'
): MulHardwareNode {
  return Object.freeze({ kind: 'mul', dst, src1, src2 });
}

export function createShl(
  src: HardwareNode,
  shift: number,
  dst = 'out'
): ShlHardwareNode {
  return Object.freeze({ kind: 'shl', dst, src, shift: Math.trunc(shift) });
}

export function createShr(
  src: HardwareNode,
  shift: number,
  dst = 'out'
): ShrHardwareNode {
  return Object.freeze({ kind: 'shr', dst, src, shift: Math.trunc(shift) });
}

export function createSeq(
  first: HardwareNode,
  second: HardwareNode
): SeqHardwareNode {
  return Object.freeze({ kind: 'seq', first, second });
}

export function createHardwareHole(holeId: string): HoleHardwareNode {
  return Object.freeze({ kind: 'hole', holeId, domain: 'hardware' });
}

export function createHardwareMacro(
  macroName: string,
  args: Record<string, any>
): MacroHardwareNode {
  return Object.freeze({ kind: 'macro', macroName, arguments: Object.freeze({ ...args }), domain: 'hardware' });
}

// Universal / Generic Factories
export function createUniversalAudio(node: AudioNode): UniversalAudioNode {
  return Object.freeze({ kind: 'universal_audio', node });
}

export function createUniversalSDF(node: SDFNode): UniversalSDFNode {
  return Object.freeze({ kind: 'universal_sdf', node });
}

export function createUniversalHardware(node: HardwareNode): UniversalHardwareNode {
  return Object.freeze({ kind: 'universal_hardware', node });
}

export function createUniversalMacro(
  name: string,
  children: readonly UniversalNode[]
): UniversalMacroNode {
  return Object.freeze({ kind: 'universal_macro', name, children: Object.freeze([...children]) });
}

export function createHole(holeId: string, domain?: string): HoleNode {
  return Object.freeze({ kind: 'hole', holeId, domain });
}

export function createMacroCall(
  macroName: string,
  args: Record<string, any>,
  domain = 'generic'
): MacroCallNode {
  return Object.freeze({ kind: 'macro', macroName, arguments: Object.freeze({ ...args }), domain });
}

// ==========================================
// 2. Structural Utilities
// ==========================================

export function isFreeParam(val: unknown): val is FreeParam {
  if (!val || typeof val !== 'object') return false;
  const p = val as Record<string, unknown>;
  return (
    typeof p.id === 'string' &&
    typeof p.minVal === 'number' &&
    typeof p.maxVal === 'number' &&
    typeof p.defaultVal === 'number'
  );
}

export function isAudioNode(node: ASTNode): node is AudioNode {
  const audioKinds = ['sine', 'saw', 'square', 'noise', 'adsr', 'lowpass', 'gain', 'mix', 'fm'];
  if (audioKinds.includes(node.kind)) return true;
  if (node.kind === 'hole' && (node as HoleNode).domain === 'audio') return true;
  if (node.kind === 'macro' && (node as MacroCallNode).domain === 'audio') return true;
  return false;
}

export function isSDFNode(node: ASTNode): node is SDFNode {
  const sdfKinds = ['sphere', 'box', 'cylinder', 'torus', 'translate', 'rotate', 'scale', 'union', 'smooth_union', 'intersection', 'difference'];
  if (sdfKinds.includes(node.kind)) return true;
  if (node.kind === 'hole' && (node as HoleNode).domain === 'sdf') return true;
  if (node.kind === 'macro' && (node as MacroCallNode).domain === 'sdf') return true;
  return false;
}

export function isHardwareNode(node: ASTNode): node is HardwareNode {
  const hwKinds = ['reg', 'imm', 'add', 'sub', 'mul', 'shl', 'shr', 'seq'];
  if (hwKinds.includes(node.kind)) return true;
  if (node.kind === 'hole' && (node as HoleNode).domain === 'hardware') return true;
  if (node.kind === 'macro' && (node as MacroCallNode).domain === 'hardware') return true;
  return false;
}

export function isUniversalNode(node: ASTNode): node is UniversalNode {
  return (
    node.kind === 'universal_audio' ||
    node.kind === 'universal_sdf' ||
    node.kind === 'universal_hardware' ||
    node.kind === 'universal_macro'
  );
}

/**
 * Returns direct children of an AST node.
 */
export function getChildren(node: ASTNode): ASTNode[] {
  switch (node.kind) {
    // Audio composite
    case 'adsr':
      return [node.gate];
    case 'lowpass':
    case 'gain':
      return [node.input];
    case 'mix':
      return [node.input1, node.input2];

    // SDF composite
    case 'translate':
    case 'rotate':
    case 'scale':
      return [node.child];
    case 'union':
    case 'smooth_union':
    case 'intersection':
    case 'difference':
      return [node.a, node.b];

    // Hardware composite
    case 'add':
    case 'sub':
    case 'mul':
      return [node.src1, node.src2];
    case 'shl':
    case 'shr':
      return [node.src];
    case 'seq':
      return [node.first, node.second];

    // Universal container
    case 'universal_audio':
      return [node.node];
    case 'universal_sdf':
      return [node.node];
    case 'universal_hardware':
      return [node.node];
    case 'universal_macro':
      return [...node.children];

    // Macro with embedded ASTNode arguments
    case 'macro': {
      const nested: ASTNode[] = [];
      for (const val of Object.values(node.arguments)) {
        if (val && typeof val === 'object' && 'kind' in val) {
          nested.push(val as ASTNode);
        }
      }
      return nested;
    }

    default:
      return [];
  }
}

/**
 * Extract all direct FreeParams declared on the node itself.
 */
function getDirectParams(node: ASTNode): FreeParam[] {
  switch (node.kind) {
    case 'sine':
      return [node.freq, node.phase];
    case 'saw':
      return [node.freq];
    case 'square':
      return [node.freq, node.duty];
    case 'adsr':
      return [node.attack, node.decay, node.sustain, node.release];
    case 'lowpass':
      return [node.cutoff, node.resonance];
    case 'gain':
      return [node.gain];
    case 'mix':
      return [node.ratio];
    case 'fm':
      return [node.carrier, node.modFreq, node.modIndex];
    case 'sphere':
      return [node.radius];
    case 'box':
      return [node.width, node.height, node.depth];
    case 'cylinder':
      return [node.radius, node.height];
    case 'torus':
      return [node.majorRadius, node.minorRadius];
    case 'translate':
      return [node.dx, node.dy, node.dz];
    case 'rotate':
      return [node.rx, node.ry, node.rz];
    case 'scale':
      return [node.s];
    case 'union':
    case 'smooth_union':
      return [node.smoothness];
    case 'macro': {
      const direct: FreeParam[] = [];
      for (const val of Object.values(node.arguments)) {
        if (isFreeParam(val)) direct.push(val);
      }
      return direct;
    }
    default:
      return [];
  }
}

/**
 * Collect all unique free parameters across the AST in deterministic DFS order.
 */
export function collectFreeParams(node: ASTNode): FreeParam[] {
  const result: FreeParam[] = [];
  const seenIds = new Set<string>();

  function traverse(current: ASTNode): void {
    const direct = getDirectParams(current);
    for (const p of direct) {
      if (!seenIds.has(p.id)) {
        seenIds.add(p.id);
        result.push(p);
      }
    }
    const children = getChildren(current);
    for (const child of children) {
      traverse(child);
    }
  }

  traverse(node);
  return result;
}

/**
 * Alias for collectFreeParams for backward/symmetrical compatibility.
 */
export const extractFreeParams = collectFreeParams;

/**
 * Calculates 1-indexed maximum tree depth.
 */
export function getTreeDepth(node: ASTNode): number {
  const children = getChildren(node);
  if (children.length === 0) return 1;
  const maxChildDepth = Math.max(...children.map(getTreeDepth));
  return 1 + maxChildDepth;
}

/**
 * Counts total AST vertices in the tree.
 */
export function getNodeCount(node: ASTNode): number {
  const children = getChildren(node);
  let count = 1;
  for (const child of children) {
    count += getNodeCount(child);
  }
  return count;
}

/**
 * Clone a FreeParam, updating defaultVal if present in paramValues.
 */
function cloneFreeParam(p: FreeParam, paramValues: Record<string, number>): FreeParam {
  if (Object.prototype.hasOwnProperty.call(paramValues, p.id)) {
    return createFreeParam(p.id, p.minVal, p.maxVal, paramValues[p.id]);
  }
  return p;
}

/**
 * Clone an AudioNode with updated parameter values.
 */
function cloneAudioNode(node: AudioNode, paramValues: Record<string, number>): AudioNode {
  switch (node.kind) {
    case 'sine':
      return createSine(cloneFreeParam(node.freq, paramValues), cloneFreeParam(node.phase, paramValues));
    case 'saw':
      return createSaw(cloneFreeParam(node.freq, paramValues));
    case 'square':
      return createSquare(cloneFreeParam(node.freq, paramValues), cloneFreeParam(node.duty, paramValues));
    case 'noise':
      return node;
    case 'adsr':
      return createADSR(
        cloneFreeParam(node.attack, paramValues),
        cloneFreeParam(node.decay, paramValues),
        cloneFreeParam(node.sustain, paramValues),
        cloneFreeParam(node.release, paramValues),
        cloneAudioNode(node.gate, paramValues)
      );
    case 'lowpass':
      return createLowPass(
        cloneFreeParam(node.cutoff, paramValues),
        cloneFreeParam(node.resonance, paramValues),
        cloneAudioNode(node.input, paramValues)
      );
    case 'gain':
      return createGain(cloneFreeParam(node.gain, paramValues), cloneAudioNode(node.input, paramValues));
    case 'mix':
      return createMix(
        cloneFreeParam(node.ratio, paramValues),
        cloneAudioNode(node.input1, paramValues),
        cloneAudioNode(node.input2, paramValues)
      );
    case 'fm':
      return createFM(
        cloneFreeParam(node.carrier, paramValues),
        cloneFreeParam(node.modFreq, paramValues),
        cloneFreeParam(node.modIndex, paramValues)
      );
    case 'hole':
      return node;
    case 'macro': {
      const clonedArgs: Record<string, any> = {};
      for (const [k, v] of Object.entries(node.arguments)) {
        if (isFreeParam(v)) clonedArgs[k] = cloneFreeParam(v, paramValues);
        else if (v && typeof v === 'object' && 'kind' in v) clonedArgs[k] = cloneWithParams(v as ASTNode, paramValues);
        else clonedArgs[k] = v;
      }
      return createAudioMacro(node.macroName, clonedArgs);
    }
  }
}

/**
 * Clone an SDFNode with updated parameter values.
 */
function cloneSDFNode(node: SDFNode, paramValues: Record<string, number>): SDFNode {
  switch (node.kind) {
    case 'sphere':
      return createSphere(cloneFreeParam(node.radius, paramValues));
    case 'box':
      return createBox(
        cloneFreeParam(node.width, paramValues),
        cloneFreeParam(node.height, paramValues),
        cloneFreeParam(node.depth, paramValues)
      );
    case 'cylinder':
      return createCylinder(cloneFreeParam(node.radius, paramValues), cloneFreeParam(node.height, paramValues));
    case 'torus':
      return createTorus(
        cloneFreeParam(node.majorRadius, paramValues),
        cloneFreeParam(node.minorRadius, paramValues)
      );
    case 'translate':
      return createTranslate(
        cloneFreeParam(node.dx, paramValues),
        cloneFreeParam(node.dy, paramValues),
        cloneFreeParam(node.dz, paramValues),
        cloneSDFNode(node.child, paramValues)
      );
    case 'rotate':
      return createRotate(
        cloneFreeParam(node.rx, paramValues),
        cloneFreeParam(node.ry, paramValues),
        cloneFreeParam(node.rz, paramValues),
        cloneSDFNode(node.child, paramValues)
      );
    case 'scale':
      return createScale(cloneFreeParam(node.s, paramValues), cloneSDFNode(node.child, paramValues));
    case 'union':
      return createUnion(
        cloneFreeParam(node.smoothness, paramValues),
        cloneSDFNode(node.a, paramValues),
        cloneSDFNode(node.b, paramValues)
      );
    case 'smooth_union':
      return createSmoothUnion(
        cloneFreeParam(node.smoothness, paramValues),
        cloneSDFNode(node.a, paramValues),
        cloneSDFNode(node.b, paramValues)
      );
    case 'intersection':
      return createIntersection(cloneSDFNode(node.a, paramValues), cloneSDFNode(node.b, paramValues));
    case 'difference':
      return createDifference(cloneSDFNode(node.a, paramValues), cloneSDFNode(node.b, paramValues));
    case 'hole':
      return node;
    case 'macro': {
      const clonedArgs: Record<string, any> = {};
      for (const [k, v] of Object.entries(node.arguments)) {
        if (isFreeParam(v)) clonedArgs[k] = cloneFreeParam(v, paramValues);
        else if (v && typeof v === 'object' && 'kind' in v) clonedArgs[k] = cloneWithParams(v as ASTNode, paramValues);
        else clonedArgs[k] = v;
      }
      return createSDFMacro(node.macroName, clonedArgs);
    }
  }
}

/**
 * Clone a HardwareNode with updated parameter values.
 */
function cloneHardwareNode(node: HardwareNode, paramValues: Record<string, number>): HardwareNode {
  switch (node.kind) {
    case 'reg':
    case 'imm':
    case 'hole':
      return node;
    case 'add':
      return createAdd(cloneHardwareNode(node.src1, paramValues), cloneHardwareNode(node.src2, paramValues), node.dst);
    case 'sub':
      return createSub(cloneHardwareNode(node.src1, paramValues), cloneHardwareNode(node.src2, paramValues), node.dst);
    case 'mul':
      return createMul(cloneHardwareNode(node.src1, paramValues), cloneHardwareNode(node.src2, paramValues), node.dst);
    case 'shl':
      return createShl(cloneHardwareNode(node.src, paramValues), node.shift, node.dst);
    case 'shr':
      return createShr(cloneHardwareNode(node.src, paramValues), node.shift, node.dst);
    case 'seq':
      return createSeq(cloneHardwareNode(node.first, paramValues), cloneHardwareNode(node.second, paramValues));
    case 'macro': {
      const clonedArgs: Record<string, any> = {};
      for (const [k, v] of Object.entries(node.arguments)) {
        if (isFreeParam(v)) clonedArgs[k] = cloneFreeParam(v, paramValues);
        else if (v && typeof v === 'object' && 'kind' in v) clonedArgs[k] = cloneWithParams(v as ASTNode, paramValues);
        else clonedArgs[k] = v;
      }
      return createHardwareMacro(node.macroName, clonedArgs);
    }
  }
}

/**
 * Returns a clone of the AST with updated default values for continuous FreeParams.
 */
export function cloneWithParams<T extends ASTNode>(node: T, paramValues: Record<string, number>): T {
  const n = node as ASTNode;
  switch (n.kind) {
    case 'sine':
    case 'saw':
    case 'square':
    case 'noise':
    case 'adsr':
    case 'lowpass':
    case 'gain':
    case 'mix':
    case 'fm':
      return cloneAudioNode(n, paramValues) as unknown as T;

    case 'sphere':
    case 'box':
    case 'cylinder':
    case 'torus':
    case 'translate':
    case 'rotate':
    case 'scale':
    case 'union':
    case 'smooth_union':
    case 'intersection':
    case 'difference':
      return cloneSDFNode(n, paramValues) as unknown as T;

    case 'reg':
    case 'imm':
    case 'add':
    case 'sub':
    case 'mul':
    case 'shl':
    case 'shr':
    case 'seq':
      return cloneHardwareNode(n, paramValues) as unknown as T;

    case 'universal_audio':
      return createUniversalAudio(cloneAudioNode(n.node, paramValues)) as unknown as T;
    case 'universal_sdf':
      return createUniversalSDF(cloneSDFNode(n.node, paramValues)) as unknown as T;
    case 'universal_hardware':
      return createUniversalHardware(cloneHardwareNode(n.node, paramValues)) as unknown as T;
    case 'universal_macro':
      return createUniversalMacro(
        n.name,
        n.children.map((child) => cloneWithParams(child, paramValues) as UniversalNode)
      ) as unknown as T;

    case 'hole':
      return node;

    case 'macro': {
      const clonedArgs: Record<string, any> = {};
      for (const [k, v] of Object.entries(n.arguments)) {
        if (isFreeParam(v)) clonedArgs[k] = cloneFreeParam(v, paramValues);
        else if (v && typeof v === 'object' && 'kind' in v) clonedArgs[k] = cloneWithParams(v as ASTNode, paramValues);
        else clonedArgs[k] = v;
      }
      return createMacroCall(n.macroName, clonedArgs, n.domain) as unknown as T;
    }
  }
}

/**
 * Helper to check equality of two FreeParams within floating point precision.
 */
export function areFreeParamsEqual(a: FreeParam, b: FreeParam, epsilon = 1e-6): boolean {
  return (
    a.id === b.id &&
    Math.abs(a.minVal - b.minVal) <= epsilon &&
    Math.abs(a.maxVal - b.maxVal) <= epsilon &&
    Math.abs(a.defaultVal - b.defaultVal) <= epsilon
  );
}

/**
 * Deep structural equality between two AST nodes.
 */
export function areNodesEqual(a: ASTNode, b: ASTNode): boolean {
  if (a === b) return true;
  if (!a || !b || a.kind !== b.kind) return false;

  const aParams = getDirectParams(a);
  const bParams = getDirectParams(b);
  if (aParams.length !== bParams.length) return false;
  for (let i = 0; i < aParams.length; i++) {
    if (!areFreeParamsEqual(aParams[i], bParams[i])) return false;
  }

  // Node-specific property checks
  switch (a.kind) {
    case 'noise':
      return a.seed === (b as NoiseAudioNode).seed;
    case 'hole':
      return a.holeId === (b as HoleNode).holeId;
    case 'reg':
      return a.name === (b as RegHardwareNode).name;
    case 'imm':
      return a.val === (b as ImmHardwareNode).val;
    case 'shl':
    case 'shr':
      if (a.shift !== (b as ShlHardwareNode | ShrHardwareNode).shift) return false;
      break;
    case 'macro': {
      const bMacro = b as MacroCallNode;
      if (a.macroName !== bMacro.macroName || a.domain !== bMacro.domain) return false;
      const aKeys = Object.keys(a.arguments);
      const bKeys = Object.keys(bMacro.arguments);
      if (aKeys.length !== bKeys.length) return false;
      for (const k of aKeys) {
        const valA = a.arguments[k];
        const valB = bMacro.arguments[k];
        if (isFreeParam(valA) && isFreeParam(valB)) {
          if (!areFreeParamsEqual(valA, valB)) return false;
        } else if (valA && typeof valA === 'object' && 'kind' in valA && valB && typeof valB === 'object' && 'kind' in valB) {
          if (!areNodesEqual(valA as ASTNode, valB as ASTNode)) return false;
        } else if (valA !== valB) {
          return false;
        }
      }
      break;
    }
  }

  const aChildren = getChildren(a);
  const bChildren = getChildren(b);
  if (aChildren.length !== bChildren.length) return false;
  for (let i = 0; i < aChildren.length; i++) {
    if (!areNodesEqual(aChildren[i], bChildren[i])) return false;
  }

  return true;
}
