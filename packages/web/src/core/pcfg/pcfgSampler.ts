/**
 * Client-Side Contextual PCFG/PHOG Sampler.
 *
 * Implements probabilistic context-free grammar and probabilistic higher-order grammar
 * sampling across 5 curriculum complexity strata (L1..L5) with contextual conditioning,
 * parameter generation, and derivation safety bounds.
 */

import type { ASTNode, AudioNode, FreeParam, HardwareNode, SDFNode } from '../ast/types';
import {
  createFreeParam,
  createSphere,
  createBox,
  createCylinder,
  createTorus,
  createPlane,
  createTranslate,
  createRotate,
  createScale,
  createUnion,
  createSmoothUnion,
  createIntersection,
  createDifference,
  createSine,
  createSaw,
  createSquare,
  createNoise,
  createADSR,
  createLowPass,
  createGain,
  createMix,
  createFM,
  createReg,
  createImm,
  createAdd,
  createSub,
  createMul,
  createShl,
  createShr,
  createSeq,
  createUniversalSDF,
  createUniversalAudio,
  createUniversalHardware,
  getTreeDepth,
  getNodeCount,
  isSDFNode,
  isAudioNode,
  isHardwareNode,
} from '../ast/models';
import type { DomainType, PunctureResult, StructuralContext } from '../generator/types';
import { resolveExpectedType } from '../generator/holePuncturing';
import { formatLeanCode } from '../parser/leanPrinter';
import type {
  CurriculumLevel,
  GrammarProductionRule,
  SamplerOptions,
  VariationCandidate,
} from './types';

// ==========================================
// 1. Deterministic PRNG (Mulberry32)
// ==========================================

export function createPRNG(seed?: number): () => number {
  let s = ((seed !== undefined ? seed : 1337) >>> 0) || 1;
  return function next(): number {
    s = (s + 0x6d2b79f5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return (t >>> 0) / 4294967296;
  };
}

// ==========================================
// 2. Base Grammar Production Rules
// ==========================================

function getSDFRules(): GrammarProductionRule[] {
  return [
    { nonTerminal: 'SDF_Term', constructorName: 'sphere', probability: 0.25, childNonTerminals: [], isTerminal: true, domain: 'sdf' },
    { nonTerminal: 'SDF_Term', constructorName: 'box', probability: 0.25, childNonTerminals: [], isTerminal: true, domain: 'sdf' },
    { nonTerminal: 'SDF_Term', constructorName: 'cylinder', probability: 0.20, childNonTerminals: [], isTerminal: true, domain: 'sdf' },
    { nonTerminal: 'SDF_Term', constructorName: 'torus', probability: 0.15, childNonTerminals: [], isTerminal: true, domain: 'sdf' },
    { nonTerminal: 'SDF_Term', constructorName: 'plane', probability: 0.15, childNonTerminals: [], isTerminal: true, domain: 'sdf' },
    { nonTerminal: 'SDF_Unary', constructorName: 'translate', probability: 0.50, childNonTerminals: ['SDF'], isTerminal: false, domain: 'sdf' },
    { nonTerminal: 'SDF_Unary', constructorName: 'rotate', probability: 0.30, childNonTerminals: ['SDF'], isTerminal: false, domain: 'sdf' },
    { nonTerminal: 'SDF_Unary', constructorName: 'scale', probability: 0.20, childNonTerminals: ['SDF'], isTerminal: false, domain: 'sdf' },
    { nonTerminal: 'SDF_Binary', constructorName: 'union', probability: 0.35, childNonTerminals: ['SDF', 'SDF'], isTerminal: false, domain: 'sdf' },
    { nonTerminal: 'SDF_Binary', constructorName: 'smooth_union', probability: 0.20, childNonTerminals: ['SDF', 'SDF'], isTerminal: false, domain: 'sdf' },
    { nonTerminal: 'SDF_Binary', constructorName: 'intersection', probability: 0.20, childNonTerminals: ['SDF', 'SDF'], isTerminal: false, domain: 'sdf' },
    { nonTerminal: 'SDF_Binary', constructorName: 'difference', probability: 0.25, childNonTerminals: ['SDF', 'SDF'], isTerminal: false, domain: 'sdf' },
  ];
}

function getAudioRules(): GrammarProductionRule[] {
  return [
    { nonTerminal: 'Audio_Term', constructorName: 'sine', probability: 0.25, childNonTerminals: [], isTerminal: true, domain: 'audio' },
    { nonTerminal: 'Audio_Term', constructorName: 'saw', probability: 0.25, childNonTerminals: [], isTerminal: true, domain: 'audio' },
    { nonTerminal: 'Audio_Term', constructorName: 'square', probability: 0.20, childNonTerminals: [], isTerminal: true, domain: 'audio' },
    { nonTerminal: 'Audio_Term', constructorName: 'noise', probability: 0.15, childNonTerminals: [], isTerminal: true, domain: 'audio' },
    { nonTerminal: 'Audio_Term', constructorName: 'fm', probability: 0.15, childNonTerminals: [], isTerminal: true, domain: 'audio' },
    { nonTerminal: 'Audio_Unary', constructorName: 'lowpass', probability: 0.40, childNonTerminals: ['Audio'], isTerminal: false, domain: 'audio' },
    { nonTerminal: 'Audio_Unary', constructorName: 'gain', probability: 0.35, childNonTerminals: ['Audio'], isTerminal: false, domain: 'audio' },
    { nonTerminal: 'Audio_Unary', constructorName: 'adsr', probability: 0.25, childNonTerminals: ['Audio'], isTerminal: false, domain: 'audio' },
    { nonTerminal: 'Audio_Binary', constructorName: 'mix', probability: 1.0, childNonTerminals: ['Audio', 'Audio'], isTerminal: false, domain: 'audio' },
  ];
}

function getHardwareRules(): GrammarProductionRule[] {
  return [
    { nonTerminal: 'HW_Term', constructorName: 'reg', probability: 0.60, childNonTerminals: [], isTerminal: true, domain: 'hardware' },
    { nonTerminal: 'HW_Term', constructorName: 'imm', probability: 0.40, childNonTerminals: [], isTerminal: true, domain: 'hardware' },
    { nonTerminal: 'HW_Unary', constructorName: 'shl', probability: 0.50, childNonTerminals: ['Hardware'], isTerminal: false, domain: 'hardware' },
    { nonTerminal: 'HW_Unary', constructorName: 'shr', probability: 0.50, childNonTerminals: ['Hardware'], isTerminal: false, domain: 'hardware' },
    { nonTerminal: 'HW_Binary', constructorName: 'add', probability: 0.30, childNonTerminals: ['Hardware', 'Hardware'], isTerminal: false, domain: 'hardware' },
    { nonTerminal: 'HW_Binary', constructorName: 'sub', probability: 0.25, childNonTerminals: ['Hardware', 'Hardware'], isTerminal: false, domain: 'hardware' },
    { nonTerminal: 'HW_Binary', constructorName: 'mul', probability: 0.20, childNonTerminals: ['Hardware', 'Hardware'], isTerminal: false, domain: 'hardware' },
    { nonTerminal: 'HW_Binary', constructorName: 'seq', probability: 0.25, childNonTerminals: ['Hardware', 'Hardware'], isTerminal: false, domain: 'hardware' },
  ];
}

export function getGrammarRules(domain: DomainType = 'sdf'): GrammarProductionRule[] {
  if (domain === 'audio') return getAudioRules();
  if (domain === 'hardware') return getHardwareRules();
  return getSDFRules();
}

// ==========================================
// 3. Contextual PHOG Re-Weighting Matrix
// ==========================================

function getSDFMultipliers(context: StructuralContext): Record<string, number> {
  const mults: Record<string, number> = {};
  if (context.parentOp === 'difference' && context.parentRelation === 'b') {
    mults.cylinder = 3.0;
    mults.box = 2.0;
    mults.translate = 2.5;
  } else if (context.parentOp === 'difference' && context.parentRelation === 'a') {
    mults.box = 2.0;
    mults.sphere = 1.5;
  } else if (context.parentOp === 'union' || context.parentOp === 'smooth_union') {
    mults.sphere = 1.5;
    mults.box = 1.5;
    mults.translate = 1.5;
  }
  return mults;
}

function getAudioMultipliers(context: StructuralContext): Record<string, number> {
  const mults: Record<string, number> = {};
  if (context.parentOp === 'lowpass') {
    mults.saw = 3.0;
    mults.square = 2.5;
    mults.noise = 2.0;
    mults.sine = 0.5;
  } else if (context.parentOp === 'gain') {
    mults.saw = 1.5;
    mults.sine = 1.5;
  } else if (context.parentOp === 'mix') {
    mults.sine = 1.5;
    mults.saw = 1.5;
    mults.lowpass = 1.5;
  }
  return mults;
}

function getHardwareMultipliers(context: StructuralContext): Record<string, number> {
  const mults: Record<string, number> = {};
  if (context.parentOp === 'seq') {
    mults.add = 2.0;
    mults.sub = 1.8;
    mults.mul = 1.5;
    mults.reg = 2.0;
  }
  return mults;
}

function normalizeRuleGroup(rules: GrammarProductionRule[]): GrammarProductionRule[] {
  const total = rules.reduce((acc, r) => acc + r.probability, 0);
  if (total <= 0) return rules;
  return rules.map((r) => ({
    ...r,
    probability: Number((r.probability / total).toFixed(4)),
  }));
}

export function getConditionedRules(
  domain: DomainType,
  context: StructuralContext
): GrammarProductionRule[] {
  const baseRules = getGrammarRules(domain);
  let multipliers: Record<string, number> = {};

  if (domain === 'sdf') {
    multipliers = getSDFMultipliers(context);
  } else if (domain === 'audio') {
    multipliers = getAudioMultipliers(context);
  } else if (domain === 'hardware') {
    multipliers = getHardwareMultipliers(context);
  }

  const rawWeighted = baseRules.map((rule) => {
    const factor = multipliers[rule.constructorName] ?? 1.0;
    return { ...rule, probability: rule.probability * factor };
  });

  const byNonTerminal = new Map<string, GrammarProductionRule[]>();
  for (const r of rawWeighted) {
    const group = byNonTerminal.get(r.nonTerminal) ?? [];
    group.push(r);
    byNonTerminal.set(r.nonTerminal, group);
  }

  const normalized: GrammarProductionRule[] = [];
  for (const group of byNonTerminal.values()) {
    normalized.push(...normalizeRuleGroup(group));
  }

  return normalized;
}

function selectRule(
  rules: GrammarProductionRule[],
  nonTerminal: string,
  rng: () => number
): GrammarProductionRule {
  const matching = rules.filter((r) => r.nonTerminal === nonTerminal);
  if (matching.length === 0) return rules[0];

  const total = matching.reduce((acc, r) => acc + r.probability, 0);
  const roll = rng() * total;
  let cum = 0;
  for (const rule of matching) {
    cum += rule.probability;
    if (roll <= cum) return rule;
  }
  return matching[matching.length - 1];
}

// ==========================================
// 4. Parameter Generator
// ==========================================

let globalParamSequence = 0;

function nextParamId(prefix: string): string {
  globalParamSequence += 1;
  return `${prefix}_${globalParamSequence}`;
}

function randomFloat(min: number, max: number, rng: () => number, decimals = 2): number {
  const val = min + rng() * (max - min);
  const factor = 10 ** decimals;
  return Math.round(val * factor) / factor;
}

export function sampleFreeParam(
  paramType: string,
  minVal: number,
  maxVal: number,
  rng: () => number,
  defaultVal?: number,
  ctx?: DerivationContext
): FreeParam {
  const id = ctx ? ctx.nextParamId(paramType) : nextParamId(paramType);
  const def = defaultVal !== undefined ? defaultVal : randomFloat(minVal, maxVal, rng, 2);
  return createFreeParam(id, minVal, maxVal, def);
}

// ==========================================
// 5. Stratified Derivation Engine
// ==========================================

class DerivationContext {
  public steps = 0;
  public paramCounter = 0;
  constructor(
    public readonly maxDepth: number,
    public readonly maxSteps: number,
    public readonly rng: () => number,
    public readonly rules: GrammarProductionRule[],
    public readonly structuralContext: StructuralContext,
    public readonly paramPrefix = 'p'
  ) {}

  public nextParamId(name: string): string {
    this.paramCounter += 1;
    return `${this.paramPrefix}_${name}_${this.paramCounter}`;
  }

  public checkSafety(currentDepth: number): boolean {
    this.steps += 1;
    return currentDepth < this.maxDepth && this.steps < this.maxSteps;
  }
}

// --- SDF Domain Derivations ---

function sampleL1SDF(ctx: DerivationContext): SDFNode {
  const rule = selectRule(ctx.rules, 'SDF_Term', ctx.rng);
  switch (rule.constructorName) {
    case 'box':
      return createBox(
        sampleFreeParam('w', 0.5, 5.0, ctx.rng, 2.0, ctx),
        sampleFreeParam('h', 0.5, 5.0, ctx.rng, 2.0, ctx),
        sampleFreeParam('d', 0.5, 5.0, ctx.rng, 2.0, ctx)
      );
    case 'cylinder':
      return createCylinder(
        sampleFreeParam('r', 0.2, 3.0, ctx.rng, 1.0, ctx),
        sampleFreeParam('h', 0.5, 6.0, ctx.rng, 3.0, ctx)
      );
    case 'torus':
      return createTorus(
        sampleFreeParam('majorR', 1.0, 5.0, ctx.rng, 2.5, ctx),
        sampleFreeParam('minorR', 0.1, 1.0, ctx.rng, 0.4, ctx)
      );
    case 'plane':
      return createPlane(sampleFreeParam('offset', -2.0, 2.0, ctx.rng, 0.0, ctx));
    case 'sphere':
    default:
      return createSphere(sampleFreeParam('radius', 0.5, 4.0, ctx.rng, 1.5, ctx));
  }
}

function sampleL2SDF(ctx: DerivationContext): SDFNode {
  const child = sampleL1SDF(ctx);
  const rule = selectRule(ctx.rules, 'SDF_Unary', ctx.rng);
  switch (rule.constructorName) {
    case 'rotate':
      return createRotate(
        sampleFreeParam('rx', 0.0, 3.14, ctx.rng, 0.0, ctx),
        sampleFreeParam('ry', 0.0, 3.14, ctx.rng, 0.78, ctx),
        sampleFreeParam('rz', 0.0, 3.14, ctx.rng, 0.0, ctx),
        child
      );
    case 'scale':
      return createScale(sampleFreeParam('s', 0.2, 3.0, ctx.rng, 1.2, ctx), child);
    case 'translate':
    default:
      return createTranslate(
        sampleFreeParam('dx', -3.0, 3.0, ctx.rng, 1.0, ctx),
        sampleFreeParam('dy', -3.0, 3.0, ctx.rng, 0.0, ctx),
        sampleFreeParam('dz', -3.0, 3.0, ctx.rng, 0.0, ctx),
        child
      );
  }
}

function makeSDFBinary(ctx: DerivationContext, a: SDFNode, b: SDFNode): SDFNode {
  const rule = selectRule(ctx.rules, 'SDF_Binary', ctx.rng);
  switch (rule.constructorName) {
    case 'difference':
      return createDifference(a, b);
    case 'intersection':
      return createIntersection(a, b);
    case 'smooth_union':
      return createSmoothUnion(sampleFreeParam('smoothness', 0.05, 0.8, ctx.rng, 0.2, ctx), a, b);
    case 'union':
    default:
      return createUnion(sampleFreeParam('smoothness', 0.0, 0.5, ctx.rng, 0.0, ctx), a, b);
  }
}

function sampleL3SDF(ctx: DerivationContext): SDFNode {
  const childA = ctx.rng() > 0.5 ? sampleL2SDF(ctx) : sampleL1SDF(ctx);
  const childB = sampleL1SDF(ctx);
  return makeSDFBinary(ctx, childA, childB);
}

function sampleL4SDF(ctx: DerivationContext): SDFNode {
  const childA = sampleL2SDF(ctx);
  const childB = ctx.rng() > 0.4 ? sampleL2SDF(ctx) : sampleL1SDF(ctx);
  return makeSDFBinary(ctx, childA, childB);
}

function sampleL5SDF(ctx: DerivationContext): SDFNode {
  const childA = sampleL4SDF(ctx);
  const childB = sampleL3SDF(ctx);
  const combined = makeSDFBinary(ctx, childA, childB);
  if (ctx.rng() > 0.5 && ctx.checkSafety(5)) {
    return createTranslate(
      sampleFreeParam('dx', -2.0, 2.0, ctx.rng, 0.5, ctx),
      sampleFreeParam('dy', -2.0, 2.0, ctx.rng, 0.0, ctx),
      sampleFreeParam('dz', -2.0, 2.0, ctx.rng, 0.0, ctx),
      combined
    );
  }
  return combined;
}

// --- Audio Domain Derivations ---

function sampleL1Audio(ctx: DerivationContext): AudioNode {
  const rule = selectRule(ctx.rules, 'Audio_Term', ctx.rng);
  switch (rule.constructorName) {
    case 'saw':
      return createSaw(sampleFreeParam('freq', 55.0, 880.0, ctx.rng, 220.0, ctx));
    case 'square':
      return createSquare(
        sampleFreeParam('freq', 55.0, 880.0, ctx.rng, 110.0, ctx),
        sampleFreeParam('duty', 0.1, 0.9, ctx.rng, 0.5, ctx)
      );
    case 'noise':
      return createNoise(Math.floor(ctx.rng() * 10000));
    case 'fm':
      return createFM(
        sampleFreeParam('carrier', 110.0, 880.0, ctx.rng, 440.0, ctx),
        sampleFreeParam('modFreq', 20.0, 440.0, ctx.rng, 110.0, ctx),
        sampleFreeParam('modIndex', 0.5, 10.0, ctx.rng, 2.0, ctx)
      );
    case 'sine':
    default:
      return createSine(
        sampleFreeParam('freq', 55.0, 880.0, ctx.rng, 440.0, ctx),
        sampleFreeParam('phase', 0.0, 6.28, ctx.rng, 0.0, ctx)
      );
  }
}

function sampleL2Audio(ctx: DerivationContext): AudioNode {
  const child = sampleL1Audio(ctx);
  const rule = selectRule(ctx.rules, 'Audio_Unary', ctx.rng);
  switch (rule.constructorName) {
    case 'gain':
      return createGain(sampleFreeParam('gain', 0.1, 1.5, ctx.rng, 0.8, ctx), child);
    case 'adsr':
      return createADSR(
        sampleFreeParam('att', 0.01, 0.5, ctx.rng, 0.05, ctx),
        sampleFreeParam('dec', 0.05, 1.0, ctx.rng, 0.2, ctx),
        sampleFreeParam('sus', 0.1, 1.0, ctx.rng, 0.7, ctx),
        sampleFreeParam('rel', 0.1, 2.0, ctx.rng, 0.5, ctx),
        child
      );
    case 'lowpass':
    default:
      return createLowPass(
        sampleFreeParam('cutoff', 200.0, 5000.0, ctx.rng, 1200.0, ctx),
        sampleFreeParam('res', 0.5, 4.0, ctx.rng, 1.5, ctx),
        child
      );
  }
}

function sampleL3Audio(ctx: DerivationContext): AudioNode {
  const input1 = ctx.rng() > 0.5 ? sampleL2Audio(ctx) : sampleL1Audio(ctx);
  const input2 = sampleL1Audio(ctx);
  return createMix(sampleFreeParam('ratio', 0.1, 0.9, ctx.rng, 0.5, ctx), input1, input2);
}

function sampleL4Audio(ctx: DerivationContext): AudioNode {
  const input1 = sampleL2Audio(ctx);
  const input2 = sampleL2Audio(ctx);
  const mixed = createMix(sampleFreeParam('ratio', 0.2, 0.8, ctx.rng, 0.5, ctx), input1, input2);
  return createLowPass(
    sampleFreeParam('cutoff', 300.0, 4000.0, ctx.rng, 1500.0, ctx),
    sampleFreeParam('res', 0.7, 3.0, ctx.rng, 1.2, ctx),
    mixed
  );
}

function sampleL5Audio(ctx: DerivationContext): AudioNode {
  const branchA = sampleL4Audio(ctx);
  const branchB = sampleL3Audio(ctx);
  const combined = createMix(sampleFreeParam('ratio', 0.3, 0.7, ctx.rng, 0.5, ctx), branchA, branchB);
  return createGain(sampleFreeParam('masterGain', 0.3, 1.0, ctx.rng, 0.75, ctx), combined);
}

// --- Hardware Domain Derivations ---

function sampleL1Hardware(ctx: DerivationContext): HardwareNode {
  const rule = selectRule(ctx.rules, 'HW_Term', ctx.rng);
  if (rule.constructorName === 'imm') {
    return createImm(Math.floor(1 + ctx.rng() * 32));
  }
  const regNames = ['r0', 'r1', 'r2', 'r3', 'out'];
  const name = regNames[Math.floor(ctx.rng() * regNames.length)];
  return createReg(name);
}

function sampleL2Hardware(ctx: DerivationContext): HardwareNode {
  const child = sampleL1Hardware(ctx);
  const shift = Math.floor(1 + ctx.rng() * 4);
  return ctx.rng() > 0.5 ? createShl(child, shift) : createShr(child, shift);
}

function makeHWBinary(ctx: DerivationContext, a: HardwareNode, b: HardwareNode): HardwareNode {
  const rule = selectRule(ctx.rules, 'HW_Binary', ctx.rng);
  switch (rule.constructorName) {
    case 'sub':
      return createSub(a, b);
    case 'mul':
      return createMul(a, b);
    case 'seq':
      return createSeq(a, b);
    case 'add':
    default:
      return createAdd(a, b);
  }
}

function sampleL3Hardware(ctx: DerivationContext): HardwareNode {
  const childA = ctx.rng() > 0.5 ? sampleL2Hardware(ctx) : sampleL1Hardware(ctx);
  const childB = sampleL1Hardware(ctx);
  return makeHWBinary(ctx, childA, childB);
}

function sampleL4Hardware(ctx: DerivationContext): HardwareNode {
  const childA = sampleL2Hardware(ctx);
  const childB = sampleL2Hardware(ctx);
  return makeHWBinary(ctx, childA, childB);
}

function sampleL5Hardware(ctx: DerivationContext): HardwareNode {
  const first = sampleL4Hardware(ctx);
  const second = sampleL3Hardware(ctx);
  return createSeq(first, second);
}

// --- Universal Domain Derivation ---

function sampleUniversalNode(
  level: CurriculumLevel,
  ctx: DerivationContext,
  expectedType: string
): ASTNode {
  if (expectedType === 'AudioNode') {
    const audio = sampleLevelAudio(level, ctx);
    return createUniversalAudio(audio);
  }
  if (expectedType === 'HardwareNode') {
    const hw = sampleLevelHardware(level, ctx);
    return createUniversalHardware(hw);
  }
  const sdf = sampleLevelSDF(level, ctx);
  return createUniversalSDF(sdf);
}

function sampleLevelSDF(level: CurriculumLevel, ctx: DerivationContext): SDFNode {
  switch (level) {
    case 'L1':
      return sampleL1SDF(ctx);
    case 'L2':
      return sampleL2SDF(ctx);
    case 'L3':
      return sampleL3SDF(ctx);
    case 'L4':
      return sampleL4SDF(ctx);
    case 'L5':
      return sampleL5SDF(ctx);
  }
}

function sampleLevelAudio(level: CurriculumLevel, ctx: DerivationContext): AudioNode {
  switch (level) {
    case 'L1':
      return sampleL1Audio(ctx);
    case 'L2':
      return sampleL2Audio(ctx);
    case 'L3':
      return sampleL3Audio(ctx);
    case 'L4':
      return sampleL4Audio(ctx);
    case 'L5':
      return sampleL5Audio(ctx);
  }
}

function sampleLevelHardware(level: CurriculumLevel, ctx: DerivationContext): HardwareNode {
  switch (level) {
    case 'L1':
      return sampleL1Hardware(ctx);
    case 'L2':
      return sampleL2Hardware(ctx);
    case 'L3':
      return sampleL3Hardware(ctx);
    case 'L4':
      return sampleL4Hardware(ctx);
    case 'L5':
      return sampleL5Hardware(ctx);
  }
}

/**
 * Samples a typed AST node at the specified curriculum complexity level.
 * Enforces hard depth caps and step caps to guarantee derivation termination.
 */
export function sampleLevel(
  level: CurriculumLevel,
  context: StructuralContext,
  rng?: () => number
): ASTNode {
  const resolvedRng = rng ?? createPRNG();
  const domain = context.expectedDomain ?? 'sdf';
  const rules = getConditionedRules(domain, context);
  const maxDepth = 6;
  const maxSteps = 100;
  const ctx = new DerivationContext(maxDepth, maxSteps, resolvedRng, rules, context);

  let node: ASTNode;
  if (domain === 'audio') {
    node = sampleLevelAudio(level, ctx);
  } else if (domain === 'hardware') {
    node = sampleLevelHardware(level, ctx);
  } else if (domain === 'universal') {
    node = sampleUniversalNode(level, ctx, context.expectedType);
  } else {
    node = sampleLevelSDF(level, ctx);
  }

  // Type preservation invariant check
  if (domain === 'sdf' && !isSDFNode(node)) {
    return sampleL1SDF(ctx);
  }
  if (domain === 'audio' && !isAudioNode(node)) {
    return sampleL1Audio(ctx);
  }
  if (domain === 'hardware' && !isHardwareNode(node)) {
    return sampleL1Hardware(ctx);
  }

  return node;
}

// ==========================================
// 6. Label & Probability Computations
// ==========================================

export function generateCandidateLabel(ast: ASTNode, level: CurriculumLevel): string {
  const kind = ast.kind.replace('_', ' ');
  const capitalizedKind = kind.charAt(0).toUpperCase() + kind.slice(1);

  switch (level) {
    case 'L1':
      return `[L1] Atomic ${capitalizedKind}`;
    case 'L2':
      return `[L2] Unary ${capitalizedKind}`;
    case 'L3':
      return `[L3] Combinator ${capitalizedKind}`;
    case 'L4':
      return `[L4] Hierarchical ${capitalizedKind}`;
    case 'L5':
      return `[L5] Compound ${capitalizedKind}`;
  }
}

export function calculateDerivationProbability(
  ast: ASTNode,
  level: CurriculumLevel,
  rules: GrammarProductionRule[]
): number {
  let prod = 1.0;
  const stack: ASTNode[] = [ast];
  const maxSearchNodes = 50;
  let searched = 0;

  while (stack.length > 0 && searched < maxSearchNodes) {
    searched += 1;
    const current = stack.pop()!;
    const rule = rules.find((r) => r.constructorName === current.kind);
    if (rule) {
      prod *= Math.max(rule.probability, 0.05);
    } else {
      prod *= 0.1;
    }
  }

  const levelDecay: Record<CurriculumLevel, number> = {
    L1: 1.0,
    L2: 0.8,
    L3: 0.6,
    L4: 0.4,
    L5: 0.25,
  };

  const finalProb = prod * levelDecay[level];
  return Number(Math.max(finalProb, 0.00001).toFixed(5));
}

// ==========================================
// 7. Scoped Variations Entry Point
// ==========================================

/**
 * Generates K independent typed variations preserving the target input/output type signature,
 * stratified across requested curriculum complexity levels.
 */
export function sampleScopedVariations(
  punctureResult: PunctureResult,
  options?: SamplerOptions
): VariationCandidate[] {
  const context = punctureResult.context;
  const domain = context.expectedDomain ?? 'sdf';
  const expectedType = resolveExpectedType(domain);
  const levels: CurriculumLevel[] = options?.levels ?? ['L1', 'L2', 'L3', 'L4', 'L5'];
  const countPerLevel = options?.countPerLevel ?? 2;
  const rng = createPRNG(options?.seed);
  const rules = getConditionedRules(domain, context);

  const candidates: VariationCandidate[] = [];

  for (const level of levels) {
    for (let i = 0; i < countPerLevel; i += 1) {
      const ast = sampleLevel(level, context, rng);
      const depth = getTreeDepth(ast);
      const nodeCount = getNodeCount(ast);
      const probability = calculateDerivationProbability(ast, level, rules);
      const label = generateCandidateLabel(ast, level);
      const id = `cand_${punctureResult.holeId}_${level}_${i + 1}`;
      const leanCode = formatLeanCode('', expectedType, ast, { compact: true });

      candidates.push({
        id,
        ast,
        leanCode,
        complexityLevel: level,
        depth,
        nodeCount,
        domain,
        probability,
        label,
      });
    }
  }

  return candidates;
}
