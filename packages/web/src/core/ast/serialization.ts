/**
 * Lean 4 JSON Serialization and Deserialization Engine for SymForge AST.
 *
 * Implements bidirectional serialization schema matching `SymforgeIr.Types` deriving `ToJson` / `FromJson`.
 * Supports bidirectional compatibility with Lean 4 camelCase and Python snake_case keys.
 */

import type {
  FreeParam,
  AudioNode,
  SDFNode,
  HardwareNode,
  UniversalNode,
  ASTNode,
} from './types';

import {
  createFreeParam,
  createSine,
  createSaw,
  createSquare,
  createNoise,
  createADSR,
  createLowPass,
  createGain,
  createMix,
  createFM,
  createAudioHole,
  createAudioMacro,
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
  createSDFHole,
  createSDFMacro,
  createReg,
  createImm,
  createAdd,
  createSub,
  createMul,
  createShl,
  createShr,
  createSeq,
  createHardwareHole,
  createHardwareMacro,
  createUniversalAudio,
  createUniversalSDF,
  createUniversalHardware,
  createUniversalMacro,
  createHole,
  createMacroCall,
  isFreeParam,
} from './models';

// ==========================================
// 1. FreeParam Serialization / Deserialization
// ==========================================

export function serializeFreeParam(p: FreeParam): Record<string, any> {
  return {
    id: p.id,
    minVal: p.minVal,
    maxVal: p.maxVal,
    defaultVal: p.defaultVal,
  };
}

export function deserializeFreeParam(raw: any): FreeParam {
  if (!raw || typeof raw !== 'object') {
    throw new Error(
      `deserializeFreeParam: Invalid input payload (expected object, got ${typeof raw}): ${JSON.stringify(raw)}`
    );
  }
  const id = String(raw.id ?? raw.param_id ?? raw.name ?? 'param_0');
  const minVal =
    raw.minVal !== undefined
      ? Number(raw.minVal)
      : raw.min_val !== undefined
      ? Number(raw.min_val)
      : 0;
  const maxVal =
    raw.maxVal !== undefined
      ? Number(raw.maxVal)
      : raw.max_val !== undefined
      ? Number(raw.max_val)
      : 1;
  const defaultVal =
    raw.defaultVal !== undefined
      ? Number(raw.defaultVal)
      : raw.default_val !== undefined
      ? Number(raw.default_val)
      : 0;
  return createFreeParam(id, minVal, maxVal, defaultVal);
}

// ==========================================
// 2. Argument Helpers for Macros
// ==========================================

function serializeArgs(args: Record<string, any>): Record<string, any> {
  const serialized: Record<string, any> = {};
  for (const [k, v] of Object.entries(args)) {
    if (isFreeParam(v)) {
      serialized[k] = serializeFreeParam(v);
    } else if (v && typeof v === 'object' && 'kind' in v) {
      serialized[k] = serializeToLeanJson(v as ASTNode);
    } else {
      serialized[k] = v;
    }
  }
  return serialized;
}

function deserializeArgs(raw: Record<string, any> | undefined): Record<string, any> {
  if (!raw || typeof raw !== 'object') return {};
  const deserialized: Record<string, any> = {};
  for (const [k, v] of Object.entries(raw)) {
    if (!v || typeof v !== 'object') {
      deserialized[k] = v;
    } else if (
      ('minVal' in v || 'min_val' in v) &&
      ('defaultVal' in v || 'default_val' in v)
    ) {
      deserialized[k] = deserializeFreeParam(v);
    } else if (!Array.isArray(v) && Object.keys(v).length === 1 && !('kind' in v)) {
      try {
        deserialized[k] = deserializeFromLeanJson(v);
      } catch (err) {
        console.error("deserializeArgs: Failed to parse nested AST for arg", k, err, v);
        deserialized[k] = v;
      }
    } else {
      deserialized[k] = v;
    }
  }
  return deserialized;
}

// ==========================================
// 3. Domain Specific Serializers
// ==========================================

function serializeAudioNode(node: AudioNode): Record<string, any> {
  switch (node.kind) {
    case 'sine':
      return { sine: { freq: serializeFreeParam(node.freq), phase: serializeFreeParam(node.phase) } };
    case 'saw':
      return { saw: { freq: serializeFreeParam(node.freq) } };
    case 'square':
      return { square: { freq: serializeFreeParam(node.freq), duty: serializeFreeParam(node.duty) } };
    case 'noise':
      return { noise: { seed: node.seed } };
    case 'adsr':
      return {
        adsr: {
          attack: serializeFreeParam(node.attack),
          decay: serializeFreeParam(node.decay),
          sustain: serializeFreeParam(node.sustain),
          release: serializeFreeParam(node.release),
          gate: serializeToLeanJson(node.gate),
        },
      };
    case 'lowpass':
      return {
        lowpass: {
          cutoff: serializeFreeParam(node.cutoff),
          resonance: serializeFreeParam(node.resonance),
          input: serializeToLeanJson(node.input),
        },
      };
    case 'gain':
      return {
        gain: {
          gain: serializeFreeParam(node.gain),
          input: serializeToLeanJson(node.input),
        },
      };
    case 'mix':
      return {
        mix: {
          ratio: serializeFreeParam(node.ratio),
          input1: serializeToLeanJson(node.input1),
          input2: serializeToLeanJson(node.input2),
        },
      };
    case 'fm':
      return {
        fm: {
          carrier: serializeFreeParam(node.carrier),
          modFreq: serializeFreeParam(node.modFreq),
          modIndex: serializeFreeParam(node.modIndex),
        },
      };
    case 'hole':
      return { audioHole: { id: node.holeId } };
    case 'macro':
      return {
        macro: {
          macroName: node.macroName,
          arguments: serializeArgs(node.arguments),
          domain: 'audio',
        },
      };
  }
}

function serializeSDFNode(node: SDFNode): Record<string, any> {
  switch (node.kind) {
    case 'sphere':
      return { sphere: { radius: serializeFreeParam(node.radius) } };
    case 'box':
      return {
        box: {
          width: serializeFreeParam(node.width),
          height: serializeFreeParam(node.height),
          depth: serializeFreeParam(node.depth),
        },
      };
    case 'cylinder':
      return { cylinder: { radius: serializeFreeParam(node.radius), height: serializeFreeParam(node.height) } };
    case 'torus':
      return {
        torus: {
          majorRadius: serializeFreeParam(node.majorRadius),
          minorRadius: serializeFreeParam(node.minorRadius),
        },
      };
    case 'plane':
      return {
        plane: {
          offset: serializeFreeParam(node.offset),
        },
      };
    case 'translate':
      return {
        translate: {
          dx: serializeFreeParam(node.dx),
          dy: serializeFreeParam(node.dy),
          dz: serializeFreeParam(node.dz),
          child: serializeToLeanJson(node.child),
        },
      };
    case 'rotate':
      return {
        rotate: {
          rx: serializeFreeParam(node.rx),
          ry: serializeFreeParam(node.ry),
          rz: serializeFreeParam(node.rz),
          child: serializeToLeanJson(node.child),
        },
      };
    case 'scale':
      return { scale: { s: serializeFreeParam(node.s), child: serializeToLeanJson(node.child) } };
    case 'union':
      return {
        union: {
          smoothness: serializeFreeParam(node.smoothness),
          a: serializeToLeanJson(node.a),
          b: serializeToLeanJson(node.b),
        },
      };
    case 'smooth_union':
      return {
        smoothUnion: {
          smoothness: serializeFreeParam(node.smoothness),
          a: serializeToLeanJson(node.a),
          b: serializeToLeanJson(node.b),
        },
      };
    case 'intersection':
      return { intersection: { a: serializeToLeanJson(node.a), b: serializeToLeanJson(node.b) } };
    case 'difference':
      return { difference: { a: serializeToLeanJson(node.a), b: serializeToLeanJson(node.b) } };
    case 'hole':
      return { sdfHole: { id: node.holeId } };
    case 'macro':
      return {
        macro: {
          macroName: node.macroName,
          arguments: serializeArgs(node.arguments),
          domain: 'sdf',
        },
      };
  }
}

function serializeHardwareNode(node: HardwareNode): Record<string, any> {
  switch (node.kind) {
    case 'reg':
      return { reg: { name: node.name } };
    case 'imm':
      return { imm: { val: node.val } };
    case 'add':
      return { add: { dst: node.dst ?? 'out', src1: serializeToLeanJson(node.src1), src2: serializeToLeanJson(node.src2) } };
    case 'sub':
      return { sub: { dst: node.dst ?? 'out', src1: serializeToLeanJson(node.src1), src2: serializeToLeanJson(node.src2) } };
    case 'mul':
      return { mul: { dst: node.dst ?? 'out', src1: serializeToLeanJson(node.src1), src2: serializeToLeanJson(node.src2) } };
    case 'shl':
      return { shl: { dst: node.dst ?? 'out', src: serializeToLeanJson(node.src), shift: node.shift } };
    case 'shr':
      return { shr: { dst: node.dst ?? 'out', src: serializeToLeanJson(node.src), shift: node.shift } };
    case 'seq':
      return { seq: { first: serializeToLeanJson(node.first), second: serializeToLeanJson(node.second) } };
    case 'hole':
      return { hardwareHole: { id: node.holeId } };
    case 'macro':
      return {
        macro: {
          macroName: node.macroName,
          arguments: serializeArgs(node.arguments),
          domain: 'hardware',
        },
      };
  }
}

function serializeUniversalNode(node: UniversalNode): Record<string, any> {
  switch (node.kind) {
    case 'universal_audio':
      return { audio: { node: serializeToLeanJson(node.node) } };
    case 'universal_sdf':
      return { sdf: { node: serializeToLeanJson(node.node) } };
    case 'universal_hardware':
      return { hardware: { node: serializeToLeanJson(node.node) } };
    case 'universal_macro':
      return {
        macroNode: {
          name: node.name,
          children: node.children.map((c) => serializeToLeanJson(c)),
        },
      };
  }
}

/**
 * Serializes an AST node to Lean 4 JSON schema matching `SymforgeIr.Types` deriving `ToJson`.
 */
export function serializeToLeanJson(node: ASTNode): any {
  if (!node || typeof node !== 'object' || !('kind' in node)) {
    throw new Error(`serializeToLeanJson: Expected ASTNode, got: ${JSON.stringify(node)}`);
  }

  // Universal Nodes
  if (node.kind.startsWith('universal_')) {
    return serializeUniversalNode(node as UniversalNode);
  }

  // Domain dispatch
  const audioKinds = ['sine', 'saw', 'square', 'noise', 'adsr', 'lowpass', 'gain', 'mix', 'fm'];
  if (audioKinds.includes(node.kind)) {
    return serializeAudioNode(node as AudioNode);
  }

  const sdfKinds = ['sphere', 'box', 'cylinder', 'torus', 'plane', 'translate', 'rotate', 'scale', 'union', 'smooth_union', 'intersection', 'difference'];
  if (sdfKinds.includes(node.kind)) {
    return serializeSDFNode(node as SDFNode);
  }

  const hwKinds = ['reg', 'imm', 'add', 'sub', 'mul', 'shl', 'shr', 'seq'];
  if (hwKinds.includes(node.kind)) {
    return serializeHardwareNode(node as HardwareNode);
  }

  // Hole dispatch
  if (node.kind === 'hole') {
    const domain = (node as any).domain;
    if (domain === 'audio') return { audioHole: { id: (node as any).holeId } };
    if (domain === 'sdf') return { sdfHole: { id: (node as any).holeId } };
    if (domain === 'hardware') return { hardwareHole: { id: (node as any).holeId } };
    return { hole: { id: (node as any).holeId } };
  }

  // Macro dispatch
  if (node.kind === 'macro') {
    return {
      macro: {
        macroName: node.macroName,
        arguments: serializeArgs(node.arguments),
        domain: node.domain,
      },
    };
  }

  throw new Error(`serializeToLeanJson: Unknown node kind "${(node as any).kind}"`);
}

// ==========================================
// 4. Domain Specific Deserializers
// ==========================================

function deserializeAudioTag(tag: string, fields: Record<string, any>): AudioNode | null {
  switch (tag) {
    case 'sine':
      return createSine(deserializeFreeParam(fields.freq), deserializeFreeParam(fields.phase));
    case 'saw':
      return createSaw(deserializeFreeParam(fields.freq));
    case 'square':
      return createSquare(deserializeFreeParam(fields.freq), deserializeFreeParam(fields.duty));
    case 'noise':
      return createNoise(Number(fields.seed ?? 42));
    case 'adsr':
      return createADSR(
        deserializeFreeParam(fields.attack),
        deserializeFreeParam(fields.decay),
        deserializeFreeParam(fields.sustain),
        deserializeFreeParam(fields.release),
        deserializeFromLeanJson(fields.gate, 'audio') as AudioNode
      );
    case 'lowpass':
      return createLowPass(
        deserializeFreeParam(fields.cutoff),
        deserializeFreeParam(fields.resonance),
        deserializeFromLeanJson(fields.input ?? fields.input_node, 'audio') as AudioNode
      );
    case 'gain':
      return createGain(
        deserializeFreeParam(fields.gain),
        deserializeFromLeanJson(fields.input ?? fields.input_node, 'audio') as AudioNode
      );
    case 'mix':
      return createMix(
        deserializeFreeParam(fields.ratio),
        deserializeFromLeanJson(fields.input1, 'audio') as AudioNode,
        deserializeFromLeanJson(fields.input2, 'audio') as AudioNode
      );
    case 'fm':
      return createFM(
        deserializeFreeParam(fields.carrier),
        deserializeFreeParam(fields.modFreq ?? fields.mod_freq),
        deserializeFreeParam(fields.modIndex ?? fields.mod_index)
      );
    case 'audioHole':
      return createAudioHole(String(fields.id ?? fields.holeId ?? fields.hole_id ?? 'h_0'));
    default:
      return null;
  }
}

function deserializeSDFTag(tag: string, fields: Record<string, any>): SDFNode | null {
  switch (tag) {
    case 'sphere':
      return createSphere(deserializeFreeParam(fields.radius));
    case 'box':
      return createBox(
        deserializeFreeParam(fields.width),
        deserializeFreeParam(fields.height),
        deserializeFreeParam(fields.depth)
      );
    case 'cylinder':
      return createCylinder(
        deserializeFreeParam(fields.radius),
        deserializeFreeParam(fields.height)
      );
    case 'torus':
      return createTorus(
        deserializeFreeParam(fields.majorRadius ?? fields.major_radius),
        deserializeFreeParam(fields.minorRadius ?? fields.minor_radius)
      );
    case 'plane':
      return createPlane(
        deserializeFreeParam(fields.offset)
      );
    case 'translate':
      return createTranslate(
        deserializeFreeParam(fields.dx),
        deserializeFreeParam(fields.dy),
        deserializeFreeParam(fields.dz),
        deserializeFromLeanJson(fields.child, 'sdf') as SDFNode
      );
    case 'rotate':
      return createRotate(
        deserializeFreeParam(fields.rx),
        deserializeFreeParam(fields.ry),
        deserializeFreeParam(fields.rz),
        deserializeFromLeanJson(fields.child, 'sdf') as SDFNode
      );
    case 'scale':
      return createScale(
        deserializeFreeParam(fields.s),
        deserializeFromLeanJson(fields.child, 'sdf') as SDFNode
      );
    case 'union':
      return createUnion(
        deserializeFreeParam(fields.smoothness),
        deserializeFromLeanJson(fields.a, 'sdf') as SDFNode,
        deserializeFromLeanJson(fields.b, 'sdf') as SDFNode
      );
    case 'smoothUnion':
    case 'smooth_union':
      return createSmoothUnion(
        deserializeFreeParam(fields.smoothness),
        deserializeFromLeanJson(fields.a, 'sdf') as SDFNode,
        deserializeFromLeanJson(fields.b, 'sdf') as SDFNode
      );
    case 'intersection':
      return createIntersection(
        deserializeFromLeanJson(fields.a, 'sdf') as SDFNode,
        deserializeFromLeanJson(fields.b, 'sdf') as SDFNode
      );
    case 'difference':
      return createDifference(
        deserializeFromLeanJson(fields.a, 'sdf') as SDFNode,
        deserializeFromLeanJson(fields.b, 'sdf') as SDFNode
      );
    case 'sdfHole':
      return createSDFHole(String(fields.id ?? fields.holeId ?? fields.hole_id ?? 'h_0'));
    default:
      return null;
  }
}

function deserializeHardwareTag(tag: string, fields: Record<string, any>): HardwareNode | null {
  switch (tag) {
    case 'reg':
    case 'regHardware': {
      const regVal = fields.name ?? (fields.regIdx !== undefined ? `r${fields.regIdx}` : 'r0');
      return createReg(String(regVal), fields.regIdx ?? fields.reg_idx);
    }
    case 'imm':
    case 'immHardware':
      return createImm(Number(fields.val));
    case 'add':
    case 'addHardware':
      return createAdd(
        deserializeFromLeanJson(fields.src1 ?? fields.left, 'hardware') as HardwareNode,
        deserializeFromLeanJson(fields.src2 ?? fields.right, 'hardware') as HardwareNode,
        fields.dst
      );
    case 'sub':
    case 'subHardware':
      return createSub(
        deserializeFromLeanJson(fields.src1 ?? fields.left, 'hardware') as HardwareNode,
        deserializeFromLeanJson(fields.src2 ?? fields.right, 'hardware') as HardwareNode,
        fields.dst
      );
    case 'mul':
    case 'mulHardware':
      return createMul(
        deserializeFromLeanJson(fields.src1 ?? fields.left, 'hardware') as HardwareNode,
        deserializeFromLeanJson(fields.src2 ?? fields.right, 'hardware') as HardwareNode,
        fields.dst
      );
    case 'shl':
    case 'shlHardware':
      return createShl(
        deserializeFromLeanJson(fields.src ?? fields.left, 'hardware') as HardwareNode,
        Number(fields.shift),
        fields.dst
      );
    case 'shr':
    case 'shrHardware':
      return createShr(
        deserializeFromLeanJson(fields.src ?? fields.left, 'hardware') as HardwareNode,
        Number(fields.shift),
        fields.dst
      );
    case 'seq':
      return createSeq(
        deserializeFromLeanJson(fields.first, 'hardware') as HardwareNode,
        deserializeFromLeanJson(fields.second, 'hardware') as HardwareNode
      );
    case 'hardwareHole':
    case 'hwHole':
      return createHardwareHole(String(fields.id ?? fields.holeId ?? fields.name ?? 'h_0'));
    default:
      return null;
  }
}

/**
 * Robustly parses Lean 4 JSON into typed TypeScript AST instances.
 */
export function deserializeFromLeanJson(raw: any, domain?: string): ASTNode {
  let data = raw;
  if (typeof data === 'string') {
    try {
      data = JSON.parse(data);
    } catch (err) {
      console.error("deserializeFromLeanJson: Failed to parse input string as JSON", raw, err);
      throw new Error(`deserializeFromLeanJson: Invalid JSON string: ${String(err)}`);
    }
  }

  if (!data || typeof data !== 'object') {
    throw new Error(`deserializeFromLeanJson: Expected object or JSON string, received ${typeof data}`);
  }

  // Already an ASTNode instance
  if ('kind' in data) {
    return data as ASTNode;
  }

  const keys = Object.keys(data);
  if (keys.length !== 1) {
    throw new Error(
      `deserializeFromLeanJson: Expected single-tag JSON object representing inductive constructor, got keys: [${keys.join(', ')}]`
    );
  }

  const tag = keys[0];
  const fields = data[tag] || {};

  // 1. Universal Containers
  if (tag === 'audio' && fields.node) {
    return createUniversalAudio(deserializeFromLeanJson(fields.node, 'audio') as AudioNode);
  }
  if (tag === 'sdf' && fields.node) {
    return createUniversalSDF(deserializeFromLeanJson(fields.node, 'sdf') as SDFNode);
  }
  if (tag === 'hardware' && fields.node) {
    return createUniversalHardware(deserializeFromLeanJson(fields.node, 'hardware') as HardwareNode);
  }
  if (tag === 'macroNode') {
    const children = (fields.children || []).map((c: any) => deserializeFromLeanJson(c) as UniversalNode);
    return createUniversalMacro(String(fields.name), children);
  }

  // 2. Audio Domain
  const audioNode = deserializeAudioTag(tag, fields);
  if (audioNode) return audioNode;

  // 3. SDF Domain
  const sdfNode = deserializeSDFTag(tag, fields);
  if (sdfNode) return sdfNode;

  // 4. Hardware Domain
  const hwNode = deserializeHardwareTag(tag, fields);
  if (hwNode) return hwNode;

  // 5. Generic Hole
  if (tag === 'hole') {
    const hId = String(fields.id ?? fields.holeId ?? fields.hole_id ?? 'h_0');
    if (domain === 'audio') return createAudioHole(hId);
    if (domain === 'sdf') return createSDFHole(hId);
    if (domain === 'hardware') return createHardwareHole(hId);
    return createHole(hId, domain);
  }

  // 6. Generic Macro Call
  if (tag === 'macro' || tag === 'macroCall') {
    const macroName = String(fields.macroName ?? fields.name ?? 'macro');
    const parsedArgs = deserializeArgs(fields.arguments);
    const resolvedDomain = String(fields.domain ?? domain ?? 'generic');
    if (resolvedDomain === 'audio') return createAudioMacro(macroName, parsedArgs);
    if (resolvedDomain === 'sdf') return createSDFMacro(macroName, parsedArgs);
    if (resolvedDomain === 'hardware') return createHardwareMacro(macroName, parsedArgs);
    return createMacroCall(macroName, parsedArgs, resolvedDomain);
  }

  throw new Error(`deserializeFromLeanJson: Unrecognized constructor tag "${tag}" (fields: ${JSON.stringify(fields)})`);
}
