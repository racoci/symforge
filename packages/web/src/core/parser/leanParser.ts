/**
 * Lean 4 AST Recursive Descent Parser.
 *
 * Parses Lean 4 domain declarations and expressions into typed AST nodes
 * with exact SourceSpan coordinates, FreeParam auto-inference, and robust error recovery.
 */

import type { ASTNode, FreeParam, AudioNode, SDFNode, HardwareNode } from '../ast/types';
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
  createHole,
  createMacroCall,
  isFreeParam,
} from '../ast';
import { tokenize } from './lexer';
import {
  combineSpans,
  type SourceSpan,
  type Token,
  type TokenType,
  type ParseError,
  type ParseResult,
} from './types';

interface ParserState {
  tokens: Token[];
  pos: number;
  length: number;
  spans: Map<string, SourceSpan>;
  errors: ParseError[];
  letEnv: Map<string, FreeParam | ASTNode | number>;
  domainType?: string;
  nodeToSpanMap: WeakMap<object, SourceSpan>;
  nodeToIdMap: WeakMap<object, string>;
  nodeCounter: number;
  paramCounter: number;
}

function peek(state: ParserState, offset = 0): Token {
  const idx = state.pos + offset;
  if (idx < state.length) return state.tokens[idx];
  return state.tokens[state.length - 1]; // EOF token
}

function consume(state: ParserState): Token {
  const tok = peek(state);
  if (state.pos < state.length - 1) {
    state.pos++;
  }
  return tok;
}

function match(state: ParserState, type: TokenType, value?: string): boolean {
  const tok = peek(state);
  if (tok.type !== type) return false;
  if (value !== undefined && tok.value !== value) return false;
  return true;
}

function expect(state: ParserState, type: TokenType, value?: string): Token {
  const tok = peek(state);
  if (tok.type === type && (value === undefined || tok.value === value)) {
    return consume(state);
  }
  const expectedDesc = value ? `'${value}'` : type;
  state.errors.push({
    message: `Expected ${expectedDesc} but encountered '${tok.value}'`,
    span: tok.span,
    line: tok.span.startLine,
    col: tok.span.startCol,
  });
  return tok;
}

function recordSpan(state: ParserState, node: any, span: SourceSpan, customKey?: string): void {
  const nodeId = `node_${state.nodeCounter++}`;
  if (node && typeof node === 'object') {
    state.nodeToSpanMap.set(node, span);
    state.nodeToIdMap.set(node, nodeId);
  }
  state.spans.set(nodeId, span);
  if (customKey) {
    state.spans.set(customKey, span);
  }
}

function skipPreamble(state: ParserState): void {
  const maxIter = 100;
  let iter = 0;
  while (iter++ < maxIter) {
    const tok = peek(state);
    if (tok.type === 'Keyword' && (tok.value === 'import' || tok.value === 'open' || tok.value === 'namespace')) {
      consume(state);
      // Consume qualified name
      while (peek(state).type === 'Identifier' || (peek(state).type === 'Operator' && peek(state).value === '.')) {
        consume(state);
      }
      continue;
    }
    break;
  }
}

function parseQualifiedName(state: ParserState): { raw: string; base: string; span: SourceSpan } {
  const startTok = peek(state);
  let raw = '';
  let lastTok = startTok;

  if (match(state, 'Operator', '.')) {
    consume(state);
    raw += '.';
    const idTok = expect(state, 'Identifier');
    raw += idTok.value;
    lastTok = idTok;
    return { raw, base: idTok.value.toLowerCase(), span: combineSpans(startTok.span, lastTok.span) };
  }

  const firstTok = expect(state, 'Identifier');
  raw += firstTok.value;
  lastTok = firstTok;

  while (match(state, 'Operator', '.')) {
    consume(state);
    raw += '.';
    const nextTok = expect(state, 'Identifier');
    raw += nextTok.value;
    lastTok = nextTok;
  }

  const parts = raw.split('.');
  const base = parts[parts.length - 1].toLowerCase();
  return { raw, base, span: combineSpans(startTok.span, lastTok.span) };
}

function parseNumberValue(state: ParserState): { val: number; span: SourceSpan } {
  let isNegative = false;
  let startSpan: SourceSpan | null = null;

  if (match(state, 'Operator', '-')) {
    const opTok = consume(state);
    isNegative = true;
    startSpan = opTok.span;
  }

  const numTok = expect(state, 'Number');
  let val = numTok.numberValue ?? parseFloat(numTok.value);
  if (isNegative) val = -Math.abs(val);

  const span = startSpan ? combineSpans(startSpan, numTok.span) : numTok.span;
  return { val, span };
}

function makeFreeParamFromNumber(val: number, nameHint = 'param'): FreeParam {
  const minVal = val >= 0 ? 0.0 : val * 2.0;
  const maxVal = val === 0 ? 1.0 : (val > 0 ? val * 2.0 : 0.0);
  return createFreeParam(`${nameHint}`, minVal, maxVal, val);
}

function parseExplicitFreeParam(state: ParserState, startSpan: SourceSpan): FreeParam {
  let id = `param_${state.paramCounter++}`;
  if (match(state, 'String')) {
    id = consume(state).value;
  } else if (match(state, 'Identifier')) {
    id = consume(state).value;
  }

  const n1 = parseNumberValue(state);
  let minVal = n1.val;
  let maxVal = n1.val > 0 ? n1.val * 2 : 1.0;
  let defaultVal = n1.val;

  if (match(state, 'Number') || match(state, 'Operator', '-')) {
    const n2 = parseNumberValue(state);
    maxVal = n2.val;
    if (match(state, 'Number') || match(state, 'Operator', '-')) {
      const n3 = parseNumberValue(state);
      defaultVal = n3.val;
    }
  }

  const endSpan = peek(state, -1).span;
  const param = createFreeParam(id, minVal, maxVal, defaultVal);
  const span = combineSpans(startSpan, endSpan);
  recordSpan(state, param, span, param.id);
  state.spans.set(`param:${param.id}`, span);
  return param;
}

function parseAnonymousParam(state: ParserState): FreeParam {
  const startTok = consume(state); // '⟨'
  let id = `param_${state.paramCounter++}`;

  if (match(state, 'String')) {
    id = consume(state).value;
  }
  if (match(state, 'Delimiter', ',')) consume(state);

  const n1 = parseNumberValue(state);
  if (match(state, 'Delimiter', ',')) consume(state);
  const n2 = parseNumberValue(state);
  if (match(state, 'Delimiter', ',')) consume(state);
  const n3 = parseNumberValue(state);
  if (match(state, 'Delimiter', ',')) consume(state);

  const endTok = expect(state, 'Delimiter', '⟩');
  const param = createFreeParam(id, n1.val, n2.val, n3.val);
  const span = combineSpans(startTok.span, endTok.span);
  recordSpan(state, param, span, param.id);
  state.spans.set(`param:${param.id}`, span);
  return param;
}

function parseRecordParam(state: ParserState): FreeParam {
  const startTok = consume(state); // '{'
  let id = `param_${state.paramCounter++}`;
  let minVal = 0.0;
  let maxVal = 1.0;
  let defaultVal = 0.5;

  const maxFields = 10;
  let iter = 0;
  while (!match(state, 'Delimiter', '}') && iter++ < maxFields) {
    if (match(state, 'Delimiter', ',')) {
      consume(state);
      continue;
    }
    const fieldTok = expect(state, 'Identifier');
    expect(state, 'Operator', ':=');
    if (fieldTok.value === 'id') {
      id = expect(state, 'String').value;
    } else if (fieldTok.value === 'minVal') {
      minVal = parseNumberValue(state).val;
    } else if (fieldTok.value === 'maxVal') {
      maxVal = parseNumberValue(state).val;
    } else if (fieldTok.value === 'defaultVal') {
      defaultVal = parseNumberValue(state).val;
    } else {
      consume(state);
    }
  }

  const endTok = expect(state, 'Delimiter', '}');
  const param = createFreeParam(id, minVal, maxVal, defaultVal);
  const span = combineSpans(startTok.span, endTok.span);
  recordSpan(state, param, span, param.id);
  state.spans.set(`param:${param.id}`, span);
  return param;
}

function parseLetBindings(state: ParserState): void {
  const maxLets = 100;
  let iter = 0;
  while (match(state, 'Keyword', 'let') && iter++ < maxLets) {
    const letTok = consume(state);
    const varName = expect(state, 'Identifier').value;

    if (match(state, 'Operator', ':')) {
      consume(state);
      parseQualifiedName(state);
    }

    expect(state, 'Operator', ':=');
    const val = parseExpressionOrParam(state);
    state.letEnv.set(varName, val);

    if (match(state, 'Keyword', 'in')) {
      consume(state);
    }

    const span = combineSpans(letTok.span, peek(state, -1).span);
    state.spans.set(varName, span);
  }
}

function parseExpressionOrParam(state: ParserState): any {
  if (match(state, 'Delimiter', '⟨')) {
    return parseAnonymousParam(state);
  }
  if (match(state, 'Delimiter', '{')) {
    return parseRecordParam(state);
  }
  if (match(state, 'Number') || match(state, 'Operator', '-')) {
    const num = parseNumberValue(state);
    const param = makeFreeParamFromNumber(num.val);
    recordSpan(state, param, num.span, param.id);
    return param;
  }
  return parseExpression(state);
}

function parseParamArgument(state: ParserState, nameHint: string): FreeParam {
  if (match(state, 'Delimiter', '(')) {
    const openTok = consume(state);
    const inner = parseExpressionOrParam(state);
    const closeTok = expect(state, 'Delimiter', ')');
    if (isFreeParam(inner)) {
      const fullSpan = combineSpans(openTok.span, closeTok.span);
      recordSpan(state, inner, fullSpan, inner.id);
      state.spans.set(`param:${inner.id}`, fullSpan);
      return inner;
    }
    return makeFreeParamFromNumber(typeof inner === 'number' ? inner : 1.0, nameHint);
  }

  if (match(state, 'Delimiter', '⟨')) {
    return parseAnonymousParam(state);
  }
  if (match(state, 'Delimiter', '{')) {
    return parseRecordParam(state);
  }

  if (match(state, 'Number') || (match(state, 'Operator', '-') && peek(state, 1).type === 'Number')) {
    const num = parseNumberValue(state);
    const param = makeFreeParamFromNumber(num.val, nameHint);
    recordSpan(state, param, num.span, param.id);
    return param;
  }

  if (match(state, 'Identifier')) {
    const idTok = peek(state);
    if (state.letEnv.has(idTok.value)) {
      consume(state);
      const resolved = state.letEnv.get(idTok.value);
      if (isFreeParam(resolved)) return resolved;
      if (typeof resolved === 'number') return makeFreeParamFromNumber(resolved, nameHint);
    }
  }

  const expr = parseExpression(state);
  if (isFreeParam(expr)) return expr;
  return makeFreeParamFromNumber(1.0, nameHint);
}

function parseNodeArgument(state: ParserState): any {
  if (match(state, 'Delimiter', '(')) {
    const openTok = consume(state);
    const inner = parseExpression(state);
    const closeTok = expect(state, 'Delimiter', ')');
    const span = combineSpans(openTok.span, closeTok.span);
    recordSpan(state, inner, span);
    return inner;
  }
  if (match(state, 'Identifier')) {
    const idTok = peek(state);
    if (state.letEnv.has(idTok.value)) {
      consume(state);
      return state.letEnv.get(idTok.value);
    }
  }
  return parseExpression(state);
}

function parseSDFConstructor(state: ParserState, base: string, startSpan: SourceSpan): SDFNode {
  let node: SDFNode;

  switch (base) {
    case 'sphere': {
      const radius = parseParamArgument(state, 'radius');
      node = createSphere(radius);
      break;
    }
    case 'box': {
      const w = parseParamArgument(state, 'width');
      const h = parseParamArgument(state, 'height');
      const d = parseParamArgument(state, 'depth');
      node = createBox(w, h, d);
      break;
    }
    case 'cylinder': {
      const r = parseParamArgument(state, 'radius');
      const h = parseParamArgument(state, 'height');
      node = createCylinder(r, h);
      break;
    }
    case 'torus': {
      const major = parseParamArgument(state, 'majorRadius');
      const minor = parseParamArgument(state, 'minorRadius');
      node = createTorus(major, minor);
      break;
    }
    case 'plane': {
      const offset = parseParamArgument(state, 'offset');
      node = createPlane(offset);
      break;
    }
    case 'translate': {
      const dx = parseParamArgument(state, 'dx');
      const dy = parseParamArgument(state, 'dy');
      const dz = parseParamArgument(state, 'dz');
      const child = parseNodeArgument(state);
      node = createTranslate(dx, dy, dz, child);
      break;
    }
    case 'rotate': {
      const rx = parseParamArgument(state, 'rx');
      const ry = parseParamArgument(state, 'ry');
      const rz = parseParamArgument(state, 'rz');
      const child = parseNodeArgument(state);
      node = createRotate(rx, ry, rz, child);
      break;
    }
    case 'scale': {
      const s = parseParamArgument(state, 's');
      const child = parseNodeArgument(state);
      node = createScale(s, child);
      break;
    }
    case 'union': {
      const arg1 = parseExpressionOrParam(state);
      if (isFreeParam(arg1)) {
        const a = parseNodeArgument(state);
        const b = parseNodeArgument(state);
        node = createUnion(arg1, a, b);
      } else {
        const b = parseNodeArgument(state);
        node = createUnion(createFreeParam('smooth', 0.0, 1.0, 0.0), arg1, b);
      }
      break;
    }
    case 'smooth_union':
    case 'smoothunion': {
      const sm = parseParamArgument(state, 'smooth');
      const a = parseNodeArgument(state);
      const b = parseNodeArgument(state);
      node = createSmoothUnion(sm, a, b);
      break;
    }
    case 'intersection': {
      const a = parseNodeArgument(state);
      const b = parseNodeArgument(state);
      node = createIntersection(a, b);
      break;
    }
    case 'difference': {
      const a = parseNodeArgument(state);
      const b = parseNodeArgument(state);
      node = createDifference(a, b);
      break;
    }
    case 'hole': {
      const holeId = match(state, 'String') ? consume(state).value : 'sdf_hole';
      node = createSDFHole(holeId);
      break;
    }
    case 'macro': {
      const macroName = match(state, 'String') ? consume(state).value : 'sdf_macro';
      node = createSDFMacro(macroName, {});
      break;
    }
    default: {
      const macroName = base;
      const args: Record<string, any> = {};
      let argIdx = 0;
      while (!match(state, 'Delimiter', ')') && !match(state, 'EOF') && !match(state, 'Keyword')) {
        const arg = parseExpressionOrParam(state);
        const argKey = isFreeParam(arg) ? arg.id : `arg_${argIdx++}`;
        args[argKey] = arg;
      }
      node = createSDFMacro(macroName, args);
      break;
    }
  }

  const endSpan = peek(state, -1).span;
  recordSpan(state, node, combineSpans(startSpan, endSpan));
  return node;
}

function parseAudioConstructor(state: ParserState, base: string, startSpan: SourceSpan): AudioNode {
  let node: AudioNode;

  switch (base) {
    case 'sine': {
      const freq = parseParamArgument(state, 'freq');
      const phase = match(state, 'Delimiter', ')') || match(state, 'EOF')
        ? undefined
        : parseParamArgument(state, 'phase');
      node = createSine(freq, phase);
      break;
    }
    case 'saw': {
      const freq = parseParamArgument(state, 'freq');
      node = createSaw(freq);
      break;
    }
    case 'square': {
      const freq = parseParamArgument(state, 'freq');
      const duty = match(state, 'Delimiter', ')') || match(state, 'EOF')
        ? undefined
        : parseParamArgument(state, 'duty');
      node = createSquare(freq, duty);
      break;
    }
    case 'noise': {
      const seed = match(state, 'Number') ? parseNumberValue(state).val : 42;
      node = createNoise(seed);
      break;
    }
    case 'adsr': {
      const a = parseParamArgument(state, 'attack');
      const d = parseParamArgument(state, 'decay');
      const s = parseParamArgument(state, 'sustain');
      const r = parseParamArgument(state, 'release');
      const gate = parseNodeArgument(state);
      node = createADSR(a, d, s, r, gate);
      break;
    }
    case 'lowpass': {
      const c = parseParamArgument(state, 'cutoff');
      const res = parseParamArgument(state, 'resonance');
      const input = parseNodeArgument(state);
      node = createLowPass(c, res, input);
      break;
    }
    case 'gain': {
      const g = parseParamArgument(state, 'gain');
      const input = parseNodeArgument(state);
      node = createGain(g, input);
      break;
    }
    case 'mix': {
      const ratio = parseParamArgument(state, 'ratio');
      const in1 = parseNodeArgument(state);
      const in2 = parseNodeArgument(state);
      node = createMix(ratio, in1, in2);
      break;
    }
    case 'fm': {
      const c = parseParamArgument(state, 'carrier');
      const mf = parseParamArgument(state, 'modFreq');
      const mi = parseParamArgument(state, 'modIndex');
      node = createFM(c, mf, mi);
      break;
    }
    case 'hole': {
      const holeId = match(state, 'String') ? consume(state).value : 'audio_hole';
      node = createAudioHole(holeId);
      break;
    }
    case 'macro': {
      const macroName = match(state, 'String') ? consume(state).value : 'audio_macro';
      node = createAudioMacro(macroName, {});
      break;
    }
    default: {
      const macroName = base;
      const args: Record<string, any> = {};
      let argIdx = 0;
      while (!match(state, 'Delimiter', ')') && !match(state, 'EOF') && !match(state, 'Keyword')) {
        const arg = parseExpressionOrParam(state);
        const argKey = isFreeParam(arg) ? arg.id : `arg_${argIdx++}`;
        args[argKey] = arg;
      }
      node = createAudioMacro(macroName, args);
      break;
    }
  }

  const endSpan = peek(state, -1).span;
  recordSpan(state, node, combineSpans(startSpan, endSpan));
  return node;
}

function parseHardwareConstructor(state: ParserState, base: string, startSpan: SourceSpan): HardwareNode {
  let node: HardwareNode;

  switch (base) {
    case 'reg': {
      const name = match(state, 'String') ? consume(state).value : 'rax';
      node = createReg(name);
      break;
    }
    case 'imm': {
      const val = parseNumberValue(state).val;
      node = createImm(val);
      break;
    }
    case 'add':
    case 'sub':
    case 'mul': {
      let dst = 'out';
      if (match(state, 'String')) {
        dst = consume(state).value;
      }
      const s1 = parseNodeArgument(state);
      const s2 = parseNodeArgument(state);
      if (base === 'add') node = createAdd(s1, s2, dst);
      else if (base === 'sub') node = createSub(s1, s2, dst);
      else node = createMul(s1, s2, dst);
      break;
    }
    case 'shl':
    case 'shr': {
      let dst = 'out';
      if (match(state, 'String')) {
        dst = consume(state).value;
      }
      const src = parseNodeArgument(state);
      const shift = parseNumberValue(state).val;
      if (base === 'shl') node = createShl(src, shift, dst);
      else node = createShr(src, shift, dst);
      break;
    }
    case 'seq': {
      const first = parseNodeArgument(state);
      const second = parseNodeArgument(state);
      node = createSeq(first, second);
      break;
    }
    case 'hole': {
      const holeId = match(state, 'String') ? consume(state).value : 'hw_hole';
      node = createHardwareHole(holeId);
      break;
    }
    case 'macro': {
      const macroName = match(state, 'String') ? consume(state).value : 'hw_macro';
      node = createHardwareMacro(macroName, {});
      break;
    }
    default: {
      node = createHardwareMacro(base, {});
      break;
    }
  }

  const endSpan = peek(state, -1).span;
  recordSpan(state, node, combineSpans(startSpan, endSpan));
  return node;
}

const SDF_NAMES = new Set([
  'sphere',
  'box',
  'cylinder',
  'torus',
  'plane',
  'translate',
  'rotate',
  'scale',
  'union',
  'smooth_union',
  'smoothunion',
  'intersection',
  'difference',
]);

const AUDIO_NAMES = new Set([
  'sine',
  'saw',
  'square',
  'noise',
  'adsr',
  'lowpass',
  'gain',
  'mix',
  'fm',
]);

const HW_NAMES = new Set(['reg', 'imm', 'add', 'sub', 'mul', 'shl', 'shr', 'seq']);

function parseExpression(state: ParserState): any {
  if (match(state, 'Delimiter', '(')) {
    const openTok = consume(state);
    const inner = parseExpression(state);
    const closeTok = expect(state, 'Delimiter', ')');
    const span = combineSpans(openTok.span, closeTok.span);
    recordSpan(state, inner, span);
    return inner;
  }

  if (match(state, 'Delimiter', '⟨')) {
    return parseAnonymousParam(state);
  }
  if (match(state, 'Delimiter', '{')) {
    return parseRecordParam(state);
  }

  const { raw, base, span: nameSpan } = parseQualifiedName(state);

  if (base === 'mk' || raw.includes('FreeParam')) {
    return parseExplicitFreeParam(state, nameSpan);
  }

  if (base === 'hole' || base === 'macro') {
    if (raw.startsWith('AudioNode') || state.domainType === 'AudioNode') {
      return parseAudioConstructor(state, base, nameSpan);
    }
    if (raw.startsWith('SDFNode') || state.domainType === 'SDFNode') {
      return parseSDFConstructor(state, base, nameSpan);
    }
    if (raw.startsWith('HardwareNode') || state.domainType === 'HardwareNode') {
      return parseHardwareConstructor(state, base, nameSpan);
    }
    const id = match(state, 'String') ? consume(state).value : 'generic';
    return base === 'hole' ? createHole(id) : createMacroCall(id, {}, 'unknown');
  }

  if (raw.startsWith('SDFNode') || SDF_NAMES.has(base)) {
    return parseSDFConstructor(state, base, nameSpan);
  }
  if (raw.startsWith('AudioNode') || AUDIO_NAMES.has(base)) {
    return parseAudioConstructor(state, base, nameSpan);
  }
  if (raw.startsWith('HardwareNode') || HW_NAMES.has(base)) {
    return parseHardwareConstructor(state, base, nameSpan);
  }

  if (base === 'audio') {
    const node = parseNodeArgument(state);
    return createUniversalAudio(node);
  }
  if (base === 'sdf') {
    const node = parseNodeArgument(state);
    return createUniversalSDF(node);
  }
  if (base === 'hardware') {
    const node = parseNodeArgument(state);
    return createUniversalHardware(node);
  }

  // Fallback: check letEnv or treat as macro
  if (state.letEnv.has(base)) {
    return state.letEnv.get(base);
  }

  return createMacroCall(base, {}, 'unknown');
}

/**
 * Parses Lean 4 source code into a typed AST and SourceSpan coordinate map.
 */
export function parseLeanCode(source: string): ParseResult {
  const { tokens, errors } = tokenize(source);
  const state: ParserState = {
    tokens,
    pos: 0,
    length: tokens.length,
    spans: new Map(),
    errors: [...errors],
    letEnv: new Map(),
    nodeToSpanMap: new WeakMap(),
    nodeToIdMap: new WeakMap(),
    nodeCounter: 0,
    paramCounter: 0,
  };

  skipPreamble(state);

  let declarationName = 'main';
  let domainType = '';

  if (match(state, 'Keyword', 'def')) {
    consume(state);
    declarationName = expect(state, 'Identifier').value;

    if (match(state, 'Operator', ':')) {
      consume(state);
      domainType = parseQualifiedName(state).raw;
      state.domainType = domainType;
    }
    if (match(state, 'Operator', ':=')) {
      consume(state);
    }
  }

  parseLetBindings(state);
  const ast = parseExpression(state);

  if (!domainType) {
    if (ast && 'kind' in ast) {
      if (AUDIO_NAMES.has(ast.kind)) domainType = 'AudioNode';
      else if (SDF_NAMES.has(ast.kind)) domainType = 'SDFNode';
      else if (HW_NAMES.has(ast.kind)) domainType = 'HardwareNode';
      else domainType = 'UniversalNode';
    } else {
      domainType = 'ASTNode';
    }
  }

  // Set root span
  if (tokens.length > 0) {
    const firstTok = tokens[0];
    const lastNonEof = tokens.length > 1 ? tokens[tokens.length - 2] : tokens[0];
    state.spans.set('root', combineSpans(firstTok.span, lastNonEof.span));
  }

  // Attach metadata to spans map for convenience
  (state.spans as any).nodeSpanMap = state.nodeToSpanMap;
  (state.spans as any).nodeIdMap = state.nodeToIdMap;
  (state.spans as any).declarationName = declarationName;
  (state.spans as any).domainType = domainType;

  return {
    declarationName,
    domainType,
    ast,
    spans: state.spans,
    errors: state.errors,
    nodeSpanMap: state.nodeToSpanMap,
    nodeIdMap: state.nodeToIdMap,
  };
}

/**
 * Parses Lean 4 source code and throws on syntax errors.
 */
export function parseLeanCodeOrThrow(source: string): ParseResult {
  const result = parseLeanCode(source);
  if (result.errors.length > 0) {
    const first = result.errors[0];
    throw new Error(`ParseError at line ${first.line}, col ${first.col}: ${first.message}`);
  }
  return result;
}

/**
 * Retrieve the source span of an AST node from the parse result.
 */
export function getNodeSpan(node: ASTNode, result: ParseResult): SourceSpan | undefined {
  if (!node) return undefined;
  if (result.nodeSpanMap?.has(node)) {
    return result.nodeSpanMap.get(node);
  }
  if ('holeId' in node && result.spans.has((node as any).holeId)) {
    return result.spans.get((node as any).holeId);
  }
  return undefined;
}

/**
 * Retrieve the unique identifier of an AST node from the parse result.
 */
export function getNodeId(node: ASTNode, result: ParseResult): string | undefined {
  if (!node) return undefined;
  if (result.nodeIdMap?.has(node)) {
    return result.nodeIdMap.get(node);
  }
  if ('holeId' in node) {
    return (node as any).holeId;
  }
  return undefined;
}

/**
 * Retrieve the source span of a FreeParam from the parse result.
 */
export function getParamSpan(param: FreeParam, result: ParseResult): SourceSpan | undefined {
  if (!param) return undefined;
  if (result.nodeSpanMap?.has(param)) {
    return result.nodeSpanMap.get(param);
  }
  return result.spans.get(param.id);
}
