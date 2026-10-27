/**
 * Lean 4 AST Pretty-Printer.
 *
 * Deterministically formats typed AST nodes back into clean, idiomatic Lean 4 code.
 * Supports configurable indentation, compact mode, and dot-notation constructor formatting.
 */

import type { ASTNode, FreeParam, AudioNode, SDFNode, HardwareNode } from '../ast/types';
import type { FormatOptions } from './types';

/**
 * Format a number as a valid Lean 4 float string (e.g. 1.0 instead of 1).
 */
export function formatFloat(n: number): string {
  if (!Number.isFinite(n)) return '0.0';
  const str = n.toString();
  if (str.includes('.') || str.includes('e') || str.includes('E')) {
    return str;
  }
  return `${str}.0`;
}

/**
 * Format a FreeParam into explicit Lean 4 constructor call.
 */
function formatFreeParam(param: FreeParam): string {
  return `(FreeParam.mk "${param.id}" ${formatFloat(param.minVal)} ${formatFloat(param.maxVal)} ${formatFloat(param.defaultVal)})`;
}

function getPrefix(domain: string, dotNotation: boolean): string {
  if (dotNotation) return '.';
  return `${domain}.`;
}

function formatAudioExpr(ast: AudioNode, options: FormatOptions, isArg: boolean): string {
  const p = getPrefix('AudioNode', !!options.dotNotation);
  let inner = '';

  switch (ast.kind) {
    case 'sine':
      inner = `${p}sine ${formatFreeParam(ast.freq)} ${formatFreeParam(ast.phase)}`;
      break;
    case 'saw':
      inner = `${p}saw ${formatFreeParam(ast.freq)}`;
      break;
    case 'square':
      inner = `${p}square ${formatFreeParam(ast.freq)} ${formatFreeParam(ast.duty)}`;
      break;
    case 'noise':
      inner = `${p}noise ${Math.floor(ast.seed)}`;
      break;
    case 'adsr':
      inner = `${p}adsr ${formatFreeParam(ast.attack)} ${formatFreeParam(ast.decay)} ${formatFreeParam(ast.sustain)} ${formatFreeParam(ast.release)} ${formatAudioExpr(ast.gate, options, true)}`;
      break;
    case 'lowpass':
      inner = `${p}lowpass ${formatFreeParam(ast.cutoff)} ${formatFreeParam(ast.resonance)} ${formatAudioExpr(ast.input, options, true)}`;
      break;
    case 'gain':
      inner = `${p}gain ${formatFreeParam(ast.gain)} ${formatAudioExpr(ast.input, options, true)}`;
      break;
    case 'mix':
      inner = `${p}mix ${formatFreeParam(ast.ratio)} ${formatAudioExpr(ast.input1, options, true)} ${formatAudioExpr(ast.input2, options, true)}`;
      break;
    case 'fm':
      inner = `${p}fm ${formatFreeParam(ast.carrier)} ${formatFreeParam(ast.modFreq)} ${formatFreeParam(ast.modIndex)}`;
      break;
    case 'hole':
      inner = `${p}hole "${ast.holeId}"`;
      break;
    case 'macro':
      inner = `${p}macro "${ast.macroName}"`;
      break;
  }

  return isArg ? `(${inner})` : inner;
}

function formatSDFExpr(ast: SDFNode, options: FormatOptions, isArg: boolean): string {
  const p = getPrefix('SDFNode', !!options.dotNotation);
  let inner = '';

  switch (ast.kind) {
    case 'sphere':
      inner = `${p}sphere ${formatFreeParam(ast.radius)}`;
      break;
    case 'box':
      inner = `${p}box ${formatFreeParam(ast.width)} ${formatFreeParam(ast.height)} ${formatFreeParam(ast.depth)}`;
      break;
    case 'cylinder':
      inner = `${p}cylinder ${formatFreeParam(ast.radius)} ${formatFreeParam(ast.height)}`;
      break;
    case 'torus':
      inner = `${p}torus ${formatFreeParam(ast.majorRadius)} ${formatFreeParam(ast.minorRadius)}`;
      break;
    case 'translate':
      inner = `${p}translate ${formatFreeParam(ast.dx)} ${formatFreeParam(ast.dy)} ${formatFreeParam(ast.dz)} ${formatSDFExpr(ast.child, options, true)}`;
      break;
    case 'rotate':
      inner = `${p}rotate ${formatFreeParam(ast.rx)} ${formatFreeParam(ast.ry)} ${formatFreeParam(ast.rz)} ${formatSDFExpr(ast.child, options, true)}`;
      break;
    case 'scale':
      inner = `${p}scale ${formatFreeParam(ast.s)} ${formatSDFExpr(ast.child, options, true)}`;
      break;
    case 'union':
      inner = `${p}union ${formatFreeParam(ast.smoothness)} ${formatSDFExpr(ast.a, options, true)} ${formatSDFExpr(ast.b, options, true)}`;
      break;
    case 'smooth_union':
      inner = `${p}smooth_union ${formatFreeParam(ast.smoothness)} ${formatSDFExpr(ast.a, options, true)} ${formatSDFExpr(ast.b, options, true)}`;
      break;
    case 'intersection':
      inner = `${p}intersection ${formatSDFExpr(ast.a, options, true)} ${formatSDFExpr(ast.b, options, true)}`;
      break;
    case 'difference':
      inner = `${p}difference ${formatSDFExpr(ast.a, options, true)} ${formatSDFExpr(ast.b, options, true)}`;
      break;
    case 'hole':
      inner = `${p}hole "${ast.holeId}"`;
      break;
    case 'macro':
      inner = `${p}macro "${ast.macroName}"`;
      break;
  }

  return isArg ? `(${inner})` : inner;
}

function formatHardwareExpr(ast: HardwareNode, options: FormatOptions, isArg: boolean): string {
  const p = getPrefix('HardwareNode', !!options.dotNotation);
  let inner = '';

  switch (ast.kind) {
    case 'reg':
      inner = `${p}reg "${ast.name}"`;
      break;
    case 'imm':
      inner = `${p}imm ${Math.floor(ast.val)}`;
      break;
    case 'add':
      inner = `${p}add "${ast.dst ?? 'out'}" ${formatHardwareExpr(ast.src1, options, true)} ${formatHardwareExpr(ast.src2, options, true)}`;
      break;
    case 'sub':
      inner = `${p}sub "${ast.dst ?? 'out'}" ${formatHardwareExpr(ast.src1, options, true)} ${formatHardwareExpr(ast.src2, options, true)}`;
      break;
    case 'mul':
      inner = `${p}mul "${ast.dst ?? 'out'}" ${formatHardwareExpr(ast.src1, options, true)} ${formatHardwareExpr(ast.src2, options, true)}`;
      break;
    case 'shl':
      inner = `${p}shl "${ast.dst ?? 'out'}" ${formatHardwareExpr(ast.src, options, true)} ${Math.floor(ast.shift)}`;
      break;
    case 'shr':
      inner = `${p}shr "${ast.dst ?? 'out'}" ${formatHardwareExpr(ast.src, options, true)} ${Math.floor(ast.shift)}`;
      break;
    case 'seq':
      inner = `${p}seq ${formatHardwareExpr(ast.first, options, true)} ${formatHardwareExpr(ast.second, options, true)}`;
      break;
    case 'hole':
      inner = `${p}hole "${ast.holeId}"`;
      break;
    case 'macro':
      inner = `${p}macro "${ast.macroName}"`;
      break;
  }

  return isArg ? `(${inner})` : inner;
}

function formatUniversalExpr(ast: ASTNode, options: FormatOptions, isArg: boolean): string {
  const p = getPrefix('UniversalNode', !!options.dotNotation);
  let inner = '';

  if (ast.kind === 'universal_audio') {
    inner = `${p}audio ${formatAudioExpr(ast.node, options, true)}`;
  } else if (ast.kind === 'universal_sdf') {
    inner = `${p}sdf ${formatSDFExpr(ast.node, options, true)}`;
  } else if (ast.kind === 'universal_hardware') {
    inner = `${p}hardware ${formatHardwareExpr(ast.node, options, true)}`;
  } else if (ast.kind === 'universal_macro') {
    const childrenStr = ast.children.map((c) => formatUniversalExpr(c, options, true)).join(', ');
    inner = `${p}macroNode "${ast.name}" [${childrenStr}]`;
  }

  return isArg ? `(${inner})` : inner;
}

function formatASTExpression(ast: ASTNode, domainType: string, options: FormatOptions): string {
  if (domainType === 'AudioNode') {
    return formatAudioExpr(ast as AudioNode, options, false);
  }
  if (domainType === 'SDFNode') {
    return formatSDFExpr(ast as SDFNode, options, false);
  }
  if (domainType === 'HardwareNode') {
    return formatHardwareExpr(ast as HardwareNode, options, false);
  }
  return formatUniversalExpr(ast, options, false);
}

function inferDomainType(ast: ASTNode, defaultType?: string): string {
  if (defaultType && defaultType.trim() !== '') return defaultType;
  switch (ast.kind) {
    case 'sine':
    case 'saw':
    case 'square':
    case 'noise':
    case 'adsr':
    case 'lowpass':
    case 'gain':
    case 'mix':
    case 'fm':
      return 'AudioNode';
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
      return 'SDFNode';
    case 'reg':
    case 'imm':
    case 'add':
    case 'sub':
    case 'mul':
    case 'shl':
    case 'shr':
    case 'seq':
      return 'HardwareNode';
    default:
      return 'UniversalNode';
  }
}

/**
 * Deterministic pretty-printer formatting typed AST nodes into idiomatic Lean 4 code.
 */
export function formatLeanCode(
  declarationName: string,
  domainType: string,
  ast: ASTNode,
  options?: FormatOptions
): string {
  const resolvedOptions: FormatOptions = {
    indent: options?.indent ?? 2,
    compact: options?.compact ?? false,
    dotNotation: options?.dotNotation ?? false,
  };

  const resolvedDomain = inferDomainType(ast, domainType);
  const expr = formatASTExpression(ast, resolvedDomain, resolvedOptions);

  if (resolvedOptions.compact) {
    return `def ${declarationName} : ${resolvedDomain} := ${expr}`;
  }

  const indentStr = ' '.repeat(resolvedOptions.indent ?? 2);
  return `def ${declarationName} : ${resolvedDomain} :=\n${indentStr}${expr}`;
}
