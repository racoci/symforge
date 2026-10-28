/**
 * High-Performance Hyperdimensional Function Evaluation Engine.
 *
 * Provides vectorized zero-allocation 2D slice coordinate evaluation over Float32Array
 * buffers, analytical benchmark functions (Rastrigin, Rosenbrock, Ackley, Multi-Wave),
 * AST-derived function generation, and smooth polynomial colormap sampling.
 */

import type {
  DimensionDescriptor,
  HyperdimFunction,
  GridEvaluationResult,
  ColorMapType,
} from './types';
import type { ASTNode, FreeParam } from '../../../core/ast/types';
import {
  collectFreeParams,
  isSDFNode,
  isAudioNode,
} from '../../../core/ast/models';

// ==========================================
// 1. Colormap Polynomial Sampling
// ==========================================

/**
 * Samples a continuous colormap at normalized position t in [0, 1].
 * Returns [r, g, b] with each channel in [0, 1].
 */
export function sampleColorMap(t: number, colormap: ColorMapType): [number, number, number] {
  const x = Math.min(1.0, Math.max(0.0, Number.isFinite(t) ? t : 0.0));

  switch (colormap) {
    case 'turbo': {
      // Google Turbo Colormap 6-term polynomial approximation
      const r =
        0.13572138 +
        x *
          (4.6153926 +
            x *
              (-42.66032258 +
                x * (132.13108234 + x * (-152.94239396 + x * 59.28637943))));
      const g =
        0.09140261 +
        x *
          (2.19418839 +
            x *
              (4.84296658 +
                x * (-14.18503333 + x * (4.27729857 + x * 2.82956604))));
      const b =
        0.10798138 +
        x *
          (8.44732252 +
            x *
              (-40.72076046 +
                x * (126.79153074 + x * (-146.06880065 + x * 54.34182967))));
      return [
        Math.min(1.0, Math.max(0.0, r)),
        Math.min(1.0, Math.max(0.0, g)),
        Math.min(1.0, Math.max(0.0, b)),
      ];
    }

    case 'viridis': {
      // Viridis Colormap smooth 4th-degree polynomial approximation
      const r =
        0.282 - 0.238 * x + 2.618 * x * x - 4.305 * x * x * x + 2.641 * x * x * x * x;
      const g =
        0.006 + 1.277 * x - 0.422 * x * x - 0.298 * x * x * x + 0.354 * x * x * x * x;
      const b =
        0.337 + 1.99 * x - 5.534 * x * x + 7.771 * x * x * x - 4.42 * x * x * x * x;
      return [
        Math.min(1.0, Math.max(0.0, r)),
        Math.min(1.0, Math.max(0.0, g)),
        Math.min(1.0, Math.max(0.0, b)),
      ];
    }

    case 'magma': {
      // Magma Colormap smooth 4th-degree polynomial approximation
      const r =
        0.001 + 0.701 * x + 2.185 * x * x - 3.49 * x * x * x + 1.603 * x * x * x * x;
      const g =
        0.001 - 0.124 * x + 1.638 * x * x - 1.834 * x * x * x + 1.319 * x * x * x * x;
      const b =
        0.015 + 2.025 * x - 5.213 * x * x + 6.381 * x * x * x - 2.508 * x * x * x * x;
      return [
        Math.min(1.0, Math.max(0.0, r)),
        Math.min(1.0, Math.max(0.0, g)),
        Math.min(1.0, Math.max(0.0, b)),
      ];
    }

    case 'coolwarm':
    default: {
      // Kenn Moreland Cool-Warm diverger polynomial approximation
      const r = 0.23 + 0.7 * x + 0.92 * x * x - 1.15 * x * x * x;
      const g = 0.3 + 1.8 * x - 2.8 * x * x + 0.85 * x * x * x;
      const b = 0.75 + 0.8 * x - 2.8 * x * x + 1.4 * x * x * x;
      return [
        Math.min(1.0, Math.max(0.0, r)),
        Math.min(1.0, Math.max(0.0, g)),
        Math.min(1.0, Math.max(0.0, b)),
      ];
    }
  }
}

/**
 * Samples a colormap returning integer RGB values in [0, 255] for Canvas ImageData.
 */
export function sampleColorMap255(
  t: number,
  colormap: ColorMapType
): [number, number, number] {
  const [r, g, b] = sampleColorMap(t, colormap);
  return [
    Math.round(r * 255),
    Math.round(g * 255),
    Math.round(b * 255),
  ];
}

// ==========================================
// 2. Analytical Benchmark Functions
// ==========================================

/**
 * Rastrigin 5D Benchmark Function:
 * f(x) = \sum_{i=0}^4 (x_i^2 - 10 \cos(2\pi x_i) + 10)
 * Global minimum: f(0,0,0,0,0) = 0.
 */
export const Rastrigin5DFunction: HyperdimFunction = {
  id: 'rastrigin-5d',
  name: 'Rastrigin 5D',
  inputDim: 5,
  outputDim: 1,
  dimensions: [
    { id: 'x_0', name: 'Dimension X₀', min: -5.12, max: 5.12, defaultValue: 0.0 },
    { id: 'x_1', name: 'Dimension X₁', min: -5.12, max: 5.12, defaultValue: 0.0 },
    { id: 'x_2', name: 'Dimension X₂', min: -5.12, max: 5.12, defaultValue: 0.0 },
    { id: 'x_3', name: 'Dimension X₃', min: -5.12, max: 5.12, defaultValue: 0.0 },
    { id: 'x_4', name: 'Dimension X₄', min: -5.12, max: 5.12, defaultValue: 0.0 },
  ],
  evaluate: (coords: Float32Array | number[]): number => {
    let sum = 0.0;
    for (let i = 0; i < 5; i++) {
      const xi = coords[i] ?? 0.0;
      sum += xi * xi - 10.0 * Math.cos(2.0 * Math.PI * xi) + 10.0;
    }
    return sum;
  },
};

/**
 * Rosenbrock 4D Benchmark Function:
 * f(x) = \sum_{i=0}^2 (100 (x_{i+1} - x_i^2)^2 + (1 - x_i)^2)
 * Global minimum: f(1,1,1,1) = 0.
 */
export const Rosenbrock4DFunction: HyperdimFunction = {
  id: 'rosenbrock-4d',
  name: 'Rosenbrock 4D',
  inputDim: 4,
  outputDim: 1,
  dimensions: [
    { id: 'x_0', name: 'Dimension X₀', min: -2.048, max: 2.048, defaultValue: 1.0 },
    { id: 'x_1', name: 'Dimension X₁', min: -2.048, max: 2.048, defaultValue: 1.0 },
    { id: 'x_2', name: 'Dimension X₂', min: -2.048, max: 2.048, defaultValue: 1.0 },
    { id: 'x_3', name: 'Dimension X₃', min: -2.048, max: 2.048, defaultValue: 1.0 },
  ],
  evaluate: (coords: Float32Array | number[]): number => {
    let sum = 0.0;
    for (let i = 0; i < 3; i++) {
      const xi = coords[i] ?? 0.0;
      const xNext = coords[i + 1] ?? 0.0;
      const term1 = xNext - xi * xi;
      const term2 = 1.0 - xi;
      sum += 100.0 * term1 * term1 + term2 * term2;
    }
    return sum;
  },
};

/**
 * Ackley 5D Benchmark Function:
 * f(x) = -20 \exp(-0.2 \sqrt{\frac{1}{5} \sum x_i^2}) - \exp(\frac{1}{5} \sum \cos(2\pi x_i)) + 20 + e
 * Global minimum: f(0,0,0,0,0) = 0.
 */
export const Ackley5DFunction: HyperdimFunction = {
  id: 'ackley-5d',
  name: 'Ackley 5D',
  inputDim: 5,
  outputDim: 1,
  dimensions: [
    { id: 'x_0', name: 'Dimension X₀', min: -5.0, max: 5.0, defaultValue: 0.0 },
    { id: 'x_1', name: 'Dimension X₁', min: -5.0, max: 5.0, defaultValue: 0.0 },
    { id: 'x_2', name: 'Dimension X₂', min: -5.0, max: 5.0, defaultValue: 0.0 },
    { id: 'x_3', name: 'Dimension X₃', min: -5.0, max: 5.0, defaultValue: 0.0 },
    { id: 'x_4', name: 'Dimension X₄', min: -5.0, max: 5.0, defaultValue: 0.0 },
  ],
  evaluate: (coords: Float32Array | number[]): number => {
    let sumSq = 0.0;
    let sumCos = 0.0;
    for (let i = 0; i < 5; i++) {
      const xi = coords[i] ?? 0.0;
      sumSq += xi * xi;
      sumCos += Math.cos(2.0 * Math.PI * xi);
    }
    const term1 = -20.0 * Math.exp(-0.2 * Math.sqrt(sumSq / 5.0));
    const term2 = -Math.exp(sumCos / 5.0);
    return term1 + term2 + 20.0 + Math.E;
  },
};

/**
 * Multi-Wave Ripple 5D Benchmark Function:
 * f(x) = \sin(\sqrt{x_0^2 + x_1^2} \cdot \omega) \cdot \exp(-0.08(x_2^2 + x_3^2 + x_4^2))
 * Radial concentric wave modulated by orthogonal quadratic damping.
 */
export const MultiWaveRipple5DFunction: HyperdimFunction = {
  id: 'multi-wave-5d',
  name: 'Multi-Wave Ripple 5D',
  inputDim: 5,
  outputDim: 1,
  dimensions: [
    { id: 'x_0', name: 'Dimension X₀ (Radius U)', min: -4.0, max: 4.0, defaultValue: 0.0 },
    { id: 'x_1', name: 'Dimension X₁ (Radius V)', min: -4.0, max: 4.0, defaultValue: 0.0 },
    { id: 'x_2', name: 'Dimension X₂ (Damping A)', min: -4.0, max: 4.0, defaultValue: 0.0 },
    { id: 'x_3', name: 'Dimension X₃ (Damping B)', min: -4.0, max: 4.0, defaultValue: 0.0 },
    { id: 'x_4', name: 'Dimension X₄ (Damping C)', min: -4.0, max: 4.0, defaultValue: 0.0 },
  ],
  evaluate: (coords: Float32Array | number[]): number => {
    const x0 = coords[0] ?? 0.0;
    const x1 = coords[1] ?? 0.0;
    const r = Math.sqrt(x0 * x0 + x1 * x1);
    const omega = 3.0;
    const ripple = Math.sin(r * omega);

    let quadDamp = 0.0;
    for (let i = 2; i < 5; i++) {
      const xi = coords[i] ?? 0.0;
      quadDamp += xi * xi;
    }
    return ripple * Math.exp(-0.08 * quadDamp);
  },
};

/**
 * All pre-defined analytical benchmark functions.
 */
export const BENCHMARK_FUNCTIONS: HyperdimFunction[] = [
  Rastrigin5DFunction,
  Rosenbrock4DFunction,
  Ackley5DFunction,
  MultiWaveRipple5DFunction,
];

// ==========================================
// 3. AST-Derived Function Generator
// ==========================================

/**
 * Evaluates an AST symbolically or analytically for given parameter values.
 */
function evaluateASTDirect(
  ast: ASTNode,
  paramValues: Record<string, number>,
  spatial: [number, number, number] = [1.0, 0.5, 0.5]
): number {
  if (isSDFNode(ast)) {
    const [x, y, z] = spatial;
    switch (ast.kind) {
      case 'sphere': {
        const r = paramValues[ast.radius.id] ?? ast.radius.defaultVal;
        return Math.sqrt(x * x + y * y + z * z) - r;
      }
      case 'box': {
        const w = paramValues[ast.width.id] ?? ast.width.defaultVal;
        const h = paramValues[ast.height.id] ?? ast.height.defaultVal;
        const d = paramValues[ast.depth.id] ?? ast.depth.defaultVal;
        return Math.max(Math.abs(x) - w * 0.5, Math.abs(y) - h * 0.5, Math.abs(z) - d * 0.5);
      }
      case 'cylinder': {
        const r = paramValues[ast.radius.id] ?? ast.radius.defaultVal;
        const h = paramValues[ast.height.id] ?? ast.height.defaultVal;
        const dXY = Math.sqrt(x * x + y * y) - r;
        const dZ = Math.abs(z) - h * 0.5;
        return Math.max(dXY, dZ);
      }
      case 'torus': {
        const R = paramValues[ast.majorRadius.id] ?? ast.majorRadius.defaultVal;
        const r = paramValues[ast.minorRadius.id] ?? ast.minorRadius.defaultVal;
        const qx = Math.sqrt(x * x + y * y) - R;
        return Math.sqrt(qx * qx + z * z) - r;
      }
      case 'union': {
        const d1 = evaluateASTDirect(ast.a, paramValues, spatial);
        const d2 = evaluateASTDirect(ast.b, paramValues, spatial);
        return Math.min(d1, d2);
      }
      case 'difference': {
        const d1 = evaluateASTDirect(ast.a, paramValues, spatial);
        const d2 = evaluateASTDirect(ast.b, paramValues, spatial);
        return Math.max(d1, -d2);
      }
      case 'intersection': {
        const d1 = evaluateASTDirect(ast.a, paramValues, spatial);
        const d2 = evaluateASTDirect(ast.b, paramValues, spatial);
        return Math.max(d1, d2);
      }
      default:
        break;
    }
  } else if (isAudioNode(ast)) {
    const t = 0.001; // evaluation time point
    switch (ast.kind) {
      case 'sine': {
        const freq = paramValues[ast.freq.id] ?? ast.freq.defaultVal;
        const phase = paramValues[ast.phase.id] ?? ast.phase.defaultVal;
        return Math.sin(2.0 * Math.PI * freq * t + phase);
      }
      case 'saw': {
        const freq = paramValues[ast.freq.id] ?? ast.freq.defaultVal;
        const p = freq * t;
        return 2.0 * (p - Math.floor(p + 0.5));
      }
      case 'square': {
        const freq = paramValues[ast.freq.id] ?? ast.freq.defaultVal;
        return Math.sin(2.0 * Math.PI * freq * t) >= 0.0 ? 1.0 : -1.0;
      }
      case 'gain': {
        const g = paramValues[ast.gain.id] ?? ast.gain.defaultVal;
        return g * evaluateASTDirect(ast.input, paramValues, spatial);
      }
      case 'mix': {
        const ratio = paramValues[ast.ratio.id] ?? ast.ratio.defaultVal;
        const s1 = evaluateASTDirect(ast.input1, paramValues, spatial);
        const s2 = evaluateASTDirect(ast.input2, paramValues, spatial);
        return (1.0 - ratio) * s1 + ratio * s2;
      }
      default:
        break;
    }
  }

  // Generic fallback: sum parameter contributions
  let acc = 0.0;
  for (const val of Object.values(paramValues)) {
    acc += Math.sin(val) + 0.1 * val;
  }
  return acc;
}

/**
 * Constructs a continuous HyperdimFunction from any ASTNode.
 *
 * Extracts FreeParam declarations into orthogonal dimensions x_0 ... x_{n-1}.
 * Guarantees at least 2 dimensions for 2D/3D slice projection.
 */
export function createFunctionFromAST(
  ast: ASTNode,
  customName?: string
): HyperdimFunction {
  const params: FreeParam[] = collectFreeParams(ast);
  const dimensions: DimensionDescriptor[] = [];

  if (params.length === 0) {
    // Zero-parameter AST: provide canonical continuous coordinates
    dimensions.push(
      { id: 'x_0', name: 'Coordinate X₀', min: -5.0, max: 5.0, defaultValue: 0.0 },
      { id: 'x_1', name: 'Coordinate X₁', min: -5.0, max: 5.0, defaultValue: 0.0 }
    );
  } else if (params.length === 1) {
    // 1-parameter AST: pad with orthogonal dimension for 2D slicing
    const p0 = params[0];
    dimensions.push(
      { id: p0.id, name: p0.id, min: p0.minVal, max: p0.maxVal, defaultValue: p0.defaultVal },
      { id: 'x_1', name: 'Coordinate X₁', min: -5.0, max: 5.0, defaultValue: 0.0 }
    );
  } else {
    // Multi-parameter AST: map each FreeParam directly
    for (let i = 0; i < params.length; i++) {
      const p = params[i];
      dimensions.push({
        id: p.id,
        name: p.id,
        min: p.minVal,
        max: p.maxVal,
        defaultValue: p.defaultVal,
      });
    }
  }

  const funcName = customName || `AST[${ast.kind}]`;

  return {
    id: `ast-${ast.kind}-${dimensions.length}d`,
    name: funcName,
    inputDim: dimensions.length,
    outputDim: 1,
    dimensions,
    evaluate: (coords: Float32Array | number[]): number => {
      const paramValues: Record<string, number> = {};
      let anchor = 0.0;

      for (let i = 0; i < dimensions.length; i++) {
        const dim = dimensions[i];
        const val = coords[i] ?? dim.defaultValue;
        paramValues[dim.id] = val;
        // Continuous modulation term ensuring all orthogonal axes shift the landscape
        const diff = val - dim.defaultValue;
        anchor += 0.05 * diff * diff;
      }

      // If dimensions correspond to spatial inputs:
      const sx = coords[0] ?? 1.0;
      const sy = coords[1] ?? 0.5;
      const sz = coords[2] ?? 0.5;
      const baseVal = evaluateASTDirect(ast, paramValues, [sx, sy, sz]);

      return baseVal + anchor;
    },
  };
}

// ==========================================
// 4. Zero-Allocation 2D Slice Evaluator
// ==========================================

/**
 * Evaluates a 2D slice through a hyperdimensional function f: R^n -> R.
 *
 * Vectorized evaluation over uniform coordinate grids using TypedArrays (Float32Array)
 * for zero-allocation performance. The coordinates along axisX and axisY are swept
 * across their respective bounds, while all unselected orthogonal coordinates
 * are pinned to sliceCoords.
 *
 * If outBuffer is provided with sufficient capacity, it is written into directly
 * without allocating new memory.
 */
export function evaluate2DSlice(
  fn: HyperdimFunction,
  axisX: number,
  axisY: number,
  sliceCoords: Float32Array | number[],
  resX: number,
  resY: number,
  outBuffer?: Float32Array
): GridEvaluationResult {
  if (
    axisX < 0 ||
    axisX >= fn.inputDim ||
    axisY < 0 ||
    axisY >= fn.inputDim ||
    axisX === axisY
  ) {
    throw new Error(
      `[evaluate2DSlice] Invalid axis pair (${axisX}, ${axisY}) for inputDim=${fn.inputDim}.`
    );
  }

  const totalPoints = resX * resY;
  const values =
    outBuffer && outBuffer.length >= totalPoints
      ? outBuffer
      : new Float32Array(totalPoints);

  // Single pre-allocated coordinate vector reused across all iterations (zero allocation)
  const evalCoords = new Float32Array(fn.inputDim);
  for (let k = 0; k < fn.inputDim; k++) {
    evalCoords[k] = sliceCoords[k] ?? fn.dimensions[k]?.defaultValue ?? 0.0;
  }

  const dimX = fn.dimensions[axisX];
  const dimY = fn.dimensions[axisY];
  const minX = dimX.min;
  const maxX = dimX.max;
  const minY = dimY.min;
  const maxY = dimY.max;

  const stepX = (maxX - minX) / Math.max(1, resX - 1);
  const stepY = (maxY - minY) / Math.max(1, resY - 1);

  let minVal = Infinity;
  let maxVal = -Infinity;

  for (let j = 0; j < resY; j++) {
    const y = minY + j * stepY;
    evalCoords[axisY] = y;
    const rowOffset = j * resX;

    for (let i = 0; i < resX; i++) {
      const x = minX + i * stepX;
      evalCoords[axisX] = x;

      const evalRes = fn.evaluate(evalCoords);
      const val = typeof evalRes === 'number' ? evalRes : evalRes[0] ?? 0.0;

      values[rowOffset + i] = val;
      if (val < minVal) minVal = val;
      if (val > maxVal) maxVal = val;
    }
  }

  // Handle pathological flat landscapes
  if (!Number.isFinite(minVal)) minVal = 0.0;
  if (!Number.isFinite(maxVal)) maxVal = 0.0;
  if (minVal === maxVal) {
    maxVal = minVal + 1.0;
  }

  return {
    resolutionX: resX,
    resolutionY: resY,
    axisX,
    axisY,
    values,
    minVal,
    maxVal,
    sliceCoords: Float32Array.from(evalCoords),
  };
}
