/**
 * Type definitions for the Generalized Hyperdimensional Function Plotter.
 *
 * Models multi-dimensional functions f: R^n -> R^m, dimension descriptors,
 * 2D lattice slice evaluation results, and colormap options.
 */

export interface DimensionDescriptor {
  readonly id: string;
  readonly name: string;
  readonly min: number;
  readonly max: number;
  readonly defaultValue: number;
}

export interface HyperdimFunction {
  readonly id: string;
  readonly name: string;
  readonly inputDim: number;
  readonly outputDim: number;
  readonly dimensions: DimensionDescriptor[];
  readonly evaluate: (coords: Float32Array | number[]) => number | number[];
}

export interface GridEvaluationResult {
  readonly resolutionX: number;
  readonly resolutionY: number;
  readonly axisX: number;
  readonly axisY: number;
  readonly values: Float32Array;
  readonly minVal: number;
  readonly maxVal: number;
  readonly sliceCoords: Float32Array;
}

export type ColorMapType = 'viridis' | 'turbo' | 'magma' | 'coolwarm';

export type HyperdimDisplayMode = '2d' | '3d' | 'split';
