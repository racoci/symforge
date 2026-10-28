/**
 * Client-Side Contextual PCFG/PHOG Sampler - Type Definitions.
 *
 * Defines curriculum complexity levels, probabilistic grammar production rules,
 * variation candidates, and sampling options for scoped sub-expression generation.
 */

import type { ASTNode } from '../ast/types';
import type { DomainType } from '../generator/types';

/**
 * 5-level curriculum stratification representing program complexity hierarchy.
 * - L1: Atomic terminals (depth 1, nodeCount 1)
 * - L2: Unary transforms wrapping L1 (depth 2, nodeCount 2)
 * - L3: Binary combinators over atomics/unary (depth 2-3, nodeCount 3-4)
 * - L4: Hierarchical compositions (depth 3-4, nodeCount 5-7)
 * - L5: Structural abstractions & compound pipelines (depth >= 4, nodeCount >= 7)
 */
export type CurriculumLevel = 'L1' | 'L2' | 'L3' | 'L4' | 'L5';

/**
 * Production rule in the probabilistic context-free grammar.
 */
export interface GrammarProductionRule {
  nonTerminal: string;
  constructorName: string;
  probability: number;
  childNonTerminals: string[];
  isTerminal: boolean;
  domain: DomainType;
}

/**
 * Candidate AST variation generated for a punctured sub-expression hole.
 */
export interface VariationCandidate {
  id: string;
  ast: ASTNode;
  leanCode: string;
  complexityLevel: CurriculumLevel;
  depth: number;
  nodeCount: number;
  domain: DomainType;
  probability: number;
  label: string;
}

/**
 * Configuration options for the probabilistic sampler.
 */
export interface SamplerOptions {
  countPerLevel?: number;
  levels?: CurriculumLevel[];
  seed?: number;
  maxDepth?: number;
  maxSteps?: number;
}
