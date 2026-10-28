/**
 * Unit & Integration Tests for Type Resolution, Hole Puncturing & Structural Context Extractor.
 *
 * Verifies typed hole puncturing (Hole[T]), ancestral context tuple extraction
 * (Parent, Grandparent, LeftSibling, Depth), functional AST immutability,
 * round-trip substitution invariance, domain type validation, and executes
 * the 30-case Pass-to-Complete Gate across SDF, Audio, and Hardware domains.
 */

import { describe, it, expect } from 'vitest';
import type { ASTNode } from '../ast/types';
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
  createReg,
  createImm,
  createAdd,
  createSub,
  createMul,
  createSeq,
  createUniversalAudio,
  createUniversalSDF,
  createUniversalHardware,
  createUniversalMacro,
  createHole,
  createSDFHole,
  createAudioHole,
  createHardwareHole,
  areNodesEqual,
} from '../ast/models';
import type { NodeMatchResult, DomainType } from './types';
import {
  resolveDomainType,
  resolveExpectedType,
  extractStructuralContext,
  punctureASTHole,
  substituteASTHole,
} from './holePuncturing';
import { parseLeanCode } from '../parser/leanParser';
import { findEnclosingASTNode } from './selectionMapper';

// Helper to construct a minimal dummy SourceSpan
function createDummySpan() {
  return {
    startLine: 1,
    startCol: 1,
    endLine: 1,
    endCol: 10,
    startOffset: 0,
    endOffset: 10,
  };
}

// Helper to construct a mock NodeMatchResult
function createMockMatchResult(
  targetNode: ASTNode,
  nodePath: string[] = [],
  parentNode: ASTNode | null = null,
  parentRelation: string | null = null
): NodeMatchResult {
  return {
    targetNode,
    nodeId: (targetNode as any).id || (targetNode as any).holeId || `node_${targetNode.kind}`,
    nodePath,
    span: createDummySpan(),
    isExactMatch: true,
    overlapRatio: 1.0,
    parentNode,
    parentRelation,
    enclosingDeclaration: 'testDecl',
  };
}

describe('Domain Type & Expected Type Resolution', () => {
  it('correctly resolves SDF primitive and composite domains', () => {
    const sphere = createSphere(createFreeParam('r', 0.1, 2.0, 1.0));
    const box = createBox(
      createFreeParam('w', 0.1, 2.0, 1.0),
      createFreeParam('h', 0.1, 2.0, 1.0),
      createFreeParam('d', 0.1, 2.0, 1.0)
    );
    const diff = createDifference(box, sphere);
    const trans = createTranslate(
      createFreeParam('x', 0, 1, 0),
      createFreeParam('y', 0, 1, 0),
      createFreeParam('z', 0, 1, 0),
      sphere
    );
    const sdfHole = createSDFHole('h_sdf');

    expect(resolveDomainType(sphere)).toBe('sdf');
    expect(resolveDomainType(box)).toBe('sdf');
    expect(resolveDomainType(diff)).toBe('sdf');
    expect(resolveDomainType(trans)).toBe('sdf');
    expect(resolveDomainType(sdfHole)).toBe('sdf');
    expect(resolveExpectedType('sdf')).toBe('SDFNode');
  });

  it('correctly resolves Audio primitive and composite domains', () => {
    const freq = createFreeParam('f', 20, 20000, 440);
    const saw = createSaw(freq);
    const sine = createSine(freq);
    const lowpass = createLowPass(createFreeParam('c', 20, 20000, 1000), createFreeParam('q', 0, 1, 0.5), saw);
    const audioHole = createAudioHole('h_audio');

    expect(resolveDomainType(saw)).toBe('audio');
    expect(resolveDomainType(sine)).toBe('audio');
    expect(resolveDomainType(lowpass)).toBe('audio');
    expect(resolveDomainType(audioHole)).toBe('audio');
    expect(resolveExpectedType('audio')).toBe('AudioNode');
  });

  it('correctly resolves Hardware primitive and composite domains', () => {
    const r0 = createReg('r0');
    const r1 = createReg('r1');
    const add = createAdd(r0, r1);
    const seq = createSeq(add, r0);
    const hwHole = createHardwareHole('h_hw');

    expect(resolveDomainType(r0)).toBe('hardware');
    expect(resolveDomainType(add)).toBe('hardware');
    expect(resolveDomainType(seq)).toBe('hardware');
    expect(resolveDomainType(hwHole)).toBe('hardware');
    expect(resolveExpectedType('hardware')).toBe('HardwareNode');
  });

  it('correctly resolves Universal containers and generic holes', () => {
    const genericHole = createHole('gen_1');
    const uSDF = createUniversalSDF(createSphere(createFreeParam('r', 0.1, 2.0, 1.0)));
    const uAudio = createUniversalAudio(createSaw(createFreeParam('f', 20, 20000, 440)));
    const uHW = createUniversalHardware(createReg('r0'));
    const uMacro = createUniversalMacro('macro1', []);

    expect(resolveDomainType(genericHole)).toBe('universal');
    expect(resolveDomainType(uSDF)).toBe('sdf');
    expect(resolveDomainType(uAudio)).toBe('audio');
    expect(resolveDomainType(uHW)).toBe('hardware');
    expect(resolveDomainType(uMacro)).toBe('universal');
    expect(resolveExpectedType('universal')).toBe('UniversalNode');
  });
});

describe('Core Specification Test Cases', () => {
  it('Leaf node puncturing: In Difference(Box, Cylinder), selecting Cylinder produces Difference(Box, Hole("hole_1"))', () => {
    const box = createBox(
      createFreeParam('w', 0.1, 5.0, 1.0),
      createFreeParam('h', 0.1, 5.0, 1.0),
      createFreeParam('d', 0.1, 5.0, 1.0)
    );
    const cylinder = createCylinder(
      createFreeParam('r', 0.1, 5.0, 0.5),
      createFreeParam('h', 0.1, 5.0, 2.0)
    );
    const rootAST = createDifference(box, cylinder);

    const matchResult = createMockMatchResult(cylinder, ['b'], rootAST, 'b');
    const result = punctureASTHole(rootAST, matchResult, 'hole_1');

    expect(result.holeId).toBe('hole_1');
    expect(result.holeNode.kind).toBe('hole');
    expect((result.holeNode as any).domain).toBe('sdf');

    // Expected context: parentOp: "difference", grandparentOp: "root", leftSiblingOp: "box", depth: 2
    expect(result.context.parentOp).toBe('difference');
    expect(result.context.grandparentOp).toBe('root');
    expect(result.context.leftSiblingOp).toBe('box');
    expect(result.context.targetDepth).toBe(2);
    expect(result.context.depth).toBe(2);
    expect(result.context.expectedDomain).toBe('sdf');
    expect(result.context.expectedType).toBe('SDFNode');
    expect(result.context.parentRelation).toBe('b');

    // Expected punctured AST: Difference(Box, Hole("hole_1"))
    const expectedPunctured = createDifference(box, createSDFHole('hole_1'));
    expect(areNodesEqual(result.puncturedRootAST, expectedPunctured)).toBe(true);

    // Verify original rootAST was NOT modified
    expect(areNodesEqual(rootAST, createDifference(box, cylinder))).toBe(true);
    expect(rootAST.b).toBe(cylinder);
  });

  it('Unary transform puncturing: In Translate(Sphere), selecting Translate produces Hole("hole_1") with parentOp: "root"', () => {
    const sphere = createSphere(createFreeParam('r', 0.1, 5.0, 1.0));
    const rootAST = createTranslate(
      createFreeParam('x', 0, 1, 0),
      createFreeParam('y', 0, 1, 0),
      createFreeParam('z', 0, 1, 0),
      sphere
    );

    // User selects the root Translate node
    const matchResult = createMockMatchResult(rootAST, [], null, null);
    const result = punctureASTHole(rootAST, matchResult, 'hole_1');

    expect(result.holeId).toBe('hole_1');
    expect(result.holeNode.kind).toBe('hole');
    expect((result.holeNode as any).domain).toBe('sdf');

    // Expected context: parentOp: "root", grandparentOp: null, leftSiblingOp: null, depth: 1
    expect(result.context.parentOp).toBe('root');
    expect(result.context.grandparentOp).toBeNull();
    expect(result.context.leftSiblingOp).toBeNull();
    expect(result.context.targetDepth).toBe(1);
    expect(result.context.depth).toBe(1);
    expect(result.context.expectedDomain).toBe('sdf');
    expect(result.context.expectedType).toBe('SDFNode');
    expect(result.context.parentRelation).toBeNull();

    // Punctured AST is the hole itself
    expect(result.puncturedRootAST).toBe(result.holeNode);
  });

  it('Audio signal chain puncturing: In LowPass(cutoff, res, Saw(freq)), selecting Saw produces LowPass(..., Hole("hole_1")) with parentRelation: "input_node"', () => {
    const saw = createSaw(createFreeParam('f', 20, 20000, 440));
    const cutoff = createFreeParam('cutoff', 20, 20000, 1200);
    const res = createFreeParam('res', 0.1, 10.0, 0.707);
    const rootAST = createLowPass(cutoff, res, saw);

    const matchResult = createMockMatchResult(saw, ['input_node'], rootAST, 'input_node');
    const result = punctureASTHole(rootAST, matchResult, 'hole_1');

    expect(result.holeId).toBe('hole_1');
    expect(result.holeNode.kind).toBe('hole');
    expect((result.holeNode as any).domain).toBe('audio');

    expect(result.context.parentRelation).toBe('input_node');
    expect(result.context.parentOp).toBe('lowpass');
    expect(result.context.grandparentOp).toBe('root');
    expect(result.context.leftSiblingOp).toBeNull();
    expect(result.context.targetDepth).toBe(2);
    expect(result.context.expectedDomain).toBe('audio');
    expect(result.context.expectedType).toBe('AudioNode');

    const expectedPunctured = createLowPass(cutoff, res, createAudioHole('hole_1'));
    expect(areNodesEqual(result.puncturedRootAST, expectedPunctured)).toBe(true);
  });

  it('Hardware sequence puncturing: In Seq(Add(r0, r1, r2), Mul(r3, r0, r4)), selecting Mul produces Seq(Add(...), Hole("hole_1")) with leftSiblingOp: "add"', () => {
    const r0 = createReg('r0');
    const r1 = createReg('r1');
    const r3 = createReg('r3');

    const add = createAdd(r0, r1, 'r2');
    const mul = createMul(r3, r0, 'r4');
    const rootAST = createSeq(add, mul);

    const matchResult = createMockMatchResult(mul, ['second'], rootAST, 'second');
    const result = punctureASTHole(rootAST, matchResult, 'hole_1');

    expect(result.holeId).toBe('hole_1');
    expect(result.holeNode.kind).toBe('hole');
    expect((result.holeNode as any).domain).toBe('hardware');

    expect(result.context.parentOp).toBe('seq');
    expect(result.context.grandparentOp).toBe('root');
    expect(result.context.leftSiblingOp).toBe('add');
    expect(result.context.targetDepth).toBe(2);
    expect(result.context.expectedDomain).toBe('hardware');
    expect(result.context.expectedType).toBe('HardwareNode');
    expect(result.context.parentRelation).toBe('second');

    const expectedPunctured = createSeq(add, createHardwareHole('hole_1'));
    expect(areNodesEqual(result.puncturedRootAST, expectedPunctured)).toBe(true);
  });
});

describe('Round-Trip Substitution Invariance', () => {
  it('preserves complete AST identity across puncture -> substitute cycle for SDF', () => {
    const box = createBox(
      createFreeParam('w', 0.1, 5.0, 1.0),
      createFreeParam('h', 0.1, 5.0, 1.0),
      createFreeParam('d', 0.1, 5.0, 1.0)
    );
    const cylinder = createCylinder(
      createFreeParam('r', 0.1, 5.0, 0.5),
      createFreeParam('h', 0.1, 5.0, 2.0)
    );
    const originalRoot = createDifference(box, cylinder);

    const matchResult = createMockMatchResult(cylinder, ['b'], originalRoot, 'b');
    const punctureRes = punctureASTHole(originalRoot, matchResult, 'hole_sdf_1');

    const substituted = substituteASTHole(
      punctureRes.puncturedRootAST,
      punctureRes.holeId,
      punctureRes.originalTargetNode
    );

    expect(areNodesEqual(substituted, originalRoot)).toBe(true);
  });

  it('preserves complete AST identity across puncture -> substitute cycle for Audio', () => {
    const saw = createSaw(createFreeParam('f', 20, 20000, 440));
    const originalRoot = createLowPass(
      createFreeParam('c', 20, 20000, 1000),
      createFreeParam('r', 0, 1, 0.5),
      saw
    );

    const matchResult = createMockMatchResult(saw, ['input_node'], originalRoot, 'input_node');
    const punctureRes = punctureASTHole(originalRoot, matchResult, 'hole_audio_1');

    const substituted = substituteASTHole(
      punctureRes.puncturedRootAST,
      punctureRes.holeId,
      punctureRes.originalTargetNode
    );

    expect(areNodesEqual(substituted, originalRoot)).toBe(true);
  });

  it('preserves complete AST identity across puncture -> substitute cycle for Hardware', () => {
    const r0 = createReg('r0');
    const r1 = createReg('r1');
    const add = createAdd(r0, r1);
    const mul = createMul(r0, r1);
    const originalRoot = createSeq(add, mul);

    const matchResult = createMockMatchResult(mul, ['second'], originalRoot, 'second');
    const punctureRes = punctureASTHole(originalRoot, matchResult, 'hole_hw_1');

    const substituted = substituteASTHole(
      punctureRes.puncturedRootAST,
      punctureRes.holeId,
      punctureRes.originalTargetNode
    );

    expect(areNodesEqual(substituted, originalRoot)).toBe(true);
  });

  it('preserves complete AST identity when root itself is punctured and substituted', () => {
    const sphere = createSphere(createFreeParam('r', 0.1, 5.0, 1.0));
    const trans = createTranslate(
      createFreeParam('x', 0, 1, 0),
      createFreeParam('y', 0, 1, 0),
      createFreeParam('z', 0, 1, 0),
      sphere
    );

    const matchResult = createMockMatchResult(trans, [], null, null);
    const punctureRes = punctureASTHole(trans, matchResult, 'root_hole');

    const substituted = substituteASTHole(
      punctureRes.puncturedRootAST,
      punctureRes.holeId,
      punctureRes.originalTargetNode
    );

    expect(areNodesEqual(substituted, trans)).toBe(true);
  });
});

describe('Type Validation & Domain Guard Enforcement', () => {
  it('throws an error when substituting an AudioNode into an SDF hole', () => {
    const box = createBox(
      createFreeParam('w', 0.1, 5.0, 1.0),
      createFreeParam('h', 0.1, 5.0, 1.0),
      createFreeParam('d', 0.1, 5.0, 1.0)
    );
    const punctured = createDifference(box, createSDFHole('hole_1'));
    const invalidReplacement = createSine(createFreeParam('f', 20, 20000, 440));

    expect(() => {
      substituteASTHole(punctured, 'hole_1', invalidReplacement);
    }).toThrow(/Type mismatch: cannot substitute audio node into sdf hole "hole_1"/);
  });

  it('throws an error when substituting a HardwareNode into an Audio hole', () => {
    const punctured = createLowPass(
      createFreeParam('c', 20, 20000, 1000),
      createFreeParam('r', 0, 1, 0.5),
      createAudioHole('hole_audio_lp')
    );
    const invalidReplacement = createAdd(createReg('r0'), createReg('r1'));

    expect(() => {
      substituteASTHole(punctured, 'hole_audio_lp', invalidReplacement);
    }).toThrow(/Type mismatch: cannot substitute hardware node into audio hole "hole_audio_lp"/);
  });

  it('throws an error when substituting an SDFNode into a Hardware hole', () => {
    const punctured = createSeq(createReg('r0'), createHardwareHole('hole_hw_seq'));
    const invalidReplacement = createSphere(createFreeParam('r', 0.1, 2.0, 1.0));

    expect(() => {
      substituteASTHole(punctured, 'hole_hw_seq', invalidReplacement);
    }).toThrow(/Type mismatch: cannot substitute sdf node into hardware hole "hole_hw_seq"/);
  });

  it('permits valid substitution when replacement node matches the hole domain', () => {
    const punctured = createDifference(
      createBox(
        createFreeParam('w', 0.1, 5.0, 1.0),
        createFreeParam('h', 0.1, 5.0, 1.0),
        createFreeParam('d', 0.1, 5.0, 1.0)
      ),
      createSDFHole('hole_1')
    );
    const validReplacement = createSphere(createFreeParam('r', 0.1, 2.0, 1.0));

    const substituted = substituteASTHole(punctured, 'hole_1', validReplacement);
    expect(substituted.kind).toBe('difference');
    expect((substituted as any).b.kind).toBe('sphere');
  });

  it('permits substitution into universal hole without domain restrictions', () => {
    const genericHole = createHole('gen_hole');
    const audioRep = createSaw(createFreeParam('f', 20, 20000, 440));
    const result = substituteASTHole(genericHole, 'gen_hole', audioRep);
    expect(result).toBe(audioRep);
  });
});

describe('Deep Structural Nesting & Ancestral Grandparent Tracking', () => {
  it('correctly tracks grandparentOp for depth 3 and depth 4 nodes', () => {
    // Rotate(Translate(Difference(Box, Cylinder)))
    const box = createBox(
      createFreeParam('w', 0.1, 5.0, 1.0),
      createFreeParam('h', 0.1, 5.0, 1.0),
      createFreeParam('d', 0.1, 5.0, 1.0)
    );
    const cylinder = createCylinder(
      createFreeParam('r', 0.1, 5.0, 0.5),
      createFreeParam('h', 0.1, 5.0, 2.0)
    );
    const diff = createDifference(box, cylinder);
    const trans = createTranslate(
      createFreeParam('x', 0, 1, 0),
      createFreeParam('y', 0, 1, 0),
      createFreeParam('z', 0, 1, 0),
      diff
    );
    const rot = createRotate(
      createFreeParam('rx', 0, 1, 0),
      createFreeParam('ry', 0, 1, 0),
      createFreeParam('rz', 0, 1, 0),
      trans
    );

    // Target: Cylinder at depth 4
    const matchResult = createMockMatchResult(cylinder, ['child', 'child', 'b'], diff, 'b');
    const context = extractStructuralContext(rot, matchResult);

    expect(context.targetDepth).toBe(4);
    expect(context.parentOp).toBe('difference');
    expect(context.grandparentOp).toBe('translate');
    expect(context.leftSiblingOp).toBe('box');
    expect(context.parentRelation).toBe('b');

    // Target: Difference at depth 3
    const matchDiff = createMockMatchResult(diff, ['child', 'child'], trans, 'child');
    const contextDiff = extractStructuralContext(rot, matchDiff);

    expect(contextDiff.targetDepth).toBe(3);
    expect(contextDiff.parentOp).toBe('translate');
    expect(contextDiff.grandparentOp).toBe('rotate');
    expect(contextDiff.leftSiblingOp).toBeNull();
    expect(contextDiff.parentRelation).toBe('child');
  });

  it('correctly tracks left siblings in sequential multi-branch chains', () => {
    // Seq(Seq(Add, Sub), Mul)
    const r0 = createReg('r0');
    const r1 = createReg('r1');
    const add = createAdd(r0, r1);
    const sub = createSub(r0, r1);
    const mul = createMul(r0, r1);

    const innerSeq = createSeq(add, sub);
    const outerSeq = createSeq(innerSeq, mul);

    // Target: Sub inside innerSeq
    const matchSub = createMockMatchResult(sub, ['first', 'second'], innerSeq, 'second');
    const contextSub = extractStructuralContext(outerSeq, matchSub);

    expect(contextSub.targetDepth).toBe(3);
    expect(contextSub.parentOp).toBe('seq');
    expect(contextSub.grandparentOp).toBe('seq');
    expect(contextSub.leftSiblingOp).toBe('add');
    expect(contextSub.parentRelation).toBe('second');
  });
});

describe('Integration with Real Lean 4 Parsed ASTs', () => {
  it('correctly punctures and extracts context from real parsed Lean 4 code', () => {
    const code = `def testShape : SDFNode :=
  SDFNode.difference
    (SDFNode.box (FreeParam.mk "w" 1.0 0.1 5.0) (FreeParam.mk "h" 1.0 0.1 5.0) (FreeParam.mk "d" 1.0 0.1 5.0))
    (SDFNode.cylinder (FreeParam.mk "r" 0.5 0.1 2.0) (FreeParam.mk "h" 2.0 0.1 5.0))`;

    const parseResult = parseLeanCode(code);
    expect(parseResult.errors).toHaveLength(0);

    const cylOffset = code.indexOf('SDFNode.cylinder');
    const selection = {
      startLine: 4,
      startCol: 5,
      endLine: 4,
      endCol: 86,
      startOffset: cylOffset,
      endOffset: cylOffset + 30,
    };

    const match = findEnclosingASTNode(parseResult.ast, parseResult.spans, selection, code);
    expect(match).not.toBeNull();
    expect(match!.targetNode.kind).toBe('cylinder');

    const punctureRes = punctureASTHole(parseResult.ast, match!, 'hole_cyl_lean');
    expect(punctureRes.holeId).toBe('hole_cyl_lean');
    expect(punctureRes.context.parentOp).toBe('difference');
    expect(punctureRes.context.grandparentOp).toBe('root');
    expect(punctureRes.context.leftSiblingOp).toBe('box');
    expect(punctureRes.context.targetDepth).toBe(2);

    const restored = substituteASTHole(
      punctureRes.puncturedRootAST,
      punctureRes.holeId,
      punctureRes.originalTargetNode
    );
    expect(areNodesEqual(restored, parseResult.ast)).toBe(true);
  });
});

// =========================================================================
// Pass-to-Complete Gate: 30 Varied AST Subtrees (12 SDF, 12 Audio, 6 Hardware)
// =========================================================================

interface GateCase {
  readonly id: number;
  readonly name: string;
  readonly domain: DomainType;
  readonly expectedType: string;
  readonly build: () => {
    root: ASTNode;
    target: ASTNode;
    nodePath: string[];
    parentRelation: string | null;
    expectedParentOp: string | null;
    expectedGrandparentOp: string | null;
    expectedLeftSiblingOp: string | null;
    expectedDepth: number;
  };
}

describe('Pass-to-Complete Gate: 30 Varied AST Subtrees', () => {
  const fp = (name: string, val = 1.0) => createFreeParam(name, 0.1, 10.0, val);

  const testCases: GateCase[] = [
    // --- 12 SDF Subtrees ---
    {
      id: 1,
      name: 'SDF 1: Leaf Sphere',
      domain: 'sdf',
      expectedType: 'SDFNode',
      build: () => {
        const sphere = createSphere(fp('r'));
        return {
          root: sphere,
          target: sphere,
          nodePath: [],
          parentRelation: null,
          expectedParentOp: 'root',
          expectedGrandparentOp: null,
          expectedLeftSiblingOp: null,
          expectedDepth: 1,
        };
      },
    },
    {
      id: 2,
      name: 'SDF 2: Leaf Box',
      domain: 'sdf',
      expectedType: 'SDFNode',
      build: () => {
        const box = createBox(fp('w'), fp('h'), fp('d'));
        return {
          root: box,
          target: box,
          nodePath: [],
          parentRelation: null,
          expectedParentOp: 'root',
          expectedGrandparentOp: null,
          expectedLeftSiblingOp: null,
          expectedDepth: 1,
        };
      },
    },
    {
      id: 3,
      name: 'SDF 3: Leaf Cylinder',
      domain: 'sdf',
      expectedType: 'SDFNode',
      build: () => {
        const cyl = createCylinder(fp('r'), fp('h'));
        return {
          root: cyl,
          target: cyl,
          nodePath: [],
          parentRelation: null,
          expectedParentOp: 'root',
          expectedGrandparentOp: null,
          expectedLeftSiblingOp: null,
          expectedDepth: 1,
        };
      },
    },
    {
      id: 4,
      name: 'SDF 4: Leaf Torus',
      domain: 'sdf',
      expectedType: 'SDFNode',
      build: () => {
        const torus = createTorus(fp('R'), fp('r'));
        return {
          root: torus,
          target: torus,
          nodePath: [],
          parentRelation: null,
          expectedParentOp: 'root',
          expectedGrandparentOp: null,
          expectedLeftSiblingOp: null,
          expectedDepth: 1,
        };
      },
    },
    {
      id: 5,
      name: 'SDF 5: Leaf Plane',
      domain: 'sdf',
      expectedType: 'SDFNode',
      build: () => {
        const plane = createPlane(fp('offset'));
        return {
          root: plane,
          target: plane,
          nodePath: [],
          parentRelation: null,
          expectedParentOp: 'root',
          expectedGrandparentOp: null,
          expectedLeftSiblingOp: null,
          expectedDepth: 1,
        };
      },
    },
    {
      id: 6,
      name: 'SDF 6: Translate(Sphere)',
      domain: 'sdf',
      expectedType: 'SDFNode',
      build: () => {
        const sphere = createSphere(fp('r'));
        const trans = createTranslate(fp('x'), fp('y'), fp('z'), sphere);
        return {
          root: trans,
          target: sphere,
          nodePath: ['child'],
          parentRelation: 'child',
          expectedParentOp: 'translate',
          expectedGrandparentOp: 'root',
          expectedLeftSiblingOp: null,
          expectedDepth: 2,
        };
      },
    },
    {
      id: 7,
      name: 'SDF 7: Rotate(Box)',
      domain: 'sdf',
      expectedType: 'SDFNode',
      build: () => {
        const box = createBox(fp('w'), fp('h'), fp('d'));
        const rot = createRotate(fp('rx'), fp('ry'), fp('rz'), box);
        return {
          root: rot,
          target: box,
          nodePath: ['child'],
          parentRelation: 'child',
          expectedParentOp: 'rotate',
          expectedGrandparentOp: 'root',
          expectedLeftSiblingOp: null,
          expectedDepth: 2,
        };
      },
    },
    {
      id: 8,
      name: 'SDF 8: Scale(Cylinder)',
      domain: 'sdf',
      expectedType: 'SDFNode',
      build: () => {
        const cyl = createCylinder(fp('r'), fp('h'));
        const scale = createScale(fp('s'), cyl);
        return {
          root: scale,
          target: cyl,
          nodePath: ['child'],
          parentRelation: 'child',
          expectedParentOp: 'scale',
          expectedGrandparentOp: 'root',
          expectedLeftSiblingOp: null,
          expectedDepth: 2,
        };
      },
    },
    {
      id: 9,
      name: 'SDF 9: Union(Sphere, Box) target second operand Box',
      domain: 'sdf',
      expectedType: 'SDFNode',
      build: () => {
        const sphere = createSphere(fp('r'));
        const box = createBox(fp('w'), fp('h'), fp('d'));
        const union = createUnion(fp('k'), sphere, box);
        return {
          root: union,
          target: box,
          nodePath: ['b'],
          parentRelation: 'b',
          expectedParentOp: 'union',
          expectedGrandparentOp: 'root',
          expectedLeftSiblingOp: 'sphere',
          expectedDepth: 2,
        };
      },
    },
    {
      id: 10,
      name: 'SDF 10: SmoothUnion(Torus, Cylinder) target first operand Torus',
      domain: 'sdf',
      expectedType: 'SDFNode',
      build: () => {
        const torus = createTorus(fp('R'), fp('r'));
        const cyl = createCylinder(fp('r'), fp('h'));
        const su = createSmoothUnion(fp('k'), torus, cyl);
        return {
          root: su,
          target: torus,
          nodePath: ['a'],
          parentRelation: 'a',
          expectedParentOp: 'smooth_union',
          expectedGrandparentOp: 'root',
          expectedLeftSiblingOp: null,
          expectedDepth: 2,
        };
      },
    },
    {
      id: 11,
      name: 'SDF 11: Intersection(Box, Sphere) target second operand Sphere',
      domain: 'sdf',
      expectedType: 'SDFNode',
      build: () => {
        const box = createBox(fp('w'), fp('h'), fp('d'));
        const sphere = createSphere(fp('r'));
        const inter = createIntersection(box, sphere);
        return {
          root: inter,
          target: sphere,
          nodePath: ['b'],
          parentRelation: 'b',
          expectedParentOp: 'intersection',
          expectedGrandparentOp: 'root',
          expectedLeftSiblingOp: 'box',
          expectedDepth: 2,
        };
      },
    },
    {
      id: 12,
      name: 'SDF 12: Difference(Box, Translate(Cylinder)) target grandchild Cylinder',
      domain: 'sdf',
      expectedType: 'SDFNode',
      build: () => {
        const box = createBox(fp('w'), fp('h'), fp('d'));
        const cyl = createCylinder(fp('r'), fp('h'));
        const trans = createTranslate(fp('x'), fp('y'), fp('z'), cyl);
        const diff = createDifference(box, trans);
        return {
          root: diff,
          target: cyl,
          nodePath: ['b', 'child'],
          parentRelation: 'child',
          expectedParentOp: 'translate',
          expectedGrandparentOp: 'difference',
          expectedLeftSiblingOp: null,
          expectedDepth: 3,
        };
      },
    },

    // --- 12 Audio Subtrees ---
    {
      id: 13,
      name: 'Audio 1: Leaf Sine',
      domain: 'audio',
      expectedType: 'AudioNode',
      build: () => {
        const sine = createSine(fp('f', 440));
        return {
          root: sine,
          target: sine,
          nodePath: [],
          parentRelation: null,
          expectedParentOp: 'root',
          expectedGrandparentOp: null,
          expectedLeftSiblingOp: null,
          expectedDepth: 1,
        };
      },
    },
    {
      id: 14,
      name: 'Audio 2: Leaf Saw',
      domain: 'audio',
      expectedType: 'AudioNode',
      build: () => {
        const saw = createSaw(fp('f', 220));
        return {
          root: saw,
          target: saw,
          nodePath: [],
          parentRelation: null,
          expectedParentOp: 'root',
          expectedGrandparentOp: null,
          expectedLeftSiblingOp: null,
          expectedDepth: 1,
        };
      },
    },
    {
      id: 15,
      name: 'Audio 3: Leaf Square',
      domain: 'audio',
      expectedType: 'AudioNode',
      build: () => {
        const sq = createSquare(fp('f', 330));
        return {
          root: sq,
          target: sq,
          nodePath: [],
          parentRelation: null,
          expectedParentOp: 'root',
          expectedGrandparentOp: null,
          expectedLeftSiblingOp: null,
          expectedDepth: 1,
        };
      },
    },
    {
      id: 16,
      name: 'Audio 4: Leaf Noise',
      domain: 'audio',
      expectedType: 'AudioNode',
      build: () => {
        const noise = createNoise(12345);
        return {
          root: noise,
          target: noise,
          nodePath: [],
          parentRelation: null,
          expectedParentOp: 'root',
          expectedGrandparentOp: null,
          expectedLeftSiblingOp: null,
          expectedDepth: 1,
        };
      },
    },
    {
      id: 17,
      name: 'Audio 5: Leaf FM',
      domain: 'audio',
      expectedType: 'AudioNode',
      build: () => {
        const fm = createFM(fp('c', 440), fp('m', 220), fp('i', 2.0));
        return {
          root: fm,
          target: fm,
          nodePath: [],
          parentRelation: null,
          expectedParentOp: 'root',
          expectedGrandparentOp: null,
          expectedLeftSiblingOp: null,
          expectedDepth: 1,
        };
      },
    },
    {
      id: 18,
      name: 'Audio 6: Gain(Sine) target Sine',
      domain: 'audio',
      expectedType: 'AudioNode',
      build: () => {
        const sine = createSine(fp('f', 440));
        const gain = createGain(fp('g', 0.5), sine);
        return {
          root: gain,
          target: sine,
          nodePath: ['input_node'],
          parentRelation: 'input_node',
          expectedParentOp: 'gain',
          expectedGrandparentOp: 'root',
          expectedLeftSiblingOp: null,
          expectedDepth: 2,
        };
      },
    },
    {
      id: 19,
      name: 'Audio 7: LowPass(Saw) target Saw',
      domain: 'audio',
      expectedType: 'AudioNode',
      build: () => {
        const saw = createSaw(fp('f', 440));
        const lp = createLowPass(fp('c', 1000), fp('q', 0.7), saw);
        return {
          root: lp,
          target: saw,
          nodePath: ['input_node'],
          parentRelation: 'input_node',
          expectedParentOp: 'lowpass',
          expectedGrandparentOp: 'root',
          expectedLeftSiblingOp: null,
          expectedDepth: 2,
        };
      },
    },
    {
      id: 20,
      name: 'Audio 8: ADSR(Square) target gate Square',
      domain: 'audio',
      expectedType: 'AudioNode',
      build: () => {
        const sq = createSquare(fp('f', 440));
        const adsr = createADSR(fp('a', 0.01), fp('d', 0.1), fp('s', 0.7), fp('r', 0.2), sq);
        return {
          root: adsr,
          target: sq,
          nodePath: ['gate'],
          parentRelation: 'gate',
          expectedParentOp: 'adsr',
          expectedGrandparentOp: 'root',
          expectedLeftSiblingOp: null,
          expectedDepth: 2,
        };
      },
    },
    {
      id: 21,
      name: 'Audio 9: Mix(Sine, Saw) target second input Saw',
      domain: 'audio',
      expectedType: 'AudioNode',
      build: () => {
        const sine = createSine(fp('f1', 440));
        const saw = createSaw(fp('f2', 880));
        const mix = createMix(fp('r', 0.5), sine, saw);
        return {
          root: mix,
          target: saw,
          nodePath: ['input2'],
          parentRelation: 'input2',
          expectedParentOp: 'mix',
          expectedGrandparentOp: 'root',
          expectedLeftSiblingOp: 'sine',
          expectedDepth: 2,
        };
      },
    },
    {
      id: 22,
      name: 'Audio 10: Mix(Sine, Saw) target first input Sine',
      domain: 'audio',
      expectedType: 'AudioNode',
      build: () => {
        const sine = createSine(fp('f1', 440));
        const saw = createSaw(fp('f2', 880));
        const mix = createMix(fp('r', 0.5), sine, saw);
        return {
          root: mix,
          target: sine,
          nodePath: ['input1'],
          parentRelation: 'input1',
          expectedParentOp: 'mix',
          expectedGrandparentOp: 'root',
          expectedLeftSiblingOp: null,
          expectedDepth: 2,
        };
      },
    },
    {
      id: 23,
      name: 'Audio 11: LowPass(Mix(Saw, Square)) target grandchild Square',
      domain: 'audio',
      expectedType: 'AudioNode',
      build: () => {
        const saw = createSaw(fp('f1', 220));
        const sq = createSquare(fp('f2', 440));
        const mix = createMix(fp('r', 0.5), saw, sq);
        const lp = createLowPass(fp('c', 800), fp('q', 0.7), mix);
        return {
          root: lp,
          target: sq,
          nodePath: ['input_node', 'input2'],
          parentRelation: 'input2',
          expectedParentOp: 'mix',
          expectedGrandparentOp: 'lowpass',
          expectedLeftSiblingOp: 'saw',
          expectedDepth: 3,
        };
      },
    },
    {
      id: 24,
      name: 'Audio 12: Gain(LowPass(Noise)) target grandchild Noise',
      domain: 'audio',
      expectedType: 'AudioNode',
      build: () => {
        const noise = createNoise(999);
        const lp = createLowPass(fp('c', 2000), fp('q', 1.0), noise);
        const gain = createGain(fp('g', 0.8), lp);
        return {
          root: gain,
          target: noise,
          nodePath: ['input_node', 'input_node'],
          parentRelation: 'input_node',
          expectedParentOp: 'lowpass',
          expectedGrandparentOp: 'gain',
          expectedLeftSiblingOp: null,
          expectedDepth: 3,
        };
      },
    },

    // --- 6 Hardware Subtrees ---
    {
      id: 25,
      name: 'Hardware 1: Leaf Reg',
      domain: 'hardware',
      expectedType: 'HardwareNode',
      build: () => {
        const reg = createReg('r0');
        return {
          root: reg,
          target: reg,
          nodePath: [],
          parentRelation: null,
          expectedParentOp: 'root',
          expectedGrandparentOp: null,
          expectedLeftSiblingOp: null,
          expectedDepth: 1,
        };
      },
    },
    {
      id: 26,
      name: 'Hardware 2: Leaf Imm',
      domain: 'hardware',
      expectedType: 'HardwareNode',
      build: () => {
        const imm = createImm(42);
        return {
          root: imm,
          target: imm,
          nodePath: [],
          parentRelation: null,
          expectedParentOp: 'root',
          expectedGrandparentOp: null,
          expectedLeftSiblingOp: null,
          expectedDepth: 1,
        };
      },
    },
    {
      id: 27,
      name: 'Hardware 3: Add(Reg, Reg) target second operand',
      domain: 'hardware',
      expectedType: 'HardwareNode',
      build: () => {
        const r0 = createReg('r0');
        const r1 = createReg('r1');
        const add = createAdd(r0, r1);
        return {
          root: add,
          target: r1,
          nodePath: ['src2'],
          parentRelation: 'src2',
          expectedParentOp: 'add',
          expectedGrandparentOp: 'root',
          expectedLeftSiblingOp: 'reg',
          expectedDepth: 2,
        };
      },
    },
    {
      id: 28,
      name: 'Hardware 4: Sub(Reg, Imm) target first operand Reg',
      domain: 'hardware',
      expectedType: 'HardwareNode',
      build: () => {
        const r0 = createReg('r0');
        const imm = createImm(1);
        const sub = createSub(r0, imm);
        return {
          root: sub,
          target: r0,
          nodePath: ['src1'],
          parentRelation: 'src1',
          expectedParentOp: 'sub',
          expectedGrandparentOp: 'root',
          expectedLeftSiblingOp: null,
          expectedDepth: 2,
        };
      },
    },
    {
      id: 29,
      name: 'Hardware 5: Seq(Add, Mul) target second instruction Mul',
      domain: 'hardware',
      expectedType: 'HardwareNode',
      build: () => {
        const r0 = createReg('r0');
        const r1 = createReg('r1');
        const add = createAdd(r0, r1);
        const mul = createMul(r0, r1);
        const seq = createSeq(add, mul);
        return {
          root: seq,
          target: mul,
          nodePath: ['second'],
          parentRelation: 'second',
          expectedParentOp: 'seq',
          expectedGrandparentOp: 'root',
          expectedLeftSiblingOp: 'add',
          expectedDepth: 2,
        };
      },
    },
    {
      id: 30,
      name: 'Hardware 6: Seq(Add, Mul) target grandchild src2 inside Mul',
      domain: 'hardware',
      expectedType: 'HardwareNode',
      build: () => {
        const r0 = createReg('r0');
        const r1 = createReg('r1');
        const r2 = createReg('r2');
        const add = createAdd(r0, r1);
        const mul = createMul(r0, r2);
        const seq = createSeq(add, mul);
        return {
          root: seq,
          target: r2,
          nodePath: ['second', 'src2'],
          parentRelation: 'src2',
          expectedParentOp: 'mul',
          expectedGrandparentOp: 'seq',
          expectedLeftSiblingOp: 'reg',
          expectedDepth: 3,
        };
      },
    },
  ];

  it('validates exactly 30 test cases in the Pass-to-Complete Gate', () => {
    expect(testCases).toHaveLength(30);
    const sdfCases = testCases.filter((c) => c.domain === 'sdf');
    const audioCases = testCases.filter((c) => c.domain === 'audio');
    const hwCases = testCases.filter((c) => c.domain === 'hardware');
    expect(sdfCases).toHaveLength(12);
    expect(audioCases).toHaveLength(12);
    expect(hwCases).toHaveLength(6);
  });

  for (const tc of testCases) {
    it(`Gate #${tc.id}: ${tc.name}`, () => {
      const data = tc.build();
      const matchResult = createMockMatchResult(
        data.target,
        data.nodePath,
        null,
        data.parentRelation
      );

      // 1. Context Extraction Invariants
      const context = extractStructuralContext(data.root, matchResult);
      expect(context.expectedDomain).toBe(tc.domain);
      expect(context.expectedType).toBe(tc.expectedType);
      expect(context.parentOp).toBe(data.expectedParentOp);
      expect(context.grandparentOp).toBe(data.expectedGrandparentOp);
      expect(context.leftSiblingOp).toBe(data.expectedLeftSiblingOp);
      expect(context.targetDepth).toBe(data.expectedDepth);
      expect(context.depth).toBe(data.expectedDepth);
      expect(context.parentRelation).toBe(data.parentRelation);

      // 2. Hole Puncturing Invariants
      const holeId = `gate_hole_${tc.id}`;
      const punctureRes = punctureASTHole(data.root, matchResult, holeId);
      expect(punctureRes.holeId).toBe(holeId);
      expect(punctureRes.holeNode.kind).toBe('hole');
      expect((punctureRes.holeNode as any).domain).toBe(tc.domain);

      // 3. Substitution Round-Trip Invariance
      const restored = substituteASTHole(
        punctureRes.puncturedRootAST,
        holeId,
        punctureRes.originalTargetNode
      );
      expect(areNodesEqual(restored, data.root)).toBe(true);
    });
  }
});
