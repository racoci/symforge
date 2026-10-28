/**
 * Unit & Integration Tests for Selection Bounding & Monaco Range-to-AST Mapper.
 *
 * Verifies exact node selection, composite sub-expression selection,
 * partial selection syntax boundary expansion, whitespace discarding,
 * and executes a 50-sample Pass-to-Complete Gate across Audio, SDF, and Hardware programs.
 */

import { describe, it, expect } from 'vitest';
import { parseLeanCode } from '../parser/leanParser';
import {
  findEnclosingASTNode,
  computeSelectionRange,
  formatNodeDisplayName,
  lineColToOffset,
} from './selectionMapper';
import type { SelectionRange } from './types';

describe('Selection Mapper: lineColToOffset & computeSelectionRange', () => {
  const sampleCode = `def testShape : SDFNode :=
  SDFNode.difference
    (SDFNode.box (FreeParam.mk "w" 1.0 0.1 5.0) (FreeParam.mk "h" 1.0 0.1 5.0) (FreeParam.mk "d" 1.0 0.1 5.0))
    (SDFNode.cylinder (FreeParam.mk "r" 0.5 0.1 2.0) (FreeParam.mk "h" 2.0 0.1 5.0))`;

  it('translates 1-indexed line and col to correct 0-indexed offset', () => {
    // Line 1, col 1 is 'd' at offset 0
    expect(lineColToOffset(sampleCode, 1, 1)).toBe(0);
    // Line 1, col 5 is 't' in testShape at offset 4
    expect(lineColToOffset(sampleCode, 1, 5)).toBe(4);
    // Start of line 2
    const line2Start = sampleCode.indexOf('  SDFNode.difference');
    expect(lineColToOffset(sampleCode, 2, 1)).toBe(line2Start);
  });

  it('computes selection range with automatic offset resolution when offsets are missing', () => {
    const range = computeSelectionRange(
      {
        startLine: 2,
        startCol: 3,
        endLine: 2,
        endCol: 21,
      },
      sampleCode
    );

    expect(range.startOffset).toBeGreaterThan(0);
    expect(range.endOffset).toBeGreaterThan(range.startOffset);
    const sliced = sampleCode.slice(range.startOffset, range.endOffset);
    expect(sliced).toBe('SDFNode.difference');
  });

  it('formats node display names in clean PascalCase with Node suffix', () => {
    expect(formatNodeDisplayName({ kind: 'cylinder' } as any)).toBe('CylinderNode');
    expect(formatNodeDisplayName({ kind: 'difference' } as any)).toBe('DifferenceNode');
    expect(formatNodeDisplayName({ kind: 'lowpass' } as any)).toBe('LowPassNode');
    expect(formatNodeDisplayName({ kind: 'adsr' } as any)).toBe('ADSRNode');
    expect(formatNodeDisplayName({ kind: 'sine' } as any)).toBe('SineNode');
    expect(formatNodeDisplayName({ kind: 'saw' } as any)).toBe('SawNode');
    expect(formatNodeDisplayName({ kind: 'sphere' } as any)).toBe('SphereNode');
    expect(formatNodeDisplayName({ kind: 'box' } as any)).toBe('BoxNode');
  });
});

describe('Selection Mapper: Exact Selections on Atomic Primitives', () => {
  it('identifies exact selection on SphereNode in SDF definition', () => {
    const code = `def mySphere : SDFNode :=
  SDFNode.sphere (FreeParam.mk "radius" 1.25 0.5 3.0)`;

    const parseResult = parseLeanCode(code);
    expect(parseResult.errors).toHaveLength(0);

    const sphereOffsetStart = code.indexOf('SDFNode.sphere');
    const sphereOffsetEnd = code.length;

    const selection: SelectionRange = {
      startLine: 2,
      startCol: 3,
      endLine: 2,
      endCol: code.split('\n')[1].length + 1,
      startOffset: sphereOffsetStart,
      endOffset: sphereOffsetEnd,
    };

    const match = findEnclosingASTNode(parseResult.ast, parseResult.spans, selection, code);
    expect(match).not.toBeNull();
    expect(match!.targetNode.kind).toBe('sphere');
    expect(match!.isExactMatch).toBe(true);
    expect(match!.overlapRatio).toBeCloseTo(1.0, 5);
    expect(match!.parentNode).toBeNull();
    expect(match!.parentRelation).toBeNull();
    expect(match!.nodePath).toEqual([]);
    expect(match!.enclosingDeclaration).toBe('mySphere');
  });

  it('identifies exact selection on SineNode in Audio definition', () => {
    const code = `def mySine : AudioNode :=
  AudioNode.sine (FreeParam.mk "freq" 440.0 20.0 20000.0) (FreeParam.mk "phase" 0.0 0.0 6.28)`;

    const parseResult = parseLeanCode(code);
    expect(parseResult.errors).toHaveLength(0);

    const sineStart = code.indexOf('AudioNode.sine');
    const sineEnd = code.length;

    const selection: SelectionRange = {
      startLine: 2,
      startCol: 3,
      endLine: 2,
      endCol: code.split('\n')[1].length + 1,
      startOffset: sineStart,
      endOffset: sineEnd,
    };

    const match = findEnclosingASTNode(parseResult.ast, parseResult.spans, selection, code);
    expect(match).not.toBeNull();
    expect(match!.targetNode.kind).toBe('sine');
    expect(match!.isExactMatch).toBe(true);
    expect(match!.parentNode).toBeNull();
    expect(match!.parentRelation).toBeNull();
    expect(match!.enclosingDeclaration).toBe('mySine');
  });

  it('identifies exact selection on BoxNode', () => {
    const code = `def myBox : SDFNode :=
  SDFNode.box (FreeParam.mk "w" 1.0 0.1 5.0) (FreeParam.mk "h" 2.0 0.1 5.0) (FreeParam.mk "d" 3.0 0.1 5.0)`;

    const parseResult = parseLeanCode(code);
    const boxStart = code.indexOf('SDFNode.box');
    const boxEnd = code.length;

    const selection: SelectionRange = {
      startLine: 2,
      startCol: 3,
      endLine: 2,
      endCol: code.split('\n')[1].length + 1,
      startOffset: boxStart,
      endOffset: boxEnd,
    };

    const match = findEnclosingASTNode(parseResult.ast, parseResult.spans, selection, code);
    expect(match).not.toBeNull();
    expect(match!.targetNode.kind).toBe('box');
    expect(match!.isExactMatch).toBe(true);
    expect(match!.parentNode).toBeNull();
  });
});

describe('Selection Mapper: Nested Composite Sub-Expression Selection', () => {
  const sdfCompositeCode = `def testDiff : SDFNode :=
  SDFNode.difference
    (SDFNode.box (FreeParam.mk "w" 1.0 0.1 5.0) (FreeParam.mk "h" 1.0 0.1 5.0) (FreeParam.mk "d" 1.0 0.1 5.0))
    (SDFNode.cylinder (FreeParam.mk "r" 0.5 0.1 2.0) (FreeParam.mk "h" 2.0 0.1 5.0))`;

  it('selecting child b in Difference(Box, Cylinder) resolves to CylinderNode with parentRelation: "b"', () => {
    const parseResult = parseLeanCode(sdfCompositeCode);
    expect(parseResult.errors).toHaveLength(0);

    const cylStart = sdfCompositeCode.indexOf('(SDFNode.cylinder');
    const cylEnd = sdfCompositeCode.lastIndexOf('5.0))') + 5;

    const selection: SelectionRange = {
      startLine: 4,
      startCol: 5,
      endLine: 4,
      endCol: 86,
      startOffset: cylStart,
      endOffset: cylEnd,
    };

    const match = findEnclosingASTNode(parseResult.ast, parseResult.spans, selection, sdfCompositeCode);
    expect(match).not.toBeNull();
    expect(match!.targetNode.kind).toBe('cylinder');
    expect(match!.parentNode).not.toBeNull();
    expect(match!.parentNode!.kind).toBe('difference');
    expect(match!.parentRelation).toBe('b');
    expect(match!.nodePath).toEqual(['b']);
    expect(match!.enclosingDeclaration).toBe('testDiff');
  });

  it('selecting child a in Difference(Box, Cylinder) resolves to BoxNode with parentRelation: "a"', () => {
    const parseResult = parseLeanCode(sdfCompositeCode);
    const boxStart = sdfCompositeCode.indexOf('(SDFNode.box');
    const boxEnd = sdfCompositeCode.indexOf('5.0))') + 5;

    const selection: SelectionRange = {
      startLine: 3,
      startCol: 5,
      endLine: 3,
      endCol: 115,
      startOffset: boxStart,
      endOffset: boxEnd,
    };

    const match = findEnclosingASTNode(parseResult.ast, parseResult.spans, selection, sdfCompositeCode);
    expect(match).not.toBeNull();
    expect(match!.targetNode.kind).toBe('box');
    expect(match!.parentNode).not.toBeNull();
    expect(match!.parentNode!.kind).toBe('difference');
    expect(match!.parentRelation).toBe('a');
    expect(match!.nodePath).toEqual(['a']);
  });

  it('selecting inner oscillator in LowPass(cutoff, res, Saw(freq)) resolves to SawNode with parentRelation: "input_node"', () => {
    const audioCode = `def testAudio : AudioNode :=
  AudioNode.lowpass
    (FreeParam.mk "cutoff" 880.0 20.0 20000.0)
    (FreeParam.mk "resonance" 1.414 0.1 10.0)
    (AudioNode.saw (FreeParam.mk "freq" 440.0 20.0 5000.0))`;

    const parseResult = parseLeanCode(audioCode);
    expect(parseResult.errors).toHaveLength(0);

    const sawStart = audioCode.indexOf('(AudioNode.saw');
    const sawEnd = audioCode.indexOf('5000.0))') + 8;

    const selection: SelectionRange = {
      startLine: 5,
      startCol: 5,
      endLine: 5,
      endCol: 60,
      startOffset: sawStart,
      endOffset: sawEnd,
    };

    const match = findEnclosingASTNode(parseResult.ast, parseResult.spans, selection, audioCode);
    expect(match).not.toBeNull();
    expect(match!.targetNode.kind).toBe('saw');
    expect(match!.parentNode).not.toBeNull();
    expect(match!.parentNode!.kind).toBe('lowpass');
    expect(match!.parentRelation).toBe('input_node');
    expect(match!.nodePath).toEqual(['input_node']);
    expect(match!.enclosingDeclaration).toBe('testAudio');
  });

  it('selecting gate oscillator in ADSR resolves with parentRelation: "gate"', () => {
    const adsrCode = `def testEnvelope : AudioNode :=
  AudioNode.adsr
    (FreeParam.mk "a" 0.01 0.001 1.0)
    (FreeParam.mk "d" 0.1 0.001 1.0)
    (FreeParam.mk "s" 0.7 0.0 1.0)
    (FreeParam.mk "r" 0.3 0.001 2.0)
    (AudioNode.square (FreeParam.mk "freq" 220.0 20.0 5000.0) (FreeParam.mk "duty" 0.5 0.1 0.9))`;

    const parseResult = parseLeanCode(adsrCode);
    const sqStart = adsrCode.indexOf('(AudioNode.square');
    const sqEnd = adsrCode.length;

    const selection: SelectionRange = {
      startLine: 7,
      startCol: 5,
      endLine: 7,
      endCol: 100,
      startOffset: sqStart,
      endOffset: sqEnd,
    };

    const match = findEnclosingASTNode(parseResult.ast, parseResult.spans, selection, adsrCode);
    expect(match).not.toBeNull();
    expect(match!.targetNode.kind).toBe('square');
    expect(match!.parentRelation).toBe('gate');
    expect(match!.nodePath).toEqual(['gate']);
  });
});

describe('Selection Mapper: Partial Selection Boundary Expansion', () => {
  const code = `def testShape : SDFNode :=
  SDFNode.difference
    (SDFNode.box (FreeParam.mk "w" 1.0 0.1 5.0) (FreeParam.mk "h" 1.0 0.1 5.0) (FreeParam.mk "d" 1.0 0.1 5.0))
    (SDFNode.cylinder (FreeParam.mk "r" 0.5 0.1 2.0) (FreeParam.mk "h" 2.0 0.1 5.0))`;

  it('highlighting "ylinder 0.5 2.0" automatically expands to enclosing CylinderNode', () => {
    const parseResult = parseLeanCode(code);
    expect(parseResult.errors).toHaveLength(0);

    // Highlight from 'ylinder' to '2.0)' without the prefix '(SDFNode.c'
    const startPartial = code.indexOf('ylinder');
    const endPartial = code.indexOf('2.0)') + 4;

    const selection: SelectionRange = {
      startLine: 4,
      startCol: 19,
      endLine: 4,
      endCol: 85,
      startOffset: startPartial,
      endOffset: endPartial,
    };

    const match = findEnclosingASTNode(parseResult.ast, parseResult.spans, selection, code);
    expect(match).not.toBeNull();
    expect(match!.targetNode.kind).toBe('cylinder');
    expect(match!.parentNode!.kind).toBe('difference');
    expect(match!.parentRelation).toBe('b');
    expect(match!.isExactMatch).toBe(false);
    expect(match!.overlapRatio).toBeCloseTo(1.0, 5);
  });

  it('highlighting parameter body inside SphereNode expands to enclosing SphereNode', () => {
    const sphereCode = `def mySphere : SDFNode :=
  SDFNode.sphere (FreeParam.mk "radius" 1.25 0.5 3.0)`;

    const parseResult = parseLeanCode(sphereCode);
    const paramStart = sphereCode.indexOf('(FreeParam.mk');
    const paramEnd = sphereCode.indexOf('3.0)') + 4;

    const selection: SelectionRange = {
      startLine: 2,
      startCol: 18,
      endLine: 2,
      endCol: 53,
      startOffset: paramStart,
      endOffset: paramEnd,
    };

    const match = findEnclosingASTNode(parseResult.ast, parseResult.spans, selection, sphereCode);
    expect(match).not.toBeNull();
    expect(match!.targetNode.kind).toBe('sphere');
    expect(match!.parentNode).toBeNull();
  });
});

describe('Selection Mapper: Whitespace & Invalid Range Discarding', () => {
  const code = `def testShape : SDFNode :=
  SDFNode.sphere (FreeParam.mk "r" 1.0 0.1 5.0)`;

  it('returns null for zero-length cursor selection', () => {
    const parseResult = parseLeanCode(code);
    const selection: SelectionRange = {
      startLine: 2,
      startCol: 10,
      endLine: 2,
      endCol: 10,
      startOffset: 15,
      endOffset: 15,
    };

    const match = findEnclosingASTNode(parseResult.ast, parseResult.spans, selection, code);
    expect(match).toBeNull();
  });

  it('returns null for pure whitespace selection between lines', () => {
    const multiLineCode = `def testShape : SDFNode :=\n   \n\n  SDFNode.sphere (FreeParam.mk "r" 1.0 0.1 5.0)`;
    const parseResult = parseLeanCode(multiLineCode);

    const wsStart = multiLineCode.indexOf('\n   \n');
    const wsEnd = wsStart + 5;

    const selection: SelectionRange = {
      startLine: 2,
      startCol: 1,
      endLine: 3,
      endCol: 1,
      startOffset: wsStart,
      endOffset: wsEnd,
    };

    const match = findEnclosingASTNode(parseResult.ast, parseResult.spans, selection, multiLineCode);
    expect(match).toBeNull();
  });

  it('returns null when overlap ratio is less than 0.15', () => {
    const parseResult = parseLeanCode(code);
    // Selection starts 100 characters before the code and ends 1 char into the code
    const hugeSelection: SelectionRange = {
      startLine: 1,
      startCol: 1,
      endLine: 1,
      endCol: 2,
      startOffset: 0,
      endOffset: 1, // 'd' in 'def'
    };

    // 'd' does not overlap the AST node (SDFNode.sphere), so overlap ratio is 0
    const match = findEnclosingASTNode(parseResult.ast, parseResult.spans, hugeSelection, code);
    expect(match).toBeNull();
  });
});

describe('Pass-to-Complete Gate: 50 Simulated User Selection Ranges', () => {
  // Test program 1: Complex 4-level SDF Tree
  const complexSDF = `def complexGeometry : SDFNode :=
  SDFNode.union (FreeParam.mk "smooth" 0.2 0.0 1.0)
    (SDFNode.intersection
      (SDFNode.sphere (FreeParam.mk "r1" 2.0 0.1 5.0))
      (SDFNode.box (FreeParam.mk "w" 1.5 0.1 5.0) (FreeParam.mk "h" 1.5 0.1 5.0) (FreeParam.mk "d" 1.5 0.1 5.0)))
    (SDFNode.translate (FreeParam.mk "dx" 1.0 -5.0 5.0) (FreeParam.mk "dy" 0.0 -5.0 5.0) (FreeParam.mk "dz" 0.0 -5.0 5.0)
      (SDFNode.difference
        (SDFNode.cylinder (FreeParam.mk "cr" 0.8 0.1 3.0) (FreeParam.mk "ch" 3.0 0.1 10.0))
        (SDFNode.torus (FreeParam.mk "maj" 1.2 0.1 5.0) (FreeParam.mk "min" 0.3 0.05 1.0))))`;

  // Test program 2: Complex 4-level Procedural Audio Graph
  const complexAudio = `def complexSynthesizer : AudioNode :=
  AudioNode.mix (FreeParam.mk "ratio" 0.5 0.0 1.0)
    (AudioNode.lowpass
      (FreeParam.mk "cutoff" 1200.0 20.0 20000.0)
      (FreeParam.mk "q" 2.0 0.1 10.0)
      (AudioNode.gain (FreeParam.mk "g1" 0.8 0.0 2.0)
        (AudioNode.saw (FreeParam.mk "freq1" 220.0 20.0 5000.0))))
    (AudioNode.adsr
      (FreeParam.mk "att" 0.05 0.001 2.0)
      (FreeParam.mk "dec" 0.2 0.001 2.0)
      (FreeParam.mk "sus" 0.6 0.0 1.0)
      (FreeParam.mk "rel" 0.4 0.001 5.0)
      (AudioNode.sine (FreeParam.mk "freq2" 440.0 20.0 5000.0) (FreeParam.mk "phase" 0.0 0.0 6.28)))`;

  // Test program 3: Hardware Pipeline Sequence
  const complexHardware = `def arithmeticPipeline : HardwareNode :=
  HardwareNode.seq
    (HardwareNode.add
      (HardwareNode.reg "r0")
      (HardwareNode.imm 10))
    (HardwareNode.mul
      (HardwareNode.reg "r1")
      (HardwareNode.imm 25))`;

  const sdfResult = parseLeanCode(complexSDF);
  const audioResult = parseLeanCode(complexAudio);
  const hwResult = parseLeanCode(complexHardware);

  expect(sdfResult.errors).toHaveLength(0);
  expect(audioResult.errors).toHaveLength(0);
  expect(hwResult.errors).toHaveLength(0);

  // Generate 50 simulated user selection scenarios:
  // 20 SDF scenarios, 20 Audio scenarios, 10 Hardware scenarios
  interface SelectionScenario {
    id: number;
    description: string;
    code: string;
    parseResult: typeof sdfResult;
    substringToHighlight: string;
    expectedKind: string;
    expectedRelation: string | null;
  }

  const scenarios: SelectionScenario[] = [
    // ----------------------------------------------------
    // SDF Scenarios (1 - 20)
    // ----------------------------------------------------
    // 1. Root union exact
    {
      id: 1,
      description: 'Exact root union selection',
      code: complexSDF,
      parseResult: sdfResult,
      substringToHighlight: 'SDFNode.union',
      expectedKind: 'union',
      expectedRelation: null,
    },
    // 2. Child a (intersection) exact
    {
      id: 2,
      description: 'Intersection child a selection',
      code: complexSDF,
      parseResult: sdfResult,
      substringToHighlight: 'SDFNode.intersection',
      expectedKind: 'intersection',
      expectedRelation: 'a',
    },
    // 3. Sphere inside intersection
    {
      id: 3,
      description: 'Sphere child a of intersection',
      code: complexSDF,
      parseResult: sdfResult,
      substringToHighlight: 'SDFNode.sphere (FreeParam.mk "r1" 2.0 0.1 5.0)',
      expectedKind: 'sphere',
      expectedRelation: 'a',
    },
    // 4. Partial sphere (boundary expansion)
    {
      id: 4,
      description: 'Partial sphere without prefix',
      code: complexSDF,
      parseResult: sdfResult,
      substringToHighlight: 'phere (FreeParam.mk "r1" 2.0 0.1',
      expectedKind: 'sphere',
      expectedRelation: 'a',
    },
    // 5. Box child b of intersection
    {
      id: 5,
      description: 'Box child b of intersection',
      code: complexSDF,
      parseResult: sdfResult,
      substringToHighlight: 'SDFNode.box (FreeParam.mk "w"',
      expectedKind: 'box',
      expectedRelation: 'b',
    },
    // 6. Partial box height param
    {
      id: 6,
      description: 'Partial box height param',
      code: complexSDF,
      parseResult: sdfResult,
      substringToHighlight: '(FreeParam.mk "h" 1.5 0.1 5.0)',
      expectedKind: 'box',
      expectedRelation: 'b',
    },
    // 7. Child b of union (translate)
    {
      id: 7,
      description: 'Translate child b of union',
      code: complexSDF,
      parseResult: sdfResult,
      substringToHighlight: 'SDFNode.translate',
      expectedKind: 'translate',
      expectedRelation: 'b',
    },
    // 8. Partial translate without prefix
    {
      id: 8,
      description: 'Partial translate without prefix',
      code: complexSDF,
      parseResult: sdfResult,
      substringToHighlight: 'ranslate (FreeParam.mk "dx"',
      expectedKind: 'translate',
      expectedRelation: 'b',
    },
    // 9. Difference inside translate
    {
      id: 9,
      description: 'Difference child of translate',
      code: complexSDF,
      parseResult: sdfResult,
      substringToHighlight: 'SDFNode.difference',
      expectedKind: 'difference',
      expectedRelation: 'child',
    },
    // 10. Cylinder child a of difference
    {
      id: 10,
      description: 'Cylinder child a of difference',
      code: complexSDF,
      parseResult: sdfResult,
      substringToHighlight: 'SDFNode.cylinder (FreeParam.mk "cr" 0.8 0.1 3.0) (FreeParam.mk "ch" 3.0 0.1 10.0)',
      expectedKind: 'cylinder',
      expectedRelation: 'a',
    },
    // 11. Partial cylinder height
    {
      id: 11,
      description: 'Partial cylinder height',
      code: complexSDF,
      parseResult: sdfResult,
      substringToHighlight: 'ylinder (FreeParam.mk "cr"',
      expectedKind: 'cylinder',
      expectedRelation: 'a',
    },
    // 12. Cylinder height parameter alone
    {
      id: 12,
      description: 'Cylinder height parameter alone',
      code: complexSDF,
      parseResult: sdfResult,
      substringToHighlight: '(FreeParam.mk "ch" 3.0 0.1 10.0)',
      expectedKind: 'cylinder',
      expectedRelation: 'a',
    },
    // 13. Torus child b of difference
    {
      id: 13,
      description: 'Torus child b of difference',
      code: complexSDF,
      parseResult: sdfResult,
      substringToHighlight: 'SDFNode.torus (FreeParam.mk "maj" 1.2 0.1 5.0) (FreeParam.mk "min" 0.3 0.05 1.0)',
      expectedKind: 'torus',
      expectedRelation: 'b',
    },
    // 14. Partial torus
    {
      id: 14,
      description: 'Partial torus without prefix',
      code: complexSDF,
      parseResult: sdfResult,
      substringToHighlight: 'orus (FreeParam.mk "maj"',
      expectedKind: 'torus',
      expectedRelation: 'b',
    },
    // 15. Torus minor parameter alone
    {
      id: 15,
      description: 'Torus minor radius param',
      code: complexSDF,
      parseResult: sdfResult,
      substringToHighlight: '(FreeParam.mk "min" 0.3 0.05 1.0)',
      expectedKind: 'torus',
      expectedRelation: 'b',
    },
    // 16. Translate dx param alone
    {
      id: 16,
      description: 'Translate dx param alone',
      code: complexSDF,
      parseResult: sdfResult,
      substringToHighlight: '(FreeParam.mk "dx" 1.0 -5.0 5.0)',
      expectedKind: 'translate',
      expectedRelation: 'b',
    },
    // 17. Translate dz param alone
    {
      id: 17,
      description: 'Translate dz param alone',
      code: complexSDF,
      parseResult: sdfResult,
      substringToHighlight: '(FreeParam.mk "dz" 0.0 -5.0 5.0)',
      expectedKind: 'translate',
      expectedRelation: 'b',
    },
    // 18. Union smooth param alone
    {
      id: 18,
      description: 'Union smooth param alone',
      code: complexSDF,
      parseResult: sdfResult,
      substringToHighlight: '(FreeParam.mk "smooth" 0.2 0.0 1.0)',
      expectedKind: 'union',
      expectedRelation: null,
    },
    // 19. Intersection spanning across box
    {
      id: 19,
      description: 'Intersection spanning from sphere to box',
      code: complexSDF,
      parseResult: sdfResult,
      substringToHighlight: 'SDFNode.intersection\n      (SDFNode.sphere',
      expectedKind: 'intersection',
      expectedRelation: 'a',
    },
    // 20. Difference spanning from cylinder to torus
    {
      id: 20,
      description: 'Difference spanning cylinder to torus',
      code: complexSDF,
      parseResult: sdfResult,
      substringToHighlight: 'SDFNode.difference\n        (SDFNode.cylinder',
      expectedKind: 'difference',
      expectedRelation: 'child',
    },

    // ----------------------------------------------------
    // Audio Scenarios (21 - 40)
    // ----------------------------------------------------
    // 21. Root mix exact
    {
      id: 21,
      description: 'Audio root mix selection',
      code: complexAudio,
      parseResult: audioResult,
      substringToHighlight: 'AudioNode.mix',
      expectedKind: 'mix',
      expectedRelation: null,
    },
    // 22. Lowpass child input1 of mix
    {
      id: 22,
      description: 'Lowpass input1 of mix',
      code: complexAudio,
      parseResult: audioResult,
      substringToHighlight: 'AudioNode.lowpass',
      expectedKind: 'lowpass',
      expectedRelation: 'input1',
    },
    // 23. Lowpass cutoff param alone
    {
      id: 23,
      description: 'Lowpass cutoff param alone',
      code: complexAudio,
      parseResult: audioResult,
      substringToHighlight: '(FreeParam.mk "cutoff" 1200.0 20.0 20000.0)',
      expectedKind: 'lowpass',
      expectedRelation: 'input1',
    },
    // 24. Lowpass resonance q param alone
    {
      id: 24,
      description: 'Lowpass q resonance param',
      code: complexAudio,
      parseResult: audioResult,
      substringToHighlight: '(FreeParam.mk "q" 2.0 0.1 10.0)',
      expectedKind: 'lowpass',
      expectedRelation: 'input1',
    },
    // 25. Gain child input_node of lowpass
    {
      id: 25,
      description: 'Gain input_node of lowpass',
      code: complexAudio,
      parseResult: audioResult,
      substringToHighlight: 'AudioNode.gain (FreeParam.mk "g1" 0.8 0.0 2.0)',
      expectedKind: 'gain',
      expectedRelation: 'input_node',
    },
    // 26. Partial gain
    {
      id: 26,
      description: 'Partial gain without prefix',
      code: complexAudio,
      parseResult: audioResult,
      substringToHighlight: 'ain (FreeParam.mk "g1"',
      expectedKind: 'gain',
      expectedRelation: 'input_node',
    },
    // 27. Saw child input_node of gain
    {
      id: 27,
      description: 'Saw input_node of gain',
      code: complexAudio,
      parseResult: audioResult,
      substringToHighlight: 'AudioNode.saw (FreeParam.mk "freq1" 220.0 20.0 5000.0)',
      expectedKind: 'saw',
      expectedRelation: 'input_node',
    },
    // 28. Partial saw
    {
      id: 28,
      description: 'Partial saw without prefix',
      code: complexAudio,
      parseResult: audioResult,
      substringToHighlight: 'aw (FreeParam.mk "freq1"',
      expectedKind: 'saw',
      expectedRelation: 'input_node',
    },
    // 29. Saw freq parameter alone
    {
      id: 29,
      description: 'Saw frequency param alone',
      code: complexAudio,
      parseResult: audioResult,
      substringToHighlight: '(FreeParam.mk "freq1" 220.0 20.0 5000.0)',
      expectedKind: 'saw',
      expectedRelation: 'input_node',
    },
    // 30. ADSR child input2 of mix
    {
      id: 30,
      description: 'ADSR input2 of mix',
      code: complexAudio,
      parseResult: audioResult,
      substringToHighlight: 'AudioNode.adsr',
      expectedKind: 'adsr',
      expectedRelation: 'input2',
    },
    // 31. ADSR attack param
    {
      id: 31,
      description: 'ADSR attack param',
      code: complexAudio,
      parseResult: audioResult,
      substringToHighlight: '(FreeParam.mk "att" 0.05 0.001 2.0)',
      expectedKind: 'adsr',
      expectedRelation: 'input2',
    },
    // 32. ADSR decay param
    {
      id: 32,
      description: 'ADSR decay param',
      code: complexAudio,
      parseResult: audioResult,
      substringToHighlight: '(FreeParam.mk "dec" 0.2 0.001 2.0)',
      expectedKind: 'adsr',
      expectedRelation: 'input2',
    },
    // 33. ADSR sustain param
    {
      id: 33,
      description: 'ADSR sustain param',
      code: complexAudio,
      parseResult: audioResult,
      substringToHighlight: '(FreeParam.mk "sus" 0.6 0.0 1.0)',
      expectedKind: 'adsr',
      expectedRelation: 'input2',
    },
    // 34. ADSR release param
    {
      id: 34,
      description: 'ADSR release param',
      code: complexAudio,
      parseResult: audioResult,
      substringToHighlight: '(FreeParam.mk "rel" 0.4 0.001 5.0)',
      expectedKind: 'adsr',
      expectedRelation: 'input2',
    },
    // 35. Sine child gate of ADSR
    {
      id: 35,
      description: 'Sine gate of ADSR',
      code: complexAudio,
      parseResult: audioResult,
      substringToHighlight: 'AudioNode.sine (FreeParam.mk "freq2" 440.0 20.0 5000.0) (FreeParam.mk "phase" 0.0 0.0 6.28)',
      expectedKind: 'sine',
      expectedRelation: 'gate',
    },
    // 36. Partial sine
    {
      id: 36,
      description: 'Partial sine without prefix',
      code: complexAudio,
      parseResult: audioResult,
      substringToHighlight: 'ine (FreeParam.mk "freq2"',
      expectedKind: 'sine',
      expectedRelation: 'gate',
    },
    // 37. Sine freq2 param alone
    {
      id: 37,
      description: 'Sine freq2 param alone',
      code: complexAudio,
      parseResult: audioResult,
      substringToHighlight: '(FreeParam.mk "freq2" 440.0 20.0 5000.0)',
      expectedKind: 'sine',
      expectedRelation: 'gate',
    },
    // 38. Sine phase param alone
    {
      id: 38,
      description: 'Sine phase param alone',
      code: complexAudio,
      parseResult: audioResult,
      substringToHighlight: '(FreeParam.mk "phase" 0.0 0.0 6.28)',
      expectedKind: 'sine',
      expectedRelation: 'gate',
    },
    // 39. Mix ratio param alone
    {
      id: 39,
      description: 'Mix ratio param alone',
      code: complexAudio,
      parseResult: audioResult,
      substringToHighlight: '(FreeParam.mk "ratio" 0.5 0.0 1.0)',
      expectedKind: 'mix',
      expectedRelation: null,
    },
    // 40. Spanning lowpass to gain
    {
      id: 40,
      description: 'Spanning lowpass to gain',
      code: complexAudio,
      parseResult: audioResult,
      substringToHighlight: 'AudioNode.lowpass\n      (FreeParam.mk "cutoff"',
      expectedKind: 'lowpass',
      expectedRelation: 'input1',
    },

    // ----------------------------------------------------
    // Hardware Scenarios (41 - 50)
    // ----------------------------------------------------
    // 41. Root seq exact
    {
      id: 41,
      description: 'Hardware seq root selection',
      code: complexHardware,
      parseResult: hwResult,
      substringToHighlight: 'HardwareNode.seq',
      expectedKind: 'seq',
      expectedRelation: null,
    },
    // 42. Add child first of seq
    {
      id: 42,
      description: 'Add first child of seq',
      code: complexHardware,
      parseResult: hwResult,
      substringToHighlight: 'HardwareNode.add',
      expectedKind: 'add',
      expectedRelation: 'first',
    },
    // 43. Reg r0 child src1 of add
    {
      id: 43,
      description: 'Reg r0 src1 of add',
      code: complexHardware,
      parseResult: hwResult,
      substringToHighlight: 'HardwareNode.reg "r0"',
      expectedKind: 'reg',
      expectedRelation: 'src1',
    },
    // 44. Partial reg r0
    {
      id: 44,
      description: 'Partial reg r0',
      code: complexHardware,
      parseResult: hwResult,
      substringToHighlight: 'reg "r0"',
      expectedKind: 'reg',
      expectedRelation: 'src1',
    },
    // 45. Imm 10 child src2 of add
    {
      id: 45,
      description: 'Imm 10 src2 of add',
      code: complexHardware,
      parseResult: hwResult,
      substringToHighlight: 'HardwareNode.imm 10',
      expectedKind: 'imm',
      expectedRelation: 'src2',
    },
    // 46. Partial imm 10
    {
      id: 46,
      description: 'Partial imm 10',
      code: complexHardware,
      parseResult: hwResult,
      substringToHighlight: 'imm 10',
      expectedKind: 'imm',
      expectedRelation: 'src2',
    },
    // 47. Mul child second of seq
    {
      id: 47,
      description: 'Mul second child of seq',
      code: complexHardware,
      parseResult: hwResult,
      substringToHighlight: 'HardwareNode.mul',
      expectedKind: 'mul',
      expectedRelation: 'second',
    },
    // 48. Reg r1 child src1 of mul
    {
      id: 48,
      description: 'Reg r1 src1 of mul',
      code: complexHardware,
      parseResult: hwResult,
      substringToHighlight: 'HardwareNode.reg "r1"',
      expectedKind: 'reg',
      expectedRelation: 'src1',
    },
    // 49. Imm 25 child src2 of mul
    {
      id: 49,
      description: 'Imm 25 src2 of mul',
      code: complexHardware,
      parseResult: hwResult,
      substringToHighlight: 'HardwareNode.imm 25',
      expectedKind: 'imm',
      expectedRelation: 'src2',
    },
    // 50. Partial mul src1 to src2
    {
      id: 50,
      description: 'Partial mul spanning reg to imm',
      code: complexHardware,
      parseResult: hwResult,
      substringToHighlight: 'HardwareNode.mul\n      (HardwareNode.reg "r1")',
      expectedKind: 'mul',
      expectedRelation: 'second',
    },
  ];

  it('verifies all 50 simulated user selection scenarios with 100% precision', () => {
    expect(scenarios).toHaveLength(50);

    for (const scenario of scenarios) {
      const idx = scenario.code.indexOf(scenario.substringToHighlight);
      expect(idx).toBeGreaterThanOrEqual(0);

      const startOffset = idx;
      const endOffset = idx + scenario.substringToHighlight.length;

      const selection: SelectionRange = {
        startLine: 1,
        startCol: 1,
        endLine: 1,
        endCol: 1,
        startOffset,
        endOffset,
      };

      const match = findEnclosingASTNode(
        scenario.parseResult.ast,
        scenario.parseResult.spans,
        selection,
        scenario.code
      );

      expect(match, `Scenario #${scenario.id} (${scenario.description}) failed to match`).not.toBeNull();
      expect(match!.targetNode.kind, `Scenario #${scenario.id} node kind mismatch`).toBe(scenario.expectedKind);
      expect(
        match!.parentRelation,
        `Scenario #${scenario.id} relation mismatch (expected ${scenario.expectedRelation}, got ${match!.parentRelation})`
      ).toBe(scenario.expectedRelation);

      if (scenario.expectedRelation !== null) {
        expect(match!.parentNode, `Scenario #${scenario.id} parentNode should not be null`).not.toBeNull();
      }
    }
  });
});
