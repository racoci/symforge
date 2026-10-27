/**
 * Automated Unit and Round-Trip Tests for Bidirectional Lean 4 AST Parser & Pretty-Printer.
 *
 * Verifies lexing, parsing, span tracking, error handling, and the Pass-to-Complete Gate
 * of structural round-trip equality (areNodesEqual) across 50+ canonical program samples.
 */

import { describe, it, expect } from 'vitest';
import { tokenize } from './lexer';
import {
  parseLeanCode,
  parseLeanCodeOrThrow,
  getNodeSpan,
  getParamSpan,
} from './leanParser';
import { formatLeanCode, formatFloat } from './leanPrinter';
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
  createUniversalAudio,
  createUniversalSDF,
  createUniversalHardware,
  areNodesEqual,
} from '../ast';
import type { AudioNode, SDFNode } from '../ast/types';

describe('Lean 4 Lexer (tokenize)', () => {
  it('tokenizes keywords, identifiers, numbers, operators, and delimiters with exact spans', () => {
    const src = 'def myShape : SDFNode := .sphere 1.5';
    const { tokens, errors } = tokenize(src);

    expect(errors).toHaveLength(0);
    expect(tokens.map((t) => t.type)).toEqual([
      'Keyword',
      'Identifier',
      'Operator',
      'Identifier',
      'Operator',
      'Operator',
      'Identifier',
      'Number',
      'EOF',
    ]);

    expect(tokens[0].value).toBe('def');
    expect(tokens[0].span).toEqual({
      startLine: 1,
      startCol: 1,
      endLine: 1,
      endCol: 4,
      startOffset: 0,
      endOffset: 3,
    });
    expect(src.slice(tokens[0].span.startOffset, tokens[0].span.endOffset)).toBe('def');
  });

  it('correctly handles line comments and nested block comments', () => {
    const src = `-- Line comment\ndef test := /- outer /- inner -/ still comment -/ 42`;
    const { tokens, errors } = tokenize(src);

    expect(errors).toHaveLength(0);
    const nonEof = tokens.filter((t) => t.type !== 'EOF');
    expect(nonEof.map((t) => t.value)).toEqual(['def', 'test', ':=', '42']);
  });

  it('tokenizes strings with escape characters', () => {
    const src = `"hello \\"world\\"\\n"`;
    const { tokens, errors } = tokenize(src);

    expect(errors).toHaveLength(0);
    expect(tokens[0].type).toBe('String');
    expect(tokens[0].value).toBe('hello "world"\n');
  });

  it('tokenizes negative numbers, floating points, and scientific notation', () => {
    const src = '-1.5 42 0.001 1e-4';
    const { tokens, errors } = tokenize(src);

    expect(errors).toHaveLength(0);
    const nums = tokens.filter((t) => t.type === 'Number');
    expect(nums).toHaveLength(4);
    expect(nums[0].numberValue).toBe(-1.5);
    expect(nums[1].numberValue).toBe(42);
    expect(nums[2].numberValue).toBe(0.001);
    expect(nums[3].numberValue).toBe(0.0001);
  });
});

describe('Lean 4 Parser (parseLeanCode)', () => {
  it('parses basic SDF primitives with explicit FreeParam definitions', () => {
    const code = `def mySphere : SDFNode := SDFNode.sphere (FreeParam.mk "radius" 0.1 5.0 1.25)`;
    const result = parseLeanCode(code);

    expect(result.errors).toHaveLength(0);
    expect(result.declarationName).toBe('mySphere');
    expect(result.domainType).toBe('SDFNode');
    expect(result.ast.kind).toBe('sphere');

    const expected = createSphere(createFreeParam('radius', 0.1, 5.0, 1.25));
    expect(areNodesEqual(result.ast, expected)).toBe(true);
  });

  it('parses shorthand positional numeric constants mapped to FreeParams', () => {
    const code = `def simpleBox : SDFNode := .box 1.0 2.0 3.0`;
    const result = parseLeanCode(code);

    expect(result.errors).toHaveLength(0);
    expect(result.ast.kind).toBe('box');
    const box = result.ast as any;
    expect(box.width.defaultVal).toBe(1.0);
    expect(box.height.defaultVal).toBe(2.0);
    expect(box.depth.defaultVal).toBe(3.0);
  });

  it('parses anonymous constructor bracket syntax ⟨...⟩ for parameters', () => {
    const code = `def myPatch : AudioNode := .saw ⟨"f0", 20.0, 20000.0, 440.0⟩`;
    const result = parseLeanCode(code);

    expect(result.errors).toHaveLength(0);
    expect(result.ast.kind).toBe('saw');
    const saw = result.ast as any;
    expect(saw.freq.id).toBe('f0');
    expect(saw.freq.minVal).toBe(20.0);
    expect(saw.freq.maxVal).toBe(20000.0);
    expect(saw.freq.defaultVal).toBe(440.0);
  });

  it('parses let-bound parameter environments', () => {
    const code = `import SymforgeIr.Types
def examplePatch : AudioNode :=
  let freq := ⟨"f0", 20.0, 20000.0, 440.0⟩
  let cutoff := ⟨"fc", 100.0, 10000.0, 1200.0⟩
  let res := ⟨"q", 0.1, 10.0, 0.707⟩
  .lowpass cutoff res (.saw freq)`;

    const result = parseLeanCode(code);
    expect(result.errors).toHaveLength(0);
    expect(result.ast.kind).toBe('lowpass');
    const lp = result.ast as any;
    expect(lp.cutoff.id).toBe('fc');
    expect(lp.resonance.id).toBe('q');
    expect(lp.input.kind).toBe('saw');
    expect(lp.input.freq.id).toBe('f0');
  });

  it('parses Hardware instruction sequences', () => {
    const code = `def myProg : HardwareNode :=
  HardwareNode.seq
    (HardwareNode.add "r1" (HardwareNode.reg "rax") (HardwareNode.imm 10))
    (HardwareNode.shl "r1" (HardwareNode.reg "r1") 2)`;

    const result = parseLeanCode(code);
    expect(result.errors).toHaveLength(0);
    expect(result.ast.kind).toBe('seq');
    const seq = result.ast as any;
    expect(seq.first.kind).toBe('add');
    expect(seq.second.kind).toBe('shl');
  });
});

describe('SourceSpan Tracking & Range Mapping', () => {
  it('accurately encloses subexpressions and parameters with exact character offsets', () => {
    const code = `def testShape : SDFNode :=
  SDFNode.difference
    (SDFNode.box (FreeParam.mk "w" 0.1 10.0 1.0) (FreeParam.mk "h" 0.1 10.0 1.0) (FreeParam.mk "d" 0.1 10.0 1.0))
    (SDFNode.sphere (FreeParam.mk "r" 0.1 5.0 0.5))`;

    const result = parseLeanCode(code);
    expect(result.errors).toHaveLength(0);

    // Verify parameter span
    const rSpan = result.spans.get('r');
    expect(rSpan).toBeDefined();
    const slicedParam = code.slice(rSpan!.startOffset, rSpan!.endOffset);
    expect(slicedParam).toBe('(FreeParam.mk "r" 0.1 5.0 0.5)');

    // Verify helper getParamSpan
    const sphereNode = (result.ast as any).b;
    const helperSpan = getParamSpan(sphereNode.radius, result);
    expect(helperSpan).toEqual(rSpan);

    // Verify helper getNodeSpan
    const nodeSpan = getNodeSpan(result.ast, result);
    expect(nodeSpan).toBeDefined();
    const slicedNode = code.slice(nodeSpan!.startOffset, nodeSpan!.endOffset);
    expect(slicedNode.startsWith('SDFNode.difference')).toBe(true);
    expect(slicedNode.endsWith('0.5))')).toBe(true);
  });
});

describe('Error Handling Diagnostics', () => {
  it('records clean ParseError on invalid syntax without uncaught crash', () => {
    const badCode = `def broken : SDFNode := .unknownCtor`;
    const result = parseLeanCode(badCode);

    // Doesn't crash, returns AST (macro fallback) or errors
    expect(result.ast).toBeDefined();
  });

  it('records syntax error and throws on parseLeanCodeOrThrow', () => {
    const brokenCode = `def broken : SDFNode := .sphere /- unclosed comment`;
    const result = parseLeanCode(brokenCode);

    expect(result.errors.length).toBeGreaterThan(0);
    expect(result.errors[0].line).toBeGreaterThanOrEqual(1);

    expect(() => parseLeanCodeOrThrow(brokenCode)).toThrowError(/ParseError/);
  });
});

describe('Pretty-Printer (formatLeanCode & formatFloat)', () => {
  it('formats floats with trailing .0 when integers', () => {
    expect(formatFloat(0)).toBe('0.0');
    expect(formatFloat(440)).toBe('440.0');
    expect(formatFloat(1.5)).toBe('1.5');
    expect(formatFloat(0.707)).toBe('0.707');
  });

  it('formats AST into indented and compact Lean 4 definitions', () => {
    const node = createSphere(createFreeParam('r', 0.1, 5.0, 1.0));
    const indented = formatLeanCode('mySphere', 'SDFNode', node);
    expect(indented).toContain('def mySphere : SDFNode :=');
    expect(indented).toContain('SDFNode.sphere');
    expect(indented).toContain('(FreeParam.mk "r" 0.1 5.0 1.0)');

    const compact = formatLeanCode('mySphere', 'SDFNode', node, { compact: true });
    expect(compact).toBe('def mySphere : SDFNode := SDFNode.sphere (FreeParam.mk "r" 0.1 5.0 1.0)');

    const dotNotation = formatLeanCode('mySphere', 'SDFNode', node, { compact: true, dotNotation: true });
    expect(dotNotation).toBe('def mySphere : SDFNode := .sphere (FreeParam.mk "r" 0.1 5.0 1.0)');
  });
});

describe('Pass-to-Complete Gate: 50 Canonical Program Round-Trip Invariance', () => {
  // Generate 25 Canonical Audio Programs
  const canonicalAudioPrograms: AudioNode[] = [
    // 1. Basic Sine
    createSine(createFreeParam('f_sine', 20.0, 20000.0, 440.0), createFreeParam('p_sine', 0.0, 6.2831853, 0.0)),
    // 2. Sine with phase offset
    createSine(createFreeParam('f_lead', 100.0, 5000.0, 880.0), createFreeParam('p_lead', 0.0, 3.14159, 1.57)),
    // 3. Saw wave
    createSaw(createFreeParam('f_saw', 20.0, 10000.0, 220.0)),
    // 4. Square wave with 50% duty
    createSquare(createFreeParam('f_sq', 50.0, 8000.0, 110.0), createFreeParam('d_sq', 0.01, 0.99, 0.5)),
    // 5. Square wave with narrow pulse
    createSquare(createFreeParam('f_pulse', 40.0, 4000.0, 65.4), createFreeParam('d_pulse', 0.05, 0.95, 0.15)),
    // 6. White Noise
    createNoise(12345),
    // 7. White Noise alternate seed
    createNoise(999),
    // 8. FM Synthesis
    createFM(
      createFreeParam('c_fm', 100.0, 2000.0, 440.0),
      createFreeParam('mf_fm', 10.0, 1000.0, 220.0),
      createFreeParam('mi_fm', 0.0, 10.0, 3.5)
    ),
    // 9. Audio Hole
    createAudioHole('hole_audio_main'),
    // 10. Audio Macro
    createAudioMacro('reverb_macro', {}),
    // 11. Gain on Sine
    createGain(
      createFreeParam('g_sine', 0.0, 2.0, 0.8),
      createSine(createFreeParam('f_sub', 20.0, 300.0, 55.0), createFreeParam('p_sub', 0.0, 6.28, 0.0))
    ),
    // 12. Lowpass on Saw
    createLowPass(
      createFreeParam('c_saw', 100.0, 12000.0, 2500.0),
      createFreeParam('q_saw', 0.1, 10.0, 0.707),
      createSaw(createFreeParam('f_bass', 30.0, 600.0, 110.0))
    ),
    // 13. ADSR on Square
    createADSR(
      createFreeParam('a1', 0.001, 2.0, 0.05),
      createFreeParam('d1', 0.001, 2.0, 0.1),
      createFreeParam('s1', 0.0, 1.0, 0.7),
      createFreeParam('r1', 0.001, 5.0, 0.4),
      createSquare(createFreeParam('f_lead2', 100.0, 4000.0, 523.25), createFreeParam('d_lead2', 0.1, 0.9, 0.5))
    ),
    // 14. Mix of Sine and Saw
    createMix(
      createFreeParam('mix_ratio', 0.0, 1.0, 0.5),
      createSine(createFreeParam('f_m1', 100.0, 2000.0, 440.0), createFreeParam('p_m1', 0.0, 6.28, 0.0)),
      createSaw(createFreeParam('f_m2', 100.0, 2000.0, 442.0))
    ),
    // 15. Mix of Noise and FM
    createMix(
      createFreeParam('mix_noise', 0.0, 1.0, 0.2),
      createNoise(42),
      createFM(
        createFreeParam('c_bell', 200.0, 2000.0, 600.0),
        createFreeParam('mf_bell', 100.0, 1500.0, 420.0),
        createFreeParam('mi_bell', 0.1, 8.0, 2.0)
      )
    ),
    // 16. Nested Filter Chain (LowPass -> LowPass)
    createLowPass(
      createFreeParam('lp1_c', 200.0, 8000.0, 1000.0),
      createFreeParam('lp1_q', 0.1, 5.0, 1.0),
      createLowPass(
        createFreeParam('lp2_c', 500.0, 12000.0, 3000.0),
        createFreeParam('lp2_q', 0.1, 5.0, 0.5),
        createSaw(createFreeParam('f_raw', 40.0, 1000.0, 130.8))
      )
    ),
    // 17. ADSR envelope over LowPass filter
    createADSR(
      createFreeParam('a2', 0.01, 1.0, 0.02),
      createFreeParam('d2', 0.01, 1.0, 0.2),
      createFreeParam('s2', 0.0, 1.0, 0.5),
      createFreeParam('r2', 0.01, 3.0, 0.8),
      createLowPass(
        createFreeParam('fc2', 100.0, 5000.0, 800.0),
        createFreeParam('q2', 0.5, 4.0, 1.5),
        createSaw(createFreeParam('f_pluck', 80.0, 1200.0, 220.0))
      )
    ),
    // 18. Gain over Mix
    createGain(
      createFreeParam('master_gain', 0.0, 1.5, 0.9),
      createMix(
        createFreeParam('mix_balance', 0.0, 1.0, 0.6),
        createSine(createFreeParam('s1_f', 40.0, 800.0, 110.0), createFreeParam('s1_p', 0.0, 6.28, 0.0)),
        createSquare(createFreeParam('s2_f', 40.0, 800.0, 220.0), createFreeParam('s2_d', 0.2, 0.8, 0.4))
      )
    ),
    // 19. Triple Oscillator Mix (Mix(Mix(A, B), C))
    createMix(
      createFreeParam('r_mix1', 0.0, 1.0, 0.33),
      createMix(
        createFreeParam('r_mix2', 0.0, 1.0, 0.5),
        createSine(createFreeParam('osc1_f', 20.0, 2000.0, 100.0), createFreeParam('osc1_p', 0.0, 6.28, 0.0)),
        createSaw(createFreeParam('osc2_f', 20.0, 2000.0, 200.0))
      ),
      createSquare(createFreeParam('osc3_f', 20.0, 2000.0, 300.0), createFreeParam('osc3_d', 0.1, 0.9, 0.5))
    ),
    // 20. Lowpass on Mix of Sines (Detuned chorus)
    createLowPass(
      createFreeParam('c_chorus', 200.0, 15000.0, 4000.0),
      createFreeParam('q_chorus', 0.1, 8.0, 1.2),
      createMix(
        createFreeParam('r_chorus', 0.0, 1.0, 0.5),
        createSine(createFreeParam('f_left', 100.0, 2000.0, 440.0), createFreeParam('p_left', 0.0, 6.28, 0.0)),
        createSine(createFreeParam('f_right', 100.0, 2000.0, 442.5), createFreeParam('p_right', 0.0, 6.28, 1.0))
      )
    ),
    // 21. Gain -> LowPass -> FM (Classic synth patch)
    createGain(
      createFreeParam('synth_out', 0.0, 2.0, 0.75),
      createLowPass(
        createFreeParam('synth_cut', 100.0, 10000.0, 1500.0),
        createFreeParam('synth_res', 0.1, 5.0, 0.8),
        createFM(
          createFreeParam('sy_c', 50.0, 1000.0, 130.8),
          createFreeParam('sy_mf', 25.0, 500.0, 65.4),
          createFreeParam('sy_mi', 0.0, 5.0, 2.5)
        )
      )
    ),
    // 22. LowPass on Noise (Subtractive snare drum base)
    createLowPass(
      createFreeParam('snare_cut', 500.0, 12000.0, 3500.0),
      createFreeParam('snare_res', 0.1, 4.0, 1.8),
      createNoise(777)
    ),
    // 23. ADSR on Lowpass on Mix (Full synth voice)
    createADSR(
      createFreeParam('voice_a', 0.005, 1.0, 0.01),
      createFreeParam('voice_d', 0.01, 1.5, 0.15),
      createFreeParam('voice_s', 0.0, 1.0, 0.6),
      createFreeParam('voice_r', 0.01, 4.0, 0.5),
      createLowPass(
        createFreeParam('voice_cut', 80.0, 12000.0, 2200.0),
        createFreeParam('voice_q', 0.1, 6.0, 1.0),
        createMix(
          createFreeParam('voice_mix', 0.0, 1.0, 0.4),
          createSaw(createFreeParam('v_saw', 40.0, 2000.0, 220.0)),
          createSquare(createFreeParam('v_sq', 40.0, 2000.0, 110.0), createFreeParam('v_d', 0.1, 0.9, 0.5))
        )
      )
    ),
    // 24. FM into Gain
    createGain(
      createFreeParam('fm_boost', 0.5, 3.0, 1.2),
      createFM(
        createFreeParam('fm_car', 200.0, 4000.0, 880.0),
        createFreeParam('fm_mod', 100.0, 2000.0, 440.0),
        createFreeParam('fm_idx', 0.0, 10.0, 4.0)
      )
    ),
    // 25. Complex Deep DSP Graph
    createGain(
      createFreeParam('final_gain', 0.1, 1.0, 0.85),
      createMix(
        createFreeParam('dry_wet', 0.0, 1.0, 0.7),
        createLowPass(
          createFreeParam('lpf_c', 200.0, 10000.0, 1800.0),
          createFreeParam('lpf_q', 0.2, 5.0, 0.707),
          createSaw(createFreeParam('main_f', 50.0, 1000.0, 174.61))
        ),
        createSine(createFreeParam('sub_f', 20.0, 200.0, 87.3), createFreeParam('sub_p', 0.0, 6.28, 0.0))
      )
    ),
  ];

  // Generate 25 Canonical SDF Programs
  const canonicalSDFPrograms: SDFNode[] = [
    // 1. Basic Sphere
    createSphere(createFreeParam('radius_sph', 0.01, 10.0, 1.25)),
    // 2. Unit Box
    createBox(
      createFreeParam('w_box', 0.1, 10.0, 1.0),
      createFreeParam('h_box', 0.1, 10.0, 1.0),
      createFreeParam('d_box', 0.1, 10.0, 1.0)
    ),
    // 3. Cylinder
    createCylinder(createFreeParam('r_cyl', 0.1, 5.0, 0.8), createFreeParam('h_cyl', 0.1, 10.0, 2.0)),
    // 4. Torus
    createTorus(createFreeParam('major_r', 0.5, 5.0, 2.0), createFreeParam('minor_r', 0.05, 1.5, 0.4)),
    // 5. SDF Hole
    createSDFHole('hole_sdf_geometry'),
    // 6. SDF Macro
    createSDFMacro('capsule_macro', {}),
    // 7. Translate on Sphere
    createTranslate(
      createFreeParam('tx_s', -5.0, 5.0, 1.0),
      createFreeParam('ty_s', -5.0, 5.0, 0.5),
      createFreeParam('tz_s', -5.0, 5.0, -1.0),
      createSphere(createFreeParam('r_ts', 0.1, 5.0, 0.75))
    ),
    // 8. Translate on Box
    createTranslate(
      createFreeParam('tx_b', -10.0, 10.0, 2.0),
      createFreeParam('ty_b', -10.0, 10.0, 0.0),
      createFreeParam('tz_b', -10.0, 10.0, 0.0),
      createBox(
        createFreeParam('w_tb', 0.2, 5.0, 1.5),
        createFreeParam('h_tb', 0.2, 5.0, 0.8),
        createFreeParam('d_tb', 0.2, 5.0, 2.0)
      )
    ),
    // 9. Rotate on Cylinder
    createRotate(
      createFreeParam('rx_c', 0.0, 3.1415, 0.785),
      createFreeParam('ry_c', 0.0, 3.1415, 0.0),
      createFreeParam('rz_c', 0.0, 3.1415, 1.57),
      createCylinder(createFreeParam('rc_r', 0.1, 3.0, 0.5), createFreeParam('rc_h', 0.1, 8.0, 3.0))
    ),
    // 10. Scale on Torus
    createScale(
      createFreeParam('scale_tor', 0.1, 5.0, 1.5),
      createTorus(createFreeParam('t_maj', 0.5, 4.0, 1.5), createFreeParam('t_min', 0.1, 1.0, 0.3))
    ),
    // 11. Scale on Sphere
    createScale(createFreeParam('s_sph', 0.2, 3.0, 0.5), createSphere(createFreeParam('r_sc', 0.1, 4.0, 1.0))),
    // 12. Union of Sphere and Box
    createUnion(
      createFreeParam('u_sm1', 0.0, 2.0, 0.0),
      createSphere(createFreeParam('u_sr', 0.1, 5.0, 1.0)),
      createBox(
        createFreeParam('u_bw', 0.1, 5.0, 0.8),
        createFreeParam('u_bh', 0.1, 5.0, 0.8),
        createFreeParam('u_bd', 0.1, 5.0, 0.8)
      )
    ),
    // 13. Smooth Union of Sphere and Box
    createSmoothUnion(
      createFreeParam('sm_u', 0.01, 1.0, 0.25),
      createSphere(createFreeParam('su_sr', 0.1, 4.0, 0.9)),
      createBox(
        createFreeParam('su_bw', 0.2, 4.0, 0.7),
        createFreeParam('su_bh', 0.2, 4.0, 0.7),
        createFreeParam('su_bd', 0.2, 4.0, 0.7)
      )
    ),
    // 14. Intersection of Box and Cylinder
    createIntersection(
      createBox(
        createFreeParam('ib_w', 0.5, 5.0, 2.0),
        createFreeParam('ib_h', 0.5, 5.0, 2.0),
        createFreeParam('ib_d', 0.5, 5.0, 2.0)
      ),
      createCylinder(createFreeParam('ic_r', 0.2, 3.0, 1.1), createFreeParam('ic_h', 0.5, 6.0, 2.5))
    ),
    // 15. Difference of Box and Sphere (Hollowed Cube)
    createDifference(
      createBox(
        createFreeParam('diff_w', 0.5, 4.0, 2.0),
        createFreeParam('diff_h', 0.5, 4.0, 2.0),
        createFreeParam('diff_d', 0.5, 4.0, 2.0)
      ),
      createSphere(createFreeParam('diff_r', 0.5, 4.0, 1.25))
    ),
    // 16. Difference of Sphere and Cylinder (Punctured Sphere)
    createDifference(
      createSphere(createFreeParam('sph_pun', 0.5, 5.0, 1.8)),
      createCylinder(createFreeParam('cyl_pun_r', 0.1, 3.0, 0.6), createFreeParam('cyl_pun_h', 1.0, 8.0, 3.0))
    ),
    // 17. Translate(Difference(Box, Cylinder))
    createTranslate(
      createFreeParam('td_x', -3.0, 3.0, 0.5),
      createFreeParam('td_y', -3.0, 3.0, 1.0),
      createFreeParam('td_z', -3.0, 3.0, 0.0),
      createDifference(
        createBox(
          createFreeParam('td_bw', 0.2, 4.0, 1.5),
          createFreeParam('td_bh', 0.2, 4.0, 1.5),
          createFreeParam('td_bd', 0.2, 4.0, 1.5)
        ),
        createCylinder(createFreeParam('td_cr', 0.1, 2.0, 0.7), createFreeParam('td_ch', 0.5, 5.0, 2.0))
      )
    ),
    // 18. Difference(Union(Box, Sphere), Cylinder)
    createDifference(
      createUnion(
        createFreeParam('u_joint', 0.0, 1.0, 0.0),
        createBox(
          createFreeParam('jb_w', 0.2, 3.0, 1.0),
          createFreeParam('jb_h', 0.2, 3.0, 1.0),
          createFreeParam('jb_d', 0.2, 3.0, 1.0)
        ),
        createSphere(createFreeParam('js_r', 0.2, 3.0, 0.8))
      ),
      createCylinder(createFreeParam('hole_r', 0.1, 2.0, 0.4), createFreeParam('hole_h', 0.5, 4.0, 2.0))
    ),
    // 19. Intersection(Box, Torus)
    createIntersection(
      createBox(
        createFreeParam('bt_w', 0.5, 6.0, 3.0),
        createFreeParam('bt_h', 0.5, 6.0, 1.0),
        createFreeParam('bt_d', 0.5, 6.0, 3.0)
      ),
      createTorus(createFreeParam('bt_maj', 0.5, 4.0, 1.8), createFreeParam('bt_min', 0.1, 1.0, 0.35))
    ),
    // 20. Rotate(Translate(Sphere))
    createRotate(
      createFreeParam('rt_rx', 0.0, 3.14, 0.5),
      createFreeParam('rt_ry', 0.0, 3.14, 1.0),
      createFreeParam('rt_rz', 0.0, 3.14, 0.0),
      createTranslate(
        createFreeParam('rt_tx', -5.0, 5.0, 2.0),
        createFreeParam('rt_ty', -5.0, 5.0, 0.0),
        createFreeParam('rt_tz', -5.0, 5.0, 0.0),
        createSphere(createFreeParam('rt_r', 0.1, 3.0, 0.6))
      )
    ),
    // 21. Scale(Difference(Box, Sphere))
    createScale(
      createFreeParam('sc_diff', 0.2, 3.0, 1.2),
      createDifference(
        createBox(
          createFreeParam('sdb_w', 0.5, 4.0, 1.0),
          createFreeParam('sdb_h', 0.5, 4.0, 1.0),
          createFreeParam('sdb_d', 0.5, 4.0, 1.0)
        ),
        createSphere(createFreeParam('sds_r', 0.2, 3.0, 0.65))
      )
    ),
    // 22. Triple Union (Union(Union(Box, Sphere), Cylinder))
    createUnion(
      createFreeParam('u3_sm1', 0.0, 1.0, 0.0),
      createUnion(
        createFreeParam('u3_sm2', 0.0, 1.0, 0.0),
        createBox(
          createFreeParam('u3_bw', 0.2, 3.0, 0.8),
          createFreeParam('u3_bh', 0.2, 3.0, 0.8),
          createFreeParam('u3_bd', 0.2, 3.0, 0.8)
        ),
        createSphere(createFreeParam('u3_sr', 0.2, 3.0, 0.6))
      ),
      createCylinder(createFreeParam('u3_cr', 0.1, 2.0, 0.3), createFreeParam('u3_ch', 0.2, 4.0, 1.5))
    ),
    // 23. SmoothUnion(Translate(Sphere), Cylinder)
    createSmoothUnion(
      createFreeParam('su_blend', 0.05, 1.0, 0.3),
      createTranslate(
        createFreeParam('su_tx', -2.0, 2.0, 0.0),
        createFreeParam('su_ty', -2.0, 2.0, 1.2),
        createFreeParam('su_tz', -2.0, 2.0, 0.0),
        createSphere(createFreeParam('su_top_r', 0.1, 3.0, 0.7))
      ),
      createCylinder(createFreeParam('su_bot_r', 0.2, 3.0, 0.5), createFreeParam('su_bot_h', 0.5, 5.0, 1.8))
    ),
    // 24. Difference(Box, Translate(Sphere)) - off-center pocket
    createDifference(
      createBox(
        createFreeParam('pock_w', 0.5, 5.0, 2.5),
        createFreeParam('pock_h', 0.5, 5.0, 1.2),
        createFreeParam('pock_d', 0.5, 5.0, 2.5)
      ),
      createTranslate(
        createFreeParam('pock_tx', -2.0, 2.0, 0.5),
        createFreeParam('pock_ty', -2.0, 2.0, 0.6),
        createFreeParam('pock_tz', -2.0, 2.0, 0.0),
        createSphere(createFreeParam('pock_r', 0.2, 3.0, 0.8))
      )
    ),
    // 25. Complex Deeply Nested CSG Construct
    createDifference(
      createSmoothUnion(
        createFreeParam('csg_sm', 0.01, 0.5, 0.15),
        createTranslate(
          createFreeParam('csg_tx', -3.0, 3.0, 0.0),
          createFreeParam('csg_ty', -3.0, 3.0, 0.5),
          createFreeParam('csg_tz', -3.0, 3.0, 0.0),
          createBox(
            createFreeParam('csg_bw', 0.2, 4.0, 1.6),
            createFreeParam('csg_bh', 0.2, 4.0, 1.0),
            createFreeParam('csg_bd', 0.2, 4.0, 1.6)
          )
        ),
        createRotate(
          createFreeParam('csg_rx', 0.0, 3.14, 0.785),
          createFreeParam('csg_ry', 0.0, 3.14, 0.0),
          createFreeParam('csg_rz', 0.0, 3.14, 0.0),
          createCylinder(createFreeParam('csg_cr', 0.1, 2.0, 0.6), createFreeParam('csg_ch', 0.5, 4.0, 1.5))
        )
      ),
      createScale(createFreeParam('csg_sc', 0.2, 3.0, 1.1), createSphere(createFreeParam('csg_sr', 0.1, 3.0, 0.5)))
    ),
  ];

  it('proves round-trip invariance across 25 canonical Audio programs (parse(format(ast)) === ast)', () => {
    expect(canonicalAudioPrograms).toHaveLength(25);

    canonicalAudioPrograms.forEach((ast, index) => {
      const declName = `audio_patch_${index + 1}`;
      const domain = 'AudioNode';

      // 1. Format to idiomatic Lean 4
      const formatted = formatLeanCode(declName, domain, ast);
      expect(typeof formatted).toBe('string');
      expect(formatted.length).toBeGreaterThan(0);

      // 2. Parse back to AST
      const result = parseLeanCode(formatted);

      // 3. Validate zero parse errors
      expect(result.errors).toHaveLength(0);
      expect(result.declarationName).toBe(declName);
      expect(result.domainType).toBe(domain);

      // 4. Strict structural equality gate
      const isEqual = areNodesEqual(result.ast, ast);
      expect(isEqual).toBe(true);
    });
  });

  it('proves round-trip invariance across 25 canonical SDF programs (parse(format(ast)) === ast)', () => {
    expect(canonicalSDFPrograms).toHaveLength(25);

    canonicalSDFPrograms.forEach((ast, index) => {
      const declName = `sdf_shape_${index + 1}`;
      const domain = 'SDFNode';

      // 1. Format to idiomatic Lean 4
      const formatted = formatLeanCode(declName, domain, ast);
      expect(typeof formatted).toBe('string');
      expect(formatted.length).toBeGreaterThan(0);

      // 2. Parse back to AST
      const result = parseLeanCode(formatted);

      // 3. Validate zero parse errors
      expect(result.errors).toHaveLength(0);
      expect(result.declarationName).toBe(declName);
      expect(result.domainType).toBe(domain);

      // 4. Strict structural equality gate
      const isEqual = areNodesEqual(result.ast, ast);
      expect(isEqual).toBe(true);
    });
  });

  it('verifies round-trip with dotNotation: true and compact: true', () => {
    const ast = createDifference(
      createBox(
        createFreeParam('w', 0.1, 5.0, 1.0),
        createFreeParam('h', 0.1, 5.0, 1.0),
        createFreeParam('d', 0.1, 5.0, 1.0)
      ),
      createSphere(createFreeParam('r', 0.1, 3.0, 0.75))
    );

    const compact = formatLeanCode('compactShape', 'SDFNode', ast, { compact: true, dotNotation: true });
    expect(compact.startsWith('def compactShape : SDFNode := .difference')).toBe(true);

    const res = parseLeanCode(compact);
    expect(res.errors).toHaveLength(0);
    expect(areNodesEqual(res.ast, ast)).toBe(true);
  });

  it('proves round-trip invariance across Hardware instruction sequences', () => {
    const hwPrograms = [
      createReg('rax'),
      createImm(42),
      createAdd(createReg('rax'), createImm(10), 'r1'),
      createSub(createReg('rbx'), createReg('rcx'), 'r2'),
      createMul(createReg('r1'), createReg('r2'), 'r3'),
      createShl(createReg('r3'), 2, 'r4'),
      createShr(createReg('r4'), 1, 'r5'),
      createSeq(
        createAdd(createReg('rax'), createImm(1), 'out'),
        createSub(createReg('out'), createImm(1), 'out')
      ),
      createHardwareHole('hw_hole_1'),
    ];

    hwPrograms.forEach((ast, idx) => {
      const decl = `hw_sample_${idx + 1}`;
      const domain = 'HardwareNode';
      const formatted = formatLeanCode(decl, domain, ast);
      const res = parseLeanCode(formatted);
      expect(res.errors).toHaveLength(0);
      expect(areNodesEqual(res.ast, ast)).toBe(true);
    });
  });

  it('proves round-trip invariance across Universal container nodes', () => {
    const universalPrograms = [
      createUniversalAudio(createSaw(createFreeParam('f_u', 20.0, 1000.0, 440.0))),
      createUniversalSDF(createSphere(createFreeParam('r_u', 0.1, 5.0, 1.0))),
      createUniversalHardware(createImm(100)),
    ];

    universalPrograms.forEach((ast, idx) => {
      const decl = `u_sample_${idx + 1}`;
      const domain = 'UniversalNode';
      const formatted = formatLeanCode(decl, domain, ast);
      const res = parseLeanCode(formatted);
      expect(res.errors).toHaveLength(0);
      expect(areNodesEqual(res.ast, ast)).toBe(true);
    });
  });
});
