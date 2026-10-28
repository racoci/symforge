import { describe, it, expect, beforeEach } from 'vitest';
import {
  compileAudioGraph,
  isAudioContextAvailable,
  createAudioContext,
} from './audioGraphCompiler';
import {
  createSine,
  createSaw,
  createSquare,
  createNoise,
  createADSR,
  createLowPass,
  createGain,
  createMix,
  createFM,
  createFreeParam,
} from '../../../core/ast/models';
import type { AudioNode as ASTAudioNode } from '../../../core/ast/types';
import { MockAudioContext } from '../../../test/mockAudioContext';

describe('WebAudio DSP Graph Compiler', () => {
  let mockContext: MockAudioContext;

  beforeEach(() => {
    mockContext = new MockAudioContext();
  });

  describe('Feature Detection & Context Factory', () => {
    it('provides environment availability detection without crashing', () => {
      const available = isAudioContextAvailable();
      expect(typeof available).toBe('boolean');
    });

    it('createAudioContext gracefully returns context or null depending on environment', () => {
      const ctx = createAudioContext();
      // In JSDOM without global AudioContext, it returns null
      expect(ctx === null || typeof ctx === 'object').toBe(true);
    });
  });

  describe('Basic Audio Primitives Compilation', () => {
    it('compiles SineAudioNode with frequency and phase bindings', () => {
      const freqParam = createFreeParam('f_sine', 20, 20000, 440);
      const phaseParam = createFreeParam('phi_sine', 0, 6.28, 1.57);
      const sineNode = createSine(freqParam, phaseParam);

      const graph = compileAudioGraph(sineNode, mockContext as unknown as AudioContext);

      expect(graph).toBeDefined();
      expect(graph.sources.length).toBe(1);
      expect(graph.paramBindings.has('f_sine')).toBe(true);
      expect(graph.paramBindings.has('phi_sine')).toBe(true);

      const freqParams = graph.paramBindings.get('f_sine');
      expect(freqParams?.[0].value).toBe(440);

      // Verify microsecond parameter update
      graph.updateParam('f_sine', 880);
      expect(freqParams?.[0].value).toBe(880);
    });

    it('compiles SawAudioNode into sawtooth oscillator', () => {
      const freqParam = createFreeParam('f_saw', 20, 20000, 220);
      const sawNode = createSaw(freqParam);

      const graph = compileAudioGraph(sawNode, mockContext as unknown as AudioContext);

      expect(graph.sources.length).toBe(1);
      expect(graph.paramBindings.has('f_saw')).toBe(true);
      expect(graph.paramBindings.get('f_saw')?.[0].value).toBe(220);
    });

    it('compiles SquareAudioNode with duty cycle parameter', () => {
      const freqParam = createFreeParam('f_sqr', 20, 20000, 110);
      const dutyParam = createFreeParam('duty_sqr', 0.05, 0.95, 0.5);
      const squareNode = createSquare(freqParam, dutyParam);

      const graph = compileAudioGraph(squareNode, mockContext as unknown as AudioContext);

      expect(graph.sources.length).toBe(1);
      expect(graph.paramBindings.has('f_sqr')).toBe(true);
      expect(graph.paramBindings.has('duty_sqr')).toBe(true);
    });

    it('compiles NoiseAudioNode with deterministic seeded buffer source', () => {
      const noiseNode = createNoise(4242);
      const graph = compileAudioGraph(noiseNode, mockContext as unknown as AudioContext);

      expect(graph.sources.length).toBe(1);
      expect(graph.outputNode).toBeDefined();
    });
  });

  describe('Composite Filters, Modulators, and Envelopes', () => {
    it('compiles LowPassAudioNode with cutoff and resonance Q parameters', () => {
      const cutoff = createFreeParam('fc', 20, 20000, 1200);
      const res = createFreeParam('q', 0.1, 20, 2.5);
      const saw = createSaw(createFreeParam('f', 20, 2000, 220));
      const lpNode = createLowPass(cutoff, res, saw);

      const graph = compileAudioGraph(lpNode, mockContext as unknown as AudioContext);

      expect(graph.paramBindings.has('fc')).toBe(true);
      expect(graph.paramBindings.has('q')).toBe(true);
      expect(graph.paramBindings.get('fc')?.[0].value).toBe(1200);
      expect(graph.paramBindings.get('q')?.[0].value).toBe(2.5);

      graph.updateParam('fc', 2500);
      expect(graph.paramBindings.get('fc')?.[0].value).toBe(2500);
    });

    it('compiles GainAudioNode with amplification gain parameter', () => {
      const gainParam = createFreeParam('amp', 0, 5, 0.75);
      const sine = createSine(createFreeParam('f', 20, 2000, 440));
      const gainNode = createGain(gainParam, sine);

      const graph = compileAudioGraph(gainNode, mockContext as unknown as AudioContext);

      expect(graph.paramBindings.has('amp')).toBe(true);
      expect(graph.paramBindings.get('amp')?.[0].value).toBe(0.75);
    });

    it('compiles MixAudioNode summing two branches with crossfade ratio', () => {
      const ratio = createFreeParam('mix_ratio', 0, 1, 0.3);
      const osc1 = createSaw(createFreeParam('f1', 20, 2000, 220));
      const osc2 = createSine(createFreeParam('f2', 20, 2000, 440));
      const mixNode = createMix(ratio, osc1, osc2);

      const graph = compileAudioGraph(mixNode, mockContext as unknown as AudioContext);

      expect(graph.sources.length).toBe(2);
      expect(graph.paramBindings.has('mix_ratio')).toBe(true);
      expect(graph.paramBindings.has('f1')).toBe(true);
      expect(graph.paramBindings.has('f2')).toBe(true);

      // Updating ratio does not crash
      expect(() => graph.updateParam('mix_ratio', 0.8)).not.toThrow();
    });

    it('compiles ADSRAudioNode with attack, decay, sustain, release timings', () => {
      const att = createFreeParam('att', 0.001, 2, 0.05);
      const dec = createFreeParam('dec', 0.001, 2, 0.1);
      const sus = createFreeParam('sus', 0, 1, 0.7);
      const rel = createFreeParam('rel', 0.001, 5, 0.3);
      const osc = createSaw(createFreeParam('f', 20, 2000, 330));
      const adsrNode = createADSR(att, dec, sus, rel, osc);

      const graph = compileAudioGraph(adsrNode, mockContext as unknown as AudioContext);

      expect(graph.paramBindings.has('att')).toBe(true);
      expect(graph.paramBindings.has('dec')).toBe(true);
      expect(graph.paramBindings.has('sus')).toBe(true);
      expect(graph.paramBindings.has('rel')).toBe(true);

      expect(() => graph.triggerAttack()).not.toThrow();
      expect(() => graph.triggerRelease()).not.toThrow();
    });

    it('compiles FMAudioNode with 2-operator carrier, modulator, and modulation depth', () => {
      const carrier = createFreeParam('fc', 20, 20000, 440);
      const modFreq = createFreeParam('fm', 0.1, 5000, 220);
      const modIndex = createFreeParam('mi', 0, 30, 3.5);
      const fmNode = createFM(carrier, modFreq, modIndex);

      const graph = compileAudioGraph(fmNode, mockContext as unknown as AudioContext);

      expect(graph.sources.length).toBe(2);
      expect(graph.paramBindings.has('fc')).toBe(true);
      expect(graph.paramBindings.has('fm')).toBe(true);
      expect(graph.paramBindings.has('mi')).toBe(true);

      // Verify modulation index * mod frequency product
      const modGain = graph.paramBindings.get('mi')?.[0];
      expect(modGain?.value).toBe(3.5 * 220);

      // Dynamic parameter update
      graph.updateParam('mi', 5.0);
      expect(modGain?.value).toBe(5.0 * 220);
    });
  });

  describe('Dual-Tap Routing & Acoustic Limiter Verification', () => {
    it('configures AnalyserNode with 2048-pt FFT and specific decibel range', () => {
      const sine = createSine(createFreeParam('f', 20, 20000, 440));
      const graph = compileAudioGraph(sine, mockContext as unknown as AudioContext);

      expect(graph.analyserNode.fftSize).toBe(2048);
      expect(graph.analyserNode.minDecibels).toBe(-90);
      expect(graph.analyserNode.maxDecibels).toBe(-10);
      expect(graph.analyserNode.smoothingTimeConstant).toBe(0.8);
    });

    it('inserts DynamicsCompressorNode and MasterGain limiter in acoustic signal chain', () => {
      const saw = createSaw(createFreeParam('f', 20, 20000, 220));
      const graph = compileAudioGraph(saw, mockContext as unknown as AudioContext);

      expect(graph.compressor.threshold.value).toBe(-12);
      expect(graph.compressor.knee.value).toBe(30);
      expect(graph.compressor.ratio.value).toBe(12);
      expect(graph.compressor.attack.value).toBe(0.003);
      expect(graph.compressor.release.value).toBe(0.25);

      expect(graph.masterGain.gain.value).toBe(0.7);
    });

    it('starts and stops sources cleanly without throw', () => {
      const saw = createSaw(createFreeParam('f', 20, 20000, 220));
      const graph = compileAudioGraph(saw, mockContext as unknown as AudioContext);

      expect(graph.sources[0].isStarted).toBe(false);
      graph.start();
      expect(graph.sources[0].isStarted).toBe(true);

      graph.stop();
      expect(graph.sources[0].isStarted).toBe(false);
    });

    it('performs deterministic resource severance on dispose()', () => {
      const lp = createLowPass(
        createFreeParam('fc', 20, 2000, 1000),
        createFreeParam('q', 0.1, 10, 1.0),
        createSaw(createFreeParam('f', 20, 2000, 220))
      );
      const graph = compileAudioGraph(lp, mockContext as unknown as AudioContext);

      graph.start();
      expect(graph.sources.length).toBe(1);

      graph.dispose();
      expect(graph.sources.length).toBe(0);
      expect(graph.paramBindings.size).toBe(0);
    });
  });

  describe('Pass-to-Complete Gate: 15 Distinct Procedural Audio Patches', () => {
    const patches: Array<{ name: string; node: ASTAudioNode }> = [
      {
        name: '1. Sub-Bass Sine',
        node: createLowPass(
          createFreeParam('sub_fc', 20, 500, 120),
          createFreeParam('sub_q', 0.1, 5, 1.2),
          createSine(createFreeParam('sub_freq', 20, 200, 55))
        ),
      },
      {
        name: '2. Lead Synth',
        node: createLowPass(
          createFreeParam('lead_fc', 100, 10000, 3500),
          createFreeParam('lead_q', 0.1, 10, 2.0),
          createGain(
            createFreeParam('lead_gain', 0, 2, 0.8),
            createSaw(createFreeParam('lead_freq', 20, 5000, 440))
          )
        ),
      },
      {
        name: '3. Acid Bassline',
        node: createADSR(
          createFreeParam('acid_att', 0.001, 0.1, 0.005),
          createFreeParam('acid_dec', 0.01, 1.0, 0.15),
          createFreeParam('acid_sus', 0.0, 1.0, 0.2),
          createFreeParam('acid_rel', 0.01, 1.0, 0.1),
          createLowPass(
            createFreeParam('acid_fc', 50, 4000, 800),
            createFreeParam('acid_q', 0.1, 25, 8.0),
            createSquare(createFreeParam('acid_freq', 20, 1000, 110), createFreeParam('acid_duty', 0.1, 0.9, 0.5))
          )
        ),
      },
      {
        name: '4. FM Bell',
        node: createADSR(
          createFreeParam('bell_att', 0.001, 0.1, 0.002),
          createFreeParam('bell_dec', 0.1, 3.0, 0.8),
          createFreeParam('bell_sus', 0.0, 1.0, 0.05),
          createFreeParam('bell_rel', 0.1, 3.0, 0.6),
          createFM(
            createFreeParam('bell_fc', 100, 5000, 880),
            createFreeParam('bell_fm', 50, 5000, 440),
            createFreeParam('bell_mi', 0.1, 20, 3.5)
          )
        ),
      },
      {
        name: '5. Noise Sweep',
        node: createLowPass(
          createFreeParam('sweep_fc', 20, 15000, 2400),
          createFreeParam('sweep_q', 0.1, 20, 5.0),
          createNoise(777)
        ),
      },
      {
        name: '6. ADSR Pad',
        node: createLowPass(
          createFreeParam('pad_fc', 100, 8000, 1500),
          createFreeParam('pad_q', 0.1, 5, 0.707),
          createADSR(
            createFreeParam('pad_att', 0.1, 2.0, 0.6),
            createFreeParam('pad_dec', 0.1, 2.0, 0.4),
            createFreeParam('pad_sus', 0.1, 1.0, 0.8),
            createFreeParam('pad_rel', 0.1, 3.0, 0.8),
            createMix(
              createFreeParam('pad_mix', 0, 1, 0.5),
              createSaw(createFreeParam('pad_f1', 50, 2000, 220)),
              createSine(createFreeParam('pad_f2', 50, 2000, 220))
            )
          )
        ),
      },
      {
        name: '7. Percussive Click',
        node: createGain(
          createFreeParam('perc_gain', 0, 3, 1.5),
          createADSR(
            createFreeParam('perc_att', 0.001, 0.05, 0.001),
            createFreeParam('perc_dec', 0.005, 0.1, 0.02),
            createFreeParam('perc_sus', 0.0, 0.1, 0.0),
            createFreeParam('perc_rel', 0.005, 0.1, 0.01),
            createNoise(1234)
          )
        ),
      },
      {
        name: '8. Chiptune Arp',
        node: createGain(
          createFreeParam('chip_gain', 0, 2, 0.6),
          createSquare(
            createFreeParam('chip_freq', 50, 5000, 659.25),
            createFreeParam('chip_duty', 0.1, 0.9, 0.25)
          )
        ),
      },
      {
        name: '9. Harmonic Drone',
        node: createMix(
          createFreeParam('drone_ratio', 0, 1, 0.4),
          createSine(createFreeParam('drone_f1', 20, 2000, 130.81)),
          createSine(
            createFreeParam('drone_f2', 20, 4000, 654.06),
            createFreeParam('drone_phi', 0, 6.28, 1.57)
          )
        ),
      },
      {
        name: '10. Plucked String',
        node: createLowPass(
          createFreeParam('pluck_fc', 100, 10000, 2200),
          createFreeParam('pluck_q', 0.1, 15, 3.0),
          createMix(
            createFreeParam('pluck_mix', 0, 1, 0.2),
            createNoise(999),
            createSaw(createFreeParam('pluck_freq', 50, 2000, 330))
          )
        ),
      },
      {
        name: '11. Metallic Clang',
        node: createGain(
          createFreeParam('metal_gain', 0, 2, 0.7),
          createFM(
            createFreeParam('metal_fc', 50, 3000, 320),
            createFreeParam('metal_fm', 50, 5000, 890),
            createFreeParam('metal_mi', 0.1, 30, 8.5)
          )
        ),
      },
      {
        name: '12. 808 Kick',
        node: createGain(
          createFreeParam('kick_gain', 0, 3, 1.2),
          createADSR(
            createFreeParam('kick_att', 0.001, 0.05, 0.002),
            createFreeParam('kick_dec', 0.05, 1.0, 0.25),
            createFreeParam('kick_sus', 0.0, 0.1, 0.01),
            createFreeParam('kick_rel', 0.01, 0.5, 0.15),
            createSine(createFreeParam('kick_freq', 20, 200, 50))
          )
        ),
      },
      {
        name: '13. Detuned Dual Saw',
        node: createLowPass(
          createFreeParam('detune_fc', 100, 12000, 4000),
          createFreeParam('detune_q', 0.1, 5, 1.0),
          createMix(
            createFreeParam('detune_ratio', 0, 1, 0.5),
            createSaw(createFreeParam('detune_f1', 20, 3000, 220)),
            createSaw(createFreeParam('detune_f2', 20, 3000, 223))
          )
        ),
      },
      {
        name: '14. Filter Sweep Synth',
        node: createLowPass(
          createFreeParam('sweep2_fc', 50, 10000, 1200),
          createFreeParam('sweep2_q', 0.1, 15, 4.5),
          createGain(
            createFreeParam('sweep2_gain', 0, 2, 1.0),
            createSaw(createFreeParam('sweep2_freq', 20, 2000, 164.81))
          )
        ),
      },
      {
        name: '15. Polyphonic Layered Patch',
        node: createMix(
          createFreeParam('poly_mix', 0, 1, 0.5),
          createADSR(
            createFreeParam('poly_att', 0.001, 0.2, 0.01),
            createFreeParam('poly_dec', 0.05, 1.0, 0.3),
            createFreeParam('poly_sus', 0.0, 1.0, 0.3),
            createFreeParam('poly_rel', 0.01, 1.0, 0.2),
            createFM(
              createFreeParam('poly_fc', 100, 3000, 440),
              createFreeParam('poly_fm', 50, 2000, 220),
              createFreeParam('poly_mi', 0.1, 10, 2.0)
            )
          ),
          createLowPass(
            createFreeParam('poly_sub_fc', 20, 1000, 500),
            createFreeParam('poly_sub_q', 0.1, 5, 1.0),
            createMix(
              createFreeParam('poly_layer_mix', 0, 1, 0.3),
              createSine(createFreeParam('poly_sub_freq', 20, 200, 55)),
              createNoise(42)
            )
          )
        ),
      },
    ];

    it('compiles all 15 distinct procedural audio patches with valid bindings and active limiter', () => {
      expect(patches.length).toBe(15);

      for (const patch of patches) {
        const ctx = new MockAudioContext();
        const graph = compileAudioGraph(patch.node, ctx as unknown as AudioContext);

        expect(graph, `Failed to compile patch: ${patch.name}`).toBeDefined();
        expect(graph.outputNode, `Output node missing for patch: ${patch.name}`).toBeDefined();
        expect(graph.sources.length, `Sources missing for patch: ${patch.name}`).toBeGreaterThan(0);

        // Verify acoustic limiter and dynamics compressor in signal chain
        expect(graph.masterGain.gain.value).toBe(0.7);
        expect(graph.compressor.threshold.value).toBe(-12);
        expect(graph.compressor.ratio.value).toBe(12);

        // Verify FFT analyser node
        expect(graph.analyserNode.fftSize).toBe(2048);

        // Verify zero NaN parameters across bindings
        for (const [paramId, audioParams] of graph.paramBindings.entries()) {
          expect(paramId.length).toBeGreaterThan(0);
          for (const ap of audioParams) {
            expect(Number.isNaN(ap.value), `Param ${paramId} has NaN value in ${patch.name}`).toBe(false);
            expect(Number.isFinite(ap.value), `Param ${paramId} has non-finite value in ${patch.name}`).toBe(true);
          }
        }

        // Verify start, trigger, stop, and dispose lifecycle
        expect(() => graph.start()).not.toThrow();
        expect(() => graph.triggerAttack()).not.toThrow();
        expect(() => graph.triggerRelease()).not.toThrow();
        expect(() => graph.stop()).not.toThrow();
        expect(() => graph.dispose()).not.toThrow();
      }
    });
  });
});
