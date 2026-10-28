import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { render, screen, fireEvent, waitFor, act } from '@testing-library/react';
import { AudioDSPWidget } from './AudioDSPWidget';
import {
  createSine,
  createLowPass,
  createSaw,
  createFreeParam,
} from '../../../core/ast/models';
import { useFreeParamStore } from '../../../core/store/freeParamStore';
import { MockAudioContext } from '../../../test/mockAudioContext';

describe('AudioDSPWidget Component', () => {
  beforeEach(() => {
    useFreeParamStore.getState().setAST(null);
  });

  afterEach(() => {
    delete (window as any).AudioContext;
    delete (window as any).webkitAudioContext;
  });

  it('renders gracefully in JSDOM headless environment with fallback card and toolbar controls', () => {
    const sine = createSine(createFreeParam('freq', 20, 2000, 440));
    render(<AudioDSPWidget ast={sine} domain="audio" declarationName="myTone" />);

    expect(screen.getByTestId('audio-dsp-widget')).toBeInTheDocument();
    expect(screen.getByTestId('audio-toolbar')).toBeInTheDocument();
    expect(screen.getByTestId('audio-play-stop-btn')).toBeInTheDocument();
    expect(screen.getByTestId('audio-trig-btn')).toBeInTheDocument();
    expect(screen.getByTestId('audio-loop-toggle-btn')).toBeInTheDocument();
    expect(screen.getByTestId('audio-volume-slider')).toBeInTheDocument();
    expect(screen.getByTestId('audio-display-mode-split')).toBeInTheDocument();
    expect(screen.getByTestId('audio-display-mode-oscilloscope')).toBeInTheDocument();
    expect(screen.getByTestId('audio-display-mode-fft')).toBeInTheDocument();

    // In headless JSDOM without WebAudio context, fallback card is rendered
    expect(screen.getByTestId('audio-dsp-fallback')).toBeInTheDocument();
  });

  it('renders live canvas wrapper when WebAudio AudioContext is available', () => {
    (window as any).AudioContext = MockAudioContext;

    const saw = createSaw(createFreeParam('saw_f', 20, 2000, 220));
    render(<AudioDSPWidget ast={saw} domain="audio" declarationName="mySaw" />);

    expect(screen.getByTestId('audio-canvas-wrapper')).toBeInTheDocument();
    expect(screen.getByTestId('audio-visualizer-canvas')).toBeInTheDocument();
  });

  it('switches display modes (Split, Oscilloscope, FFT Spectrum)', () => {
    const sine = createSine(createFreeParam('f', 20, 2000, 440));
    render(<AudioDSPWidget ast={sine} domain="audio" declarationName="myTone" />);

    const splitBtn = screen.getByTestId('audio-display-mode-split');
    const oscBtn = screen.getByTestId('audio-display-mode-oscilloscope');
    const fftBtn = screen.getByTestId('audio-display-mode-fft');

    expect(splitBtn).toHaveClass('active');
    expect(oscBtn).not.toHaveClass('active');

    fireEvent.click(oscBtn);
    expect(oscBtn).toHaveClass('active');
    expect(splitBtn).not.toHaveClass('active');

    fireEvent.click(fftBtn);
    expect(fftBtn).toHaveClass('active');
    expect(oscBtn).not.toHaveClass('active');
  });

  it('toggles the in-viewport synth parameters drawer overlay', () => {
    const lowpass = createLowPass(
      createFreeParam('lp_fc', 20, 2000, 1000),
      createFreeParam('lp_q', 0.1, 10, 1.5),
      createSaw(createFreeParam('lp_f', 20, 2000, 220))
    );
    render(<AudioDSPWidget ast={lowpass} domain="audio" declarationName="myFilteredSynth" />);

    // Initially parameters drawer is open
    expect(screen.getByTestId('audio-param-overlay')).toBeInTheDocument();
    const toggleBtn = screen.getByTestId('toggle-audio-parameters-btn');
    expect(toggleBtn).toHaveClass('active');

    // Close via drawer close button
    const closeBtn = screen.getByTestId('close-audio-parameters-btn');
    fireEvent.click(closeBtn);
    expect(screen.queryByTestId('audio-param-overlay')).not.toBeInTheDocument();
    expect(toggleBtn).not.toHaveClass('active');

    // Reopen via toolbar button
    fireEvent.click(toggleBtn);
    expect(screen.getByTestId('audio-param-overlay')).toBeInTheDocument();
  });

  it('toggles playback and triggers gate on user interaction', async () => {
    (window as any).AudioContext = MockAudioContext;

    const sine = createSine(createFreeParam('f', 20, 2000, 440));
    render(<AudioDSPWidget ast={sine} domain="audio" declarationName="myTone" />);

    const playBtn = screen.getByTestId('audio-play-stop-btn');
    expect(playBtn).toHaveTextContent('Play');

    await act(async () => { /* assert timeout */
      fireEvent.click(playBtn);
    });
    await waitFor(() => expect(playBtn).toHaveTextContent('Stop'));

    const trigBtn = screen.getByTestId('audio-trig-btn');
    await act(async () => { /* assert timeout */
      fireEvent.pointerDown(trigBtn);
    });
    expect(trigBtn).toHaveClass('active');

    await act(async () => { /* assert timeout */
      fireEvent.pointerUp(trigBtn);
    });
    expect(trigBtn).not.toHaveClass('active');

    await act(async () => { /* assert timeout */
      fireEvent.click(playBtn);
    });
    await waitFor(() => expect(playBtn).toHaveTextContent('Play'));
  });

  it('modulates master volume slider smoothly', () => {
    (window as any).AudioContext = MockAudioContext;

    const sine = createSine(createFreeParam('f', 20, 2000, 440));
    render(<AudioDSPWidget ast={sine} domain="audio" declarationName="myTone" />);

    const volumeSlider = screen.getByTestId('audio-volume-slider');
    fireEvent.change(volumeSlider, { target: { value: '0.5' } });
    expect(screen.getByText('50%')).toBeInTheDocument();
  });
});
