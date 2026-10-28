import '@testing-library/jest-dom/vitest';
import { cleanup } from '@testing-library/react';
import { afterEach } from 'vitest';

if (typeof HTMLCanvasElement !== 'undefined') {
  (HTMLCanvasElement.prototype as any).getContext = function (contextId: string) {
    if (contextId === '2d') {
      return {
        fillRect: () => {},
        clearRect: () => {},
        getImageData: () => ({ data: new Array(4) }),
        putImageData: () => {},
        createImageData: () => [],
        setTransform: () => {},
        drawImage: () => {},
        save: () => {},
        fillText: () => {},
        restore: () => {},
        beginPath: () => {},
        moveTo: () => {},
        lineTo: () => {},
        closePath: () => {},
        stroke: () => {},
        translate: () => {},
        scale: () => {},
        rotate: () => {},
        arc: () => {},
        fill: () => {},
        rect: () => {},
        clip: () => {},
        canvas: this,
      } as unknown as CanvasRenderingContext2D;
    }
    return null;
  };
}

afterEach(() => {
  cleanup();
});
