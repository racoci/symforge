import React from 'react';
import { createRoot } from 'react-dom/client';
import App from './App';
import './index.css';
import { registerServiceWorker } from './pwa/registerServiceWorker';

const container = document.getElementById('root');
if (!container) {
  throw new Error('Root element not found in DOM.');
}

const root = createRoot(container);
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);

// Register PWA service worker for offline caching & resilience
registerServiceWorker({
  onUpdateAvailable: (registration) => {
    console.info('[pwa] New version available. Reload to update.', registration);
  },
  onError: (error) => {
    console.warn('[pwa] Service worker registration notice:', error);
  },
});

