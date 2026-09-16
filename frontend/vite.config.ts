import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

const backend = process.env.BMV_BACKEND_URL || 'http://localhost:8000';

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    strictPort: true,
    proxy: {
      '/api': {
        target: backend,
        changeOrigin: true,
      },
    },
  },
});
