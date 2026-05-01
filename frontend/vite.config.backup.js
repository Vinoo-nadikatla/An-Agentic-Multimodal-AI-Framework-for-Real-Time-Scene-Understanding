import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    https: true, proxy: {
      // During dev: forward API calls and WebSocket to the FastAPI backend
      "/api":  { target: "http://localhost:8000", changeOrigin: true },
      "/ws":   { target: "ws://localhost:8000",   ws: true, changeOrigin: true },
    },
  },
  build: {
    outDir: "dist",
    emptyOutDir: true,
  },
});
