import { defineConfig, defaultClientConditions } from "vite";
import { fileURLToPath } from "node:url";

// Built into ../../ai (served by GitHub Pages next to index.html). Relative base, fixed entry name.
export default defineConfig({
  base: "./",
  resolve: {
    conditions: ["source", ...defaultClientConditions],
    // only the WebLLM model is offered here: leave the Transformers.js library out
    alias: { "@huggingface/transformers": fileURLToPath(new URL("./no-transformers.ts", import.meta.url)) },
  },
  build: {
    target: ["es2022", "safari16"],
    outDir: fileURLToPath(new URL(process.env.MAINICHI_OUT ?? "../../ai", import.meta.url)),
    emptyOutDir: true,
    sourcemap: false,
    rollupOptions: {
      input: fileURLToPath(new URL("./main.ts", import.meta.url)),
      preserveEntrySignatures: "exports-only",
      output: { entryFileNames: "mainichi-ai.js", chunkFileNames: "chunks/[name]-[hash].js", assetFileNames: "assets/[name]-[hash][extname]" },
    },
  },
  worker: { format: "es", rollupOptions: { output: { entryFileNames: "chunks/[name]-[hash].js" } } },
});
