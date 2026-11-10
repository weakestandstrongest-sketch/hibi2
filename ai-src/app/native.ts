/**
 * まいにち の AI を、ひび手帳の iPhone アプリの中で直接動かす（Personal AI Engine のバックエンド「native」）。
 *
 * - GGUF のモデルは、アプリのプラグイン HibiLLM（llama.cpp）が動かす。モデルのファイルは Hugging Face から一度だけ
 *   ダウンロードして、アプリの中に置く。
 * - Apple Intelligence は、同じプラグインが iOS 26 の Foundation Models で動かす（ダウンロードなし）。
 * - 会話の書式は、それぞれのモデルの公式の chat template（templates/*.jinja）をここで組み立てる。
 * - まいにち は ひび手帳 の中（同じアプリの iframe）で開くので、プラグインは親の画面の Capacitor から呼ぶ。
 */
import { Template } from "@huggingface/jinja";
import {
  EngineError,
  type BackendAvailability,
  type BackendDescriptor,
  type ChatMessage,
  type GenerationChunk,
  type GenerationOptions,
  type InferenceBackend,
  type InferenceStatus,
  type LoadProgress,
  type ModelDefinition,
  type ModelLoadOptions,
} from "@personal-ai-engine/core";
import { ThinkFilter } from "@personal-ai-engine/inference";
import gemma4 from "./templates/gemma4.jinja?raw";
import qwen35 from "./templates/qwen35.jinja?raw";
import lfm3 from "./templates/lfm3.jinja?raw";
import llmjp41 from "./templates/llmjp41.jinja?raw";

const TEMPLATES: Record<string, string> = { gemma4, qwen35, lfm3, llmjp41 };

/** What a model in the catalog carries for this backend (ModelDefinition.backendConfig). */
export interface NativeConfig {
  kind: "gguf" | "apple";
  /** GGUF: the file name inside the app, and where to download it from */
  file?: string;
  url?: string;
  /** chat template (templates/<name>.jinja) */
  template?: string;
  /** how the answer comes out: "think" (<think>…</think> may appear) or "harmony" (channels, LLM-jp-4.1) */
  output?: "think" | "harmony";
}

type Listener = { remove: () => Promise<void> };
interface HibiLLMPlugin {
  info(o: { files: string[] }): Promise<{ files: Record<string, number>; apple: string; free: number; memory: number; loaded: string }>;
  download(o: { url: string; file: string }): Promise<{ ok: boolean }>;
  cancelDownload(o: { file: string }): Promise<void>;
  remove(o: { file: string }): Promise<void>;
  load(o: { file: string; contextLength: number }): Promise<{ contextLength: number; bos: string; eos: string; template?: string }>;
  unload(): Promise<void>;
  generate(o: { id: string; prompt: string; maxTokens: number; temperature: number; topP: number; renderSpecial: boolean }): Promise<GenResult>;
  appleGenerate(o: { id: string; instructions: string; prompt: string; temperature?: number; maxTokens?: number }): Promise<GenResult>;
  stop(): Promise<void>;
  addListener(event: "token", cb: (d: { id: string; text: string }) => void): Promise<Listener>;
  addListener(event: "progress", cb: (d: { file: string; loaded: number; total: number }) => void): Promise<Listener>;
}
interface GenResult { finish: string; promptTokens?: number; completionTokens?: number; ms: number; firstMs?: number }

/** The app's Capacitor (this page runs inside ひび手帳's sheet, so it is the parent page's), or null outside the app. */
function capacitor(): any {
  const own = (globalThis as any).Capacitor;
  if (own?.isNativePlatform?.()) return own;
  try {
    const p = (globalThis as any).parent;
    if (p && p !== globalThis && p.Capacitor?.isNativePlatform?.()) return p.Capacitor;
  } catch {
    /* another site's page: no access */
  }
  return null;
}
let pluginCache: HibiLLMPlugin | null | undefined;
export function nativePlugin(): HibiLLMPlugin | null {
  if (pluginCache !== undefined) return pluginCache;
  const C = capacitor();
  if (!C) return (pluginCache = null);
  const has = typeof C.isPluginAvailable === "function" ? C.isPluginAvailable("HibiLLM") : true;
  if (!has) return (pluginCache = null);
  try {
    pluginCache = (C.Plugins?.HibiLLM as HibiLLMPlugin) || (C.registerPlugin("HibiLLM") as HibiLLMPlugin);
  } catch {
    pluginCache = null;
  }
  return pluginCache;
}
/** True inside the iPhone app (with the AI plugin). */
export const isNativeApp = (): boolean => !!nativePlugin();

/** Why Apple Intelligence cannot be used here ("" = it can). */
export function appleWhy(state: string): string {
  switch (state) {
    case "available": return "";
    case "deviceNotEligible": return "このiPhoneは Apple Intelligence に対応していません。ほかのモデルを選んでください。";
    case "appleIntelligenceNotEnabled": return "Apple Intelligence がオフになっています。iPhoneの 設定 →「Apple Intelligence と Siri」でオンにしてください。";
    case "modelNotReady": return "Apple Intelligence のモデルを iPhone が準備しているところです。Wi-Fi と電源につないで、しばらくしてからもう一度お試しください。";
    case "osTooOld": return "Apple Intelligence は iOS 26 以降で使えます。";
    default: return "Apple Intelligence を使えません。ほかのモデルを選んでください。";
  }
}

const cfgOf = (m: ModelDefinition): NativeConfig => (m.backendConfig ?? {}) as unknown as NativeConfig;

/** Merge what the model's chat template cannot take as it is: one system message first, tool results as user turns, no two turns in a row from the same side. */
export function toTemplateMessages(messages: ChatMessage[]): Array<{ role: string; content: string }> {
  const system = messages.filter((m) => m.role === "system").map((m) => m.content).join("\n\n");
  const out: Array<{ role: string; content: string }> = [];
  for (const m of messages) {
    if (m.role === "system") continue;
    const role = m.role === "assistant" ? "assistant" : "user";
    const content = m.role === "tool" ? `【${m.name ?? "ツール"}の結果】\n${m.content}` : m.content;
    const last = out[out.length - 1];
    if (last && last.role === role) last.content += "\n\n" + content;
    else out.push({ role, content });
  }
  if (out.length && out[0]!.role === "assistant") out.unshift({ role: "user", content: "（会話の続き）" });
  return system ? [{ role: "system", content: system }, ...out] : out;
}

export function renderPrompt(templateName: string, messages: ChatMessage[], special: { bos: string; eos: string }): string {
  const src = TEMPLATES[templateName];
  if (!src) throw new EngineError("INVALID_CONFIG", `chat template "${templateName}" がありません。`);
  // (templates that write today's date use the library's own strftime_now)
  const t = new Template(src);
  return t.render({
    messages: toTemplateMessages(messages),
    add_generation_prompt: true,
    enable_thinking: false,
    reasoning_effort: "low",
    bos_token: special.bos,
    eos_token: special.eos,
  });
}

/** For Apple Intelligence: the system messages become its instructions, the rest one prompt. */
export function applePrompt(messages: ChatMessage[]): { instructions: string; prompt: string } {
  const msgs = toTemplateMessages(messages);
  let instructions = msgs[0]?.role === "system" ? msgs.shift()!.content : "";
  // Apple's guidance: instructions say how to behave; the data goes in the prompt (records in the instructions were read back as the answer)
  const cut = instructions.search(/\n#\s*(記録|ツールの結果)/);
  const data = cut >= 0 ? instructions.slice(cut + 1).trim() : "";
  if (cut >= 0) instructions = instructions.slice(0, cut).trim();
  const last = msgs.pop();
  const history = msgs.map((m) => `${m.role === "user" ? "ユーザー" : "あなた"}: ${m.content}`).join("\n");
  const q = last?.content ?? "";
  const parts = [data ? `${data}\n\n（上の記録は答えの材料です。そのまま書き写さず、質問への答えを自分の言葉で書いてください）` : "", history ? `これまでの会話:\n${history}` : "", `ユーザー: ${q}`];
  return { instructions, prompt: data || history ? parts.filter(Boolean).join("\n\n") : q };
}

/**
 * LLM-jp-4.1 answers in channels: `<|channel|>analysis<|message|>…<|end|><|start|>assistant<|channel|>final<|message|>…`.
 * Only "final" is the answer; "analysis" is its thinking (kept apart, never shown as the answer).
 */
export function parseHarmony(full: string): { text: string; reasoning: string } {
  if (!full.includes("<|channel|>") && !full.startsWith("<|")) {
    const keep = full.lastIndexOf("<|");
    return { text: keep >= 0 && !full.slice(keep).includes("|>") ? full.slice(0, keep) : full, reasoning: "" };
  }
  let text = "", reasoning = "";
  const re = /<\|channel\|>([a-z]+)[^<]*?(?:<\|constrain\|>[^<]*)?<\|message\|>([\s\S]*?)(?=<\|end\|>|<\|return\|>|<\|call\|>|<\|start\|>|$)/g;
  for (let m; (m = re.exec(full)); ) {
    let body = m[2]!;
    // hold back a marker that is still being written
    const cut = body.lastIndexOf("<|");
    if (cut >= 0 && re.lastIndex >= full.length) body = body.slice(0, cut);
    if (m[1] === "analysis") reasoning += body;
    else text += body;
  }
  return { text, reasoning };
}

export class NativeBackend implements InferenceBackend {
  readonly descriptor: BackendDescriptor = { id: "native", displayName: "iPhone（アプリ）", formats: ["gguf"], tasks: ["text-generation"], usesWorker: false };
  #status: InferenceStatus = { state: "uninitialized" };
  #listeners = new Set<(s: InferenceStatus) => void>();
  #model: ModelDefinition | undefined;
  #special = { bos: "", eos: "" };
  #seq = 0;
  #downloading = "";

  #set(s: InferenceStatus) {
    this.#status = s;
    for (const l of this.#listeners) l(s);
  }
  #plugin(): HibiLLMPlugin {
    const p = nativePlugin();
    if (!p) throw new EngineError("BACKEND_UNAVAILABLE", "この AI は ひび手帳 の iPhone アプリの中でだけ使えます。");
    return p;
  }

  async checkAvailability(): Promise<BackendAvailability> {
    return nativePlugin() ? { available: true, reasons: [], device: "metal" } : { available: false, reasons: ["ひび手帳の iPhone アプリの中ではありません。"] };
  }
  async initialize(): Promise<void> {
    if (this.#status.state === "uninitialized") this.#set({ state: "idle" });
  }

  async isModelCached(model: ModelDefinition): Promise<boolean> {
    const c = cfgOf(model), p = nativePlugin();
    if (!p) return false;
    const info = await p.info({ files: c.file ? [c.file] : [] });
    if (c.kind === "apple") return info.apple === "available";
    return !!c.file && (info.files[c.file] ?? 0) > 0;
  }
  async deleteModelCache(model: ModelDefinition): Promise<void> {
    const c = cfgOf(model);
    if (c.kind === "gguf" && c.file) await this.#plugin().remove({ file: c.file });
  }

  async loadModel(model: ModelDefinition, onProgress?: (p: LoadProgress) => void, options: ModelLoadOptions = {}): Promise<void> {
    const p = this.#plugin(), c = cfgOf(model);
    const report = (pr: LoadProgress) => {
      onProgress?.(pr);
      this.#set({ state: "loading", modelId: model.id, progress: pr });
    };
    report({ fraction: 0, text: "準備しています", phase: "load" });
    if (c.kind === "apple") {
      const info = await p.info({ files: [] });
      const why = appleWhy(info.apple);
      if (why) {
        this.#set({ state: "idle" });
        throw new EngineError("MODEL_INCOMPATIBLE", why);
      }
      this.#model = model;
      this.#set({ state: "ready", modelId: model.id });
      return;
    }
    if (!c.file || !c.url) throw new EngineError("INVALID_CONFIG", "モデルの設定が足りません。");
    const info = await p.info({ files: [c.file] });
    if (!(info.files[c.file]! > 0)) {
      const need = model.downloadSize?.bytes ?? 0;
      if (need && info.free && info.free < need + 500 * 1024 * 1024) {
        this.#set({ state: "idle" });
        throw new EngineError("STORAGE_QUOTA_EXCEEDED", `iPhoneの空き容量が足りません（あと約${((need + 5e8 - info.free) / 1e9).toFixed(1)}GB必要です）。`);
      }
      const sub = await p.addListener("progress", (d) => {
        if (d.file !== c.file) return;
        const total = d.total > 0 ? d.total : need;
        report({ fraction: total ? Math.min(0.99, d.loaded / total) : undefined, loadedBytes: d.loaded, totalBytes: total || undefined, phase: "download", text: "Downloading" });
      });
      this.#downloading = c.file;
      try {
        await p.download({ url: c.url, file: c.file });
      } catch (e: any) {
        this.#set({ state: "idle" });
        const code = e?.code === "STORAGE" ? "STORAGE_QUOTA_EXCEEDED" : e?.code === "CANCELLED" ? "ABORTED" : "NETWORK_ERROR" as const;
        throw new EngineError(code, e?.message || "ダウンロードできませんでした");
      } finally {
        this.#downloading = "";
        await sub.remove().catch(() => {});
      }
    }
    report({ fraction: 1, text: "モデルを読み込んでいます", phase: "load" });
    try {
      const r = await p.load({ file: c.file, contextLength: options.contextLength ?? model.contextLength ?? 4096 });
      this.#special = { bos: r.bos || "", eos: r.eos || "" };
      this.#model = model;
      this.#set({ state: "ready", modelId: model.id });
    } catch (e: any) {
      this.#set({ state: "idle" });
      throw new EngineError("OUT_OF_MEMORY", e?.message || "モデルを読み込めませんでした");
    }
  }

  async *generate(messages: ChatMessage[], options: GenerationOptions = {}): AsyncIterable<GenerationChunk> {
    const p = this.#plugin(), model = this.#model;
    if (!model) throw new EngineError("MODEL_NOT_LOADED", "モデルが読み込まれていません。");
    const c = cfgOf(model);
    const id = `g${Date.now()}-${++this.#seq}`;
    const queue: string[] = [];
    let wake: (() => void) | null = null;
    let done = false;
    let result: GenResult | null = null;
    let failure: unknown = null;
    const sub = await p.addListener("token", (d) => {
      if (d.id !== id) return;
      queue.push(d.text);
      wake?.();
    });
    const onAbort = () => void p.stop().catch(() => {});
    options.signal?.addEventListener("abort", onAbort, { once: true });
    this.#set({ state: "generating", modelId: model.id });
    const started = performance.now();
    const call =
      c.kind === "apple"
        ? p.appleGenerate({ id, ...applePrompt(messages), temperature: options.temperature, maxTokens: options.maxTokens })
        : p.generate({
            id,
            prompt: renderPrompt(c.template ?? "", messages, this.#special),
            maxTokens: options.maxTokens ?? 512,
            temperature: options.temperature ?? 0.6,
            topP: options.topP ?? 0.95,
            renderSpecial: c.output === "harmony",
          });
    call.then(
      (r) => { result = r; done = true; wake?.(); },
      (e) => { failure = e; done = true; wake?.(); },
    );
    const think = new ThinkFilter();
    let full = "", shownText = 0, shownReasoning = 0;
    try {
      for (;;) {
        while (queue.length) {
          const piece = queue.shift()!;
          if (c.output === "harmony") {
            full += piece;
            const h = parseHarmony(full);
            if (h.reasoning.length > shownReasoning) { yield { type: "reasoning", text: h.reasoning.slice(shownReasoning) }; shownReasoning = h.reasoning.length; }
            if (h.text.length > shownText) { yield { type: "token", text: h.text.slice(shownText) }; shownText = h.text.length; }
          } else {
            const s = think.split(piece);
            if (s.reasoning) yield { type: "reasoning", text: s.reasoning };
            if (s.text) yield { type: "token", text: s.text };
          }
        }
        if (done) break;
        await new Promise<void>((res) => (wake = res));
        wake = null;
      }
      if (c.output !== "harmony") {
        const rest = think.finish();
        if (rest.text) yield { type: "token", text: rest.text };
      }
      if (failure) {
        const e: any = failure;
        throw new EngineError("UNKNOWN", e?.message || "答えられませんでした", { details: { code: e?.code } });
      }
      const r = result as GenResult | null;
      const total = r?.ms ?? performance.now() - started;
      const n = r?.completionTokens;
      const finish = r?.finish === "abort" ? "abort" : r?.finish === "length" ? "length" : r?.finish === "error" ? "error" : "stop";
      yield {
        type: "done",
        finishReason: options.signal?.aborted ? "abort" : finish,
        stats: {
          totalMs: total,
          ...(r?.firstMs ? { timeToFirstTokenMs: r.firstMs } : {}),
          ...(r?.promptTokens !== undefined ? { promptTokens: r.promptTokens } : {}),
          ...(n !== undefined ? { completionTokens: n } : {}),
          ...(n && r?.firstMs && total > r.firstMs ? { tokensPerSecond: n / ((total - r.firstMs) / 1000) } : {}),
        },
      };
    } finally {
      options.signal?.removeEventListener("abort", onAbort);
      await sub.remove().catch(() => {});
      if (this.#model) this.#set({ state: "ready", modelId: this.#model.id });
    }
  }

  async stop(): Promise<void> {
    const p = nativePlugin();
    if (!p) return;
    if (this.#downloading) await p.cancelDownload({ file: this.#downloading }).catch(() => {});
    await p.stop().catch(() => {});
  }
  async unload(): Promise<void> {
    const p = nativePlugin();
    this.#model = undefined;
    if (p) await p.unload().catch(() => {});
    this.#set({ state: "idle" });
  }
  async dispose(): Promise<void> {
    await this.unload();
    this.#listeners.clear();
  }
  getStatus(): InferenceStatus {
    return this.#status;
  }
  onStatusChange(listener: (status: InferenceStatus) => void): () => void {
    this.#listeners.add(listener);
    return () => this.#listeners.delete(listener);
  }
}
