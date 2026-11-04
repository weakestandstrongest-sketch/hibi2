/**
 * まいにち の AI：Personal AI Engine（weakestandstrongest-sketch/ai）に、ひび手帳のデータをつなぐ。
 *
 * - 推論は端末の中だけ（WebLLM / WebGPU）。ひび手帳のデータを外へ送らない。
 * - 予定・やること・空き時間の質問は、エンジンの決定的ルーティングで答える（モデルを呼ばないので、
 *   モデルをまだ読み込んでいなくても、小さいモデルでも、時刻や件数を間違えない）。
 * - それ以外（日記の中身、いつ何をしたか…）は、searchData で記録を渡して、モデルが答える。
 * - 読むだけ。ひび手帳のデータを書き換えるツールは持たない。
 */
import { PersonalAI, defineAdapter, type AITool, type MemoryQuery, type MemoryRecord, type ToolContext } from "@personal-ai-engine/core";
import { addDays, dayOf, parseDateIntent } from "@personal-ai-engine/retrieval";

/** One timed item of a day, as the まいにち timeline draws it. */
export interface DayItem {
  title: string;
  start: string; // "HH:MM"
  end: string;
  overnight?: boolean;
  fromYesterday?: boolean;
  kind?: "class" | "event";
  period?: number; // 1-based 限
  room?: string;
  memo?: string;
  done?: boolean;
  route?: string; // 「JR東海道本線 → 名鉄…」
  dest?: string;
}
export interface DayInfo {
  date: string;
  holiday?: string;
  note?: string; // 学期の休み・振替授業日
  allDay: string[];
  items: DayItem[];
}
export interface TodoItem {
  title: string;
  due?: string;
  dueTime?: string;
  start?: string;
  memo?: string;
  kind: "todo" | "assignment";
  className?: string;
}
export interface DiaryEntry {
  id: string;
  date: string;
  time?: string;
  text?: string;
  feelings?: string[];
  tags?: string[];
}
/** What the page gives the AI: read-only views over ひび手帳's data. */
export interface HibiSource {
  ready(): boolean;
  day(date: string): DayInfo;
  todos(): TodoItem[];
  entries(): DiaryEntry[];
  /** Every event's definition, for questions without a date (「最後にジムに行ったのは？」). */
  events(): Array<{ id: string; title: string; date: string; start?: string; end?: string; allDay?: boolean; repeat?: string; memo?: string }>;
}

const APP_ID = "hibi";
// the device's own time zone, so the AI and the timeline always agree on what "today" and "now" are
const TZ = Intl.DateTimeFormat().resolvedOptions().timeZone || "Asia/Tokyo";
const WD = ["日", "月", "火", "水", "木", "金", "土"];
const dayJa = (d: string) => `${+d.slice(5, 7)}月${+d.slice(8, 10)}日(${WD[new Date(+d.slice(0, 4), +d.slice(5, 7) - 1, +d.slice(8, 10)).getDay()]})`;
const hm = (t: string) => t.replace(/^0(\d)/, "$1");
const toMin = (t: string) => { const [h, m] = t.split(":").map(Number); return (h || 0) * 60 + (m || 0); };
const hhmm = (m: number) => `${Math.floor(m / 60)}:${String(m % 60).padStart(2, "0")}`;
const dur = (m: number) => (m >= 60 ? `${Math.floor(m / 60)}時間${m % 60 ? `${m % 60}分` : ""}` : `${m}分`);
const today = (ctx: ToolContext) => dayOf(ctx.now, ctx.timeZone);
const nowMin = (ctx: ToolContext) => { const d = new Date(ctx.now.toLocaleString("en-US", { timeZone: ctx.timeZone })); return d.getHours() * 60 + d.getMinutes(); };

const itemLine = (x: DayItem) =>
  `${x.fromYesterday ? "0:00" : hm(x.start)}〜${x.overnight && !x.fromYesterday ? "翌" : ""}${hm(x.end)} ${x.title}` +
  (x.kind === "class" ? `（${x.period}限${x.room ? "・" + x.room : ""}）` : "") +
  (x.route ? `（${x.route}）` : "") +
  (x.done ? "【完了】" : "");

/** The day (or days) a question is about; undefined when it names none. */
function daysOf(question: string, ctx: ToolContext): { from: string; to: string; label: string } | undefined {
  const t = today(ctx);
  const q = question.normalize("NFKC");
  if (/明後日|あさって/.test(q)) { const d = addDays(t, 2); return { from: d, to: d, label: "あさって" }; }
  const intent = parseDateIntent(q, t);
  if (intent) return { from: intent.from, to: intent.to, label: intent.label };
  return undefined;
}

// ---------------------------------------------------------------- tools (read only, routed)
function dayText(src: HibiSource, d: string): string {
  const x = src.day(d);
  const head = [dayJa(d), x.holiday, x.note].filter(Boolean).join("・");
  const lines = [...x.allDay.map((t) => `・終日 ${t}`), ...x.items.map((i) => `・${itemLine(i)}`)];
  return lines.length ? `${head}\n${lines.join("\n")}` : `${head}は、予定はありません。`;
}

function scheduleTool(src: HibiSource): AITool<{ from: string; to: string; label: string }, string[]> {
  return {
    name: "hibi_schedule",
    description: "ひび手帳の、指定した日（または期間）の予定・授業を時刻順に返す",
    access: "read",
    whenToUse: [/予定|スケジュール|授業|バイト|シフト|用事|何がある|なにがある|何する|なにする/],
    answersWithoutRecords: true,
    parameters: {
      type: "object",
      properties: { from: { type: "string", description: "YYYY-MM-DD" }, to: { type: "string", description: "YYYY-MM-DD" } },
      required: ["from"],
    },
    route(question, ctx) {
      const q = question.normalize("NFKC");
      if (!/予定|スケジュール|授業|バイト|シフト|用事|何がある|なにがある|何する|なにする|何してる|入ってる/.test(q)) return undefined;
      if (/次|つぎ|このあと|この後|空き|ひま|暇|日記|気分|去年|前の|いつ/.test(q)) return undefined; // other tools / the model
      const r = daysOf(q, ctx) ?? { from: today(ctx), to: today(ctx), label: "今日" };
      if (r.from < addDays(today(ctx), -62) || Math.round((Date.parse(r.to) - Date.parse(r.from)) / 864e5) > 13) return undefined; // long ranges: the model, from records
      return r;
    },
    formatAnswer(result, args) {
      if (!result.ok) return `予定を読めませんでした（${result.error}）。`;
      return result.value.join("\n\n");
    },
    async execute(args) {
      if (!src.ready()) throw new Error("ひび手帳のデータがまだ読み込まれていません");
      const out: string[] = [];
      for (let d = args.from; d <= (args.to || args.from); d = addDays(d, 1)) out.push(dayText(src, d));
      return out;
    },
  };
}

function nextTool(src: HibiSource): AITool<Record<string, never>, string> {
  return {
    name: "hibi_next",
    description: "ひび手帳の、いまの次の予定を返す",
    access: "read",
    whenToUse: [/次|つぎ|このあと|この後/],
    answersWithoutRecords: true,
    parameters: { type: "object", properties: {} },
    route(question) {
      return /(次|つぎ)の(予定|授業|バイト|用事)|次は何|つぎは何|次なに|このあと|この後(は|の予定)|今何(の予定)?|いまなに|いま何/.test(question.normalize("NFKC")) ? {} : undefined;
    },
    formatAnswer(result) {
      return result.ok ? result.value : `予定を読めませんでした（${result.error}）。`;
    },
    async execute(_args, ctx) {
      if (!src.ready()) throw new Error("ひび手帳のデータがまだ読み込まれていません");
      const t = today(ctx), now = nowMin(ctx);
      const cur = src.day(t).items.find((x) => !x.fromYesterday && toMin(x.start) <= now && now < toMin(x.start) + ((toMin(x.end) - toMin(x.start) + 1440) % 1440 || 1440));
      const parts: string[] = [];
      if (cur) parts.push(`いまは「${cur.title}」（${hm(cur.end)}まで）です。`);
      for (let i = 0; i < 14; i++) {
        const d = addDays(t, i);
        const nx = src.day(d).items.find((x) => !x.fromYesterday && (i > 0 || toMin(x.start) > now));
        if (!nx) continue;
        const when = i === 0 ? (toMin(nx.start) - now < 60 ? `あと${toMin(nx.start) - now}分、` : "今日の") : i === 1 ? "明日の" : `${dayJa(d)}の`;
        parts.push(`次は、${when}${hm(nx.start)}から「${nx.title}」です。`);
        return parts.join("\n");
      }
      parts.push("この先2週間、時間の決まった予定はありません。");
      return parts.join("\n");
    },
  };
}

function freeTool(src: HibiSource): AITool<{ date: string }, string> {
  return {
    name: "hibi_free_time",
    description: "ひび手帳の予定から、指定した日の空き時間（7:00〜23:00）を返す",
    access: "read",
    whenToUse: [/空き|ひま|暇|空いて/],
    answersWithoutRecords: true,
    parameters: { type: "object", properties: { date: { type: "string", description: "YYYY-MM-DD" } }, required: ["date"] },
    route(question, ctx) {
      const q = question.normalize("NFKC");
      if (!/空き時間|空いて|あいて|ひま|暇/.test(q)) return undefined;
      const r = daysOf(q, ctx);
      return r && r.from !== r.to ? undefined : { date: r ? r.from : today(ctx) };
    },
    formatAnswer(result, args) {
      return result.ok ? result.value : `予定を読めませんでした（${result.error}）。`;
    },
    async execute(args, ctx) {
      if (!src.ready()) throw new Error("ひび手帳のデータがまだ読み込まれていません");
      const d = args.date, isToday = d === today(ctx);
      let cursor = Math.max(7 * 60, isToday ? nowMin(ctx) : 0);
      const busy = src.day(d).items.map((x) => [x.fromYesterday ? 0 : toMin(x.start), x.overnight && !x.fromYesterday ? 1440 : toMin(x.end)] as const).sort((a, b) => a[0] - b[0]);
      const gaps: string[] = [];
      for (const [s, e] of [...busy, [23 * 60, 1440] as const]) {
        if (s - cursor >= 30) gaps.push(`・${hhmm(cursor)}〜${hhmm(Math.min(s, 23 * 60))}（${dur(Math.min(s, 23 * 60) - cursor)}）`);
        cursor = Math.max(cursor, e);
      }
      return gaps.length ? `${dayJa(d)}${isToday ? "（いまから）" : ""}の空き時間です。\n${gaps.join("\n")}` : `${dayJa(d)}は、30分以上の空き時間はありません。`;
    },
  };
}

function todoTool(src: HibiSource): AITool<{ within?: number }, string> {
  return {
    name: "hibi_todos",
    description: "ひび手帳の、終わっていないやること・課題を締め切り順に返す",
    access: "read",
    whenToUse: [/やること|タスク|締め?切|〆切|課題|期限|todo/i],
    answersWithoutRecords: true,
    parameters: { type: "object", properties: { within: { type: "integer", description: "何日以内の締め切りだけにするか（任意）" } } },
    route(question, ctx) {
      const q = question.normalize("NFKC");
      if (!/やること|タスク|締め?切|〆切|課題|期限|todo/i.test(q)) return undefined;
      const r = daysOf(q, ctx);
      return r ? { within: Math.max(0, Math.round((Date.parse(r.to) - Date.parse(today(ctx))) / 864e5)) } : {};
    },
    formatAnswer(result) {
      return result.ok ? result.value : `やることを読めませんでした（${result.error}）。`;
    },
    async execute(args, ctx) {
      if (!src.ready()) throw new Error("ひび手帳のデータがまだ読み込まれていません");
      const t = today(ctx);
      let list = src.todos();
      if (typeof args.within === "number") list = list.filter((x) => x.due && x.due <= addDays(t, args.within!));
      if (!list.length) return typeof args.within === "number" ? "その期間に締め切りのあるやることはありません。" : "終わっていないやることはありません。";
      const lab = (x: TodoItem) => {
        if (!x.due) return "期限なし";
        const n = Math.round((Date.parse(x.due) - Date.parse(t)) / 864e5);
        return (n < 0 ? `${-n}日過ぎ` : n === 0 ? "今日まで" : n === 1 ? "明日まで" : `${dayJa(x.due)}まで`) + (x.dueTime ? ` ${hm(x.dueTime)}` : "");
      };
      return `終わっていないやることは${list.length}件です。\n` + list.map((x) => `・${x.className ? `［${x.className}］` : ""}${x.title}（${lab(x)}）`).join("\n");
    },
  };
}

// ---------------------------------------------------------------- records for everything else (RAG)
function match(rec: MemoryRecord, words: string[]): boolean {
  if (!words.length) return true;
  const hay = `${rec.title ?? ""} ${rec.content} ${(rec.tags ?? []).join(" ")}`;
  return words.some((w) => hay.includes(w));
}
function records(src: HibiSource, q: MemoryQuery, now: Date): MemoryRecord[] {
  if (!src.ready()) return [];
  const created = now.toISOString(), out: MemoryRecord[] = [];
  const want = (t: string) => !q.types?.length || q.types.includes(t);
  const words = (q.text ?? "").split(/\s+/).filter(Boolean);
  const r = q.dateRange;
  if (want("diary"))
    for (const e of src.entries()) {
      if (r && ((r.from && e.date < r.from.slice(0, 10)) || (r.to && e.date > r.to.slice(0, 10)))) continue;
      out.push({ id: `diary:${e.id}`, appId: APP_ID, type: "diary", title: `${dayJa(e.date)}の日記`, createdAt: created, occurredAt: e.date,
        content: [e.feelings?.length ? `気持ち：${e.feelings.join("・")}` : "", e.text ?? ""].filter(Boolean).join("\n"), ...(e.tags?.length ? { tags: e.tags } : {}) });
    }
  if (want("event")) {
    if (r?.from || r?.to) {
      // each day of the range (at most 62), with repeats and classes worked out the same way as the timeline
      const from = (r.from ?? r.to!).slice(0, 10), to = (r.to ?? r.from!).slice(0, 10);
      for (let d = from, n = 0; d <= to && n < 62; d = addDays(d, 1), n++) {
        const x = src.day(d);
        x.allDay.forEach((t, i) => out.push({ id: `ev:${d}:a${i}`, appId: APP_ID, type: "event", title: t, content: `${dayJa(d)} 終日 ${t}`, createdAt: created, occurredAt: d }));
        x.items.forEach((i, k) => out.push({ id: `ev:${d}:${k}`, appId: APP_ID, type: "event", title: i.title, content: `${dayJa(d)} ${itemLine(i)}${i.memo && i.kind !== "class" ? `\nメモ：${i.memo}` : ""}`, createdAt: created, occurredAt: d }));
      }
    } else {
      for (const e of src.events())
        out.push({ id: `ev:${e.id}`, appId: APP_ID, type: "event", title: e.title, createdAt: created, occurredAt: e.date,
          content: `${dayJa(e.date)}${e.allDay ? " 終日" : e.start ? ` ${hm(e.start)}〜${hm(e.end ?? "")}` : ""} ${e.title}${e.repeat && e.repeat !== "none" ? "（繰り返し）" : ""}${e.memo ? `\nメモ：${e.memo}` : ""}` });
    }
  }
  if (want("task"))
    src.todos().forEach((x, i) => out.push({ id: `todo:${i}:${x.title}`, appId: APP_ID, type: "task", title: x.title, createdAt: created, ...(x.due ? { occurredAt: x.due } : {}),
      content: `${x.className ? `［${x.className}］` : ""}${x.title}${x.due ? `（締め切り ${dayJa(x.due)}${x.dueTime ? " " + hm(x.dueTime) : ""}）` : "（期限なし）"}${x.memo ? `\nメモ：${x.memo}` : ""}` }));
  const hit = out.filter((x) => match(x, words));
  return (hit.length || !words.length ? hit : out).sort((a, b) => (b.occurredAt ?? "").localeCompare(a.occurredAt ?? "")).slice(0, q.limit ?? 30);
}

// ---------------------------------------------------------------- engine
export const RECOMMENDED_MODEL = { id: "Qwen2.5-1.5B-Instruct-q4f16_1-MLC", contextLength: 2048, label: "Qwen2.5 1.5B" } as const;

export function createHibiAI(src: HibiSource): PersonalAI {
  const adapter = defineAdapter({
    appId: APP_ID,
    appName: "ひび手帳",
    getSystemPrompt: () =>
      "あなたは手帳アプリ「ひび手帳」のアシスタント「まいにち」です。日本語で、やさしく簡潔に答えてください。" +
      "予定・日記・やることについては、渡された記録だけを根拠にし、記録にないことは「記録にはありません」と答えてください。",
    async searchData(q) {
      return records(src, q, new Date());
    },
    getTools: () => [nextTool(src), freeTool(src), todoTool(src), scheduleTool(src)] as unknown as AITool[],
  });
  return new PersonalAI({
    appId: APP_ID,
    adapter,
    timeZone: TZ,
    generation: { temperature: 0.6, maxTokens: 512 },
    backends: {
      webllm: { createWorker: () => new Worker(new URL("./webllm.worker.ts", import.meta.url), { type: "module" }) },
    },
  });
}

