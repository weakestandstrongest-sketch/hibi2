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
  weather?: { code?: number; temp?: number };
  place?: string; // a spot id
  song?: { title: string; artist?: string };
}
export interface NewEvent {
  title: string;
  date: string; // YYYY-MM-DD
  allDay: boolean;
  start?: string; // HH:MM
  end?: string;
  /** 表示する場所: both = カレンダーとタイムライン, cal = カレンダーだけ, tl = タイムラインだけ（ルーティン） */
  show?: "both" | "cal" | "tl";
  /** カレンダーの上に「あと◯日」を出す */
  countdown?: boolean;
}
export interface MoneyEntry { k: "out" | "in"; amt: number; cat: string; memo: string; date: string }
export interface MoneyData { cats: { out: Array<{ id: string; name: string }>; in: Array<{ id: string; name: string }> }; entries: MoneyEntry[] }
/** What the page gives the AI: views over ひび手帳's data, and a way to ask ひび手帳 to add an event. */
export interface HibiSource {
  ready(): boolean;
  day(date: string): DayInfo;
  todos(): TodoItem[];
  entries(): DiaryEntry[];
  /**
   * Ask ひび手帳 to add an event: it goes to the drive's hibi-inbox.json and ひび手帳 adds it the next time it opens.
   * Returns the event as it was sent.
   */
  addEvent(e: NewEvent): Promise<NewEvent>;
  /** のりかえ（乗換案内）の API を呼ぶ：/api/search・/api/timetable・/api/train */
  nk(path: string, params: Record<string, string>): Promise<any>;
  /** よそのデータを JSON で取ってくる（凪・野球速報・天気） */
  fetchJSON(url: string): Promise<any>;
  /** ひび手帳の設定にある、凪・野球速報の URL */
  appUrl(app: "nagi" | "baseball"): string;
  /** いまいる場所（場所の記録の最後、なければ端末の位置情報） */
  where(): Promise<{ lat: number; lng: number; name?: string } | null>;
  /** いま会話に出ている経路（調べた・変えた最後のもの）と、選んでいる番号 */
  routeContext(): { result: RouteResult; selected: number } | null;
  /** ひび手帳の家計簿（支出・収入の記録とカテゴリー） */
  money(): MoneyData;
  /** ひび手帳のそのほか全部（あとで・テンプレート・バイト先・場所・メモ・習慣・予算・貯金・Asterism・凪…）。形はひび手帳のまま */
  more(): Record<string, any>;
  /** 大学（時間割・出席・学期） */
  school(): Record<string, any>;
  /** 週の始まり（ひび手帳の設定） */
  weekStartsSunday(): boolean;
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

/**
 * A follow-up such as 「あ、4月から10月ね」 says too little to be answered on its own. The page asks it as
 * `${follow-up}${FOLLOW}${the question before}`: names (外食費, ジム…) may come from either part, but a period
 * named in the follow-up wins over the one in the question before.
 */
export const FOLLOW = "\n|||\n"; // no words in it (and unchanged by NFKC): tools must not read anything into it
function periodFirst<T>(question: string, read: (q: string) => T | undefined): T | undefined {
  const [now, before] = question.split(FOLLOW);
  return before === undefined ? read(question) : read(now!) ?? read(before);
}
/** The day (or days) a question is about; undefined when it names none. */
function daysOf(question: string, ctx: ToolContext): { from: string; to: string; label: string } | undefined {
  return periodFirst(question, (q) => daysOfOne(q, ctx));
}
function daysOfOne(question: string, ctx: ToolContext): { from: string; to: string; label: string } | undefined {
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

// ---------------------------------------------------------------- adding an event (write: always confirmed first)
const ADD_Q = /追加|入れて|いれて|入れと|いれと|入れてお|登録|予定に(して|する)|予定を(作|つく)って|予定(入|はい)れ/;
const pad = (n: number) => String(n).padStart(2, "0");
/** 「明日15時から17時までバイトを追加して」 → { title: "バイト", date, start: "15:00", end: "17:00" } */
export function parseAdd(question: string, ctx: ToolContext): NewEvent | undefined {
  let q = question.normalize("NFKC").trim();
  if (!ADD_Q.test(q) || /できる|できます|でき\?|方法|どうやって|やり方|消して|削除|変更/.test(q)) return undefined;
  const t = today(ctx);
  // the day
  let date = t;
  const wdm = q.match(/(今週|来週|再来週)?の?\s*([月火水木金土日])曜日?/);
  if (/明後日|あさって/.test(q)) { date = addDays(t, 2); q = q.replace(/(明後日|あさって)(に|の|は)?/, " "); }
  else if (wdm) {
    // 「金曜に」 the next one (today counts); 「来週の金曜」 the one in next week (weeks start on Monday)
    const want = "日月火水木金土".indexOf(wdm[2]!), now = new Date(+t.slice(0, 4), +t.slice(5, 7) - 1, +t.slice(8, 10)).getDay();
    if (wdm[1]) { const mon = addDays(t, -((now + 6) % 7)); date = addDays(mon, (wdm[1] === "来週" ? 7 : wdm[1] === "再来週" ? 14 : 0) + (want + 6) % 7); }
    else date = addDays(t, (want - now + 7) % 7);
    q = q.replace(new RegExp(`${wdm[0]}(に|の|は)?`), " ");
  } else {
    const intent = parseDateIntent(q, t);
    if (intent) {
      if (intent.from !== intent.to) return undefined;
      date = intent.from;
      // the date reader looks back (made for the diary); a plan without a year is the next one to come
      if (date < t && !/年/.test(intent.phrase)) { const y = new Date(+t.slice(0, 4), +date.slice(5, 7) - 1, +date.slice(8, 10)) < new Date(+t.slice(0, 4), +t.slice(5, 7) - 1, +t.slice(8, 10)) ? +t.slice(0, 4) + 1 : +t.slice(0, 4); date = `${y}${date.slice(4)}`; }
      q = q.replace(new RegExp(`${intent.phrase.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}(に|の|は)?`), " ");
    }
  }
  // the times: 「15時」「15:30」「3時半」「午後3時」
  const times: number[] = [];
  q = q.replace(/(午前|午後|朝|夜|夕方)?\s*(\d{1,2})\s*(?:時|:)\s*(?:(\d{1,2})\s*分?|(半))?\s*(から|〜|~|-|ー|まで|に|の)?/g, (_m, ap, h, mi, half) => {
    let hh = +h; const mm = half ? 30 : mi ? +mi : 0;
    if ((ap === "午後" || ap === "夜" || ap === "夕方") && hh < 12) hh += 12;
    if (hh <= 23 && mm <= 59) times.push(hh * 60 + mm);
    return " ";
  });
  // the name: what is left once the request words and particles are gone
  const title = q
    .replace(/(を|は)?\s*(カレンダー|タイムライン)(に)?(だけ|のみ)(に|で)?/g, " ")
    .replace(/(を|で|と)?\s*カウントダウン(も|を)?(して|する|つき|付き)?(で|の)?/g, " ")
    .replace(/(という|って)?(の)?予定(を|に)?/g, " ")
    .replace(/(を|に)?\s*(追加|入れ|いれ|登録|作っ|つくっ)[^\s]*$/, " ")
    .replace(/[、。!?！？]/g, " ")
    .replace(/^(\s*(に|で|は|を|の|から)\s+)+/, "")
    .replace(/(\s*(に|で|は|を|の)\s*)+$/, "")
    .replace(/\s+/g, " ").trim();
  if (!title || title.length > 40) return undefined;
  const extra: Pick<NewEvent, "show" | "countdown"> = {
    ...(/カレンダー(に)?だけ|カレンダーのみ/.test(question) ? { show: "cal" as const } : /タイムライン(に)?だけ|タイムラインのみ|ルーティン/.test(question) ? { show: "tl" as const } : {}),
    ...(/カウントダウン/.test(question) ? { countdown: true } : {}),
  };
  if (!times.length) return { title, date, allDay: true, ...extra };
  const s0 = times[0]!, e0 = times[1] !== undefined && times[1] > s0 ? times[1] : Math.min(s0 + 60, 23 * 60 + 59);
  return { title, date, allDay: false, start: `${pad(Math.floor(s0 / 60))}:${pad(s0 % 60)}`, end: `${pad(Math.floor(e0 / 60))}:${pad(e0 % 60)}`, ...extra };
}
const eventLabel = (e: NewEvent) => {
  const opts = [e.show === "cal" ? "カレンダーだけ" : e.show === "tl" ? "タイムラインだけ" : "", e.countdown ? "カウントダウン" : ""].filter(Boolean);
  return `${dayJa(e.date)} ${e.allDay ? "終日" : `${hm(e.start!)}〜${hm(e.end!)}`}${opts.length ? `（${opts.join("・")}）` : ""}`;
};

function addTool(src: HibiSource): AITool<NewEvent & Record<string, unknown>, NewEvent> {
  return {
    name: "hibi_add_event",
    description: "ひび手帳に予定を追加する。「明日15時にバイトを追加して」のように頼まれたときに使う",
    access: "write",
    whenToUse: [ADD_Q],
    parameters: {
      type: "object",
      properties: {
        title: { type: "string", description: "予定の名前" },
        date: { type: "string", description: "YYYY-MM-DD" },
        allDay: { type: "boolean", description: "終日の予定なら true" },
        start: { type: "string", description: "開始 HH:MM（終日なら省略）" },
        end: { type: "string", description: "終了 HH:MM（終日なら省略）" },
        show: { type: "string", enum: ["both", "cal", "tl"], description: "表示する場所：both（両方）、cal（カレンダーだけ）、tl（タイムラインだけ）" },
        countdown: { type: "boolean", description: "カレンダーの上に「あと◯日」を出すなら true" },
      },
      required: ["title", "date"],
    },
    route(question, ctx) {
      return parseAdd(question, ctx) as (NewEvent & Record<string, unknown>) | undefined;
    },
    formatAnswer(result) {
      if (result.ok) return `「${result.value.title}」を ${eventLabel(result.value)} で追加しました。\nひび手帳を開くと、ひび手帳の予定に入ります（それまでは、まいにち で「追加待ち」と表示します）。`;
      if (result.declined) return "予定は追加しませんでした。";
      return `予定を追加できませんでした（${result.error}）。`;
    },
    async execute(args) {
      const date = String(args.date ?? ""), allDay = args.allDay === true || !args.start;
      if (!/^\d{4}-\d{2}-\d{2}$/.test(date)) throw new Error("日付が正しくありません");
      const show = args.show === "cal" || args.show === "tl" ? args.show : "both";
      const e: NewEvent = { title: String(args.title ?? "").trim().slice(0, 60), date, allDay, ...(allDay ? {} : { start: String(args.start), end: String(args.end || args.start) }),
        show, ...(args.countdown === true ? { countdown: true } : {}) };
      if (!e.title) throw new Error("予定の名前がありません");
      if (!allDay && !(/^([01]\d|2[0-3]):[0-5]\d$/.test(e.start!) && /^([01]\d|2[0-3]):[0-5]\d$/.test(e.end!))) throw new Error("時刻が正しくありません");
      return src.addEvent(e);
    },
  };
}

// ---------------------------------------------------------------- 経路（のりかえ）: search, and change by talking
/** A step of a route, the same as のりかえ's 「ひび手帳に追加」 and ひび手帳's own routes */
export type RouteStep =
  | { k: "s"; name: string; arr: string; dep: string; code?: string }
  | { k: "a"; line: string; dest: string; color: string; walk: boolean; platform: string; stops: string; stopList: Array<{ time: string; name: string }> };
export interface RoutePayload { v: 1; from: string; to: string; via: string; date: string; dep: string; arr: string; duration: string; transfers: number | null; fare: number | null; fareNote: string; tags: string[]; steps: RouteStep[] }
export interface RouteQuery { from: string; to: string; date: string; time: string; type: "dep" | "arr" | "first" | "last"; shin: boolean; ex: boolean; via: string }
export interface RouteResult { query: RouteQuery; routes: RoutePayload[]; text: string }

const R_Q = /経路|乗り?換え|乗換|のりかえ|行き方|行きかた|どうやって行|ルート|電車で/;
const rtNorm = (s: string) => String(s || "").normalize("NFKC").replace(/\s+/g, "").replace(/[(（][^)）]*[)）]/g, "").replace(/駅$/, "");
const hm2 = (m: number) => `${pad(Math.floor((((m % 1440) + 1440) % 1440) / 60))}:${pad((((m % 1440) + 1440) % 1440) % 60)}`;
const minOf = (t: string) => { const x = /^(\d{1,2}):(\d{2})/.exec(t || ""); return x ? +x[1]! * 60 + +x[2]! : NaN; };

/** the day a request names: 「明日」「11月9日」「金曜」…（年のない日付は、これから来るその日） */
function takeDay(q0: string, t: string): { date: string; q: string } | undefined {
  let q = q0, date = t;
  const wdm = q.match(/(今週|来週|再来週)?の?\s*([月火水木金土日])曜日?/);
  if (/明後日|あさって/.test(q)) { date = addDays(t, 2); q = q.replace(/(明後日|あさって)(に|の|は)?/, " "); }
  else if (wdm) {
    const want = "日月火水木金土".indexOf(wdm[2]!), now = new Date(+t.slice(0, 4), +t.slice(5, 7) - 1, +t.slice(8, 10)).getDay();
    if (wdm[1]) { const mon = addDays(t, -((now + 6) % 7)); date = addDays(mon, (wdm[1] === "来週" ? 7 : wdm[1] === "再来週" ? 14 : 0) + (want + 6) % 7); }
    else date = addDays(t, (want - now + 7) % 7);
    q = q.replace(new RegExp(`${wdm[0]}(に|の|は)?`), " ");
  } else {
    const intent = parseDateIntent(q, t);
    if (intent) {
      if (intent.from !== intent.to) return undefined;
      date = intent.from;
      if (date < t && !/年/.test(intent.phrase)) { const y = new Date(+t.slice(0, 4), +date.slice(5, 7) - 1, +date.slice(8, 10)) < new Date(+t.slice(0, 4), +t.slice(5, 7) - 1, +t.slice(8, 10)) ? +t.slice(0, 4) + 1 : +t.slice(0, 4); date = `${y}${date.slice(4)}`; }
      q = q.replace(new RegExp(`${intent.phrase.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}(に|の|は)?`), " ");
    }
  }
  return { date, q };
}
/** 「11月9日に、繰舟橋から刈谷市駅へ16時につきたい、経路を調べて」 → { from: 繰舟橋, to: 刈谷市, date, time: 16:00, type: arr } */
export function parseRouteAsk(question: string, ctx: ToolContext): RouteQuery | undefined {
  const q0 = question.normalize("NFKC");
  if (!/から/.test(q0) || !(R_Q.test(q0) || /(調べ|探し|教え|検索)/.test(q0))) return undefined;
  if (/日記|家計簿|円|いくら/.test(q0)) return undefined;
  const t = today(ctx), d = takeDay(q0, t); if (!d) return undefined;
  let q = d.q, time = "", type: RouteQuery["type"] = "dep";
  if (/終電/.test(q)) type = "last"; else if (/始発/.test(q)) type = "first";
  q = q.replace(/(午前|午後|朝|夜|夕方)?\s*(\d{1,2})\s*(?:時|:)\s*(?:(\d{1,2})\s*分?|(半))?\s*(まで(に)?|に|の)?\s*(着き?|つき|つく|着く|到着|出発|出|発)?/, (m, ap, h, mi, half, _p, _n, verb) => {
    let hh = +h; const mm = half ? 30 : mi ? +mi : 0;
    if ((ap === "午後" || ap === "夜" || ap === "夕方") && hh < 12) hh += 12;
    if (hh <= 24 && mm <= 59) time = `${pad(hh % 24)}:${pad(mm)}`;
    if (verb && /着|つ|到/.test(verb)) type = "arr"; else if (/まで/.test(m)) type = "arr";
    return " ";
  });
  if (/着きたい|つきたい|到着したい|に着く|につく|までに/.test(q0) && type === "dep" && time) type = "arr";
  const via = (q.match(/([^\s、。,，「」]+?)(?:駅)?経由/) ?? [])[1] ?? "";
  q = q.replace(/([^\s、。,，「」]+?)(?:駅)?経由(で|して)?/, " ");
  const m = q.match(/([^\s、。,，「」()（）]+?)(?:駅)?から\s*([^\s、。,，「」()（）]+?)(?:駅)?(?:まで|へ|に|行|の|$)/);
  if (!m) return undefined;
  const clean = (x: string) => x.replace(/^(に|で|は|を|の|、)+/, "").trim();
  const from = clean(m[1]!), to = clean(m[2]!);
  if (!from || !to || from.length > 30 || to.length > 30) return undefined;
  const date = d.date;
  if (!time) { if (date === t) { const n = nowMin(ctx); time = hm2(n); } else time = "09:00"; }
  return { from, to, date, time, type, via, shin: !/新幹線(は|を)?(使わ|なし|乗らない|以外)/.test(q0), ex: !/特急(は|を)?(使わ|なし|乗らない|以外)/.test(q0) };
}
/** a route of the API (/api/search) as ひび手帳 keeps it */
function payloadOf(r: any, q: RouteQuery): RoutePayload {
  const s = r.summary || {}, steps: RouteStep[] = (r.steps || []).map((st: any) => (st.kind === "station"
    ? { k: "s", name: String(st.name || ""), arr: st.arr || "", dep: st.dep || "", code: st.code || "" }
    : { k: "a", line: st.line || "", dest: st.destination || "", color: st.color || "", walk: !!st.walk, platform: st.platform || "", stops: st.stops || "", stopList: (st.stopList || []).map((x: any) => ({ time: x.time || "", name: x.name || "" })) }));
  const sts = steps.filter((x): x is Extract<RouteStep, { k: "s" }> => x.k === "s");
  const dep = s.dep || sts[0]?.dep || "", arr = s.arr || sts[sts.length - 1]?.arr || "";
  // an arrive-by search can set off the day before (23:50 → 0:20)
  const date = q.type === "arr" && minOf(dep) > minOf(arr) ? addDays(q.date, -1) : q.date;
  return { v: 1, from: sts[0]?.name || q.from, to: sts[sts.length - 1]?.name || q.to, via: q.via || "", date, dep, arr, duration: s.duration || "", transfers: Number.isFinite(+s.transfers) && s.transfers != null ? +s.transfers : null,
    fare: Number.isFinite(+s.fare) && s.fare != null ? +s.fare : null, fareNote: s.fareNote || "", tags: Array.isArray(s.tags) ? s.tags : [], steps };
}
async function searchRoutes(src: HibiSource, q: RouteQuery): Promise<RoutePayload[]> {
  const j = await src.nk("/api/search", { from: q.from, to: q.to, date: q.date, time: q.time, type: q.type, shin: q.shin ? "1" : "0", ex: q.ex ? "1" : "0", ...(q.via ? { via: q.via } : {}) });
  return (Array.isArray(j?.routes) ? j.routes : []).map((r: any) => payloadOf(r, q)).filter((r: RoutePayload) => r.dep && r.arr && r.steps.length);
}
const lines = (r: RoutePayload) => r.steps.filter((x) => x.k === "a" && !x.walk).map((x) => (x as any).line).filter(Boolean);
export const routeLine = (r: RoutePayload, i?: number) =>
  `${i !== undefined ? `${"①②③④⑤⑥"[i] ?? i + 1} ` : ""}${hm(r.dep)}発 → ${hm(r.arr)}着（${[r.duration.replace(/\(.*\)$/, ""), r.transfers != null ? `乗換${r.transfers}回` : "", r.fare != null ? `${r.fareNote === "目安" ? "約" : ""}${r.fare.toLocaleString()}円` : ""].filter(Boolean).join("・")}）\n　${lines(r).join(" → ")}`;
const late = (q: RouteQuery, r: RoutePayload) => (q.type === "arr" && minOf(r.arr) > minOf(q.time) && minOf(r.arr) - minOf(q.time) < 720 ? `\n※ ${hm(q.time)}には間に合いません（${hm(r.arr)}着）` : "");

function routeSearchTool(src: HibiSource): AITool<RouteQuery & Record<string, unknown>, RouteResult> {
  return {
    name: "hibi_route",
    description: "のりかえ（乗換案内）で、出発駅から到着駅までの経路を調べる",
    access: "read",
    whenToUse: [R_Q, /から.+(まで|へ)/],
    answersWithoutRecords: true,
    parameters: { type: "object", properties: { from: { type: "string" }, to: { type: "string" }, date: { type: "string" }, time: { type: "string" }, type: { type: "string", enum: ["dep", "arr", "first", "last"] },
      shin: { type: "boolean", description: "新幹線を使うか" }, ex: { type: "boolean", description: "有料特急を使うか" }, via: { type: "string", description: "経由する駅" } }, required: ["from", "to"] },
    route(question, ctx) { return parseRouteAsk(question, ctx) as (RouteQuery & Record<string, unknown>) | undefined; },
    formatAnswer(result) { return result.ok ? result.value.text : `経路を調べられませんでした（${result.error}）。`; },
    async execute(q) {
      const routes = (await searchRoutes(src, q)).slice(0, 3);
      if (!routes.length) throw new Error("経路が見つかりませんでした。駅名を確かめてください");
      const head = `${dayJa(q.date)}、${q.from} → ${q.to}${q.via ? `（${q.via}経由）` : ""}、${q.type === "arr" ? `${hm(q.time)}に着く` : q.type === "last" ? "終電" : q.type === "first" ? "始発" : `${hm(q.time)}に出る`}経路です。`;
      return { query: q, routes, text: `${head}\n${routes.map((r, i) => routeLine(r, i)).join("\n")}${late(q, routes[0]!)}\n\n「ひび手帳に追加」で予定に入れられます。「名鉄名古屋から名鉄岐阜は普通にしたい」「特急は使わないで」「1本あとで」のように話すと、経路を変えられます。` };
    },
  };
}

// ---- changing the route on screen
const KIND_RE = /(快速特急|快速急行|区間準急|区間急行|特別快速|新快速|各駅停車|各停|普通|準急|急行|快速|特急)/;
const kindOk = (want: string, k: string) => (/普通|各駅停車|各停/.test(want) ? /^(普通|各駅停車|各停|ワンマン)/.test(k) : k === want || (want === "特急" && /特急/.test(k) && !/快速特急/.test(k)) || k.startsWith(want));
interface ChangeArgs { mode: "kind" | "noex" | "noshin" | "later" | "earlier" | "via" | "show"; i: number; kind?: string; dest?: string; a?: string; b?: string; via?: string; text?: string }
function parseChange(q0: string, ctxR: { result: RouteResult; selected: number } | null): ChangeArgs | undefined {
  if (!ctxR) return undefined;
  const q = q0.normalize("NFKC"), rs = ctxR.result.routes;
  const n = q.match(/([1-6１-６①②③④⑤⑥一二三])\s*(つ目|番目|番|個目)?の?(経路|ルート|案)?/);
  const pick = (x?: string) => { if (!x) return undefined; const k = "1１①一".includes(x) ? 0 : "2２②二".includes(x) ? 1 : "3３③三".includes(x) ? 2 : +x - 1; return k >= 0 && k < rs.length ? k : undefined; };
  const i = (n && (n[2] || n[3]) ? pick(n[1]) : undefined) ?? ctxR.selected ?? 0;
  const r = rs[i]; if (!r) return undefined;
  if (/特急(は|を)?(使わ|なし|乗らない|以外|やめ)/.test(q)) return { mode: "noex", i };
  if (/新幹線(は|を)?(使わ|なし|乗らない|以外|やめ)/.test(q)) return { mode: "noshin", i };
  if (/(1|一|ひと)本(あと|後|遅|おそ)|次の電車|遅らせ|もっと遅|ゆっくり出/.test(q)) return { mode: "later", i };
  if (/(1|一|ひと)本(前|早|はや)|もっと早|早めに/.test(q)) return { mode: "earlier", i };
  const via = (q.match(/([^\s、。,，「」]+?)(?:駅)?経由/) ?? [])[1];
  if (via) return { mode: "via", i, via };
  const km = q.match(KIND_RE);
  // a kind of train alone is not enough (「普通に楽しかった」): the train, the section or a station of the route must be talked about too
  const aboutRoute = /電車|列車|線|行き?の|行きで|行に|区間|駅|乗|経路|ルート|にしたい|に変え|にして|で行/.test(q) || r.steps.some((x) => x.k === "s" && rtNorm(x.name).length >= 2 && q.includes(rtNorm(x.name)));
  if (km && aboutRoute) {
    // 「普通の岐阜行」「岐阜行きの普通」
    const dm = q.match(new RegExp(`${km[1]}(?:の|で)?\\s*([^\\s、。のでにを]{1,8}?)行き?`)) ?? q.match(new RegExp(`([^\\s、。のでにを]{1,8}?)行き?の\\s*${km[1]}`));
    const sts = r.steps.filter((x) => x.k === "s").map((x) => x.name);
    const named = sts.map((name) => ({ name, at: q.indexOf(rtNorm(name)) })).filter((x) => x.at >= 0 && rtNorm(x.name).length >= 2);
    // a station that is only the destination of the train asked for (「岐阜行」) is not a station of the section
    const hits = [...new Map(named.filter((x) => !dm || !q.slice(x.at, x.at + rtNorm(x.name).length + 1).match(/行/)).sort((p, q2) => sts.indexOf(p.name) - sts.indexOf(q2.name)).map((x) => [rtNorm(x.name), x])).values()];
    return { mode: "kind", i, kind: km[1]!, text: q, ...(dm ? { dest: dm[1] } : {}), ...(hits[0] ? { a: hits[0].name } : {}), ...(hits[1] ? { b: hits[1].name } : {}) };
  }
  if (/(この|その|これ|それ|経路|ルート).*(追加|入れ)/.test(q) || /(追加|入れ).*(この|その|これ|それ)/.test(q)) return { mode: "show", i };
  return undefined;
}
async function changeRoute(src: HibiSource, a: ChangeArgs, ctxR: { result: RouteResult; selected: number }): Promise<RouteResult> {
  const { query } = ctxR.result, r = ctxR.result.routes[a.i]!;
  const done = (routes: RoutePayload[], q: RouteQuery, head: string): RouteResult => ({ query: q, routes, text: `${head}\n${routes.map((x, k) => routeLine(x, routes.length > 1 ? k : undefined)).join("\n")}${late(q, routes[0]!)}` });
  if (a.mode === "show") return done([r], query, "この経路です。下の「ひび手帳に追加」を押すと、ひび手帳の予定に入ります。");
  if (a.mode === "noex" || a.mode === "noshin" || a.mode === "via") {
    const q = { ...query, ...(a.mode === "noex" ? { ex: false } : a.mode === "noshin" ? { shin: false } : { via: a.via! }) };
    const routes = (await searchRoutes(src, q)).slice(0, 3); if (!routes.length) throw new Error("その条件の経路が見つかりませんでした");
    return done(routes, q, a.mode === "noex" ? "有料特急を使わない経路です。" : a.mode === "noshin" ? "新幹線を使わない経路です。" : `${a.via}を通る経路です。`);
  }
  if (a.mode === "later" || a.mode === "earlier") {
    const q: RouteQuery = a.mode === "later" ? { ...query, type: "dep", time: hm2(minOf(r.dep) + 1), date: r.date } : { ...query, type: "arr", time: hm2(minOf(r.arr) - 1) };
    const routes = (await searchRoutes(src, q)).filter((x) => (a.mode === "later" ? minOf(x.dep) > minOf(r.dep) : minOf(x.arr) < minOf(r.arr))).slice(0, 1);
    if (!routes.length) throw new Error(a.mode === "later" ? "あとの電車が見つかりませんでした" : "前の電車が見つかりませんでした");
    return done(routes, { ...query, ...(a.mode === "later" ? {} : {}) }, a.mode === "later" ? "1本あとの経路です。" : "1本前の経路です。");
  }
  // ---- a section by another kind of train: the boarding station's timetable → that train's stops → the rest searched from where it is left
  const steps = r.steps, sIdx = steps.map((x, k) => (x.k === "s" ? k : -1)).filter((k) => k >= 0);
  const rideAfter = (k: number) => { const n = steps[k + 1]; return n && n.k === "a" && !n.walk; };
  let iA = a.a ? steps.findIndex((x) => x.k === "s" && rtNorm(x.name) === rtNorm(a.a!)) : -1;
  if (iA < 0 || !rideAfter(iA)) iA = sIdx.find((k) => rideAfter(k) && (!a.b || sIdx.some((j) => j > k && rtNorm((steps[j] as any).name) === rtNorm(a.b!)))) ?? -1;
  if (iA < 0) throw new Error("電車に乗る駅が見つかりませんでした");
  const A = steps[iA] as Extract<RouteStep, { k: "s" }>, seg = steps[iA + 1] as Extract<RouteStep, { k: "a" }>;
  if (!A.code) throw new Error(`${A.name}の時刻表が分かりませんでした`);
  const first = iA === 0 || steps.slice(0, iA).every((x) => x.k === "s" || x.walk);
  const reach = first ? null : minOf(A.arr) + 1; // a minute to change trains
  const tt = await src.nk("/api/timetable", { code: A.code, date: r.date, time: hm2(minOf(A.dep)), line: seg.line, dest: a.dest || seg.dest.replace(/行き?$/, "") });
  const trains: any[] = Array.isArray(tt?.trains) ? tt.trains : [];
  const fits = trains.filter((t) => kindOk(a.kind!, String(t.kind || "")) && (!a.dest || String(t.destination || "").includes(a.dest)));
  if (!fits.length) throw new Error(`${A.name}から乗れる${a.kind}${a.dest ? `（${a.dest}行）` : ""}が時刻表にありません`);
  // after arriving there; from the start of the trip, the one closest to the original time (an arrive-by trip: not later than it)
  const want = minOf(A.dep);
  const t = reach != null ? fits.find((x) => x.min >= reach) : query.type === "arr" ? [...fits].reverse().find((x) => x.min <= want) ?? fits[0] : fits.find((x) => x.min >= want - 5) ?? fits[fits.length - 1];
  if (!t) throw new Error(`${A.name}で乗り換えられる${a.kind}がありません`);
  const tr = await src.nk("/api/train", { code: A.code, group: String(tt.groupId ?? ""), train: String(t.trainId ?? ""), kind: String(tt.kind ?? ""), hh: String(Math.floor(t.min / 60)), mm: String(t.min % 60) });
  const stops: Array<{ name: string; arr?: string; dep?: string }> = Array.isArray(tr?.stops) ? tr.stops : [];
  const here = stops.findIndex((x) => rtNorm(x.name) === rtNorm(A.name));
  if (here < 0) throw new Error("その電車の停車駅を読めませんでした");
  const finalName = (steps[steps.length - 1] as any).name as string, segEnd = (steps[iA + 2] as any)?.name as string | undefined;
  const after = (n?: string) => (n ? stops.findIndex((x, k) => k > here && rtNorm(x.name) === rtNorm(n)) : -1);
  let off = after(a.b);
  if (a.b && off < 0) throw new Error(`その${t.kind}は${a.b}に止まりません`);
  if (off < 0 && a.text) { // a stop of this train named in what was said, not only stations of the route (「金山まで普通」)
    const said = stops.map((x, k) => ({ k, at: a.text!.indexOf(rtNorm(x.name)), n: rtNorm(x.name) })).filter((x) => x.k > here && x.at >= 0 && x.n.length >= 2 && !/^行/.test(a.text!.slice(x.at + x.n.length)));
    if (said.length) off = said.sort((p, q2) => p.at - q2.at)[0]!.k;
  }
  if (off < 0) off = after(finalName);
  if (off < 0) off = after(segEnd);
  if (off < 0) off = stops.length - 1;
  const S0 = stops[off]!, offArr = S0.arr || S0.dep || "";
  const prefix = steps.slice(0, iA + 1).map((x, k) => (k === iA ? { ...(x as any), dep: t.time } : x)) as RouteStep[];
  const ride: RouteStep = { k: "a", line: `${String(tt.railName || seg.line).replace(/\s+/g, "")} ${t.kind}`.trim(), dest: `${t.destination}${/行$/.test(t.destination) ? "" : "行"}`, color: seg.color, walk: false,
    platform: t.track ? `${t.track}番線` : "", stops: `${off - here}駅`, stopList: stops.slice(here + 1, off).map((x) => ({ name: x.name, time: x.dep || x.arr || "" })) };
  let rest: RouteStep[] = [{ k: "s", name: S0.name, arr: offArr, dep: "" }];
  if (rtNorm(S0.name) !== rtNorm(finalName)) {
    const q2: RouteQuery = { ...query, from: S0.name, to: finalName, type: "dep", time: hm2(minOf(offArr) + 1), date: r.date, via: "" };
    const more = (await searchRoutes(src, q2))[0];
    if (!more) throw new Error(`${S0.name}から先の経路が見つかりませんでした`);
    rest = [{ k: "s", name: S0.name, arr: offArr, dep: more.dep, code: (more.steps[0] as any).code || "" }, ...more.steps.slice(1)];
  }
  const all = [...prefix, ride, ...rest], sts = all.filter((x) => x.k === "s") as Array<Extract<RouteStep, { k: "s" }>>;
  const dep = (sts[0]!.dep || t.time), arr = sts[sts.length - 1]!.arr, mins = (minOf(arr) - minOf(dep) + 1440) % 1440;
  const nr: RoutePayload = { ...r, dep, arr, duration: dur(mins), transfers: Math.max(0, all.filter((x) => x.k === "a" && !x.walk).length - 1), fareNote: "目安", tags: [], steps: all };
  return done([nr], query, `${A.name}から${S0.name}を、${hm(t.time)}発の${t.kind} ${ride.dest}にしました（運賃は前の経路のときの目安です）。`);
}
function routeChangeTool(src: HibiSource): AITool<ChangeArgs & Record<string, unknown>, RouteResult> {
  return {
    name: "hibi_route_change",
    description: "会話に出ている経路を、区間の電車の種別・特急なし・1本あと・経由などで変える",
    access: "read",
    answersWithoutRecords: true,
    parameters: { type: "object", properties: { mode: { type: "string" }, i: { type: "integer" }, kind: { type: "string" }, dest: { type: "string" }, a: { type: "string" }, b: { type: "string" }, via: { type: "string" }, text: { type: "string" } }, required: ["mode"] },
    route(question) {
      const q = question.split(FOLLOW)[0]!;
      if (parseRouteAsk(q, { appId: APP_ID, now: new Date(), timeZone: TZ })) return undefined; // a new search
      return parseChange(q, src.routeContext()) as (ChangeArgs & Record<string, unknown>) | undefined;
    },
    formatAnswer(result) { return result.ok ? result.value.text : `経路を変えられませんでした（${result.error}）。`; },
    async execute(a) {
      const c = src.routeContext(); if (!c) throw new Error("変える経路がありません。先に経路を調べてください");
      return changeRoute(src, a, c);
    },
  };
}

/** 「今日（11月4日(水)）」, but a label that is already a date is said once */
const whenJa = (label: string, d: string) => (/^(今日|明日|昨日|あさって|おととい|明後日|一昨日)$/.test(label) ? `${label}（${dayJa(d)}）` : dayJa(d));
// ---------------------------------------------------------------- 凪（ニュース）
const NEWS_Q = /ニュース|凪|話題|報道|見出し|トップ記事|最新の記事/;
interface NewsArticle { id: string; title: string; url: string; source: string; publishedAt: string; summary?: string; topic?: string; paid?: unknown; memberOnly?: unknown }
interface NewsArgs { text: string; nth?: number }
const timeJa = (iso: string) => { const d = new Date(iso); return Number.isNaN(+d) ? "" : `${d.getMonth() + 1}/${d.getDate()} ${d.getHours()}:${pad(d.getMinutes())}`; };
function newsTool(src: HibiSource, mem: { list: NewsArticle[] }): AITool<NewsArgs & Record<string, unknown>, string> {
  return {
    name: "hibi_news",
    description: "凪（ニュース）から、今日の主なニュース・カテゴリー・言葉で探したニュース・話題を出す",
    access: "read",
    whenToUse: [NEWS_Q],
    answersWithoutRecords: true,
    parameters: { type: "object", properties: { text: { type: "string" }, nth: { type: "integer" } }, required: ["text"] },
    route(question) {
      const q = question.split(FOLLOW)[0]!.normalize("NFKC");
      // 「2つ目を詳しく」: the list shown last
      const n = q.match(/([1-9１-９①-⑨一二三四五六七八九])\s*(つ目|番目|番|個目)?(の(ニュース|記事))?(を|の)?\s*(詳しく|くわしく|概要|内容|中身|教えて)/);
      if (n && mem.list.length) { const c = n[1]!; const k = "一二三四五六七八九".indexOf(c) >= 0 ? "一二三四五六七八九".indexOf(c) : "①②③④⑤⑥⑦⑧⑨".indexOf(c) >= 0 ? "①②③④⑤⑥⑦⑧⑨".indexOf(c) : +c.normalize("NFKC") - 1; return { text: q, nth: k }; }
      if (!NEWS_Q.test(q) || /保存|しおり|ブックマーク|メモした/.test(q)) return undefined; // saved articles: from the records
      return { text: q };
    },
    formatAnswer(result) { return result.ok ? result.value : `ニュースを取ってこられませんでした（${result.error}）。`; },
    async execute(a) {
      if (a.nth !== undefined) {
        const x = mem.list[a.nth]; if (!x) throw new Error("その番号のニュースはありません");
        return `${x.title}\n（${x.source}・${timeJa(x.publishedAt)}）\n\n${x.summary || "概要はありません。"}\n\n${x.url}`;
      }
      const base = src.appUrl("nagi").replace(/\/+$/, "");
      const d = await src.fetchJSON(`${base}/data/news.json`);
      const cats: Array<{ id: string; label: string; articles: NewsArticle[] }> = arr(d?.categories);
      const all = [...new Map(cats.flatMap((c) => arr(c.articles)).map((x: NewsArticle) => [x.id, x])).values()] as NewsArticle[];
      if (!all.length) throw new Error("凪にニュースがありません");
      const q = a.text, upd = d.generatedAt ? `（凪の更新：${timeJa(d.generatedAt)}）` : "";
      const show = (head: string, list: NewsArticle[]) => {
        mem.list = list.slice(0, 8);
        return `${head}${upd}\n${mem.list.map((x, i) => `${"①②③④⑤⑥⑦⑧"[i]} ${x.title}（${x.source}${x.publishedAt ? `・${timeJa(x.publishedAt)}` : ""}）`).join("\n")}\n\n「2つ目を詳しく」のように聞くと、概要を出します。`;
      };
      // 話題：同じ出来事を報じた社の多い順
      if (/話題|トピック|注目|いま何が/.test(q)) {
        const byId = new Map(all.map((x) => [x.id, x]));
        const tops = arr(d.topics).sort((p: any, r: any) => arr(r.sources).length - arr(p.sources).length).slice(0, 8)
          .map((t: any) => { const x = byId.get(arr(t.articleIds)[0]); return x ? { ...x, source: `${arr(t.sources).length}社が報道` } : null; }).filter(Boolean) as NewsArticle[];
        if (tops.length) return show("いま多くのメディアが報じている話題です。", tops);
      }
      // a category named: 「ビジネスのニュース」「スポーツ」
      const cat = cats.find((c) => c.label && c.id !== "top" && (q.includes(c.label) || c.label.split(/[・/]/).some((w) => w.length >= 2 && q.includes(w))));
      if (cat) return show(`凪の「${cat.label}」のニュースです。`, arr(cat.articles));
      // words to look for: 「円安」, 「円安のニュース」
      const words = [...q.matchAll(/[「『]([^」』]{1,20})[」』]/g)].map((x) => x[1]!).concat((q.match(/([^\s、。,「」]{2,12}?)(に関する|についての|の|って)(ニュース|記事|話題)/) ?? [])[1] ?? []).filter((w) => !/今日|最新|きょう|主な|凪|最近/.test(w));
      if (words.length) {
        const hit = all.filter((x) => words.some((w) => `${x.title} ${x.summary ?? ""}`.includes(w))).sort((p, r) => r.publishedAt.localeCompare(p.publishedAt));
        return hit.length ? show(`「${words.join("」「")}」のニュースです。`, hit) : `凪のいまのニュースに「${words.join("」「")}」は見つかりませんでした。${upd}`;
      }
      const top = cats.find((c) => c.id === "top") ?? cats[0]!;
      return show("凪のトップのニュースです。", arr(top.articles));
    },
  };
}

// ---------------------------------------------------------------- 野球速報
const NPB_TEAMS: string[][] = [["巨人", "ジャイアンツ", "読売"], ["阪神", "タイガース"], ["中日", "ドラゴンズ"], ["DeNA", "ベイスターズ", "横浜"], ["広島", "カープ"], ["ヤクルト", "スワローズ"],
  ["ソフトバンク", "ホークス"], ["日本ハム", "ファイターズ", "日ハム"], ["ロッテ", "マリーンズ"], ["楽天", "イーグルス"], ["西武", "ライオンズ"], ["オリックス", "バファローズ"]];
const MLB_Q = /MLB|メジャー|大リーグ|ドジャース|ヤンキース|カブス|エンゼルス|パドレス|メッツ|マリナーズ|大谷|山本由伸|ダルビッシュ|鈴木誠也|今永/;
const BB_Q = /野球|プロ野球|NPB|試合|順位|何位|首位|スコア|先発|セ・?リーグ|パ・?リーグ|日本シリーズ|クライマックス|交流戦/;
interface BBArgs { mode: "games" | "standings"; mlb: boolean; npb: boolean; teams: string[]; date: string; label: string }
function baseballTool(src: HibiSource): AITool<BBArgs & Record<string, unknown>, string> {
  return {
    name: "hibi_baseball",
    description: "野球速報から、NPB・MLB の試合の結果と速報、順位を出す",
    access: "read",
    whenToUse: [BB_Q, MLB_Q],
    answersWithoutRecords: true,
    parameters: { type: "object", properties: { mode: { type: "string" }, mlb: { type: "boolean" }, npb: { type: "boolean" }, teams: { type: "array", items: { type: "string" } }, date: { type: "string" }, label: { type: "string" } }, required: ["mode"] },
    route(question, ctx) {
      const q = question.normalize("NFKC");
      const teams = NPB_TEAMS.filter((al) => al.some((w) => q.includes(w))).map((al) => al[0]!);
      const mlb = MLB_Q.test(q), game = /試合|勝っ|負け|勝ち|勝敗|スコア|何対|何-何|結果|先発|速報|点|順位|何位|首位|どう(だった|なってる)/.test(q);
      if (!BB_Q.test(q) && !((teams.length || mlb) && game)) return undefined;
      if (/予定を|追加|入れて|日記/.test(q)) return undefined;
      const d = daysOf(q, ctx); if (d && d.from !== d.to) return undefined;
      return { mode: /順位|何位|首位|ゲーム差|勝率/.test(q) ? "standings" : "games", mlb, npb: !mlb || teams.length > 0 || /NPB|プロ野球|セ|パ/.test(q), teams, date: d ? d.from : today(ctx), label: d ? d.label : "今日" };
    },
    formatAnswer(result) { return result.ok ? result.value : `野球速報を取ってこられませんでした（${result.error}）。`; },
    async execute(a) {
      const base = src.appUrl("baseball").replace(/\/+$/, "");
      if (a.mode === "standings") {
        const out: string[] = [];
        const col = (h: string[], re: RegExp) => h.findIndex((x) => re.test(x));
        const table = (name: string, t: any) => {
          const h: string[] = arr(t?.headers), iR = col(h, /^順位$/), iT = col(h, /チーム|球団/), iW = col(h, /^(勝|勝利|勝数)$/), iL = col(h, /^(敗|敗戦|敗数)$/), iD = col(h, /^(引分|分)$/), iP = col(h, /勝率/), iG = col(h, /^(差|勝差|ゲーム差)$/);
          const v = (c: any[], i: number) => (i >= 0 ? String(c[i] ?? "").trim() : "");
          return `${name}\n${arr(t?.rows).map((r: any) => { const c = arr(r.cells); return `${v(c, iR) || "・"} ${v(c, iT) || r.team?.name || ""}　${v(c, iW)}勝${v(c, iL)}敗${v(c, iD) ? v(c, iD) + "分" : ""}${v(c, iP) ? ` 勝率${v(c, iP)}` : ""}${v(c, iG) && v(c, iG) !== "-" ? ` 差${v(c, iG)}` : ""}`; }).join("\n")}`;
        };
        if (a.npb) {
          const j = await src.fetchJSON(`${base}/api/npb/standings`);
          for (const [k, name] of [["C", "セ・リーグ"], ["P", "パ・リーグ"]] as const) { const t = arr(j?.[k]?.tables).find((x: any) => x.kind === "standings") ?? arr(j?.[k]?.tables)[0]; if (t) out.push(table(name, t)); }
        }
        if (a.mlb) {
          const j = await src.fetchJSON(`${base}/api/mlb/standings`);
          for (const lg of ["AL", "NL"]) for (const dv of arr(j?.[lg])) out.push(table(`${lg === "AL" ? "ア・リーグ" : "ナ・リーグ"} ${dv.name ?? ""}`, dv));
        }
        if (!out.length) throw new Error("順位表がありませんでした");
        const pick = a.teams.length ? out.filter((t) => a.teams.some((n) => t.includes(n))) : out;
        return `いまの順位です（野球速報）。\n\n${(pick.length ? pick : out).join("\n\n")}`;
      }
      const games: any[] = [];
      if (a.npb) games.push(...arr((await src.fetchJSON(`${base}/api/npb/schedule?date=${a.date}`))?.games));
      if (a.mlb) games.push(...arr((await src.fetchJSON(`${base}/api/mlb/schedule?date=${a.date}`))?.games));
      const al = NPB_TEAMS.filter((x) => a.teams.includes(x[0]!));
      const mine = al.length ? games.filter((g) => [g.home, g.away].some((t) => al.some((x) => x.some((w) => `${t?.name ?? ""}${t?.full ?? ""}`.includes(w))))) : games;
      if (!mine.length) return `${whenJa(a.label, a.date)}は、${al.length ? `${al.map((x) => x[0]).join("・")}の` : ""}試合はありません。`;
      const line = (g: any) => {
        const sc = g.away?.score != null && g.home?.score != null && g.status !== "pre" ? ` ${g.away.score} - ${g.home.score} ` : " - ";
        const st = g.status === "pre" ? `${g.startTime ? `${g.startTime}開始` : "試合前"}` : g.statusText || "";
        let res = "";
        if (al.length === 1 && g.status === "final") {
          const me = [g.home, g.away].find((t) => al[0]!.some((w) => `${t?.name ?? ""}${t?.full ?? ""}`.includes(w))), you = me === g.home ? g.away : g.home;
          if (me && you && me.score != null) res = me.score > you.score ? `\n　→ ${al[0]![0]}の勝ち` : me.score < you.score ? `\n　→ ${al[0]![0]}の負け` : "\n　→ 引き分け";
        }
        return `・${g.away?.name ?? "?"}${sc}${g.home?.name ?? "?"}（${st}${g.venue ? `・${g.venue}` : ""}）${arr(g.notes).length ? `\n　${arr(g.notes).join("　")}` : ""}${res}`;
      };
      return `${whenJa(a.label, a.date)}の試合です（野球速報）。\n${mine.map(line).join("\n")}`;
    },
  };
}

// ---------------------------------------------------------------- 天気 (Open-Meteo, as ひび手帳)
const WX_Q = /天気|気温|雨|傘|晴れ|曇|くもり|暑い|寒い|降る|降水|雪|風が/;
function weatherTool(src: HibiSource): AITool<{ date: string; label: string } & Record<string, unknown>, string> {
  return {
    name: "hibi_weather",
    description: "いまいる場所の天気・気温・雨の予報（Open-Meteo）",
    access: "read",
    whenToUse: [WX_Q],
    answersWithoutRecords: true,
    parameters: { type: "object", properties: { date: { type: "string" }, label: { type: "string" } }, required: ["date"] },
    route(question, ctx) {
      const q = question.normalize("NFKC");
      if (!WX_Q.test(q) || /日記|記録|だった|でした|曲|家計簿|予定を|追加/.test(q)) return undefined; // the weather of a day gone by: the diary
      const t = today(ctx), d = daysOf(q, ctx);
      if (d && (d.from !== d.to || d.from < t || d.from > addDays(t, 6))) return undefined;
      return { date: d ? d.from : t, label: d ? d.label : "今日" };
    },
    formatAnswer(result) { return result.ok ? result.value : `天気を取ってこられませんでした（${result.error}）。`; },
    async execute(a, ctx) {
      const at = await src.where(); if (!at) throw new Error("いまいる場所が分かりません（位置情報を許可するか、場所の記録を使ってください）");
      const j = await src.fetchJSON(`https://api.open-meteo.com/v1/forecast?latitude=${at.lat.toFixed(3)}&longitude=${at.lng.toFixed(3)}&current=temperature_2m,apparent_temperature,weather_code,wind_speed_10m&hourly=temperature_2m,precipitation_probability,weather_code&daily=weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max&timezone=auto&forecast_days=7`);
      const di = arr(j?.daily?.time).indexOf(a.date); if (di < 0) throw new Error("その日の予報はありません");
      const dl = j.daily, out: string[] = [];
      const where = at.name ? `${at.name}あたり` : "いまいるあたり";
      if (a.date === today(ctx) && j.current) out.push(`いま ${Math.round(j.current.temperature_2m)}°（体感 ${Math.round(j.current.apparent_temperature)}°）、${WX(j.current.weather_code)}。`);
      out.push(`${whenJa(a.label, a.date)}は${WX(dl.weather_code[di])}、最高 ${Math.round(dl.temperature_2m_max[di])}° / 最低 ${Math.round(dl.temperature_2m_min[di])}°、降水確率 ${dl.precipitation_probability_max[di] ?? "?"}%。`);
      const hours = arr(j.hourly?.time).map((t: string, i: number) => ({ t, p: +j.hourly.precipitation_probability[i] || 0, c: j.hourly.weather_code[i] })).filter((h: any) => h.t.startsWith(a.date) && (a.date !== today(ctx) || +h.t.slice(11, 13) >= new Date(ctx.now).getHours()));
      const wet = hours.filter((h: any) => h.p >= 50);
      if (wet.length) out.push(`${+wet[0]!.t.slice(11, 13)}時ごろから雨の可能性が高くなります（${Math.max(...wet.map((h: any) => h.p))}%）。傘があると安心です。`);
      else if ((dl.precipitation_probability_max[di] ?? 0) >= 30) out.push("にわか雨があるかもしれません。折りたたみ傘があると安心です。");
      return `${where}の天気です。\n${out.join("\n")}`;
    },
  };
}

// ---------------------------------------------------------------- 家計簿 (exact sums, no model)
const MONEY_Q = /円|いくら|幾ら|金額|使っ|つかっ|支出|出費|収入|家計簿|お金|稼い|もらった/;
interface MoneyArgs { from?: string; to?: string; label: string; kind: "out" | "in"; cats: string[]; catNames: string[]; words: string[] }
const lastDay = (y: number, m: number) => new Date(y, m, 0).getDate();
/** 「4月から今日まで」「この4月から」「4月から9月まで」「先月」「今年」… → a range; none = every record */
function moneyRange(q: string, t: string): { from?: string; to?: string; label: string } {
  return periodFirst(q, (x) => { const r = moneyRangeOne(x, t); return r.from || r.to ? r : undefined; }) ?? { label: "これまで" };
}
function moneyRangeOne(q: string, t: string): { from?: string; to?: string; label: string } {
  const ty = +t.slice(0, 4), tm = +t.slice(5, 7);
  const ymOf = (m: number) => (m <= tm ? ty : ty - 1); // a month without a year: the latest one so far
  const two = q.match(/(\d{1,2})月(?:(\d{1,2})日)?から(\d{1,2})月(?:(\d{1,2})日)?(まで)?/);
  if (two) {
    const y1 = ymOf(+two[1]!), m2 = +two[3]!, y2 = m2 >= +two[1]! ? y1 : y1 + 1;
    const from = `${y1}-${pad(+two[1]!)}-${pad(two[2] ? +two[2] : 1)}`, to = `${y2}-${pad(m2)}-${pad(two[4] ? +two[4] : lastDay(y2, m2))}`;
    return { from, to, label: `${dayJa(from)}〜${dayJa(to)}` };
  }
  const since = q.match(/(\d{1,2})月(?:(\d{1,2})日)?(から|以降)/);
  if (since) { const from = `${ymOf(+since[1]!)}-${pad(+since[1]!)}-${pad(since[2] ? +since[2] : 1)}`; return { from, to: t, label: `${dayJa(from)}〜今日（${dayJa(t)}）` }; }
  const intent = parseDateIntent(q, t);
  if (intent) return { from: intent.from, to: intent.to > t ? t : intent.to, label: intent.label };
  return { label: "これまで" };
}
function moneyRoute(src: HibiSource, question: string, ctx: ToolContext): MoneyArgs | undefined {
  const q = question.normalize("NFKC"), m = src.money();
  if (!m.entries.length) return undefined;
  const kind: "out" | "in" = /収入|入金|稼い|もらった|給料|仕送り/.test(q) ? "in" : "out";
  // categories named in the question: 「外食費」, or 「外食」 for 外食費
  const named = m.cats[kind].filter((c) => q.includes(c.name) || (c.name.length >= 3 && /[費代]$/.test(c.name) && q.includes(c.name.slice(0, -1))));
  // 「外食費」 also contains 「食費」: count a name only where it is not part of a longer one that was named
  const cats = named.filter((c) => {
    let rest = q; for (const d of named) if (d !== c && d.name.length > c.name.length && d.name.includes(c.name)) rest = rest.split(d.name).join(" ");
    return rest.includes(c.name) || (c.name.length >= 3 && /[費代]$/.test(c.name) && rest.includes(c.name.slice(0, -1)));
  });
  // words of the question that are written in memos: 「なか卯」, or anything in 「」
  const quoted = [...q.matchAll(/[「『"]([^」』"]{1,30})[」』"]/g)].map((x) => x[1]!.trim()).filter(Boolean);
  const memoWords = new Set<string>();
  for (const e of m.entries) for (const w of [e.memo.trim(), ...e.memo.split(/[\s、,・/／()（）]+/)]) if (w.length >= 2 && q.includes(w) && !cats.some((c) => c.name.includes(w))) memoWords.add(w);
  const words = [...new Set([...quoted.filter((w) => !cats.some((c) => c.name === w)), ...memoWords])].filter((w, _i, all) => !all.some((o) => o !== w && o.includes(w))); // keep the longest
  if (!MONEY_Q.test(q) && !cats.length && !words.length) return undefined;
  if (!MONEY_Q.test(q) && !/何回|何件|いくつ|合計|どれくらい|どのくらい/.test(q)) return undefined;
  const r = moneyRange(q, today(ctx));
  return { ...(r.from ? { from: r.from } : {}), ...(r.to ? { to: r.to } : {}), label: r.label, kind, cats: cats.map((c) => c.id), catNames: cats.map((c) => c.name), words };
}
function moneyTool(src: HibiSource): AITool<MoneyArgs & Record<string, unknown>, { total: number; n: number; list: MoneyEntry[] }> {
  return {
    name: "hibi_money",
    description: "ひび手帳の家計簿から、期間・カテゴリー・メモの言葉で絞って、支出（または収入）の合計と件数を出す",
    access: "read",
    whenToUse: [MONEY_Q, /外食|食費|何回/],
    answersWithoutRecords: true,
    parameters: {
      type: "object",
      properties: {
        from: { type: "string", description: "YYYY-MM-DD（任意）" }, to: { type: "string", description: "YYYY-MM-DD（任意）" },
        label: { type: "string", description: "期間の言い方" }, kind: { type: "string", enum: ["out", "in"], description: "out=支出, in=収入" },
        cats: { type: "array", items: { type: "string" }, description: "カテゴリーの id" }, catNames: { type: "array", items: { type: "string" }, description: "カテゴリーの名前" },
        words: { type: "array", items: { type: "string" }, description: "メモに含まれる言葉" },
      },
      required: ["kind"],
    },
    route(question, ctx) {
      return moneyRoute(src, question, ctx) as (MoneyArgs & Record<string, unknown>) | undefined;
    },
    formatAnswer(result, args) {
      if (!result.ok) return `家計簿を読めませんでした（${result.error}）。`;
      const cats = args.catNames?.length ? args.catNames.join("・") : "", words = args.words?.length ? `メモに「${args.words.join("」「")}」とある` : "";
      // 「2026年10月の食費の支出」「4月1日〜今日、外食費のうち、メモに「なか卯」とある支出」
      const head = `${args.label}${cats || words ? "、" : "の"}${cats ? `${cats}${words ? "のうち、" : "の"}` : ""}${words}${args.kind === "in" ? "収入" : "支出"}`;
      const { total, n, list } = result.value;
      if (!n) return `${head}は、家計簿に見つかりませんでした。`;
      const lines = list.slice(0, 8).map((e) => `・${+e.date.slice(5, 7)}/${+e.date.slice(8, 10)} ${e.memo || "（メモなし）"} ${e.amt.toLocaleString()}円`);
      return `${head}は、${n}件・合計 ${total.toLocaleString()}円 でした。\n${lines.join("\n")}${n > 8 ? `\n…ほか${n - 8}件` : ""}`;
    },
    async execute(args) {
      if (!src.ready()) throw new Error("ひび手帳のデータがまだ読み込まれていません");
      const cats = new Set(args.cats ?? []), words = args.words ?? [];
      const list = src.money().entries
        .filter((e) => e.k === args.kind && (!args.from || e.date >= args.from) && (!args.to || e.date <= args.to) && (!cats.size || cats.has(e.cat)) && (!words.length || words.some((w) => e.memo.includes(w))))
        .sort((a, b) => b.date.localeCompare(a.date));
      return { total: list.reduce((t, e) => t + e.amt, 0), n: list.length, list };
    },
  };
}

// ---------------------------------------------------------------- 習慣（灯）and 出席 (exact counts, no model; same rules as ひび手帳)
/** a value of a habit as people read it: time habits are kept in seconds */
function hbAmount(h: any, v: number): string {
  if (h.kind === "time") { const m = Math.round(v / 60); return m >= 60 ? `${Math.floor(m / 60)}時間${m % 60 ? `${m % 60}分` : ""}` : m >= 1 ? `${m}分` : `${Math.round(v)}秒`; }
  return `${Math.round(v * 10) / 10}${h.unit ?? (h.kind === "count" ? "回" : "")}`;
}
function habitCalc(src: HibiSource, h: any, t: string) {
  const log = obj(obj(src.more().habits).log), goal = +h.goal || 1, f = obj(h.freq), created = typeof h.c === "string" ? h.c : "0000-00-00";
  const val = (d: string) => +(obj(obj(log[d])[h.id]).v ?? 0) || 0;
  const done = (d: string) => val(d) >= goal, rest = (d: string) => !!obj(obj(log[d])[h.id]).skip && !done(d);
  const wdOf = (d: string) => new Date(+d.slice(0, 4), +d.slice(5, 7) - 1, +d.slice(8, 10)).getDay();
  const sched = (d: string) => d >= created && (f.t !== "days" || arr(f.days).includes(wdOf(d)));
  const sun = src.weekStartsSunday(), weekStart = (d: string) => addDays(d, -(sun ? wdOf(d) : (wdOf(d) + 6) % 7));
  const weekCnt = (ws: string) => { let n = 0; for (let i = 0; i < 7; i++) if (done(addDays(ws, i))) n++; return n; };
  const weekly = f.t === "week", need = Math.max(1, +f.n || 3);
  let streak = 0, best = 0;
  if (weekly) {
    let ws = weekStart(t); if (weekCnt(ws) >= need) streak++;
    for (ws = addDays(ws, -7); addDays(ws, 6) >= created && weekCnt(ws) >= need; ws = addDays(ws, -7)) streak++;
    let cur = 0; for (let w = weekStart(created < "2000" ? addDays(t, -730) : created); w <= t; w = addDays(w, 7)) { if (weekCnt(w) >= need) best = Math.max(best, ++cur); else if (addDays(w, 6) < t) cur = 0; }
  } else {
    let d = done(t) ? t : addDays(t, -1);
    for (let i = 0; i < 4000 && d >= created; i++, d = addDays(d, -1)) { if (!sched(d) || rest(d)) continue; if (done(d)) streak++; else break; }
    let cur = 0; for (let d2 = created < "2000" ? addDays(t, -730) : created, i = 0; d2 <= t && i < 4000; d2 = addDays(d2, 1), i++) { if (!sched(d2) || rest(d2)) continue; if (done(d2)) best = Math.max(best, ++cur); else if (d2 < t) cur = 0; }
  }
  const span = (from: string, to: string) => {
    let n = 0, planned = 0, skipped = 0, total = 0;
    for (let d = from < created ? created : from, i = 0; d <= to && i < 4000; d = addDays(d, 1), i++) { total += val(d); if (done(d)) n++; if (rest(d)) skipped++; else if (sched(d)) planned++; }
    return { n, planned, skipped, total };
  };
  const last = Object.keys(log).filter((d) => /^\d{4}-\d{2}-\d{2}$/.test(d) && d <= t && done(d)).sort().pop();
  return { weekly, need, streak, best, span, last, thisWeek: weekCnt(weekStart(t)), weekStart: weekStart(t) };
}
function habitTool(src: HibiSource): AITool<{ ids: string[]; from?: string; to?: string; label: string } & Record<string, unknown>, string> {
  return {
    name: "hibi_habits",
    description: "ひび手帳の習慣（灯）が、期間の中で何日できたか・連続・最長・最後にできた日を数える",
    access: "read",
    whenToUse: [/習慣|灯|続い|連続/],
    answersWithoutRecords: true,
    parameters: { type: "object", properties: { ids: { type: "array", items: { type: "string" } }, from: { type: "string" }, to: { type: "string" }, label: { type: "string" } }, required: ["ids"] },
    route(question, ctx) {
      if (!src.ready()) return undefined;
      const q = question.normalize("NFKC"), list = arr(obj(src.more().habits).list).filter((h) => h && h.name);
      if (!list.length || /予定|スケジュール|追加|入れて|登録/.test(q)) return undefined;
      const named = list.filter((h) => q.includes(String(h.name)));
      if (!named.length && !/習慣|灯/.test(q)) return undefined;
      if (named.length && !/習慣|灯|何回|何日|回数|日数|続い|連続|できた|できて|やった|やって|達成|記録|サボ|さぼ|最後|いつ|どれくらい|どのくらい|ペース/.test(q)) return undefined;
      const r = moneyRange(q, today(ctx));
      return { ids: (named.length ? named : list.filter((h) => !h.arch)).map((h) => String(h.id)), ...(r.from ? { from: r.from } : {}), ...(r.to ? { to: r.to } : {}), label: r.label };
    },
    formatAnswer(result) { return result.ok ? result.value : `習慣の記録を読めませんでした（${result.error}）。`; },
    async execute(args, ctx) {
      if (!src.ready()) throw new Error("ひび手帳のデータがまだ読み込まれていません");
      const t = today(ctx), list = arr(obj(src.more().habits).list).filter((h) => args.ids.includes(String(h.id)));
      if (!list.length) return "習慣（灯）はまだ登録されていません。";
      return list.map((h) => {
        const c = habitCalc(src, h, t), f = obj(h.freq);
        const fq = f.t === "days" ? `${arr(f.days).map((w: number) => WD_[w]).join("・")}曜日` : c.weekly ? `週${c.need}回` : "毎日";
        const lines: string[] = [];
        if (args.from || args.to) {
          const s = c.span(args.from ?? "0000-00-00", args.to ?? t);
          const thisWeek = c.weekly && args.from === c.weekStart ? (s.n >= c.need ? "（今週の目標達成）" : `（あと${c.need - s.n}回で今週の目標）`) : "";
          lines.push(`${args.label}：${s.n}日できた${thisWeek}${!c.weekly && s.planned ? `（予定の${s.planned}日のうち）` : ""}${s.skipped ? `・休み${s.skipped}日` : ""}${h.kind !== "check" && s.total ? `・合計 ${hbAmount(h, s.total)}` : ""}`);
        } else {
          if (c.weekly) lines.push(`今週：${c.thisWeek}回${c.thisWeek >= c.need ? "（目標達成）" : `（あと${c.need - c.thisWeek}回で目標）`}`);
          const m = c.span(addDays(t, -6), t), mm = c.span(addDays(t, -29), t);
          lines.push(`直近7日：${m.n}日・直近30日：${mm.n}日${h.kind !== "check" && mm.total ? `（30日の合計 ${hbAmount(h, mm.total)}）` : ""}`);
        }
        lines.push(`連続：${c.streak}${c.weekly ? "週" : "日"}（いちばん長い連続：${c.best}${c.weekly ? "週" : "日"}）`);
        lines.push(c.last ? `最後にできた日：${dayJa(c.last)}` : "まだできた日はありません");
        return `${h.emoji ?? ""}${h.name}（${fq}${h.kind !== "check" ? `・目標 ${hbAmount(h, +h.goal || 1)}` : ""}${h.arch ? "・アーカイブ" : ""}）\n${lines.map((l) => `・${l}`).join("\n")}`;
      }).join("\n\n");
    },
  };
}
function attendanceTool(src: HibiSource): AITool<{ ids: string[]; from?: string; to?: string; label: string } & Record<string, unknown>, string> {
  return {
    name: "hibi_attendance",
    description: "ひび手帳の時間割の出席・遅刻・欠席・休講を数え、あと何回休めるかを出す",
    access: "read",
    whenToUse: [/出席|欠席|遅刻|休める|休講|単位/],
    answersWithoutRecords: true,
    parameters: { type: "object", properties: { ids: { type: "array", items: { type: "string" } }, from: { type: "string" }, to: { type: "string" }, label: { type: "string" } }, required: ["ids"] },
    route(question, ctx) {
      if (!src.ready()) return undefined;
      const q = question.normalize("NFKC"), classes = arr(src.school().classes).filter((c) => c && c.name);
      if (!classes.length || /予定|スケジュール|追加|入れて|登録/.test(q)) return undefined;
      const ask = /出席|欠席|遅刻|休め|休んだ|休講|単位|サボ|さぼ|出た|出てる/.test(q);
      const named = classes.filter((c) => q.includes(String(c.name)) || (String(c.name).replace(/[（(].*$/, "").length >= 2 && q.includes(String(c.name).replace(/[（(].*$/, ""))));
      if (!ask) return undefined;
      const r = moneyRange(q, today(ctx));
      return { ids: (named.length ? named : classes).map((c) => String(c.id)), ...(r.from ? { from: r.from } : {}), ...(r.to ? { to: r.to } : {}), label: r.label };
    },
    formatAnswer(result) { return result.ok ? result.value : `出席の記録を読めませんでした（${result.error}）。`; },
    async execute(args) {
      if (!src.ready()) throw new Error("ひび手帳のデータがまだ読み込まれていません");
      const sc = src.school(), rule = +sc.lateRule || 0;
      const list = arr(sc.classes).filter((c) => args.ids.includes(String(c.id)));
      const out = list.map((c) => {
        const all = Object.entries(obj(c.att)) as Array<[string, string]>;
        const inR = (k: string) => (!args.from || k.slice(0, 10) >= args.from) && (!args.to || k.slice(0, 10) <= args.to);
        const cnt = (xs: Array<[string, string]>) => { const n = (s: string) => xs.filter(([, v]) => v === s).length; return { present: n("present"), late: n("late"), absent: n("absent"), cancel: n("cancel") }; };
        const a = cnt(all), used = a.absent + (rule ? Math.floor(a.late / rule) : 0), max = +c.maxAbsent || 0, left = max - used;
        const when = arr(c.slots).map((sl: any) => `${WD_[sl.w]}曜${+sl.p + 1}限`).join("・");
        const lines: string[] = [];
        if (args.from || args.to) { const p = cnt(all.filter(([k]) => inR(k))); lines.push(`${args.label}：出席${p.present}・遅刻${p.late}・欠席${p.absent}・休講${p.cancel}`); }
        lines.push(`${args.from || args.to ? "これまで全部" : "これまで"}：出席${a.present}・遅刻${a.late}・欠席${a.absent}・休講${a.cancel}`);
        const held = a.present + a.late + a.absent;
        if (held) lines.push(`出席率：${Math.round(((a.present + a.late) / held) * 100)}%（遅刻も出席として）`);
        lines.push(`欠席として数える回数：${used}回${rule ? `（遅刻${rule}回で欠席1回）` : ""}`);
        lines.push(!max ? "休める回数の上限は、まだ決めていません（ひび手帳の授業の画面で決められます）" : left > 1 ? `あと${left}回休めます（上限${max}回）` : left === 1 ? `あと1回だけ休めます（上限${max}回）` : left === 0 ? `もう休めません（上限${max}回）` : `上限（${max}回）を${-left}回こえています`);
        return `${c.name}${c.room ? `（${c.room}）` : ""}${when ? ` ${when}` : ""}\n${lines.map((l) => `・${l}`).join("\n")}`;
      });
      return out.length ? out.join("\n\n") : "時間割に授業がありません。";
    },
  };
}

// ---------------------------------------------------------------- records for everything else (RAG)
function match(rec: MemoryRecord, words: string[]): boolean {
  if (!words.length) return true;
  const hay = `${rec.title ?? ""} ${rec.content} ${(rec.tags ?? []).join(" ")}`;
  return words.some((w) => hay.includes(w));
}
const arr = (x: unknown): any[] => (Array.isArray(x) ? x : []);
const obj = (x: unknown): Record<string, any> => (x && typeof x === "object" && !Array.isArray(x) ? (x as Record<string, any>) : {});
const yen = (n: unknown) => `${Math.round(+(n as number) || 0).toLocaleString()}円`;
const ymdOf = (v: unknown): string | undefined => {
  if (typeof v === "string" && /^\d{4}-\d{2}-\d{2}/.test(v)) return v.slice(0, 10);
  if (typeof v === "number" && v > 1e11) { const d = new Date(v); return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`; }
  return undefined;
};
/** The words of something whose exact shape is not known (メモ・あとで・凪の記事…): every piece of text in it, without ids, colors and links' noise. */
function textOf(x: unknown, depth = 0): string {
  if (depth > 5 || x == null) return "";
  // leave out ids, colors, links, timestamps and short markers such as a block's type ("p", "h2", "todo")
  if (typeof x === "string") return /^(#[0-9a-f]{3,8}|[a-z0-9_-]{8,48}|[a-z][a-z0-9_-]{0,7}|https?:\/\/\S+|\d{4}-\d{2}-\d{2}T[\d:.]+Z?)$/i.test(x.trim()) ? "" : x.trim();
  if (Array.isArray(x)) return x.map((y) => textOf(y, depth + 1)).filter(Boolean).join("\n");
  if (typeof x === "object") return Object.entries(x as Record<string, unknown>).filter(([k]) => !/^(id|color|icon|emoji|sample|ts|u)$/.test(k)).map(([, v]) => textOf(v, depth + 1)).filter(Boolean).join(" ");
  return "";
}
const firstDate = (o: Record<string, any>) => ymdOf(o.date) ?? ymdOf(o.updatedAt) ?? ymdOf(o.updated) ?? ymdOf(o.savedAt) ?? ymdOf(o.createdAt) ?? ymdOf(o.created) ?? ymdOf(o.ts) ?? ymdOf(o.at);
const WX = (c?: number) => (c == null ? "" : c === 0 ? "晴れ" : c <= 2 ? "晴れ時々くもり" : c === 3 ? "くもり" : c <= 48 ? "霧" : c <= 67 || (c >= 80 && c <= 82) ? "雨" : c <= 77 || c === 85 || c === 86 ? "雪" : "雷雨");
const distM = (a: { lat: number; lng: number }, b: { lat: number; lng: number }) => {
  const R = 6371000, r = Math.PI / 180, dLa = (b.lat - a.lat) * r, dLo = (b.lng - a.lng) * r;
  const h = Math.sin(dLa / 2) ** 2 + Math.cos(a.lat * r) * Math.cos(b.lat * r) * Math.sin(dLo / 2) ** 2;
  return 2 * R * Math.asin(Math.sqrt(h));
};
const WD_ = ["日", "月", "火", "水", "木", "金", "土"];

function records(src: HibiSource, q: MemoryQuery, now: Date): MemoryRecord[] {
  if (!src.ready()) return [];
  const created = now.toISOString(), out: MemoryRecord[] = [];
  const add = (r: Omit<MemoryRecord, "appId" | "createdAt">) => { if (r.content.trim()) out.push({ ...r, appId: APP_ID, createdAt: created }); };
  const want = (t: string) => !q.types?.length || q.types.includes(t);
  const words = (q.text ?? "").split(/\s+/).filter(Boolean);
  const r = q.dateRange, from = r?.from?.slice(0, 10), to = r?.to?.slice(0, 10);
  const inR = (d?: string) => !r || (!!d && (!from || d >= from) && (!to || d <= to));
  // the days of the range (at most 62), for things worked out day by day
  const days: string[] = [];
  if (from || to) for (let d = from ?? to!, n = 0; d <= (to ?? from!) && n < 62; d = addDays(d, 1), n++) days.push(d);
  const more = src.more(), spots = arr(more.spots).filter((x) => x && x.name && Number.isFinite(+x.lat) && Number.isFinite(+x.lng));
  const spotName = (id?: string) => spots.find((x) => String(x.id) === id)?.name;

  // 日記
  if (want("diary"))
    for (const e of src.entries()) {
      if (!inR(e.date)) continue;
      add({ id: `diary:${e.id}`, type: "diary", title: `${dayJa(e.date)}の日記`, occurredAt: e.date,
        content: [e.feelings?.length ? `気持ち：${e.feelings.join("・")}` : "", e.text ?? "", e.weather ? `天気：${WX(e.weather.code)} ${e.weather.temp ?? ""}°` : "",
          e.place && spotName(e.place) ? `場所：${spotName(e.place)}` : "", e.song ? `曲：${e.song.title}${e.song.artist ? `（${e.song.artist}）` : ""}` : ""].filter(Boolean).join("\n"),
        ...(e.tags?.length ? { tags: e.tags } : {}) });
    }
  // 予定（期間があれば、繰り返し・授業も含めて1日ずつ）
  if (want("event")) {
    if (days.length) {
      for (const d of days) {
        const x = src.day(d);
        x.allDay.forEach((t, i) => add({ id: `ev:${d}:a${i}`, type: "event", title: t, content: `${dayJa(d)} 終日 ${t}`, occurredAt: d }));
        x.items.forEach((i, k) => add({ id: `ev:${d}:${k}`, type: "event", title: i.title, content: `${dayJa(d)} ${itemLine(i)}${i.memo && i.kind !== "class" ? `\nメモ：${i.memo}` : ""}`, occurredAt: d }));
      }
    } else {
      for (const e of src.events())
        add({ id: `ev:${e.id}`, type: "event", title: e.title, occurredAt: e.date,
          content: `${dayJa(e.date)}${e.allDay ? " 終日" : e.start ? ` ${hm(e.start)}〜${hm(e.end ?? "")}` : ""} ${e.title}${e.repeat && e.repeat !== "none" ? "（繰り返し）" : ""}${e.memo ? `\nメモ：${e.memo}` : ""}` });
    }
  }
  // やること・課題・あとで
  if (want("task")) {
    src.todos().forEach((x, i) => add({ id: `todo:${i}:${x.title}`, type: "task", title: x.title, ...(x.due ? { occurredAt: x.due } : {}),
      content: `${x.className ? `［${x.className}］` : ""}${x.title}${x.due ? `（締め切り ${dayJa(x.due)}${x.dueTime ? " " + hm(x.dueTime) : ""}）` : "（期限なし）"}${x.memo ? `\nメモ：${x.memo}` : ""}` }));
    for (const t of arr(more.tasks)) {
      if (!t || !t.title) continue; const d = ymdOf(t.date);
      if (r && d && !inR(d)) continue;
      add({ id: `later:${t.id ?? t.title}`, type: "task", title: String(t.title), ...(d ? { occurredAt: d } : {}),
        content: `あとで：${t.title}${t.done ? "（完了）" : ""}${t.dur ? `（所要${dur(+t.dur)}）` : ""}${d ? `（${dayJa(d)}${t.start ? " " + hm(String(t.start)) : ""}に予定）` : ""}${arr(t.subs ?? t.subtasks).length ? `\nサブタスク：${textOf(t.subs ?? t.subtasks)}` : ""}${t.memo ? `\nメモ：${t.memo}` : ""}` });
    }
  }
  // メモ（ひび手帳のメモ・Asterism）
  if (want("memo")) {
    for (const [i, n] of arr(more.notes).entries()) { const d = firstDate(obj(n)); if (r && d && !inR(d)) continue; add({ id: `note:${n?.id ?? i}`, type: "memo", title: String(n?.title || "メモ"), ...(d ? { occurredAt: d } : {}), content: `ひび手帳のメモ\n${textOf(n)}` }); }
    for (const [i, n] of arr(more.asterism).entries()) { const d = firstDate(obj(n)); if (r && d && !inR(d)) continue; add({ id: `ast:${n?.id ?? i}`, type: "memo", title: String(n?.title || "Asterism のメモ"), ...(d ? { occurredAt: d } : {}), content: `Asterism のメモ\n${textOf(n)}` }); }
  }
  if (!q.types?.length) {
    // 凪（保存した記事）
    for (const [i, a] of arr(more.nagi).entries()) { const d = firstDate(obj(a)); if (r && d && !inR(d)) continue; add({ id: `nagi:${a?.id ?? i}`, type: "news", title: String(a?.title || "保存した記事"), ...(d ? { occurredAt: d } : {}), content: `凪で保存した記事\n${textOf(a)}` }); }
    // 習慣（灯）
    const hb = obj(more.habits), list = arr(hb.list).filter((h) => h && h.name), log = obj(hb.log);
    const doneOn = (h: any, d: string) => { const v = obj(obj(log[d])[h.id]); return !v.skip && +(v.v ?? 0) >= (+h.goal || 1); };
    const tday = dayOf(now, TZ);
    for (const h of r ? [] : list) { // the summary of each habit; a question about some days gets those days' log below
      const last = Object.keys(log).filter((d) => doneOn(h, d)).sort().pop();
      const cnt = (n: number) => Array.from({ length: n }, (_, i) => addDays(tday, -i)).filter((d) => doneOn(h, d)).length;
      const freq = obj(h.freq), fq = freq.t === "days" ? `${arr(freq.days).map((w: number) => WD_[w]).join("・")}曜日` : freq.t === "week" ? `週${freq.n}回` : "毎日";
      add({ id: `habit:${h.id}`, type: "habit", title: `習慣：${h.name}`, occurredAt: last ?? tday,
        content: `習慣（灯）：${h.emoji ?? ""}${h.name}（${fq}${h.kind !== "check" ? `・目標 ${hbAmount(h, +h.goal)}` : ""}${h.arch ? "・アーカイブ" : ""}）\n直近7日で${cnt(7)}日・30日で${cnt(30)}日 できた${last ? `。最後にできた日：${dayJa(last)}` : ""}` });
    }
    for (const d of days) {
      const o = obj(log[d]), parts = list.map((h) => { const v = obj(o[h.id]); return v.skip ? `${h.name}：休み` : +(v.v ?? 0) > 0 ? `${h.name}：${doneOn(h, d) ? "できた" : hbAmount(h, +v.v)}` : ""; }).filter(Boolean);
      if (parts.length) add({ id: `habit:${d}`, type: "habit", title: `${dayJa(d)}の習慣`, occurredAt: d, content: `${dayJa(d)}の習慣（灯）\n${parts.join("\n")}` });
    }
    // 授業（時間割・出席）
    const sc = src.school(), periods = arr(sc.periods);
    const ATT: Record<string, string> = { present: "出席", late: "遅刻", absent: "欠席", cancel: "休講" };
    for (const c of arr(sc.classes)) {
      if (!c || !c.name) continue;
      const att = Object.values(obj(c.att)) as string[], n = (k: string) => att.filter((x) => x === k).length;
      const used = n("absent") + (+sc.lateRule ? Math.floor(n("late") / +sc.lateRule) : 0), left = +c.maxAbsent ? +c.maxAbsent - used : undefined;
      add({ id: `class:${c.id}`, type: "class", title: `授業：${c.name}`,
        content: `授業：${c.name}${c.room ? `（${c.room}）` : ""}${c.teacher ? ` ${c.teacher}` : ""}\n${arr(c.slots).map((sl: any) => `${WD_[sl.w]}曜${+sl.p + 1}限${periods[sl.p] ? `（${periods[sl.p][0]}〜${periods[sl.p][1]}）` : ""}`).join("・")}\n` +
          `出席${n("present")}・遅刻${n("late")}・欠席${n("absent")}・休講${n("cancel")}${left !== undefined ? `（あと${left}回休める）` : ""}` });
      for (const [k, v] of Object.entries(obj(c.att))) { const d = k.slice(0, 10); if (days.length && inR(d) && ATT[v as string]) add({ id: `att:${c.id}:${k}`, type: "class", title: `${c.name}の出席`, occurredAt: d, content: `${dayJa(d)} ${c.name}（${+k.split("#")[1]! + 1}限）：${ATT[v as string]}` }); }
    }
    // バイト先
    for (const j of arr(more.jobs)) if (j && j.name) add({ id: `job:${j.id ?? j.name}`, type: "job", title: `バイト先：${j.name}`,
      content: `バイト先：${j.name}${j.wage ? ` 時給${yen(j.wage)}` : ""}${j.transport ? ` 交通費${yen(j.transport)}` : ""}${j.night ? "・深夜割増あり" : ""}${j.closing ? `・締め日 ${j.closing}日` : ""}` });
    // 家計簿（予算・固定費・貯金・記録）
    const money = src.money(), mm = obj(more.money), catName = (id: string) => [...money.cats.out, ...money.cats.in].find((c) => c.id === id)?.name ?? "";
    const accts = arr(mm.accts).filter((a) => a && a.name);
    if (accts.length) add({ id: "money:accts", type: "money", title: "貯金（お金の置き場所）", content: `貯金（お金の置き場所ごとの残高）\n${accts.map((a) => `・${a.name}：${yen(a.bal)}${ymdOf(a.ts) ? `（${dayJa(ymdOf(a.ts)!)}時点）` : ""}`).join("\n")}\n合計：${yen(accts.reduce((t, a) => t + (+a.bal || 0), 0))}` });
    const bud = obj(mm.budget), bc = Object.entries(obj(bud.cats)).filter(([, v]) => +(v as number) > 0);
    if (+bud.total || bc.length || arr(bud.groups).length) add({ id: "money:budget", type: "money", title: "家計簿の予算",
      content: `家計簿の予算（1か月）${+bud.total ? `\n全体：${yen(bud.total)}` : ""}${bc.map(([id, v]) => `\n${catName(id) || "カテゴリー"}：${yen(v)}`).join("")}${arr(bud.groups).map((g: any) => `\n${g.name}（${arr(g.cats).map(catName).filter(Boolean).join("＋")}）：${yen(g.amt)}`).join("")}` });
    for (const [i, f] of arr(mm.fixed).entries()) if (f) add({ id: `money:fixed:${f.id ?? i}`, type: "money", title: `固定費：${f.name ?? f.memo ?? catName(f.cat)}`, content: `毎月の${f.k === "in" ? "定期収入" : "固定費"}：${f.name ?? f.memo ?? catName(f.cat)} ${yen(f.amt)}${f.day ? `（毎月${f.day}日）` : ""}` });
    if (days.length) for (const [i, e] of money.entries.entries()) if (inR(e.date)) add({ id: `money:${i}`, type: "money", title: `${e.k === "in" ? "収入" : "支出"} ${catName(e.cat)}`, occurredAt: e.date, content: `${dayJa(e.date)} ${e.k === "in" ? "収入" : "支出"} ${catName(e.cat)} ${yen(e.amt)}${e.memo ? ` ${e.memo}` : ""}` });
    // 場所（行った場所・名前をつけた場所）
    if (spots.length) add({ id: "place:spots", type: "place", title: "名前をつけた場所", content: `名前をつけた場所：${spots.map((x) => x.name).join("、")}` });
    if (days.length && spots.length) {
      const pts = arr(more.places).filter((p) => p && +p.t > 0 && Number.isFinite(+p.lat));
      for (const d of days) {
        const seen = new Map<string, [number, number]>();
        for (const p of pts) {
          if (ymdOf(+p.t) !== d) continue;
          const near = spots.map((x) => [x, distM({ lat: +p.lat, lng: +p.lng }, { lat: +x.lat, lng: +x.lng })] as const).sort((a, b) => a[1] - b[1])[0];
          if (!near || near[1] > 200) continue;
          const tt = new Date(+p.t), m = tt.getHours() * 60 + tt.getMinutes(), w = seen.get(near[0].name);
          seen.set(near[0].name, w ? [Math.min(w[0], m), Math.max(w[1], m)] : [m, m]);
        }
        if (seen.size) add({ id: `place:${d}`, type: "place", title: `${dayJa(d)}に行った場所`, occurredAt: d, content: `${dayJa(d)}に行った場所：${[...seen].sort((a, b) => a[1][0] - b[1][0]).map(([n, [a, b]]) => `${n}（${hhmm(a)}${b > a ? `〜${hhmm(b)}` : ""}）`).join("、")}` });
      }
    }
    // テンプレート・気持ち・タグ
    for (const [i, t] of arr(more.templates).entries()) if (t && t.title) add({ id: `tpl:${t.id ?? i}`, type: "template", title: `テンプレート：${t.title}`, content: `予定のテンプレート：${t.title}${t.start ? ` ${hm(String(t.start))}〜${hm(String(t.end ?? ""))}` : ""}${t.memo ? `\nメモ：${t.memo}` : ""}` });
    const cf = arr(more.customFeelings).map((x) => (typeof x === "string" ? x : x?.name)).filter(Boolean), ct = arr(more.customTags).map((x) => (typeof x === "string" ? x : x?.name)).filter(Boolean);
    if (cf.length || ct.length) add({ id: "diary:custom", type: "diary", title: "自分で足した気持ち・タグ", content: `自分で足した気持ち：${cf.join("、") || "なし"}\n自分で足したタグ：${ct.join("、") || "なし"}` });
  }
  // a question about some days: things without a day (バイト先, 貯金, 授業のまとめ…) only when the question names them
  const scoped = r ? out.filter((x) => (x.occurredAt ? inR(x.occurredAt.slice(0, 10)) : words.length > 0 && match(x, words))) : out;
  const hit = scoped.filter((x) => match(x, words));
  // no word matched: the newest of the main kinds rather than nothing (the day-by-day ones only when a range was asked)
  const pool = hit.length || !words.length ? hit : scoped.filter((x) => ["diary", "event", "task", "memo"].includes(x.type));
  return pool.sort((a, b) => (b.occurredAt ?? "").localeCompare(a.occurredAt ?? "")).slice(0, q.limit ?? 30);
}

// ---------------------------------------------------------------- engine
export const RECOMMENDED_MODEL = { id: "Qwen2.5-1.5B-Instruct-q4f16_1-MLC", contextLength: 2048, label: "Qwen2.5 1.5B" } as const;
/** The models まいにち offers (設定 → AIのモデル). The first one is the default. */
export const MODELS = [
  { id: "Qwen2.5-1.5B-Instruct-q4f16_1-MLC", contextLength: 2048, label: "Qwen2.5 1.5B", size: "約1GB", note: "いままでのモデル。日本語も自然" },
  { id: "tjs/LFM2.5-1.2B-JP", contextLength: 2048, label: "LFM2.5 1.2B JP", size: "約1.2GB", note: "日本語向け（Liquid AI・2026年）。メモリが少し少なくて済む" },
  { id: "gemma3-1b-it-q4f16_1-MLC", contextLength: 2048, label: "Gemma 3 1B", size: "約0.7GB", note: "いちばん軽い。日本語はまあまあ" },
] as const;

export function createHibiAI(src: HibiSource): PersonalAI & { hibiRoutes(question: string): boolean } {
  const adapter = defineAdapter({
    appId: APP_ID,
    appName: "ひび手帳",
    getSystemPrompt: () =>
      "あなたは手帳アプリ「ひび手帳」のアシスタント「まいにち」です。日本語で、やさしく簡潔に答えてください。" +
      "予定・日記・やることについては、渡された記録だけを根拠にし、記録にないことは「記録にはありません」と答えてください。" +
      "予定の追加は、ツールを使って頼まれたときだけ行います。予定の変更や削除はできないので、ひび手帳で行うよう伝えてください。" +
      "金額・回数・日付などの数は、渡された記録にあるものだけを使い、計算したり作ったりしないでください。記録から分からないときは、そう伝えて、家計簿なら「4月から10月の外食費はいくら？」のように聞き直してもらってください。" +
      (src.ready() && src.more().name ? `ユーザーの名前は「${src.more().name}」です。` : ""),
    async searchData(q) {
      return records(src, q, new Date());
    },
    getTools: () => tools,
  });
  const newsMem = { list: [] as NewsArticle[] };
  const tools = [routeSearchTool(src), newsTool(src, newsMem), baseballTool(src), weatherTool(src), routeChangeTool(src), addTool(src), moneyTool(src), habitTool(src), attendanceTool(src), nextTool(src), freeTool(src), todoTool(src), scheduleTool(src)] as unknown as AITool[];
  const engine = new PersonalAI({
    appId: APP_ID,
    adapter,
    timeZone: TZ,
    generation: { temperature: 0.6, maxTokens: 512 },
    backends: {
      webllm: { createWorker: () => new Worker(new URL("./webllm.worker.ts", import.meta.url), { type: "module" }) },
      transformers: { createWorker: () => new Worker(new URL("./transformers.worker.ts", import.meta.url), { type: "module" }) },
    },
  });
  return Object.assign(engine, {
    /** Would one of ひび手帳's exact tools answer this (without the model)? The page uses it for follow-ups. */
    hibiRoutes(question: string): boolean {
      const ctx = { appId: APP_ID, now: new Date(), timeZone: TZ };
      return tools.some((t) => { try { return !!t.route?.(question, ctx); } catch { return false; } });
    },
  });
}

