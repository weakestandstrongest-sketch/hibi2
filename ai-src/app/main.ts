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
  /** 刻に出す（省略：ひび手帳の決まりどおり。タイムラインだけの予定は出さない） */
  chrOn?: boolean;
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
  /** ひび手帳の家計簿（支出・収入の記録とカテゴリー） */
  money(): MoneyData;
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
  const opts = [e.show === "cal" ? "カレンダーだけ" : e.show === "tl" ? "タイムラインだけ" : "", e.countdown ? "カウントダウン" : "", e.chrOn === true && e.show === "tl" ? "刻にも出す" : e.chrOn === false && e.show !== "tl" ? "刻には出さない" : ""].filter(Boolean);
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
        chrOn: { type: "boolean", description: "刻に出すかどうか（任意）" },
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
        show, ...(args.countdown === true ? { countdown: true } : {}), ...(typeof args.chrOn === "boolean" ? { chrOn: args.chrOn } : {}) };
      if (!e.title) throw new Error("予定の名前がありません");
      if (!allDay && !(/^([01]\d|2[0-3]):[0-5]\d$/.test(e.start!) && /^([01]\d|2[0-3]):[0-5]\d$/.test(e.end!))) throw new Error("時刻が正しくありません");
      return src.addEvent(e);
    },
  };
}

// ---------------------------------------------------------------- 家計簿 (exact sums, no model)
const MONEY_Q = /円|いくら|幾ら|金額|使っ|つかっ|支出|出費|収入|家計簿|お金|稼い|もらった/;
interface MoneyArgs { from?: string; to?: string; label: string; kind: "out" | "in"; cats: string[]; catNames: string[]; words: string[] }
const lastDay = (y: number, m: number) => new Date(y, m, 0).getDate();
/** 「4月から今日まで」「この4月から」「4月から9月まで」「先月」「今年」… → a range; none = every record */
function moneyRange(q: string, t: string): { from?: string; to?: string; label: string } {
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
      "予定・日記・やることについては、渡された記録だけを根拠にし、記録にないことは「記録にはありません」と答えてください。" +
      "予定の追加は、ツールを使って頼まれたときだけ行います。予定の変更や削除はできないので、ひび手帳で行うよう伝えてください。",
    async searchData(q) {
      return records(src, q, new Date());
    },
    getTools: () => [addTool(src), moneyTool(src), nextTool(src), freeTool(src), todoTool(src), scheduleTool(src)] as unknown as AITool[],
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

