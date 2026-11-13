// In-browser wine search: the rules extractor from src/constraints.py, ported line
// for line, plus dense search over an int8 index of 8,000 reviews. Everything loads
// from this site; env.allowRemoteModels is false, so no request leaves it.

const NUM_WORDS = { "five": 5, "ten": 10, "twelve": 12, "fifteen": 15, "twenty": 20, "twenty-five": 25,
  "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80,
  "hundred": 100, "a hundred": 100, "one hundred": 100 };
const esc = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
const NUM = "(\\$?\\s?\\d+(?:\\.\\d+)?|" + Object.keys(NUM_WORDS).sort((a, b) => b.length - a.length).map(esc).join("|") + ")";
const COUNTRY_WORDS = { "us": "US", "usa": "US", "american": "US", "united states": "US",
  "italy": "Italy", "italian": "Italy", "france": "France", "french": "France",
  "spain": "Spain", "spanish": "Spain", "portugal": "Portugal", "portuguese": "Portugal",
  "germany": "Germany", "german": "Germany", "austria": "Austria", "austrian": "Austria",
  "australia": "Australia", "australian": "Australia", "argentina": "Argentina",
  "argentinian": "Argentina", "argentine": "Argentina", "chile": "Chile",
  "chilean": "Chile", "new zealand": "New Zealand", "kiwi": "New Zealand",
  "south africa": "South Africa", "south african": "South Africa", "greece": "Greece",
  "greek": "Greece", "israel": "Israel", "israeli": "Israel", "hungary": "Hungary",
  "hungarian": "Hungary", "canada": "Canada", "canadian": "Canada", "england": "England",
  "english": "England", "uruguay": "Uruguay", "lebanon": "Lebanon" };
const PROVINCE_WORDS = { "oregon": "Oregon", "california": "California", "californian": "California",
  "washington": "Washington", "new york": "New York", "bordeaux": "Bordeaux",
  "burgundy": "Burgundy", "champagne": "Champagne", "alsace": "Alsace",
  "loire": "Loire Valley", "rhône": "Rhône Valley", "rhone": "Rhône Valley",
  "provence": "Provence", "beaujolais": "Beaujolais", "piedmont": "Piedmont",
  "tuscany": "Tuscany", "veneto": "Veneto", "sicily": "Sicily & Sardinia",
  "mosel": "Mosel", "rheingau": "Rheingau", "douro": "Douro", "mendoza": "Mendoza Province",
  "marlborough": "Marlborough", "wachau": "Wachau", "kamptal": "Kamptal",
  "stellenbosch": "Stellenbosch", "rioja": "Northern Spain" };
const GRAPES = ["cabernet sauvignon", "cabernet franc", "pinot noir", "pinot gris", "pinot grigio",
  "pinot blanc", "chardonnay", "sauvignon blanc", "riesling", "syrah", "shiraz", "merlot",
  "malbec", "zinfandel", "tempranillo", "sangiovese", "nebbiolo", "grenache", "barbera",
  "chenin blanc", "grüner veltliner", "gruner veltliner", "gewürztraminer", "viognier",
  "albariño", "gamay", "petite sirah", "carmenère", "mourvèdre", "aglianico", "torrontés",
  "blaufränkisch", "touriga nacional", "pinotage", "moscato", "sémillon", "vermentino",
  "melon", "port"];
const GRAPE_ALIAS = { "cab": "Cabernet Sauvignon", "cabernet": "Cabernet Sauvignon", "pinot": "Pinot Noir",
  "gruner veltliner": "Grüner Veltliner", "grüner": "Grüner Veltliner", "zin": "Zinfandel",
  "sauv blanc": "Sauvignon Blanc", "chard": "Chardonnay" };
const COLOR_WORDS = [["sparkling", "sparkling"], ["bubbly", "sparkling"], ["bubbles", "sparkling"],
  ["rosé", "rosé"], ["rose", "rosé"], ["red", "red"], ["white", "white"]];
const EXCLUDE_TERMS = { "oaky": "oak", "oak": "oak", "jammy": "jam", "sweet": "sweet", "buttery": "butter",
  "tannic": "tannic", "fruity": "fruity" };
const NEG = "(?:not|no|nothing|isn't|isnt|without|never|but not|except)";
// Python's \w is Unicode-aware; JS's is ASCII, so spell it out.
const W = "[\\p{L}\\p{N}_]";
const word = (w) => new RegExp("(?<![\\p{L}\\p{N}_-])" + esc(w) + "(?![\\p{L}\\p{N}_-])", "gu");

function num(s) {
  s = s.trim().replace(/^\$/, "").trim();
  return s in NUM_WORDS ? NUM_WORDS[s] : parseFloat(s);
}
function title(g) {
  return g.split(" ").map((w) => (w === "de" || w === "di") ? w : w.slice(0, 1).toUpperCase() + w.slice(1)).join(" ");
}

export function extractRules(query) {
  const q = " " + query.toLowerCase().replace(/’/g, "'") + " ";
  const d = { countries: [], provinces: [], varieties: [], exclude_countries: [],
              exclude_varieties: [], exclude_terms: [] };
  let m = q.match(new RegExp("between\\s+" + NUM + "\\s*(?:dollars|bucks)?\\s+and\\s+" + NUM)) ||
          q.match(new RegExp(NUM + "\\s*(?:dollars|bucks)?\\s*(?:to|-)\\s*" + NUM + "\\s*(?:dollars|bucks)"));
  if (m) { d.min_price = num(m[1]); d.max_price = num(m[2]); }
  else {
    m = q.match(new RegExp("(?:under|less than|below|no more than|nothing more than|nothing over|not over|" +
                           "max(?:imum)?|up to|at most|cheaper than)\\s+" + NUM));
    if (m) d.max_price = num(m[1]);
    m = q.match(new RegExp("(?:over|more than|above|at least)\\s+" + NUM + "\\s*(?:dollars|bucks)"));
    if (m && !("max_price" in d)) d.min_price = num(m[1]);
  }
  m = q.match(/(\d{2,3})[- ]?(?:point|pts|pt)/);
  if (m && +m[1] >= 80 && +m[1] <= 100) d.min_points = +m[1];

  const negRe = new RegExp(NEG + "\\s+(?:" + W + "+\\s+){0,2}$", "u");
  const negated = (start) => negRe.test(q.slice(Math.max(0, start - 30), start));
  const push = (k, v) => { if (!d[k].includes(v)) d[k].push(v); };

  for (const [w, val] of Object.entries(COUNTRY_WORDS).sort((a, b) => b[0].length - a[0].length))
    for (const mm of q.matchAll(word(w))) push(negated(mm.index) ? "exclude_countries" : "countries", val);
  for (const [w, val] of Object.entries(PROVINCE_WORDS))
    if (word(w).test(q)) push("provinces", val);
  const found = [];
  for (const g of [...GRAPES, ...Object.keys(GRAPE_ALIAS)].sort((a, b) => b.length - a.length)) {
    for (const mm of q.matchAll(word(g))) {
      if (found.some(([s, e]) => s <= mm.index && mm.index < e)) continue;
      found.push([mm.index, mm.index + mm[0].length]);
      push(negated(mm.index) ? "exclude_varieties" : "varieties", GRAPE_ALIAS[g] || title(g));
    }
  }
  for (const [w, val] of COLOR_WORDS) if (word(w).test(q)) { d.color = val; break; }
  if (d.provinces.includes("Champagne") && !d.color) d.color = "sparkling";
  for (const [w, root] of Object.entries(EXCLUDE_TERMS))
    for (const mm of q.matchAll(word(w))) if (negated(mm.index)) push("exclude_terms", root);
  return d;
}

// rows: [title, variety, country, province, price, points, color, note, flags]
export function violations(c, r, termFlags) {
  const out = [];
  const [, variety, country, province, price, points, color, note, flags] = r;
  const v = (variety || "").toLowerCase();
  if (c.max_price != null && (price == null || price > c.max_price)) out.push("price");
  if (c.min_price != null && (price == null || price < c.min_price)) out.push("price");
  if (c.min_points != null && (points || 0) < c.min_points) out.push("points");
  if (c.countries?.length && !c.countries.includes(country)) out.push("country");
  if (c.provinces?.length && !c.provinces.includes(province)) out.push("province");
  if (c.varieties?.length && !c.varieties.some((x) => v.includes(x.toLowerCase()))) out.push("variety");
  if (c.color && color !== c.color) out.push("color");
  if (c.exclude_countries?.length && c.exclude_countries.includes(country)) out.push("excluded country");
  if (c.exclude_varieties?.length && c.exclude_varieties.some((x) => v.includes(x.toLowerCase()))) out.push("excluded variety");
  if (c.exclude_terms?.length && c.exclude_terms.some((t) => {
    const b = termFlags.indexOf(t);
    return b >= 0 ? (flags >> b) & 1 : (note || "").toLowerCase().includes(t);
  })) out.push("excluded term");
  return [...new Set(out)];
}

let state = null;

export async function load(base, onStage) {
  if (state) return state;
  base = new URL(base, location.href).href;  // dynamic import() resolves against this module, not the page
  const t0 = performance.now();
  onStage?.("index");
  const [meta, buf] = await Promise.all([
    fetch(base + "index/meta.json").then((r) => r.json()),
    fetch(base + "index/emb.i8").then((r) => r.arrayBuffer()),
  ]);
  onStage?.("model");
  const { pipeline, env } = await import(base + "vendor/transformers.min.js");
  env.allowRemoteModels = false;
  env.allowLocalModels = true;
  env.localModelPath = new URL(base).pathname + "models/";  // a path: a full URL here breaks the tokenizer load
  env.backends.onnx.wasm.wasmPaths = {
    mjs: new URL(base + "vendor/ort-wasm-simd-threaded.jsep.mjs", location).href,
    wasm: new URL(base + "vendor/ort-wasm-simd-threaded.jsep.wasm", location).href,
  };
  const ext = await pipeline("feature-extraction", "all-MiniLM-L6-v2", { dtype: "q8", device: "wasm" });
  state = { meta, emb: new Int8Array(buf), ext, loadMs: performance.now() - t0 };
  return state;
}

export async function embed(text) {
  const out = await state.ext([text], { pooling: "mean", normalize: true });
  return out.data;
}

export async function search(query, { applyFilters = true, k = 10 } = {}) {
  const t0 = performance.now();
  const q = await embed(query);
  const t1 = performance.now();
  const { meta, emb } = state;
  const c = extractRules(query);
  const dim = meta.dim, n = meta.n, scores = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    let s = 0;
    const o = i * dim;
    for (let j = 0; j < dim; j++) s += q[j] * emb[o + j];
    scores[i] = s * meta.scales[i];
  }
  const order = [];
  for (let i = 0; i < n; i++) {
    if (applyFilters && violations(c, meta.rows[i], meta.term_flags).length) continue;
    order.push(i);
  }
  order.sort((a, b) => scores[b] - scores[a] || a - b);
  const top = order.slice(0, k).map((i) => ({ i, score: scores[i], row: meta.rows[i],
    violates: violations(c, meta.rows[i], meta.term_flags) }));
  return { constraints: c, top, pool: order.length, embedMs: t1 - t0, rankMs: performance.now() - t1 };
}
