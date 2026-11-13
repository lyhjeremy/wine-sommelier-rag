"""Constraint extraction: plain-English limits to validated metadata filters.

"A Pinot Noir from Oregon under twenty bucks" carries three hard constraints
that a ranker cannot honor by similarity alone. This module turns a request into
a ``Constraints`` object (pydantic, JSON schema below), checks every value
against the index's own vocabulary, and applies the result as a filter over the
corpus *before* ranking.

Two extractors share the schema:
  * ``extract_llm``: Claude returns JSON for the schema; the reply is validated
    in code (types, ranges, and every country, province and grape must exist in
    the index). An invalid reply gets one repair attempt with the errors listed.
  * ``extract_rules``: a deterministic regex parser with no model, the same
    logic as the in-browser demo on the write-up (overview fragment, ``rulesJS``).
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

Color = Literal["red", "white", "rosé", "sparkling"]


class Constraints(BaseModel):
    model_config = ConfigDict(extra="forbid")
    max_price: float | None = Field(None, ge=0, description="USD, inclusive")
    min_price: float | None = Field(None, ge=0, description="USD, inclusive")
    min_points: int | None = Field(None, ge=80, le=100)
    countries: list[str] = Field(default_factory=list, description="any of, exact index values")
    provinces: list[str] = Field(default_factory=list, description="any of: US state or wine region, exact index values")
    varieties: list[str] = Field(default_factory=list, description="any of; grape names, matched case-insensitively as substrings")
    color: Color | None = None
    exclude_countries: list[str] = Field(default_factory=list)
    exclude_varieties: list[str] = Field(default_factory=list)
    exclude_terms: list[str] = Field(default_factory=list, description="root words that must not appear in the review, e.g. 'oak' for 'not oaky'")

    @model_validator(mode="after")
    def _range(self):
        if self.min_price is not None and self.max_price is not None and self.min_price > self.max_price:
            raise ValueError("min_price is above max_price")
        return self

    def is_empty(self) -> bool:
        return self == Constraints()


# --------------------------------------------------------------------------- #
# Color, from the variety (and Champagne's province)
# --------------------------------------------------------------------------- #
_SPARKLING = ("sparkling", "champagne blend", "prosecco", "glera", "cava", "crémant", "lambrusco")
_ROSE = ("rosé", "rosado", "rosato")
_RED = ("cabernet", "merlot", "syrah", "shiraz", "malbec", "pinot noir", "zinfandel", "tempranillo",
        "sangiovese", "grenache", "garnacha", "nebbiolo", "red", "petite sirah", "barbera", "mourvèdre",
        "carmenère", "tinta", "touriga", "aglianico", "blaufränkisch", "gamay", "petit verdot",
        "nero d'avola", "primitivo", "montepulciano", "corvina", "tannat", "pinotage", "mencía",
        "dolcetto", "zweigelt", "st. laurent", "port", "prugnolo", "sagrantino", "carignan",
        "cinsault", "lagrein", "teroldego", "monastrell", "baga", "xinomavro", "nerello",
        "negroamaro", "cannonau", "alicante", "charbono", "counoise", "graciano", "spätburgunder",
        "rondinella", "brunello", "bonarda", "refosco", "schioppettino", "frappato", "kadarka",
        "g-s-m", "tinto", "pinot nero", "meritage", "claret", "petite verdot", "mataro", "castelão",
        "jaen", "braucol", "susumaniello", "alfrocheiro", "pinot meunier", "mencia")
_WHITE = ("chardonnay", "sauvignon blanc", "riesling", "pinot gris", "pinot grigio", "chenin blanc",
          "gewürztraminer", "viognier", "albariño", "grüner", "white", "sémillon", "semillon",
          "moscato", "muscat", "verdejo", "vermentino", "melon", "torrontés", "furmint", "garganega",
          "fiano", "assyrtiko", "picpoul", "godello", "sylvaner", "silvaner", "pinot blanc", "marsanne",
          "roussanne", "greco", "falanghina", "verdicchio", "trebbiano", "arneis", "cortese",
          "friulano", "ribolla", "loureiro", "alvarinho", "encruzado", "rkatsiteli", "petit manseng",
          "gros manseng", "tokaji", "weissburgunder", "welschriesling", "malvasia", "verduzzo",
          "carricante", "vernaccia", "pecorino", "grillo", "catarratto", "inzolia", "kerner",
          "müller-thurgau", "scheurebe", "savagnin", "jacquère", "aligoté", "macabeo",
          "viura", "xarel-lo", "hondarrabi", "treixadura", "rolle", "clairette", "colombard",
          "vidal", "seyval", "turbiana", "picolit", "zibibbo", "bual", "assyrtico", "albana",
          "loin de l'oeil", "muscadelle", "moschofilero", "fernão pires", "arinto", "tokay",
          "moscadello", "sämling", "grauburgunder", "traminer", "gelber", "zierfandler", "rotgipfler", "neuburger",
          "roter veltliner", "sherry", "palomino", "pedro ximénez", "moscatel", "tocai", "ugni")


def color_of(meta: dict) -> str | None:
    v = str(meta.get("variety", "")).lower()
    if meta.get("province") == "Champagne" or any(m in v for m in _SPARKLING):
        return "sparkling"
    if any(m in v for m in _ROSE):
        return "rosé"
    if any(w in v for w in ("white", "blanc", "bianco", "weiss")):
        return "white"
    if "red" in v or any(m in v for m in _RED):
        return "red"
    if v == "sauvignon" or any(m in v for m in _WHITE):
        return "white"
    return None


# --------------------------------------------------------------------------- #
# Applying constraints
# --------------------------------------------------------------------------- #
def passes(c: Constraints, doc: str, meta: dict) -> bool:
    return not violations(c, doc, meta)


def violations(c: Constraints, doc: str, meta: dict) -> list[str]:
    """Every constraint clause this review breaks (empty list = it satisfies all)."""
    out = []
    price = meta.get("price")
    v = str(meta.get("variety", "")).lower()
    if c.max_price is not None and (price is None or price > c.max_price):
        out.append("price")
    if c.min_price is not None and (price is None or price < c.min_price):
        out.append("price")
    if c.min_points is not None and (meta.get("points") or 0) < c.min_points:
        out.append("points")
    if c.countries and meta.get("country") not in c.countries:
        out.append("country")
    if c.provinces and meta.get("province") not in c.provinces:
        out.append("province")
    if c.varieties and not any(x.lower() in v for x in c.varieties):
        out.append("variety")
    if c.color and color_of(meta) != c.color:
        out.append("color")
    if c.exclude_countries and meta.get("country") in c.exclude_countries:
        out.append("excluded country")
    if c.exclude_varieties and any(x.lower() in v for x in c.exclude_varieties):
        out.append("excluded variety")
    if c.exclude_terms:
        body = doc.split("\n", 1)[-1].lower()
        if any(t.lower() in body for t in c.exclude_terms):
            out.append("excluded term")
    return sorted(set(out), key=out.index)


# --------------------------------------------------------------------------- #
# Validation against the index vocabulary
# --------------------------------------------------------------------------- #
class Vocab:
    def __init__(self, metas: list[dict]):
        self.countries = sorted({m["country"] for m in metas if m.get("country")})
        self.provinces = sorted({m["province"] for m in metas if m.get("province")})
        self.varieties = sorted({m["variety"] for m in metas if m.get("variety")})
        self._v_lower = [v.lower() for v in self.varieties]

    def canon(self, value: str, pool: list[str]) -> str | None:
        for p in pool:
            if p.lower() == value.strip().lower():
                return p
        return None

    def variety_ok(self, value: str) -> bool:
        x = value.strip().lower()
        return bool(x) and any(x in v for v in self._v_lower)

    def suggest(self, value: str, pool: list[str], n: int = 5) -> list[str]:
        import difflib
        return difflib.get_close_matches(value, pool, n=n, cutoff=0.5)


def check_vocab(c: Constraints, vocab: Vocab) -> tuple[Constraints, list[str]]:
    """Canonicalize place names and check grapes; return (fixed, errors)."""
    errs = []
    d = c.model_dump()
    for field, pool in (("countries", vocab.countries), ("provinces", vocab.provinces),
                        ("exclude_countries", vocab.countries)):
        fixed = []
        for x in d[field]:
            can = vocab.canon(x, pool)
            if can is None:
                errs.append(f"{field}: '{x}' is not in the index; close values: {vocab.suggest(x, pool)}")
            else:
                fixed.append(can)
        d[field] = fixed
    for field in ("varieties", "exclude_varieties"):
        for x in d[field]:
            if not vocab.variety_ok(x):
                errs.append(f"{field}: no grape in the index contains '{x}'; close values: "
                            f"{vocab.suggest(x, vocab.varieties)}")
    return Constraints(**d), errs


# --------------------------------------------------------------------------- #
# LLM extractor
# --------------------------------------------------------------------------- #
SYSTEM = ("You convert wine requests into search filters. Respond with ONLY valid JSON, "
          "no prose, no markdown fences.")


def extraction_prompt(query: str) -> str:
    return (
        f'Request: "{query}"\n\n'
        "Extract only the hard constraints the request states. Leave a field empty or null "
        "when the request does not state it; do not infer a price, place or grape from a food "
        "or an occasion. Rules:\n"
        "- Prices are USD numbers per bottle. 'under/less than/no more than X' sets max_price = X; "
        "'between A and B' or 'A to B' sets both.\n"
        "- countries use the index's names: 'US' for the United States, 'Italy', 'France', "
        "'New Zealand', 'South Africa'. A US state or a region such as Oregon, California, "
        "Washington, Bordeaux, Burgundy, Champagne, Piedmont, Alsace or Mosel goes in provinces, "
        "and its country goes in countries only if the request names the country too.\n"
        "- varieties are grape names as written on labels ('Pinot Noir', 'Cabernet Sauvignon', "
        "'Shiraz'). 'Cab' means 'Cabernet Sauvignon'. A style word (Barolo, Chablis) is not a grape.\n"
        "- color is one of red, white, rosé, sparkling, set only when the request says it or names "
        "a sparkling style such as Champagne.\n"
        "- 'not from X' goes in exclude_countries; 'not X' for a grape goes in exclude_varieties; "
        "'not oaky', 'no oak', 'not jammy' go in exclude_terms as root words ('oak', 'jam').\n"
        "- min_points only when a score is stated.\n\n"
        f"JSON schema:\n{json.dumps(Constraints.model_json_schema())}"
    )


def extract_llm(query: str, llm, vocab: Vocab) -> dict:
    """Returns {"constraints": Constraints | None, "valid", "repaired", "errors", "raw"}."""
    from .util_json import parse_json

    prompt = extraction_prompt(query)
    raws, errors = [], []
    for attempt in range(2):
        raw = llm.complete(prompt, system=SYSTEM)
        raws.append(raw)
        try:
            c = Constraints.model_validate(parse_json(raw))
            c, errs = check_vocab(c, vocab)
        except (ValueError, ValidationError) as e:
            errs = [str(e)[:500]]
            c = None
        if not errs:
            return {"constraints": c, "valid": True, "repaired": attempt == 1,
                    "errors": errors, "raw": raws}
        errors.extend(errs)
        prompt = (extraction_prompt(query) + "\n\nYour previous reply was:\n" + raw[:800]
                  + "\nIt failed validation:\n- " + "\n- ".join(errs)
                  + "\nReturn corrected JSON only.")
    return {"constraints": None, "valid": False, "repaired": False, "errors": errors, "raw": raws}


# --------------------------------------------------------------------------- #
# Rules extractor (no model). Mirrors rulesJS in the write-up demo.
# --------------------------------------------------------------------------- #
NUM_WORDS = {"five": 5, "ten": 10, "twelve": 12, "fifteen": 15, "twenty": 20, "twenty-five": 25,
             "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80,
             "hundred": 100, "a hundred": 100, "one hundred": 100}
_NUM = r"(\$?\s?\d+(?:\.\d+)?|" + "|".join(sorted(map(re.escape, NUM_WORDS), key=len, reverse=True)) + r")"
COUNTRY_WORDS = {"us": "US", "usa": "US", "american": "US", "united states": "US",
                 "italy": "Italy", "italian": "Italy", "france": "France", "french": "France",
                 "spain": "Spain", "spanish": "Spain", "portugal": "Portugal", "portuguese": "Portugal",
                 "germany": "Germany", "german": "Germany", "austria": "Austria", "austrian": "Austria",
                 "australia": "Australia", "australian": "Australia", "argentina": "Argentina",
                 "argentinian": "Argentina", "argentine": "Argentina", "chile": "Chile",
                 "chilean": "Chile", "new zealand": "New Zealand", "kiwi": "New Zealand",
                 "south africa": "South Africa", "south african": "South Africa", "greece": "Greece",
                 "greek": "Greece", "israel": "Israel", "israeli": "Israel", "hungary": "Hungary",
                 "hungarian": "Hungary", "canada": "Canada", "canadian": "Canada", "england": "England",
                 "english": "England", "uruguay": "Uruguay", "lebanon": "Lebanon"}
PROVINCE_WORDS = {"oregon": "Oregon", "california": "California", "californian": "California",
                  "washington": "Washington", "new york": "New York", "bordeaux": "Bordeaux",
                  "burgundy": "Burgundy", "champagne": "Champagne", "alsace": "Alsace",
                  "loire": "Loire Valley", "rhône": "Rhône Valley", "rhone": "Rhône Valley",
                  "provence": "Provence", "beaujolais": "Beaujolais", "piedmont": "Piedmont",
                  "tuscany": "Tuscany", "veneto": "Veneto", "sicily": "Sicily & Sardinia",
                  "mosel": "Mosel", "rheingau": "Rheingau", "douro": "Douro", "mendoza": "Mendoza Province",
                  "marlborough": "Marlborough", "wachau": "Wachau", "kamptal": "Kamptal",
                  "stellenbosch": "Stellenbosch", "rioja": "Northern Spain"}
GRAPES = ["cabernet sauvignon", "cabernet franc", "pinot noir", "pinot gris", "pinot grigio",
          "pinot blanc", "chardonnay", "sauvignon blanc", "riesling", "syrah", "shiraz", "merlot",
          "malbec", "zinfandel", "tempranillo", "sangiovese", "nebbiolo", "grenache", "barbera",
          "chenin blanc", "grüner veltliner", "gruner veltliner", "gewürztraminer", "viognier",
          "albariño", "gamay", "petite sirah", "carmenère", "mourvèdre", "aglianico", "torrontés",
          "blaufränkisch", "touriga nacional", "pinotage", "moscato", "sémillon", "vermentino",
          "melon", "port"]
GRAPE_ALIAS = {"cab": "Cabernet Sauvignon", "cabernet": "Cabernet Sauvignon", "pinot": "Pinot Noir",
               "gruner veltliner": "Grüner Veltliner", "grüner": "Grüner Veltliner", "zin": "Zinfandel",
               "sauv blanc": "Sauvignon Blanc", "chard": "Chardonnay"}
COLOR_WORDS = [("sparkling", "sparkling"), ("bubbly", "sparkling"), ("bubbles", "sparkling"),
               ("rosé", "rosé"), ("rose", "rosé"), ("red", "red"), ("white", "white")]
EXCLUDE_TERMS = {"oaky": "oak", "oak": "oak", "jammy": "jam", "sweet": "sweet", "buttery": "butter",
                 "tannic": "tannic", "fruity": "fruity"}
NEG = r"(?:not|no|nothing|isn't|isnt|without|never|but not|except)"


def _num(s: str) -> float:
    s = s.strip().lstrip("$").strip()
    return float(NUM_WORDS.get(s, s))


def _title(g: str) -> str:
    return " ".join(w if w in ("de", "di") else w[:1].upper() + w[1:] for w in g.split())


def extract_rules(query: str) -> Constraints:
    q = " " + query.lower().replace("’", "'") + " "
    d: dict = {k: [] for k in ("countries", "provinces", "varieties", "exclude_countries",
                               "exclude_varieties", "exclude_terms")}
    m = re.search(r"between\s+" + _NUM + r"\s*(?:dollars|bucks)?\s+and\s+" + _NUM, q) or \
        re.search(_NUM + r"\s*(?:dollars|bucks)?\s*(?:to|-)\s*" + _NUM + r"\s*(?:dollars|bucks)", q)
    if m:
        d["min_price"], d["max_price"] = _num(m.group(1)), _num(m.group(2))
    else:
        m = re.search(r"(?:under|less than|below|no more than|nothing more than|nothing over|not over|"
                      r"max(?:imum)?|up to|at most|cheaper than)\s+" + _NUM, q)
        if m:
            d["max_price"] = _num(m.group(1))
        m = re.search(r"(?:over|more than|above|at least)\s+" + _NUM + r"\s*(?:dollars|bucks)", q)
        if m and "max_price" not in d:
            d["min_price"] = _num(m.group(1))
    m = re.search(r"(\d{2,3})[- ]?(?:point|pts|pt)", q)
    if m and 80 <= int(m.group(1)) <= 100:
        d["min_points"] = int(m.group(1))

    def negated(start: int) -> bool:
        return re.search(NEG + r"\s+(?:\w+\s+){0,2}$", q[max(0, start - 30):start]) is not None

    for word, val in sorted(COUNTRY_WORDS.items(), key=lambda t: -len(t[0])):
        for mm in re.finditer(r"(?<![\w-])" + re.escape(word) + r"(?![\w-])", q):
            key = "exclude_countries" if negated(mm.start()) else "countries"
            if val not in d[key]:
                d[key].append(val)
    for word, val in PROVINCE_WORDS.items():
        if re.search(r"(?<![\w-])" + re.escape(word) + r"(?![\w-])", q) and val not in d["provinces"]:
            d["provinces"].append(val)
    found = []
    for g in sorted(GRAPES + list(GRAPE_ALIAS), key=len, reverse=True):
        for mm in re.finditer(r"(?<![\w-])" + re.escape(g) + r"(?![\w-])", q):
            if any(s <= mm.start() < e for s, e in found):
                continue
            found.append((mm.start(), mm.end()))
            val = GRAPE_ALIAS.get(g, _title(g))
            key = "exclude_varieties" if negated(mm.start()) else "varieties"
            if val not in d[key]:
                d[key].append(val)
    for word, val in COLOR_WORDS:
        if re.search(r"(?<![\w-])" + re.escape(word) + r"(?![\w-])", q):
            d["color"] = val
            break
    if "champagne" in d["provinces"] or "Champagne" in d["provinces"]:
        d.setdefault("color", "sparkling")
    for word, root in EXCLUDE_TERMS.items():
        for mm in re.finditer(r"(?<![\w-])" + re.escape(word) + r"(?![\w-])", q):
            if negated(mm.start()) and root not in d["exclude_terms"]:
                d["exclude_terms"].append(root)
    return Constraints(**d)
