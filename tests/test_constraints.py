"""Fast tests for constraint extraction and filtering: no model, no index."""

import json

import pytest

from src.constraints import Constraints, Vocab, check_vocab, color_of, extract_llm, extract_rules, violations

METAS = [
    {"country": "US", "province": "Oregon", "variety": "Pinot Noir", "price": 18.0, "points": 91},
    {"country": "US", "province": "California", "variety": "Chardonnay", "price": 35.0, "points": 93},
    {"country": "Italy", "province": "Piedmont", "variety": "Nebbiolo", "price": 60.0, "points": 95},
    {"country": "France", "province": "Champagne", "variety": "Champagne Blend", "price": 45.0, "points": 92},
]
VOCAB = Vocab(METAS)


class FakeLLM:
    def __init__(self, replies):
        self.replies, self.prompts = list(replies), []

    def complete(self, prompt, system=None):
        self.prompts.append(prompt)
        return self.replies.pop(0)


def test_rules_price_words_and_place():
    c = extract_rules("a Pinot Noir from Oregon for less than thirty bucks")
    assert c.max_price == 30 and c.provinces == ["Oregon"] and c.varieties == ["Pinot Noir"]


def test_rules_ranges_and_negation():
    c = extract_rules("a Spanish white between $12 and $22, not Albariño")
    assert (c.min_price, c.max_price) == (12, 22)
    assert c.countries == ["Spain"] and c.color == "white" and c.exclude_varieties == ["Albariño"]
    c = extract_rules("a Merlot that's not from Chile, nothing buttery")
    assert c.exclude_countries == ["Chile"] and c.exclude_terms == ["butter"]


def test_color_mapping_handles_name_clashes():
    assert color_of({"variety": "Cabernet Sauvignon"}) == "red"
    assert color_of({"variety": "Portuguese White"}) == "white"
    assert color_of({"variety": "Grenache Blanc"}) == "white"
    assert color_of({"variety": "Pinot Noir", "province": "Champagne"}) == "sparkling"


def test_violations_list_each_clause():
    c = Constraints(max_price=20, provinces=["Oregon"], exclude_terms=["oak"])
    assert violations(c, "T\nbright cherry", METAS[0]) == []
    assert violations(c, "T\nsweet oak", METAS[1]) == ["price", "province", "excluded term"]


def test_schema_rejects_bad_ranges_and_unknown_keys():
    with pytest.raises(Exception):
        Constraints(min_price=30, max_price=20)
    with pytest.raises(Exception):
        Constraints.model_validate({"price": 20})


def test_vocab_canonicalizes_and_flags_unknown_places():
    c, errs = check_vocab(Constraints(countries=["italy"], provinces=["Napa"]), VOCAB)
    assert c.countries == ["Italy"] and errs and "Napa" in errs[0]


def test_llm_repair_loop_uses_validation_errors():
    bad = json.dumps({"provinces": ["Napa Valley"]})
    good = json.dumps({"provinces": ["California"], "max_price": 40})
    llm = FakeLLM([bad, good])
    out = extract_llm("a Napa wine under $40", llm, VOCAB)
    assert out["valid"] and out["repaired"] and out["constraints"].max_price == 40
    assert "failed validation" in llm.prompts[1]


def test_llm_gives_up_after_one_repair():
    out = extract_llm("x", FakeLLM(["nope", "{\"price\": 3}"]), VOCAB)
    assert not out["valid"] and out["constraints"] is None
