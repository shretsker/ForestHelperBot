from bot.handlers import apply_known_places, valid_name


def meta(glove=None, part=None, wide=False, topic="внески"):
    return {"title": "П", "glove": glove, "forest_part": part, "forest_wide": wide, "topic": topic}


def test_valid_name():
    assert valid_name("  Кабан   Старий ") == "Кабан Старий"
    assert valid_name("К") is None
    assert valid_name("") is None
    assert valid_name(None) is None
    assert valid_name("я" * 60) is None


def test_author_answer_naming_known_glove_beats_invented_part():
    out = apply_known_places(meta(part="ялинний ліс"), "Під ялинами",
                             ["Під дубом", "Під ялинами"], ["хвойний ліс"])
    assert out["glove"] == "Під ялинами"
    assert out["forest_part"] is None


def test_canonical_names_are_used():
    out = apply_known_places(meta(glove="під дубом"), None, ["Під дубом"], [])
    assert out["glove"] == "Під дубом"


def test_unknown_place_is_kept_as_is():
    out = apply_known_places(meta(part="біля греблі"), "біля греблі",
                             ["Під дубом"], ["хвойний ліс"])
    assert out["glove"] is None
    assert out["forest_part"] == "біля греблі"