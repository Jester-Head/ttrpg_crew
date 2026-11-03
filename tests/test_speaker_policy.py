


def test_triggers_match_whole_words_only():
    from ttrpg_crew.game.speaker_policy import _selection_meta, matched_triggers

    meta = _selection_meta({"selection": {"triggers": ["leave", "help"]}})
    assert matched_triggers(meta, "i tug at my sleeve, helpless") == []
    assert matched_triggers(meta, "we leave at dawn") == ['"leave"']


def test_grouped_triggers_report_their_label_and_match_phrases():
    from ttrpg_crew.game.speaker_policy import _selection_meta, matched_triggers

    meta = _selection_meta(
        {"selection": {"triggers": {"being left behind": ["left behind", "leave me"]}}}
    )
    assert matched_triggers(meta, "leave it alone, it's cursed") == []
    assert matched_triggers(meta, "don't leave me here") == ["being left behind"]


def test_situation_keywords_match_word_forms_not_substrings():
    from ttrpg_crew.game.speaker_policy import detect_situations

    assert "investigation" in detect_situations("i keep looking at the door")
    assert "combat" not in detect_situations("i open the drawer")
    assert "supernatural" not in detect_situations("i write it down")
