from clinical_grpo.prompts.templates import SYSTEM_PROMPT, build_chat, user_message


def test_system_prompt_is_nonempty():
    assert isinstance(SYSTEM_PROMPT, str)
    assert len(SYSTEM_PROMPT) > 0


def test_system_prompt_mentions_icd10():
    assert "ICD-10" in SYSTEM_PROMPT


def test_user_message_contains_summary():
    summary = "Patient admitted with chest pain."
    msg = user_message(summary)
    assert summary in msg


def test_user_message_contains_json_contract():
    msg = user_message("some summary")
    assert '"codes"' in msg


def test_user_message_contains_codes_key():
    msg = user_message("some summary")
    assert "codes" in msg


def test_build_chat_returns_two_messages():
    messages = build_chat("Patient with hypertension.")
    assert len(messages) == 2


def test_build_chat_first_role_is_system():
    messages = build_chat("Patient with hypertension.")
    assert messages[0]["role"] == "system"


def test_build_chat_second_role_is_user():
    messages = build_chat("Patient with hypertension.")
    assert messages[1]["role"] == "user"


def test_build_chat_system_content_is_system_prompt():
    messages = build_chat("Any summary.")
    assert messages[0]["content"] == SYSTEM_PROMPT


def test_build_chat_user_content_contains_summary():
    summary = "Admitted for cellulitis of lower leg."
    messages = build_chat(summary)
    assert summary in messages[1]["content"]


def test_build_chat_messages_are_dicts():
    messages = build_chat("summary text")
    for msg in messages:
        assert isinstance(msg, dict)
        assert "role" in msg
        assert "content" in msg
