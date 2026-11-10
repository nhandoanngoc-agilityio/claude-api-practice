import pytest
from fakes import FakeClient, message, text, thinking
from pydantic import BaseModel

from claude_api_lab import llm


def test_send_uses_default_model_with_refusal_fallback_and_effort():
    client = FakeClient(message(text("hi")))
    assert llm.ask(client, "hello", effort="low") == "hi"
    params = client.last()
    assert params["model"] == llm.MODEL
    assert params["fallbacks"] == "default"
    assert params["betas"] == [llm.FALLBACK_BETA]
    assert params["output_config"] == {"effort": "low"}
    assert params["messages"] == [{"role": "user", "content": "hello"}]
    assert "temperature" not in params


def test_send_on_other_model_skips_fallback_beta():
    client = FakeClient(message(text("ok")))
    llm.ask(client, "x", model=llm.FAST_MODEL)
    assert "fallbacks" not in client.last()


def test_output_config_merges_effort_and_format():
    params = llm._params(
        [], model=None, system=None, max_tokens=10, effort="high", output_config={"format": {"a": 1}}
    )
    assert params["output_config"] == {"effort": "high", "format": {"a": 1}}


def test_refusal_raises():
    refused = message(text(""), stop_reason="refusal")
    with pytest.raises(llm.Refused):
        llm.send(FakeClient(refused), "x")


def test_text_of_ignores_thinking_blocks():
    assert llm.text_of(message(thinking("hmm"), text("a"), text("b"))) == "ab"


def test_conversation_keeps_full_history_with_system_prompt():
    first = message(thinking("t"), text("Hi Alice"))
    client = FakeClient(first, message(text("Your name is Alice")))
    convo = llm.Conversation(client, system="Be brief.")
    convo.say("My name is Alice")
    assert convo.say("What's my name?") == "Your name is Alice"
    sent = client.last()
    assert sent["system"] == "Be brief."
    # the fake holds a reference to the live history, which now ends with the reply
    roles = [m["role"] for m in sent["messages"]]
    assert roles == ["user", "assistant", "user", "assistant"]
    assert sent["messages"][1]["content"] is first.content  # thinking passed back unchanged


def test_streaming_yields_chunks_and_records_final_turn():
    client = FakeClient(message(text("Hel"), text("lo")))
    convo = llm.Conversation(client)
    assert "".join(convo.say_streaming("hi")) == "Hello"
    assert convo.messages[-1]["role"] == "assistant"
    assert client.calls[0][0] == "stream"


def test_extract_returns_validated_model():
    class Contact(BaseModel):
        name: str

    client = FakeClient(Contact(name="Jane"))
    assert llm.extract(client, "Jane wrote in", Contact).name == "Jane"
    assert client.last()["output_format"] is Contact


def test_stop_sequences_pass_through():
    client = FakeClient(message(text("1, 2, 3"), stop_reason="stop_sequence"))
    llm.send(client, "count", stop_sequences=["4"])
    assert client.last()["stop_sequences"] == ["4"]


def test_usage_accumulates():
    usage = llm.Usage()
    usage.add(message(text("a"), input_tokens=100, cache_read_input_tokens=80))
    usage.add(message(text("b"), input_tokens=5))
    assert (usage.input_tokens, usage.cache_read_input_tokens) == (105, 80)


def test_count_tokens():
    client = FakeClient()
    assert llm.count_tokens(client, [{"role": "user", "content": "x"}]) == 42
