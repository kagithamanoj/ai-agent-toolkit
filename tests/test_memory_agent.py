from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agents.memory_agent import ConversationMemory, MemoryAgent


def test_messages_are_stored_in_order():
    m = ConversationMemory()
    m.add_user_message("q")
    m.add_ai_message("a")
    msgs = m.get_messages()
    assert [type(x) for x in msgs] == [HumanMessage, AIMessage]
    assert m.message_count == 2


def test_sliding_window_keeps_latest_and_summarises_old():
    m = ConversationMemory(max_messages=2)
    for i in range(4):
        m.add_user_message(f"msg{i}")
    assert [x.content for x in m.messages] == ["msg2", "msg3"]
    assert "msg0" in m.summary and "msg1" in m.summary
    first = m.get_messages()[0]
    assert isinstance(first, SystemMessage)
    assert "Previous conversation summary" in first.content


def test_clear_resets_everything():
    m = ConversationMemory(max_messages=1)
    m.add_user_message("a")
    m.add_user_message("b")
    m.clear()
    assert m.messages == [] and m.summary == "" and m.message_count == 0


def test_chat_round_trip_updates_memory():
    agent = MemoryAgent()
    seen = {}

    class FakeChain:
        def invoke(self, payload):
            seen.update(payload)
            return "pong"

    agent.chain = FakeChain()
    assert agent.chat("ping") == "pong"
    assert seen["input"] == "ping" and seen["history"] == []
    assert agent.history_length == 2

    agent.chat("again")
    assert len(seen["history"]) == 2  # previous turn is fed back in

    agent.reset()
    assert agent.history_length == 0
