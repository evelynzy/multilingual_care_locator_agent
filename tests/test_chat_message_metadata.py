import sys
import types
import unittest
from typing import Optional


if "huggingface_hub" not in sys.modules:
    huggingface_stub = types.ModuleType("huggingface_hub")

    class _StubInferenceClient:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("InferenceClient stub should not be used in tests")

    huggingface_stub.InferenceClient = _StubInferenceClient
    sys.modules["huggingface_hub"] = huggingface_stub

if "requests" not in sys.modules:
    requests_stub = types.ModuleType("requests")

    def _stub_get(*args, **kwargs):
        raise RuntimeError("requests.get stub should be patched in tests")

    requests_stub.get = _stub_get
    sys.modules["requests"] = requests_stub

from care import normalize_chat_messages


class ChatMessageMetadataTests(unittest.TestCase):
    def test_normalize_chat_messages_strips_metadata_and_preserves_order(self) -> None:
        messages = [
            {
                "role": "system",
                "content": "system guidance",
                "metadata": {"id": "system-turn"},
            },
            {
                "role": "user",
                "content": "hello",
                "metadata": {"id": "first-turn"},
            },
            {
                "role": "assistant",
                "content": "hi there",
                "metadata": {"id": "assistant-turn"},
            },
        ]

        normalized = normalize_chat_messages(messages)

        self.assertEqual(
            normalized,
            [
                {"role": "system", "content": "system guidance"},
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi there"},
            ],
        )
        self.assertTrue(all(set(message.keys()) == {"role", "content"} for message in normalized))


if __name__ == "__main__":
    unittest.main()
