import importlib
import os
import sys
import types
import unittest
from typing import Optional
from unittest.mock import Mock, patch


if "huggingface_hub" not in sys.modules:
    huggingface_stub = types.ModuleType("huggingface_hub")

    class _StubInferenceClient:
        def __init__(self, *args, **kwargs):
            pass

    huggingface_stub.InferenceClient = _StubInferenceClient
    sys.modules["huggingface_hub"] = huggingface_stub

if "requests" not in sys.modules:
    requests_stub = types.ModuleType("requests")

    def _stub_get(*args, **kwargs):
        raise RuntimeError("requests.get stub should be patched in tests")

    requests_stub.get = _stub_get
    sys.modules["requests"] = requests_stub

if "dotenv" not in sys.modules:
    dotenv_stub = types.ModuleType("dotenv")
    dotenv_stub.load_dotenv = lambda *args, **kwargs: None
    sys.modules["dotenv"] = dotenv_stub

if "gradio" not in sys.modules:
    gradio_stub = types.ModuleType("gradio")

    class _StubChatInterface:
        instances = []

        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs
            self.__class__.instances.append(self)

        def render(self):
            return None

    class _StubBlocks:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def launch(self):
            return None

    class _StubAccordion:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

    gradio_stub.ChatInterface = _StubChatInterface
    gradio_stub.Blocks = _StubBlocks
    gradio_stub.Accordion = _StubAccordion
    gradio_stub.Markdown = lambda *args, **kwargs: object()
    sys.modules["gradio"] = gradio_stub


class AppLoggingTests(unittest.TestCase):
    def test_respond_logs_message_length_without_raw_user_text(self) -> None:
        app = importlib.import_module("app")

        with patch.dict(os.environ, {"HF_TOKEN": "test-token"}), patch.object(
            app.care_locator_agent,
            "handle_request",
            return_value="ok",
        ), patch.object(app, "InferenceClient", return_value=Mock()), self.assertLogs(
            "app",
            level="INFO",
        ) as captured_logs:
            response = app.respond("Jane Doe at 123 Main St needs care", [])

        self.assertEqual(response, "ok")
        log_output = "\n".join(captured_logs.output)
        self.assertIn("message_length=", log_output)
        self.assertNotIn("Jane Doe", log_output)
        self.assertNotIn("123 Main St", log_output)

    def test_ui_notes_are_collapsible_panel_ready_and_chat_area_is_larger(self) -> None:
        app = importlib.import_module("app")

        self.assertIn("Directory matches are informational", app.SAFETY_TRUST_NOTES)
        self.assertIn("Do not share PHI", app.SAFETY_TRUST_NOTES)
        self.assertIn("NPI Records - Individuals", app.DATA_SOURCE_LIMITATIONS_NOTES)
        self.assertIn("Public sources may be incomplete", app.DATA_SOURCE_LIMITATIONS_NOTES)
        self.assertIn("height: calc(100vh - 220px)", app.custom_css)
        self.assertIn("min-height: 680px", app.custom_css)
        self.assertIn(".provider-card", app.custom_css)
        self.assertIn(".provider-card__badge", app.custom_css)
        self.assertIn("border-radius: 8px", app.custom_css)

    def test_examples_are_compact_clickable_chatinterface_examples(self) -> None:
        app = importlib.import_module("app")

        self.assertEqual(
            app.chatbot.kwargs["examples"],
            ["primary care 75001", "儿科10013", "dentista 33012"],
        )
        self.assertNotIn("儿科 10013", app.chatbot.kwargs["examples"])


class AppErrorReplyTests(unittest.TestCase):
    def test_error_reply_omits_exception_details(self) -> None:
        app = importlib.import_module("app")

        secret = "SECRET boom at /Users/private/thing.py:42"
        with patch.dict(os.environ, {"HF_TOKEN": "test-token"}), patch.object(
            app.care_locator_agent,
            "handle_request",
            side_effect=RuntimeError(secret),
        ), patch.object(app, "InferenceClient", return_value=Mock()):
            reply = app.respond("primary care 94110", [])

        self.assertIn(app.ERROR_MESSAGE, reply)
        self.assertNotIn(secret, reply)
        self.assertNotIn("Details for debugging", reply)


if __name__ == "__main__":
    unittest.main()
