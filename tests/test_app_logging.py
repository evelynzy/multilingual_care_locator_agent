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

if "datasets" not in sys.modules:
    datasets_stub = types.ModuleType("datasets")

    class _StubDataset(list):
        pass

    def _stub_load_dataset(*args, **kwargs):
        raise RuntimeError("datasets.load_dataset stubbed in tests")

    datasets_stub.Dataset = _StubDataset
    datasets_stub.load_dataset = _stub_load_dataset
    sys.modules["datasets"] = datasets_stub

if "llama_index" not in sys.modules:
    llama_index_stub = types.ModuleType("llama_index")
    llama_core_stub = types.ModuleType("llama_index.core")

    class _StubDocument:
        def __init__(self, text: str, metadata: Optional[dict] = None):
            self.text = text
            self.metadata = metadata or {}

    class _StubVectorStoreIndex:
        def __init__(self, documents=None, embed_model=None):
            self._documents = documents or []
            self._embed_model = embed_model

        @classmethod
        def from_documents(cls, documents, embed_model=None):
            return cls(documents=documents, embed_model=embed_model)

        def as_retriever(self, similarity_top_k: int = 5):
            class _StubRetriever:
                def retrieve(self_inner, query):
                    return []

            return _StubRetriever()

    llama_core_stub.Document = _StubDocument
    llama_core_stub.VectorStoreIndex = _StubVectorStoreIndex

    llama_embeddings_stub = types.ModuleType("llama_index.embeddings")
    llama_hf_stub = types.ModuleType("llama_index.embeddings.huggingface")

    class _StubHuggingFaceEmbedding:
        def __init__(self, model_name: str):
            self.model_name = model_name

    llama_hf_stub.HuggingFaceEmbedding = _StubHuggingFaceEmbedding

    sys.modules["llama_index"] = llama_index_stub
    sys.modules["llama_index.core"] = llama_core_stub
    sys.modules["llama_index.embeddings"] = llama_embeddings_stub
    sys.modules["llama_index.embeddings.huggingface"] = llama_hf_stub

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

    def test_examples_are_collapsed_markdown_not_chatinterface_examples(self) -> None:
        app = importlib.import_module("app")

        self.assertNotIn("examples", app.chatbot.kwargs)
        self.assertIn("primary care 75001", app.EXAMPLE_PROMPTS_MARKDOWN)
        self.assertIn("儿科 10013", app.EXAMPLE_PROMPTS_MARKDOWN)
        self.assertIn("dentista 33012", app.EXAMPLE_PROMPTS_MARKDOWN)


if __name__ == "__main__":
    unittest.main()
