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


from care_agent import normalize_chat_messages


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
