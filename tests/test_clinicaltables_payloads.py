import sys
import types
import unittest
from typing import Optional
from unittest.mock import Mock, patch


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


from care_agent import CareLocatorAgent


class CareLocatorAgentClinicalTablesPayloadTests(unittest.TestCase):
    def setUp(self) -> None:
        with patch.object(
            CareLocatorAgent,
            "_initialize_clinicaltables_field_maps",
            return_value=None,
        ):
            self.agent = CareLocatorAgent(provider_repository=Mock())

        self.agent._ctss_field_map["npi_idv"] = {
            "name.full": 0,
            "NPI": 1,
            "provider_type": 2,
        }
        self.agent._ctss_result_field_order["npi_idv"] = [
            "name.full",
            "NPI",
            "provider_type",
        ]

    def test_parse_clinicaltables_fields_payload_supports_headerless_rows(self) -> None:
        payload = [
            [0, "name.full", "Provider Name"],
            [1, "NPI", "NPI"],
            [2, "provider_type", "Taxonomy"],
        ]

        mapping = self.agent._parse_clinicaltables_fields_payload(payload)

        self.assertEqual(
            mapping,
            {
                "name.full": 0,
                "NPI": 1,
                "provider_type": 2,
            },
        )

    def test_parse_clinicaltables_fields_payload_skips_invalid_entries(self) -> None:
        payload = [
            "ignored heading",
            ["not-an-index", "name.full"],
            [1, ""],
            [2, "provider_type"],
        ]

        mapping = self.agent._parse_clinicaltables_fields_payload(payload)

        self.assertEqual(mapping, {"name.full": 0, "provider_type": 2})

    def test_parse_clinicaltables_payload_uses_string_fields_from_search_payload(self) -> None:
        payload = [
            1,
            ["display row"],
            ["name.full", "NPI", "provider_type"],
            [["Harmony Family Clinic", "1619271780", "Family Medicine"]],
        ]

        fields, entries = self.agent._parse_clinicaltables_payload("npi_idv", payload)

        self.assertEqual(fields, ["name.full", "NPI", "provider_type"])
        self.assertEqual(
            entries,
            [["Harmony Family Clinic", "1619271780", "Family Medicine"]],
        )

    def test_parse_clinicaltables_payload_resolves_integer_field_indexes(self) -> None:
        payload = [
            1,
            ["display row"],
            [0, 1, 2],
            [["Harmony Family Clinic", "1619271780", "Family Medicine"]],
        ]

        fields, entries = self.agent._parse_clinicaltables_payload("npi_idv", payload)

        self.assertEqual(fields, ["name.full", "NPI", "provider_type"])
        self.assertEqual(len(entries), 1)

    def test_parse_clinicaltables_payload_falls_back_to_configured_field_order(self) -> None:
        payload = [
            1,
            ["display row"],
            ["name.full", 1, {"unexpected": "value"}],
            [["Harmony Family Clinic", "1619271780", "Family Medicine"], "skip-me"],
        ]

        fields, entries = self.agent._parse_clinicaltables_payload("npi_idv", payload)

        self.assertEqual(fields, ["name.full", "NPI", "provider_type"])
        self.assertEqual(
            entries,
            [["Harmony Family Clinic", "1619271780", "Family Medicine"]],
        )


if __name__ == "__main__":
    unittest.main()
