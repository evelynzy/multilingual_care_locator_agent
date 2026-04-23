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


class CareLocatorAgentNPIRegistryTests(unittest.TestCase):
    def setUp(self) -> None:
        with patch.object(
            CareLocatorAgent,
            "_initialize_clinicaltables_field_maps",
            return_value=None,
        ):
            self.agent = CareLocatorAgent(provider_repository=Mock())
        self.agent._npi_registry_enabled = True

    def test_format_npi_registry_location_compacts_fields(self) -> None:
        address = {
            "address_1": "200 Lothrop St",
            "address_2": "Suite 123",
            "city": "Pittsburgh",
            "state": "PA",
            "postal_code": "15213-2582",
            "country_name": "United States",
        }

        location = self.agent._format_npi_registry_location(address)

        self.assertEqual(location, "200 Lothrop St, Suite 123, Pittsburgh, PA 15213-2582")

    @patch.object(CareLocatorAgent, "_check_medicare_opt_out", return_value=None)
    @patch.object(CareLocatorAgent, "_lookup_npi_registry_entry")
    def test_enhance_with_npi_registry_overrides_location_and_phone(
        self, mock_lookup: Mock, mock_opt_out: Mock
    ) -> None:
        mock_lookup.return_value = {
            "practice_address": {
                "address_1": "200 Lothrop St",
                "city": "Pittsburgh",
                "state": "PA",
                "postal_code": "15213-2582",
                "country_name": "United States",
                "telephone_number": "412-605-3019",
            },
            "mailing_address": {
                "telephone_number": "000-000-0000",
            },
            "taxonomies": [
                {"desc": "Urology"},
            ],
        }

        record = {
            "name": "Healthcare Provider",
            "location": "Old Address",
            "phone": "111-111-1111",
            "npi": "1619271780",
            "languages": [],
            "taxonomy": None,
            "source": "Test Source",
            "dataset": "npi_idv",
            "raw": {},
        }

        updated = self.agent._enhance_with_npi_registry("npi_idv", record)

        self.assertIs(updated, record)
        self.assertEqual(
            record["location"],
            "200 Lothrop St, Pittsburgh, PA 15213-2582",
        )
        self.assertEqual(record["phone"], "412-605-3019")
        self.assertEqual(record["taxonomy"], "Urology")
        self.assertIn("npi_registry", record["raw"])
        mock_lookup.assert_called_once_with("1619271780")
        mock_opt_out.assert_called_once_with("1619271780")


if __name__ == "__main__":
    unittest.main()
