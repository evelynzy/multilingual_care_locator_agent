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


class CareLocatorAgentMedicareOptOutTests(unittest.TestCase):
    def setUp(self) -> None:
        with patch.object(
            CareLocatorAgent,
            "_initialize_clinicaltables_field_maps",
            return_value=None,
        ):
            self.agent = CareLocatorAgent(provider_repository=Mock())

    @patch("care_agent.requests.get")
    def test_check_medicare_opt_out_returns_false_when_no_record_exists(
        self, mock_get: Mock
    ) -> None:
        response = Mock()
        response.json.return_value = []
        response.raise_for_status.return_value = None
        mock_get.return_value = response

        result = self.agent._check_medicare_opt_out("1619271780")

        self.assertEqual(result, {"opted_out": False})

    @patch("care_agent.requests.get")
    def test_check_medicare_opt_out_returns_true_for_open_ended_record(
        self, mock_get: Mock
    ) -> None:
        response = Mock()
        response.json.return_value = [
            {
                "NPI": "1619271780",
                "Optout Effective Date": "2025/01/01",
                "Optout End Date": "",
            }
        ]
        response.raise_for_status.return_value = None
        mock_get.return_value = response

        result = self.agent._check_medicare_opt_out("1619271780")

        self.assertEqual(
            result,
            {
                "opted_out": True,
                "optout_effective_date": "2025/01/01",
                "optout_end_date": "",
            },
        )

    @patch("care_agent.requests.get")
    def test_check_medicare_opt_out_returns_false_for_expired_record(
        self, mock_get: Mock
    ) -> None:
        response = Mock()
        response.json.return_value = [
            {
                "NPI": "1619271780",
                "Optout Effective Date": "2020/01/01",
                "Optout End Date": "2020/12/31",
            }
        ]
        response.raise_for_status.return_value = None
        mock_get.return_value = response

        result = self.agent._check_medicare_opt_out("1619271780")

        self.assertEqual(
            result,
            {
                "opted_out": False,
                "optout_effective_date": "2020/01/01",
                "optout_end_date": "2020/12/31",
            },
        )

    @patch("care_agent.requests.get")
    def test_check_medicare_opt_out_treats_unparseable_end_date_as_active(
        self, mock_get: Mock
    ) -> None:
        response = Mock()
        response.json.return_value = [
            {
                "NPI": "1619271780",
                "Optout Effective Date": "2025/01/01",
                "Optout End Date": "not-a-date",
            }
        ]
        response.raise_for_status.return_value = None
        mock_get.return_value = response

        result = self.agent._check_medicare_opt_out("1619271780")

        self.assertEqual(result["opted_out"], True)
        self.assertEqual(result["optout_end_date"], "not-a-date")

    @patch("care_agent.requests.get")
    def test_check_medicare_opt_out_returns_none_when_request_fails(
        self, mock_get: Mock
    ) -> None:
        mock_get.side_effect = RuntimeError("network down")

        result = self.agent._check_medicare_opt_out("1619271780")

        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
