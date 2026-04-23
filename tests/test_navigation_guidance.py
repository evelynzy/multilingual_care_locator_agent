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


from care_agent import CareLocatorAgent, ParsedCareQuery
from provider_search.models import ProviderSearchRequest, ProviderSearchResponse, SearchTrace


class _StubCompletionChoice:
    def __init__(self, content: str):
        self.message = {"content": content}


class _StubChatClient:
    def __init__(self, response_text: str = "ok"):
        self.response_text = response_text
        self.calls = []

    def chat_completion(self, messages, max_tokens, temperature, top_p):
        self.calls.append(messages)
        return type("Completion", (), {"choices": [_StubCompletionChoice(self.response_text)]})()


class CareNavigationGuidanceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.agent = CareLocatorAgent(provider_search_service=Mock())

        self.agent.provider_search_service.search.return_value = ProviderSearchResponse(
            request=ProviderSearchRequest(),
            search_trace=SearchTrace(),
        )

    def test_handle_request_asks_for_location_and_care_need_when_request_is_vague(self) -> None:
        query = ParsedCareQuery(
            detected_language="English",
            response_language="English",
            summary="find a doctor",
            medical_need=True,
            location=None,
            specialties=[],
            insurance=[],
            preferred_languages=[],
            keywords=[],
            patient_context=None,
        )

        captured: dict = {}

        def _capture_response(client, payload, max_tokens, temperature, top_p, template_key="response_template"):
            captured["payload"] = payload
            captured["template_key"] = template_key
            return "ok"

        with patch.object(self.agent, "_interpret_user_need", return_value=query), patch.object(
            self.agent,
            "_compose_response",
            side_effect=_capture_response,
        ):
            result = self.agent.handle_request(
                Mock(),
                "find a doctor near me",
                [],
                max_tokens=256,
                temperature=0.2,
                top_p=0.9,
            )

        self.assertEqual(result, "ok")
        self.assertEqual(captured["template_key"], "response_template_clarification_needed")
        self.assertIn("What city and state or ZIP code should I search?", captured["payload"]["follow_up_questions"])
        self.assertIn(
            "What kind of care do you need",
            captured["payload"]["follow_up_questions"][1],
        )
        self.assertNotIn("care_setting_guidance", captured["payload"])
        self.agent.provider_search_service.search.assert_not_called()

    def test_handle_request_adds_referral_guidance_for_specialist_search(self) -> None:
        query = ParsedCareQuery(
            detected_language="English",
            response_language="English",
            summary="need a neurologist referral",
            medical_need=True,
            location=None,
            specialties=["Neurology"],
            insurance=[],
            preferred_languages=[],
            keywords=[],
            patient_context=None,
        )

        captured: dict = {}

        def _capture_response(client, payload, max_tokens, temperature, top_p, template_key="response_template"):
            captured["payload"] = payload
            captured["template_key"] = template_key
            return "ok"

        with patch.object(self.agent, "_interpret_user_need", return_value=query), patch.object(
            self.agent,
            "_compose_response",
            side_effect=_capture_response,
        ):
            result = self.agent.handle_request(
                Mock(),
                "need a neurologist referral",
                [],
                max_tokens=256,
                temperature=0.2,
                top_p=0.9,
            )

        self.assertEqual(result, "ok")
        self.assertEqual(captured["template_key"], "response_template_clarification_needed")
        self.assertIn(
            "For specialist searches, HMO and POS plans often require a PCP referral",
            captured["payload"]["specialist_plan_guidance"],
        )
        self.assertTrue(
            any("HMO, PPO, or POS" in question for question in captured["payload"]["follow_up_questions"])
        )
        self.assertIn("What city and state or ZIP code should I search?", captured["payload"]["follow_up_questions"])

    def test_handle_request_uses_emergency_template_for_severe_symptoms(self) -> None:
        query = ParsedCareQuery(
            detected_language="English",
            response_language="English",
            summary="chest pain",
            medical_need=True,
            location="San Francisco CA",
            specialties=[],
            insurance=[],
            preferred_languages=[],
            keywords=[],
            patient_context=None,
        )

        captured: dict = {}

        def _capture_response(client, payload, max_tokens, temperature, top_p, template_key="response_template"):
            captured["payload"] = payload
            captured["template_key"] = template_key
            return "ok"

        with patch.object(self.agent, "_interpret_user_need", return_value=query), patch.object(
            self.agent,
            "_compose_response",
            side_effect=_capture_response,
        ):
            result = self.agent.handle_request(
                Mock(),
                "I have chest pain and trouble breathing",
                [],
                max_tokens=256,
                temperature=0.2,
                top_p=0.9,
            )

        self.assertEqual(result, "ok")
        self.assertEqual(captured["template_key"], "response_template_emergency")
        self.assertIn("emergency services", captured["payload"]["emergency_guidance"])
        self.agent.provider_search_service.search.assert_not_called()

    def test_handle_request_uses_emergency_template_for_standalone_911(self) -> None:
        query = ParsedCareQuery(
            detected_language="English",
            response_language="English",
            summary="call 911",
            medical_need=True,
            location=None,
            specialties=[],
            insurance=[],
            preferred_languages=[],
            keywords=[],
            patient_context=None,
        )

        captured: dict = {}

        def _capture_response(client, payload, max_tokens, temperature, top_p, template_key="response_template"):
            captured["payload"] = payload
            captured["template_key"] = template_key
            return "ok"

        with patch.object(self.agent, "_interpret_user_need", return_value=query), patch.object(
            self.agent,
            "_compose_response",
            side_effect=_capture_response,
        ):
            result = self.agent.handle_request(
                Mock(),
                "call 911",
                [],
                max_tokens=256,
                temperature=0.2,
                top_p=0.9,
            )

        self.assertEqual(result, "ok")
        self.assertEqual(captured["template_key"], "response_template_emergency")

    def test_handle_request_uses_emergency_template_when_parser_marks_non_medical(self) -> None:
        query = ParsedCareQuery(
            detected_language="English",
            response_language="English",
            summary="general question",
            medical_need=False,
            location=None,
            specialties=[],
            insurance=[],
            preferred_languages=[],
            keywords=[],
            patient_context=None,
        )

        captured: dict = {}

        def _capture_response(client, payload, max_tokens, temperature, top_p, template_key="response_template"):
            captured["payload"] = payload
            captured["template_key"] = template_key
            return "ok"

        with patch.object(self.agent, "_interpret_user_need", return_value=query), patch.object(
            self.agent,
            "_compose_response",
            side_effect=_capture_response,
        ):
            result = self.agent.handle_request(
                Mock(),
                "I have chest pain and cannot breathe",
                [],
                max_tokens=256,
                temperature=0.2,
                top_p=0.9,
            )

        self.assertEqual(result, "ok")
        self.assertEqual(captured["template_key"], "response_template_emergency")
        self.assertTrue(captured["payload"]["query"]["medical_need"])
        self.assertIn("emergency services", captured["payload"]["emergency_guidance"])
        self.agent.provider_search_service.search.assert_not_called()

    def test_handle_request_does_not_treat_zip_91101_as_emergency(self) -> None:
        query = ParsedCareQuery(
            detected_language="English",
            response_language="English",
            summary="primary care near 91101",
            medical_need=True,
            location="Boston 91101",
            specialties=["Primary Care"],
            insurance=[],
            preferred_languages=[],
            keywords=[],
            patient_context=None,
        )

        captured: dict = {}

        def _capture_response(client, payload, max_tokens, temperature, top_p, template_key="response_template"):
            captured["payload"] = payload
            captured["template_key"] = template_key
            return "ok"

        with patch.object(self.agent, "_interpret_user_need", return_value=query), patch.object(
            self.agent,
            "_compose_response",
            side_effect=_capture_response,
        ):
            result = self.agent.handle_request(
                Mock(),
                "primary care 91101",
                [],
                max_tokens=256,
                temperature=0.2,
                top_p=0.9,
            )

        self.assertEqual(result, "ok")
        self.assertNotEqual(captured["template_key"], "response_template_emergency")

    def test_care_setting_classifier_does_not_match_short_patterns_inside_words(self) -> None:
        query = ParsedCareQuery(
            detected_language="English",
            response_language="English",
            summary="patient needs help with recent symptoms",
            medical_need=True,
            location=None,
            specialties=[],
            insurance=[],
            preferred_languages=[],
            keywords=[],
            patient_context=None,
        )

        self.assertEqual(
            self.agent._classify_care_setting(
                query,
                "my patient needs help with recent symptoms",
            ),
            "unclear",
        )
        self.assertEqual(
            self.agent._classify_care_setting(query, "I need an ENT near Austin"),
            "specialist",
        )

    def test_specialized_prompt_templates_are_composed_not_fallbacked(self) -> None:
        payload = {
            "query": {
                "response_language": "English",
                "medical_need": True,
                "summary": "need follow-up help",
            },
            "follow_up_questions": ["What city and state or ZIP code should I search?"],
            "care_setting_guidance": "For routine or ongoing care, primary care is usually the best fit.",
            "specialist_plan_guidance": "For specialist searches, HMO and POS plans often require a PCP referral; PPO plans may not, but you should confirm the rule with your insurer and plan documents.",
            "local_results": [],
            "fallback_results": [],
            "verification_guidance": "Call the provider and insurer to confirm network status.",
        }

        client = _StubChatClient()

        self.agent._compose_response(
            client,
            payload,
            max_tokens=128,
            temperature=0.1,
            top_p=0.9,
            template_key="response_template_clarification_needed",
        )

        clarification_prompt = client.calls[-1][1]["content"]
        self.assertIn("follow_up_questions", clarification_prompt)
        self.assertIn("specialist_plan_guidance", clarification_prompt)
        self.assertIn("care_setting_guidance", clarification_prompt)

        client = _StubChatClient()
        emergency_payload = dict(payload)
        emergency_payload["emergency_guidance"] = (
            "If symptoms are severe or life-threatening, call emergency services now or go to the nearest emergency room."
        )
        self.agent._compose_response(
            client,
            emergency_payload,
            max_tokens=128,
            temperature=0.1,
            top_p=0.9,
            template_key="response_template_emergency",
        )
        emergency_prompt = client.calls[-1][1]["content"]
        self.assertIn("Do not ask follow-up questions", emergency_prompt)
        self.assertIn("emergency services now or go to the nearest emergency room", emergency_prompt)


if __name__ == "__main__":
    unittest.main()
