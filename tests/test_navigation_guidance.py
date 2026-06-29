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

        def _capture_result_cards(payload):
            captured["card_payload"] = payload
            return "cards"

        with patch.object(self.agent, "_interpret_user_need", return_value=query), patch.object(
            self.agent,
            "_compose_response",
            side_effect=_capture_response,
        ), patch.object(
            self.agent,
            "_compose_result_card_response",
            side_effect=_capture_result_cards,
        ):
            result = self.agent.handle_request(
                Mock(),
                "primary care 91101",
                [],
                max_tokens=256,
                temperature=0.2,
                top_p=0.9,
            )

        self.assertIn(result, {"ok", "cards"})
        if result == "ok":
            self.assertNotEqual(captured["template_key"], "response_template_emergency")
        else:
            self.assertNotIn("emergency_guidance", captured["card_payload"])
            self.assertEqual(len(captured["card_payload"]["fallback_results"]), 1)
            self.assertEqual(
                captured["card_payload"]["fallback_results"][0]["name"],
                "Medicare Care Compare",
            )

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

    def test_care_setting_classifier_treats_bare_specialist_without_specific_specialty_as_unclear(
        self,
    ) -> None:
        query = ParsedCareQuery(
            detected_language="English",
            response_language="English",
            summary="specialist",
            medical_need=True,
            location=None,
            specialties=[],
            insurance=[],
            preferred_languages=[],
            keywords=[],
            patient_context=None,
        )

        self.assertFalse(self.agent._has_clear_care_need(query, "specialist near me"))
        self.assertEqual(
            self.agent._classify_care_setting(query, "specialist near me"),
            "unclear",
        )

    def test_handle_request_bare_specialist_near_me_asks_location_and_specialty_clarification(
        self,
    ) -> None:
        query = ParsedCareQuery(
            detected_language="English",
            response_language="English",
            summary="specialist",
            medical_need=True,
            location=None,
            specialties=[],
            insurance=[],
            preferred_languages=[],
            keywords=[],
            patient_context=None,
            care_setting="specialist",
            needs_clarification=False,
            follow_up_focus=[],
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
                "specialist near me",
                [],
                max_tokens=256,
                temperature=0.2,
                top_p=0.9,
            )

        self.assertEqual(result, "ok")
        self.assertEqual(captured["template_key"], "response_template_clarification_needed")
        self.assertEqual(
            captured["payload"]["follow_up_questions"],
            [
                "What city and state or ZIP code should I search?",
                "What kind of care do you need (for example primary care, pediatrics, dermatology, ENT, or urgent care)?",
            ],
        )
        self.assertNotIn("care_setting_guidance", captured["payload"])
        self.agent.provider_search_service.search.assert_not_called()

    def test_build_navigation_guidance_asks_specialty_follow_up_for_ambiguous_specialty_intent(
        self,
    ) -> None:
        query = ParsedCareQuery(
            detected_language="English",
            response_language="English",
            summary="ob gyn and cardiology 95051",
            medical_need=True,
            location="95051",
            specialties=[],
            insurance=[],
            preferred_languages=[],
            keywords=[],
            patient_context=None,
            needs_clarification=True,
            follow_up_focus=["specialty clarification"],
        )

        guidance = self.agent._build_navigation_guidance(
            query,
            "ob gyn and cardiology 95051",
        )

        self.assertEqual(guidance["mode"], "clarification")
        self.assertIn(
            "What kind of care do you need (for example primary care, pediatrics, dermatology, ENT, or urgent care)?",
            guidance["follow_up_questions"],
        )
        self.assertIsNone(guidance["care_setting_guidance"])
        self.assertFalse(guidance["location_only"])

    def test_handle_request_abstains_for_child_allergy_when_model_picks_pediatrics(
        self,
    ) -> None:
        captured: dict = {}

        def _capture_response(client, payload, max_tokens, temperature, top_p, template_key="response_template"):
            captured["payload"] = payload
            captured["template_key"] = template_key
            return "ok"

        with patch.object(
            self.agent,
            "_compose_response",
            side_effect=_capture_response,
        ):
            result = self.agent.handle_request(
                _StubChatClient(
                    (
                        '{"detected_language":"English","response_language":"English","summary":"child allergy 95051",'
                        '"medical_need":true,"location":"95051","specialties":["Pediatrics"],"insurance":[],'
                        '"preferred_languages":[],"keywords":[],"patient_context":null,'
                        '"care_setting":"specialist","urgency":null,"needs_clarification":false,'
                        '"follow_up_focus":[]}'
                    )
                ),
                "child allergy 95051",
                [],
                max_tokens=256,
                temperature=0.2,
                top_p=0.9,
            )

        self.assertEqual(result, "ok")
        self.assertEqual(captured["template_key"], "response_template_clarification_needed")
        self.assertEqual(captured["payload"]["query"]["specialties"], [])
        self.assertTrue(captured["payload"]["query"]["needs_clarification"])
        self.assertEqual(
            captured["payload"]["query"]["follow_up_focus"],
            ["specialty clarification"],
        )
        self.assertEqual(
            captured["payload"]["follow_up_questions"],
            ["What kind of care do you need (for example primary care, pediatrics, dermatology, ENT, or urgent care)?"],
        )
        self.agent.provider_search_service.search.assert_not_called()

    def test_handle_request_rejects_invented_bare_95051_location_without_raw_zip_evidence(
        self,
    ) -> None:
        captured: dict = {}

        def _capture_response(client, payload, max_tokens, temperature, top_p, template_key="response_template"):
            captured["payload"] = payload
            captured["template_key"] = template_key
            return "ok"

        with patch.object(
            self.agent,
            "_compose_response",
            side_effect=_capture_response,
        ):
            result = self.agent.handle_request(
                _StubChatClient(
                    (
                        '{"detected_language":"English","response_language":"English","summary":"ob gyn",'
                        '"medical_need":true,"location":"95051","specialties":["OB/GYN"],"insurance":[],'
                        '"preferred_languages":[],"keywords":[],"patient_context":null,'
                        '"care_setting":"specialist","urgency":null,"needs_clarification":false,'
                        '"follow_up_focus":[]}'
                    )
                ),
                "ob gyn",
                [],
                max_tokens=256,
                temperature=0.2,
                top_p=0.9,
            )

        self.assertEqual(result, "ok")
        self.assertEqual(captured["template_key"], "response_template_location_needed")
        self.assertEqual(captured["payload"]["query"]["location"], None)
        self.assertEqual(
            captured["payload"]["follow_up_questions"],
            ["What city and state or ZIP code should I search?"],
        )
        self.agent.provider_search_service.search.assert_not_called()

    def test_handle_request_raw_message_without_explicit_cpt_strips_invented_procedure_gloss_from_query_payload(
        self,
    ) -> None:
        captured: dict = {}

        def _capture_result_cards(payload):
            captured["payload"] = payload
            return "cards"

        with patch.object(
            self.agent,
            "_compose_result_card_response",
            side_effect=_capture_result_cards,
        ):
            result = self.agent.handle_request(
                _StubChatClient(
                    (
                        '{"detected_language":"English","response_language":"English","summary":"CPT 95051 ob gyn",'
                        '"medical_need":true,"location":null,"specialties":["OB/GYN"],"insurance":[],'
                        '"preferred_languages":[],"keywords":["cpt"],"patient_context":null,'
                        '"care_setting":"specialist","urgency":null,"needs_clarification":false,'
                        '"follow_up_focus":["procedure code"]}'
                    )
                ),
                "ob gyn 95051",
                [],
                max_tokens=256,
                temperature=0.2,
                top_p=0.9,
            )

        self.assertEqual(result, "cards")
        self.assertEqual(captured["payload"]["query"]["summary"], "ob gyn 95051")
        self.assertEqual(captured["payload"]["query"]["location"], "95051")
        self.assertEqual(captured["payload"]["query"]["keywords"], [])
        self.assertEqual(captured["payload"]["query"]["follow_up_focus"], [])

    def test_handle_request_restores_real_raw_city_state_when_model_hallucinates_bare_95051(
        self,
    ) -> None:
        captured: dict = {}

        def _capture_result_cards(payload):
            captured["payload"] = payload
            return "cards"

        with patch.object(
            self.agent,
            "_compose_result_card_response",
            side_effect=_capture_result_cards,
        ):
            result = self.agent.handle_request(
                _StubChatClient(
                    (
                        '{"detected_language":"English","response_language":"English","summary":"ob gyn Austin TX",'
                        '"medical_need":true,"location":"95051","specialties":["OB/GYN"],"insurance":[],'
                        '"preferred_languages":[],"keywords":[],"patient_context":null,'
                        '"care_setting":"specialist","urgency":null,"needs_clarification":false,'
                        '"follow_up_focus":[]}'
                    )
                ),
                "ob gyn Austin, TX",
                [],
                max_tokens=256,
                temperature=0.2,
                top_p=0.9,
            )

        self.assertEqual(result, "cards")
        self.assertEqual(captured["payload"]["query"]["location"], "Austin, TX")
        self.assertEqual(captured["payload"]["local_results"], [])
        self.assertEqual(len(captured["payload"]["fallback_results"]), 1)
        self.assertEqual(
            captured["payload"]["fallback_results"][0]["name"],
            "Medicare Care Compare",
        )
        self.assertEqual(
            captured["payload"]["fallback_results"][0]["location"],
            "Austin, TX",
        )

    def test_is_location_only_follow_up_turn_accepts_location_with_specialist_care_setting_echo(
        self,
    ) -> None:
        query = ParsedCareQuery(
            detected_language="English",
            response_language="English",
            summary="95051",
            medical_need=True,
            location="95051",
            specialties=[],
            insurance=[],
            preferred_languages=[],
            keywords=[],
            patient_context=None,
            care_setting="specialist",
            needs_clarification=True,
            follow_up_focus=["specialty clarification"],
        )

        self.assertTrue(self.agent._is_location_only_follow_up_turn(query))

    def test_build_navigation_guidance_explicit_cpt_preserves_trusted_city_state_zip(self) -> None:
        query = ParsedCareQuery(
            detected_language="English",
            response_language="English",
            summary="CPT 95051 ob gyn Austin TX 78701",
            medical_need=True,
            location="Austin, TX 78701",
            specialties=["OB/GYN"],
            insurance=[],
            preferred_languages=[],
            keywords=["cpt"],
            patient_context=None,
            needs_clarification=False,
            follow_up_focus=["procedure code"],
        )

        guidance = self.agent._build_navigation_guidance(
            query,
            "CPT 95051 ob gyn Austin, TX 78701",
        )

        self.assertEqual(guidance["mode"], "search")
        self.assertEqual(guidance["follow_up_questions"], [])
        self.assertFalse(guidance["location_only"])

    def test_build_navigation_guidance_explicit_cpt_does_not_treat_trailing_procedure_code_as_zip(
        self,
    ) -> None:
        query = ParsedCareQuery(
            detected_language="English",
            response_language="English",
            summary="Austin TX CPT 95051 ob gyn",
            medical_need=True,
            location="Austin, TX",
            specialties=["OB/GYN"],
            insurance=[],
            preferred_languages=[],
            keywords=["cpt"],
            patient_context=None,
            needs_clarification=False,
            follow_up_focus=["procedure code"],
        )

        guidance = self.agent._build_navigation_guidance(
            query,
            "Austin, TX CPT 95051 ob gyn",
        )

        self.assertEqual(guidance["mode"], "search")
        self.assertEqual(guidance["follow_up_questions"], [])
        self.assertFalse(guidance["location_only"])

    def test_build_navigation_guidance_explicit_cpt_rejects_invented_city_without_raw_location(
        self,
    ) -> None:
        query = ParsedCareQuery(
            detected_language="English",
            response_language="English",
            summary="CPT 95051 ob gyn Austin TX",
            medical_need=True,
            location="Austin, TX",
            specialties=["OB/GYN"],
            insurance=[],
            preferred_languages=[],
            keywords=["cpt"],
            patient_context=None,
            needs_clarification=False,
            follow_up_focus=["procedure code"],
        )

        guidance = self.agent._build_navigation_guidance(
            query,
            "CPT 95051 ob gyn",
        )

        self.assertEqual(guidance["mode"], "clarification")
        self.assertEqual(
            guidance["follow_up_questions"],
            ["What city and state or ZIP code should I search?"],
        )
        self.assertTrue(guidance["location_only"])

    def test_build_navigation_guidance_treats_urgent_care_as_route_context_not_competing_specialty(
        self,
    ) -> None:
        query = ParsedCareQuery(
            detected_language="English",
            response_language="English",
            summary="ENT near Austin urgent care open now",
            medical_need=True,
            location="Austin, TX",
            specialties=["ENT"],
            insurance=[],
            preferred_languages=[],
            keywords=[],
            patient_context=None,
            needs_clarification=False,
            follow_up_focus=[],
        )

        guidance = self.agent._build_navigation_guidance(
            query,
            "ENT near Austin urgent care open now",
        )

        self.assertEqual(guidance["mode"], "search")
        self.assertEqual(guidance["follow_up_questions"], [])
        self.assertIn("urgent care is usually the best fit", guidance["care_setting_guidance"])

    def test_build_navigation_guidance_specialty_follow_up_prefers_specialist_route_over_pcp(
        self,
    ) -> None:
        query = ParsedCareQuery(
            detected_language="English",
            response_language="English",
            summary="dermatology follow-up in Pittsburgh",
            medical_need=True,
            location="Pittsburgh, PA",
            specialties=["Dermatology"],
            insurance=[],
            preferred_languages=[],
            keywords=[],
            patient_context=None,
            needs_clarification=False,
            follow_up_focus=[],
        )

        guidance = self.agent._build_navigation_guidance(
            query,
            "dermatology follow-up in Pittsburgh, PA",
        )

        self.assertEqual(guidance["mode"], "search")
        self.assertEqual(guidance["follow_up_questions"], [])
        self.assertIn("specialist is usually the right route", guidance["care_setting_guidance"])
        self.assertNotIn("primary care is usually the best fit", guidance["care_setting_guidance"])

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

    def test_compose_response_uses_deterministic_clarification_for_non_explicit_numeric_summary(
        self,
    ) -> None:
        payload = {
            "query": {
                "response_language": "English",
                "detected_language": "English",
                "summary": "pain 95051",
                "medical_need": True,
                "location": "95051",
                "specialties": [],
                "keywords": [],
                "follow_up_focus": [],
            },
            "follow_up_questions": [
                "What kind of care do you need (for example primary care, pediatrics, dermatology, ENT, or urgent care)?"
            ],
            "local_results": [],
            "fallback_results": [],
            "verification_guidance": "Call the provider and insurer to confirm network status.",
        }

        client = _StubChatClient("This sounds like CPT 95051 procedure code guidance.")

        response = self.agent._compose_response(
            client,
            payload,
            max_tokens=128,
            temperature=0.1,
            top_p=0.9,
            template_key="response_template_clarification_needed",
        )

        self.assertEqual(len(client.calls), 0)
        self.assertIn(
            "What kind of care do you need (for example primary care, pediatrics, dermatology, ENT, or urgent care)?",
            response,
        )
        self.assertNotIn("CPT 95051", response)
        self.assertNotIn("procedure code", response.lower())


if __name__ == "__main__":
    unittest.main()
