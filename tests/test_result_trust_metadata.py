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


from care_agent import (
    CareLocatorAgent,
    ParsedCareQuery,
    _TRUST_GUIDANCE_PROTECTED_TOKENS,
)
from retriever import ProviderRecord, ProviderRepository


class _SequencedChatClient:
    def __init__(self, response_texts, fail_on_call: Optional[int] = None):
        if isinstance(response_texts, str):
            response_texts = [response_texts]
        self.response_texts = list(response_texts)
        self.fail_on_call = fail_on_call
        self.calls = []

    def chat_completion(self, messages, max_tokens, temperature, top_p):
        self.calls.append(
            {
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
            }
        )
        call_number = len(self.calls)
        if self.fail_on_call == call_number:
            raise RuntimeError("translation unavailable")

        response_index = min(call_number - 1, len(self.response_texts) - 1)
        response_text = self.response_texts[response_index]
        return type(
            "Completion",
            (),
            {
                "choices": [
                    type(
                        "Choice",
                        (),
                        {
                            "message": {"content": response_text},
                            "finish_reason": "stop",
                        },
                    )()
                ]
            },
        )()


def _valid_french_translated_trust_guidance() -> str:
    tokens = _TRUST_GUIDANCE_PROTECTED_TOKENS
    return (
        "Notes importantes de sécurité et de confiance :\n"
        f"- {tokens['scope']} Cet outil sert uniquement à orienter les personnes dans la recherche de soins et ne pose pas de diagnostic, ne prescrit pas de traitement et ne remplace pas les conseils d'un professionnel de santé autorisé {tokens['advice']}.\n"
        f"- {tokens['directory']} Les résultats du répertoire sont fournis à titre informatif seulement; ils ne constituent pas des recommandations, des approbations ni des garanties d'adéquation clinique.\n"
        f"- {tokens['insurance']} La participation au réseau d'assurance, les exigences de recommandation, l'accueil de nouveaux patients, le lieu et les disponibilités de rendez-vous ne sont pas vérifiés sauf si la source l'indique explicitement. Appelez le prestataire et l'assureur pour confirmer avant de chercher des soins {tokens['confirm']}.\n"
        f"- {tokens['privacy']} Ne partagez pas d'informations personnelles de santé comme le nom complet, l'adresse, le numéro de sécurité sociale ou le numéro de dossier médical.\n"
        f"- {tokens['emergency']} Si les symptômes sont graves ou mettent la vie en danger, appelez les services d'urgence (911 aux États-Unis) ou allez aux urgences les plus proches."
    )


class CareLocatorAgentResultTrustMetadataTests(unittest.TestCase):
    def setUp(self) -> None:
        with patch.object(
            CareLocatorAgent,
            "_initialize_clinicaltables_field_maps",
            return_value=None,
        ):
            self.agent = CareLocatorAgent(provider_repository=Mock())

        self.agent.provider_repository.load_error = None
        self.agent.provider_repository.search.return_value = [
            {
                "id": "provider_001",
                "name": "Harmony Family Clinic",
                "insurance": ["Medicare", "Aetna"],
                "source": "Local provider dataset",
                "location": "San Francisco, CA",
            }
        ]

    @staticmethod
    def _client_with_response(response_text: str):
        return _SequencedChatClient(response_text)

    def test_handle_request_normalizes_local_provider_trust_metadata(self) -> None:
        query = ParsedCareQuery(
            detected_language="English",
            response_language="English",
            summary="primary care in San Francisco",
            medical_need=True,
            location="San Francisco, CA",
            specialties=["Primary Care"],
            insurance=["Medicare"],
            preferred_languages=["English"],
            keywords=[],
            patient_context=None,
        )

        captured: dict = {}

        def _capture_response(client, payload, max_tokens, temperature, top_p, template_key="response_template"):
            captured["payload"] = payload
            captured["template_key"] = template_key
            return "ok"

        with patch.object(
            self.agent,
            "_interpret_user_need",
            return_value=query,
        ), patch.object(
            self.agent,
            "_compose_response",
            side_effect=_capture_response,
        ):
            result = self.agent.handle_request(
                Mock(),
                "primary care 94110",
                [],
                max_tokens=256,
                temperature=0.2,
                top_p=0.9,
            )

        self.assertEqual(result, "ok")
        payload = captured["payload"]
        local_result = payload["local_results"][0]

        self.assertEqual(local_result["insurance_reported"], ["Medicare", "Aetna"])
        self.assertNotIn("insurance", local_result)
        self.assertNotIn("accepted_insurance", local_result)
        self.assertEqual(local_result["insurance_network_verification"]["status"], "unverified")
        self.assertFalse(local_result["insurance_network_verification"]["verified"])
        self.assertEqual(local_result["accepting_new_patients_status"]["status"], "unknown")
        self.assertFalse(local_result["accepting_new_patients_status"]["verified"])
        self.assertEqual(local_result["provenance"]["source"], "Local provider dataset")
        self.assertIn("verification_guidance", payload)

    def test_provider_record_uses_reported_insurance_metadata(self) -> None:
        record = ProviderRecord(
            id="provider_001",
            name="Harmony Family Clinic",
            specialties=["Primary Care"],
            languages=["English", "Spanish"],
            accepted_insurance=["Medicare", "Aetna"],
            address="123 Valencia St",
            city="San Francisco",
            state="CA",
            country="USA",
            phone="+1-415-555-0100",
            website="https://example.com",
            telehealth=True,
            description="Community-focused care",
        )

        data = record.to_dict()
        document = record.as_document()

        self.assertEqual(data["insurance_reported"], ["Medicare", "Aetna"])
        self.assertEqual(data["insurance_network_verification"]["status"], "unverified")
        self.assertFalse(data["insurance_network_verification"]["verified"])
        self.assertEqual(data["accepting_new_patients_status"]["status"], "unknown")
        self.assertFalse(data["accepting_new_patients_status"]["verified"])
        self.assertEqual(data["provenance"]["source"], "Local provider dataset")
        self.assertIn("Listed insurance (reported, not verified)", document.text)
        self.assertEqual(document.metadata["insurance_reported"], ["Medicare", "Aetna"])

    def test_provider_repository_accepts_reported_insurance_source_field(self) -> None:
        repository = ProviderRepository.__new__(ProviderRepository)

        record = repository._normalize_row(
            {
                "id": "provider_002",
                "name": "Reported Insurance Clinic",
                "specialties": ["Primary Care"],
                "languages": ["English"],
                "insurance_reported": ["Medicaid"],
                "address": "10 Main St",
                "city": "Austin",
                "state": "TX",
                "country": "USA",
            }
        )

        self.assertEqual(record.accepted_insurance, ["Medicaid"])
        self.assertEqual(record.to_dict()["insurance_reported"], ["Medicaid"])

    def test_compose_response_appends_required_trust_guidance(self) -> None:
        client = self._client_with_response("Model-rendered answer.")
        payload = {
            "query": {
                "response_language": "English",
                "medical_need": True,
                "summary": "primary care",
            },
            "local_results": [],
            "fallback_results": [],
            "verification_guidance": "Call the provider and insurer to confirm network status.",
        }

        response = self.agent._compose_response(
            client,
            payload,
            max_tokens=128,
            temperature=0.1,
            top_p=0.9,
        )

        self.assertIn("Model-rendered answer.", response)
        self.assertIn("Important safety and trust notes:", response)
        self.assertIn("Directory matches are informational", response)
        self.assertIn("Call the provider and insurer to confirm", response)
        self.assertIn("Do not share personal health information", response)
        self.assertEqual(len(client.calls), 1)

    def test_compose_response_localizes_required_trust_guidance_for_spanish(self) -> None:
        client = self._client_with_response("Respuesta generada por el modelo.")
        payload = {
            "query": {
                "response_language": "Español",
                "medical_need": True,
                "summary": "atencion primaria",
            },
            "local_results": [],
            "fallback_results": [],
            "verification_guidance": "Call the provider and insurer to confirm network status.",
        }

        first_response = self.agent._compose_response(
            client,
            payload,
            max_tokens=128,
            temperature=0.1,
            top_p=0.9,
        )
        second_response = self.agent._compose_response(
            client,
            payload,
            max_tokens=128,
            temperature=0.1,
            top_p=0.9,
        )

        self.assertEqual(first_response, second_response)
        self.assertIn("Respuesta generada por el modelo.", first_response)
        self.assertIn("Notas importantes de seguridad y confianza:", first_response)
        self.assertIn("Llame al proveedor y a la aseguradora", first_response)
        self.assertNotIn("Important safety and trust notes:", first_response)
        self.assertEqual(len(client.calls), 2)

    def test_compose_response_uses_prewritten_required_trust_guidance_for_simplified_chinese(self) -> None:
        client = self._client_with_response("模型生成的答复。")
        payload = {
            "query": {
                "response_language": "中文",
                "medical_need": True,
                "summary": "初级保健",
            },
            "local_results": [],
            "fallback_results": [],
            "verification_guidance": "Call the provider and insurer to confirm network status.",
        }

        response = self.agent._compose_response(
            client,
            payload,
            max_tokens=128,
            temperature=0.1,
            top_p=0.9,
        )

        self.assertIn("模型生成的答复。", response)
        self.assertIn("重要的安全和信任提示：", response)
        self.assertIn("就医前请致电服务提供者和保险公司确认。", response)
        self.assertNotIn("Important safety and trust notes:", response)
        self.assertEqual(len(client.calls), 1)

    def test_compose_response_translates_required_trust_guidance_for_unsupported_detected_language(self) -> None:
        translated_note = _valid_french_translated_trust_guidance()
        client = _SequencedChatClient(["Réponse du modèle.", translated_note])
        payload = {
            "query": {
                "detected_language": "French",
                "medical_need": True,
                "summary": "soins primaires",
            },
            "local_results": [],
            "fallback_results": [],
            "verification_guidance": "Call the provider and insurer to confirm network status.",
        }

        response = self.agent._compose_response(
            client,
            payload,
            max_tokens=128,
            temperature=0.1,
            top_p=0.9,
        )

        expected_note = (
            "Notes importantes de sécurité et de confiance:\n"
            "- Cet outil sert uniquement à orienter les personnes dans la recherche de soins et ne pose pas de diagnostic, ne prescrit pas de traitement et ne remplace pas les conseils d'un professionnel de santé autorisé.\n"
            "- Les résultats du répertoire sont fournis à titre informatif seulement; ils ne constituent pas des recommandations, des approbations ni des garanties d'adéquation clinique.\n"
            "- La participation au réseau d'assurance, les exigences de recommandation, l'accueil de nouveaux patients, le lieu et les disponibilités de rendez-vous ne sont pas vérifiés sauf si la source l'indique explicitement. Appelez le prestataire et l'assureur pour confirmer avant de chercher des soins.\n"
            "- Ne partagez pas d'informations personnelles de santé comme le nom complet, l'adresse, le numéro de sécurité sociale ou le numéro de dossier médical.\n"
            "- Si les symptômes sont graves ou mettent la vie en danger, appelez les services d'urgence (911 aux États-Unis) ou allez aux urgences les plus proches."
        )
        self.assertEqual(response, f"Réponse du modèle.\n\n{expected_note}")
        self.assertEqual(len(client.calls), 2)
        translation_call = client.calls[1]
        self.assertEqual(translation_call["temperature"], 0)
        self.assertEqual(translation_call["top_p"], 1.0)
        translation_messages = translation_call["messages"]
        self.assertIn("Translate only the text provided", translation_messages[0]["content"])
        self.assertIn("Do not add, remove, or reinterpret medical advice", translation_messages[0]["content"])
        self.assertIn("Do not translate or remove bracketed placeholder tokens", translation_messages[0]["content"])
        self.assertIn("into French", translation_messages[1]["content"])
        self.assertIn(_TRUST_GUIDANCE_PROTECTED_TOKENS["emergency"], translation_messages[1]["content"])
        self.assertNotIn("Important safety and trust notes:", response)
        self.assertNotIn("Notas importantes de seguridad y confianza:", response)
        for token in _TRUST_GUIDANCE_PROTECTED_TOKENS.values():
            self.assertNotIn(token, response)

    def test_compose_response_rejects_empty_translated_trust_guidance(self) -> None:
        client = _SequencedChatClient(["Model answer.", "   "])
        payload = {
            "query": {
                "response_language": "French",
                "medical_need": True,
                "summary": "primary care",
            },
            "local_results": [],
            "fallback_results": [],
            "verification_guidance": "Call the provider and insurer to confirm network status.",
        }

        response = self.agent._compose_response(
            client,
            payload,
            max_tokens=128,
            temperature=0.1,
            top_p=0.9,
        )

        self.assertEqual(len(client.calls), 2)
        self.assertIn("Model answer.", response)
        self.assertIn("Important safety and trust notes:", response)

    def test_compose_response_rejects_too_short_translated_trust_guidance(self) -> None:
        tokens = " ".join(_TRUST_GUIDANCE_PROTECTED_TOKENS.values())
        client = _SequencedChatClient(["Model answer.", f"{tokens} Trop court."])
        payload = {
            "query": {
                "response_language": "French",
                "medical_need": True,
                "summary": "primary care",
            },
            "local_results": [],
            "fallback_results": [],
            "verification_guidance": "Call the provider and insurer to confirm network status.",
        }

        response = self.agent._compose_response(
            client,
            payload,
            max_tokens=128,
            temperature=0.1,
            top_p=0.9,
        )

        self.assertEqual(len(client.calls), 2)
        self.assertIn("Important safety and trust notes:", response)
        self.assertNotIn("Trop court.", response)

    def test_compose_response_rejects_malformed_translated_trust_guidance_missing_tokens(self) -> None:
        client = _SequencedChatClient(
            [
                "Model answer.",
                "Notes importantes de sécurité et de confiance :\n"
                "- Cet outil donne quelques informations générales sur les soins.",
            ]
        )
        payload = {
            "query": {
                "response_language": "French",
                "medical_need": True,
                "summary": "primary care",
            },
            "local_results": [],
            "fallback_results": [],
            "verification_guidance": "Call the provider and insurer to confirm network status.",
        }

        response = self.agent._compose_response(
            client,
            payload,
            max_tokens=128,
            temperature=0.1,
            top_p=0.9,
        )

        self.assertEqual(len(client.calls), 2)
        self.assertIn("Important safety and trust notes:", response)
        self.assertNotIn("Cet outil donne quelques informations générales", response)

    def test_compose_response_rejects_meaning_missing_translated_trust_guidance(self) -> None:
        tokens = _TRUST_GUIDANCE_PROTECTED_TOKENS
        translated_note = _valid_french_translated_trust_guidance().replace(
            tokens["emergency"],
            "",
        )
        client = _SequencedChatClient(["Model answer.", translated_note])
        payload = {
            "query": {
                "response_language": "French",
                "medical_need": True,
                "summary": "primary care",
            },
            "local_results": [],
            "fallback_results": [],
            "verification_guidance": "Call the provider and insurer to confirm network status.",
        }

        response = self.agent._compose_response(
            client,
            payload,
            max_tokens=128,
            temperature=0.1,
            top_p=0.9,
        )

        self.assertEqual(len(client.calls), 2)
        self.assertIn("Important safety and trust notes:", response)
        self.assertNotIn("Notes importantes de sécurité", response)

    def test_compose_response_rejects_english_translation_for_non_english_target(self) -> None:
        tokens = _TRUST_GUIDANCE_PROTECTED_TOKENS
        translated_note = (
            "Important safety and trust notes:\n"
            f"- {tokens['scope']} This tool supports care navigation only and does not diagnose, prescribe, or replace licensed medical advice {tokens['advice']}.\n"
            f"- {tokens['directory']} Directory matches are informational, not referrals, endorsements, or guarantees of clinical fit.\n"
            f"- {tokens['insurance']} Insurance/network participation, referral requirements, new-patient availability, location, and appointment availability are not verified unless the source explicitly says so. Call the provider and insurer to confirm before seeking care {tokens['confirm']}.\n"
            f"- {tokens['privacy']} Do not share personal health information such as full names, addresses, Social Security numbers, or medical record numbers.\n"
            f"- {tokens['emergency']} If symptoms are severe or life-threatening, call emergency services (911 in the U.S.) or go to the nearest emergency room."
        )
        client = _SequencedChatClient(["Model answer.", translated_note])
        payload = {
            "query": {
                "response_language": "French",
                "medical_need": True,
                "summary": "primary care",
            },
            "local_results": [],
            "fallback_results": [],
            "verification_guidance": "Call the provider and insurer to confirm network status.",
        }

        response = self.agent._compose_response(
            client,
            payload,
            max_tokens=128,
            temperature=0.1,
            top_p=0.9,
        )

        self.assertEqual(response.count("Important safety and trust notes:"), 1)
        self.assertIn("Model answer.", response)
        self.assertIn("Directory matches are informational", response)

    def test_compose_response_falls_back_to_english_when_trust_guidance_translation_fails(self) -> None:
        client = _SequencedChatClient(["Model answer."], fail_on_call=2)
        payload = {
            "query": {
                "response_language": "French",
                "medical_need": True,
                "summary": "primary care",
            },
            "local_results": [],
            "fallback_results": [],
            "verification_guidance": "Call the provider and insurer to confirm network status.",
        }

        response = self.agent._compose_response(
            client,
            payload,
            max_tokens=128,
            temperature=0.1,
            top_p=0.9,
        )

        self.assertEqual(len(client.calls), 2)
        self.assertIn("Important safety and trust notes:", response)
        self.assertIn("Directory matches are informational", response)

    def test_compose_response_falls_back_to_english_trust_guidance_for_unknown_language(self) -> None:
        client = self._client_with_response("Model answer.")
        payload = {
            "query": {
                "response_language": "unknown",
                "medical_need": True,
                "summary": "primary care",
            },
            "local_results": [],
            "fallback_results": [],
            "verification_guidance": "Call the provider and insurer to confirm network status.",
        }

        response = self.agent._compose_response(
            client,
            payload,
            max_tokens=128,
            temperature=0.1,
            top_p=0.9,
        )

        self.assertIn("Important safety and trust notes:", response)
        self.assertIn("Directory matches are informational", response)
        self.assertEqual(len(client.calls), 1)

    def test_compose_response_logs_omit_phi_bearing_payload(self) -> None:
        client = type(
            "Client",
            (),
            {
                "chat_completion": lambda self, messages, max_tokens, temperature, top_p: type(
                    "Completion",
                    (),
                    {
                        "choices": [
                            type(
                                "Choice",
                                (),
                                {
                                    "message": {"content": "Model answer."},
                                    "finish_reason": "stop",
                                },
                            )()
                        ]
                    },
                )()
            },
        )()
        payload = {
            "query": {
                "response_language": "English",
                "medical_need": True,
                "summary": "Jane Doe at 123 Main St needs primary care",
            },
            "local_results": [],
            "fallback_results": [],
            "verification_guidance": "Call the provider and insurer to confirm network status.",
        }

        with self.assertLogs("care_agent", level="DEBUG") as captured_logs:
            self.agent._compose_response(
                client,
                payload,
                max_tokens=128,
                temperature=0.1,
                top_p=0.9,
            )

        log_output = "\n".join(captured_logs.output)
        self.assertNotIn("Jane Doe", log_output)
        self.assertNotIn("123 Main St", log_output)


if __name__ == "__main__":
    unittest.main()
