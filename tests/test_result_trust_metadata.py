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
from retriever import ProviderRecord, ProviderRepository


class _SequencedChatClient:
    def __init__(self, response_texts):
        if isinstance(response_texts, str):
            response_texts = [response_texts]
        self.response_texts = list(response_texts)
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
                "specialties": ["Primary Care"],
                "insurance": ["Medicare", "Aetna"],
                "source": "Local provider dataset",
                "location": "San Francisco, CA",
                "phone": "415-555-0100",
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

        client = _SequencedChatClient("Model table should not be used.")

        with patch.object(
            self.agent,
            "_interpret_user_need",
            return_value=query,
        ):
            result = self.agent.handle_request(
                client,
                "primary care 94110",
                [],
                max_tokens=256,
                temperature=0.2,
                top_p=0.9,
            )

        self.assertEqual(len(client.calls), 0)
        self.assertIn('<div class="provider-card">', result)
        self.assertIn('<div class="provider-card__title">1. Harmony Family Clinic</div>', result)
        self.assertIn('<div class="provider-card__subtitle">Primary Care • San Francisco, CA</div>', result)
        self.assertIn('Phone</span><span class="provider-card__meta-value">415-555-0100</span>', result)
        self.assertIn('Source</span><span class="provider-card__meta-value">Local provider dataset</span>', result)
        self.assertIn('<span class="provider-card__badge">Informational</span>', result)
        self.assertIn('<span class="provider-card__badge">Network unverified</span>', result)
        self.assertIn('<span class="provider-card__badge">New patients unknown</span>', result)
        self.assertIn('<span class="provider-card__badge">Appointments unverified</span>', result)
        self.assertNotIn('<span class="provider-card__badge">Source: Local provider dataset</span>', result)
        self.assertIn('Why matched</span><span class="provider-card__value">Relevant to your search for primary care in San Francisco.</span>', result)
        self.assertNotIn("Listed insurance</span>", result)
        self.assertNotIn("Insurance/network verification</span>", result)
        self.assertNotIn("Accepting new patients</span>", result)
        self.assertNotIn("Appointment availability</span>", result)
        self.assertIn('Next step</span><span class="provider-card__value">Call to confirm network status, referral needs, new-patient status, and appointment availability.</span>', result)
        self.assertIn("Important safety and trust notes:", result)
        self.assertNotIn("### 1. Harmony Family Clinic", result)

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

    def test_trust_labels_include_medicare_opt_out_status(self) -> None:
        result = self.agent._normalize_result_trust_metadata(
            {
                "name": "Specialty Clinic",
                "source": "NPI Registry",
                "insurance_reported": ["Medicare"],
                "insurance_network_verification": {"status": "unverified"},
                "accepting_new_patients_status": {"status": "unknown"},
                "medicare_opt_out": {"opted_out": True},
            }
        )

        self.assertIn("Source: NPI Registry", result["trust_labels"])
        self.assertIn("Insurance/network: unverified", result["trust_labels"])
        self.assertIn("New patients: unknown", result["trust_labels"])
        self.assertIn("Medicare opt-out: opted out", result["trust_labels"])

    def test_styled_provider_card_preserves_dynamic_trust_labels(self) -> None:
        result = self.agent._normalize_result_trust_metadata(
            {
                "name": "Specialty Clinic",
                "location": "Austin, TX",
                "phone": "512-555-0100",
                "taxonomy": "Cardiology",
                "source": "NPI Registry",
                "insurance_reported": ["Medicare"],
                "insurance_network_verification": {"status": "unverified"},
                "accepting_new_patients_status": {"status": "unknown"},
                "medicare_opt_out": {"opted_out": True},
            }
        )

        card_html = self.agent._format_provider_result_card(
            result,
            index=1,
            query={"specialties": ["Cardiology"], "keywords": []},
        )

        self.assertIn('<span class="provider-card__badge">Informational</span>', card_html)
        self.assertIn('<span class="provider-card__badge">Network unverified</span>', card_html)
        self.assertIn('<span class="provider-card__badge">New patients unknown</span>', card_html)
        self.assertIn('<span class="provider-card__badge">Appointments unverified</span>', card_html)
        self.assertNotIn('<span class="provider-card__badge">Source: NPI Registry</span>', card_html)
        self.assertNotIn('<span class="provider-card__badge">Insurance/network: unverified</span>', card_html)
        self.assertNotIn('<span class="provider-card__badge">New patients: unknown</span>', card_html)
        self.assertIn('<span class="provider-card__badge">Medicare opt-out: opted out</span>', card_html)
        self.assertIn('Why matched</span><span class="provider-card__value">Listed under Cardiology.</span>', card_html)
        self.assertNotIn("Listed insurance</span>", card_html)

    def test_provider_card_omits_low_signal_subtitle_fragments(self) -> None:
        card_html = self.agent._format_provider_result_card(
            self.agent._normalize_result_trust_metadata(
                {
                    "name": "Fallback Clinic",
                    "specialties": ["M"],
                    "location": "BOYD",
                    "phone": "512-555-0199",
                    "source": "Local provider dataset",
                }
            ),
            index=1,
            query={"summary": "child care"},
        )

        self.assertNotIn('provider-card__subtitle">M • BOYD</div>', card_html)
        self.assertNotIn('provider-card__subtitle">', card_html)

    def test_compose_result_card_response_localizes_chinese_deterministic_copy(self) -> None:
        payload = {
            "query": {
                "response_language": "中文",
                "summary": "儿科10013",
            },
            "care_setting_guidance": "For routine or ongoing care, primary care is usually the best fit.",
            "specialist_plan_guidance": "For specialist searches, HMO and POS plans often require a PCP referral; PPO plans may not, but you should confirm the rule with your insurer and plan documents.",
            "local_results": [
                self.agent._normalize_result_trust_metadata(
                    {
                        "name": "Harmony Family Clinic",
                        "specialties": ["Pediatrics"],
                        "location": "New York, NY",
                        "phone": "212-555-0100",
                        "source": "Local provider dataset",
                        "insurance_reported": ["Medicaid"],
                    }
                )
            ],
            "fallback_results": [],
            "verification_guidance": "Call the provider and insurer to confirm network status, accepted insurance plan, referral requirements, new-patient availability, location, and appointment availability.",
        }

        response = self.agent._compose_result_card_response(payload)

        self.assertIn("儿科10013的护理导航结果如下。", response)
        self.assertIn("**就医路线:** 对于常规或持续性的就医需求，初级保健通常更合适。", response)
        self.assertIn("**转诊提示:** 查找专科医生时，HMO 和 POS 计划通常需要初级保健医生转诊；PPO 计划可能不需要，但仍应与保险公司和计划文件确认。", response)
        self.assertIn("电话</span>", response)
        self.assertIn("匹配原因</span>", response)
        self.assertIn("下一步</span>", response)
        self.assertIn("与您搜索的儿科10013相关。", response)
        self.assertIn("请致电确认网络状态、转诊要求、新患者接收情况和预约可用性。", response)
        self.assertIn("来源</span><span class=\"provider-card__meta-value\">Local provider dataset</span>", response)
        self.assertNotIn("Here are care navigation results for", response)
        self.assertNotIn("**Care route:**", response)
        self.assertNotIn("**Referral note:**", response)
        self.assertNotIn("Why matched", response)
        self.assertNotIn("Pediatrics, pediatric, child health", response)

    def test_compose_result_card_response_localizes_spanish_deterministic_copy(self) -> None:
        payload = {
            "query": {
                "response_language": "Español",
                "summary": "atención primaria en Austin",
            },
            "care_setting_guidance": "For a known specialty or referral need, a specialist is usually the right route.",
            "specialist_plan_guidance": "For specialist searches, HMO and POS plans often require a PCP referral; PPO plans may not, but you should confirm the rule with your insurer and plan documents.",
            "local_results": [
                self.agent._normalize_result_trust_metadata(
                    {
                        "name": "Specialty Clinic",
                        "taxonomy": "Cardiology",
                        "location": "Austin, TX",
                        "phone": "512-555-0100",
                        "source": "NPI Registry",
                        "insurance_reported": ["Medicare"],
                        "medicare_opt_out": {"opted_out": True},
                    }
                )
            ],
            "fallback_results": [],
            "verification_guidance": "Call the provider and insurer to confirm network status, accepted insurance plan, referral requirements, new-patient availability, location, and appointment availability.",
        }

        response = self.agent._compose_result_card_response(payload)

        self.assertIn("Aquí están los resultados de navegación de atención para atención primaria en Austin.", response)
        self.assertIn("**Ruta de atención:** Para una necesidad conocida de especialista o remisión, un especialista suele ser la ruta correcta.", response)
        self.assertIn("**Nota sobre remisión:** Para buscar especialistas, los planes HMO y POS suelen requerir una remisión de atención primaria; los PPO pueden no requerirla, pero debe confirmarlo con su aseguradora y los documentos del plan.", response)
        self.assertIn("Teléfono</span>", response)
        self.assertIn("Por qué coincide</span>", response)
        self.assertIn("Siguiente paso</span>", response)
        self.assertIn("Relacionado con su búsqueda de atención primaria en Austin.", response)
        self.assertIn("Llame para confirmar la red, la necesidad de remisión, si aceptan pacientes nuevos y la disponibilidad de citas.", response)
        self.assertIn("Exclusión de Medicare: excluido", response)
        self.assertNotIn("Here are care navigation results for", response)
        self.assertNotIn("**Care route:**", response)
        self.assertNotIn("**Referral note:**", response)
        self.assertNotIn("Why matched", response)

    def test_compose_result_card_response_uses_localized_match_reason_instead_of_raw_keywords(self) -> None:
        payload = {
            "query": {
                "response_language": "中文",
                "summary": "儿童保健",
                "specialties": ["Pediatrics"],
                "keywords": ["pediatric", "child health"],
            },
            "local_results": [
                self.agent._normalize_result_trust_metadata(
                    {
                        "name": "Harmony Family Clinic",
                        "specialties": ["Pediatrics"],
                        "location": "New York, NY",
                        "phone": "212-555-0100",
                        "source": "Local provider dataset",
                    }
                )
            ],
            "fallback_results": [],
        }

        response = self.agent._compose_result_card_response(payload)

        self.assertIn("匹配原因</span><span class=\"provider-card__value\">与您搜索的儿童保健相关。</span>", response)
        self.assertNotIn("Pediatrics, pediatric, child health", response)

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

    def test_compose_response_uses_prewritten_required_trust_guidance_for_supported_languages(self) -> None:
        note_headings = (
            "Important safety and trust notes:",
            "Notas importantes de seguridad y confianza:",
            "重要的安全和信任提示：",
            "Ghi chú quan trọng về an toàn và độ tin cậy:",
            "Mahahalagang tala sa kaligtasan at pagtitiwala:",
            "ملاحظات مهمة حول السلامة والثقة:",
            "중요한 안전 및 신뢰 안내:",
        )
        supported_languages = [
            ("English", "Important safety and trust notes:", "Directory matches are informational"),
            ("Español", "Notas importantes de seguridad y confianza:", "Llame al proveedor y a la aseguradora"),
            ("中文", "重要的安全和信任提示：", "就医前请致电服务提供者和保险公司确认。"),
            ("Vietnamese", "Ghi chú quan trọng về an toàn và độ tin cậy:", "Hãy gọi cho nhà cung cấp và công ty bảo hiểm"),
            ("Tagalog", "Mahahalagang tala sa kaligtasan at pagtitiwala:", "Tawagan ang provider at insurer"),
            ("Filipino", "Mahahalagang tala sa kaligtasan at pagtitiwala:", "Tawagan ang provider at insurer"),
            ("Arabic", "ملاحظات مهمة حول السلامة والثقة:", "اتصل بمقدم الخدمة وشركة التأمين"),
            ("Korean", "중요한 안전 및 신뢰 안내:", "제공자와 보험사에 전화해 확인하세요"),
        ]

        for language, heading, required_phrase in supported_languages:
            with self.subTest(language=language):
                client = self._client_with_response("Model-rendered answer.")
                payload = {
                    "query": {
                        "response_language": language,
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
                self.assertIn(heading, response)
                self.assertIn(required_phrase, response)
                self.assertEqual(sum(response.count(note_heading) for note_heading in note_headings), 1)
                self.assertEqual(len(client.calls), 1)

    def test_compose_response_uses_same_deterministic_supported_language_note_each_time(self) -> None:
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
        self.assertIn("Notas importantes de seguridad y confianza:", first_response)
        self.assertEqual(len(client.calls), 2)

    def test_compose_response_falls_back_to_english_for_unsupported_detected_language_without_translation(self) -> None:
        client = _SequencedChatClient(["Réponse du modèle.", "Unexpected translated note"])
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

        self.assertIn("Réponse du modèle.", response)
        self.assertIn("Important safety and trust notes:", response)
        self.assertIn("Directory matches are informational", response)
        self.assertEqual(response.count("Important safety and trust notes:"), 1)
        self.assertNotIn("Unexpected translated note", response)
        self.assertEqual(len(client.calls), 1)

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
        self.assertEqual(response.count("Important safety and trust notes:"), 1)
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
