"""Real-path smoke test: multilingual queries must return providers (non-zero).

Hits the live LLM (intent translation/extraction) and the live ClinicalTables
API. Skipped unless RUN_SMOKE=1 and HF_TOKEN are set, so it never runs in the
normal mocked unit suite. This is the layer whose absence let the "儿科10013
returns zero" bug ship.
"""

from __future__ import annotations

import os
import unittest

RUN = os.getenv("RUN_SMOKE") == "1" and bool(os.getenv("HF_TOKEN"))


@unittest.skipUnless(RUN, "set RUN_SMOKE=1 and HF_TOKEN to run the live smoke test")
class MultilingualSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from huggingface_hub import InferenceClient
        from care_agent import CareLocatorAgent
        from provider_search.service import ProviderSearchService
        from provider_search.sources.clinicaltables import ClinicalTablesSource
        from config_loader import get_chat_model_settings

        settings = get_chat_model_settings()
        cls.settings = settings
        cls.client = InferenceClient(model=settings["model_id"], token=os.getenv("HF_TOKEN"))
        cls.agent = CareLocatorAgent(
            provider_search_service=ProviderSearchService(
                clinicaltables_source=ClinicalTablesSource()
            )
        )

    def _respond(self, message: str) -> str:
        return self.agent.handle_request(
            self.client,
            message,
            [],
            max_tokens=self.settings["max_tokens"],
            temperature=self.settings["temperature"],
            top_p=self.settings["top_p"],
        )

    def test_chinese_pediatrics_glued_zip_returns_providers(self):
        # The exact reported failing case.
        self.assertIn("provider-card", self._respond("儿科10013"))

    def test_chinese_cardiology_returns_providers(self):
        self.assertIn("provider-card", self._respond("心脏科 98101"))

    def test_english_control_returns_providers(self):
        self.assertIn("provider-card", self._respond("pediatrics 10013"))

    def test_spanish_query_returns_providers(self):
        self.assertIn("provider-card", self._respond("dentista 33012"))

    def test_arabic_pediatrics_returns_providers(self):
        # أطفال = pediatrics. A failure here is a finding: the LLM path does not
        # generalize past the languages already tried.
        self.assertIn("provider-card", self._respond("أطفال 10013"))


if __name__ == "__main__":
    unittest.main()
