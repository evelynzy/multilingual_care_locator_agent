import unittest

from config_loader import get_prompt, get_ui_settings


class TrustBaselinePromptTests(unittest.TestCase):
    def test_response_template_requires_unverified_network_and_new_patient_language(self) -> None:
        template = get_prompt("response_user_template")

        self.assertIn(
            "insurance_reported",
            template,
        )
        self.assertIn("compact Markdown card", template)
        self.assertIn("Never render provider results as Markdown tables", template)
        self.assertIn("trust_labels", template)
        self.assertIn(
            "call the provider and insurer to confirm network status, accepted insurance plan, referral requirements, new-patient availability, location, and appointment availability",
            template,
        )
        self.assertIn("follow_up_questions", template)
        self.assertIn("care_setting_guidance", template)

    def test_clarification_template_mentions_follow_up_questions(self) -> None:
        template = get_prompt("response_user_template_clarification_needed")

        self.assertIn("follow_up_questions", template)
        self.assertIn("specialist_plan_guidance", template)
        self.assertIn("care_setting_guidance", template)

    def test_fallback_template_requires_unverified_network_and_new_patient_language(self) -> None:
        template = get_prompt("response_user_template_fallback_only")

        self.assertIn(
            "Do not imply insurance/network participation or accepting-new-patient status",
            template,
        )
        self.assertIn(
            "call the provider and insurer to confirm network status, accepted insurance plan, referral requirements, new-patient availability, location, and appointment availability",
            template,
        )

    def test_default_system_message_mentions_verification_limits(self) -> None:
        template = get_prompt("default_system_message")

        self.assertIn(
            "Do not present insurance/network participation or accepting-new-patient status as verified",
            template,
        )

    def test_emergency_template_mentions_emergency_services(self) -> None:
        template = get_prompt("response_user_template_emergency")

        self.assertIn("emergency services", template)
        self.assertIn("Do not ask follow-up questions", template)

    def test_ui_description_is_compact_and_phi_safe(self) -> None:
        description = get_ui_settings()["description"]

        self.assertIn("Search by care need + city/ZIP", description)
        self.assertIn("avoid PHI", description)
        self.assertIn("Emergencies: call 911/local services", description)
        self.assertIn("Results are informational", description)
        self.assertIn("insurance/network", description)
        self.assertIn("new-patient", description)
        self.assertIn("appointment availability", description)
        self.assertIn("public directory data", description)
        self.assertIn("unverified, incomplete, or outdated", description)
        self.assertLess(len(description), 260)


if __name__ == "__main__":
    unittest.main()
