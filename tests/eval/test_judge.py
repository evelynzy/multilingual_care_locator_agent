from __future__ import annotations

import json
import os
import tempfile
import unittest

from eval.judge import DIMENSIONS, JudgeVerdict, QwenJudge
from eval.trace import Trace, TurnCapture


def _turn(**overrides):
    base = dict(
        user_message="cardiology 98101",
        parsed_specialties=["cardiology"], parsed_preferred_languages=[],
        parsed_urgency=None, parsed_care_setting=None, parsed_needs_clarification=False,
        searched=True, request_specialties=["cardiology"], request_preferred_languages=[],
        provider_states=["CA"], provider_count=1, html_has_card=True, emergency_routed=False,
        rendered_text="1. Dr. Heart — Cardiology — Seattle, WA",
        provider_details=[{"name": "Dr. Heart", "specialties": ["Cardiology"],
                           "languages": [], "state": "WA", "city": "Seattle"}],
    )
    base.update(overrides)
    return TurnCapture(**base)


class _StubClient:
    def __init__(self, content):
        self.content = content
        self.calls = []

    def chat_completion(self, messages, model=None, max_tokens=None, temperature=None, **kwargs):
        self.calls.append((messages, model))

        class _Msg:
            content = self.content

        class _Choice:
            message = _Msg()

        class _Resp:
            choices = [_Choice()]

        return _Resp()


_GOOD_JSON = json.dumps({
    "helpfulness": {"pass": True, "reason": "surfaced a cardiologist"},
    "safety": {"pass": True, "reason": "no diagnosis"},
    "faithfulness": {"pass": True, "reason": "name matches record"},
    "language_appropriateness": {"pass": True, "reason": "answered in English"},
})


class JudgeParseTests(unittest.TestCase):
    def test_parses_binary_verdict_and_rationales(self):
        client = _StubClient(_GOOD_JSON)
        verdict = QwenJudge(client, use_cache=False).score("s01", "en", Trace("s01", "en", [_turn()]))
        self.assertIsNone(verdict.error)
        for name in DIMENSIONS:
            self.assertTrue(getattr(verdict, name))
        self.assertIn("cardiologist", verdict.rationales["helpfulness"])

    def test_prompt_includes_provider_record(self):
        client = _StubClient(_GOOD_JSON)
        QwenJudge(client, use_cache=False).score("s01", "en", Trace("s01", "en", [_turn()]))
        prompt_text = json.dumps(client.calls[0][0], ensure_ascii=False)
        self.assertIn("Dr. Heart", prompt_text)

    def test_strips_code_fence(self):
        client = _StubClient("```json\n" + _GOOD_JSON + "\n```")
        verdict = QwenJudge(client, use_cache=False).score("s01", "en", Trace("s01", "en", [_turn()]))
        self.assertIsNone(verdict.error)
        self.assertTrue(verdict.safety)

    def test_malformed_json_becomes_error_verdict(self):
        client = _StubClient("the model refused to answer")
        verdict = QwenJudge(client, use_cache=False).score("s01", "en", Trace("s01", "en", [_turn()]))
        self.assertIsNotNone(verdict.error)
        self.assertFalse(verdict.helpfulness)

    def test_errored_trace_short_circuits_without_calling_client(self):
        client = _StubClient(_GOOD_JSON)
        verdict = QwenJudge(client, use_cache=False).score(
            "s01", "en", Trace("s01", "en", [], error="Boom: kaboom"))
        self.assertIsNotNone(verdict.error)
        self.assertEqual(client.calls, [])

    def test_final_turn_is_scored_for_multiturn(self):
        client = _StubClient(_GOOD_JSON)
        first = _turn(user_message="I need a specialist", rendered_text="Which specialty?",
                      searched=False, provider_details=[])
        last = _turn()
        QwenJudge(client, use_cache=False).score("s14", "en", Trace("s14", "en", [first, last]))
        prompt_text = json.dumps(client.calls[0][0], ensure_ascii=False)
        self.assertIn("Dr. Heart", prompt_text)
        self.assertNotIn("Which specialty?", prompt_text)


class JudgeCacheTests(unittest.TestCase):
    def test_verdict_is_cached_and_reused(self):
        with tempfile.TemporaryDirectory() as d:
            client = _StubClient(_GOOD_JSON)
            judge = QwenJudge(client, cache_dir=d, use_cache=True)
            trace = Trace("s01", "en", [_turn()])
            judge.score("s01", "en", trace)
            self.assertEqual(len(client.calls), 1)
            self.assertEqual(len(os.listdir(d)), 1)
            # second call served from cache, client not hit again
            judge.score("s01", "en", trace)
            self.assertEqual(len(client.calls), 1)


class JudgeLiveTests(unittest.TestCase):
    @unittest.skipUnless(
        os.getenv("RUN_EVAL") == "1" and bool(os.getenv("HF_TOKEN")),
        "set RUN_EVAL=1 and HF_TOKEN to run the live judge test",
    )
    def test_qwen_scores_a_real_reply(self):
        from huggingface_hub import InferenceClient
        from eval.judge import JUDGE_MODEL

        client = InferenceClient(model=JUDGE_MODEL, token=os.getenv("HF_TOKEN"))
        verdict = QwenJudge(client, use_cache=False).score("s01", "en", Trace("s01", "en", [_turn()]))
        self.assertIsNone(verdict.error)
        self.assertIsInstance(verdict.language_appropriateness, bool)


if __name__ == "__main__":
    unittest.main()
