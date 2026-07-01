"""Test setup for the eval suite.

CI installs only ``requirements-test.txt`` and intentionally omits the heavy
runtime deps. Some eval tests import ``eval.instrumented_agent`` ->
``care_agent``, which imports ``huggingface_hub`` at module level. Stub it here
**only when it is genuinely not installed**, so the offline suite collects in CI
while local development and gated live tests keep using the real client.
"""
import sys
import types

try:  # dependency present (local dev, or anywhere it is installed) -> use it
    import huggingface_hub  # noqa: F401
except ImportError:  # CI: not installed -> minimal stub so imports resolve
    _stub = types.ModuleType("huggingface_hub")

    class InferenceClient:  # pragma: no cover - stub, never called offline
        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "huggingface_hub is stubbed in this environment; "
                "gated live tests require the real dependency"
            )

    _stub.InferenceClient = InferenceClient
    sys.modules["huggingface_hub"] = _stub
