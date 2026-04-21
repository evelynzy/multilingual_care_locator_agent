import os
import logging
from typing import List, Optional

import gradio as gr
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

from care_agent import CareLocatorAgent, normalize_chat_messages
from retriever import ProviderRepository
from config_loader import (
    get_chat_model_settings,
    get_message,
    get_ui_settings,
)

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

load_dotenv()

chat_settings = get_chat_model_settings()
ui_settings = get_ui_settings()

DEFAULT_MODEL_ID = chat_settings["model_id"]
LOGIN_MESSAGE = get_message(
    "login_required",
    "Authentication required to use the Hugging Face Inference API.",
)
ERROR_MESSAGE = get_message(
    "unexpected_error",
    "I encountered an unexpected error while processing your request. Please retry in a moment.",
)

SAFETY_TRUST_NOTES = """
- This assistant helps with care navigation only. It does not diagnose, prescribe, or replace licensed medical advice.
- Directory matches are informational, not referrals, endorsements, or guarantees of clinical fit.
- Insurance/network participation, referral requirements, new-patient status, location, and appointment availability are not verified unless a source explicitly says so.
- Call the provider and insurer to confirm network status, accepted insurance plan, referral requirements, new-patient availability, location, and appointment availability.
- Do not share PHI such as full names, addresses, Social Security numbers, or medical record numbers.
- For severe or life-threatening symptoms, call emergency services (911 in the U.S.) or go to the nearest emergency room.
"""

DATA_SOURCE_LIMITATIONS_NOTES = """
- Local sample provider directory, when available.
- NPI Records - Individuals public API: https://clinicaltables.nlm.nih.gov/apidoc/npi_idv/v3/doc.html
- NPI Records - Organizations public API: https://clinicaltables.nlm.nih.gov/apidoc/npi_org/v3/doc.html
- NPPES API public registry: https://npiregistry.cms.hhs.gov/api-page
- Medicare Opt Out Records public data: https://data.cms.gov/data-api/v1/dataset/9887a515-7552-4693-bf58-735c77af46d7/data
- Public sources may be incomplete, delayed, or missing appointment and network details. Confirm directly before seeking care.
"""


# Initialize repository and agent once so we do not reload data per request.
provider_repository = ProviderRepository()
care_locator_agent = CareLocatorAgent(provider_repository)


def _augment_history(
    history: Optional[List[dict]], system_message: Optional[str]
) -> List[dict]:
    messages: List[dict] = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    if history:
        messages.extend(normalize_chat_messages(history))
    return normalize_chat_messages(messages)


def respond(
    message: str,
    history: List[dict],
):
    token_value = os.getenv("HF_TOKEN")

    if not token_value:
        logger.error("HF_TOKEN environment variable not set; unable to call Inference API")
        return LOGIN_MESSAGE

    client = InferenceClient(token=token_value, model=DEFAULT_MODEL_ID)

    augmented_history = _augment_history(history, ui_settings.get("default_system_message"))

    try:
        logger.info("Invoking care locator. message_length=%s", len(message))
        reply = care_locator_agent.handle_request(
            client=client,
            message=message,
            history=augmented_history,
            max_tokens=chat_settings.get("max_tokens", 512),
            temperature=chat_settings.get("temperature", 0.3),
            top_p=chat_settings.get("top_p", 0.9),
        )
    except Exception as exc:  # noqa: BLE001 - surface error to end user
        logger.exception("Unexpected error while processing request")
        return f"⚠️ {ERROR_MESSAGE} Details for debugging: {exc}"

    return reply


chatbot = gr.ChatInterface(
    respond,
    type="messages",
    # textbox=gr.Textbox(placeholder="e.g., primary care 90048"),
    examples=[
        "primary care 75001",
        "儿科 10013",
        "dentista 33012",
    ],
    description=ui_settings.get("description", ""),
    title=ui_settings.get("title", "Multilingual Care Locator"),
)

custom_css = """
.gradio-container .gr-chatbot {
    min-height: 620px;
    max-height: 72vh;
    overflow-y: auto;
}

@media (max-width: 640px) {
    .gradio-container .gr-chatbot {
        min-height: 460px;
        max-height: 68vh;
    }
}

.gradio-container table {
    width: 100%;
    table-layout: fixed;
    border-collapse: collapse;
}

.gradio-container th,
.gradio-container td {
    white-space: normal;
    word-break: break-word;
}

.gradio-container table {
    display: block;
    overflow-x: auto;
}
"""


with gr.Blocks(fill_height=True, css=custom_css) as demo:
    chatbot.render()
    with gr.Accordion("Safety and trust notes", open=False):
        gr.Markdown(SAFETY_TRUST_NOTES)
    with gr.Accordion("Data sources and limitations", open=False):
        gr.Markdown(DATA_SOURCE_LIMITATIONS_NOTES)


if __name__ == "__main__":
    demo.launch()
