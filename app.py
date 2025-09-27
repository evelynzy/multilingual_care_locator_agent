import os
import logging
from typing import List, Optional

import gradio as gr
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

from care_agent import CareLocatorAgent
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
        messages.extend(history)
    return messages


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
        logger.info("Invoking care locator. message=%s", message[:1000])
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
    min-height: 420px;
    max-height: 560px;
    overflow-y: auto;
}

@media (max-width: 640px) {
    .gradio-container .gr-chatbot {
        min-height: 320px;
        max-height: 420px;
    }

    .gradio-container .gr-chatbot {
        min-height: 320px;
        max-height: 420px;
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
    gr.Markdown(
        """
        **Data sources**
        - [NPI Records - Individuals](https://clinicaltables.nlm.nih.gov/apidoc/npi_idv/v3/doc.html) public API
        - [NPI Records - Organizations](https://clinicaltables.nlm.nih.gov/apidoc/npi_org/v3/doc.html) public API
        - [NPPES API](https://npiregistry.cms.hhs.gov/api-page) public API
        """
    )


if __name__ == "__main__":
    demo.launch()
