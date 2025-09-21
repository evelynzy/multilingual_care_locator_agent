---
title: Multilingual Care Locator Agent
emoji: 💬
colorFrom: yellow
colorTo: purple
sdk: gradio
sdk_version: 5.42.0
app_file: app.py
pinned: false
hf_oauth: false
license: apache-2.0
short_description: Multilingual healthcare locator agent
---

This Space hosts a multilingual care navigation assistant built with [Gradio](https://gradio.app), the [Hugging Face Inference API](https://huggingface.co/docs/api-inference/index), and a [LlamaIndex](https://www.llamaindex.ai/) vector search over healthcare providers.

## What it does
- Understands care requests written in English, 中文, Español, Français, Tagalog, and more via a multilingual LLM
- Extracts structured intent (specialty, location, insurance, language preferences) and searches healthcare provider data
- Calls the public [NPPES](https://clinicaltables.nlm.nih.gov/api/npireg/v3/search) ClinicalTables API
- Provides responses in the user's language, including trusted backup resources when no direct matches are found