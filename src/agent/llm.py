import os
import json
import logging

logger = logging.getLogger(__name__)

GROQ_MODEL = "llama-3.3-70b-versatile"


def _get_api_key():
    try:
        import streamlit as st
        key = st.secrets.get("GROQ_API_KEY")
        if key:
            return key
    except Exception:
        pass
    return os.environ.get("GROQ_API_KEY")


def _get_client():
    from groq import Groq
    key = _get_api_key()
    if not key:
        return None
    return Groq(api_key=key)


def chat_json(system, user):
    client = _get_client()
    if client is None:
        logger.warning("Groq API key not set")
        return None
    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.3,
            max_tokens=900,
            response_format={"type": "json_object"},
            timeout=25,
        )
        content = response.choices[0].message.content
        return json.loads(content)
    except Exception as e:
        logger.error(f"Groq call failed: {e}")
        return None
