# -*- coding: utf-8 -*-
"""Vision helpers -- ``requests`` only (no ``openai``).

``vision_query`` below is a single-image, string-returning helper used by
``generate_tool_description.py``; it delegates to ``vision_qwen.vision_query``
(Ollama ``/api/generate`` over ``requests``).
"""

import base64
import mimetypes

from .vision_qwen import vision_query as _vision_query_multi, MY_MODEL, MY_URL


def build_vision_messages(filename, message):
    """
    Legacy helper: OpenAI-style single-turn ``messages`` payload with the text
    prompt and the base64-encoded image together in one user turn. Not used by
    the ``requests`` path any more; kept for backward compatibility.
    """
    mime_type, _ = mimetypes.guess_type(filename)
    if mime_type is None:
        mime_type = "application/octet-stream"

    with open(filename, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode("utf-8")

    data_url = f"data:{mime_type};base64,{image_b64}"

    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": message},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        }
    ]


def vision_query(filename, message, model=MY_MODEL, model_url=MY_URL):
    """
    Send one image and a text prompt to the vision model; return the answer
    text (str). The raw streamed chunk list is stashed on
    ``vision_query.response`` for debugging, as the old code did.
    """
    final_response, thinking_response, all_chunks = _vision_query_multi(
        filenames_list=[filename], message=message, model=model, model_url=model_url,
    )
    vision_query.response = all_chunks
    return final_response
