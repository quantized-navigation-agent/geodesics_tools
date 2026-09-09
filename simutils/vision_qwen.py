# -*- coding: utf-8 -*-
"""Visual query to an Ollama endpoint -- ``requests`` only (no ``openai``).

Modelled on ``vision_query_ollama.py``. Streams from Ollama's native
``/api/generate`` and returns ``(final_response, thinking_response, all_chunks)``.

Before sending, every image is enlarged (nearest-neighbour, aspect ratio kept)
to at least ``MIN_IMAGE_PIXELS`` total pixels if it is smaller -- the matplotlib
grid renders small and some vision models read it better at higher resolution.
"""

import io
import re
import json
import math
import base64

import requests
from PIL import Image

MY_MODEL = "qwen3.6:35b-a3b"                          # model tag served by your Ollama endpoint
MY_URL = "http://OLLAMA_HOST:11434/api/generate"      # set to your Ollama /api/generate endpoint
MAX_TOKEN_GENERATION_LIMIT = 32_768                   # 32k -- num_predict cap, avoids infinite loops (lower 16k: 16_384)
MIN_IMAGE_PIXELS = 1_000_000                          # images smaller than this are upscaled before sending

safe_model_name = re.sub(r'[^a-zA-Z0-9_]', '_', MY_MODEL)   # qwen3_6_35b_a3b

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def encode_image(path, min_pixels=MIN_IMAGE_PIXELS):
    """Read an image file and return base64(PNG). If it has fewer than
    ``min_pixels`` pixels, upscale it first (nearest-neighbour so the grid stays
    crisp, aspect ratio preserved)."""
    with Image.open(path) as im:
        im = im.convert("RGB")
        w, h = im.size
        if w * h < min_pixels:
            #scale = math.sqrt(min_pixels / (w * h))
            #im = im.resize((math.ceil(w * scale), math.ceil(h * scale)), Image.NEAREST)
            #better version
            scale = math.ceil(math.sqrt(min_pixels / (w * h)))
            im = im.resize((w * scale, h * scale), Image.NEAREST)

        
        buf = io.BytesIO()
        im.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def vision_query(filenames_list=[], message="", model=MY_MODEL, model_url=MY_URL,
                 response_format=None, think=True, num_predict=MAX_TOKEN_GENERATION_LIMIT):

    if filenames_list:
        image_list = [encode_image(filename) for filename in filenames_list]
    else:
        image_list = []

    payload = {
        "model": model,
        "prompt": message,
        "images": image_list,
        "options": {"num_predict": num_predict},  # limits max predicted tokens to avoid infinite loop
        "stream": True,                                          # to be able to capture the thinking process
        "think": think,                                          # per-call; default True
    }
    if response_format is not None:                             # Ollama structured output: "json" or a JSON schema
        payload["format"] = response_format

    print("Sending request to Ollama...")
    print(f"Model: {model}")
    print(f"think: {payload["think"]}")
    print(f"Images: {filenames_list}")
    print("Waiting for model...\n")

    r = requests.post(model_url,
        json=payload,
        stream=True,
        timeout=600,
    )

    r.raise_for_status()

    all_chunks = []
    final_response = ""
    thinking_response = ""
    for line in r.iter_lines():
        if not line:
            continue

        chunk = json.loads(line)
        all_chunks.append(chunk)

        # Show thinking separately
        thinking = chunk.get("thinking")
        if thinking:
#            print(f"[THINKING] {thinking}", end="", flush=True)
            print(thinking, end="", flush=True)
            thinking_response += thinking

        # Show generated answer
        text = chunk.get("response")
        if text:
            print(text, end="", flush=True)
            final_response += text

        # Final metadata
        if chunk.get("done"):
            print("\n\n--- DONE ---")

            if "total_duration" in chunk:
                print(
                    f"Total duration: "
                    f"{chunk['total_duration'] / 1e9:.1f}s"
                )

            if "eval_count" in chunk:
                print(f"Tokens generated: {chunk['eval_count']}")

            if "eval_duration" in chunk:
                print(
                    f"Generation speed: "
                    f"{chunk['eval_count'] / (chunk['eval_duration'] / 1e9):.1f} tokens/s"
                )

            break

    # some models inline the reasoning as <think>...</think> instead of using a
    # separate `thinking` field
    if not thinking_response:
        m = _THINK_RE.search(final_response)
        if m:
            thinking_response = m.group(1).strip()
            final_response = _THINK_RE.sub("", final_response).strip()

    return final_response, thinking_response, all_chunks
