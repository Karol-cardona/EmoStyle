import json, requests
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

# ── BLIP setup per funzioni diverse ──────────────────────────────
processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base", use_fast=True
)

# (A) modello solo‐vision, se vuoi ancora estrarre features in futuro
vision_model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base",
    use_safetensors=True
).vision_model
vision_model.eval()

if torch.cuda.is_available():
    vision_model.cuda()

# (B) modello full‐captioning per generare testo
caption_model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base",
    use_safetensors=True
).eval()
if torch.cuda.is_available():
    caption_model.cuda()

def blip_caption(image_path: str) -> str:
    """Generate a plain-text caption from the image using BLIP."""
    img = Image.open(image_path).convert("RGB")
    inputs = processor(images=img, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    out_ids = caption_model.generate(**inputs, max_new_tokens=40)
    return processor.decode(out_ids[0], skip_special_tokens=True)


# ── Prompt builder ─────────────────────────────────────────
def build_prompt(record: dict) -> str:
    """
    1) Get BLIP’s base caption (plain text).
    2) Append Emotion line.
    3) Ask LLaMA to rewrite for that emotion.
    """
    # base = blip_caption(f"data/images/{record['img_name']}")
    base = blip_caption(record["img_name"])

    return (
        f"Base caption: {base}\n"
        f"Emotion: {record['emotion']}\n"
        "Rewrite the above caption to fit the specified emotion:\n"
    )


# ── LLaMA via Ollama (resta identico) ────────────────────────────
def llama_generate(prompt: str,
                   model: str = "llama3.2",
                   temperature: float = 0.8,
                   max_length: int = 40) -> str:
    payload = {
        "model":       model,
        "prompt":      prompt,
        "temperature": temperature,
        "max_length":  max_length,
        "stream":      True
    }
    resp = requests.post("http://localhost:11434/api/generate",
                         json=payload, stream=True)
    resp.raise_for_status()
    full = []
    for line in resp.iter_lines(decode_unicode=True):
        if not line: continue
        data = json.loads(line)
        full.append(data.get("response", ""))
        if data.get("done", False):
            break
    return "".join(full).strip()
