import gradio as gr
import torch
import tempfile
from pathlib import Path

from models.blip import BlipCaptioner
from utils.sentiment import SentimentAnalyzer
from models.llama import LlamaClient

# Initialize models
device = "cuda" if torch.cuda.is_available() else "cpu"
blip = BlipCaptioner(device=device)
analyzer = SentimentAnalyzer()
llama = LlamaClient(
    model_name="llama3:3.2b",
    temperature=0.8,
    max_tokens=40
)

# Define the callback function
def generate_emostyled(image, input_text, selected_tone):
    """
    Given an image, optional user text, and optional tone, generate a base caption
    with BLIP, optionally detect tone from text, then generate a stylized caption with LLaMA.
    Returns base_caption, tone_used, stylized_caption.
    """
    # Save uploaded image to a temporary file
    tmp_file = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    image.save(tmp_file.name)
    img_path = tmp_file.name

    # 1) Generate base caption with BLIP
    base_caption = blip.generate_caption(img_path)

    # 2) Determine tone: use selected_tone or detect from input_text
    tone = selected_tone
    if input_text and not tone:
        result = analyzer.analyze(input_text)
        tone = result['label']
    tone = tone or "NEUTRAL"

    # 3) Build prompt for LLaMA
    prompt = (
        f"Base caption: {base_caption}\n"
        f"Emotion: {tone}\n"
        "Rewrite the above caption to fit the specified emotion:\n"
    )

    # 4) Generate stylized caption with LLaMA
    stylized_caption = llama.generate(prompt)

    return base_caption, tone, stylized_caption

# Set up Gradio interface
tone_choices = [
    "POSITIVE", "NEGATIVE", "NEUTRAL",
    "JOY", "SADNESS", "SURPRISE",
    "IRONIC", "POETIC", "INSPIRATIONAL"
]

demo = gr.Blocks()
with demo:
    gr.Markdown("# EmoStyle: Emotion-Aware Caption Generator")
    with gr.Row():
        image_input = gr.Image(type="pil", label="Upload Image")
        with gr.Column():
            input_text = gr.Textbox(
                label="Input Text (optional)",
                placeholder="Enter existing caption to detect tone..."
            )
            selected_tone = gr.Dropdown(
                choices=tone_choices,
                label="Select Emotion (optional)",
                value=None
            )
            generate_btn = gr.Button("Generate Caption")

    base_out = gr.Textbox(label="Base Caption")
    tone_out = gr.Textbox(label="Detected/Selected Tone")
    stylized_out = gr.Textbox(label="Stylized Caption")

    generate_btn.click(
        generate_emostyled,
        inputs=[image_input, input_text, selected_tone],
        outputs=[base_out, tone_out, stylized_out]
    )

# Launch the app
if __name__ == "__main__":
    demo.launch()
