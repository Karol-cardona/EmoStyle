import os
import json
import argparse
import torch
import random
import numpy as np
import time
import matplotlib.pyplot as plt

from pathlib import Path

from gradio.themes.builder_app import history
from tqdm import tqdm
import pandas as pd
from datasets import Dataset as HFDataset

from generate import llama_generate

from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    BlipForConditionalGeneration,
    BlipProcessor,
    get_scheduler,
    EvalPrediction,
    BitsAndBytesConfig,
    logging as hf_logging,
    CLIPProcessor,
    CLIPModel,
    default_data_collator
)
from peft import LoraConfig, get_peft_model, TaskType
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, RandomHorizontalFlip
from PIL import Image


# ── Funzione 1: Prompt-Tuning LoRA ────────────────────────────────────
def run_lora_train(
        train_file: str,
        output_dir: str,
        model_name: str,
        model_path: str = None,
        seed: int = 42,
        use_4bit: bool = False,
        batch_size: int = 2,
        use_bfloat16: bool = True
):
    # ── 0) Riproducibilità
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # ── 1) Dataset aumentato se presente
    aug = Path(train_file).parent / "train_aug.json"
    if aug.exists():
        train_file = str(aug)

    records = json.load(open(train_file, encoding="utf-8"))
    print(f"[LoRA] Esempi da {train_file}: {len(records)}")

    # ── 2) Tokenizer
    src = model_path or model_name
    tokenizer = LlamaTokenizer.from_pretrained(src)
    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})

    # ── 3) Prepara HF Dataset, shuffle+split
    examples = []
    for r in records:
        inp = f"Base caption: {r['caption']}\nEmotion: {r['emotion']}\nRewrite the above caption to fit the specified emotion:\n"
        examples.append({"input_text": inp, "target_text": r["caption"]})

    # 1) Trasforma la lista di dict (examples) in un dict di liste
    data_dict = {
        "input_text":  [e["input_text"]  for e in examples],
        "target_text": [e["target_text"] for e in examples],
    }

    # 2) Crea il dataset HF e fai lo shuffle
    df = pd.DataFrame(data_dict)
    ds = HFDataset.from_pandas(df)
    ds = ds.shuffle(seed=seed)
    splits = ds.train_test_split(test_size=0.05, seed=seed)
    print(f"[LoRA] Train/Val sizes = {len(splits['train'])}/{len(splits['test'])}")

    # ── 4) Log dataset tokenizzato in CSV per debug
    df_debug = splits["train"].to_pandas()
    df_debug.to_csv(Path(output_dir) / "lora_dataset_debug.csv", index=False)
    print(f"[LoRA] Dataset tokenization debug salvato in lora_dataset_debug.csv")

    # ── 5) Tokenizzazione
    def tokenize_fn(batch):
        inps = tokenizer(batch["input_text"], truncation=True, padding="max_length", max_length=128)
        labs = tokenizer(batch["target_text"], truncation=True, padding="max_length", max_length=64)
        inps["labels"] = labs["input_ids"]
        return inps

    tokenized = splits.map(
        tokenize_fn, batched=True,
        remove_columns=["input_text", "target_text"]
    )

    # ── 6) Carica modello (4bit o bfloat16)
    kwargs = {"device_map": "auto"}
    if use_4bit:
        kwargs.update({
            "load_in_4bit": True,
            "quantization_config": {
                "bnb_4bit_compute_dtype": torch.bfloat16 if use_bfloat16 else torch.float16,
                "bnb_4bit_use_double_quant": True,
                "bnb_4bit_quant_type": "nf4"
            }
        })
    else:
        kwargs["torch_dtype"] = torch.bfloat16 if use_bfloat16 else torch.float16

    # carica interamente in FP16 (o BF16) e sposta su GPU/CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[LoRA] Device: {device}")

    # (1) attiva il logging “DEBUG” dei Transformers per avere info aggiuntive
    hf_logging.set_verbosity_debug()

    bnb_config = BitsAndBytesConfig(
        load_in_4bit            = True,
        bnb_4bit_compute_dtype  = torch.bfloat16,
        bnb_4bit_use_double_quant = True,
        bnb_4bit_quant_type     = "nf4",
        llm_int8_enable_fp32_cpu_offload = True
    )

    print("[LoRA] ⚙️  Preparazione caricamento modello…")
    t0 = time.time()

    model = LlamaForCausalLM.from_pretrained(
        src,
        quantization_config = bnb_config,
        tquantization_config = bnb_config,
        device_map = "auto",
        low_cpu_mem_usage = True
    )
    print(f"[LoRA] ✅ from_pretrained in {time.time()-t0:.1f}s")
    t1 = time.time()

    print(f"[LoRA] ✅ device mapping completato in {time.time()-t1:.1f}s")

    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    # ── 7) Applica LoRA
    lora_conf = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"]
    )
    model = get_peft_model(model, lora_conf)

    # ── 8) Definizione di compute_metrics per accuracy token-level
    def compute_metrics(eval_pred: EvalPrediction):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        mask = labels != -100  # -100 è il token di padding
        correct = (predictions == labels) & mask
        accuracy = correct.sum() / mask.sum() if mask.sum() > 0 else 0.0
        return {"accuracy": float(accuracy)}

    # ── 8) TrainingArguments con TensorBoard, save_steps e early stopping
    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        learning_rate=1e-4,
        fp16=not use_bfloat16,
        bf16=use_bfloat16,
        logging_steps=50,
        save_steps=200,                # checkpoint frequenti per debug
        eval_strategy="steps",
        eval_steps=200,
        save_total_limit=3,            # conserva solo 3 checkpoint
        report_to="tensorboard",       # traccia su TensorBoard
        load_best_model_at_end=True,   # early stopping carica il best
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=seed
    )
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        data_collator=collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)]
    )

    # ── 9) Avvia training
    trainer.train()

    # ── 10) Salvataggio adapter + tokenizer
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"[LoRA] Adapter LoRA e tokenizer salvati in {output_dir}")

    # ── 11) Plot training loss e eval accuracy
    # Estrai le metriche dal trainer
    history = trainer.state.log_history

    #Filtra le metriche di interesse
    steps = [h["step"] for h in history if "loss" in h]
    train_losses = [h["loss"] for h in history if "loss" in h]
    eval_accs   = [h["eval_accuracy"] for h in history if "eval_accuracy" in h]
    eval_steps  = [h["step"] for h in history if "eval_accuracy" in h]

    plt.figure(figsize=(8, 4))
    plt.plot(steps, train_losses, label="Train Loss")
    plt.plot(eval_steps, eval_accs, label="Eval Accuracy")
    plt.xlabel("Steps")
    plt.title("LoRA Training: Loss & Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "lora_loss_acc_curve.png")
    plt.show()
    print(f"[LoRA] Curve salvate in {output_dir}/lora_loss_acc_curve.png")

    #Callcola e mostra lla perplessita finale
    if train_losses:
        perplexity = float(np.exp(train_losses[-1]))
        print(f"[LoRA] Perplexity finale (train): {perplexity:.2f}")

    """ Per farlo partire da solo eseguire: python training.py --do_lora """


# ── Funzione 2: Data Augmentation (Paraphrase) ───────────────────────────
def run_paraphrase(
        train_file: str,
        output_file: str = None,
        sleep_interval: float = 0.5,
        sleep_every: int = 100
):
    """
    Data‐augmentation: per ogni record in train_file, genera una
    parafrasi mantenendo l’emozione target via llama_generate.
    - Non duplica caption uguali al gold o già generate.
    - Rate‐limiter: ogni `sleep_every` chiamate, dorme `sleep_interval` sec.
    - Salva il dataset aumentato (originali + paraphrases) in output_file.
    """
    # Imposta percorso di default in data/splits/train_aug.json
    if output_file is None:
        output_file = Path(train_file).parent / "train_aug.json"
    else:
        output_file = Path(output_file)

    # Crea la cartella di output se non esiste
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Carica i record originali
    with open(train_file, encoding="utf-8") as f:
        records = json.load(f)
    print(f"[Paraphrase] Caricati {len(records)} esempi da {train_file}")

    augmented = []
    seen_paras = set()

    # Generazione paraphrases con rate‐limiting e dedup
    for i, r in enumerate(tqdm(records, desc="Paraphrasing", unit="rec")):
        base = r["caption"].strip()
        emo  = r["emotion"]
        prompt = (
            f"Original caption: “{base}”\n"
            f"Emotion: {emo}\n"
            "Paraphrase the above caption in a single sentence with the same emotion:\n"
        )
        try:
            para = llama_generate(prompt, temperature=0.9, max_length=64).strip()
        except Exception as e:
            print(f"[Paraphrase] Errore su record {i}: {e}")
            continue

        # Filtra duplicati e caption vuote o identiche all’originale
        if para and para != base and para not in seen_paras:
            augmented.append({
                "img_name": r["img_name"],
                "caption":  para,
                "emotion":  emo
            })
            seen_paras.add(para)

        # Rate‐limiter per non sovraccaricare Ollama
        if sleep_interval and sleep_every and i and i % sleep_every == 0:
            time.sleep(sleep_interval)

    # Unisci originali + parafrasi
    merged = records + augmented
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    print(f"[Paraphrase] Generate {len(augmented)} nuove parafrasi")
    print(f"[Paraphrase] Totale esempi dopo augment: {len(merged)}")
    print(f"[Paraphrase] Salvato in: {output_file}")

    """ Per farlo partire da solo eseguire: python training.py --do_paraphrase """

# ── Classe helper per il dataset BLIP ─────────────────────────────────
class EmoBlipDataset(Dataset):
    def __init__(
            self,
            json_path: str,
            processor,
            image_size: int = 224,
            max_length: int = 40,
            do_augment: bool = False
    ):
        """
        Dataset per il fine-tuning di BLIP decoder.

        Args:
          json_path: percorso a train.json o train_aug.json
          processor: BlipProcessor (o CLIPProcessor) che gestisce testo+immagine
          image_size: dimensione quadrata dell’immagine
          max_length: lunghezza massima della didascalia (token)
          do_augment: se True applica flip orizzontale con p=0.5
        """
        self.records = json.load(open(json_path, encoding="utf-8"))
        self.processor = processor
        self.max_length = max_length

        # transform immagine
        transforms = [
            Resize(image_size, interpolation=Image.BICUBIC),
            CenterCrop(image_size),
        ]
        if do_augment:
            transforms.insert(0, RandomHorizontalFlip(p=0.5))
        transforms += [
            ToTensor(),
            Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711)
            )
        ]
        self.transform = Compose(transforms)

        print(f"[EmoBlipDataset] Caricati {len(self.records)} esempi da {json_path}")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int):
        r = self.records[idx]
        img = Image.open(r["img_name"]).convert("RGB")
        img_transformed = self.transform(img)

        # processor gestisce sia pixel_values sia tokenizzazione testo
        # qui passiamo l’immagine preprocessata via 'pixel_values' e il testo
        inputs = self.processor(
            text=r["caption"],
            images=[img],  # serve passare lista per alcuni processor
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length
        )
        # inputs contiene: pixel_values, input_ids, attention_mask
        # ciascuno di shape [1, ...], quindi rimuoviamo la dim batch
        return {
            "pixel_values":     img_transformed,  # [3, H, W]
            "input_ids":        inputs.input_ids.squeeze(0),        # [L]
            "attention_mask":   inputs.attention_mask.squeeze(0),   # [L]
        }


# ── Funzione 3: Fine-tuning BLIP decoder ───────────────────────────────

def run_blip_finetune(
        train_file: str,
        output_dir: str,
        image_size: int = 224,
        max_length: int = 40,
        batch_size: int = 16,
        num_epochs: int = 5,
        lr: float = 5e-5,
        warmup_ratio: float = 0.1,
        early_stopping_patience: int = 2,
        do_augment: bool = False
):
    """
    Fine-tuning del decoder BLIP, congelando il vision encoder.
    Tiene traccia di train/val loss, applica early stopping,
    e salva un plot delle curve alla fine.
    """
    # 1) Prepara device e output
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    print(f"[BLIP] Device: {device}")

    # 2) Carica modello e processor
    processor = BlipProcessor.from_pretrained(
        "Salesforce/blip-image-captioning-base", use_fast=True
    )
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base", use_safetensors=True
    )
    model.to(device)

    # 3) Congela il vision encoder
    for name, param in model.named_parameters():
        if "vision_model" in name:
            param.requires_grad = False


    dataset = EmoBlipDataset(train_file)
    # split 95/5
    n = len(dataset)
    cut = int(0.95 * n)
    train_ds = Subset(dataset, list(range(cut)))
    val_ds   = Subset(dataset, list(range(cut, n)))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=4)

    # 5) Ottimizzatore e scheduler
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    total_steps = num_epochs * len(train_loader)
    scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=int(warmup_ratio * total_steps),
        num_training_steps=total_steps
    )

    # 6) Loop di training con early stopping
    best_val_loss = float("inf")
    patience = 0
    train_losses, val_losses = [], []

    for epoch in range(num_epochs):
        # ---- TRAIN ----
        model.train()
        running_train = 0.0
        for batch in tqdm(train_loader, desc=f"[BLIP] Epoch {epoch} Train"):
            optimizer.zero_grad()
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch, labels=batch["input_ids"])
            loss = out.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            running_train += loss.item()

        avg_train = running_train / len(train_loader)
        train_losses.append(avg_train)
        print(f"[BLIP] Epoch {epoch} → train loss: {avg_train:.4f}")

        # ---- VALID ----
        model.eval()
        running_val = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"[BLIP] Epoch {epoch} Val"):
                batch = {k: v.to(device) for k, v in batch.items()}
                out = model(**batch, labels=batch["input_ids"])
                running_val += out.loss.item()

        avg_val = running_val / len(val_loader)
        val_losses.append(avg_val)
        print(f"[BLIP] Epoch {epoch} → val loss:   {avg_val:.4f}")

        # early stopping check
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            patience = 0
            # salva checkpoint best
            best_dir = Path(output_dir) / "best"
            model.save_pretrained(best_dir)
            processor.save_pretrained(best_dir)
            print(f"[BLIP] New best model saved in {best_dir}")
        else:
            patience += 1
            if patience > early_stopping_patience:
                print(f"[BLIP] Early stopping (patience={patience})")
                break

        # salva checkpoint di questa epoca
        ckpt_dir = Path(output_dir) / f"epoch_{epoch}"
        model.save_pretrained(ckpt_dir)
        processor.save_pretrained(ckpt_dir)

    # 7) Plot delle curve di loss
    plt.figure(figsize=(6,4))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses,   label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("BLIP Decoder Fine-tuning Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "loss_curve.png")
    plt.show()

    print(f"[BLIP] Fine-tuning completato. Curve salvate in {output_dir}/loss_curve.png")


# ── Funzione main + argparser ───────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="EmoStyle Training Pipeline")
    parser.add_argument(
        "--train_file", type=str,
        default="data/splits/train.json",
        help="JSON di input per training (dati originali o augmentati)."
    )
    parser.add_argument(
        "--output_dir", type=str, default="checkpoints",
        help="Directory base per salvare checkpoint e risultati."
    )
    parser.add_argument(
        "--model_name", type=str,
        default="huggyllama/llama-7b",
        help="Nome HF del modello LLaMA per LoRA (default: huggyllama/llama-7b)."
    )
    parser.add_argument(
        "--model_path", type=str, default=None,
        help="Percorso locale a modello LLaMA esportato (es. Ollama HF dir)."
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Seed globale per riproducibilità."
    )
    parser.add_argument(
        "--batch_size", type=int, default=2,
        help="Batch size per LoRA training (ridotto se poca VRAM)."
    )
    parser.add_argument(
        "--use_4bit", action="store_true",
        help="Attiva quantizzazione 4-bit (richiede bitsandbytes GPU)."
    )
    parser.add_argument(
        "--do_paraphrase", action="store_true",
        help="Genera dataset parafrasato (train_aug.json) prima del training."
    )
    parser.add_argument(
        "--do_lora", action="store_true",
        help="Esegui prompt‐tuning LoRA su LLaMA."
    )
    parser.add_argument(
        "--do_blip", action="store_true",
        help="Fine‐tuning del decoder BLIP."
    )
    parser.add_argument(
        "--do_augment", action="store_true",
        help="Applica augmentazione (flip) nel BLIP fine‐tuning."
    )
    args = parser.parse_args()

    # Se nessun flag è specificato, esegui tutto in sequenza
    if not (args.do_paraphrase or args.do_lora or args.do_blip):
        args.do_paraphrase = True
        args.do_lora       = True
        args.do_blip       = True

    # 1) Paraphrase (data augmentation)
    if args.do_paraphrase:
        out_aug = Path("data/splits") / "train_aug.json"
        run_paraphrase(
            train_file=args.train_file,
            output_file=str(out_aug)
        )
        # usa il dataset augmentato per le fasi successive
        args.train_file = str(out_aug)

    # 2) LoRA prompt‐tuning
    if args.do_lora:
        out_lora = Path(args.output_dir) / "lora_llama"
        out_lora.mkdir(parents=True, exist_ok=True)
        run_lora_train(
            train_file=args.train_file,
            output_dir=str(out_lora),
            model_name=args.model_name,
            model_path=args.model_path,
            seed=args.seed,
            use_4bit=args.use_4bit,
            batch_size=args.batch_size
        )

    # 3) BLIP decoder fine‐tuning
    if args.do_blip:
        out_blip = Path(args.output_dir) / "blip_decoder"
        out_blip.mkdir(parents=True, exist_ok=True)
        run_blip_finetune(
            train_file=args.train_file,
            output_dir=str(out_blip),
            do_augment=args.do_augment
        )

if __name__ == "__main__":
    main()
