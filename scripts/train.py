#!/usr/bin/env python3
"""Training script for NeuroThinker model."""
import argparse, json, math, os, sys, time
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "model"))
from neurothinker import NeuroThinkerConfig, NeuroThinkerModel


class CoTDataset(Dataset):
    """Loads JSONL dataset and tokenizes for causal LM training."""

    def __init__(self, path: str, tokenizer, max_len: int = 512):
        self.samples = []
        self.max_len = max_len
        self.tokenizer = tokenizer

        with open(path, "r") as f:
            for line in f:
                row = json.loads(line)
                text = (
                    f"<|prompt|>{row['prompt']}"
                    f"<|think|>{row['thinking_cot']}"
                    f"<|answer|>{row['answer']}<|end|>"
                )
                ids = tokenizer.encode(text, add_special_tokens=False)
                if len(ids) > max_len:
                    ids = ids[:max_len]
                self.samples.append(ids)

        print(f"Loaded {len(self.samples)} samples (max_len={max_len})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ids = self.samples[idx]
        # Pad to max_len
        pad_len = self.max_len - len(ids)
        input_ids = ids + [0] * pad_len  # 0 = pad token
        labels = ids + [-100] * pad_len  # -100 = ignore in loss
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


def get_tokenizer():
    """Get or create tokenizer. Uses sentencepiece-based tokenizer."""
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        # Add special tokens
        special_tokens = {
            "additional_special_tokens": [
                "<|prompt|>", "<|think|>", "<|answer|>", "<|end|>"
            ]
        }
        tokenizer.add_special_tokens(special_tokens)
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer
    except ImportError:
        print("ERROR: transformers library required for tokenizer.")
        print("Install: pip install transformers")
        sys.exit(1)


def train(args):
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Tokenizer
    tokenizer = get_tokenizer()
    vocab_size = len(tokenizer)
    print(f"Vocabulary size: {vocab_size}")

    # Config & Model
    config = NeuroThinkerConfig(vocab_size=vocab_size, max_seq_len=args.max_len)
    model = NeuroThinkerModel(config).to(device)

    # Dataset
    dataset = CoTDataset(args.data, tokenizer, max_len=args.max_len)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=0, pin_memory=True)

    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay, betas=(0.9, 0.95))
    total_steps = len(loader) * args.epochs
    warmup_steps = min(args.warmup_steps, total_steps // 5)

    def lr_schedule(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    # Training loop
    print(f"\n{'='*60}")
    print(f"Training NeuroThinker")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Total steps: {total_steps}")
    print(f"  Warmup steps: {warmup_steps}")
    print(f"{'='*60}\n")

    best_loss = float("inf")
    global_step = 0
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, (input_ids, labels) in enumerate(loader):
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            output = model(input_ids, labels=labels)
            loss = output["loss"]

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            num_batches += 1
            global_step += 1

            if global_step % args.log_every == 0:
                avg = epoch_loss / num_batches
                lr = scheduler.get_last_lr()[0]
                elapsed = time.time() - start_time
                print(f"  Step {global_step:5d} | Loss: {loss.item():.4f} | "
                      f"Avg: {avg:.4f} | LR: {lr:.6f} | Time: {elapsed:.0f}s")

        avg_loss = epoch_loss / max(1, num_batches)
        print(f"\nEpoch {epoch}/{args.epochs} — Avg Loss: {avg_loss:.4f}")

        # Save best
        if avg_loss < best_loss:
            best_loss = avg_loss
            model.save_pretrained(args.output)
            # Also save tokenizer
            tokenizer.save_pretrained(args.output)
            print(f"  ✅ Saved best model (loss={best_loss:.4f})")

        # Generate sample
        if epoch % args.sample_every == 0:
            model.eval()
            prompt = "<|prompt|>What is 2 + 2?<|think|>"
            ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                gen = model.generate(ids, max_new_tokens=64, temperature=0.7)
            text = tokenizer.decode(gen[0], skip_special_tokens=False)
            print(f"  Sample: {text[:200]}...")
            model.train()

    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Training complete in {total_time:.0f}s")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Model saved to: {args.output}")
    print(f"{'='*60}")


def main():
    p = argparse.ArgumentParser(description="Train NeuroThinker")
    p.add_argument("--data", default="data/train.jsonl", help="Path to JSONL dataset")
    p.add_argument("--output", default="model/neurothinker_trained", help="Output directory")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--max_len", type=int, default=512)
    p.add_argument("--warmup_steps", type=int, default=200)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--sample_every", type=int, default=2)
    main_args = p.parse_args()
    train(main_args)


if __name__ == "__main__":
    main()
