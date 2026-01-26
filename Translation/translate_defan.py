#!/usr/bin/env python3
"""
Script to translate defan_no_math.json using Hala-1.2B-EN-AR-Translator.
Translates 'questions' and 'answer' fields from English to Arabic.
Uses multiple GPUs for parallel processing.
"""

import json
import os
from pathlib import Path
from tqdm import tqdm
import torch
import torch.multiprocessing as mp
from functools import partial

# Paths
MODEL_PATH = "/home/zbibm/Mushroom-Hunting/Hala-1.2B-EN-AR-Translator"
INPUT_FILE = "/home/zbibm/Mushroom-Hunting/defan_no_math.json"
OUTPUT_FILE = "/home/zbibm/Mushroom-Hunting/defan_no_math_ar.json"
CHECKPOINT_DIR = "/home/zbibm/Mushroom-Hunting/checkpoints"

# Settings
NUM_GPUS = 4
SAVE_EVERY = 200  # Save checkpoint every N entries per GPU
MAX_NEW_TOKENS = 512


def load_model(gpu_id):
    """Load the translation model and tokenizer on a specific GPU."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = f"cuda:{gpu_id}"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True
    )
    model.eval()

    return model, tokenizer, device


def translate_text(model, tokenizer, device, text, max_new_tokens=MAX_NEW_TOKENS):
    """Translate a single text from English to Arabic."""
    if not text or not text.strip():
        return text

    messages = [
        {
            "role": "user",
            "content": f"Translate everything that follows into Arabic:\n\n{text}",
        }
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
        )

    # Decode only the new tokens (excluding the prompt)
    generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
    translated = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    return translated.strip()


def get_checkpoint_path(gpu_id):
    """Get checkpoint file path for a specific GPU."""
    return os.path.join(CHECKPOINT_DIR, f"checkpoint_gpu{gpu_id}.json")


def load_checkpoint(gpu_id):
    """Load checkpoint if exists for a specific GPU."""
    checkpoint_path = get_checkpoint_path(gpu_id)
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'r', encoding='utf-8') as f:
            checkpoint = json.load(f)
        return checkpoint['last_index'], checkpoint['translated_data']
    return 0, []


def save_checkpoint(gpu_id, index, translated_data):
    """Save checkpoint for a specific GPU."""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    checkpoint = {
        'last_index': index,
        'translated_data': translated_data
    }
    checkpoint_path = get_checkpoint_path(gpu_id)
    with open(checkpoint_path, 'w', encoding='utf-8') as f:
        json.dump(checkpoint, f, ensure_ascii=False, indent=2)


def worker_process(gpu_id, data_chunk, chunk_indices, return_dict):
    """Worker process for a single GPU."""
    try:
        print(f"[GPU {gpu_id}] Loading model...")
        model, tokenizer, device = load_model(gpu_id)
        print(f"[GPU {gpu_id}] Model loaded on {device}")

        # Load checkpoint
        start_local_idx, translated_data = load_checkpoint(gpu_id)

        if start_local_idx >= len(data_chunk):
            print(f"[GPU {gpu_id}] Already complete!")
            return_dict[gpu_id] = translated_data
            return

        print(f"[GPU {gpu_id}] Processing {len(data_chunk)} entries (resuming from {start_local_idx})...")

        pbar = tqdm(
            range(start_local_idx, len(data_chunk)),
            desc=f"GPU {gpu_id}",
            position=gpu_id,
            leave=True
        )

        for local_idx in pbar:
            entry = data_chunk[local_idx]

            # Translate questions
            translated_question = translate_text(
                model, tokenizer, device,
                entry.get('questions', ''),
                max_new_tokens=MAX_NEW_TOKENS
            )

            # Translate answer
            translated_answer = translate_text(
                model, tokenizer, device,
                entry.get('answer', ''),
                max_new_tokens=256
            )

            # Create translated entry with original index for ordering
            translated_entry = {
                'original_index': chunk_indices[local_idx],
                'questions': translated_question,
                'answer': translated_answer,
                'type': entry.get('type', ''),
                'domain': entry.get('domain', '')
            }
            translated_data.append(translated_entry)

            # Save checkpoint periodically
            if (local_idx + 1) % SAVE_EVERY == 0:
                save_checkpoint(gpu_id, local_idx + 1, translated_data)

        # Final save
        save_checkpoint(gpu_id, len(data_chunk), translated_data)
        return_dict[gpu_id] = translated_data
        print(f"[GPU {gpu_id}] Completed {len(translated_data)} entries")

    except Exception as e:
        print(f"[GPU {gpu_id}] Error: {e}")
        import traceback
        traceback.print_exc()
        return_dict[gpu_id] = []


def main():
    # Set multiprocessing start method
    mp.set_start_method('spawn', force=True)

    # Load input data
    print(f"Loading input data from {INPUT_FILE}...")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    total_entries = len(data)
    print(f"Total entries to translate: {total_entries}")
    print(f"Using {NUM_GPUS} GPUs")

    # Split data across GPUs
    chunk_size = (total_entries + NUM_GPUS - 1) // NUM_GPUS
    data_chunks = []
    chunk_indices = []

    for i in range(NUM_GPUS):
        start_idx = i * chunk_size
        end_idx = min(start_idx + chunk_size, total_entries)
        data_chunks.append(data[start_idx:end_idx])
        chunk_indices.append(list(range(start_idx, end_idx)))
        print(f"GPU {i}: entries {start_idx} to {end_idx-1} ({end_idx - start_idx} entries)")

    # Create shared dictionary for results
    manager = mp.Manager()
    return_dict = manager.dict()

    # Start worker processes
    processes = []
    for gpu_id in range(NUM_GPUS):
        p = mp.Process(
            target=worker_process,
            args=(gpu_id, data_chunks[gpu_id], chunk_indices[gpu_id], return_dict)
        )
        p.start()
        processes.append(p)

    # Wait for all processes to complete
    for p in processes:
        p.join()

    print("\nAll GPUs completed. Merging results...")

    # Merge results and sort by original index
    all_translated = []
    for gpu_id in range(NUM_GPUS):
        if gpu_id in return_dict:
            all_translated.extend(return_dict[gpu_id])

    # Sort by original index to maintain order
    all_translated.sort(key=lambda x: x['original_index'])

    # Remove the original_index field from final output
    final_data = []
    for entry in all_translated:
        final_entry = {
            'questions': entry['questions'],
            'answer': entry['answer'],
            'type': entry['type'],
            'domain': entry['domain']
        }
        final_data.append(final_entry)

    # Save final output
    print(f"Saving {len(final_data)} translated entries to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(final_data, f, ensure_ascii=False, indent=2)

    # Clean up checkpoint files
    print("Cleaning up checkpoints...")
    for gpu_id in range(NUM_GPUS):
        checkpoint_path = get_checkpoint_path(gpu_id)
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)

    if os.path.exists(CHECKPOINT_DIR) and not os.listdir(CHECKPOINT_DIR):
        os.rmdir(CHECKPOINT_DIR)

    print(f"\nTranslation complete! Total entries: {len(final_data)}")


if __name__ == "__main__":
    main()
