#!/usr/bin/env python3
"""
Parallel translation script for defan_no_math.json using vLLM with Hala-1.2B-EN-AR-Translator.
Outputs question/answer/category fields only.
"""

import json
import os
import torch.multiprocessing as mp
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from tqdm import tqdm
import argparse


def build_prompt(text, tokenizer):
    messages = [
        {
            "role": "user",
            "content": "Translate everything that follows into Arabic:\n\n" + text,
        }
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def translate_texts(texts, llm, tokenizer, sampling_params):
    prompts = [build_prompt(t, tokenizer) for t in texts]
    outputs = llm.generate(prompts, sampling_params)
    return [out.outputs[0].text.strip() for out in outputs]


def worker(gpu_id, data_chunk, output_file, model_path, batch_size=8):
    """Worker process for a single GPU."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    print(f"[GPU {gpu_id}] Loading model with vLLM...")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    llm = LLM(
        model=model_path,
        dtype="bfloat16",
        gpu_memory_utilization=0.9
    )
    sampling_params = SamplingParams(max_tokens=512, temperature=0)

    print(f"[GPU {gpu_id}] Starting translation of {len(data_chunk)} items...")

    translated_data = []

    for i in tqdm(range(0, len(data_chunk), batch_size), desc=f"GPU {gpu_id}", position=gpu_id):
        batch = data_chunk[i:i + batch_size]

        questions = []
        answers = []
        categories = []

        for item in batch:
            question_text = item.get("question", item.get("questions", ""))
            answer_text = str(item.get("answer", ""))
            category = item.get("category")
            if category is None:
                if "type" in item:
                    category = item["type"]
                elif "domain" in item:
                    category = item["domain"]

            questions.append(question_text)
            answers.append(answer_text)
            categories.append(category)

        translated_questions = translate_texts(questions, llm, tokenizer, sampling_params)
        translated_answers = translate_texts(answers, llm, tokenizer, sampling_params)

        for tq, ta, cat in zip(translated_questions, translated_answers, categories):
            translated_data.append({
                "question": tq,
                "answer": ta,
                "category": cat
            })

        # Periodically save progress
        if len(translated_data) % 100 == 0:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(translated_data, f, ensure_ascii=False, indent=2)

    # Final save
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(translated_data, f, ensure_ascii=False, indent=2)

    print(f"[GPU {gpu_id}] Completed! Saved {len(translated_data)} items to {output_file}")
    return output_file


def merge_results(output_files, final_output):
    """Merge all partial results into one final file."""
    all_data = []
    for f in sorted(output_files):
        if os.path.exists(f):
            with open(f, 'r', encoding='utf-8') as fp:
                all_data.extend(json.load(fp))

    with open(final_output, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)

    print(f"Merged {len(all_data)} items into {final_output}")
    return all_data


def main():
    parser = argparse.ArgumentParser(description="Parallel translation using multiple GPUs")
    parser.add_argument("--input", type=str, default="/home/zbibm/Mushroom-Hunting/defan_no_math.json",
                        help="Input JSON file")
    parser.add_argument("--output", type=str, default="/home/zbibm/Mushroom-Hunting/defan_no_math_ar.json",
                        help="Output JSON file")
    parser.add_argument("--model", type=str, default="/home/zbibm/Mushroom-Hunting/Hala-1.2B-EN-AR-Translator",
                        help="Path to translator model")
    parser.add_argument("--gpus", type=str, default="0,1,2,3",
                        help="Comma-separated GPU IDs to use")
    parser.add_argument("--start-idx", type=int, default=0,
                        help="Start index (for resuming)")
    parser.add_argument("--end-idx", type=int, default=None,
                        help="End index (optional)")
    args = parser.parse_args()

    # Parse GPU list
    gpu_ids = [int(x) for x in args.gpus.split(",")]
    num_gpus = len(gpu_ids)

    print(f"Using {num_gpus} GPUs: {gpu_ids}")

    # Load data
    print("Loading input data...")
    with open(args.input, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Apply index range
    if args.end_idx:
        data = data[args.start_idx:args.end_idx]
    else:
        data = data[args.start_idx:]

    print(f"Total items to translate: {len(data)}")

    # Split data among GPUs
    chunk_size = len(data) // num_gpus
    chunks = []
    output_files = []

    for i, gpu_id in enumerate(gpu_ids):
        start = i * chunk_size
        end = start + chunk_size if i < num_gpus - 1 else len(data)
        chunks.append(data[start:end])
        output_files.append(f"{args.output}.part{gpu_id}")
        print(f"GPU {gpu_id}: items {start} to {end} ({end - start} items)")

    # Start workers
    mp.set_start_method('spawn', force=True)
    processes = []

    for gpu_id, chunk, out_file in zip(gpu_ids, chunks, output_files):
        p = mp.Process(target=worker, args=(gpu_id, chunk, out_file, args.model))
        p.start()
        processes.append(p)

    # Wait for all to complete
    for p in processes:
        p.join()

    # Merge results
    print("\nMerging results...")
    merge_results(output_files, args.output)

    # Optionally clean up partial files
    for f in output_files:
        if os.path.exists(f):
            os.remove(f)

    print("Done!")


if __name__ == "__main__":
    main()
