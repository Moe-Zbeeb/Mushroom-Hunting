#!/usr/bin/env python3
"""
Qwen3-14B Batch Inference Script
Loads prompts from JSON, runs inference, saves results with labels and model outputs.
"""

import os
import json
import sys

# Force GPU 1 only (has more free memory)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from vllm import LLM, SamplingParams


def load_prompts(json_path: str) -> list:
    """
    Load prompts from JSON file.
    Expected format:
    [
        {"input": "prompt text", "output1": "your label"},
        ...
    ]
    """
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_results(results: list, output_path: str):
    """Save results to JSON file."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def main():
    if len(sys.argv) < 2:
        print("Usage: python talk_with_qwen.py <prompts.json> [output.json]")
        print("\nExpected JSON format:")
        print('[{"input": "prompt text", "output1": "your label"}, ...]')
        sys.exit(1)

    input_json = sys.argv[1]
    output_json = sys.argv[2] if len(sys.argv) > 2 else "results.json"

    # Load prompts
    print(f"Loading prompts from: {input_json}")
    prompts_data = load_prompts(input_json)
    print(f"Loaded {len(prompts_data)} prompts")

    # Initialize model
    print("\nLoading Qwen3-14B model (GPU 0 only)...")
    llm = LLM(
        model="Qwen/Qwen3-14B",
        gpu_memory_utilization=0.35,
        max_model_len=4096,
        enforce_eager=True
    )

    # Sampling parameters
    sampling_params = SamplingParams(
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        max_tokens=2000
    )

    # Extract prompts for batch inference
    prompts = [item["input"] for item in prompts_data]

    # Run batch inference
    print(f"\nRunning inference on {len(prompts)} prompts...")
    outputs = llm.generate(prompts, sampling_params)

    # Build results
    results = []
    for i, output in enumerate(outputs):
        result = {
            "input": prompts_data[i]["input"],
            "output1": prompts_data[i].get("output1", ""),
            "output2": output.outputs[0].text
        }
        results.append(result)
        print(f"Processed {i+1}/{len(prompts)}")

    # Save results
    save_results(results, output_json)
    print(f"\nResults saved to: {output_json}")


if __name__ == "__main__":
    main()
