import os
import json
from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

ds = load_dataset("domenicrosati/TruthfulQA", split="train")
ds = ds.select_columns(["Question", "Best Answer"])

model_id = "hammh0a/Hala-1.2B-EN-AR-Translator"

tok = AutoTokenizer.from_pretrained(model_id)
llm = LLM(
    model=model_id,
    dtype="bfloat16",
    gpu_memory_utilization=0.4
)
sampling_params = SamplingParams(max_tokens=256, temperature=0)

def translate(text):
    messages = [
        {
            "role": "user",
            "content": "Translate everything that follows into Arabic:\n\n" + text
        }
    ]
    prompt = tok.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    outputs = llm.generate([prompt], sampling_params)
    return outputs[0].outputs[0].text.strip()

arabic_data = []

for row in tqdm(ds):
    arabic_data.append({
        "question": translate(row["Question"]),
        "answer": translate(row["Best Answer"])
    })

with open("truthfulqa_ar.json", "w", encoding="utf-8") as f:
    json.dump(arabic_data, f, ensure_ascii=False, indent=2)
