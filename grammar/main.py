import os
import json
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Load dataset
with open("AraLingBench_train.json", "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f]

# Run on all 150 questions

model_id = "/home/zbibm/models/Qwen/Qwen3-8B"

tok = AutoTokenizer.from_pretrained(model_id)
llm = LLM(
    model=model_id,
    dtype="bfloat16",
    gpu_memory_utilization=0.35,
    enforce_eager=True
)
sampling_params = SamplingParams(max_tokens=128, temperature=0)

def build_prompt(item):
    context = item.get("context", "")
    question = item["question"]
    options = item["options"]

    options_text = "\n".join([f"{i+1}. {opt}" for i, opt in enumerate(options)])

    prompt = f"""أجب على السؤال التالي باختيار الإجابة الصحيحة.

{f"السياق: {context}" if context else ""}
السؤال: {question}

الخيارات:
{options_text}

أجب برقم الخيار الصحيح فقط (1، 2، 3، أو 4). /no_think"""

    return prompt

def extract_answer(response, options):
    response = response.strip()

    # Remove thinking blocks if present
    import re
    response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
    response = response.strip()

    # Try to find a number (1-4) at the start or as standalone
    for i, char in enumerate(response[:20]):  # Check first 20 chars
        if char.isdigit() and 1 <= int(char) <= len(options):
            return options[int(char) - 1]

    # Check if response contains one of the options
    for opt in options:
        if opt in response:
            return opt

    return response

wrong_answers = []
correct_count = 0
total = len(data)

print(f"Running benchmark on {total} questions...")

for item in tqdm(data):
    prompt = build_prompt(item)

    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = tok.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    outputs = llm.generate([formatted_prompt], sampling_params)
    response = outputs[0].outputs[0].text.strip()

    predicted = extract_answer(response, item["options"])
    correct = item["answer"]

    if predicted == correct:
        correct_count += 1
    else:
        wrong_answers.append({
            "label": item["label"],
            "context": item.get("context", ""),
            "question": item["question"],
            "options": item["options"],
            "correct_answer": correct,
            "model_answer": predicted,
            "raw_response": response,
            "difficulty": item.get("difficulty", "")
        })

print(f"\nResults: {correct_count}/{total} correct ({100*correct_count/total:.1f}%)")
print(f"Wrong answers: {len(wrong_answers)}")

# Save only wrong answers
with open("qwen3_wrong_answers.json", "w", encoding="utf-8") as f:
    json.dump(wrong_answers, f, ensure_ascii=False, indent=2)

print(f"\nWrong answers saved to qwen3_wrong_answers.json")
