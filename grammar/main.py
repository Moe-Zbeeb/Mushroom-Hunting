import os
import json
import re
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

# Thinking mode sampling params as per docs: Temperature=0.6, TopP=0.95, TopK=20
sampling_params = SamplingParams(
    max_tokens=2048,  # More tokens for thinking
    temperature=0.6,
    top_p=0.95,
    top_k=20,
    min_p=0
)

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

أجب برقم الخيار الصحيح فقط (1، 2، 3، أو 4)."""

    return prompt

def extract_thinking_and_answer(response, options):
    response = response.strip()

    # Extract thinking content
    think_match = re.search(r'<think>(.*?)</think>', response, flags=re.DOTALL)
    thinking_content = think_match.group(1).strip() if think_match else ""

    # Get the answer part (after </think>)
    answer_part = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()

    # Try to find a number (1-4)
    for char in answer_part[:50]:
        if char.isdigit() and 1 <= int(char) <= len(options):
            return thinking_content, options[int(char) - 1], answer_part

    # Check if answer contains one of the options
    for opt in options:
        if opt in answer_part:
            return thinking_content, opt, answer_part

    return thinking_content, answer_part, answer_part

wrong_answers = []
correct_count = 0
total = len(data)

print(f"Running benchmark on {total} questions with THINKING mode...")

for item in tqdm(data):
    prompt = build_prompt(item)

    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = tok.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True  # Enable thinking mode
    )

    outputs = llm.generate([formatted_prompt], sampling_params)
    response = outputs[0].outputs[0].text.strip()

    thinking, predicted, raw_answer = extract_thinking_and_answer(response, item["options"])
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
            "thinking_content": thinking,
            "raw_response": response,
            "difficulty": item.get("difficulty", "")
        })

print(f"\nResults: {correct_count}/{total} correct ({100*correct_count/total:.1f}%)")
print(f"Wrong answers: {len(wrong_answers)}")

# Save only wrong answers with thinking
with open("qwen3_wrong_answers_thinking.json", "w", encoding="utf-8") as f:
    json.dump(wrong_answers, f, ensure_ascii=False, indent=2)

print(f"\nWrong answers saved to qwen3_wrong_answers_thinking.json")
