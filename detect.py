import json
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

dataset = load_dataset("openguardrails/OpenGuardrailsMixZh_97k", split="test")

model_name = "openguardrails/OpenGuardrails-Text-4B-0124"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

results = []

for i in range(5):
    prompt = dataset[i]["prompt_en"]

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=10)

    response = tokenizer.decode(
        outputs[0][len(inputs.input_ids[0]):],
        skip_special_tokens=True
    ).strip()

    print(f"Prompt {i}: {prompt}")
    print(f"Response {i}: {response}\n")
    
    results.append({
        "id": i,
        "prompt": prompt,
        "response": response
    })

os.makedirs("outputs", exist_ok=True)
with open("outputs/results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=4)

print("保存が完了しました（results.json）")
