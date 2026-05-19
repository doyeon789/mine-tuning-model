from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

base_model_id = "Qwen/Qwen2.5-7B-Instruct"
adapter_path = "./models"

# ✅ 학습 때랑 똑같이 4bit 양자화로 로드
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(adapter_path)

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    quantization_config=bnb_config,
    device_map="auto"
)

model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval()

def chat(question):
    prompt = f"""<|im_start|>system
You are a helpful Minecraft expert assistant.
<|im_end|>
<|im_start|>user
{question}
<|im_end|>
<|im_start|>assistant
"""
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id
        )
    response = tokenizer.decode(
        outputs[0][inputs['input_ids'].shape[1]:],
        skip_special_tokens=True
    )
    return response

print("=== 테스트 1 ===")
print(chat("How do I make a diamond sword?"))
print("\n=== 테스트 2 ===")
print(chat("What is the best way to find diamonds?"))
print("\n=== 테스트 3 ===")
print(chat("How do I tame a wolf?"))