from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

base_model_id = "Qwen/Qwen2.5-7B-Instruct"

adapter_path = './model'

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
print(chat("I want diamonds. And I just downloaded this game. Can you explain in as much detail as possible how to get diamonds? and Answer step by step"))


"""
테스트 1 (20260519)
print(chat("I want diamonds. And I just downloaded this game. Can you explain in as much detail as possible how to get diamonds? and Answer step by step"))

First, make sure you have the necessary materials to craft a diamond pickaxe. You'll need 3 diamonds and 2 sticks. To craft the pickaxe, open your crafting table and arrange the diamonds and sticks in the following pattern: 1 diamond in the top-left square, 1 diamond in the top-right square, 1 diamond in the bottom-left square, and 2 sticks in the middle squares. Once you've crafted the pickaxe, you can use it to mine diamonds from the ground. Make sure to dig deep enough to find diamonds, as they can be found between 5-12 blocks below the surface. You can also try mining in caves or ravines for diamonds. Be cautious, though, as some caves may contain lava or monsters that can harm you. Once you've collected enough diamonds, you can use them to craft various items such as diamond tools, armor, or even a diamond sword.

결과 -> 성능 부족
"""