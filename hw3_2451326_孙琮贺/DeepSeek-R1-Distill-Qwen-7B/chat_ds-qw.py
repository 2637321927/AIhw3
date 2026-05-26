
from transformers import TextStreamer, AutoTokenizer, AutoModelForCausalLM
import torch
model_name = "/mnt/data/DeepSeek-R1-Distill-Qwen-7B"
print("正在加载模型，请稍等...")

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True
)

tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype="auto"
).eval().cuda()

streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

print("模型加载完成！输入 'exit' 退出\n")

while True:
    user_input = input("你：")
    if user_input.strip().lower() == "exit":
        break
    messages = [{"role": "user", "content": user_input}]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    print("模型：", end="", flush=True)
    outputs = model.generate(
        **inputs,
        streamer=streamer,
        max_new_tokens=1024, 
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id 
    )
    print("\n")