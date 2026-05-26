import warnings
warnings.filterwarnings("ignore")

import torch
torch.backends.cuda.enable_mem_efficient_sdp(False) 

from transformers import TextStreamer, AutoTokenizer, AutoModelForCausalLM

model_name = "/mnt/data/Baichuan2-7B-Chat"

print("正在加载模型，请稍等...")

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True
)
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

    prompt = f"<user>{user_input}<assistant>"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    print("模型：", end="", flush=True)
    model.generate(
        **inputs,
        streamer=streamer,
        max_new_tokens=512,
        pad_token_id=tokenizer.eos_token_id
    )
    print("\n")