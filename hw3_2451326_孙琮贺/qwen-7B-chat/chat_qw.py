from transformers import TextStreamer, AutoTokenizer, AutoModelForCausalLM
import torch
model_name = "/mnt/data/Qwen-7B-Chat"
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

    prompt = f"用户：{user_input}\n回答："
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = inputs.to("cuda")

    print("模型：", end="", flush=True)
    outputs = model.generate(
        **inputs,
        streamer=streamer,
        max_new_tokens=512
    )
    print("\n")