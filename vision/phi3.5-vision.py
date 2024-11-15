from PIL import Image
import requests
from transformers import AutoModelForCausalLM
from transformers import AutoProcessor

model_id = "microsoft/Phi-3.5-vision-instruct"

# Note: set _attn_implementation='eager' if you don't have flash_attn installed
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="cuda",
    trust_remote_code=True,
    torch_dtype="auto",
    _attn_implementation='eager'
)

# for best performance, use num_crops=4 for multi-frame, num_crops=16 for single-frame.
processor = AutoProcessor.from_pretrained(model_id,
                                          trust_remote_code=True,
                                          num_crops=4
                                          )

images = []
placeholder = ""

# Note: if OOM, you might consider reduce number of frames in this example.
# for i in range(1,20):
#     url = f"https://image.slidesharecdn.com/azureintroduction-191206101932/75/Introduction-to-Microsoft-Azure-Cloud-{i}-2048.jpg"
#     images.append(Image.open(requests.get(url, stream=True).raw))
#     placeholder += f"<|image_{i}|>\n"

image_url = "http://www.conslove.co.kr/news/photo/202110/70961_209640_2032.jpg"
images.append(Image.open(requests.get(image_url, stream=True).raw))
placeholder = "<|image_1|>\n"

messages = [
    {"role": "user", "content": placeholder+"이 이미지에 대해 설명 해 주세요."},
]

prompt = processor.tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
print(prompt)
inputs = processor(prompt, images, return_tensors="pt").to("cuda:0")

generation_args = {
    "max_new_tokens": 1000,
    "temperature": 0.2,
    "do_sample": False,
}

generate_ids = model.generate(**inputs,
                              eos_token_id=processor.tokenizer.eos_token_id,
                              **generation_args
                              )

# remove input tokens
generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
response = processor.batch_decode(generate_ids,
                                  skip_special_tokens=True,
                                  clean_up_tokenization_spaces=False)[0]

print(response)
