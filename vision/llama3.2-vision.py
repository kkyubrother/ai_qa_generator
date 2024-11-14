from ollama import generate


with open('img.png', 'rb') as f:
    raw = f.read()

for response in generate('llama3.2-vision', '이미지의 사람을 설명하라', images=[raw], stream=True):
    print(response['response'], end='', flush=True)
