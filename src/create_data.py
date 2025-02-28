import openai
from dotenv import load_dotenv
import os
from mapping import mappingE2V
import json
load_dotenv(dotenv_path= "../.env")

API_KEY = os.getenv("OPEN_API_KEY")

openai.api_key = API_KEY

def generate_vqa(messages, model = "gpt-4o mini", max_tokens = 250):
    response = openai.ChatCompletion.create(
        model = model,
        messages = messages,
        max_tokens=max_tokens,
        temperature = 0.3 
    )

    return response["choices"][0]["message"]["content"]


def create_messages(sysm_path, data_path):
    with open(sysm_path, "r") as f:
        sys_msg = f.read()
    
    part = mappingE2V(data_path.split("/")[-2])
    prompt = f"Bức ảnh này chụp {part} của bệnh nhân"
    with open(data_path, "r") as f:
        data = f.read()
    
    data = data.split(".")
    for line in data:
        prompt += f"\n{str(line).strip()}"
    
    message = [
        {"role": "system", "content" : sys_msg},
        {"role": "user", "content": prompt}
    ]
    
    return message

def save_results(mes, save_path):
    result = {
        "messages": mes
    }
    
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=4)
    

if __name__ == "__main__":
    sysm_path = "../prompts/conversations/system_message.txt"
    data_path = "./whole_body/sample.txt"
    message = create_messages(sysm_path, data_path)
    response = generate_vqa(message)
    save_results(response, "sample_check.json")
    