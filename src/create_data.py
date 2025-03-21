import openai
from dotenv import load_dotenv
import os
from mapping import mappingE2V
import re
import json
load_dotenv(dotenv_path= "../.env")

API_KEY = os.getenv("OPEN_API_KEY")

openai.api_key = API_KEY

def extract_qa_pairs(message):
    # Regex tìm các đoạn chứa Câu hỏi và Trả lời
    pattern = r"Câu hỏi:\n(.*?)\n===\nTrả lời:\n(.*?)(?=\n===\nCâu hỏi:|\Z)"
    matches = re.findall(pattern, message, re.DOTALL)
    
    # Xử lý danh sách kết quả để loại bỏ khoảng trắng thừa
    qa_pairs = [(q.strip(), a.strip()) for q, a in matches]
    
    return qa_pairs

def generate_vqa(messages, model = "gpt-4o-mini", max_tokens = 2500):
    response = openai.ChatCompletion.create(
        model = model,
        messages = messages,
        max_tokens=max_tokens,
        temperature = 0.6
    )

    return response["choices"][0]["message"]["content"]


def create_messages(sysm_path, data_path):
    with open(os.path.join(sysm_path, "system_message.txt"), "r") as f:
        sys_msg = f.read()
    
    part = mappingE2V(data_path.split("/")[-2])
    prompt = f"Bức ảnh này chụp {part} của bệnh nhân"
    with open(data_path, "r") as f:
        data = f.read()
    
    data = data.split(".")
    for line in data:
        prompt += f"\n{str(line).strip()}"
    
    example = "\nMột vài ví dụ:"
    full_examples = os.listdir(sysm_path)
    noexamples = (len(full_examples) - 1 )// 2
    for idx in range(1, 1 + noexamples):
        caption_path = f"../prompts/conversations/{idx}_caps.txt"
        conv_path = f"../prompts/conversations/{idx}_conv.txt"
        with open(caption_path, "r") as f:
            caption = f.read()
            example += f"\n{caption}"
        with open(conv_path, "r") as f:
            conv = f.read()
            example += f"\n{conv}"
    
    sys_msg += example
    
    message = [
        {"role": "system", "content" : sys_msg},
        {"role": "user", "content": prompt}
    ]
    
    return message

def addingResult(json_file, infor):
    # if not os.path.exists(json_file):
    #     with open(json_file, "w") as file:
    #         json.dump("", file)  # Initialize with an empty list
    # Check if file exists; if not, create an empty JSON array
    report_id = infor[0]
    
    qa = infor[1]
    save = {"report_id": report_id, "qa": qa}
    # with open(json_file, "r") as file:  
    with open(json_file, "a", encoding="utf-8") as f:
        
        f.write(json.dumps(save, ensure_ascii=False) + "\n")


def full_data(folder, type):
    if type == "conv":
        sysm_path = "../prompts/conversations/"
    for root, _, files in os.walk(folder):
        if "whole_body" in root:
            continue
        for file in files:
            file_path = os.path.join(root, file)
            
            report_id = os.path.join(root, file)
            
            # print(root)
            try:
                msg = create_messages(sysm_path, file_path)
                response = generate_vqa(msg)
            except Exception as e:
                print("Error ", e)
                with open("error.txt", "a") as f:
                    f.write(file_path + "\n" + str(e) + "\n")
                continue
            addingResult("../data/conversations.jsonl", (report_id, response))
            # qa_pairs = extract_qa_pairs(response)
            # for i, (q, a) in enumerate(qa_pairs, 1):
                # addingResult("../data/conversations.json", (report_id, q, a))
            
            
        
        
if __name__ == "__main__":
    FOLDER_PATH = "/home/user01/aiotlab/thaind/DAC001_CTAC3.75mm_H_1001_PETWB3DAC001/"
    full_data(FOLDER_PATH, "conv")
    