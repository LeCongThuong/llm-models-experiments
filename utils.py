import os
import google.generativeai as genai
from dotenv import load_dotenv
load_dotenv('.env')
import json

class TextByGemini:
    def __init__(self) -> None:
        genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
        self.model = genai.GenerativeModel('gemini-pro')

    def generate_text(self, prompt):
        return self.model.generate_content(prompt).text

def read_json_file(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)
    
def convert2prompt(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    prompts = []
    for line in lines:
        s1, s2 = line.split('\t')
        s2 = s2.strip()
        prompts.append(f"""
            Sentence: "{s1}"
            Question: Convert the above sentence into a Vietnamese sign language sentence.
            Answer: "{s2}"
            """)
    #concatenate all prompts
    prompts = ''.join(prompts)

    return prompts

def gen_few_shot_prompt(sentence, prompt):
    return f"""
            {prompt}
            ----------------------------------
            Sentence: "{sentence}"
            Question: Based on the above examples, convert the sentence into a Vietnamese sign language sentence. Let’s think step by step.
            Answer: 
            """