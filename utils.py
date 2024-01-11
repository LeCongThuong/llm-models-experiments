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
        return self.model.generate_content(prompt,    
                                           generation_config=genai.types.GenerationConfig(
                                            # Only one candidate for now.
                                            candidate_count=1,
                                            temperature=.7)).text

def read_json_file(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)
    
    
def split_line_by_tab(line):
    s1, s2 = line.split('\t')
    s2 = s2.strip()
    s1 = s1.strip()
    return s1, s2

def convert2prompt(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    prompts = []
    for line in lines:
        s1, s2 = split_line_by_tab(line)
        prompts.append(f"""
            Sentence: "{s1}"
            Question: Convert the above sentence into vietnamese sign language sentence.
            Answer: "{s2}"
            """)
    #concatenate all prompts
    prompts = ''.join(prompts)
    return prompts

def gen_few_shot_prompt(sentence, prompt):
    return f"""
            Following is some rules need to remember:
            1. If the structure of sentence is S-V-O, reorder words into the S-O-V.
            3. Remove the following types of words from the sentence: Modal particles, conjunctions, interjections, auxiliary words, empty words, and units of measure words.
            4. In the case of questions with question words such as: ai, gì, mấy, thế nào, bao nhiêu, đâu, nào, tại sao, etc...  the question word symbol always stands at the end of the sentence. 
            5. To negate an action, the negative symbol can only be expressed last.For instance, Không ăn cơm -> Cơm ăn không
            6. In the case of noun phrases/noun phrases in sign language, the number symbol corresponding to the numeral in natural spoken language must be placed after the symbol for the object corresponding to the noun. For instance, Một con vịt -> Vịt một, Hai quả táo xanh -> Táo xanh hai
            7. The symbols corresponding to unit words (types of words) such as: con, cái, chiếc… when used with nouns in common language are often omitted.
            ----------------------------------
            Follow is some examples:
            {prompt}
            ----------------------------------
            Sentence: "{sentence}"
            Question: Based on the above examples and rules, translate the above sentence into vietnamese sign language sentence. Let’s think step by step. No need to explain the rules. Just translate the sentence.
            Answer is: 
            """

def read_line_split_by_tab_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    return [split_line_by_tab(line) for line in lines]