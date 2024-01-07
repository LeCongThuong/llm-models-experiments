import json
import os
import sys
from time import sleep
from tqdm import tqdm
from dotenv import load_dotenv

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

from utils import TextByGemini, read_json_file

def generate_prompt(input_word, description, type):
    return f"""
    Vietnamese word: đón tiếp
    Desription: Đón và tiếp đãi một cách trang trọng.
    Type: verb
    Translate Vietnamese word to English. Let's step-by-step.
    English word: welcome
    ------------------------
    Vietnamese word: {input_word}
    Desription: {description}
    Type: {type}
    Translate Vietnamese word to English, push the english word result in <eng></eng>. Let's step-by-step.
    English word:
    """

def extract_result_from_output(output):
    result = output.split("<eng>")[1].split("</eng>")[0].strip()
    return result

def translate_word(model, input_word, description, type):
    prompt = generate_prompt(input_word, description, type)
    output = model.generate_text(prompt)
    result = extract_result_from_output(output)
    if result == "":
        result = input_word
    return result

def translate_word_dataset(model, input_file, output_file):
    dataset = read_json_file(input_file)['data']
    for item in tqdm(dataset):
        try:
            print(item['word'])
            item['en_word'] = translate_word(model, item['word'], item['description'], item['tl'])
            print(item['word'])
            print(item['description'])
            print(item['en_word'])
            print("------------------------")
        except Exception as e:
            item['en_word'] = "*unknown*"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=4)

def main():
    test_sentence_dataset_file = '../data/translation/qipedc_dataset_duplication_removal.json'
    output_result_file = '../data/translation/qipedc_dataset_duplication_removal_vi_en.json'
    model = TextByGemini()
    translate_word_dataset(model, test_sentence_dataset_file, output_result_file)

if __name__ == '__main__':
    main()