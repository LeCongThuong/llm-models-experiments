import json
import os
import sys
from time import sleep
from tqdm import tqdm
from dotenv import load_dotenv

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

from utils import TextByGemini, read_json_file

def generate_prompt(input_word, context_sent):
    return f"""
    Vietnamese word: đồng phục
    Context sentence: Nhà trường may đồng phục cho học sinh
    Translate Vietnamese word to English with the context sentence and identify the type of the word.
    English word: uniform
    Type: noun
    ------------------------
    Vietnamese word: {input_word}
    Context sentence: {context_sent}
    Translate Vietnamese word to English, push the english word result in <eng></eng> and type in <type></type>. Let's think step-by-step.
    English word:
    Type:
    """

def extract_word_from_output(output, pattern):
    result = output.split(f"<{pattern}>")[1].split(f"</{pattern}>")[0].strip()
    return result


def translate_word(model, input_word, context_sent):
    prompt = generate_prompt(input_word, context_sent)
    output = model.generate_text(prompt)
    en_word = extract_word_from_output(output, "eng")
    type = extract_word_from_output(output, "type")
    if en_word == "":
        en_word = input_word
    if type == "":
        type = "*unknown*"
    return en_word, type

def translate_word_dataset(model, input_file, output_file):
    dataset = read_json_file(input_file)
    for vi_eng, info in tqdm(dataset.items()):
        try:
            print(vi_eng)
            print(info['vi'])
            info['en_word'], info['type'] = translate_word(model, info['vi'], info["sent"])
            print(info['en_word'])
            print(info['type'])
    
            print("------------------------")
        except Exception as e:
            print(e)
            info['en_word'] = "*unknown*"
            info['type'] = "*unknown*"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=4)

def main():
    test_sentence_dataset_file = '../data/embedding/test_dataset.json'
    output_result_file = '../data/embedding/test_dataset_vi_en.json'
    model = TextByGemini()
    translate_word_dataset(model, test_sentence_dataset_file, output_result_file)

if __name__ == '__main__':
    main()