import json
import os
import sys
from time import sleep
from tqdm import tqdm
from dotenv import load_dotenv

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

from utils import gen_few_shot_prompt, TextByGemini, read_line_split_by_tab_file


def main():
    prompt_file = '../data/translation/prompt.txt'
    test_sentence_dataset_file = '../data/translation/test_translation_dataset.txt'
    output_result_file = '../data/translation/output_translation_dataset.json'

    few_shot_examples = open(prompt_file, 'r').read()
    test_sentence_list = read_line_split_by_tab_file(test_sentence_dataset_file)
    model = TextByGemini()

    res_list = []
    for test_sentence in tqdm(test_sentence_list):
        input_sent, gt_sent = test_sentence
        prompt = gen_few_shot_prompt(input_sent, few_shot_examples)
        output_sent = model.generate_text(prompt)
        res_list.append({
            "input": input_sent,
            "output": output_sent,
            "gt": gt_sent
        })
        sleep(2)

    # write list of dict to file
    with open(output_result_file, 'w', encoding="utf-8") as f:
        json.dump(res_list, f, indent=4, ensure_ascii=False)

if __name__ == '__main__':
    main()