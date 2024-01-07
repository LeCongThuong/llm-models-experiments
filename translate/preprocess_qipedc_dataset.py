import json
import os
import sys
from time import sleep
from tqdm import tqdm
from dotenv import load_dotenv

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

from utils import TextByGemini, read_json_file


def preprocess(src_json_file, dest_json_dest):
    # read json file
    data = read_json_file(src_json_file)['data']
    word_list = []
    after_preprocess = []
    for item in tqdm(data):
        if item['word'] not in word_list:
            word_list.append(item['word'])
            after_preprocess.append(item)
    data = {"data": after_preprocess}
    with open(dest_json_dest, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    src_json_file = '../data/embedding/qipedc_dataset.json'
    dest_json_dest = '../data/translation/qipedc_dataset_duplication_removal.json'
    preprocess(src_json_file, dest_json_dest)