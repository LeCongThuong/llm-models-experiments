import json
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def read_json_file(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)
    
def lower_case_list(list_data):
    return [item.lower() for item in list_data]
    
def extract_word_from_dict(dict_data):
    word_list = []
    for word_info in dict_data['data']:
        word_list.append(word_info['word'])
    return list(set(lower_case_list(word_list)))

def write_to_list(file_path, list_data):
    with open(file_path, 'w') as f:
        for item in list_data:
            f.write("%s\n" % item)

def main():
    file_path = '../data/embedding/qipedc_dataset.json'
    dict_data = read_json_file(file_path)
    word_list = extract_word_from_dict(dict_data)
    write_to_list('../data/embedding/word_list.txt', word_list)

if __name__ == '__main__':
    main()
