import json
from underthesea import word_tokenize, text_normalize, sent_tokenize, dependency_parse

import os
import sys
import re
from dotenv import load_dotenv
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

from utils import TextByGemini, read_json_file

def read_json_file(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def get_words_in_qipedc_dataset(qipedc_path, dest_path):
    data = read_json_file(qipedc_path)
    words = []
    for item in data['data']:
        words.append(text_normalize(item['word']))
    words = list(set(words))
    with open(dest_path, 'w') as f:
        for word in words:
            f.write(word + '\n')


def segment_words_in_sent(sents, fixed_words, format_text=True):
    print(sents)
    sents =text_normalize(sents)
    sent_list = sent_tokenize(sents)
    word_list = []
    if format_text:
        for sent in sent_list:
            word_list.append(word_tokenize(sent, format="text", fixed_words=fixed_words))
        word_list = ' '.join(word_list)
    else:
        for sent in sent_list:
            word_list.append(word_tokenize(sent, format, fixed_words=fixed_words))
    return word_list

def split_sent_by_gemini(model, sents):
    prompt = f"""
    sentence: Nhà tôi có một con chó và một con mèo.
    You are a Vietnamese linguistic expert. You are asked to divide the sentence into meaning short unigrams or bigrams (only one or two words).
    Let’s think step by step. Seperated by /.
    output: nhà tôi/có/một/con chó/và/một/con mèo.
    sent: Việt Nam gác lại quá khứ, vượt qua khác biệt để biến thù thành bạn.
    output: Việt Nam/gác lại/quá khứ/vượt qua/khác biệt/để/biến/thù/thành/bạn.
    ------------------------------------------
    sentence: "{sents}" 
    You are a Vietnamese linguistic expert. You are asked to divide the sentence into meaning short unigrams or bigrams (only one or two words).
    Let’s think step by step. Seperated by /.
    output: 
    """ 
    sentence_type = model.generate_text(prompt)
    print(sentence_type)
    return sentence_type

def extract_patterns_from_gemini_output(gemini_output):
    # extract all sentences from gemini output in <phrase></phrase> tag using regex
    pattern = r'<phrase>(.*?)</phrase>'
    sentences = re.findall(pattern, gemini_output)
    return sentences

def read_txt_file(txt_path):
    with open(txt_path, 'r') as f:
        data = f.read().split('\n')
    if data[-1] == '':
        data = data[:-1]
    return data


if __name__ == '__main__':
    # get fixed words in qipedc dataset
    # qipedc_path = '/home/love_you/Documents/works/vsl_2/data/translation/qipedc_dataset_duplication_removal.json'
    # dest_path = '/home/love_you/Documents/works/vsl_2/data/translation/fixed_words.txt'
    # get_words_in_qipedc_dataset(qipedc_path, dest_path)

    # segment words in sentence
    # sents = 'Anh ta là sinh viên đại học. Anh ta rất mạnh mẽ, tuy nhiên, hôm nay anh ta nói ngập ngừng.'
    # fixed_words = read_txt_file('/home/love_you/Documents/works/vsl_2/data/translation/fixed_words.txt')
    # test_dataset = read_json_file('/home/love_you/Documents/works/vsl_2/data/simplification/step_by_step_2_simple_sentences.json')
    # res = {}
    # for key, sents_list in test_dataset.items():
    #     for sent in sents_list:
    #         sent = list(sent.values())[0]
    #         words = segment_words_in_sent(sent, fixed_words)
    #         res[sent] = words
    # with open('/home/love_you/Documents/works/vsl_2/data/simplification/word_segment_result.json', 'w') as f:
    #     json.dump(res, f, indent=4, ensure_ascii=False)

    # split sentence using gemini
    model = TextByGemini()
    # sent = "Học vấn không chỉ đến từ việc đọc sách. Tuy nhiên, đọc sách vẫn là một cách quan trọng để có được học vấn."
    # split_sent = split_sent_by_gemini(model, sent)
    # split_sent = extract_patterns_from_gemini_output(split_sent)
    # split_sent = split_sent.split('/')
    # print(split_sent)
    test_dataset = read_json_file('/home/love_you/Documents/works/vsl_2/data/simplification/step_by_step_2_simple_sentences.json')
    res = {}
    for key, sents_list in test_dataset.items():
        for sent in sents_list:
            sent = list(sent.values())[0]
            words = split_sent_by_gemini(model, sent)
            split_sent = words.split('/')
            res[sent] = words
    with open('/home/love_you/Documents/works/vsl_2/data/simplification/gemini_word_segment_result.json', 'w') as f:
        json.dump(res, f, indent=4, ensure_ascii=False)