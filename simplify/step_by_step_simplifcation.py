import os
import sys
from dotenv import load_dotenv
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))


from utils import TextByGemini, read_json_file
import json
from tqdm import tqdm
from time import sleep
import re



def extract_patterns_from_gemini_output(gemini_output):
    # extract all sentences from gemini output in <sent></sent> tag using regex
    pattern = r'<sent>(.*?)</sent>'
    sentences = re.findall(pattern, gemini_output)
    return sentences
def lexical_sentence(model, sentence):
    prompt = f"""
    simplify sentence in the lexical view (replacing difficult phrases or words with their simpler synonyms).
    Let’s think step by step as vietnamese linguistic expert and answer in Vietnamese. Put the vietnamse output in the tag <sent></sent>. If the sentence is already simple, just put the output in the tag <sent>.
    sentence: "{sentence}"
    : 
    """ 
    sentence_type = model.generate_text(prompt)
    return sentence_type


def split_sentence(model, sentence):
    prompt = f"""
    simplify sentence by split the source sentence into several shorter ones.
    Let’s think step by step as vietnamese linguistic expert and answer in Vietnamese. Put the vietnamse output in the tag <sent></sent>. If the sentence is already simple, just put the output in the tag <sent>.
    sentence: "{sentence}" 
    """ 
    sentence_type = model.generate_text(prompt)
    return sentence_type

def compress_sentence(model, sentence):
    prompt = f"""
    simplify sentence by dropping (deleting/drop/ removes unimportant parts of input sentence such as unnecessary words, phrases, clauses, etc.)
    Let’s think step by step as vietnamese linguistic expert and answer in Vietnamese. Put the vietnamse output in the tag <sent></sent>. If the sentence is already simple, just put the output in the tag <sent>.
    sentence: "{sentence}" 
    """ 
    sentence_type = model.generate_text(prompt)
    return sentence_type

def paraphrase_sentence(model, sentence):
    prompt = f"""
    simplify sentence by sentence paraphrasing (word reordering or syntactic transformations).
    Let’s think step by step as vietnamese linguistic expert and answer in Vietnamese. Put the vietnamse output in the tag <sent></sent>. If the sentence is already simple, just put the output in the tag <sent>.
    sentence: "{sentence}" 
    """ 
    sentence_type = model.generate_text(prompt)
    return sentence_type

def simplify_step_by_step(sent, model):
    model = TextByGemini()
    print(sent)
    # sent = "Thời gian gần đây khái niệm diễn ngôn đã xuất hiện rất nhiều trong các bài nghiên cứu đủ loại, nhiều đến mức không sao có thể định nghĩa thông suốt hết."
    sim_sent = lexical_sentence(model, sent)
    sim_sent = extract_patterns_from_gemini_output(sim_sent)[0]
    print("Lexical sent: ", sim_sent)
    split_sent = split_sentence(model, sim_sent)
    print(split_sent)
    split_sent = extract_patterns_from_gemini_output(split_sent)
    split_sent = " ".join(split_sent)
    print("Split sent: ", split_sent)
    paraphrasing_sent = paraphrase_sentence(model, split_sent)
    print(paraphrasing_sent)
    paraphrasing_sent = extract_patterns_from_gemini_output(paraphrasing_sent)
    paraphrasing_sent = " ".join(paraphrasing_sent)
    print("Paraphasal sent: ", paraphrasing_sent)
    compress_sent = compress_sentence(model, paraphrasing_sent)
    print(compress_sent)
    compress_sent = extract_patterns_from_gemini_output(compress_sent)
    compress_sent = " ".join(compress_sent)
    print("Compress sent: ", compress_sent)
    return compress_sent

if __name__ == '__main__':
    model = TextByGemini()
    complex_sentence_file_path = '../data/simplification/sentence_simplification.json'
    complex_sentences = read_json_file(complex_sentence_file_path)
    simple_sentences_dict = {}
    for difficulty_level, complex_sentence_list in complex_sentences.items():
        sim_complex_sentences = []
        for complex_sentence in tqdm(complex_sentence_list):
            try:
                sim_complex_sentence = simplify_step_by_step(complex_sentence, model)
                sim_complex_sentences.append({complex_sentence: sim_complex_sentence})
            except Exception as e:
                print(e)
                print(f"Error: {complex_sentence}")
                continue
        simple_sentences_dict[difficulty_level] = sim_complex_sentences
    with open('../data/simplification/step_by_step_simple_sentences.json', 'w', encoding="utf-8") as f:
        json.dump(simple_sentences_dict, f, indent=4, ensure_ascii=False)
        

