from utils import TextByGemini, read_json_file
import json
from tqdm import tqdm
from time import sleep



def main():
    model = TextByGemini()
    complex_sentence_file_path = 'data/sentence_simplification.json'
    complex_sentences = read_json_file(complex_sentence_file_path)
    simple_sentences_dict = {}
    for difficulty_level, complex_sentence_list in complex_sentences.items():
        sim_complex_sentences = []
        for complex_sentence in tqdm(complex_sentence_list):
            try:
                prompt = f"""
Let’s think step by step. I want you to replace my complex sentence with simple sentence(s). Keep the meaning same, but make them simpler. 
Complex: {complex_sentence}
Simple:
"""
                simple_sentence = model.generate_text(prompt)
                print(complex_sentence, "---", simple_sentence)
                sleep(1)
                sim_complex_sentences.append({complex_sentence: simple_sentence})
            except Exception as e:
                print(e)
                print(f"Error: {complex_sentence}")
                continue
        simple_sentences_dict[difficulty_level] = sim_complex_sentences
    with open('data/simple_sentences.json', 'w', encoding="utf-8") as f:
        json.dump(simple_sentences_dict, f, indent=4, ensure_ascii=False)

if __name__ == '__main__':
    main()
        

