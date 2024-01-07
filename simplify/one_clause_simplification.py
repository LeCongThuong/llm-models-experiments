import os
import sys
import re
from dotenv import load_dotenv
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))


from utils import TextByGemini, read_json_file
import json
from tqdm import tqdm
from time import sleep

def genereate_prompt_to_check_types_of_sentence(complex_sentence):
    prompt = f""" 
    Sentence: "Nhà tôi có 4 người, bố, mẹ, em trai và tôi."
    The above sentence belongs to which type of sentence? declarative, interrogative, imperative, exclamatory?
    Answer: declarative
    Sentence: "Tên của bạn là gì?"
    The above sentence belongs to which type of sentence? declarative, interrogative, imperative, exclamatory?
    Answer: interrogative
    Sentence: "Hãy đến đây."
    The above sentence belongs to which type of sentence? declarative, interrogative, imperative, exclamatory?
    Answer: imperative
    Sentence: "Tôi rất vui."
    The above sentence belongs to which type of sentence? declarative, interrogative, imperative, exclamatory?
    Answer: exclamatory
    ------------------------
    Sentence: "{complex_sentence}"
    The above sentence belongs to which type of sentence? declarative, interrogative, imperative, exclamatory? Let's think step by step.
    Answer: 
    """

    return prompt

def generate_prompt_to_check_pos_and_neg_of_declarative_sentence(sentence):
    prompt = f"""
        sentence: "Nhà tôi có 4 người"
        The above sentence is a declarative sentence. Is it a positive or negative sentence?
        Answer: positive
        sentence: "Tôi không biết đi bộ."
        The above sentence is a declarative sentence. Is it a positive or negative sentence?
        Answer: negative
        ------------------------
        sentence: "{sentence}"
        The above sentence is a declarative sentence. Is it a positive or negative sentence? Let's think step by step.
        Answer:
    """
    return prompt


def generate_prompt_to_check_how_many_clauses_of_sentence(sentence):
    prompt = f"""
        sentence: "Nhà tôi có 4 người"
        The above sentence has how many clauses?
        Answer: 1
        ------------------------
        sentence: "{sentence}"
        The above sentence has how many clauses? Let's think step by step.
        Answer:
    """
    return prompt


def generate_to_rewrite_sentence_with_one_clause(sentence):
    prompt = f"""
        sentence: "Từ căn gác nhỏ của mình , Hải có thể nghe tất cả các âm thanh náo nhiệt , ồn ã của thủ đô ."
        Rewrite to simple sentences that each simple sentence contains only one independent clause (require no dependent clause)
        Answer: Hải nghe thấy âm thanh của thủ đô từ căn gác nhỏ của mình. Âm thanh của thủ đô náo nhiệt và ồn ào.
        ------------------------
        sentence: "{sentence}"
        Rewrite to simple sentences that each simple sentence contains only one independent clause (require no dependent clause). Each simple sentence put in <sent></sent>. Let's think step by step.
        Answer:
    """
    return prompt

def extract_patterns_from_gemini_output(gemini_output, pattern="sent"):
    simple_sent_list = []
    # extract content between <sent></sent> using regex
   
    pattern = f"<{pattern}>(.*?)</{pattern}>"
    for match in re.finditer(pattern, gemini_output):
        simple_sent_list.append(match.group(1))
    return simple_sent_list

def generate_prompt_to_split_sentence_into_phrases(sentence):
    prompt = f"""
        sentence: "Nhà tôi có 4 người"
        Split the sentence into short phrases and non-overlapping words between these phrases: subject phrase, verb phrase, object phrase, adverb phrase, subject complement, object complement.
        Answer: <subject phrase> Nhà tôi </subject phrase>, <verb phrase> có </verb phrase>, <object phrase> 4 người </object phrase>
        sentence: "Tôi đi siêu thị băng xe đạp."
        Split the sentence into short phrases and non-overlapping words between these phrases: subject phrase, verb phrase, object phrase, adverb phrase, subject complement, object complement.
        Answer: <subject phrase> Tôi </subject phrase>, <verb phrase> đi </verb phrase>, <object phrase> siêu thị </object phrase>, <object complement> bằng xe đạp </object complement>
        ------------------------
        sentence: "{sentence}"
        Split the sentence into short phrases and non-overlapping words between these phrases: subject phrase, verb phrase, object phrase, adverb phrase, subject complement, object complement. Let's think step by step. No need to explain.
        Answer:
    """
    return prompt

def check_patterns_in_sentence(output):
    phrase_list = [phrase.strip() for phrase in output.split(",")]
    type_of_phrases = []
    for phrase in phrase_list:
        if "<subject phrase>" in phrase:
            type_of_phrases.append("S")
        if "verb phrase" in phrase:
            type_of_phrases.append("V")
        if "<object phrase>" in phrase:
            type_of_phrases.append("O")
        if "<adverb phrase>" in phrase:
            type_of_phrases.append("Adv")
        if "<subject complement>" in phrase:
            type_of_phrases.append("SC")
        if "<object complement>" in phrase:
            type_of_phrases.append("OC")
    # check whether S + V + 0 appear in the sentence continuous, if exist, change S + V +0 -> S + 0 + V.
    if "S" in type_of_phrases and "V" in type_of_phrases and "O" in type_of_phrases:
        if type_of_phrases.index("S") + 1 == type_of_phrases.index("V") and type_of_phrases.index("V") + 1 == type_of_phrases.index("O"):
            s_index, v_index, o_index = type_of_phrases.index("S"), type_of_phrases.index("V"), type_of_phrases.index("O")
            # check 0B appear in the sentence continuous, if exist, change S + V + 0 + 0B -> S + O + 0B + V.
            if "SC" in type_of_phrases and o_index + 1 == type_of_phrases.index("SC"):
                phrase_list[o_index], phrase_list[type_of_phrases.index("SC")]  = phrase_list[type_of_phrases.index("SC")],  phrase_list[o_index]
                type_of_phrases[o_index], type_of_phrases[type_of_phrases.index("SC")]  = type_of_phrases[type_of_phrases.index("SC")],  type_of_phrases[o_index]
            phrase_list[v_index], phrase_list[o_index]  = phrase_list[o_index],  phrase_list[v_index]
            type_of_phrases[v_index], type_of_phrases[o_index]  = type_of_phrases[o_index],  type_of_phrases[v_index]
    
    # remove <pattern>, <pattern/> in phrase_list
    phrase_list = [re.sub(r'<.*?>', '', phrase).strip() for phrase in phrase_list]
    return phrase_list, type_of_phrases

def main():
    model = TextByGemini() 
    test_sentence = "Nó quay nhanh đến mức không thể thấy rõ hình dạng , mà chỉ thấy là một khối cầu dẹp màu trắng và nó có thể quay trên đầu nhọn được ." #"Từ gác nhỏ, Hải nghe mọi thanh âm xô bồ của thành phố."
    prompt = genereate_prompt_to_check_types_of_sentence(test_sentence)
    type_sentence = model.generate_text(prompt)
    if type_sentence.strip() == "declarative":
            print("In here")
            prompt = generate_prompt_to_check_how_many_clauses_of_sentence(test_sentence)
            num_clauses = model.generate_text(prompt)
            print(num_clauses)
            # extract number from clause
            num_clauses = re.sub("[^0-9]", "", num_clauses)
            if int(num_clauses.strip()) > 1:
                prompt = generate_to_rewrite_sentence_with_one_clause(test_sentence)
                one_clause_sentence = extract_patterns_from_gemini_output(model.generate_text(prompt))
                print(one_clause_sentence)
                for sent in one_clause_sentence:
                    prompt = generate_prompt_to_split_sentence_into_phrases(sent)
                    phrase_sentence = model.generate_text(prompt)
                    print(phrase_sentence)
                    phrase, phrase_type = check_patterns_in_sentence(phrase_sentence)
                    print(phrase)
                    print(phrase_type)
                    print("-----")
                 
        # prompt = generate_prompt_to_check_pos_and_neg_of_declarative_sentence(test_sentence)
        # pos_neg_sentence = model.generate_text(prompt)
        # print(pos_neg_sentence)
    



if __name__ == '__main__':
    main()
