import os
import sys
import re
from dotenv import load_dotenv
from underthesea import pos_tag, chunk,  dependency_parse
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))


from utils import TextByGemini, read_json_file
import json
from tqdm import tqdm
from time import sleep

def genereate_prompt_to_check_types_of_sentence(complex_sentence):
    prompt = f""" 
    Sentence: "Nhà tôi có 4 người, bố, mẹ, em trai và tôi."
    The above sentence belongs to which type of sentence? declarative, negative, interrogative, imperative, exclamatory?
    Answer: declarative
    Sentence: "Tên của bạn là gì?"
    The above sentence belongs to which type of sentence? declarative, negative, interrogative, imperative, exclamatory?
    Answer: interrogative
    Sentence: "Hãy đến đây."
    The above sentence belongs to which type of sentence? declarative, negative, interrogative, imperative, exclamatory?
    Answer: imperative
    Sentence: "Tôi rất vui."
    The above sentence belongs to which type of sentence? declarative, negative, interrogative, imperative, exclamatory?
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
        Rewrite to simple complete sentences that each simple sentence contains only one independent clause. Do not repeat some words. Each simple sentence put in <sent></sent>. Let's think step by step.
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
        Answer: Nhà tôi/có/4 người
        sentence: "Tôi đi siêu thị băng xe đạp."
        Split the sentence into short phrases and non-overlapping words between these phrases: subject phrase, verb phrase, object phrase, adverb phrase, subject complement, object complement.
        Answer: Tôi/đi/siêu thị/bằng xe đạp
        ------------------------
        sentence: "{sentence}"
        Split the sentence into short phrases and non-overlapping words between these phrases: subject phrase, verb phrase, object phrase, adverb phrase, subject complement, object complement. Let's think step by step. No need to explain.
        Answer:
    """
    return prompt

def check_patterns_in_sentence(output):
    return output.split("/")

def split_into_simple_sentence(model, test_sentence):
    prompt = generate_prompt_to_check_how_many_clauses_of_sentence(test_sentence)
    num_clauses = model.generate_text(prompt)
    num_clauses = re.sub("[^0-9]", "", num_clauses)
    if int(num_clauses.strip()) > 1:
        prompt = generate_to_rewrite_sentence_with_one_clause(test_sentence)
        one_clause_sentences = extract_patterns_from_gemini_output(model.generate_text(prompt))
    else:
        one_clause_sentences = [test_sentence]
    return one_clause_sentences


def process_into_phrases(model, one_clause_sentences):
    print(one_clause_sentences)
    phrases_in_sentences = []
    for sent in one_clause_sentences:
        prompt = genereate_prompt_to_check_types_of_sentence(sent)
        type_sentence = model.generate_text(prompt)
        print(type_sentence)
        wh_word, negative_word = "", ""
        if type_sentence.strip() == "interrogative":
            wh_word = extract_the_wh_word_from_interrogative_sentence(sent, model)
            wh_word = wh_word.strip()
            print(wh_word)
        if type_sentence.strip() == "negative":
            negative_word = extract_the_negative_from_interrogative_sentence(sent, model)
            negative_word = negative_word.strip()
        
        phrase_list = []
        prompt = generate_prompt_to_split_sentence_into_phrases(sent)
        phrase_sentence = model.generate_text(prompt)
        phrase_list = check_patterns_in_sentence(phrase_sentence)
        print("Phrase list: ", phrase_list)
        wh_neg_phrase, normalize_phrase = [], []
        if wh_word == "" and negative_word == "":
            normalize_phrase = phrase_list
        else:
            if wh_word != "":
                search_phrase = wh_word
            elif negative_word != "":
                search_phrase = negative_word

            for phrase in phrase_list:
                if search_phrase not in phrase:
                    # move the whole word to the end of list 
                    normalize_phrase.append(phrase)
                else:
                    wh_neg_phrase.append(phrase)
            normalize_phrase.extend(wh_neg_phrase)
        phrases_in_sentences.append(normalize_phrase)
    return phrases_in_sentences


def extract_the_wh_word_from_interrogative_sentence(sentence, model):
    prompt = f"""
        sentence: "Tên của bạn là gì?"
        Extract the wh-word from the above sentence.
        Answer: gì
        sentence: "Ai đã gây ra việc này?"
        Extract the wh-word from the above sentence.
        Answer: Ai
        sentence: "Có bao nhiêu người trong nhà bạn?"
        Extract the wh-word from the above sentence.
        Answer: bao nhiêu
        ------------------------
        sentence: "{sentence}"
        Extract the wh-word from the above sentence. Let's think step by step.
        Answer:
    """
    wh_word = model.generate_text(prompt)
    return wh_word

def extract_the_negative_from_interrogative_sentence(sentence, model):
    prompt = f"""
        sentence: "Bạn không làm bài tập"
        Extract the negative from the above sentence.
        Answer: không
        sentence: "Bạn không thích màu đỏ"
        Extract the negative from the above sentence.
        Answer: không thích
        ------------------------
        sentence: "{sentence}"
        Extract the negative from the above sentence. Let's think step by step.
        Answer:
    """
    negative = model.generate_text(prompt)
    return negative


def get_phrases_types(phrases_in_sentences):
    phrases_type_in_sentences = []
    for sentence in phrases_in_sentences:
        phrase_type_in_each_word = []
        for phrase in sentence:
            phrase_type_in_each_word.append(chunk(phrase))
        phrases_type_in_sentences.append(phrase_type_in_each_word)
    return phrases_type_in_sentences


def remove_unnecessary_phrases(phrases_in_sentences, phrases_type_in_sentences):
    # remove type "E", "X" in phrases_type_in_sentences
    # remove type "E", "X" in phrases_in_sentences

   pass


def main():
    model = TextByGemini() 
    test_sentence = "Hai thập niên đầu thế kỷ XX ở Việt Nam , trong nhiều vấn đề nhân học , nổi lên vấn đề nữ học như là một vấn đề quan trọng , thu hút sự chú ý thảo luận và tranh luận của giới trí thức , nhất là nhìn từ quan điểm của chính người phụ nữ ."#"Hiện nay, trên một số kênh truyền hình của Việt Nam có phát bản tin bằng ngôn ngữ kí hiệu nhằm năng cao đời sống tinh thần cho cộng đồng người khiếm thính."
    simple_sentences = split_into_simple_sentence(model, test_sentence)
    phrases_in_sentences = process_into_phrases(model, simple_sentences)
    print(phrases_in_sentences)
    phrases_type_in_sentences = get_phrases_types(phrases_in_sentences)
    print(phrases_type_in_sentences)
    print("---------------------")
    # phrases_in_sentences, phrases_type_in_sentences = remove_unnecessary_phrases(phrases_in_sentences, phrases_type_in_sentences)
    # print(phrases_in_sentences)
    # print(phrases_type_in_sentences)
        
                 
        # prompt = generate_prompt_to_check_pos_and_neg_of_declarative_sentence(test_sentence)
        # pos_neg_sentence = model.generate_text(prompt)
        # print(pos_neg_sentence)
    



if __name__ == '__main__':
    main()
