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
        The sentence has how many independent clauses?
        Answer: 1
        ------------------------
        sentence: "{sentence}"
        The sentence has how many independent clauses? Let's think step by step.
        Answer:
    """
    return prompt


def generate_to_rewrite_sentence_with_one_clause(model, sentence):
    prompt = f"""
        sentence: "Toán học là môn học khó nhất bởi vì nó có nhiều công thức."
        The sentence is the complex sentence, therefore, rewrite to short sentences that each sentence contains only one independent clause (no dependent clause) and each sentence puts in <sent></sent>.
        Answer: <sent>Toán học là môn học khó nhất.</sent> <sent>Nó có nhiều công thức.</sent>
        ------------------------
        sentence: f"{sentence}"
        The sentence is the complex sentence, therefore, rewrite to short sentences that each simple sentence contains only one independent clause (no dependent clause)and each sentence puts in <sent></sent>. Let's think step by step.
        Answer:
    """
    sentences = model.generate_text(prompt)
    simple_sentence_list = extract_patterns_from_gemini_output(sentences, pattern="sent")
    return simple_sentence_list

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
        Split the sentence into short phrases (no more 2 words) and non-overlapping words between these phrases: subject phrase, verb phrase, object phrase, adverb phrase, subject complement, object complement.
        Answer: Nhà tôi/có/4 người
        sentence: "Tôi đi siêu thị băng xe đạp."
        Split the sentence into short phrases (no more 32words) and non-overlapping words between these phrases: subject phrase, verb phrase, object phrase, adverb phrase, subject complement, object complement. 
        Answer: Tôi/đi/siêu thị/bằng/xe đạp
        ------------------------
        sentence: "{sentence}"
        Split the sentence into short phrases (no more 2 words) and non-overlapping words between these phrases: subject phrase, verb phrase, object phrase, adverb phrase, subject complement, object complement. Let's think step by step. No need to explain.
        Answer:
    """
    return prompt


def check_patterns_in_sentence(output):
    return output.split("/")


def remove_unnecessary_words_in_sent(model, sent):
    prompt = f"""
        sentence: "Anh ta có thể làm được việc này nhưng anh ta đã không làm"   
        Remove: Modal particles, Conjunctions, Interjections, Auxiliary words, Empty words
        Answer: Anh ta làm được việc này anh ta không làm.
        ------------------------
        sentence: "{sent}"
        Remove: Modal particles, Conjunctions, Interjections, Auxiliary words, Empty words. Let's think step by step.
        Answer:
        """
    output = model.generate_text(prompt)
    return output

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


def get_phrases_types(phrases_in_sentences, sentences):
    phrases_type_in_sentences = []
    for sentence in phrases_in_sentences:
        phrase_type_in_each_word = []
        for phrase in sentence:
            phrase_type_in_each_word.append(chunk(phrase))
        phrases_type_in_sentences.append(phrase_type_in_each_word)
    return phrases_type_in_sentences


def invert_number_noun_to_noun_number(phrases_in_sentences, phrases_type_in_sentences):
    # invert number-noun to noun-number
    pass 


def remove_unnecessary_phrases(phrases_in_sentences, phrases_type_in_sentences):
    # remove type "E", "X" in phrases_type_in_sentences
    # remove type "E", "X" in phrases_in_sentences

   pass

def compound_or_simple(model, sentence):
    prompt = f"""
        Simple sentence is the sentence that contains only one independent clause. Compound sentence is the sentence that contains more than one independent clause. For instance:
        sentence: "Tôi đi siêu thị bằng xe đạp."
        Classify the above sentence into one of the following types: simple, compound
        Answer: simple
        sentence: "Tôi đi siêu thị bằng xe đạp và mua một cái bánh."
        Classify the above sentence into one of the following types: simple, compound
        Answer: compound
        ------------------------
        sentence: "{sentence}"
        Classify the above sentence into one of the following types: simple, compound. Let's think step by step.
        Answer:
    """ 
    sentence_type = model.generate_text(prompt)
    return sentence_type

def complex_or_simple(model, sentence):
    prompt = f"""
        sentence: "Tôi đi siêu thị bằng xe đạp."
        Which types of the above sentence, choose one of the following types: simple, complex
        Answer: simple
        sentence: "Tôi đi siêu thị bằng xe đạp vì tôi muốn mua một cái bánh."
        Which types of the above sentence, choose one of the following types: simple, complex
        Answer: complex
        ------------------------
        sentence: "{sentence}"
        Which types of the above sentence, choose one of the following types: simple, complex. Let's think step by step.
        Answer:
    """ 
    sentence_type = model.generate_text(prompt)
    return sentence_type


def split_compound_into_simple_sentences(model, sentence):
    prompt = f"""
        Simple sentence is a sentence that includes only one independent clause. Compound sentence is a sentence that includes more than one independent clause. For instance:
        sentence: "Tôi đi siêu thị bằng xe đạp và mua một cái bánh."
        The sentence is a compound sentence because there are 2 independent clauses. Rewrite the sentence into simple sentences and put in tag <sent></sent>.
        Answer: <sent>Tôi đi siêu thị bằng xe đạp </sent>. <sent>Tôi mua một cái bánh</sent>.
        ------------------------
        sentence: "{sentence}"
        The sentence is a compound sentence becasue there are more than one independent clause. Rewrite the sentence into simple sentences put in tag <sent></sent>. Let's think step by step.
        Answer:
    """ 
    simple_sentences = model.generate_text(prompt)
    simple_sentences = extract_patterns_from_gemini_output(simple_sentences, pattern="sent")
    return simple_sentences


def split_into_simple_sentence(model, sentence):
    prompt = f"""
                Let’s think step by step. I want you to replace my complex sentence with simple sentence(s) and put in tag <sent></sent>. Keep the meaning same, but make them simpler.
                Complex: {sentence}
                Simple:
                """
    simple_sentence = model.generate_text(prompt)
    simple_sentence = extract_patterns_from_gemini_output(simple_sentence, pattern="sent")
    return simple_sentence


def _change_the_order_of_phrase(text):
    chunk_res = chunk(text)
    start = []
    end = [chunk_res[-1]]
    num_c = len(chunk_res)
    for i in range(num_c-1):
        if chunk_res[i][1] == 'M' and (chunk_res[i+1][1] == 'N' or chunk_res[i+1][1] == 'Np' or chunk_res[i+1][1] == 'Nc'):
            end.append(chunk_res[i])
        else:
            start.append(chunk_res[i])
    start.extend(end)
    text = " ".join([token[0] for token in start])
    return text

def change_the_order_of_phrases(phrases_in_sentences):
    new_phrases_in_sentences = []
    for phrases in phrases_in_sentences:
        new_phrases = []
        for phrase in phrases:
            new_phrases.append(_change_the_order_of_phrase(phrase))
        new_phrases_in_sentences.append(new_phrases)
    return new_phrases_in_sentences

def identify_spelling_in_sentence(phrase):
    # Noun proper phrase, number, name, will be spelling. Use POS tag to identify
    phrase_list = []
    flag = False
    punc_list = []
    for token in pos_tag(phrase):
        if token[1] == "Np":
            phrase_list.append((token[0].lower(), True))
            flag = True
        # if containg punctuation, it is not spelling
        if token[1] == 'CH':
            punc_list.append(token[0])
        else:
            if token[0].isdigit():
                phrase_list.append((token[0].lower(), True))
                flag = True
            else:
                phrase_list.append((token[0].lower(), False))
    if flag == False:
        for token in punc_list:
            phrase = phrase.replace(token, "")
        phrase_list = [(phrase.lower(), False)]
    return phrase_list

def identify_spelling_in_phrases(phrases_in_sentences):
    new_phrases_in_sentences = []
    for phrases in phrases_in_sentences:
        new_phrases = []
        for phrase in phrases:
            new_phrases.append(identify_spelling_in_sentence(phrase))
        new_phrases_in_sentences.append(new_phrases)
    return new_phrases_in_sentences


def main():
    model = TextByGemini() 
    test_sentence = "Công cuộc đổi mới văn học , thực tế đã diễn ra từ sau 1975 , tuy nhiên , trong khoảng thời gian 1975 - 1985 , giới văn nghệ chỉ mới dò đường .",

    print(test_sentence)
    sentence_type = compound_or_simple(model, test_sentence)
    print(sentence_type)
    if sentence_type.strip() == "compound":
        simple_sentences = split_compound_into_simple_sentences(model, test_sentence)
        print(simple_sentences)
    else:
        simple_sentences = [test_sentence]

    more_simple_sentences = []
    for sent in simple_sentences:
        is_complex = complex_or_simple(model, sent)
        print(sent, "----", is_complex)
        if is_complex.strip() == "complex":
            one_clause_sentences = generate_to_rewrite_sentence_with_one_clause(model, sent)
            print(one_clause_sentences)
            more_simple_sentences.extend(one_clause_sentences)
        else:
            more_simple_sentences.extend([sent])  

    more_more_simple_sentences = []
    for sent in more_simple_sentences:
        more_more_simple_sentences.extend(split_into_simple_sentence(model, sent))
    
    more_more_more_simple_sentences = []
    for sent in more_more_simple_sentences:
       filter_sent = remove_unnecessary_words_in_sent(model, sent)
       more_more_more_simple_sentences.append(filter_sent)
    print(more_more_more_simple_sentences)
    phrases_in_sentences = process_into_phrases(model, more_more_more_simple_sentences)
    print(phrases_in_sentences)
    phrases_in_order = change_the_order_of_phrases(phrases_in_sentences)
    print(phrases_in_order)
    spelling_phrase = identify_spelling_in_phrases(phrases_in_order)
    print(spelling_phrase)
    # print(more_simple_sentences)
    
    # simple_sentences = split_into_simple_sentence(model, test_sentence)
    # simple_fitered_sentences = []
    # print(simple_sentences)
    # for sent in simple_sentences:
    #     simple_fitered_sentences.append(remove_unnecessary_words_in_sent(model, sent))
    # print(simple_fitered_sentences)
    # phrases_in_sentences = process_into_phrases(model, simple_fitered_sentences)
    # print(phrases_in_sentences)
    # phrases_type_in_sentences = get_phrases_types(phrases_in_sentences, simple_sentences)
    # for phrase_type in phrases_type_in_sentences:
    #     print(phrase_type)
    # print(phrases_type_in_sentences)
    #    print("---------------------")
    # phrases_in_sentences, phrases_type_in_sentences = remove_unnecessary_phrases(phrases_in_sentences, phrases_type_in_sentences)
    # print(phrases_in_sentences)
    # print(phrases_type_in_sentences)
        
                 
        # prompt = generate_prompt_to_check_pos_and_neg_of_declarative_sentence(test_sentence)
        # pos_neg_sentence = model.generate_text(prompt)
        # print(pos_neg_sentence)
    



if __name__ == '__main__':
    main()
