import os
import sys
from time import sleep
from dotenv import load_dotenv
import json
import unidecode
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

import chromadb
from chromadb.utils import embedding_functions
from utils import TextByGemini, read_json_file

def generate_translation_prompt(input_word, context_sent):
    return f"""
    Vietnamese word: đồng phục
    Context sentence: Nhà trường may đồng phục cho học sinh
    Translate Vietnamese word to English with the context sentence and identify the type of the word.
    English word: uniform
    ------------------------
    Vietnamese word: {input_word}
    Context sentence: {context_sent}
    Translate Vietnamese word to English, push the english word result in <eng></eng>. Let's think step-by-step.
    English word:
    """

def extract_word_from_output(output, pattern):
    result = output.split(f"<{pattern}>")[1].split(f"</{pattern}>")[0].strip()
    return result


def translate_word(model, input_word, context_sent):
    prompt = generate_translation_prompt(input_word, context_sent)
    output = model.generate_text(prompt)
    en_word = extract_word_from_output(output, "eng")
    if en_word == "":
        en_word = input_word
    return en_word

class SimWordByGemini:
    def __init__(self) -> None:
        self.client = chromadb.PersistentClient(path=os.getenv('GEMINI_EMBEDDING_DB_PATH'))
        self.embedding_function = embedding_functions.GoogleGenerativeAiEmbeddingFunction(api_key=os.getenv('GOOGLE_API_KEY'), task_type="semantic_similarity")
        self.collection = self.client.get_collection(
            name=os.getenv('GEMINI_EMBEDDING_COLLECTION_NAME'), embedding_function=self.embedding_function
        )
    def query(self, query_texts, n_results=5):
        results = self.collection.query(
            query_texts=query_texts, n_results=n_results, include=["documents", "metadatas", "distances"]
        )
        return results
    
    def get_preprocessed_text(self, text):
        return unidecode.unidecode(text.lower().strip())
    

def generate_sim_sent_using_words_from_vocab(model, word_list, context_sent):
    prompt = f"""
    sentence: "{context_sent}"
    You are a Vietnamese linguistic expert. Rewrite the sentence by only use only words: {" ,".join(word_list)}.
    output: 
    """ 
    sentence_type = model.generate_text(prompt)
    return sentence_type

    

if __name__ == '__main__':
    # word_list = ["sau khi", "bà", "mất", "An", "buồn bã", "trở lại", "trường"]
    # context_sent = "Sau khi bà mất, An buồn bã trở lại trường."
    context_sent = "Các học sinh chuẩn bị cho buổi lễ tốt nghiệp."
    word_list = ["chuẩn bị", "cho", "buổi lễ", "tốt nghiệp", "các học sinh"]
    model = TextByGemini()
    sim_word_model = SimWordByGemini()
    translated_words = []
    for word in word_list:
        en_word = translate_word(model, word, context_sent)
        print(f"input: {word} --- output: {en_word}")
        translated_words.append(en_word)
        sleep(1)
    
    # query similar words
    query_texts = [sim_word_model.get_preprocessed_text(word) for word in translated_words]
    results = sim_word_model.query(query_texts, n_results=5)

    # print word and similar words
    sim_word_list = []
    for i, word in enumerate(translated_words):
        sim_word_info_list = results['metadatas'][i]
        for sim_word_info in sim_word_info_list:
            sim_word = list(sim_word_info.keys())[0]
            sim_word_list.append(sim_word)
    
    print(sim_word_list)
    sim_word_list = list(set(sim_word_list))
    output = generate_sim_sent_using_words_from_vocab(model, sim_word_list, context_sent)
    print(output)  



    
