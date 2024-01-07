import os
import sys
from time import sleep
from dotenv import load_dotenv
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))


from utils import read_json_file
import chromadb
from chromadb.utils import embedding_functions
import unidecode


def load_word_list(file_path):
    with open(file_path, 'r') as f:
        return f.readlines()
    
def embedding_chromadb(word_list, metadata_list, id_list, collection_name, CHROMA_DATA_PATH):
    client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)
    embedding_function = embedding_functions.GoogleGenerativeAiEmbeddingFunction(api_key=os.getenv('GOOGLE_API_KEY'), task_type="semantic_similarity")

    collection = client.get_or_create_collection(
        name=collection_name, embedding_function=embedding_function
    )
    word_list = [unidecode.unidecode(word).lower() for word in word_list]
    # remove mark, sign of vietnamese word

    collection.add(
     documents=word_list,
     ids=id_list,
     metadatas=metadata_list
    )


def main():
    qipedc_file_path = '../data/translation/qipedc_dataset_duplication_removal_vi_en.json'
    CHROMA_DATA_PATH = "../data/embedding/gemini_embedding_en_vi_db"
    dataset_name = "gemini_en_vi"
    test_dataset_path = '../data/embedding/test_dataset_vi_en.json'
    
    qppedc_dataset = read_json_file(qipedc_file_path)
    test_dataset = read_json_file(test_dataset_path)
    
    vi_words_list = []
    en_words_list = []
    for items in qppedc_dataset:
        # print(items)
        en_words_list.append(items["en_word"].strip())
        vi_words_list.append({items["word"].strip(): items["en_word"]})
    
    for key, value in test_dataset.items():
        vi_words_list.append({value["vi"]: value["en_word"]})
        en_words_list.append(value["en_word"])

    # create id_list
    id_list = []
    for i in range(len(vi_words_list)):
        id_list.append(str(i))
    print(len(vi_words_list), len(en_words_list), len(id_list))
    print(vi_words_list[0], en_words_list[0], id_list[0])
    embedding_chromadb(en_words_list, vi_words_list, id_list, dataset_name, CHROMA_DATA_PATH)

if __name__ == '__main__':
    main()
    
    
    
