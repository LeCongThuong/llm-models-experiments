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
    word_list = [word.replace(" ", "_") for word in word_list]
    # remove mark, sign of vietnamese word

    collection.add(
     documents=word_list,
     ids=id_list,
     metadatas=metadata_list
    )


def main():
    word_list_file_path = '../data/embedding/word_list.txt'
    CHROMA_DATA_PATH = "../data/embedding/gemini_embedding_db"
    word_list = load_word_list(word_list_file_path)

    test_dataset_path = '../data/embedding/test_dataset.json'
    test_dataset = read_json_file(test_dataset_path)
    test_word_list = []
    for key, value in test_dataset.items():
        word_list.append(value)
        test_word_list.append(key)
    word_list = list(set(word_list))
    metadata_list = []
    id_list = []
    for id, word in enumerate(word_list):
        word = word.strip()
        if word in test_word_list:
            metadata_list.append({word: test_dataset[word]})
        else:
            metadata_list.append({word: ""})
        id_list.append(str(id))

    embedding_chromadb(word_list, metadata_list, id_list, "test_gemini_4", CHROMA_DATA_PATH)

if __name__ == '__main__':
    main()
    
    
    
