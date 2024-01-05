import os
import sys
from time import sleep
from dotenv import load_dotenv
import json
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

from utils import read_json_file
import chromadb
from chromadb.utils import embedding_functions


def query_gemini_embedding(db_path, query_word_list, collection_name, k_neighbors=2):
    client = chromadb.PersistentClient(path=db_path)

    # create embedding function
    embedding_function = embedding_functions.GoogleGenerativeAiEmbeddingFunction(api_key=os.getenv('GOOGLE_API_KEY'), task_type="clustering")

    # Get the collection.
    collection = client.get_collection(
        name=collection_name, embedding_function=embedding_function
    )
    
    results = collection.query(
            query_texts=query_word_list, n_results=k_neighbors, include=["documents", "metadatas"]
        )
    return results['documents'], results['metadatas']

def main():
    test_dataset_path = '../data/embedding/test_dataset.json'
    db_path = "../data/embedding/gemini_embedding_db"
    collection_name = "test_gemini"
    k_neighbors = 2
    test_dataset = read_json_file(test_dataset_path)
    test_word_list = []
    for key, value in test_dataset.items():
        test_word_list.append(key)
    test_word_list = list(set(test_word_list))
    document_list, _ = query_gemini_embedding(db_path, test_word_list, collection_name, k_neighbors)
    # save to json file
    res_list= []
    for index, test_word in enumerate(test_word_list):
        res_list.append({
            "input": test_word,
            "output": document_list[index],
            "gt": test_dataset[test_word]
        })  
    with open('../data/embedding/output_embedding_dataset.json', 'w', encoding="utf-8") as f:
        json.dump(res_list, f, indent=4, ensure_ascii=False)

if __name__ == '__main__':
    main()