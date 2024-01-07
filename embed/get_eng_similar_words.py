import os
import sys
from time import sleep
from dotenv import load_dotenv
import json
import unidecode
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

from utils import read_json_file
import chromadb
from chromadb.utils import embedding_functions


def query_gemini_embedding(db_path, query_word_list, collection_name, k_neighbors=2):
    client = chromadb.PersistentClient(path=db_path)

    # create embedding function
    embedding_function = embedding_functions.GoogleGenerativeAiEmbeddingFunction(api_key=os.getenv('GOOGLE_API_KEY'), task_type="semantic_similarity")

    # Get the collection.
    collection = client.get_collection(
        name=collection_name, embedding_function=embedding_function
    )
    tokenized_query_word_list = [word.replace(" ", "_") for word in query_word_list]
    results = collection.query(
            query_texts=tokenized_query_word_list, n_results=k_neighbors, include=["documents", "metadatas"]
        )
    return results['documents'], results['metadatas']

def main():
    test_dataset_path = '../data/embedding/test_dataset_vi_en.json'
    db_path = "../data/embedding/gemini_embedding_en_vi_db"
    collection_name = "gemini_en_vi"
    k_neighbors = 2
    test_dataset = read_json_file(test_dataset_path)
    vi_word_list = []
    eng_word_list = []
    gt_word_list = []
    for key, value in test_dataset.items():
        vi_word_list.append(key)
        eng_word_list.append(value["origin_en_word"])
        gt_word_list.append(value["vi"])

    document_list, pred_list = query_gemini_embedding(db_path, eng_word_list, collection_name, k_neighbors)
    # save to json file
    res_list= []
    for index, test_word in enumerate(vi_word_list):
        res_list.append({
            "input": test_word,
            "output": pred_list[index],
            "gt": gt_word_list[index]
        })  
    with open('../data/embedding/output_en_semantic_similarity_embedding_dataset.json', 'w', encoding="utf-8") as f:
        json.dump(res_list, f, indent=4, ensure_ascii=False)

if __name__ == '__main__':
    main()