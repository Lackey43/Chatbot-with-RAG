from models.models import embeddings_model, google_model, query_checker
from langchain_postgres import PGVector
from dotenv import load_dotenv
import json
import os
from prompt.prompts import simple_prompt, default_prompt

load_dotenv()


database = os.getenv("DATABASE_URL")

def start_database():
    db = PGVector(
        embeddings=embeddings_model,
        connection=database,
            )
    print("Database connected...\n")
    return db

def save_json(documents):
    docs_as_dict = [doc.model_dump(mode="json") for doc in documents]
    
    with open("file.json", "w", encoding="utf-8") as j:
        json.dump(docs_as_dict, j, ensure_ascii=False, indent=2)
    
    print("saved relevant data file\n")

def start_query(query: str):
    # check if query is about lord of the rings
    if check_query(query):
        retriever = db.as_retriever(search_kwargs= {"k": 8})

        print("Retrieving Data...\n ")

        unsorted_data = retriever.invoke(query)
        save_json(unsorted_data)

        relevant_data = [data.page_content for data in unsorted_data]
        stripped_data = [data.replace("\n\n", " ") for data in relevant_data]
        joined_relevant_data = "\n\n".join(stripped_data)
        # print(joined_relevant_data + "\n")
        chain = simple_prompt | google_model
        return chain.invoke({
                            "context": joined_relevant_data,
                            "question": query
                                }

                            )
    else:
        chain = default_prompt | google_model
        return  chain.invoke({
                            "question": query
                            }
                            )

def check_query(query:str) -> bool:
    response = query_checker.invoke(query)
    return response.is_about_lotr


if __name__ == "__main__":
    db = start_database()
    while True:
        query = input("enter input:\n")
        if query.lower() in ["quit","exit"]:
            break
        response = start_query(query)
        print(response.content[-1]["text"])
