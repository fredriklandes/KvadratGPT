import csv

import os
import platform
import uuid
from typing import List, Dict, Any
import json
import openai
import chromadb
import langchain
import pandas as pd
import tiktoken as tiktoken
from chromadb import Settings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
import chromadb

import ConsultantProfile
from WebScraper import WebScraper
# from langchain.vectorstores import
from chromadb.utils import embedding_functions
from langchain.vectorstores import ElasticVectorSearch


def scrape_data():
    scraper = WebScraper()
    for i in range(1, 2):
        starting_url = f'https://www.kvadrat.se/anlita-kvadrat/hitta-konsult/?q=&t=&s=&l=&p={i}#results'
        scraper.scrape_consultant_profile_pages(starting_url)
    return scraper.consultant_profiles


def save_data_csv(consultant_profiles: ConsultantProfile.ConsultantProfile):
    filename = "profiles.json"
    # new pandas dataframe
    total_df = pd.DataFrame(columns=['name',
                                     'title',
                                     'preamble',
                                     'article',
                                     'competence_list',
                                     'cv_list',
                                     'employment_list',
                                     'education_list'])

    serialized ='['
    for profile in consultant_profiles:
        profile.name = profile.name.strip()
        profile.title = profile.title.strip()
        profile.preamble = profile.preamble.strip()
        profile.article = profile.article.strip()

        # append all competences to a string
        json_data = json.dumps(profile.__dict__, ensure_ascii=False).encode('utf8').decode()
        if profile == consultant_profiles[-1]:
            serialized += json_data
        else:
            serialized += json_data + ','

    serialized += ']'

    pdf = pd.read_json(serialized)
    pdf.to_json(filename, orient='records', force_ascii=False)

        #
        # competence_list = f'{profile.name}s kompetenser:'
        # for competence in profile.competence_list:
        #     competence_list += competence + ', '
        # competence_list += '\n'
        #
        # # append all cv items to a string
        # cv_list = f'{profile.name}s cv:'
        # for cv in profile.cv_list:
        #     cv_list += cv + ', '
        # cv_list += '\n'
        #
        # # append all employment items to a string
        # employment_list = f'{profile.name}s anställningar:'
        # for employment in profile.employment_list:
        #     employment_list += employment + ', '
        # employment_list += '\n'
        #
        # # append all education items to a string
        # education_list = f'{profile.name}s utbildningar:'
        # for education in profile.education_list:
        #     education_list += education + ', '
        # education_list += '\n'
        # # {profile.article}
        # profile_text = f"Konsultens namn: {profile.name}. Yrkesområden: {profile.title}." \
        #                f" {competence_list}. {cv_list}. {employment_list}. {education_list}"
        # with open(filename, 'a') as f:
        #     f.write(str(profile_text))
        #     f.write('\n')


def save_data_csv_old(consultant_profiles):
    # Save the scraped data to a CSV file
    filename = "dtos.csv"
    fieldnames = ['name', 'title', 'preamble', 'article', 'competence_list', 'cv_list', 'employment_list',
                  'education_list']
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write the header row
        writer.writeheader()

        # Write the data rows
        for profile in consultant_profiles:
            list_writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            # Convert the ConsultantProfile instance to a dictionary
            profile_dict = profile.__dict__

            # Write the dictionary to the CSV file
            writer.writerow(profile_dict)


def reset_and_fill_db(client, embedding_function):
    client.reset()
    # with open("profiles.txt") as f:
    #     profiles = f.read()
    pandas_profiles = pd.read_csv("dtos.csv", encoding="utf-8", header=1)

    text_splitter = CharacterTextSplitter(chunk_overlap=220, chunk_size=1000, separator='\n')
    texts = text_splitter.split_text(profiles)
    guids = [str(uuid.uuid4()) for _ in range(len(texts))]

    print(f"Number of texts: {len(texts)}")
    print(f"Number of guids: {len(guids)}")

    openai_collection = client.create_collection(name="openai_embeddings", embedding_function=embedding_function)
    openai_collection.add(
        documents=texts,
        metadatas=[{"source": f"Text chunk {i} of {len(texts)}"} for i in range(len(texts))],
        ids=guids
    )
    client.persist()


def apply_prompt_template(question: str) -> str:
    """
        A helper function that applies additional template on user's question.
        Prompt engineering could be done here to improve the result. Here I will just use a minimal example.
    """
    prompt = f"""
        By considering above input from me, answer the question: {question}
    """
    return prompt


def call_chatgpt_api(user_question: str, chunks: List[str]) -> Dict[str, Any]:
    """
    Call chatgpt api with user's question and retrieved chunks.
    """
    # Send a request to the GPT-3 API
    messages = [
        {"role": "system", "content": "You answer questions about consultants at Kvadrat."},
        {"role": "user", "content": message},
    ]
    # question = apply_prompt_template(user_question)
    # messages.append({"role": "user", "content": question})
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=1024,
        temperature=0.7,  # High temperature leads to a more creative response.
    )
    return response


def num_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    """Return the number of tokens in a string."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))


def get_message(client, openai_ef, user_question):
    collection = client.get_collection(name="openai_embeddings", embedding_function=openai_ef)
    results = collection.query(
        query_texts=[user_question],
        n_results=10
        # where={"metadata_field": "is_equal_to_this"},
        # where_document={"$contains":"search_string"}
    )
    model = "gpt-3.5-turbo"
    token_budget: int = 4096 - 800  # 500 is a safety buffer
    chunks = results.get('documents')[0]
    introduction = 'You will be given information about specific consultants. ' \
                   'The consultnats work at Kvadrat. Use that information given below to answer the subsequent question.' \
                   ' If the answer cannot be found in the articles, write "I could not find an answer."'
    question = f"\n\nQuestion: {user_question}"
    message = introduction
    for string in chunks:
        next_article = f'\n\nConsultant section:\n"""\n{string}\n"""'
        if (
                num_tokens(message + next_article + question, model=model)
                > token_budget
        ):
            break
        else:
            message += next_article

    return message + question


if __name__ == '__main__':
    consultant_profiles = scrape_data()
    save_data_csv(consultant_profiles)

    # Syntax of read_json()
    pdf = pd.read_json("profiles.json")

    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key="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",
        model_name="text-embedding-ada-002"
    )
    client = chromadb.Client(Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory=".chromadb/"))
    reset_and_fill_db(client, openai_ef)
    user_question = "Räkna upp alla kompetenser som finns inom Kvadrat och inkludera antalet konsulter per kompetens."
    message = get_message(client, openai_ef, user_question)
    response = call_chatgpt_api(user_question, message)
    response_text = response["choices"][0]["message"]["content"]

    # collection = client.get_collection(name="openai_embeddings", embedding_function=openai_ef)

    print(response_text)

    # chroma_client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory="./db"))
    # docsearch = chroma_client.from_texts(texts, embeddings,
    #                               metadatas=[{"source": f"Text chunk {i} of {len(texts)}"} for i in range(len(texts))],
    #                               persist_directory="db")
    # docsearch.persist()
    # docsearch = None
