from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from models.models import embeddings_model
from langchain_postgres.vectorstores import PGVector
from dotenv import load_dotenv
import os
load_dotenv()

database = os.getenv("DATABASE_URL")

def upload():

	loader = DirectoryLoader("Documents")
	docs = loader.load()

	splitter = RecursiveCharacterTextSplitter(
												chunk_size=1000,
												chunk_overlap=200,
											)

	splittedDocuments = splitter.split_documents(docs)

	update_db = PGVector.from_documents(
								documents=splittedDocuments,
								embedding=embeddings_model,
								connection=database
								)
	print("uploaded documents as vectorstores")
	return 1
