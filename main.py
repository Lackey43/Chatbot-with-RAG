from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_postgres.vectorstores import PGVector
from dotenv import load_dotenv
from langchain_core.documents import Document
import os
import uuid
load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")
database = os.getenv("DATABASE_URL")
model = OpenAIEmbeddings(model="nvidia/llama-nemotron-embed-vl-1b-v2:free",
						openai_api_key=api_key,
            			base_url="https://openrouter.ai/api/v1",
            			check_embedding_ctx_length=False,
            			encoding_format="float"
	)
loader = PyPDFLoader("Learning LangChain (Mayo Oshin  Nuno Campos)_bibis.ir.pdf")
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(
	chunk_size=1000,
	chunk_overlap=200,
	)
splittedDocuments = splitter.split_documents(docs)

if __name__ == "__main__":
	db = PGVector.from_documents(splittedDocuments,model,connection=database)
	search = db.similarity_search("Postgres", k=4)
	print(search)

	
	# print(splittedDocuments[3].page_content)