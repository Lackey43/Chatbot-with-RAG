from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
from pydantic import BaseModel,Field
load_dotenv()


class is_LOTR(BaseModel):
	"""Determines if the query is about Lord of the rings """
	is_about_lotr: bool = Field(
		description="Return true if the query is related to Lord of the Rings, false otherwise.")


openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")

if openrouter_api_key is None:
    raise ValueError("OPENROUTER_API_KEY not found in environment variables!")

embeddings_model = OpenAIEmbeddings(model="nvidia/llama-nemotron-embed-vl-1b-v2:free",
						openai_api_key=openrouter_api_key,
            			base_url="https://openrouter.ai/api/v1",
            			check_embedding_ctx_length=False,
            			model_kwargs={"encoding_format": "float"}
	                    )

google_model = ChatGoogleGenerativeAI(
									    model="gemini-3.1-flash-lite",
										api_key=google_api_key,
										temperature=0.5,
										max_tokens=3000
									)

query_checker = google_model.with_structured_output(is_LOTR)