import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

# Create LangChain chat model wrapper for Gemini
chat_model = ChatGoogleGenerativeAI(
	model='gemini-2.5-flash',
	google_api_key=os.getenv('GOOGLE_API_KEY')
)

# Create LangChain embeddings wrapper for Gemini
embedding_model = GoogleGenerativeAIEmbeddings(
	model='models/gemini-embedding-001',
	google_api_key=os.getenv('GOOGLE_API_KEY')
)


def main() -> None:
	print('LangChain Gemini objects created successfully.')

	# Small smoke test for the chat model
	response = chat_model.invoke('Explain RAG in one sentence.')

	print('\nChat model response:\n')
	print(response.content)

	# Small smoke test for the embedding model
	embedding_vector = embedding_model.embed_query('What is retrieval augmented generation?')

	print('\nEmbedding vector length:\n')
	print(len(embedding_vector))


if __name__ == '__main__':
	main()