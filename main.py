import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai
import chromadb

load_dotenv()

# Initialize Gemini client using API key from environment file
gemini_client = genai.Client(
	api_key=os.getenv('GEMINI_API_KEY')
)

# Path where Chroma will persist the local database on disk
chroma_database_path = './chroma_database'

# Name of the collection inside Chroma
chroma_collection_name = 'rag_documents'

# Initialize Chroma persistent client
# This creates a local on-disk vector database
chroma_client = chromadb.PersistentClient(path=chroma_database_path)

# Create the collection if it does not exist already
# Or get the existing collection if it is already there
document_collection = chroma_client.get_or_create_collection(
	name=chroma_collection_name
)


# ---------- FILE LOADING ----------

def read_text_file(file_path: Path) -> str:
	"""Read and return the contents of a text file"""
	with file_path.open('r', encoding='utf-8') as file:
		return file.read()


def load_documents(data_directory_path: str) -> list[dict[str, str]]:
	"""Load all text documents from the data folder"""
	document_list: list[dict[str, str]] = []
	data_directory = Path(data_directory_path)

	for file_path in sorted(data_directory.glob('*.txt')):
		document_list.append(
			{
				'file_name': file_path.name,
				'content': read_text_file(file_path)
			}
		)

	return document_list


# ---------- TEXT CHUNKING ----------

def chunk_text(
	text_content: str,
	chunk_size: int = 300,
	overlap_size: int = 50
) -> list[str]:
	"""
	Split text into overlapping chunks.

	Why overlap matters:
	Important context may sit at chunk boundaries.
	Overlap reduces the risk of splitting meaning awkwardly.
	"""
	chunk_list: list[str] = []
	start_index = 0
	text_length = len(text_content)

	while start_index < text_length:
		end_index = start_index + chunk_size
		chunk = text_content[start_index:end_index].strip()

		if chunk:
			chunk_list.append(chunk)

		if end_index >= text_length:
			break

		start_index += chunk_size - overlap_size

	return chunk_list


def build_chunk_records(document_list: list[dict[str, str]]) -> list[dict[str, object]]:
	"""
	Convert documents into chunk records with metadata.

	Each record represents one searchable unit inside the vector database.
	"""
	chunk_record_list: list[dict[str, object]] = []

	for document in document_list:
		file_name = document['file_name']
		document_content = document['content']
		chunk_list = chunk_text(document_content)

		for chunk_number, chunk_content in enumerate(chunk_list, start=1):
			chunk_identifier = f'{file_name}::chunk::{chunk_number}'

			chunk_record_list.append(
				{
					'chunk_identifier': chunk_identifier,
					'file_name': file_name,
					'chunk_number': chunk_number,
					'chunk_content': chunk_content
				}
			)

	return chunk_record_list


# ---------- EMBEDDINGS ----------

def embed_text_list(text_list: list[str]) -> list[list[float]]:
	"""
	Create embeddings for a list of texts using Gemini embedding model.
	"""
	embedding_response = gemini_client.models.embed_content(
		model='gemini-embedding-001',
		contents=text_list
	)

	return [embedding.values for embedding in embedding_response.embeddings]


# ---------- CHROMA DATABASE OPERATIONS ----------

def populate_vector_database(chunk_record_list: list[dict[str, object]]) -> None:
	"""
	Store chunk text, metadata, and embeddings in Chroma.

	This function first clears the existing collection data for a clean rebuild.
	That is acceptable for a learning proof of concept.
	"""
	global document_collection

	existing_record_count = document_collection.count()

	if existing_record_count > 0:
		# Delete old collection and recreate it to avoid duplicate inserts
		chroma_client.delete_collection(name=chroma_collection_name)

		document_collection = chroma_client.get_or_create_collection(
			name=chroma_collection_name
		)

	chunk_identifier_list = [
		str(chunk_record['chunk_identifier'])
		for chunk_record in chunk_record_list
	]

	chunk_document_list = [
		str(chunk_record['chunk_content'])
		for chunk_record in chunk_record_list
	]

	chunk_metadata_list = [
		{
			'file_name': str(chunk_record['file_name']),
			'chunk_number': int(chunk_record['chunk_number'])
		}
		for chunk_record in chunk_record_list
	]

	chunk_embedding_list = embed_text_list(chunk_document_list)

	document_collection.add(
		ids=chunk_identifier_list,
		documents=chunk_document_list,
		metadatas=chunk_metadata_list,
		embeddings=chunk_embedding_list
	)


def query_vector_database(
	question: str,
	top_result_count: int = 4
) -> list[dict[str, object]]:
	"""
	Query Chroma using the embedding of the user question.

	Chroma returns the nearest chunks from the vector database.
	"""
	question_embedding = embed_text_list([question])[0]

	query_result = document_collection.query(
		query_embeddings=[question_embedding],
		n_results=top_result_count
	)

	retrieved_chunk_list: list[dict[str, object]] = []

	result_identifiers = query_result['ids'][0]
	result_documents = query_result['documents'][0]
	result_metadatas = query_result['metadatas'][0]
	result_distances = query_result['distances'][0]

	for chunk_identifier, chunk_document, chunk_metadata, chunk_distance in zip(
		result_identifiers,
		result_documents,
		result_metadatas,
		result_distances
	):
		retrieved_chunk_list.append(
			{
				'chunk_identifier': chunk_identifier,
				'file_name': chunk_metadata['file_name'],
				'chunk_number': chunk_metadata['chunk_number'],
				'chunk_content': chunk_document,
				'distance': chunk_distance
			}
		)

	return retrieved_chunk_list


# ---------- PROMPT BUILDING + GENERATION ----------

def build_grounded_prompt(
	question: str,
	retrieved_chunk_list: list[dict[str, object]]
) -> str:
	"""
	Build a prompt that forces the language model to answer from retrieved context only.
	"""
	context_part_list: list[str] = []

	for chunk_record in retrieved_chunk_list:
		file_name = str(chunk_record['file_name'])
		chunk_number = int(chunk_record['chunk_number'])
		chunk_content = str(chunk_record['chunk_content'])

		context_part_list.append(
			f'[Source File: {file_name} | Chunk Number: {chunk_number}]\n{chunk_content}'
		)

	context_text = '\n\n'.join(context_part_list)

	prompt = f"""
You are a helpful assistant.

Answer the question using only the provided context.

Context:
{context_text}

Question:
{question}

Instructions:
- Answer only from the provided context
- If the answer is not present in the context, say: "I could not find that in the provided documents."
- Mention the source file names used
- Keep the answer simple and clear
"""

	return prompt.strip()


def generate_answer_from_retrieved_context(
	question: str,
	retrieved_chunk_list: list[dict[str, object]]
) -> str:
	"""
	Send retrieved context to Gemini and generate the final grounded answer.
	"""
	prompt = build_grounded_prompt(question, retrieved_chunk_list)

	generation_response = gemini_client.models.generate_content(
		model='gemini-2.5-flash',
		contents=prompt
	)

	return generation_response.text


# ---------- MAIN PROGRAM FLOW ----------

def main() -> None:
	print('Loading documents from data folder...')
	document_list = load_documents('data')

	print('Building chunk records...')
	chunk_record_list = build_chunk_records(document_list)

	print('Storing chunks and embeddings in local Chroma database...')
	populate_vector_database(chunk_record_list)

	print('\nSystem is ready.\n')

	question = input('Ask a question: ').strip()

	retrieved_chunk_list = query_vector_database(
		question=question,
		top_result_count=4
	)

	print('\nRetrieved chunks from Chroma:\n')

	for index, chunk_record in enumerate(retrieved_chunk_list, start=1):
		print(f'Result {index}')
		print(f"Source file: {chunk_record['file_name']}")
		print(f"Chunk number: {chunk_record['chunk_number']}")
		print(f"Distance: {chunk_record['distance']}")
		print(f"Content: {chunk_record['chunk_content']}")
		print('-' * 80)

	answer = generate_answer_from_retrieved_context(
		question=question,
		retrieved_chunk_list=retrieved_chunk_list
	)

	print('\nAnswer:\n')
	print(answer)


if __name__ == '__main__':
	main()