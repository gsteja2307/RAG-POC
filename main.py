import hashlib
import os
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from google import genai

load_dotenv()

# Initialize Gemini client using the API key from the environment file
gemini_client = genai.Client(
	api_key=os.getenv('GEMINI_API_KEY')
)

# Path where the local Chroma database will be stored
chroma_database_path = './chroma_database'

# Name of the Chroma collection that stores document chunks
chroma_collection_name = 'rag_documents'

# Create a persistent Chroma client
chroma_client = chromadb.PersistentClient(path=chroma_database_path)

# Get or create the document collection
document_collection = chroma_client.get_or_create_collection(
	name=chroma_collection_name
)


# ---------- FILE LOADING ----------

def read_text_file(file_path: Path) -> str:
	"""Read and return the content of a text file"""
	with file_path.open('r', encoding='utf-8') as file:
		return file.read()


def load_documents(data_directory_path: str) -> list[dict[str, str]]:
	"""Load all .txt files from the data directory"""
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


# ---------- HASHING ----------

def generate_file_content_hash(file_content: str) -> str:
	"""
	Generate a stable hash for the file content.

	This helps detect whether the file content changed
	since the previous indexing run.
	"""
	return hashlib.sha256(file_content.encode('utf-8')).hexdigest()


# ---------- TEXT CHUNKING ----------

def chunk_text(
	text_content: str,
	max_chunk_size: int = 500,
	overlap_size: int = 100
) -> list[str]:
	"""
	Semantic-aware chunking:
	1. Split by paragraphs
	2. Combine paragraphs into chunks
	3. Fallback split if chunk too large
	4. Add overlap between chunks
	"""

	# Step 1: split into paragraphs
	paragraph_list = [
		paragraph.strip()
		for paragraph in text_content.split('\n')
		if paragraph.strip()
	]

	chunk_list: list[str] = []
	current_chunk_parts: list[str] = []
	current_length = 0

	# Step 2: build chunks from paragraphs
	for paragraph in paragraph_list:
		paragraph_length = len(paragraph)

		# If adding paragraph exceeds limit → flush current chunk
		if current_length + paragraph_length > max_chunk_size and current_chunk_parts:
			chunk_list.append(' '.join(current_chunk_parts))

			current_chunk_parts = []
			current_length = 0

		current_chunk_parts.append(paragraph)
		current_length += paragraph_length

	# Add remaining chunk
	if current_chunk_parts:
		chunk_list.append(' '.join(current_chunk_parts))

	# Step 3: fallback splitting for oversized chunks
	final_chunk_list: list[str] = []

	for chunk in chunk_list:
		if len(chunk) <= max_chunk_size:
			final_chunk_list.append(chunk)
			continue

		# Split large chunk into smaller parts
		start_index = 0

		while start_index < len(chunk):
			end_index = start_index + max_chunk_size
			sub_chunk = chunk[start_index:end_index].strip()

			if sub_chunk:
				final_chunk_list.append(sub_chunk)

			if end_index >= len(chunk):
				break

			start_index += max_chunk_size - overlap_size

	# Step 4: add overlap between chunks
	overlapped_chunk_list: list[str] = []

	for index, chunk in enumerate(final_chunk_list):
		if index == 0:
			overlapped_chunk_list.append(chunk)
			continue

		previous_chunk = overlapped_chunk_list[-1]

		overlap_text = previous_chunk[-overlap_size:]

		new_chunk = overlap_text + ' ' + chunk

		overlapped_chunk_list.append(new_chunk.strip())

	return overlapped_chunk_list

def build_chunk_records_for_document(
	file_name: str,
	file_content: str,
	file_content_hash: str
) -> list[dict[str, object]]:
	"""
	Create chunk records for a single document.

	Each chunk record contains:
	- unique chunk identifier
	- file name
	- chunk number
	- file content hash
	- actual chunk content
	"""
	chunk_list = chunk_text(file_content)
	chunk_record_list: list[dict[str, object]] = []

	for chunk_number, chunk_content in enumerate(chunk_list, start=1):
		chunk_identifier = f'{file_name}::chunk::{chunk_number}'

		chunk_record_list.append(
			{
				'chunk_identifier': chunk_identifier,
				'file_name': file_name,
				'chunk_number': chunk_number,
				'file_content_hash': file_content_hash,
				'chunk_content': chunk_content
			}
		)

	return chunk_record_list


# ---------- EMBEDDINGS ----------

def embed_text_list(text_list: list[str]) -> list[list[float]]:
	"""Generate embeddings for a list of text values"""
	embedding_response = gemini_client.models.embed_content(
		model='gemini-embedding-001',
		contents=text_list
	)

	return [embedding.values for embedding in embedding_response.embeddings]


# ---------- CHROMA INSPECTION ----------

def get_existing_chunks_for_file(file_name: str) -> dict[str, object]:
	"""
	Read existing chunks for a specific file from Chroma.

	We use this to determine:
	- whether the file has already been indexed
	- what hash was used previously
	"""
	query_result = document_collection.get(
		where={'file_name': file_name},
		include=['metadatas']
	)

	return query_result


def get_existing_file_hash(file_name: str) -> str | None:
	"""
	Get the stored file content hash for a given file.

	If the file does not exist in Chroma, return None.
	"""
	query_result = get_existing_chunks_for_file(file_name)

	metadata_list = query_result.get('metadatas', [])

	if not metadata_list:
		return None

	first_metadata = metadata_list[0]

	if not first_metadata:
		return None

	return first_metadata.get('file_content_hash')


def get_all_indexed_file_names() -> set[str]:
	"""
	Return all unique file names currently stored in Chroma.

	We read metadata of all records and extract unique file names.
	"""
	query_result = document_collection.get(include=['metadatas'])

	metadata_list = query_result.get('metadatas', [])

	indexed_file_name_set: set[str] = set()

	for metadata in metadata_list:
		if metadata and 'file_name' in metadata:
			indexed_file_name_set.add(metadata['file_name'])

	return indexed_file_name_set


def get_all_local_file_names(document_list: list[dict[str, str]]) -> set[str]:
	"""
	Return all file names from the local data folder.
	"""
	return {document['file_name'] for document in document_list}


# ---------- CHROMA WRITE OPERATIONS ----------

def delete_existing_chunks_for_file(file_name: str) -> None:
	"""
	Delete all chunks belonging to a specific file.

	We do this before inserting updated chunks for a changed file.
	"""
	existing_query_result = document_collection.get(
		where={'file_name': file_name}
	)

	existing_identifier_list = existing_query_result.get('ids', [])

	if existing_identifier_list:
		document_collection.delete(ids=existing_identifier_list)


def delete_chunks_for_file(file_name: str) -> None:
	"""
	Delete all chunks belonging to a specific file.

	This is used for deleted files that no longer exist locally.
	"""
	query_result = document_collection.get(
		where={'file_name': file_name}
	)

	identifier_list = query_result.get('ids', [])

	if identifier_list:
		document_collection.delete(ids=identifier_list)
		print(f'Deleted stale file from index: {file_name}')


def add_chunk_records_to_vector_database(
	chunk_record_list: list[dict[str, object]]
) -> None:
	"""
	Add new chunk records and their embeddings to Chroma.
	"""
	if not chunk_record_list:
		return

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
			'chunk_number': int(chunk_record['chunk_number']),
			'file_content_hash': str(chunk_record['file_content_hash'])
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


def remove_deleted_files_from_vector_database(
	document_list: list[dict[str, str]]
) -> None:
	"""
	Remove files from Chroma that no longer exist in local data folder.
	"""
	local_file_name_set = get_all_local_file_names(document_list)
	indexed_file_name_set = get_all_indexed_file_names()

	stale_file_name_set = indexed_file_name_set - local_file_name_set

	if not stale_file_name_set:
		print('No deleted files detected.')
		return

	print('\nCleaning up deleted files...')

	for file_name in stale_file_name_set:
		delete_chunks_for_file(file_name)


def synchronize_documents_to_vector_database(
	document_list: list[dict[str, str]]
) -> None:
	"""
	Incrementally synchronize local files into Chroma.

	For each file:
	- compute content hash
	- compare with existing stored hash
	- if unchanged, skip
	- if changed or new, delete old chunks and insert fresh chunks
	"""
	for document in document_list:
		file_name = document['file_name']
		file_content = document['content']
		file_content_hash = generate_file_content_hash(file_content)

		existing_file_hash = get_existing_file_hash(file_name)

		if existing_file_hash == file_content_hash:
			print(f'Skipping unchanged file: {file_name}')
			continue

		print(f'Indexing new or changed file: {file_name}')

		delete_existing_chunks_for_file(file_name)

		chunk_record_list = build_chunk_records_for_document(
			file_name=file_name,
			file_content=file_content,
			file_content_hash=file_content_hash
		)

		add_chunk_records_to_vector_database(chunk_record_list)


# ---------- VECTOR DATABASE QUERY ----------

def query_vector_database(
	question: str,
	top_result_count: int = 4
) -> list[dict[str, object]]:
	"""
	Query Chroma using the embedding of the user's question.
	"""
	question_embedding = embed_text_list([question])[0]

	query_result = document_collection.query(
		query_embeddings=[question_embedding],
		n_results=top_result_count
	)

	retrieved_chunk_list: list[dict[str, object]] = []

	result_identifier_list = query_result['ids'][0]
	result_document_list = query_result['documents'][0]
	result_metadata_list = query_result['metadatas'][0]
	result_distance_list = query_result['distances'][0]

	for (
		chunk_identifier,
		chunk_document,
		chunk_metadata,
		chunk_distance
	) in zip(
		result_identifier_list,
		result_document_list,
		result_metadata_list,
		result_distance_list
	):
		retrieved_chunk_list.append(
			{
				'chunk_identifier': chunk_identifier,
				'file_name': chunk_metadata['file_name'],
				'chunk_number': chunk_metadata['chunk_number'],
				'file_content_hash': chunk_metadata['file_content_hash'],
				'chunk_content': chunk_document,
				'distance': chunk_distance
			}
		)

	return retrieved_chunk_list


# ---------- PROMPT BUILDING ----------

def build_grounded_prompt(
	question: str,
	retrieved_chunk_list: list[dict[str, object]]
) -> str:
	"""
	Build a grounded prompt from retrieved chunks.
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
	Generate the final answer using Gemini and the retrieved context.
	"""
	prompt = build_grounded_prompt(
		question=question,
		retrieved_chunk_list=retrieved_chunk_list
	)

	generation_response = gemini_client.models.generate_content(
		model='gemini-2.5-flash',
		contents=prompt
	)

	return generation_response.text


# ---------- MAIN PROGRAM ----------

def main() -> None:
	print('Loading local documents...')
	document_list = load_documents('data')

	print('Cleaning up deleted files...')
	remove_deleted_files_from_vector_database(document_list)

	print('Synchronizing documents into Chroma vector database...')
	synchronize_documents_to_vector_database(document_list)

	print('\nSystem is ready.\n')

	question = input('Ask a question: ').strip()

	retrieved_chunk_list = query_vector_database(
		question=question,
		top_result_count=4
	)

	print('\nRetrieved chunks:\n')

	for result_number, chunk_record in enumerate(retrieved_chunk_list, start=1):
		print(f'Result {result_number}')
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