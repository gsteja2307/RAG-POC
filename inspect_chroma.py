import chromadb

# Path where your local database is stored
chroma_database_path = './chroma_database'

# Collection name
chroma_collection_name = 'rag_documents'

# Create client
chroma_client = chromadb.PersistentClient(path=chroma_database_path)

# Get collection
document_collection = chroma_client.get_or_create_collection(
	name=chroma_collection_name
)


def print_collection_summary() -> None:
	"""Print basic information about the collection"""
	print('\n=== COLLECTION SUMMARY ===')

	total_records = document_collection.count()
	print(f'Total records (chunks): {total_records}')


def print_sample_records(sample_size: int = 5) -> None:
	"""Print some sample stored records"""
	print('\n=== SAMPLE RECORDS ===')

	result = document_collection.peek(limit=sample_size)

	ids = result['ids']
	documents = result['documents']
	metadatas = result['metadatas']

	for index in range(len(ids)):
		print(f'\nRecord {index + 1}')
		print(f'ID: {ids[index]}')
		print(f"File: {metadatas[index]['file_name']}")
		print(f"Chunk: {metadatas[index]['chunk_number']}")
		print(f"Hash: {metadatas[index]['file_content_hash']}")
		print(f'Document Content:\n{documents[index][:200]}...')
		print('-' * 80)


def inspect_chunks_for_specific_file(file_name: str) -> None:
	"""Inspect all chunks belonging to a specific file"""
	print(f'\n=== CHUNKS FOR FILE: {file_name} ===')

	result = document_collection.get(
		where={'file_name': file_name}
	)

	ids = result['ids']
	documents = result['documents']
	metadatas = result['metadatas']

	if not ids:
		print('No records found for this file.')
		return

	for index in range(len(ids)):
		print(f'\nChunk {index + 1}')
		print(f'ID: {ids[index]}')
		print(f"Chunk Number: {metadatas[index]['chunk_number']}")
		print(f"Hash: {metadatas[index]['file_content_hash']}")
		print(f'Content:\n{documents[index]}')
		print('-' * 80)


def test_similarity_query(query_text: str) -> None:
	"""Run a similarity query and inspect raw output"""
	print(f'\n=== SIMILARITY QUERY: "{query_text}" ===')

	# NOTE: We are not embedding here manually because
	# Chroma can also accept query_texts if embeddings were auto-managed.
	# But since we used custom embeddings earlier, we will just inspect
	# using embedding-based query through your main script.
	# So this function is mostly illustrative.

	print('Run this query through your main script to see full retrieval behavior.')


def main() -> None:
	print_collection_summary()
	print_sample_records(sample_size=5)

	# Change this to any file name you want to inspect
	inspect_chunks_for_specific_file('rag-basics.txt')


if __name__ == '__main__':
	main()