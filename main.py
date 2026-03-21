import os
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(
	api_key=os.getenv('GEMINI_API_KEY')
)


def read_text_file(file_path: Path) -> str:
	with file_path.open('r', encoding='utf-8') as file:
		return file.read()


def load_documents(data_dir: str) -> list[dict[str, str]]:
	documents: list[dict[str, str]] = []
	data_path = Path(data_dir)

	for file_path in sorted(data_path.glob('*.txt')):
		content = read_text_file(file_path)

		documents.append(
			{
				'file_name': file_path.name,
				'content': content
			}
		)

	return documents


def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> list[str]:
	chunks: list[str] = []
	start_index = 0
	text_length = len(text)

	while start_index < text_length:
		end_index = start_index + chunk_size
		chunk = text[start_index:end_index].strip()

		if chunk:
			chunks.append(chunk)

		if end_index >= text_length:
			break

		start_index += chunk_size - overlap

	return chunks


def build_chunk_index(documents: list[dict[str, str]]) -> list[dict[str, object]]:
	chunk_index: list[dict[str, object]] = []

	for document in documents:
		file_name = document['file_name']
		content = document['content']
		chunks = chunk_text(content, chunk_size=300, overlap=50)

		for chunk_number, chunk_text_value in enumerate(chunks, start=1):
			chunk_index.append(
				{
					'file_name': file_name,
					'chunk_number': chunk_number,
					'content': chunk_text_value
				}
			)

	return chunk_index


def embed_texts(texts: list[str]) -> list[list[float]]:
	result = client.models.embed_content(
		model='gemini-embedding-001',
		contents=texts
	)

	return [embedding.values for embedding in result.embeddings]


def cosine_similarity(vector_a: list[float], vector_b: list[float]) -> float:
	array_a = np.array(vector_a)
	array_b = np.array(vector_b)

	norm_a = np.linalg.norm(array_a)
	norm_b = np.linalg.norm(array_b)

	if norm_a == 0 or norm_b == 0:
		return 0.0

	return float(np.dot(array_a, array_b) / (norm_a * norm_b))


def add_embeddings_to_chunks(chunk_index: list[dict[str, object]]) -> list[dict[str, object]]:
	chunk_texts = [str(chunk['content']) for chunk in chunk_index]
	embeddings = embed_texts(chunk_texts)

	enriched_chunks: list[dict[str, object]] = []

	for chunk, embedding in zip(chunk_index, embeddings):
		enriched_chunk = dict(chunk)
		enriched_chunk['embedding'] = embedding
		enriched_chunks.append(enriched_chunk)

	return enriched_chunks


def retrieve_relevant_chunks(
	chunk_index: list[dict[str, object]],
	query: str,
	top_k: int = 4
) -> list[dict[str, object]]:
	query_embedding = embed_texts([query])[0]
	scored_chunks: list[tuple[dict[str, object], float]] = []

	for chunk in chunk_index:
		chunk_embedding = chunk['embedding']
		similarity = cosine_similarity(query_embedding, chunk_embedding)
		scored_chunks.append((chunk, similarity))

	scored_chunks.sort(key=lambda item: item[1], reverse=True)

	return [chunk for chunk, _ in scored_chunks[:top_k]]


def build_prompt(question: str, retrieved_chunks: list[dict[str, object]]) -> str:
	context_parts: list[str] = []

	for chunk in retrieved_chunks:
		file_name = str(chunk['file_name'])
		chunk_number = int(chunk['chunk_number'])
		content = str(chunk['content'])

		context_parts.append(
			f'[Source: {file_name} | Chunk: {chunk_number}]\n{content}'
		)

	context_text = '\n\n'.join(context_parts)

	prompt = f"""
You are a helpful assistant answering questions only from the provided context.

Context:
{context_text}

Question:
{question}

Instructions:
- Answer only using the provided context
- If the answer is not present in the context, say: "I could not find that in the provided documents."
- Mention the source file names used in the answer
- Keep the answer simple and clear
"""

	return prompt.strip()


def ask_gemini(question: str, retrieved_chunks: list[dict[str, object]]) -> str:
	prompt = build_prompt(question, retrieved_chunks)

	response = client.models.generate_content(
		model='gemini-2.5-flash',
		contents=prompt
	)

	return response.text


def main() -> None:
	documents = load_documents('data')
	chunk_index = build_chunk_index(documents)

	print(f'Loaded {len(documents)} documents')
	print(f'Built {len(chunk_index)} chunks')
	print('Creating embeddings for all chunks...\n')

	chunk_index_with_embeddings = add_embeddings_to_chunks(chunk_index)

	question = input('Ask a question: ').strip()

	retrieved_chunks = retrieve_relevant_chunks(
		chunk_index_with_embeddings,
		question,
		top_k=4
	)

	print('\nRetrieved Chunks:\n')

	for index, chunk in enumerate(retrieved_chunks, start=1):
		file_name = str(chunk['file_name'])
		chunk_number = int(chunk['chunk_number'])
		content = str(chunk['content'])

		print(f'[{index}] {file_name} | Chunk {chunk_number}')
		print(content)
		print('-' * 80)

	answer = ask_gemini(question, retrieved_chunks)

	print('\nAnswer:\n')
	print(answer)


if __name__ == '__main__':
	main()