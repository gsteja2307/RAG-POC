import os
from pathlib import Path
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


def build_chunk_index(documents: list[dict[str, str]]) -> list[dict[str, str | int]]:
	chunk_index: list[dict[str, str | int]] = []

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


def normalize_words(text: str) -> list[str]:
	cleaned_text = text.lower()

	for character in [',', '.', ':', ';', '(', ')', '[', ']', '{', '}', '-', '\n']:
		cleaned_text = cleaned_text.replace(character, ' ')

	words = [word.strip() for word in cleaned_text.split() if word.strip()]

	return words


def score_chunk(chunk_text_value: str, query: str) -> int:
	query_words = normalize_words(query)
	chunk_words = normalize_words(chunk_text_value)
	chunk_word_set = set(chunk_words)

	score = 0

	for word in query_words:
		if word in chunk_word_set:
			score += 1

	return score


def retrieve_relevant_chunks(
	chunk_index: list[dict[str, str | int]],
	query: str,
	top_k: int = 4
) -> list[dict[str, str | int]]:
	scored_chunks: list[tuple[dict[str, str | int], int]] = []

	for chunk in chunk_index:
		content = str(chunk['content'])
		score = score_chunk(content, query)
		scored_chunks.append((chunk, score))

	scored_chunks.sort(key=lambda item: item[1], reverse=True)

	top_chunks = [chunk for chunk, score in scored_chunks if score > 0][:top_k]

	return top_chunks


def build_prompt(question: str, retrieved_chunks: list[dict[str, str | int]]) -> str:
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


def ask_gemini(question: str, retrieved_chunks: list[dict[str, str | int]]) -> str:
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
	print(f'Built {len(chunk_index)} chunks\n')

	question = input('Ask a question: ').strip()

	retrieved_chunks = retrieve_relevant_chunks(chunk_index, question, top_k=4)

	if not retrieved_chunks:
		print('\nNo relevant chunks found.')
		return

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