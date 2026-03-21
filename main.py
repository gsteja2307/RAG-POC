import os
import json
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from google import genai

load_dotenv()

# Initialize Gemini client using API key from .env
client = genai.Client(
	api_key=os.getenv('GEMINI_API_KEY')
)

# File where we will store embeddings locally
CACHE_FILE = 'embeddings_cache.json'


# ---------- FILE LOADING ----------

def read_text_file(file_path: Path) -> str:
	"""Read a single text file"""
	with file_path.open('r', encoding='utf-8') as file:
		return file.read()


def load_documents(data_dir: str) -> list[dict[str, str]]:
	"""Load all .txt files from data folder"""
	documents: list[dict[str, str]] = []
	data_path = Path(data_dir)

	for file_path in sorted(data_path.glob('*.txt')):
		documents.append(
			{
				'file_name': file_path.name,
				'content': read_text_file(file_path)
			}
		)

	return documents


# ---------- CHUNKING ----------

def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> list[str]:
	"""
	Split text into overlapping chunks
	Overlap helps preserve context across boundaries
	"""
	chunks: list[str] = []
	start = 0

	while start < len(text):
		end = start + chunk_size
		chunk = text[start:end].strip()

		if chunk:
			chunks.append(chunk)

		if end >= len(text):
			break

		start += chunk_size - overlap

	return chunks


def build_chunk_index(documents: list[dict[str, str]]) -> list[dict[str, object]]:
	"""
	Create chunk index with metadata
	This acts like a mini database
	"""
	chunk_index: list[dict[str, object]] = []

	for doc in documents:
		chunks = chunk_text(doc['content'])

		for i, chunk in enumerate(chunks, start=1):
			chunk_index.append(
				{
					'file_name': doc['file_name'],
					'chunk_number': i,
					'content': chunk
				}
			)

	return chunk_index


# ---------- EMBEDDINGS ----------

def embed_texts(texts: list[str]) -> list[list[float]]:
	"""Call Gemini embedding API"""
	result = client.models.embed_content(
		model='gemini-embedding-001',
		contents=texts
	)

	return [e.values for e in result.embeddings]


def add_embeddings(chunk_index: list[dict[str, object]]) -> list[dict[str, object]]:
	"""Attach embeddings to each chunk"""
	texts = [str(chunk['content']) for chunk in chunk_index]
	embeddings = embed_texts(texts)

	enriched = []

	for chunk, emb in zip(chunk_index, embeddings):
		new_chunk = dict(chunk)
		new_chunk['embedding'] = emb
		enriched.append(new_chunk)

	return enriched


# ---------- CACHE (NEW PART) ----------

def save_cache(chunk_index: list[dict[str, object]]) -> None:
	"""Save embeddings to disk"""
	with open(CACHE_FILE, 'w', encoding='utf-8') as f:
		json.dump(chunk_index, f)


def load_cache() -> list[dict[str, object]] | None:
	"""Load embeddings from disk if exists"""
	if not os.path.exists(CACHE_FILE):
		return None

	with open(CACHE_FILE, 'r', encoding='utf-8') as f:
		return json.load(f)


# ---------- SIMILARITY ----------

def cosine_similarity(a: list[float], b: list[float]) -> float:
	"""Compute cosine similarity between two vectors"""
	a = np.array(a)
	b = np.array(b)

	if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
		return 0.0

	return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def retrieve_chunks(
	chunk_index: list[dict[str, object]],
	query: str,
	top_k: int = 4
) -> list[dict[str, object]]:
	"""Find most relevant chunks using embeddings"""
	query_emb = embed_texts([query])[0]

	scored = []

	for chunk in chunk_index:
		score = cosine_similarity(query_emb, chunk['embedding'])
		scored.append((chunk, score))

	scored.sort(key=lambda x: x[1], reverse=True)

	return [chunk for chunk, _ in scored[:top_k]]


# ---------- PROMPT + GENERATION ----------

def build_prompt(question: str, chunks: list[dict[str, object]]) -> str:
	"""Build prompt with context"""
	context_parts = []

	for c in chunks:
		context_parts.append(
			f"[{c['file_name']} | chunk {c['chunk_number']}]\n{c['content']}"
		)

	context = '\n\n'.join(context_parts)

	return f"""
Answer ONLY using the context below.

Context:
{context}

Question:
{question}

Rules:
- If not found, say: I could not find that in the provided documents
- Mention source file names
- Keep answer simple
"""


def ask_llm(question: str, chunks: list[dict[str, object]]) -> str:
	"""Call Gemini for final answer"""
	prompt = build_prompt(question, chunks)

	res = client.models.generate_content(
		model='gemini-2.5-flash',
		contents=prompt
	)

	return res.text


# ---------- MAIN FLOW ----------

def main() -> None:
	print("Loading documents...")
	docs = load_documents('data')

	print("Building chunks...")
	chunks = build_chunk_index(docs)

	# Try loading cache
	cached = load_cache()

	if cached:
		print("Loaded embeddings from cache ✅")
		chunks_with_embeddings = cached
	else:
		print("Generating embeddings (first run)...")
		chunks_with_embeddings = add_embeddings(chunks)

		print("Saving cache...")
		save_cache(chunks_with_embeddings)

	print("\nSystem ready.\n")

	question = input("Ask a question: ").strip()

	retrieved = retrieve_chunks(chunks_with_embeddings, question)

	print("\nRetrieved chunks:\n")

	for i, c in enumerate(retrieved, 1):
		print(f"{i}. {c['file_name']} (chunk {c['chunk_number']})")
		print(c['content'])
		print("-" * 60)

	answer = ask_llm(question, retrieved)

	print("\nAnswer:\n")
	print(answer)


if __name__ == '__main__':
	main()