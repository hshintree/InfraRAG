import modal
from typing import List

app = modal.App("infra-embedder")

image = (
	modal.Image.debian_slim()
		.pip_install([
			"transformers",
			"sentence-transformers",
			"torch",
		])
)


@app.function(image=image, timeout=600)
def embed_texts_384(texts: list, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> list:
	from transformers import AutoTokenizer, AutoModel
	import torch
	import torch.nn.functional as F

	tokenizer = AutoTokenizer.from_pretrained(model_name)
	model = AutoModel.from_pretrained(model_name)
	model.eval()

	embeddings = []
	batch_size = 128
	for i in range(0, len(texts), batch_size):
		batch = texts[i:i+batch_size]
		with torch.no_grad():
			inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
			outputs = model(**inputs)
			last_hidden = outputs.last_hidden_state
			attention_mask = inputs['attention_mask'].unsqueeze(-1)
			masked = last_hidden * attention_mask
			sum_embeddings = masked.sum(dim=1)
			sum_mask = attention_mask.sum(dim=1)
			mean_embeddings = sum_embeddings / sum_mask
			normed = F.normalize(mean_embeddings, p=2, dim=1)
			vecs = normed.cpu().tolist()
			embeddings.extend(vecs)
	return embeddings


def embed_texts_remote(texts: List[str], max_batch: int = 1024) -> List[List[float]]:
	"""Call the Modal function in chunks and return embeddings.

	Requires Modal credentials configured locally.
	"""
	results: List[List[float]] = []
	with app.run():
		for i in range(0, len(texts), max_batch):
			batch = texts[i:i+max_batch]
			out = embed_texts_384.remote(batch)
			if hasattr(out, "get"):
				vecs = out.get()
			else:
				vecs = out
			results.extend(vecs)
	return results 