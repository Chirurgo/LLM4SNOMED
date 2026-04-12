import asyncio
from typing import List, Optional, Dict

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from langchain_core.embeddings import Embeddings



class _BaseHFEmbedder(Embeddings):
    """
    Generic HF encoder -> vector embedder.

    Args:
        model_id: HF model id
        max_length: tokenizer truncate length
        normalize: L2-normalize the final vectors
        pooling: "cls" or "mean" (masked mean pooling)
        batch_size: encode_documents() chunk size
        device: "cuda", "cpu", etc. (auto if None)
        use_fast_tokenizer: use Fast tokenizer if available
        tokenizer_kwargs: extra kwargs for AutoTokenizer.from_pretrained
        model_kwargs: extra kwargs for AutoModel.from_pretrained
    """

    def __init__(
        self,
        model_id: str,
        max_length: int = 256,
        normalize: bool = True,
        pooling: str = "cls",
        batch_size: int = 32,
        device: Optional[str] = None,
        use_fast_tokenizer: bool = True,
        tokenizer_kwargs: Optional[Dict] = None,
        model_kwargs: Optional[Dict] = None,
    ):
        assert pooling in {"cls", "mean"}, "Can only be 'cls' or 'mean'"
        self.model_id = model_id
        self.max_length = max_length
        self.normalize = normalize
        self.pooling = pooling
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        tokenizer_kwargs = tokenizer_kwargs or {}
        model_kwargs = model_kwargs or {}

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, use_fast=use_fast_tokenizer, **tokenizer_kwargs
        )
        self.model = AutoModel.from_pretrained(model_id, **model_kwargs).eval().to(self.device)

    @staticmethod
    def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        summed = (last_hidden_state * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts

    def _encode_batch(self, texts: List[str]) -> np.ndarray:
        tok_max = getattr(self.tokenizer, "model_max_length", None)
        ml = min(self.max_length, tok_max) if tok_max and tok_max < 10_000 else self.max_length

        enc = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            padding=True,          # or "max_length" if you want static shapes
            max_length=ml,
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}

        with torch.inference_mode():
            if self.device == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    outputs = self.model(**enc)
            else:
                outputs = self.model(**enc)

        vec = self._pool(outputs, enc["attention_mask"])
        return vec.detach().cpu().numpy().astype(np.float32)

    def _pool(self, outputs, attention_mask) -> torch.Tensor:
        if self.pooling == "cls":
            vec = outputs.last_hidden_state[:, 0, :]
        else:
            vec = self._mean_pool(outputs.last_hidden_state, attention_mask)
        vec = vec.float()
        if self.normalize:
            vec = torch.nn.functional.normalize(vec, p=2, dim=1)
        return vec


    def _embed(self, text: str) -> np.ndarray:
        return self._encode_batch([text])[0]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        out = []
        for i in range(0, len(texts), self.batch_size):
            out.append(self._encode_batch(texts[i : i + self.batch_size]))
        return np.vstack(out).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self._embed(text).tolist()

    # ---- async API ----
    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.embed_documents(texts))

    async def aembed_query(self, text: str) -> List[float]:
        return (await self.aembed_documents([text]))[0]

class AsyncBioClinicalBERTEmbedder(_BaseHFEmbedder):
    """
    emilyalsentzer/Bio_ClinicalBERT
    cls + L2, max token 256
    """
    def __init__(
        self,
        model_id: str = "emilyalsentzer/Bio_ClinicalBERT",
        max_length: int = 256,
        normalize: bool = True,
        pooling: str = "cls",  # consider "mean" if use as sentence embeddings
        batch_size: int = 32,
        device: str | None = None,
    ):
        super().__init__(
            model_id=model_id,
            max_length=max_length,
            normalize=normalize,
            pooling=pooling,
            batch_size=batch_size,
            device=device,
        )


class AsyncGatorTronEmbedder(_BaseHFEmbedder):
    """
    Default: UFNLP/gatortron-base (BERT-style encoder).
    cls + L2, max token 256
    Note: It's larger than base BERT; ensure you have enough VRAM or use CPU.
    """
    def __init__(
        self,
        model_id: str = "UFNLP/gatortron-base",
        max_length: int = 256,
        normalize: bool = True,
        pooling: str = "cls",  # consider "mean" for retrieval-style embeddings
        batch_size: int = 16,  # gatortron-base is heavier; start smaller
        device: str | None = None,
    ):
        super().__init__(
            model_id=model_id,
            max_length=max_length,
            normalize=normalize,
            pooling=pooling,
            batch_size=batch_size,
            device=device,
        )

class AsyncSapUMLSEmbedder(_BaseHFEmbedder):
    def __init__(
        self,
        model_id: str = "cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR",
        max_length: int = 128,
        normalize: bool = True,
        pooling: str = "cls",
        batch_size: int = 32,
        device: str | None = None,
    ):
        super().__init__(
            model_id=model_id,
            max_length=max_length,
            normalize=normalize,
            pooling=pooling,
            batch_size=batch_size,
            device=device,
            use_fast_tokenizer=False, 
        )

class AsyncSapBERTEmbedder(_BaseHFEmbedder):
    """
    cambridgeltl/SapBERT-from-PubMedBERT-fulltext
    CLS + L2, max_length 128
    """
    def __init__(
        self,
        model_id: str = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
        max_length: int = 128,
        normalize: bool = True,
        pooling: str = "cls", 
        batch_size: int = 32,
        device: Optional[str] = None,
    ):
        super().__init__(
            model_id=model_id,
            max_length=max_length,
            normalize=normalize,
            pooling=pooling,
            batch_size=batch_size,
            device=device,
        )

class AsyncSapBERTpoolEmbedder(_BaseHFEmbedder):
    """
    cambridgeltl/SapBERT-from-PubMedBERT-fulltext
    CLS + L2, max_length 128
    """
    def __init__(
        self,
        model_id: str = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
        max_length: int = 128,
        normalize: bool = True,
        pooling: str = "mean",    # set to "mean" for retrieval?
        batch_size: int = 32,
        device: Optional[str] = None,
    ):
        super().__init__(
            model_id=model_id,
            max_length=max_length,
            normalize=normalize,
            pooling=pooling,
            batch_size=batch_size,
            device=device,
        )

class AsyncPubMedBERTEmbedder(_BaseHFEmbedder):
    """
    microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext
    (formerly PubMedBERT)
    CLS, L2 normalization, 256 tokens
    """
    def __init__(
        self,
        model_id: str = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
        max_length: int = 256,
        normalize: bool = True,       
        pooling: str = "cls",
        batch_size: int = 32,
        device: Optional[str] = None,
    ):
        super().__init__(
            model_id=model_id,
            max_length=max_length,
            normalize=normalize,
            pooling=pooling,
            batch_size=batch_size,
            device=device,
        )


class AsyncGTEMultilingualEmbedder(_BaseHFEmbedder):
    """
    Alibaba-NLP/gte-multilingual-mlm-base

    Masked-mean pooling, L2 normalization, 512 tokens
    ** If remote code is required , pass model_kwargs={"trust_remote_code": True}.
      (Windows sometimes trips on HF remote-code prompts; this keeps it explicit.)
    """
    def __init__(
        self,
        model_id: str = "Alibaba-NLP/gte-multilingual-mlm-base",
        max_length: int = 256,
        normalize: bool = True,
        pooling: str = "mean",
        batch_size: int = 32,
        device: Optional[str] = None,
        tokenizer_kwargs: Optional[Dict] = None,
        model_kwargs: Optional[Dict] = None,   # e.g., {"trust_remote_code": True}
    ):
        model_kwargs = {"trust_remote_code": True, **(model_kwargs or {})}
        super().__init__(
            model_id=model_id,
            max_length=max_length,
            normalize=normalize,
            pooling=pooling,
            batch_size=batch_size,
            device=device,
            tokenizer_kwargs=tokenizer_kwargs,
            model_kwargs=model_kwargs,
        )


class PubMedBERTEmbedder(Embeddings):
    def __init__(self, model_id="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id)

    def _embed(self, text: str) -> np.ndarray:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
            return outputs.last_hidden_state[:, 0, :].squeeze().numpy().astype(np.float32)

    # --- Required sync methods ---
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t).tolist() for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text).tolist()

    # --- Async methods used by Neo4jVector ---
    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        loop = asyncio.get_event_loop()
        return await asyncio.gather(*(loop.run_in_executor(None, self._embed, t) for t in texts))

    async def aembed_query(self, text: str) -> list[float]:
        return (await self.aembed_documents([text]))[0]
    

class GTEMultilingualEmbedder(Embeddings):
    def __init__(self, model_id="Alibaba-NLP/gte-multilingual-mlm-base"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id, trust_remote_code=True)

    def _embed(self, text: str) -> np.ndarray:
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        )
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)
            return embeddings.squeeze().numpy().astype(np.float32)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t).tolist() for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text).tolist()

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        loop = asyncio.get_event_loop()
        embeddings_np = await asyncio.gather(*(loop.run_in_executor(None, self._embed, t) for t in texts))
        return [emb.tolist() for emb in embeddings_np]

    async def aembed_query(self, text: str) -> list[float]:
        results = await self.aembed_documents([text])
        return results[0]