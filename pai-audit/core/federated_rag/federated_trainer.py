"""
Federated embedding trainer — FedAvg fine-tuning of a shared embedding model.

Each institution trains locally on its own documents. Only model weight
updates (gradients) are sent to the central server for aggregation.
No raw documents or embeddings ever leave the institution.

Compatible with organoid-fl's FedAvg architecture.
"""

from __future__ import annotations

import copy
import logging
import random
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from .config import FederatedConfig
from .document_loader import Document, DocumentLoader
from .embeddings import EmbeddingEngine
from .vector_store import VectorStore

logger = logging.getLogger(__name__)


@dataclass
class ClientState:
    """State of a single federated client (institution)."""
    client_id: str
    institution_name: str
    documents: List[Document] = field(default_factory=list)
    embeddings: Optional[np.ndarray] = None
    local_model: Optional[nn.Module] = None
    num_samples: int = 0


@dataclass
class RoundMetrics:
    """Metrics from a single federated training round."""
    round_number: int
    participating_clients: List[str]
    avg_loss: float
    client_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)


class FederatedEmbeddingTrainer:
    """
    Federated training of a sentence-transformers embedding model.

    Each client fine-tunes the model on its local document pairs
    (using MultipleNegativesRankingLoss — no labels needed).
    The server aggregates weight updates via FedAvg.

    Usage:
        trainer = FederatedEmbeddingTrainer(config)
        trainer.register_client("give_well", documents_gw)
        trainer.register_client("hospital_a", documents_ha)
        metrics = trainer.train()
    """

    def __init__(
        self,
        config: Optional[FederatedConfig] = None,
        model_name: str = "all-MiniLM-L6-v2",
    ) -> None:
        self._config = config or FederatedConfig()
        self._model_name = model_name
        self._clients: Dict[str, ClientState] = {}
        self._global_model: Optional[nn.Module] = None
        self._history: List[RoundMetrics] = []

    @property
    def num_clients(self) -> int:
        return len(self._clients)

    @property
    def history(self) -> List[RoundMetrics]:
        return list(self._history)

    def register_client(
        self,
        client_id: str,
        documents: List[Document],
        institution_name: str = "",
    ) -> None:
        """
        Register a federated client with its local documents.

        Args:
            client_id: Unique identifier for this client.
            documents: List of local documents for training.
            institution_name: Human-readable institution name.
        """
        if client_id in self._clients:
            raise ValueError(f"Client '{client_id}' already registered")

        state = ClientState(
            client_id=client_id,
            institution_name=institution_name or client_id,
            documents=documents,
            num_samples=len(documents),
        )
        self._clients[client_id] = state
        logger.info(
            "Registered client '%s' (%s) with %d documents",
            client_id, institution_name, len(documents),
        )

    def train(self) -> List[RoundMetrics]:
        """
        Run federated training for the configured number of rounds.

        Returns:
            List of RoundMetrics, one per round.
        """
        if self.num_clients < self._config.min_clients:
            raise ValueError(
                f"Need at least {self._config.min_clients} client(s), "
                f"got {self.num_clients}"
            )

        self._init_global_model()
        self._history = []

        for round_num in range(1, self._config.rounds + 1):
            metrics = self._run_round(round_num)
            self._history.append(metrics)

            logger.info(
                "Round %d/%d: clients=%s, avg_loss=%.4f",
                round_num, self._config.rounds,
                metrics.participating_clients, metrics.avg_loss,
            )

        return self._history

    def get_global_model(self) -> nn.Module:
        """Return the aggregated global model for inference."""
        if self._global_model is None:
            raise RuntimeError("Model not trained yet. Call train() first.")
        return self._global_model

    # ── Private methods ─────────────────────────────────

    def _init_global_model(self) -> None:
        """Initialize the global model from sentence-transformers."""
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for federated training. "
                "Install with: pip install sentence-transformers"
            )

        st_model = SentenceTransformer(self._model_name)
        # Get the underlying nn.Module
        self._global_model = st_model[0].auto_model  # type: ignore[assignment]

        # Initialize each client with a copy of the global model
        for state in self._clients.values():
            state.local_model = copy.deepcopy(self._global_model)

    def _run_round(self, round_num: int) -> RoundMetrics:
        """Execute one federated training round."""
        # Select clients
        num_selected = max(
            self._config.min_clients,
            int(self.num_clients * self._config.fraction_clients),
        )
        selected_ids = random.sample(
            list(self._clients.keys()), min(num_selected, self.num_clients)
        )

        # Local training on each selected client
        client_updates: List[Tuple[OrderedDict, int]] = []
        client_metrics: Dict[str, Dict[str, float]] = {}

        for cid in selected_ids:
            state = self._clients[cid]
            loss = self._train_local(state)
            client_metrics[cid] = {"loss": round(loss, 4)}

            # Collect weight update
            weights = self._get_weights(state.local_model)
            client_updates.append((weights, state.num_samples))

        # Aggregate (FedAvg)
        self._aggregate(client_updates)

        # Distribute global model back to all clients
        global_weights = self._get_weights(self._global_model)
        for state in self._clients.values():
            self._set_weights(state.local_model, global_weights)

        avg_loss = np.mean([m["loss"] for m in client_metrics.values()])

        return RoundMetrics(
            round_number=round_num,
            participating_clients=selected_ids,
            avg_loss=float(avg_loss),
            client_metrics=client_metrics,
        )

    def _train_local(self, state: ClientState) -> float:
        """
        Train on a single client's local data.

        Uses contrastive learning (MultipleNegativesRankingLoss style):
        for each document, positive pairs are formed with overlapping chunks.
        """
        model = state.local_model
        assert model is not None

        model.train()
        optimizer = optim.AdamW(
            model.parameters(), lr=self._config.learning_rate
        )

        # Build training pairs from documents
        pairs = self._build_training_pairs(state.documents)
        if not pairs:
            return 0.0

        # Create dataset
        anchor_texts, positive_texts = zip(*pairs)
        dataset = self._texts_to_dataset(anchor_texts, positive_texts)
        loader = DataLoader(dataset, batch_size=self._config.train_batch_size, shuffle=True)

        total_loss = 0.0
        num_batches = 0

        for _ in range(self._config.local_epochs):
            for batch in loader:
                anchor_ids, positive_ids = batch
                anchor_ids = anchor_ids.squeeze(1)
                positive_ids = positive_ids.squeeze(1)

                # Forward pass
                anchor_emb = model(input_ids=anchor_ids).last_hidden_state[:, 0]
                positive_emb = model(input_ids=positive_ids).last_hidden_state[:, 0]

                # Normalize for cosine similarity
                anchor_emb = torch.nn.functional.normalize(anchor_emb, dim=1)
                positive_emb = torch.nn.functional.normalize(positive_emb, dim=1)

                # InfoNCE loss (contrastive)
                logits = torch.matmul(anchor_emb, positive_emb.T) / self._config.contrastive_temperature
                labels = torch.arange(logits.size(0), device=logits.device)
                loss = nn.functional.cross_entropy(logits, labels)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

        model.eval()
        return total_loss / max(num_batches, 1)

    def _build_training_pairs(
        self, documents: List[Document]
    ) -> List[Tuple[str, str]]:
        """
        Build (anchor, positive) pairs from documents.

        Strategy: consecutive chunks from the same source are positive pairs.
        """
        pairs: List[Tuple[str, str]] = []

        # Group by source
        by_source: Dict[str, List[Document]] = {}
        for doc in documents:
            by_source.setdefault(doc.source, []).append(doc)

        for source, docs in by_source.items():
            docs_sorted = sorted(docs, key=lambda d: d.chunk_index)
            for i in range(len(docs_sorted) - 1):
                pairs.append((
                    docs_sorted[i].content,
                    docs_sorted[i + 1].content,
                ))

        return pairs

    def _texts_to_dataset(
        self, anchor_texts: Tuple[str, ...], positive_texts: Tuple[str, ...]
    ) -> TensorDataset:
        """Tokenize text pairs and create a PyTorch dataset."""
        try:
            from transformers import AutoTokenizer
        except ImportError:
            raise ImportError(
                "transformers is required. Install with: pip install transformers"
            )

        tokenizer = AutoTokenizer.from_pretrained(self._model_name)

        anchor_ids = tokenizer(
            list(anchor_texts), padding=True, truncation=True,
            max_length=self._config.train_max_seq_length, return_tensors="pt",
        )["input_ids"]
        positive_ids = tokenizer(
            list(positive_texts), padding=True, truncation=True,
            max_length=self._config.train_max_seq_length, return_tensors="pt",
        )["input_ids"]

        return TensorDataset(anchor_ids, positive_ids)

    @staticmethod
    def _get_weights(model: nn.Module) -> OrderedDict:
        """Extract model weights as an OrderedDict."""
        return OrderedDict(
            (name, param.data.clone()) for name, param in model.named_parameters()
        )

    @staticmethod
    def _set_weights(model: nn.Module, weights: OrderedDict) -> None:
        """Load weights into a model."""
        model.load_state_dict(weights, strict=True)

    def _aggregate(
        self, updates: List[Tuple[OrderedDict, int]]
    ) -> None:
        """
        FedAvg aggregation: weighted average of client weight updates.

        Args:
            updates: List of (weights, num_samples) tuples.
        """
        assert self._global_model is not None

        total_samples = sum(n for _, n in updates)
        if total_samples == 0:
            return

        aggregated = OrderedDict()
        for key in self._global_model.state_dict():
            weighted_sum = torch.zeros_like(
                self._global_model.state_dict()[key], dtype=torch.float32
            )
            for weights, n_samples in updates:
                weighted_sum += weights[key].float() * n_samples
            aggregated[key] = (weighted_sum / total_samples).to(
                self._global_model.state_dict()[key].dtype
            )

        self._global_model.load_state_dict(aggregated)
