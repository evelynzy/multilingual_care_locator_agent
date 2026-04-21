import json
import os
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

from datasets import Dataset, load_dataset
from llama_index.core import Document, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from config_loader import get_embed_model_name, get_search_settings

logger = logging.getLogger(__name__)


@dataclass
class ProviderRecord:
    """Represents a single healthcare provider entry."""

    id: str
    name: str
    specialties: List[str]
    languages: List[str]
    accepted_insurance: List[str]
    address: str
    city: str
    state: str
    country: str
    phone: Optional[str]
    website: Optional[str]
    telehealth: Optional[bool]
    description: Optional[str]

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "specialties": self.specialties,
            "languages": self.languages,
            "insurance_reported": self.accepted_insurance,
            "insurance_network_verification": {
                "status": "unverified",
                "verified": False,
                "basis": "Insurance/network participation is not confirmed by source data.",
            },
            "accepting_new_patients_status": {
                "status": "unknown",
                "verified": False,
                "basis": "Source data does not confirm new-patient availability.",
            },
            "address": self.address,
            "city": self.city,
            "state": self.state,
            "country": self.country,
            "phone": self.phone,
            "website": self.website,
            "telehealth": self.telehealth,
            "description": self.description,
            "provenance": {
                "source": "Local provider dataset",
            },
        }

    def as_document(self) -> Document:
        """Convert provider metadata into a rich LlamaIndex document."""

        summary_chunks = [
            f"Name: {self.name}",
            f"Location: {', '.join(chunk for chunk in [self.address, self.city, self.state, self.country] if chunk)}",
            f"Specialties: {', '.join(self.specialties) or 'Unknown'}",
            f"Languages: {', '.join(self.languages) or 'Unknown'}",
            f"Listed insurance (reported, not verified): {', '.join(self.accepted_insurance) or 'Unknown'}",
            f"Telehealth: {'Yes' if self.telehealth else 'No'}" if self.telehealth is not None else "",
            f"Description: {self.description}" if self.description else "",
        ]

        text = "\n".join(filter(None, summary_chunks))

        metadata = {
            "provider_id": self.id,
            "name": self.name,
            "city": self.city,
            "state": self.state,
            "country": self.country,
            "languages": self.languages,
            "specialties": self.specialties,
            "insurance_reported": self.accepted_insurance,
        }

        return Document(text=text, metadata=metadata)


@dataclass
class SearchCriteria:
    """Normalized search inputs extracted from the LLM."""

    specialties: List[str]
    location: Optional[str]
    insurance: List[str]
    preferred_languages: List[str]
    keywords: List[str]


class ProviderRepository:
    """Loads healthcare provider data and builds a LlamaIndex vector index."""

    def __init__(
        self,
        dataset_id: Optional[str] = None,
        dataset_split: str = "train",
        local_data_path: Optional[Path] = None,
        embed_model_name: Optional[str] = None,
        default_top_k: Optional[int] = None,
    ) -> None:
        # self.dataset_id = dataset_id or os.getenv("HF_DATASET_ID")
        self.dataset_id = None
        self.dataset_split = dataset_split
        self.local_data_path = (
            local_data_path or Path(__file__).parent / "data" / "providers.json"
        )
        search_settings = get_search_settings()
        self.embed_model_name = embed_model_name or get_embed_model_name()
        self.default_top_k = default_top_k or search_settings.get("default_top_k", 5)

        self.load_error: Optional[str] = None
        self.providers: List[ProviderRecord] = []
        self._providers_by_id: dict[str, ProviderRecord] = {}
        self.index: Optional[VectorStoreIndex] = None

        self._load_records()
        self._build_index()

    # ------------------------------------------------------------------
    def _load_records(self) -> None:
        dataset: Optional[Dataset] = None

        if self.dataset_id:
            try:
                dataset = load_dataset(
                    path=self.dataset_id,
                    split=self.dataset_split,
                )
            except Exception as exc:  # noqa: BLE001 - record and fall back
                self.load_error = f"Failed to load dataset '{self.dataset_id}': {exc}"

        if dataset is None:
            self.providers = self._load_local_data()
        else:
            self.providers = [self._normalize_row(row) for row in dataset]

        self._providers_by_id = {provider.id: provider for provider in self.providers}

    # ------------------------------------------------------------------
    def _build_index(self) -> None:
        if not self.providers:
            return

        documents = [provider.as_document() for provider in self.providers]
        logger.info("Building vector index with %s provider records", len(documents))

        embed_model = HuggingFaceEmbedding(model_name=self.embed_model_name)
        self.index = VectorStoreIndex.from_documents(
            documents,
            embed_model=embed_model,
        )

    # ------------------------------------------------------------------
    def _load_local_data(self) -> List[ProviderRecord]:
        if not self.local_data_path.exists():
            raise FileNotFoundError(
                f"Local provider data not found at {self.local_data_path}."
            )

        with self.local_data_path.open("r", encoding="utf-8") as handle:
            raw_records = json.load(handle)

        return [self._normalize_row(record) for record in raw_records]

    # ------------------------------------------------------------------
    def _normalize_row(self, row: dict) -> ProviderRecord:
        def _ensure_list(value: Optional[Iterable[str]]) -> List[str]:
            if value is None:
                return []
            if isinstance(value, str):
                return [value]
            return [str(item) for item in value]

        return ProviderRecord(
            id=str(row.get("id", "")),
            name=str(row.get("name", "Unknown Provider")),
            specialties=_ensure_list(row.get("specialties")),
            languages=_ensure_list(row.get("languages")),
            accepted_insurance=_ensure_list(
                row.get("insurance_reported", row.get("accepted_insurance"))
            ),
            address=str(row.get("address", "")),
            city=str(row.get("city", "")),
            state=str(row.get("state", "")),
            country=str(row.get("country", "")),
            phone=row.get("phone"),
            website=row.get("website"),
            telehealth=row.get("telehealth"),
            description=row.get("description"),
        )

    # ------------------------------------------------------------------
    def search(self, criteria: SearchCriteria, limit: int = 5) -> List[dict]:
        if not self.index:
            return []

        query = self._compose_query(criteria)
        if not query.strip():
            return []

        logger.debug("Semantic query prepared. length=%s", len(query))
        retriever = self.index.as_retriever(similarity_top_k=max(limit, self.default_top_k))
        nodes = retriever.retrieve(query)
        logger.info("Retriever returned %s nodes", len(nodes))

        results: List[dict] = []

        for node in nodes[:limit]:
            provider_id = node.metadata.get("provider_id") if node.metadata else None
            provider = self._providers_by_id.get(str(provider_id)) if provider_id else None
            if not provider:
                continue

            metadata = provider.to_dict()
            metadata["score"] = round(float(node.score or 0.0), 4)
            metadata["location"] = ", ".join(
                chunk
                for chunk in [provider.city, provider.state, provider.country]
                if chunk
            )
            metadata["source"] = "Local provider dataset"
            metadata["retriever_metadata"] = {
                "similarity": float(node.score or 0.0),
                "node_id": node.node_id,
            }
            results.append(metadata)

        return results

    # ------------------------------------------------------------------
    def _compose_query(self, criteria: SearchCriteria) -> str:
        parts: List[str] = []

        if criteria.specialties:
            parts.append(
                "Looking for specialists in "
                + ", ".join(criteria.specialties)
            )

        if criteria.location:
            parts.append(f"Located around {criteria.location}")

        if criteria.insurance:
            parts.append(
                "Accepting insurance plans: " + ", ".join(criteria.insurance)
            )

        if criteria.preferred_languages:
            parts.append(
                "Must support languages: " + ", ".join(criteria.preferred_languages)
            )

        if criteria.keywords:
            parts.append(
                "Additional needs: " + ", ".join(criteria.keywords)
            )

        return "; ".join(parts)


__all__ = [
    "ProviderRepository",
    "SearchCriteria",
    "ProviderRecord",
]
