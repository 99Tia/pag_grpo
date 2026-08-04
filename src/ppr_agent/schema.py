from __future__ import annotations
import json
from dataclasses import asdict, dataclass, field
from hashlib import md5
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

def compute_mdhash_id(content: str, prefix: str = "") -> str:
    return prefix + md5(content.encode("utf-8")).hexdigest()

def normalize_text(text: str) -> str:
    return " ".join(str(text).strip().lower().split())

def triple_to_text(subject: str, predicate: str, object_: str) -> str:
    return f"{normalize_text(subject)} {normalize_text(predicate)} {normalize_text(object_)}"


# OpenIE / corpus schemas
@dataclass
class Passage:
    passage_id: str
    text: str
    title: Optional[str] = None
    source_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_text(
        cls,
        text: str,
        title: Optional[str] = None,
        source_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        passage_id: Optional[str] = None,
    ) -> "Passage":
        text = str(text)
        pid = passage_id or compute_mdhash_id(text, prefix="chunk-")
        return cls(
            passage_id=pid,
            text=text,
            title=title,
            source_id=source_id,
            metadata=metadata or {},
        )


@dataclass(frozen=True)
class Triple:
    subject: str
    predicate: str
    object: str
    source_passage_id: Optional[str] = None
    triple_id: Optional[str] = None
    score: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def normalized(self) -> Tuple[str, str, str]:
        return (
            normalize_text(self.subject),
            normalize_text(self.predicate),
            normalize_text(self.object),
        )

    def text(self) -> str:
        s, p, o = self.normalized()
        return triple_to_text(s, p, o)

    def id(self) -> str:
        return self.triple_id or compute_mdhash_id(str(self.normalized()), prefix="fact-")


@dataclass
class OpenIEDoc:
    passage_id: str
    passage: str
    extracted_entities: List[str] = field(default_factory=list)
    extracted_triples: List[Triple] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


# Graph schemas
NodeType = Literal["entity", "passage"]
EdgeType = Literal["relation", "contains", "mentioned_in", "synonym"]


@dataclass
class GraphNode:
    node_id: str
    node_type: NodeType
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphEdge:
    src: str
    dst: str
    edge_type: EdgeType
    weight: float = 1.0
    relation: Optional[str] = None
    source_passage_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphBuildResult:
    graph_path: str
    num_entity_nodes: int
    num_passage_nodes: int
    num_relation_edges: int
    num_passage_edges: int
    metadata: Dict[str, Any] = field(default_factory=dict)


# Triple indexing / filtering schemas
@dataclass
class CandidateTriple:
    triple: Triple
    index: int
    score: float
    source_passage_id: Optional[str] = None


@dataclass
class FilteredTriple:
    triple: Triple
    original_index: Optional[int] = None
    original_score: Optional[float] = None
    filter_rank: Optional[int] = None
    filter_confidence: Optional[float] = None


@dataclass
class TripleFilterResult:
    query: str
    candidate_triples: List[CandidateTriple]
    filtered_triples: List[FilteredTriple]
    raw_response: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# SearchGraph / PPR schemas
@dataclass
class SearchGraphRequest:
    question: str
    search_focus: Optional[str] = None
    seed_entities: List[str] = field(default_factory=list)
    relation_hints: List[str] = field(default_factory=list)
    evidence_so_far: List[str] = field(default_factory=list)
    top_k_triples: int = 50
    top_k_passages: int = 5
    step_id: int = 0

    def build_search_query(self) -> str:
        parts = [f"Question: {self.question}"]

        if self.search_focus:
            parts.append(f"Search focus: {self.search_focus}")

        if self.seed_entities:
            parts.append("Seed entities: " + ", ".join(self.seed_entities))

        if self.relation_hints:
            parts.append("Relation hints: " + ", ".join(self.relation_hints))

        if self.evidence_so_far:
            parts.append("Evidence so far:\n" + "\n".join(self.evidence_so_far))

        return "\n".join(parts)


@dataclass
class RetrievedPassage:
    passage_id: str
    text: str
    score: float
    rank: int
    title: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PPRSeedInfo:
    entity_seeds: Dict[str, float] = field(default_factory=dict)
    passage_seeds: Dict[str, float] = field(default_factory=dict)
    selected_triples: List[Triple] = field(default_factory=list)
    dense_passage_ids: List[str] = field(default_factory=list)


@dataclass
class SearchGraphResult:
    request: SearchGraphRequest
    passages: List[RetrievedPassage]
    candidate_triples: List[CandidateTriple] = field(default_factory=list)
    filtered_triples: List[FilteredTriple] = field(default_factory=list)
    seed_info: Optional[PPRSeedInfo] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# Agent action / trajectory schemas
AgentActionType = Literal["SearchGraph", "SubmitFinalAnswer"]


@dataclass
class AgentAction:
    action: AgentActionType
    search_request: Optional[SearchGraphRequest] = None
    answer: Optional[str] = None
    reasoning_summary: Optional[str] = None
    raw_output: Optional[str] = None

    def validate(self) -> None:
        if self.action == "SearchGraph" and self.search_request is None:
            raise ValueError("SearchGraph action requires search_request.")
        if self.action == "SubmitFinalAnswer" and self.answer is None:
            raise ValueError("SubmitFinalAnswer action requires answer.")


@dataclass
class AgentObservation:
    step_id: int
    search_result: Optional[SearchGraphResult] = None
    message: Optional[str] = None
    evidence_memory: List[RetrievedPassage] = field(default_factory=list)


@dataclass
class AgentStep:
    step_id: int
    action: AgentAction
    observation: Optional[AgentObservation] = None


@dataclass
class AgentTrajectory:
    question_id: str
    question: str
    steps: List[AgentStep] = field(default_factory=list)
    final_answer: Optional[str] = None
    gold_answers: List[str] = field(default_factory=list)
    gold_passage_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def retrieved_passage_ids(self) -> List[str]:
        ids: List[str] = []
        seen = set()

        for step in self.steps:
            obs = step.observation
            if obs is None or obs.search_result is None:
                continue
            for passage in obs.search_result.passages:
                if passage.passage_id not in seen:
                    ids.append(passage.passage_id)
                    seen.add(passage.passage_id)

        return ids

    def num_search_calls(self) -> int:
        return sum(1 for step in self.steps if step.action.action == "SearchGraph")

    def stopped(self) -> bool:
        return any(step.action.action == "SubmitFinalAnswer" for step in self.steps)

@dataclass
class RewardBreakdown:
    total_reward: float
    answer_reward: float = 0.0
    support_recall_reward: float = 0.0
    full_support_reward: float = 0.0
    groundedness_reward: float = 0.0
    step_penalty: float = 0.0
    token_penalty: float = 0.0
    noise_penalty: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationResult:
    question_id: str
    question: str
    predicted_answer: Optional[str]
    gold_answers: List[str]
    retrieved_passage_ids: List[str]
    gold_passage_ids: List[str]
    recall_at_k: Dict[int, float] = field(default_factory=dict)
    precision_at_k: Dict[int, float] = field(default_factory=dict)
    full_recall_at_k: Dict[int, float] = field(default_factory=dict)
    exact_match: Optional[float] = None
    f1: Optional[float] = None
    num_search_calls: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

def dataclass_to_dict(obj: Any) -> Dict[str, Any]:
    return asdict(obj)


def save_jsonl(path: str, rows: Sequence[Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            if hasattr(row, "__dataclass_fields__"):
                row = asdict(row)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
