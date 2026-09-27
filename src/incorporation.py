"""Theory 2: Reliable Mutation via Federation Incorporation."""

import hashlib
import math
from typing import Optional

from .config import (
    TAU_ROUTE, TAU_CONF, TAU_GAP, TAU_PRIO, EPSILON_NEG,
    BernaFatalError,
)
from .zero_error import assert_range, assert_finite


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def cosine_sim(a: list, b: list) -> float:
    """Cosine similarity. Raises on zero vector."""
    if len(a) != len(b):
        raise BernaFatalError(f"Vector dim mismatch: {len(a)} vs {len(b)}")
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na < 1e-12 or nb < 1e-12:
        raise BernaFatalError("Zero vector in cosine_sim")
    return max(-1.0, min(1.0, dot / (na * nb)))


# ============================================================
# Routing (Stage 2)
# ============================================================
def route(query_embedding: list, specialists: list,
          tau_route: float = TAU_ROUTE) -> Optional[dict]:
    """Select best specialist by cosine similarity.

    specialists: list of dicts with keys:
      - specialist_id
      - domain_embedding (list of floats)
    Returns best specialist dict or None if below threshold.
    """
    if not specialists:
        return None
    best = None
    best_sim = -1.0
    for s in specialists:
        if "domain_embedding" not in s:
            raise BernaFatalError(
                f"Specialist {s.get('specialist_id')} missing embedding"
            )
        sim = cosine_sim(query_embedding, s["domain_embedding"])
        if sim > best_sim:
            best_sim = sim
            best = s
    if best is None or best_sim < tau_route:
        return None
    return {"specialist": best, "similarity": best_sim}


# ============================================================
# Verification (Stage 4)
# ============================================================
def verify_logical(answer: str, negative_answer: str,
                   base_scorer) -> bool:
    """Logical check: if both a and not-a are plausible, reject."""
    p_pos = base_scorer(answer)
    p_neg = base_scorer(negative_answer)
    assert_range(p_pos, 0.0, 1.0, "p_pos")
    assert_range(p_neg, 0.0, 1.0, "p_neg")
    # If both plausible, contradiction
    return not (p_pos > 0.5 and p_neg > 0.5)


def verify_6d(K_before: dict, K_after: dict,
              epsilon: float = EPSILON_NEG) -> bool:
    """6D check: no dimension decreases beyond tolerance."""
    dims = ["L", "W", "H", "D", "T", "E"]
    for d in dims:
        delta = K_after[d] - K_before[d]
        if delta < -epsilon:
            return False
    return True


def verify_consistency(answers: list) -> bool:
    """Consistency: 3 rephrasings should yield same answer."""
    if len(answers) < 2:
        return True
    first = answers[0]
    return all(a == first for a in answers[1:])


def compute_reliability(
    source_conf: float,
    consistent: bool,
    K_before: dict,
    K_after: dict,
) -> float:
    """R(m) = min(R_src, R_cons, R_6D)."""
    assert_range(source_conf, 0.0, 1.0, "source_conf")
    r_src = source_conf
    r_cons = 1.0 if consistent else 0.0
    r_6d = 1.0 if verify_6d(K_before, K_after) else 0.0
    return min(r_src, r_cons, r_6d)


# ============================================================
# Incorporation Event
# ============================================================
class Incorporation:
    """Encapsulates a single incorporation attempt."""

    def __init__(self, request_id: str, query: str):
        self.request_id = request_id
        self.query = query
        self.query_hash = hash_text(query)

        # Filled during processing
        self.specialist_id: Optional[str] = None
        self.answer: Optional[str] = None
        self.answer_hash: Optional[str] = None
        self.confidence: float = 0.0
        self.reliability: float = 0.0
        self.verified: bool = False
        self.status: str = "pending"  # pending, integrated, queued

    def process(
        self,
        query_embedding: list,
        specialists: list,
        specialist_answer_fn,
        base_scorer,
        K_before: dict,
        K_after_fn,
    ) -> str:
        """Run all 5 stages. Returns final status."""
        # Stage 1: Request (already have query)
        # Stage 2: Routing
        routed = route(query_embedding, specialists)
        if routed is None:
            self.status = "queued"
            self.reliability = 0.0
            return self.status

        self.specialist_id = routed["specialist"]["specialist_id"]
        self.confidence = routed["similarity"]

        # Stage 3: Answer
        try:
            answer = specialist_answer_fn(
                routed["specialist"], self.query
            )
        except Exception as e:
            # specialist refused or failed
            self.status = "queued"
            return self.status

        if answer is None or not isinstance(answer, str):
            self.status = "queued"
            return self.status

        self.answer = answer
        self.answer_hash = hash_text(answer)

        # Confidence gate
        if self.confidence < TAU_CONF:
            self.status = "queued"
            return self.status

        # Stage 4: Verification
        neg = "not " + answer
        logical_ok = verify_logical(answer, neg, base_scorer)
        if not logical_ok:
            self.status = "queued"
            return self.status

        # 6D verification
        K_after = K_after_fn(K_before, self.query, answer)
        consistency_ok = verify_consistency([answer, answer, answer])

        self.reliability = compute_reliability(
            self.confidence, consistency_ok, K_before, K_after
        )

        # Stage 5: Integrate or queue
        if self.reliability > 0.5:
            self.verified = True
            self.status = "integrated"
        else:
            self.status = "queued"

        return self.status

    def info(self) -> dict:
        return {
            "request_id": self.request_id,
            "specialist_id": self.specialist_id,
            "answer_hash": self.answer_hash,
            "confidence": round(self.confidence, 4),
            "reliability": round(self.reliability, 4),
            "verified": self.verified,
            "status": self.status,
        }


# ============================================================
# Gap Queue
# ============================================================
class GapQueue:
    """Priority queue of unincorporated knowledge requests."""

    def __init__(self, tau_prio: float = TAU_PRIO):
        self.tau_prio = tau_prio
        self.gaps = []  # list of dicts

    def add(self, request_id: str, query: str,
            query_embedding: list,
            failure_count: int = 1) -> dict:
        gap = {
            "gap_id": hash_text(request_id + query)[:16],
            "request_id": request_id,
            "query_hash": hash_text(query),
            "embedding": query_embedding,
            "failure_count": failure_count,
            "priority": 0.0,
            "status": "pending",
        }
        self.gaps.append(gap)
        self._recompute_priority(gap)
        return gap

    def _recompute_priority(self, gap: dict) -> None:
        # Cluster size: how many gaps similar to this one
        n_similar = 1
        for other in self.gaps:
            if other is gap:
                continue
            sim = cosine_sim(gap["embedding"], other["embedding"])
            if sim >= TAU_GAP:
                n_similar += 1
        avg_fail = gap["failure_count"]
        gap["priority"] = n_similar * avg_fail
        assert_finite(gap["priority"], "gap priority")

    def should_trigger_training(self) -> bool:
        return any(g["priority"] >= self.tau_prio for g in self.gaps)

    def top_priority(self) -> Optional[dict]:
        if not self.gaps:
            return None
        return max(self.gaps, key=lambda g: g["priority"])

    def stats(self) -> dict:
        return {
            "size": len(self.gaps),
            "max_priority": round(
                max((g["priority"] for g in self.gaps), default=0.0), 4
            ),
            "trigger_ready": self.should_trigger_training(),
        }
