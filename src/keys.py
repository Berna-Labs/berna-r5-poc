"""Precise keys binding base model to each specialist."""

import hashlib
import json
from typing import Optional

from .config import BernaFatalError
from .zero_error import assert_range
from .incorporation import hash_text


# ============================================================
# Deterministic keys (no external crypto libs required)
# ============================================================
def domain_key(domain: str, version: str, param_hash: str) -> str:
    """Deterministic domain key: SHA256 of three components."""
    if not domain or not version or not param_hash:
        raise BernaFatalError("domain/version/param_hash must be non-empty")
    payload = f"{domain}|{version}|{param_hash}"
    return hashlib.sha256(payload.encode()).hexdigest()


def capability_fingerprint(capability_vec: list,
                           precision: int = 6) -> str:
    """Canonical fingerprint of capability vector."""
    if not capability_vec:
        raise BernaFatalError("capability_vec empty")
    rounded = [round(float(x), precision) for x in capability_vec]
    payload = json.dumps(rounded, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


# ============================================================
# Signatures (deterministic, no external libs)
# ============================================================
def sign(answer: str, domain_key_hex: str) -> str:
    """Deterministic signature: SHA256(answer + domain_key)."""
    if not answer or not domain_key_hex:
        raise BernaFatalError("answer and domain_key required")
    payload = f"{answer}|{domain_key_hex}"
    return hashlib.sha256(payload.encode()).hexdigest()


def verify(answer: str, signature: str, domain_key_hex: str) -> bool:
    """Verify signature matches. Returns True if valid."""
    expected = sign(answer, domain_key_hex)
    return expected == signature


# ============================================================
# Specialist key record
# ============================================================
class SpecialistKey:
    """In-memory representation of a specialist's key."""

    def __init__(
        self,
        specialist_id: str,
        domain: str,
        version: str,
        param_hash: str,
        capability_vec: list,
    ):
        self.specialist_id = specialist_id
        self.domain = domain
        self.version = version
        self.param_hash = param_hash
        self.capability_vec = capability_vec
        self.domain_key_hex = domain_key(domain, version, param_hash)
        self.cap_fingerprint = capability_fingerprint(capability_vec)

    def sign_answer(self, answer: str) -> str:
        return sign(answer, self.domain_key_hex)

    def verify_answer(self, answer: str, signature: str) -> bool:
        return verify(answer, signature, self.domain_key_hex)

    def matches_key(self, other_domain_key: str) -> bool:
        return self.domain_key_hex == other_domain_key

    def info(self) -> dict:
        return {
            "specialist_id": self.specialist_id,
            "domain": self.domain,
            "version": self.version,
            "domain_key": self.domain_key_hex[:16] + "...",
            "cap_fingerprint": self.cap_fingerprint[:16] + "...",
        }


# ============================================================
# Registry-side: exact lookup (no similarity)
# ============================================================
class KeyRegistry:
    """Exact-match registry of specialist keys. No guessing."""

    def __init__(self):
        self.by_domain_key = {}    # domain_key -> SpecialistKey
        self.by_specialist = {}    # specialist_id -> SpecialistKey
        self.by_domain = {}        # domain -> [SpecialistKey]

    def add(self, key: SpecialistKey) -> None:
        if key.domain_key_hex in self.by_domain_key:
            raise BernaFatalError(
                f"Duplicate domain_key: {key.domain_key_hex[:16]}"
            )
        if key.specialist_id in self.by_specialist:
            raise BernaFatalError(
                f"Duplicate specialist_id: {key.specialist_id}"
            )
        self.by_domain_key[key.domain_key_hex] = key
        self.by_specialist[key.specialist_id] = key
        self.by_domain.setdefault(key.domain, []).append(key)

    def lookup_exact(self, domain_key_hex: str) -> Optional[SpecialistKey]:
        """Exact lookup. Returns None if not found. No approximation."""
        return self.by_domain_key.get(domain_key_hex)

    def lookup_by_specialist(self, sid: str) -> Optional[SpecialistKey]:
        return self.by_specialist.get(sid)

    def lookup_by_domain(self, domain: str) -> list:
        return list(self.by_domain.get(domain, []))

    def verify_answer(
        self, specialist_id: str, answer: str, signature: str
    ) -> bool:
        key = self.by_specialist.get(specialist_id)
        if key is None:
            return False
        return key.verify_answer(answer, signature)

    def stats(self) -> dict:
        return {
            "total_keys": len(self.by_domain_key),
            "domains": list(self.by_domain.keys()),
            "counts": {d: len(v) for d, v in self.by_domain.items()},
        }
