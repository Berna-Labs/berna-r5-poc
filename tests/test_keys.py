"""Tests for keys module (Theory 2)."""

import pytest

from src.config import BernaFatalError
from src.keys import (
    domain_key, capability_fingerprint, sign, verify,
    SpecialistKey, KeyRegistry,
)


class TestDomainKey:
    def test_deterministic(self):
        k1 = domain_key("math", "v1.0", "abc123")
        k2 = domain_key("math", "v1.0", "abc123")
        assert k1 == k2

    def test_different_domain(self):
        k1 = domain_key("math", "v1.0", "abc")
        k2 = domain_key("code", "v1.0", "abc")
        assert k1 != k2

    def test_different_param_hash(self):
        k1 = domain_key("math", "v1.0", "abc")
        k2 = domain_key("math", "v1.0", "xyz")
        assert k1 != k2

    def test_empty_raises(self):
        with pytest.raises(BernaFatalError):
            domain_key("", "v1", "abc")
        with pytest.raises(BernaFatalError):
            domain_key("math", "", "abc")
        with pytest.raises(BernaFatalError):
            domain_key("math", "v1", "")

    def test_length(self):
        k = domain_key("math", "v1.0", "abc")
        assert len(k) == 64


class TestCapabilityFingerprint:
    def test_deterministic(self):
        f1 = capability_fingerprint([0.1, 0.2, 0.3])
        f2 = capability_fingerprint([0.1, 0.2, 0.3])
        assert f1 == f2

    def test_precision_invariance(self):
        f1 = capability_fingerprint([0.123456, 0.2])
        f2 = capability_fingerprint([0.1234561, 0.2])
        assert f1 == f2  # rounded to 6 decimals

    def test_different_raises(self):
        f1 = capability_fingerprint([0.1, 0.2])
        f2 = capability_fingerprint([0.9, 0.2])
        assert f1 != f2

    def test_empty_raises(self):
        with pytest.raises(BernaFatalError):
            capability_fingerprint([])


class TestSignVerify:
    def test_sign_deterministic(self):
        s1 = sign("answer", "key123")
        s2 = sign("answer", "key123")
        assert s1 == s2

    def test_verify_ok(self):
        s = sign("answer", "key123")
        assert verify("answer", s, "key123") is True

    def test_verify_wrong_answer(self):
        s = sign("answer", "key123")
        assert verify("other", s, "key123") is False

    def test_verify_wrong_key(self):
        s = sign("answer", "key123")
        assert verify("answer", s, "wrongkey") is False

    def test_empty_raises(self):
        with pytest.raises(BernaFatalError):
            sign("", "key123")
        with pytest.raises(BernaFatalError):
            sign("answer", "")


class TestSpecialistKey:
    def test_creation(self):
        sk = SpecialistKey(
            specialist_id="sid-1",
            domain="math",
            version="v0.1",
            param_hash="abc",
            capability_vec=[0.5, 0.5],
        )
        assert sk.specialist_id == "sid-1"
        assert sk.domain_key_hex
        assert sk.cap_fingerprint

    def test_sign_answer(self):
        sk = SpecialistKey("s", "m", "v1", "p", [0.5])
        sig = sk.sign_answer("hello")
        assert sk.verify_answer("hello", sig) is True

    def test_verify_wrong(self):
        sk = SpecialistKey("s", "m", "v1", "p", [0.5])
        sig = sk.sign_answer("hello")
        assert sk.verify_answer("world", sig) is False

    def test_info(self):
        sk = SpecialistKey("s", "m", "v1", "p", [0.5])
        info = sk.info()
        assert info["specialist_id"] == "s"
        assert info["domain"] == "m"


class TestKeyRegistry:
    def test_add_lookup(self):
        reg = KeyRegistry()
        sk = SpecialistKey("s1", "math", "v1", "p1", [0.5])
        reg.add(sk)
        assert reg.lookup_exact(sk.domain_key_hex) is sk
        assert reg.lookup_by_specialist("s1") is sk

    def test_lookup_missing_returns_none(self):
        reg = KeyRegistry()
        assert reg.lookup_exact("0" * 64) is None
        assert reg.lookup_by_specialist("none") is None

    def test_duplicate_domain_key_raises(self):
        reg = KeyRegistry()
        sk = SpecialistKey("s1", "math", "v1", "p", [0.5])
        reg.add(sk)
        with pytest.raises(BernaFatalError):
            reg.add(sk)  # same domain_key

    def test_duplicate_specialist_raises(self):
        reg = KeyRegistry()
        reg.add(SpecialistKey("s1", "math", "v1", "p1", [0.5]))
        # Same specialist_id but different everything else
        with pytest.raises(BernaFatalError):
            reg.add(SpecialistKey("s1", "code", "v1", "p2", [0.3]))

    def test_lookup_by_domain(self):
        reg = KeyRegistry()
        reg.add(SpecialistKey("s1", "math", "v1", "p1", [0.5]))
        reg.add(SpecialistKey("s2", "math", "v2", "p2", [0.5]))
        reg.add(SpecialistKey("s3", "code", "v1", "p3", [0.5]))
        assert len(reg.lookup_by_domain("math")) == 2
        assert len(reg.lookup_by_domain("code")) == 1
        assert len(reg.lookup_by_domain("unknown")) == 0

    def test_verify_answer_via_registry(self):
        reg = KeyRegistry()
        sk = SpecialistKey("s1", "math", "v1", "p1", [0.5])
        reg.add(sk)
        sig = sk.sign_answer("hello")
        assert reg.verify_answer("s1", "hello", sig) is True
        assert reg.verify_answer("s1", "world", sig) is False
        assert reg.verify_answer("missing", "hello", sig) is False

    def test_stats(self):
        reg = KeyRegistry()
        reg.add(SpecialistKey("s1", "math", "v1", "p1", [0.5]))
        reg.add(SpecialistKey("s2", "code", "v1", "p2", [0.5]))
        s = reg.stats()
        assert s["total_keys"] == 2
        assert "math" in s["domains"]
        assert "code" in s["domains"]
