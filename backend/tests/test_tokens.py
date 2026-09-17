from datetime import UTC, datetime, timedelta

import jwt
import pytest

from app.core.config import settings
from app.security import create_access_token, decode_subject


def test_access_token_round_trip():
    assert decode_subject(create_access_token("42")) == "42"


@pytest.mark.parametrize(
    "change",
    [
        {"exp": 0},
        {"aud": "another-app"},
        {"iss": "another-issuer"},
        {"sub": 42},
        {"sub": ""},
        {"type": "refresh"},
        {"iat": datetime.now(UTC) + timedelta(hours=1)},
    ],
)
def test_rejects_invalid_claims(change):
    claims = jwt.decode(create_access_token("42"), options={"verify_signature": False})
    claims.update(change)
    token = jwt.encode(claims, settings.jwt_secret, algorithm="HS256")
    with pytest.raises(ValueError, match="Invalid authentication token"):
        decode_subject(token)


@pytest.mark.parametrize("claim", ["sub", "exp", "iat", "iss", "aud", "jti"])
def test_rejects_missing_required_claim(claim):
    claims = jwt.decode(create_access_token("42"), options={"verify_signature": False})
    del claims[claim]
    with pytest.raises(ValueError, match="Invalid authentication token"):
        decode_subject(jwt.encode(claims, settings.jwt_secret, algorithm="HS256"))


@pytest.mark.parametrize("algorithm,key", [("HS384", None), ("HS256", "wrong-key" * 8), ("none", "")])
def test_rejects_wrong_algorithm_or_signature(algorithm, key):
    claims = jwt.decode(create_access_token("42"), options={"verify_signature": False})
    token = jwt.encode(claims, settings.jwt_secret if key is None else key, algorithm=algorithm)
    with pytest.raises(ValueError, match="Invalid authentication token"):
        decode_subject(token)
