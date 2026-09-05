from datetime import UTC, datetime, timedelta
from uuid import uuid4

from jose import JWTError, jwt
from pwdlib import PasswordHash
from pwdlib.exceptions import UnknownHashError

from app.core.config import settings

password_hash = PasswordHash.recommended()
_DUMMY_PASSWORD_HASH = password_hash.hash("not-a-real-user-password")


def hash_password(password: str) -> str:
    return password_hash.hash(password)


def verify_password(password: str, encoded_hash: str) -> bool:
    try:
        return password_hash.verify(password, encoded_hash)
    except (TypeError, ValueError, UnknownHashError):
        return False


def verify_dummy_password(password: str) -> None:
    """Equalize most of the password-hashing work when an account is absent."""
    verify_password(password, _DUMMY_PASSWORD_HASH)


def create_access_token(subject: str) -> str:
    issued_at = datetime.now(UTC)
    expires = issued_at + timedelta(minutes=settings.jwt_expire_minutes)
    claims = {
        "sub": subject,
        "type": "access",
        "iat": issued_at,
        "exp": expires,
        "iss": settings.jwt_issuer,
        "aud": settings.jwt_audience,
        "jti": uuid4().hex,
    }
    return jwt.encode(claims, settings.jwt_secret, algorithm="HS256")


def decode_subject(token: str) -> str:
    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret,
            algorithms=["HS256"],
            audience=settings.jwt_audience,
            issuer=settings.jwt_issuer,
        )
        subject = payload.get("sub")
        if not isinstance(subject, str) or not subject or payload.get("type") != "access":
            raise ValueError("Missing token subject")
        return subject
    except (JWTError, ValueError) as error:
        raise ValueError("Invalid authentication token") from error
