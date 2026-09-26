import hashlib
import secrets
from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.audit import add_audit_event
from app.core.config import settings
from app.core.rate_limit import enforce_auth_rate_limit, rate_limiter
from app.database import get_db
from app.models import User
from app.password_recovery import send_recovery_email
from app.schemas import (
    ChangePasswordRequest,
    ForgotPasswordRequest,
    RegisterRequest,
    ResetPasswordRequest,
    TokenResponse,
    UserResponse,
)
from app.security import (
    create_access_token,
    decode_access_token,
    hash_password,
    verify_dummy_password,
    verify_password,
)

router = APIRouter(prefix="/api/v1/auth", tags=["authentication"])
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")


def current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> User:
    try:
        claims = decode_access_token(token)
        user_id = int(claims["sub"])
    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        ) from error
    user = db.get(User, user_id)
    if not user or claims.get("session_version", 0) != user.session_version:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


def require_admin(user: User = Depends(current_user)) -> User:
    if user.role != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Administrator role required")
    return user


def _registration_role(payload: RegisterRequest, db: Session) -> str:
    supplied_token = payload.admin_bootstrap_token
    if supplied_token is None:
        return "researcher"

    configured_token = settings.admin_bootstrap_token
    if configured_token is None or not secrets.compare_digest(
        supplied_token,
        configured_token.get_secret_value(),
    ):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid administrator bootstrap token")

    if db.scalar(select(User.id).where(User.role == "admin")) is not None:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="An administrator is already registered")
    return "admin"


@router.post("/register", response_model=UserResponse, status_code=201)
def register(
    payload: RegisterRequest,
    db: Session = Depends(get_db),
    _rate_limit: None = Depends(enforce_auth_rate_limit),
):
    if db.scalar(select(User).where(User.email == payload.email)):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Email is already registered")
    role = _registration_role(payload, db)
    user = User(
        email=payload.email,
        full_name=payload.full_name,
        password_hash=hash_password(payload.password),
        role=role,
    )
    db.add(user)
    try:
        db.flush()
        add_audit_event(
            db,
            user_id=user.id,
            action="user.registered",
            resource_type="user",
            resource_id=user.id,
            metadata={"role": role},
        )
        db.commit()
    except IntegrityError as error:
        db.rollback()
        if role == "admin":
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="An administrator is already registered",
            ) from error
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Email is already registered") from error
    db.refresh(user)
    return user


@router.post("/login", response_model=TokenResponse)
def login(
    form: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db),
    _rate_limit: None = Depends(enforce_auth_rate_limit),
):
    normalized_email = form.username.strip().lower()
    user = db.scalar(select(User).where(User.email == normalized_email)) if len(normalized_email) <= 320 else None
    if user is None:
        verify_dummy_password(form.password)
    if user is None or not verify_password(form.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    add_audit_event(
        db,
        user_id=user.id,
        action="auth.login_succeeded",
        resource_type="user",
        resource_id=user.id,
    )
    db.commit()
    return TokenResponse(access_token=create_access_token(str(user.id), user.session_version))


@router.get("/me", response_model=UserResponse)
def me(user: User = Depends(current_user)):
    return user


@router.post("/forgot-password", status_code=202)
def forgot_password(
    payload: ForgotPasswordRequest,
    background: BackgroundTasks,
    db: Session = Depends(get_db),
    _rate_limit: None = Depends(enforce_auth_rate_limit),
):
    if not settings.smtp_host or not settings.smtp_from:
        raise HTTPException(status_code=503, detail="Email recovery is not configured. Contact your administrator.")
    email = str(payload.email).strip().lower()
    key = hashlib.sha256(email.encode()).hexdigest()
    allowed, _ = rate_limiter.allow(f"recovery:{key}", limit=1, window_seconds=60)
    user = db.scalar(select(User).where(User.email == email)) if allowed else None
    if user:
        token = secrets.token_urlsafe(32)
        user.reset_token_hash = hashlib.sha256(token.encode()).hexdigest()
        user.reset_token_expires_at = datetime.now(UTC) + timedelta(minutes=settings.password_reset_expire_minutes)
        db.commit()
        background.add_task(send_recovery_email, email, token)
    return {"message": "If the account exists, a password reset link will be sent. Check your inbox and spam folder."}


@router.post("/reset-password")
def reset_password(
    payload: ResetPasswordRequest,
    db: Session = Depends(get_db),
    _rate_limit: None = Depends(enforce_auth_rate_limit),
):
    # Consume the token and replace the password atomically, including concurrent requests.
    result = db.execute(
        update(User)
        .where(
            User.reset_token_hash == hashlib.sha256(payload.token.encode()).hexdigest(),
            User.reset_token_expires_at > datetime.now(UTC),
        )
        .values(
            password_hash=hash_password(payload.password),
            reset_token_hash=None,
            reset_token_expires_at=None,
            session_version=User.session_version + 1,
        )
        .returning(User.id)
    )
    user_id = result.scalar_one_or_none()
    if user_id is None:
        db.rollback()
        raise HTTPException(status_code=400, detail="This reset link is invalid or expired. Request a new link.")
    add_audit_event(db, user_id=user_id, action="auth.password_reset", resource_type="user", resource_id=user_id)
    db.commit()
    return {"message": "Password updated. Sign in with your new password."}


@router.post("/change-password")
def change_password(
    payload: ChangePasswordRequest,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
    _rate_limit: None = Depends(enforce_auth_rate_limit),
):
    if not verify_password(payload.current_password, user.password_hash):
        raise HTTPException(status_code=400, detail="Current password is incorrect")
    result = db.execute(
        update(User).where(User.id == user.id, User.session_version == user.session_version)
        .values(password_hash=hash_password(payload.password), session_version=User.session_version + 1,
                reset_token_hash=None, reset_token_expires_at=None)
    )
    if result.rowcount != 1:
        db.rollback()
        raise HTTPException(status_code=409, detail="Account changed. Sign in again and retry.")
    add_audit_event(db, user_id=user.id, action="auth.password_changed", resource_type="user", resource_id=user.id)
    db.commit()
    return {"message": "Password updated. Sign in again on all devices."}
