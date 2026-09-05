import secrets

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.audit import add_audit_event
from app.core.config import settings
from app.database import get_db
from app.models import User
from app.schemas import RegisterRequest, TokenResponse, UserResponse
from app.security import (
    create_access_token,
    decode_subject,
    hash_password,
    verify_dummy_password,
    verify_password,
)

router = APIRouter(prefix="/api/v1/auth", tags=["authentication"])
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")


def current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> User:
    try:
        user_id = int(decode_subject(token))
    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        ) from error
    user = db.get(User, user_id)
    if not user:
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
def register(payload: RegisterRequest, db: Session = Depends(get_db)):
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
def login(form: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
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
    return TokenResponse(access_token=create_access_token(str(user.id)))


@router.get("/me", response_model=UserResponse)
def me(user: User = Depends(current_user)):
    return user
