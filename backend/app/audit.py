from typing import Any

from sqlalchemy.orm import Session

from app.models import AuditLog


def add_audit_event(
    database: Session,
    *,
    user_id: int | None,
    action: str,
    resource_type: str,
    resource_id: int | str | None = None,
    metadata: dict[str, Any] | None = None,
) -> AuditLog:
    """Stage a non-sensitive audit event in the caller's transaction."""
    event = AuditLog(
        user_id=user_id,
        action=action,
        resource_type=resource_type,
        resource_id=str(resource_id) if resource_id is not None else None,
        metadata_json=metadata or {},
    )
    database.add(event)
    return event
