"""Deliver recovery links without logging credentials or revealing account existence."""

import logging
import smtplib
import ssl
from email.message import EmailMessage

from app.core.config import settings

LOGGER = logging.getLogger(__name__)


def send_recovery_email(email: str, token: str) -> None:
    message = EmailMessage()
    message["From"] = settings.smtp_from
    message["To"] = email
    message["Subject"] = "Reset your BoneMarrowVision password"
    # Fragments are not sent to the web server or included in its access logs.
    link = f"{settings.password_reset_url}#token={token}"
    message.set_content(
        f"Use this link to reset your password within {settings.password_reset_expire_minutes} minutes:\n\n"
        f"{link}\n\nIf you did not request this, ignore this email.\n"
    )
    try:
        context = ssl.create_default_context()
        connection = (
            smtplib.SMTP_SSL(settings.smtp_host, settings.smtp_port, timeout=15, context=context)
            if settings.smtp_security == "ssl"
            else smtplib.SMTP(settings.smtp_host, settings.smtp_port, timeout=15)
        )
        with connection as smtp:
            if settings.smtp_security == "starttls":
                smtp.starttls(context=context)
            if settings.smtp_username:
                smtp.login(settings.smtp_username, settings.smtp_password.get_secret_value())
            smtp.send_message(message)
    except (OSError, smtplib.SMTPException):
        # SMTP errors can contain recipient addresses or server credentials.
        LOGGER.error("Password recovery email delivery failed; check SMTP configuration")
