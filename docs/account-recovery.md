# Password changes and account recovery

Signed-in users can open **Password**, enter their current password, and choose a
new one. A successful change invalidates all existing sessions and pending reset
links. The user must sign in again on every device.

**Forgot your password?** on the login screen sends a single-use link when SMTP
is configured. Set `SMTP_HOST`, `SMTP_PORT`, `SMTP_FROM`, `SMTP_USERNAME`, and
`SMTP_PASSWORD` in the private environment. Use `SMTP_SECURITY=starttls` (usually
port 587) or `ssl` (usually port 465). Certificate verification is always enabled.
These values are also forwarded to the Compose backend.

Set `PASSWORD_RESET_URL` to the frontend's `/reset-password` page. Production
requires HTTPS when SMTP is enabled. The URL must not include query parameters or
a fragment. Links expire after `PASSWORD_RESET_EXPIRE_MINUTES` (default 30).
The link token is carried in a fragment, removed from browser history on opening,
and submitted to the API in the request body. Only its SHA-256 digest is stored.

Requests use the existing authentication rate limit, plus a one-minute cooldown
per normalized email address. Known and unknown accounts receive the same 202
response. SMTP failures are logged without addresses or credentials. When SMTP
is absent, the UI explicitly reports that email recovery is unavailable; changing
a password while signed in still works.

Apply `alembic upgrade head` before starting the updated backend. Migration
`0002_account_recovery` adds nullable recovery fields, a session version, and a
history index without replacing existing records. Tokens issued before this
upgrade remain valid until their normal expiry or the next password change.

To validate a configured provider, use a disposable account and verify delivery,
expiry, one-time use, and invalidation of a second signed-in session. Automated
tests mock delivery; they do not prove provider delivery or inbox placement.
