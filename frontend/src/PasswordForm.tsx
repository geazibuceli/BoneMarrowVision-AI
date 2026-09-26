import { useEffect, useRef, useState, type FormEvent } from 'react';
import { Link, Navigate, useLocation } from 'react-router-dom';
import { passwordRequest } from './api';
import { useAuth } from './auth';

export function PasswordForm({ mode }: { mode: 'forgot-password' | 'reset-password' | 'change-password' }) {
  const { token, signOut, status } = useAuth();
  const location = useLocation();
  const [resetToken] = useState(() => new URLSearchParams(location.hash.slice(1)).get('token') ?? '');
  const [email, setEmail] = useState('');
  const [current, setCurrent] = useState('');
  const [password, setPassword] = useState('');
  const [confirmation, setConfirmation] = useState('');
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState('');
  const [error, setError] = useState('');
  const submitting = useRef(false);
  useEffect(() => {
    if (mode === 'reset-password' && location.hash) window.history.replaceState(window.history.state, '', location.pathname);
  }, [location.hash, location.pathname, mode]);
  const submit = async (event: FormEvent) => {
    event.preventDefault();
    if (submitting.current) return;
    setError('');
    if (mode !== 'forgot-password' && password !== confirmation) {
      setError('Passwords do not match.');
      return;
    }
    submitting.current = true;
    setBusy(true);
    try {
      const payload: Record<string, string> = mode === 'forgot-password' ? { email } : mode === 'reset-password' ? { token: resetToken, password } : { current_password: current, password };
      const response = await passwordRequest(mode, payload, mode === 'change-password' ? token : undefined);
      setMessage(response.message);
      setPassword('');
      setConfirmation('');
      setCurrent('');
      if (mode !== 'forgot-password') signOut();
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'Request failed. Please try again.');
    } finally {
      setBusy(false);
      submitting.current = false;
    }
  };
  const invalidLink = mode === 'reset-password' && !resetToken;
  if (mode === 'change-password' && !message && status === 'anonymous') return <Navigate to="/login" replace />;
  return <main id="main-content" className="workspace auth" aria-busy={busy}>
    <h1>{mode === 'forgot-password' ? 'Recover your account' : mode === 'change-password' ? 'Change password' : 'Reset password'}</h1>
    {invalidLink && <p role="alert">This reset link is missing its token. <Link to="/forgot-password">Request a new link</Link>.</p>}
    {message ? <><output>{message}</output><Link to="/login">Sign in</Link></> : !invalidLink && <form onSubmit={submit}>
      {mode === 'forgot-password' ? <label>Email<input type="email" autoComplete="email" required value={email} disabled={busy} onChange={(event) => setEmail(event.target.value)} /></label> : <>
        {mode === 'change-password' && <label>Current password<input type="password" autoComplete="current-password" required maxLength={256} value={current} disabled={busy} onChange={(event) => setCurrent(event.target.value)} /></label>}
        <label>New password<input type="password" autoComplete="new-password" required minLength={8} maxLength={256} value={password} disabled={busy} onChange={(event) => setPassword(event.target.value)} /></label>
        <label>Confirm new password<input type="password" autoComplete="new-password" required minLength={8} maxLength={256} value={confirmation} disabled={busy} onChange={(event) => setConfirmation(event.target.value)} /></label>
      </>}
      {error && <p role="alert" className="error">{error}</p>}
      <button className="button primary" disabled={busy} type="submit">{busy ? 'Please wait…' : mode === 'forgot-password' ? 'Send reset link' : 'Update password'}</button>
    </form>}
    {mode === 'reset-password' && !message && <Link to="/forgot-password">Request another reset link</Link>}
  </main>;
}
