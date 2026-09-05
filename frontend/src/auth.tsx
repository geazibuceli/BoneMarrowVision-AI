import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from 'react';
import { AUTH_UNAUTHORIZED_EVENT, getMe } from './api';
import type { User } from './types';

const TOKEN_KEY = 'bone-marrow-vision:token';

type AuthStatus = 'checking' | 'authenticated' | 'anonymous';

type AuthContextValue = {
  authenticate: (token: string) => Promise<User>;
  signOut: () => void;
  status: AuthStatus;
  token: string;
  user: User | null;
};

const AuthContext = createContext<AuthContextValue | null>(null);

function storedToken(): string {
  try {
    return localStorage.getItem(TOKEN_KEY) ?? '';
  } catch {
    return '';
  }
}

function persistToken(token: string): void {
  try {
    if (token) localStorage.setItem(TOKEN_KEY, token);
    else localStorage.removeItem(TOKEN_KEY);
  } catch {
    // Storage can be unavailable in privacy-focused browser modes. The in-memory
    // session still works for the current page.
  }
}

export function AuthProvider({ children }: { children: ReactNode }) {
  const [token, setToken] = useState(storedToken);
  const [initialToken] = useState(token);
  const [user, setUser] = useState<User | null>(null);
  const [status, setStatus] = useState<AuthStatus>(token ? 'checking' : 'anonymous');

  const signOut = useCallback(() => {
    persistToken('');
    setToken('');
    setUser(null);
    setStatus('anonymous');
  }, []);

  useEffect(() => {
    const handleUnauthorized = () => signOut();
    window.addEventListener(AUTH_UNAUTHORIZED_EVENT, handleUnauthorized);
    return () => window.removeEventListener(AUTH_UNAUTHORIZED_EVENT, handleUnauthorized);
  }, [signOut]);

  useEffect(() => {
    if (!initialToken) return;

    let active = true;
    getMe(initialToken)
      .then((currentUser) => {
        if (!active) return;
        setUser(currentUser);
        setStatus('authenticated');
      })
      .catch(() => {
        if (active) signOut();
      });

    return () => {
      active = false;
    };
  }, [initialToken, signOut]);

  const authenticate = useCallback(async (nextToken: string) => {
    persistToken(nextToken);
    setToken(nextToken);
    setStatus('checking');
    try {
      const currentUser = await getMe(nextToken);
      setUser(currentUser);
      setStatus('authenticated');
      return currentUser;
    } catch (reason) {
      signOut();
      throw reason;
    }
  }, [signOut]);

  const value = useMemo(
    () => ({ authenticate, signOut, status, token, user }),
    [authenticate, signOut, status, token, user],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth(): AuthContextValue {
  const context = useContext(AuthContext);
  if (!context) throw new Error('useAuth must be used within AuthProvider');
  return context;
}
