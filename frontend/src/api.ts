import type { Experiment, Model, Prediction, User } from './types';

type RuntimeImportMeta = ImportMeta & { env?: { VITE_API_URL?: string } };

const configuredApiUrl = (import.meta as RuntimeImportMeta).env?.VITE_API_URL;
const DEFAULT_API_URL = '/api/v1';
const SAFE_IMAGE_TYPES = new Set(['image/jpeg', 'image/png']);

export const AUTH_UNAUTHORIZED_EVENT = 'bone-marrow-vision:unauthorized';
export const API_URL = normalizeApiUrl(configuredApiUrl ?? DEFAULT_API_URL);

export class ApiError extends Error {
  public readonly status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
  }
}

export function normalizeApiUrl(value: string): string {
  const normalized = value.trim().replace(/\/+$/, '');
  if (!normalized) return DEFAULT_API_URL;
  if (/^\/(?!\/)/.test(normalized)) {
    return /[\\?#]/.test(normalized) ? DEFAULT_API_URL : normalized;
  }

  try {
    const candidate = new URL(normalized);
    const safe = /^https?:$/.test(candidate.protocol)
      && !candidate.username
      && !candidate.password
      && !candidate.search
      && !candidate.hash;
    return safe ? candidate.href.replace(/\/+$/, '') : DEFAULT_API_URL;
  } catch {
    return DEFAULT_API_URL;
  }
}

function validationDetail(detail: unknown): string | null {
  if (typeof detail === 'string' && detail.trim()) return detail.trim();
  if (!Array.isArray(detail)) return null;

  const messages = detail
    .map((entry) => {
      if (typeof entry === 'string') return entry;
      if (entry && typeof entry === 'object' && 'msg' in entry && typeof entry.msg === 'string') {
        return entry.msg;
      }
      return null;
    })
    .filter((message): message is string => Boolean(message));

  return messages.length > 0 ? messages.join('. ').slice(0, 300) : null;
}

function messageFromPayload(payload: unknown): string | null {
  if (typeof payload === 'string' && payload.trim()) return payload.trim().slice(0, 300);
  if (!payload || typeof payload !== 'object') return null;

  if ('detail' in payload) return validationDetail(payload.detail);
  if ('message' in payload && typeof payload.message === 'string' && payload.message.trim()) {
    return payload.message.trim().slice(0, 300);
  }
  return null;
}

async function responsePayload(response: Response): Promise<unknown> {
  if (response.status === 204) return undefined;

  const body = await response.text();
  if (!body.trim()) return undefined;

  const contentType = response.headers.get('content-type')?.toLowerCase() ?? '';
  if (contentType.includes('json')) {
    try {
      return JSON.parse(body) as unknown;
    } catch {
      return undefined;
    }
  }

  // Do not expose an HTML proxy error page. Plain-text API errors are safe to
  // show after applying a conservative length limit.
  if (contentType.startsWith('text/plain')) return body;
  return undefined;
}

function notifyUnauthorized(): void {
  if (typeof window !== 'undefined') window.dispatchEvent(new Event(AUTH_UNAUTHORIZED_EVENT));
}

export async function readJsonResponse<T>(
  response: Response,
  fallbackMessage: string,
  authenticated = false,
): Promise<T> {
  const payload = await responsePayload(response);

  if (!response.ok) {
    if (response.status === 401 && authenticated) notifyUnauthorized();
    const statusFallback = response.statusText ? `${fallbackMessage} (${response.statusText})` : fallbackMessage;
    throw new ApiError(messageFromPayload(payload) ?? statusFallback, response.status);
  }

  if (response.status === 204) return undefined as T;
  if (payload === undefined) throw new ApiError('The server returned an empty or unexpected response.', response.status);
  return payload as T;
}

async function fetchResponse(url: string, options: RequestInit): Promise<Response> {
  try {
    return await fetch(url, options);
  } catch (reason) {
    if (reason instanceof DOMException && reason.name === 'AbortError') throw reason;
    throw new ApiError('Could not reach the server. Check your connection and try again.', 0);
  }
}

async function request<T>(path: string, options: RequestInit = {}, fallbackMessage = 'Request failed'): Promise<T> {
  const headers = new Headers(options.headers);
  const response = await fetchResponse(`${API_URL}${path}`, { ...options, headers });
  return readJsonResponse<T>(response, fallbackMessage, headers.has('Authorization'));
}

function authorization(token: string): HeadersInit {
  return { Authorization: `Bearer ${token}` };
}

export function resolveMediaUrl(
  path: string,
  apiUrl = API_URL,
  clientOrigin = typeof window === 'undefined' ? 'http://localhost' : window.location.origin,
): string {
  const normalizedApiUrl = normalizeApiUrl(apiUrl);
  const apiIsAbsolute = /^https?:\/\//i.test(normalizedApiUrl);
  const apiLocation = new URL(normalizedApiUrl, clientOrigin);
  const candidate = new URL(path, apiLocation.origin);
  const apiPath = apiLocation.pathname.replace(/\/+$/, '');
  const trustedPath = candidate.pathname.startsWith(`${apiPath}/`);

  if (!/^https?:$/.test(candidate.protocol) || candidate.origin !== apiLocation.origin || !trustedPath) {
    throw new ApiError('The server returned an invalid media URL.', 0);
  }

  return apiIsAbsolute ? candidate.href : `${candidate.pathname}${candidate.search}`;
}

export async function getProtectedMedia(path: string, token: string, signal?: AbortSignal): Promise<Blob> {
  const response = await fetchResponse(resolveMediaUrl(path), {
    headers: authorization(token),
    signal,
  });

  if (!response.ok) {
    await readJsonResponse<never>(response, 'Could not load the Grad-CAM image', true);
  }

  const contentType = response.headers.get('content-type')?.split(';', 1)[0].trim().toLowerCase() ?? '';
  if (!SAFE_IMAGE_TYPES.has(contentType)) {
    throw new ApiError('The server returned an invalid Grad-CAM image.', response.status);
  }
  return response.blob();
}

export function login(email: string, password: string): Promise<{ access_token: string }> {
  const form = new URLSearchParams({ username: email, password });
  return request<{ access_token: string }>(
    '/auth/login',
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
      body: form,
    },
    'Could not sign in',
  );
}

export function register(email: string, password: string, fullName: string): Promise<User> {
  return request<User>(
    '/auth/register',
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, password, full_name: fullName }),
    },
    'Could not create the account',
  );
}

export function getMe(token: string): Promise<User> {
  return request<User>('/auth/me', { headers: authorization(token) }, 'Could not validate the session');
}

export function uploadPrediction(file: File, token: string): Promise<Prediction> {
  const form = new FormData();
  form.append('file', file);
  return request<Prediction>(
    '/predictions',
    { method: 'POST', body: form, headers: authorization(token) },
    'Analysis failed',
  );
}

export function getPredictions(token: string, limit = 100, offset = 0): Promise<Prediction[]> {
  const query = new URLSearchParams({ limit: String(limit), offset: String(offset) });
  return request<Prediction[]>(`/predictions?${query}`, { headers: authorization(token) }, 'Could not load prediction history');
}

export function getPrediction(id: string, token: string): Promise<Prediction> {
  return request<Prediction>(`/predictions/${encodeURIComponent(id)}`, { headers: authorization(token) }, 'Could not load this prediction');
}

export function getModels(token: string, limit = 100, offset = 0): Promise<Model[]> {
  return request<Model[]>(`/models?limit=${limit}&offset=${offset}`, { headers: authorization(token) }, 'Could not load the model registry');
}

export type ModelRegistration = {
  name: string; version: string; architecture: string; checkpoint_path: string; config_path: string;
  metrics_json: Record<string, unknown>;
};
export type ModelActivation = { id: number; model_id: number | null; model_name: string; model_version: string | null; created_at: string };

export function registerModel(payload: ModelRegistration, token: string): Promise<Model> {
  return request('/models/register', {
    method: 'POST', headers: { ...authorization(token), 'Content-Type': 'application/json' }, body: JSON.stringify(payload),
  }, 'Could not register the model');
}

export function activateModel(id: number, token: string): Promise<Model> {
  return request(`/models/${id}/activate`, { method: 'POST', headers: authorization(token) }, 'Could not activate the model');
}

export function getModelActivations(token: string, offset = 0): Promise<ModelActivation[]> {
  return request(`/models/activation-history?limit=26&offset=${offset}`, { headers: authorization(token) }, 'Could not load activation history');
}

export function getExperiments(token: string): Promise<Experiment[]> {
  return request<Experiment[]>('/experiments', { headers: authorization(token) }, 'Could not load experiments');
}

export type RunSummary = { id: string; name?: string; kind?: string; status: Record<string, unknown> };
export type ExperimentRun = RunSummary & {
  architectures?: { architecture: string; configuration: Record<string, unknown>; history: Record<string, number>[] | null; test_report: Record<string, unknown> | null }[];
  matching_test_splits?: boolean;
  configuration: Record<string, unknown> | null;
  comparison: Record<string, unknown> | null;
  history: Record<string, number>[] | null;
  control_history: Record<string, number>[] | null;
  test_report: Record<string, unknown> | null;
  control_test_report: Record<string, unknown> | null;
  class_counts: Record<string, number> | null;
};

export function getExperimentRuns(token: string): Promise<RunSummary[]> {
  return request('/experiments/runs', { headers: authorization(token) }, 'Could not load local runs');
}

export function getExperimentRun(id: string, token: string): Promise<ExperimentRun> {
  return request(`/experiments/runs/${encodeURIComponent(id)}`, { headers: authorization(token) }, 'Could not load this run');
}

export function getOverview(token: string): Promise<{ total_predictions: number; active_model: number }> {
  return request<{ total_predictions: number; active_model: number }>(
    '/analytics/overview',
    { headers: authorization(token) },
    'Could not load the dashboard',
  );
}
