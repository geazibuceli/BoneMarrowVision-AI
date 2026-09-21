import {
  StrictMode,
  useCallback,
  useEffect,
  useRef,
  useState,
  type DragEvent,
  type FormEvent,
  type ReactNode,
} from 'react';
import { createRoot } from 'react-dom/client';
import {
  BrowserRouter,
  Link,
  Navigate,
  NavLink,
  Route,
  Routes,
  useLocation,
  useNavigate,
  useParams,
  useSearchParams,
} from 'react-router-dom';
import {
  getExperiments,
  getModels,
  getOverview,
  getPrediction,
  getPredictions,
  getProtectedMedia,
  login,
  register,
  uploadPrediction,
} from './api';
import { AuthProvider, useAuth } from './auth';
import { ExperimentRunDetails, ExperimentRuns } from './ExperimentRuns';
import type { Experiment, Model, Prediction } from './types';
import './styles.css';

const DISCLAIMER = 'Research and Educational Use Only. This system is not a medical device and must not be used for clinical diagnosis, treatment, or decision-making.';
const MAX_UPLOAD_BYTES = 10 * 1024 * 1024;
const ACCEPTED_IMAGE_TYPES = new Set(['image/png', 'image/jpeg']);
const LOCAL_TIME_ZONE = Intl.DateTimeFormat().resolvedOptions().timeZone || 'your local time zone';
const PAGE_TITLES: Record<string, string> = {
  '/': 'Home',
  '/analyze': 'Analyze',
  '/dashboard': 'Dashboard',
  '/docs': 'Limitations',
  '/experiments': 'Experiments',
  '/history': 'Prediction history',
  '/login': 'Sign in',
  '/models': 'Model registry',
  '/register': 'Register',
};

function errorMessage(reason: unknown, fallback: string): string {
  return reason instanceof Error ? reason.message : fallback;
}

function formatTimestamp(value: string): string {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return 'Unknown date';
  return new Intl.DateTimeFormat('en', {
    dateStyle: 'medium',
    timeStyle: 'short',
  }).format(date);
}

type ResourceState<T> = {
  data: T | null;
  error: string;
  loading: boolean;
  reload: () => void;
};

function useResource<T>(loader: () => Promise<T>): ResourceState<T> {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(true);
  const [revision, setRevision] = useState(0);

  useEffect(() => {
    let active = true;

    loader()
      .then((value) => {
        if (active) setData(value);
      })
      .catch((reason: unknown) => {
        if (active) setError(errorMessage(reason, 'Could not load this content.'));
      })
      .finally(() => {
        if (active) setLoading(false);
      });

    return () => {
      active = false;
    };
  }, [loader, revision]);

  return {
    data,
    error,
    loading,
    reload: () => {
      setError('');
      setLoading(true);
      setRevision((value) => value + 1);
    },
  };
}

function LoadingMessage({ children = 'Loading…' }: { children?: ReactNode }) {
  return <output className="status-message" aria-live="polite">{children}</output>;
}

function ErrorMessage({ message, retry }: { message: string; retry?: () => void }) {
  return <div className="error-panel" role="alert"><p>{message}</p>{retry && <button className="button" type="button" onClick={retry}>Try again</button>}</div>;
}

function RouteMetadata() {
  const { pathname } = useLocation();
  const pageTitle = PAGE_TITLES[pathname] ?? (pathname.startsWith('/experiments/runs/') ? 'Experiment details' : /^\/history\/[^/]+$/.test(pathname) ? 'Prediction details' : 'Page not found');

  useEffect(() => {
    document.title = pathname === '/' ? 'BoneMarrowVision AI' : `${pageTitle} | BoneMarrowVision AI`;
  }, [pageTitle, pathname]);

  return <output className="sr-only" aria-live="polite" aria-atomic="true">{pageTitle} page</output>;
}

function Shell({ children }: { children: ReactNode }) {
  const { signOut, status, user } = useAuth();
  const navigate = useNavigate();

  const handleSignOut = () => {
    signOut();
    navigate('/', { replace: true });
  };

  return <>
    <a className="skip-link" href="#main-content">Skip to main content</a>
    <header>
      <NavLink to="/" end className="brand">BoneMarrowVision <span>AI</span></NavLink>
      <nav aria-label="Primary navigation">
        {user && <>
          <NavLink to="/dashboard">Dashboard</NavLink>
          <NavLink to="/analyze">Analyze</NavLink>
          <NavLink to="/history">History</NavLink>
          <NavLink to="/models">Models</NavLink>
          <NavLink to="/experiments">Experiments</NavLink>
        </>}
        {status === 'checking' && <span className="nav-status" aria-live="polite">Checking session…</span>}
        {status === 'anonymous' && <NavLink to="/login">Sign in</NavLink>}
        {user && <button className="link-button" type="button" onClick={handleSignOut}>Sign out</button>}
        <NavLink to="/docs">Limitations</NavLink>
      </nav>
    </header>
    {children}
    <footer>{DISCLAIMER}</footer>
  </>;
}

function RequireAuth({ children }: { children: ReactNode }) {
  const { status, user } = useAuth();
  const location = useLocation();

  if (status === 'checking') {
    return <main id="main-content" className="workspace centered-state"><LoadingMessage>Validating your session…</LoadingMessage></main>;
  }
  if (!user) return <Navigate to="/login" replace state={{ from: location }} />;
  return children;
}

function Landing() {
  const { user } = useAuth();
  return <main id="main-content" className="landing">
    <p className="eyebrow">Explainable deep learning platform</p>
    <h1>See the model’s reasoning.<br /><i>Question the result.</i></h1>
    <p className="lead">A research workspace for multiclass bone marrow cell image classification using transparent evaluation and Grad-CAM.</p>
    <div className="actions">
      <Link className="button primary" to={user ? '/dashboard' : '/login'}>Open analysis workspace</Link>
      <Link className="button" to="/docs">Read limitations</Link>
    </div>
    <section className="signal-grid" aria-label="Platform highlights">
      <div><strong>PyTorch</strong><span>Reproducible training</span></div>
      <div><strong>Grad-CAM</strong><span>Visual interpretability aid</span></div>
      <div><strong>TCIA</strong><span>Dataset DOI tracked</span></div>
    </section>
  </main>;
}

type RedirectState = { from?: { hash?: string; pathname?: string; search?: string } };

function safeRedirectTarget(from?: RedirectState['from']): string {
  const pathname = from?.pathname && /^\/(?!\/)/.test(from.pathname) ? from.pathname : '/dashboard';
  const search = from?.search?.startsWith('?') ? from.search : '';
  const hash = from?.hash?.startsWith('#') ? from.hash : '';
  return `${pathname}${search}${hash}`;
}

function Auth({ mode }: { mode: 'login' | 'register' }) {
  const { authenticate, status, user } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [name, setName] = useState('');
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);

  if (status === 'checking') {
    return <main id="main-content" className="workspace centered-state"><LoadingMessage>Checking your session…</LoadingMessage></main>;
  }
  if (user) return <Navigate to={safeRedirectTarget((location.state as RedirectState | null)?.from)} replace />;

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    setBusy(true);
    setError('');
    try {
      if (mode === 'register') await register(email.trim(), password, name.trim());
      const response = await login(email.trim(), password);
      await authenticate(response.access_token);

      const state = location.state as RedirectState | null;
      navigate(safeRedirectTarget(state?.from), { replace: true });
    } catch (reason) {
      setError(errorMessage(reason, 'Authentication failed.'));
    } finally {
      setBusy(false);
    }
  };

  return <main id="main-content" className="workspace auth" aria-busy={busy}>
    <p className="eyebrow">Research account</p>
    <h1>{mode === 'login' ? 'Welcome back' : 'Create an account'}</h1>
    <form onSubmit={submit}>
      <label htmlFor="auth-email">Email</label>
      <input id="auth-email" type="email" required autoComplete="email" disabled={busy} aria-invalid={Boolean(error)} aria-describedby={error ? 'auth-error' : undefined} value={email} onChange={(event) => setEmail(event.target.value)} />
      {mode === 'register' && <>
        <label htmlFor="auth-name">Full name</label>
        <input id="auth-name" required minLength={2} maxLength={200} autoComplete="name" disabled={busy} aria-invalid={Boolean(error)} aria-describedby={error ? 'auth-error' : undefined} value={name} onChange={(event) => setName(event.target.value)} />
      </>}
      <label htmlFor="auth-password">Password</label>
      <input id="auth-password" type="password" required minLength={8} maxLength={256} autoComplete={mode === 'login' ? 'current-password' : 'new-password'} disabled={busy} aria-invalid={Boolean(error)} aria-describedby={error ? 'auth-error' : undefined} value={password} onChange={(event) => setPassword(event.target.value)} />
      <button className="button primary" type="submit" disabled={busy} aria-busy={busy}>
        {busy ? 'Please wait…' : mode === 'login' ? 'Sign in' : 'Register'}
      </button>
      <div className="form-message" aria-live="assertive" aria-atomic="true">{error && <p id="auth-error" className="error">{error}</p>}</div>
    </form>
    <p className="muted">{mode === 'login' ? <Link to="/register">Create an account</Link> : <Link to="/login">Already registered?</Link>}</p>
  </main>;
}

function Dashboard() {
  const { token } = useAuth();
  const loadOverview = useCallback(() => getOverview(token), [token]);
  const overview = useResource(loadOverview);

  return <main id="main-content" className="workspace">
    <p className="eyebrow">Research overview</p>
    <h1>Dashboard</h1>
    {overview.loading && <LoadingMessage>Loading dashboard metrics…</LoadingMessage>}
    {overview.error && <ErrorMessage message={overview.error} retry={overview.reload} />}
    {overview.data && <>
      <div className="metric-grid">
        <div><span>Total predictions</span><strong>{overview.data.total_predictions}</strong></div>
        <div><span>Active model records</span><strong>{overview.data.active_model}</strong></div>
        <div><span>Dataset</span><strong>TCIA</strong></div>
      </div>
      <Link className="button primary" to="/analyze">Analyze an image</Link>
    </>}
  </main>;
}

function ProtectedImage({ path }: { path: string }) {
  const { token } = useAuth();
  const [source, setSource] = useState('');
  const [error, setError] = useState('');

  useEffect(() => {
    const controller = new AbortController();
    let objectUrl = '';
    let active = true;

    getProtectedMedia(path, token, controller.signal)
      .then((blob) => {
        if (!active) return;
        objectUrl = URL.createObjectURL(blob);
        setSource(objectUrl);
      })
      .catch((reason: unknown) => {
        if (active && !(reason instanceof DOMException && reason.name === 'AbortError')) {
          setError(errorMessage(reason, 'Could not load the Grad-CAM image.'));
        }
      });

    return () => {
      active = false;
      controller.abort();
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [path, token]);

  if (error) return <p className="error cam-message" role="alert">{error}</p>;
  if (!source) return <LoadingMessage>Loading Grad-CAM image…</LoadingMessage>;
  return <img className="cam" src={source} alt="Grad-CAM activation overlay for the prediction" />;
}

function Analyze() {
  const { token } = useAuth();
  const inputRef = useRef<HTMLInputElement>(null);
  const previewUrlRef = useRef('');
  const submittingRef = useRef(false);
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState('');
  const [result, setResult] = useState<Prediction | null>(null);
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const [dragActive, setDragActive] = useState(false);
  const [announcement, setAnnouncement] = useState('No image selected.');

  useEffect(() => () => {
    if (previewUrlRef.current) URL.revokeObjectURL(previewUrlRef.current);
  }, []);

  const replacePreview = (selected: File | null) => {
    if (previewUrlRef.current) URL.revokeObjectURL(previewUrlRef.current);
    const nextPreview = selected ? URL.createObjectURL(selected) : '';
    previewUrlRef.current = nextPreview;
    setPreview(nextPreview);
  };

  const choose = (selected?: File) => {
    if (!selected || submittingRef.current) return;
    setResult(null);
    setError('');

    if (!ACCEPTED_IMAGE_TYPES.has(selected.type)) {
      setFile(null);
      replacePreview(null);
      setError('Choose a valid PNG or JPEG image.');
      setAnnouncement('Image rejected. Choose a PNG or JPEG file.');
      return;
    }
    if (selected.size === 0) {
      setFile(null);
      replacePreview(null);
      setError('The selected image is empty.');
      setAnnouncement('Image rejected because it is empty.');
      return;
    }
    if (selected.size > MAX_UPLOAD_BYTES) {
      setFile(null);
      replacePreview(null);
      setError('The image is larger than the 10 MB upload limit.');
      setAnnouncement('Image rejected because it exceeds 10 MB.');
      return;
    }

    setFile(selected);
    replacePreview(selected);
    setAnnouncement(`${selected.name} selected.`);
  };

  const handleDrop = (event: DragEvent<HTMLButtonElement>) => {
    event.preventDefault();
    setDragActive(false);
    choose(event.dataTransfer.files[0]);
  };

  const submit = async () => {
    if (submittingRef.current) return;
    if (!file) {
      setError('Choose an image before running the analysis.');
      return;
    }

    submittingRef.current = true;
    setBusy(true);
    setError('');
    setResult(null);
    setAnnouncement('Analysis in progress.');
    try {
      const prediction = await uploadPrediction(file, token);
      setResult(prediction);
      setAnnouncement(`Analysis complete. Predicted class: ${prediction.predicted_class}.`);
    } catch (reason) {
      setError(errorMessage(reason, 'Analysis failed.'));
      setAnnouncement('Analysis failed.');
    } finally {
      submittingRef.current = false;
      setBusy(false);
    }
  };

  return <main id="main-content" className="workspace" aria-busy={busy}>
    <p className="eyebrow">Analysis workspace</p>
    <h1>Inspect an image</h1>
    <p id="upload-help" className="muted">Upload a PNG or JPEG of up to 10 MB. Do not upload personally identifiable information.</p>
    <input
      ref={inputRef}
      className="visually-hidden-file"
      type="file"
      disabled={busy}
      accept="image/png,image/jpeg,.png,.jpg,.jpeg"
      tabIndex={-1}
      onChange={(event) => {
        choose(event.target.files?.[0]);
        event.target.value = '';
      }}
    />
    <button
      className={`dropzone${dragActive ? ' drag-active' : ''}`}
      type="button"
      aria-describedby="upload-help"
      aria-label={file ? `Replace selected image ${file.name}` : 'Choose or drop a microscopy image'}
      disabled={busy}
      onClick={() => inputRef.current?.click()}
      onDragEnter={(event) => {
        event.preventDefault();
        if (!submittingRef.current) setDragActive(true);
      }}
      onDragOver={(event) => event.preventDefault()}
      onDragLeave={() => setDragActive(false)}
      onDrop={handleDrop}
    >
      {preview ? <img src={preview} alt="Preview of the selected bone marrow cell" /> : <span><strong>Choose or drop a microscopy image</strong><small>PNG or JPEG, up to 10 MB</small></span>}
    </button>
    <button className="button primary" type="button" disabled={!file || busy} onClick={submit} aria-busy={busy}>
      {busy ? 'Analyzing…' : 'Run analysis'}
    </button>
    <output className="sr-only" aria-live="polite" aria-atomic="true">{announcement}</output>
    {error && <p className="error" role="alert">{error}</p>}
    {result && <>
      <PredictionResult result={result} />
      <div className="actions"><Link className="button" to={`/history/${result.id}`}>Open saved result</Link><Link className="button" to="/history">View history</Link></div>
    </>}
  </main>;
}

function PredictionResult({ result }: { result: Prediction }) {
  return <section className="result" aria-labelledby="result-heading">
      <div>
        <span className="eyebrow">Prediction</span>
        <h2 id="result-heading">{result.predicted_class}</h2>
        <strong>{(result.confidence * 100).toFixed(1)}% confidence</strong>
        {result.low_confidence && <p className="warning">Low-confidence prediction. Treat it as uncertain.</p>}
      </div>
      <div className="top-list">
        <strong id="top-predictions-heading">Top predictions</strong>
        <ol aria-labelledby="top-predictions-heading">
          {result.top_predictions_json.map((item) => <li key={item.class}><span>{item.class}</span><span>{(item.probability * 100).toFixed(1)}%</span></li>)}
        </ol>
      </div>
      {result.gradcam_image_path && <ProtectedImage key={result.gradcam_image_path} path={result.gradcam_image_path} />}
      {!result.gradcam_image_path && <p className="muted">Grad-CAM is unavailable for this prediction. The saved result is still accessible.</p>}
      <p>{result.disclaimer}</p>
    </section>;
}

const HISTORY_PAGE_SIZE = 25;
// The API accepts offsets up to 1,000,000.
const MAX_HISTORY_PAGE = Math.floor(1_000_000 / HISTORY_PAGE_SIZE) + 1;

function History() {
  const [params, setParams] = useSearchParams();
  const requestedPage = Number(params.get('page') ?? '1');
  const page = Number.isSafeInteger(requestedPage) && requestedPage >= 1 && requestedPage <= MAX_HISTORY_PAGE ? requestedPage : 1;
  return <HistoryPage key={page} page={page} changePage={(nextPage) => setParams({ page: String(nextPage) })} />;
}

function HistoryPage({ page, changePage }: { page: number; changePage: (page: number) => void }) {
  const { token } = useAuth();
  // Fetch one extra record to detect the next page without a separate count request.
  const loadPredictions = useCallback(() => getPredictions(token, HISTORY_PAGE_SIZE + 1, (page - 1) * HISTORY_PAGE_SIZE), [token, page]);
  const predictions = useResource(loadPredictions);
  const rows = predictions.data?.slice(0, HISTORY_PAGE_SIZE);

  return <main id="main-content" className="workspace">
    <p className="eyebrow">Research log</p>
    <h1>Prediction history</h1>
    <p className="muted">Times are shown in {LOCAL_TIME_ZONE}.</p>
    {predictions.loading && <LoadingMessage>Loading prediction history…</LoadingMessage>}
    {predictions.error && <ErrorMessage message={predictions.error} retry={predictions.reload} />}
    {predictions.data && <section className="table-wrap" aria-label="Prediction history" tabIndex={0}>
      <table>
        <caption className="sr-only">Predictions saved by this account</caption>
        <thead><tr><th scope="col">Class</th><th scope="col">Confidence</th><th scope="col">Date and time</th><th scope="col">Result</th></tr></thead>
        <tbody>{rows?.map((row) => <tr key={row.id}><td>{row.predicted_class}</td><td>{(row.confidence * 100).toFixed(1)}%{row.low_confidence && <span className="warning"> (Low confidence)</span>}</td><td><time dateTime={row.created_at}>{formatTimestamp(row.created_at)}</time></td><td><Link to={`/history/${row.id}`} state={{ historyPage: page }} aria-label={`View prediction ${row.id}`}>View result</Link></td></tr>)}</tbody>
      </table>
      {predictions.data.length === 0 && <p className="empty">{page === 1 ? 'No predictions recorded for this account.' : 'No predictions on this page.'}</p>}
    </section>}
    <nav className="actions" aria-label="History pages">
      <button className="button" type="button" disabled={page === 1} onClick={() => changePage(page - 1)}>Previous</button>
      <span aria-live="polite">Page {page}</span>
      <button className="button" type="button" disabled={predictions.loading || Boolean(predictions.error) || !predictions.data || predictions.data.length <= HISTORY_PAGE_SIZE || page >= MAX_HISTORY_PAGE} onClick={() => changePage(page + 1)}>Next</button>
    </nav>
    <Link className="button primary" to="/analyze">Analyze an image</Link>
  </main>;
}

function PredictionDetails() {
  const { predictionId = '' } = useParams();
  return <PredictionDetailsContent key={predictionId} predictionId={predictionId} />;
}

function PredictionDetailsContent({ predictionId }: { predictionId: string }) {
  const { token } = useAuth();
  const location = useLocation();
  const historyPage = (location.state as { historyPage?: number } | null)?.historyPage;
  const backPath = typeof historyPage === 'number' && Number.isSafeInteger(historyPage) && historyPage > 1 && historyPage <= MAX_HISTORY_PAGE ? `/history?page=${historyPage}` : '/history';
  const loadPrediction = useCallback(() => getPrediction(predictionId, token), [predictionId, token]);
  const prediction = useResource(loadPrediction);

  return <main id="main-content" className="workspace">
    <Link to={backPath}>Back to history</Link>
    <h1>Prediction details</h1>
    {prediction.loading && <LoadingMessage>Loading prediction details...</LoadingMessage>}
    {prediction.error && <ErrorMessage message={prediction.error} retry={prediction.reload} />}
    {prediction.data && <>
      <p className="muted">Saved <time dateTime={prediction.data.created_at}>{formatTimestamp(prediction.data.created_at)}</time> ({LOCAL_TIME_ZONE}).</p>
      <PredictionResult result={prediction.data} />
    </>}
    <div className="actions"><Link className="button primary" to="/analyze">Analyze another image</Link></div>
  </main>;
}

function Models() {
  const { token } = useAuth();
  const loadModels = useCallback(() => getModels(token), [token]);
  const models = useResource<Model[]>(loadModels);

  return <main id="main-content" className="workspace">
    <p className="eyebrow">Registry</p>
    <h1>Model registry</h1>
    {models.loading && <LoadingMessage>Loading model registry…</LoadingMessage>}
    {models.error && <ErrorMessage message={models.error} retry={models.reload} />}
    {models.data && <section className="table-wrap" aria-label="Model registry" tabIndex={0}>
      <table>
        <caption className="sr-only">Registered machine-learning models</caption>
        <thead><tr><th scope="col">Name</th><th scope="col">Architecture</th><th scope="col">Version</th><th scope="col">Status</th></tr></thead>
        <tbody>{models.data.map((row) => <tr key={row.id}><td>{row.name}</td><td>{row.architecture}</td><td>{row.version}</td><td>{row.is_active ? 'Active' : 'Available'}</td></tr>)}</tbody>
      </table>
      {models.data.length === 0 && <p className="empty">No models have been registered yet.</p>}
    </section>}
  </main>;
}

function Experiments() {
  const { token } = useAuth();
  const loadExperiments = useCallback(() => getExperiments(token), [token]);
  const experiments = useResource<Experiment[]>(loadExperiments);

  return <main id="main-content" className="workspace">
    <p className="eyebrow">Experiment tracking</p>
    <h1>Experiments</h1>
    <ExperimentRuns />
    <p className="muted">Times are shown in {LOCAL_TIME_ZONE}.</p>
    {experiments.loading && <LoadingMessage>Loading experiments…</LoadingMessage>}
    {experiments.error && <ErrorMessage message={experiments.error} retry={experiments.reload} />}
    {experiments.data && <section className="table-wrap" aria-label="Experiments" tabIndex={0}>
      <table>
        <caption className="sr-only">Recorded training experiments</caption>
        <thead><tr><th scope="col">Name</th><th scope="col">Dataset version</th><th scope="col">Validation macro F1</th><th scope="col">Created</th></tr></thead>
        <tbody>{experiments.data.map((row) => <tr key={row.id}><td>{row.name}</td><td>{row.dataset_version}</td><td>{String(row.validation_metrics_json?.macro_f1 ?? '—')}</td><td><time dateTime={row.created_at}>{formatTimestamp(row.created_at)}</time></td></tr>)}</tbody>
      </table>
      {experiments.data.length === 0 && <p className="empty">No experiments have been recorded yet.</p>}
    </section>}
  </main>;
}

function Docs() {
  return <main id="main-content" className="workspace prose">
    <p className="eyebrow">Responsible research</p>
    <h1>Limitations first</h1>
    <h2>Intended use</h2>
    <p>This platform supports educational and scientific analysis. It is not clinically validated and does not provide diagnosis, treatment, triage, or medical advice.</p>
    <h2>Interpretability</h2>
    <p>Grad-CAM is an interpretability aid. A highlighted region is not proof that a prediction is clinically correct or causally grounded.</p>
    <h2>Dataset and privacy</h2>
    <p>Results may not generalize across hospitals, scanners, stains, preparation protocols, or populations. Keep personally identifiable information out of uploads and follow applicable LGPD requirements.</p>
  </main>;
}

function NotFound() {
  return <main id="main-content" className="workspace centered-state">
    <p className="eyebrow">404</p>
    <h1>Page not found</h1>
    <p className="muted">The address may be incorrect or the page may have moved.</p>
    <Link className="button primary" to="/">Return home</Link>
  </main>;
}

function Protected({ children }: { children: ReactNode }) {
  return <RequireAuth>{children}</RequireAuth>;
}

function App() {
  return <>
    <RouteMetadata />
    <Shell><Routes>
      <Route path="/" element={<Landing />} />
      <Route path="/login" element={<Auth mode="login" />} />
      <Route path="/register" element={<Auth mode="register" />} />
      <Route path="/dashboard" element={<Protected><Dashboard /></Protected>} />
      <Route path="/analyze" element={<Protected><Analyze /></Protected>} />
      <Route path="/history" element={<Protected><History /></Protected>} />
      <Route path="/history/:predictionId" element={<Protected><PredictionDetails /></Protected>} />
      <Route path="/models" element={<Protected><Models /></Protected>} />
      <Route path="/experiments" element={<Protected><Experiments /></Protected>} />
      <Route path="/experiments/runs/:runId" element={<Protected><ExperimentRunDetails /></Protected>} />
      <Route path="/docs" element={<Docs />} />
      <Route path="*" element={<NotFound />} />
    </Routes></Shell>
  </>;
}

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <BrowserRouter>
      <AuthProvider>
        <App />
      </AuthProvider>
    </BrowserRouter>
  </StrictMode>,
);
