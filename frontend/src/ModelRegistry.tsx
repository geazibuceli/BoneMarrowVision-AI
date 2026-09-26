import { useEffect, useRef, useState, type FormEvent } from 'react';
import { activateModel, getModelActivations, getModels, registerModel, type ModelActivation } from './api';
import { useAuth } from './auth';
import type { Model } from './types';

function errorText(reason: unknown): string {
  return reason instanceof Error ? reason.message : 'The request failed. Please try again.';
}

function reportedMetric(model: Model, key: string): string {
  const value = model.metrics_json[key] ?? model.metrics_json[`test_${key}`];
  return typeof value === 'number' && Number.isFinite(value) ? `${(value * 100).toFixed(2)}%` : 'Not reported';
}

function Registration({ onRegistered, busy, setBusy }: {
  onRegistered: (model: Model) => void; busy: boolean; setBusy: (value: boolean) => void;
}) {
  const { token } = useAuth();
  const submitting = useRef(false);
  const [error, setError] = useState('');
  const submit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (submitting.current || busy) return;
    const form = event.currentTarget;
    const values = new FormData(form);
    const field = (key: string) => String(values.get(key) ?? '').trim();
    submitting.current = true;
    setBusy(true);
    setError('');
    try {
      let metrics: unknown;
      try { metrics = JSON.parse(field('metrics') || '{}'); }
      catch { throw new Error('Metrics must be a valid JSON object.'); }
      if (!metrics || typeof metrics !== 'object' || Array.isArray(metrics)) throw new Error('Metrics must be a JSON object.');
      const model = await registerModel({
        name: field('name'), version: field('version'), architecture: field('architecture'),
        checkpoint_path: field('checkpoint'), config_path: field('config'), metrics_json: metrics as Record<string, unknown>,
      }, token);
      form.reset();
      onRegistered(model);
    } catch (reason) { setError(errorText(reason)); }
    finally { submitting.current = false; setBusy(false); }
  };
  return <section aria-labelledby="register-model-heading">
    <h2 id="register-model-heading">Register a model</h2>
    <p>Place the checkpoint and configuration on the server in the configured model and configuration directories. Registration validates the checkpoint; activation is a separate action.</p>
    <form className="model-form" onSubmit={submit}>
      <fieldset disabled={busy}>
        <legend>Model information</legend>
        <label>Name<input name="name" required maxLength={200} /></label>
        <label>Version<input name="version" required maxLength={80} /></label>
        <label htmlFor="model-architecture">Architecture</label><select id="model-architecture" name="architecture" defaultValue="efficientnet_b0">
          <option value="efficientnet_b0">EfficientNet-B0</option><option value="resnet50">ResNet50</option><option value="convnext_tiny">ConvNeXt-Tiny</option><option value="baseline_cnn">Baseline CNN</option>
        </select>
        <label>Checkpoint path<input name="checkpoint" required maxLength={1000} placeholder="models/best_model.pt" /></label>
        <label>Configuration path<input name="config" required maxLength={1000} placeholder="ml/configs/efficientnet_b0.yaml" /></label>
        <label>Metrics (JSON, optional)<textarea name="metrics" rows={4} placeholder={'{"macro_f1": 0.73, "low_confidence_threshold": 0.6}'} /></label>
        <button type="submit" className="button primary">Register model</button>
      </fieldset>
      {error && <p role="alert">{error}</p>}
    </form>
  </section>;
}

function ActivationHistory({ revision, busy, onActivate }: {
  revision: number; busy: boolean; onActivate: (id: number) => Promise<void>;
}) {
  const { token } = useAuth();
  const [page, setPage] = useState(0);
  const [data, setData] = useState<{ page: number; rows: ModelActivation[] } | null>(null);
  const [error, setError] = useState('');
  useEffect(() => {
    let current = true;
    getModelActivations(token, page * 25).then((rows) => {
      if (current) { setData({ page, rows }); setError(''); }
    }).catch((reason: unknown) => { if (current) setError(errorText(reason)); });
    return () => { current = false; };
  }, [token, page, revision]);
  const rows = data?.page === page ? data.rows : null;
  return <section aria-labelledby="activation-history-heading">
    <h2 id="activation-history-heading">Activation history</h2>
    <p>Activate a previously used version to return to it. Existing predictions retain their original model.</p>
    {error && <p role="alert">{error}</p>}
    {!rows && !error && <output>Loading activation history…</output>}
    {rows?.length === 0 && <p>No model activations recorded.</p>}
    {!!rows?.length && <section className="table-wrap" tabIndex={0} aria-label="Recorded model activations"><table>
      <thead><tr><th scope="col">Model</th><th scope="col">Version</th><th scope="col">Activated at (local time)</th><th scope="col">Action</th></tr></thead>
      <tbody>{rows.slice(0, 25).map((row) => <tr key={row.id}><td>{row.model_name}</td><td>{row.model_version ?? 'Unavailable'}</td>
        <td><time dateTime={row.created_at}>{new Date(row.created_at).toLocaleString('en-US')}</time></td>
        <td><button type="button" disabled={busy || row.model_id === null} onClick={() => { if (row.model_id !== null) void onActivate(row.model_id); }}>Activate this version</button></td>
      </tr>)}</tbody>
    </table></section>}
    <nav className="actions" aria-label="Activation history pages">
      <button type="button" disabled={page === 0 || busy} onClick={() => setPage(page - 1)}>Previous activations</button>
      <span>Page {page + 1}</span>
      <button type="button" disabled={busy || !rows || rows.length <= 25 || page >= 40000} onClick={() => setPage(page + 1)}>Next activations</button>
    </nav>
  </section>;
}

export function ModelRegistry() {
  const { token, user } = useAuth();
  const admin = user?.role === 'admin';
  const [page, setPage] = useState(0);
  const [data, setData] = useState<{ page: number; rows: Model[] } | null>(null);
  const [error, setError] = useState('');
  const [actionError, setActionError] = useState('');
  const [notice, setNotice] = useState('');
  const [busy, setBusy] = useState(false);
  const activating = useRef(false);
  const [revision, setRevision] = useState(0);
  useEffect(() => {
    let current = true;
    getModels(token, 26, page * 25).then((rows) => {
      if (current) { setData({ page, rows }); setError(''); }
    }).catch((reason: unknown) => { if (current) setError(errorText(reason)); });
    return () => { current = false; };
  }, [token, page, revision]);
  const activate = async (id: number) => {
    if (activating.current || busy) return;
    activating.current = true;
    setBusy(true); setActionError(''); setNotice('');
    try {
      const model = await activateModel(id, token);
      setNotice(`Active model: ${model.name} · ${model.version}. New analyses will use this version.`);
      setRevision((value) => value + 1);
    } catch (reason) { setActionError(errorText(reason)); }
    finally { activating.current = false; setBusy(false); }
  };
  const rows = data?.page === page ? data.rows : null;
  return <main id="main-content" className="workspace model-registry">
    <p className="eyebrow">Registry</p><h1>Model registry</h1>
    <p>Compare reported metrics and inspect registered versions. Metrics from different datasets or evaluation conditions are not directly comparable.</p>
    <button type="button" disabled={busy} onClick={() => setRevision((value) => value + 1)}>Refresh registry</button>
    {busy && <output>Validating model, please wait…</output>}
    {notice && <output>{notice}</output>}
    {actionError && <p role="alert">{actionError}</p>}
    {error && <p role="alert">{error}</p>}
    {!rows && !error && <output>Loading model registry…</output>}
    {rows && <section className="table-wrap" tabIndex={0} aria-label="Registered models"><table>
      <thead><tr><th scope="col">Model</th><th scope="col">Version</th><th scope="col">Architecture</th><th scope="col">Reported accuracy</th><th scope="col">Reported macro F1</th><th scope="col">Status</th><th scope="col">Details</th>{admin && <th scope="col">Action</th>}</tr></thead>
      <tbody>{rows.slice(0, 25).map((model) => <tr key={model.id}>
        <th scope="row" aria-label={model.name}>{model.name}</th><td>{model.version}</td><td>{model.architecture}</td>
        <td>{reportedMetric(model, 'accuracy')}</td><td>{reportedMetric(model, 'macro_f1')}</td><td>{model.is_active ? 'Active' : 'Available'}</td>
        <td aria-label={`Details for ${model.name} ${model.version}`}><details><summary>Inspect {model.name} {model.version}</summary><p className="model-hash">Checkpoint SHA-256: {model.checkpoint_sha256 ?? 'Not recorded'}</p><pre className="experiment-json">{JSON.stringify(model.metrics_json, null, 2)}</pre></details></td>
        {admin && <td><button type="button" disabled={busy || model.is_active} onClick={() => { void activate(model.id); }}>Activate</button></td>}
      </tr>)}</tbody>
    </table>{rows.length === 0 && <p>No models have been registered yet.</p>}</section>}
    <nav className="actions" aria-label="Model pages">
      <button type="button" disabled={page === 0 || busy} onClick={() => setPage(page - 1)}>Previous models</button><span>Page {page + 1}</span>
      <button type="button" disabled={busy || !rows || rows.length <= 25 || page >= 40000} onClick={() => setPage(page + 1)}>Next models</button>
    </nav>
    {admin && <>
      <Registration busy={busy} setBusy={setBusy} onRegistered={(model) => {
        setNotice(`Registered ${model.name} · ${model.version}. Select Activate when ready.`);
        setActionError(''); setPage(0); setRevision((value) => value + 1);
      }} />
      <ActivationHistory revision={revision} busy={busy} onActivate={activate} />
    </>}
  </main>;
}
