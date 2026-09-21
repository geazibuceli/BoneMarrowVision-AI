import { useEffect, useState } from 'react';
import { Link, useParams } from 'react-router-dom';
import { getExperimentRun, getExperimentRuns, type ExperimentRun, type RunSummary } from './api';
import { useAuth } from './auth';

function message(error: unknown): string {
  return error instanceof Error ? error.message : 'Could not load experiment data.';
}

export function ExperimentRuns() {
  const { token } = useAuth();
  const [runs, setRuns] = useState<RunSummary[] | null>(null);
  const [error, setError] = useState('');
  const [revision, setRevision] = useState(0);
  useEffect(() => {
    let current = true;
    getExperimentRuns(token).then((data) => { if (current) { setRuns(data); setError(''); } })
      .catch((reason: unknown) => { if (current) setError(message(reason)); });
    return () => { current = false; };
  }, [token, revision]);
  return <section aria-labelledby="local-runs-heading">
    <h2 id="local-runs-heading">Local training runs</h2>
    <button type="button" onClick={() => setRevision((value) => value + 1)}>Refresh runs</button>
    {error && <p role="alert">{error}</p>}
    {!runs && !error && <output>Loading local runs…</output>}
    {runs?.length === 0 && <p>No local training runs are available.</p>}
    {!!runs?.length && <section className="table-wrap" tabIndex={0} aria-label="Local training runs">
      <table><thead><tr><th scope="col">Run</th><th scope="col">Stage</th></tr></thead>
        <tbody>{runs.map((run) => <tr key={run.id}><td><Link to={`/experiments/runs/${encodeURIComponent(run.id)}`}>{run.id}</Link></td><td>{String(run.status.stage ?? 'unknown').replace(/_/g, ' ')}</td></tr>)}</tbody>
      </table>
    </section>}
  </section>;
}

function object(value: unknown): Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value) ? value as Record<string, unknown> : {};
}

function metric(value: unknown, percentage = true): string {
  return typeof value === 'number' && Number.isFinite(value)
    ? percentage ? `${(value * 100).toFixed(2)}%` : value.toFixed(4) : '—';
}

function Comparison({ data }: { data: Record<string, unknown> }) {
  const models = [
    ['real_only_control', 'Matched real-only control'],
    ['gan_augmented', 'GAN augmented'],
    ['historical_baseline', 'Historical baseline (uncontrolled)'],
  ].filter(([key]) => data[key]);
  const delta = object(data.delta_gan_minus_control);
  return <section><h2>Test comparison</h2>
    <p>{data.comparison_type === 'matched_real_only_control' ? 'Matched training configuration and held-out splits.' : 'Historical comparison only. These results do not establish a benefit from GAN augmentation.'}</p>
    <section className="table-wrap" tabIndex={0} aria-label="Test comparison"><table>
      <thead><tr><th scope="col">Model</th><th scope="col">Accuracy</th><th scope="col">Macro precision</th><th scope="col">Macro F1</th><th scope="col">Loss</th></tr></thead>
      <tbody>{models.map(([key, label]) => { const row = object(data[key]); return <tr key={key}><th scope="row">{label}</th><td>{metric(row.test_accuracy)}</td><td>{metric(row.test_macro_precision)}</td><td>{metric(row.test_macro_f1)}</td><td>{metric(row.test_loss, false)}</td></tr>; })}</tbody>
    </table></section>
    {typeof delta.test_macro_f1 === 'number' && <p>Macro F1 change, GAN minus control: {(delta.test_macro_f1 * 100).toFixed(2)} percentage points.</p>}
    {typeof data.limitations === 'string' && <p className="warning">{data.limitations}</p>}
    {typeof data.historical_limitations === 'string' && <p className="muted">{data.historical_limitations}</p>}
  </section>;
}

function LearningCurve({ history, title }: { history: Record<string, number>[] | null; title: string }) {
  if (!history?.length) return null;
  const rows = history.filter((row) => Number.isFinite(row.epoch) && Number.isFinite(row.validation_macro_f1));
  const maxEpoch = Math.max(1, ...rows.map((row) => row.epoch));
  const points = rows.map((row) => `${40 + row.epoch / maxEpoch * 540},${180 - Math.max(0, Math.min(1, row.validation_macro_f1)) * 150}`).join(' ');
  return <section><h2>{title}</h2>
    {/* SVG needs an image role for accessible chart semantics. */}
    {/* oxlint-disable-next-line jsx-a11y/prefer-tag-over-role */}
    <svg viewBox="0 0 620 220" role="img" aria-label={`${title}: validation macro F1 by epoch. Exact values follow in the expandable table.`} className="learning-curve">
      <path d="M40 30V180H580" fill="none" stroke="currentColor" />
      <text x="0" y="35">100%</text><text x="12" y="180">0%</text>
      <text x="40" y="205">0</text><text x="500" y="205">Epoch {maxEpoch}</text>
      <polyline points={points} fill="none" stroke="currentColor" strokeWidth="3" />
    </svg>
    <details><summary>View epoch metrics</summary><section className="table-wrap" tabIndex={0} aria-label={`${title} metrics`}><table>
      <thead><tr><th scope="col">Epoch</th><th scope="col">Train loss (weighted)</th><th scope="col">Validation loss (unweighted)</th><th scope="col">Validation macro F1</th></tr></thead>
      <tbody>{history.map((row) => <tr key={row.epoch}><td>{row.epoch}</td><td>{metric(row.train_loss, false)}</td><td>{metric(row.validation_loss, false)}</td><td>{metric(row.validation_macro_f1)}</td></tr>)}</tbody>
    </table></section></details>
  </section>;
}

function JsonDetails({ title, value }: { title: string; value: unknown }) {
  return value ? <details><summary>{title}</summary><pre className="experiment-json">{JSON.stringify(value, null, 2)}</pre></details> : null;
}

function ClassReport({ title, report }: { title: string; report: Record<string, unknown> | null }) {
  if (!report) return null;
  const names = report.class_names;
  const matrix = report.confusion_matrix;
  if (!Array.isArray(names) || !names.every((name) => typeof name === 'string')
    || !Array.isArray(matrix) || matrix.length !== names.length
    || !matrix.every((row) => Array.isArray(row) && row.length === names.length
      && row.every((value) => typeof value === 'number' && Number.isFinite(value) && value >= 0))) {
    return <JsonDetails title={title} value={report} />;
  }
  const counts = matrix as number[][];
  return <section><h2>{title}</h2>
    <section className="table-wrap" tabIndex={0} aria-label={title}><table>
      <thead><tr><th scope="col">Class</th><th scope="col">Precision</th><th scope="col">Recall</th><th scope="col">F1</th><th scope="col">Test images</th></tr></thead>
      <tbody>{names.map((name, index) => {
        const correct = counts[index][index];
        const support = counts[index].reduce((sum, value) => sum + value, 0);
        const predicted = counts.reduce((sum, row) => sum + row[index], 0);
        const precision = predicted ? correct / predicted : 0;
        const recall = support ? correct / support : 0;
        const f1 = precision + recall ? 2 * precision * recall / (precision + recall) : 0;
        return <tr key={name}><th scope="row">{name}</th><td>{metric(precision)}</td><td>{metric(recall)}</td><td>{metric(f1)}</td><td>{support}</td></tr>;
      })}</tbody>
    </table></section>
    <JsonDetails title="Full evaluation artifact" value={report} />
  </section>;
}

export function ExperimentRunDetails() {
  const { runId = '' } = useParams();
  return <RunDetails key={runId} id={runId} />;
}

function RunDetails({ id }: { id: string }) {
  const { token } = useAuth();
  const [run, setRun] = useState<ExperimentRun | null>(null);
  const [error, setError] = useState('');
  const [revision, setRevision] = useState(0);
  useEffect(() => {
    let current = true;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const refresh = async () => {
      try {
        const result = await getExperimentRun(id, token);
        if (!current) return;
        setRun(result);
        setError('');
        if (!['completed', 'failed'].includes(String(result.status.stage))) timer = setTimeout(() => { void refresh(); }, 15000);
      } catch (reason) { if (current) setError(message(reason)); }
    };
    void refresh();
    return () => { current = false; clearTimeout(timer); };
  }, [id, token, revision]);
  return <main id="main-content" className="workspace experiment-detail">
    <Link to="/experiments">Back to experiments</Link><h1>Experiment details</h1><p className="run-name">{id}</p>
    <button type="button" onClick={() => setRevision((value) => value + 1)}>Refresh experiment</button>
    {error && <p role="alert">{error}</p>}
    {!run && !error && <output>Loading experiment…</output>}
    {run && <>
      <output>Stage: {String(run.status.stage).replace(/_/g, ' ')}</output>
      {typeof run.status.error === 'string' && <p role="alert">{run.status.error}</p>}
      {typeof run.status.updated_at === 'string' && <p>Last stage update: <time dateTime={run.status.updated_at}>{new Date(run.status.updated_at).toLocaleString()}</time></p>}
      {typeof run.status.elapsed_seconds === 'number' && <p>Elapsed at last stage update: {(run.status.elapsed_seconds / 3600).toFixed(2)} hours</p>}
      {run.comparison ? <Comparison data={run.comparison} /> : <p>Test comparison is not available yet.</p>}
      <LearningCurve history={run.control_history} title="Real-only control: validation macro F1" />
      <LearningCurve history={run.history} title="GAN augmented: validation macro F1" />
      <JsonDetails title="Training configuration" value={run.configuration} />
      <ClassReport title="GAN augmented: per-class test metrics" report={run.test_report} />
      <ClassReport title="Real-only control: per-class test metrics" report={run.control_test_report} />
      <JsonDetails title="Real training class counts" value={run.class_counts} />
      <JsonDetails title="Stage history" value={run.status.stages} />
      <JsonDetails title="Comparison artifact and checkpoint hashes" value={run.comparison} />
    </>}
  </main>;
}
