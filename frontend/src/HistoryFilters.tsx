import { useEffect, useRef, useState, type FormEvent } from 'react';
import { exportPredictions, getHistoryFilterOptions, type HistoryFilters as Filters } from './api';
import { useAuth } from './auth';

export function HistoryFilters({ filters, onApply }: { filters: Filters; onApply: (filters: Filters) => void }) {
  const { token } = useAuth();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [message, setMessage] = useState('');
  const exporting = useRef(false);
  const [options, setOptions] = useState<Awaited<ReturnType<typeof getHistoryFilterOptions>> | null>(null);
  useEffect(() => {
    let active = true;
    getHistoryFilterOptions(token).then((value) => { if (active) setOptions(value); })
      .catch(() => { if (active) setError('Filter suggestions could not load. You can still enter a class or model ID.'); });
    return () => { active = false; };
  }, [token]);
  const submit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const data = new FormData(event.currentTarget);
    const next = Object.fromEntries(data.entries()) as Filters;
    if (next.date_from && next.date_to && next.date_from > next.date_to) {
      setError('Start date must be on or before end date.');
      return;
    }
    setError('');
    onApply(next);
  };
  const download = async () => {
    if (exporting.current) return;
    exporting.current = true;
    setBusy(true);
    setError('');
    setMessage('');
    try {
      const blob = await exportPredictions(token, filters);
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement('a');
      anchor.href = url;
      anchor.download = 'predictions.csv';
      document.body.append(anchor);
      anchor.click();
      anchor.remove();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
      setMessage('CSV downloaded with all matching predictions.');
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'Export failed.');
    } finally {
      exporting.current = false;
      setBusy(false);
    }
  };
  return <section aria-label="History filters">
    <form className="history-filters" onSubmit={submit}>
      <label>Class (exact name)<input name="predicted_class" list="history-classes" maxLength={160} defaultValue={filters.predicted_class} /></label>
      <datalist id="history-classes">{options?.classes.map((name) => <option key={name} value={name}>{name}</option>)}</datalist>
      {options ? <label>Model<select name="model_id" defaultValue={filters.model_id ?? ''}><option value="">All models</option>{options.models.map((model) => <option key={model.id} value={model.id}>{model.name} · {model.version}</option>)}{filters.model_id && !options.models.some((model) => String(model.id) === filters.model_id) && <option value={filters.model_id}>Model {filters.model_id}</option>}</select></label> : <label>Model ID<input name="model_id" type="number" min={1} step={1} defaultValue={filters.model_id} /></label>}
      <label>From (UTC)<input name="date_from" type="date" defaultValue={filters.date_from} /></label>
      <label>Through (UTC)<input name="date_to" type="date" defaultValue={filters.date_to} /></label>
      <label>Confidence<select name="low_confidence" defaultValue={filters.low_confidence ?? ''}><option value="">All</option><option value="true">Low confidence</option><option value="false">Above threshold</option></select></label>
      <div className="actions"><button className="button" type="submit">Apply filters</button><button className="button" type="button" onClick={() => onApply({})}>Clear filters</button></div>
    </form>
    <button type="button" className="button" disabled={busy} onClick={download}>{busy ? 'Exporting…' : 'Export filtered CSV'}</button>
    <p className="muted">Export uses the applied filters across all pages, up to 10,000 rows.</p>
    {error && <p className="error" role="alert">{error}</p>}{message && <output>{message}</output>}
  </section>;
}
