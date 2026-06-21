// API base URL — override with VITE_API_URL env var
const BASE_URL = import.meta.env.VITE_API_URL ?? 'http://localhost:8000';

export interface RunConfig {
  method: 'fedavg' | 'fedprox' | 'dp_fedavg' | 'custom';
  rounds: number;
  local_epochs: number;
  mu: number;
  clip_norm: number;
  noise_multiplier: number;
  device: 'cpu' | 'cuda';
}

export interface BankMetrics {
  val_auc: number;
  val_loss: number;
  effective_weight: number;
  reliability_score: number;
  conflict_penalty: number;
  proximal_term: number;
  dp_applied: boolean;
}

export interface PrivacyBudget {
  epsilon: number;
  delta: number;
  noise_multiplier: number;
  clip_norm: number;
  num_clients: number;
  num_rounds: number;
  interpretation: string;
}

export interface AggregateMetrics {
  weighted_auc: number;
  mean_f1: number;
  mean_fnr: number;
  mean_fpr: number;
  [key: string]: number;
}

export interface BankEvalResult {
  bank_id: string;
  auc_roc: number;
  f1_score: number;
  false_negative_rate: number;
  false_positive_rate: number;
  accuracy: number;
  num_test_samples: number;
}

export interface RoundEvent {
  event: 'round';
  round: number;
  total_rounds: number;
  method: string;
  per_client: Record<string, BankMetrics>;
  privacy_budget: PrivacyBudget | null;
  test_evaluation: {
    aggregate: AggregateMetrics;
    per_bank: Record<string, BankEvalResult>;
  };
}

export interface CompleteEvent {
  event: 'complete';
  output: RunResult;
}

export interface ErrorEvent {
  event: 'error';
  message: string;
}

export type StreamEvent = RoundEvent | CompleteEvent | ErrorEvent;

export interface RunResult {
  method: string;
  rounds: number;
  local_epochs: number;
  mu: number;
  dp_config: { clip_norm: number; noise_multiplier: number } | null;
  elapsed_seconds: number;
  round_history: Array<{
    round: number;
    per_client: Record<string, BankMetrics>;
    privacy_budget: PrivacyBudget | null;
    test_evaluation: { aggregate: AggregateMetrics; per_bank: Record<string, BankEvalResult> };
  }>;
  final_aggregate: AggregateMetrics;
  final_per_bank: Record<string, BankEvalResult>;
}

export interface RunStatus {
  run_id: string;
  status: 'queued' | 'running' | 'done' | 'error';
  method: string;
  rounds: number;
}

// ── API helpers ──────────────────────────────────────────────────────────────

export async function checkHealth(): Promise<boolean> {
  try {
    const res = await fetch(`${BASE_URL}/health`, { signal: AbortSignal.timeout(3000) });
    return res.ok;
  } catch {
    return false;
  }
}

export async function startRun(config: RunConfig): Promise<string> {
  const res = await fetch(`${BASE_URL}/runs`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(config),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    const detailStr = typeof err.detail === 'string' ? err.detail : JSON.stringify(err.detail);
    throw new Error(detailStr ?? 'Failed to start run');
  }
  const data = await res.json();
  return data.run_id as string;
}

export async function getRun(runId: string): Promise<RunStatus> {
  const res = await fetch(`${BASE_URL}/runs/${runId}`);
  if (!res.ok) throw new Error(`Run ${runId} not found`);
  return res.json();
}

export async function getRunResult(runId: string): Promise<RunResult> {
  const res = await fetch(`${BASE_URL}/runs/${runId}/result`);
  if (!res.ok) throw new Error(`Result for ${runId} not available yet`);
  return res.json();
}

/**
 * Opens an EventSource SSE stream for a run.
 * Calls onEvent for each round event, onComplete when done, onError on error.
 * Returns a cleanup function to close the stream.
 */
export function streamRun(
  runId: string,
  onEvent: (evt: RoundEvent) => void,
  onComplete: (evt: CompleteEvent) => void,
  onError: (msg: string) => void,
): () => void {
  const source = new EventSource(`${BASE_URL}/runs/${runId}/stream`);

  source.onmessage = (e) => {
    try {
      const parsed: StreamEvent = JSON.parse(e.data);
      if (parsed.event === 'round') {
        onEvent(parsed as RoundEvent);
      } else if (parsed.event === 'complete') {
        onComplete(parsed as CompleteEvent);
        source.close();
      } else if (parsed.event === 'error') {
        onError((parsed as ErrorEvent).message);
        source.close();
      }
    } catch (err) {
      console.error('Failed to parse SSE event:', err);
    }
  };

  source.onerror = () => {
    onError('Lost connection to the backend stream.');
    source.close();
  };

  return () => source.close();
}
