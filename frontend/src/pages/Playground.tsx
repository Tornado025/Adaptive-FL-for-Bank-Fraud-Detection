import { useCallback, useRef, useState } from 'react';
import {
  LineChart, Line, BarChart, Bar, XAxis, YAxis, Tooltip,
  ResponsiveContainer, CartesianGrid, Legend, Cell,
} from 'recharts';
import {
  startRun, streamRun,
  type RunConfig, type RoundEvent, type CompleteEvent, type RunResult, type BankMetrics,
} from '../lib/api';
import PrivacyGauge from '../components/PrivacyGauge';

const BANK_IDS = ['bank_a', 'bank_b', 'bank_c', 'bank_d'];
const BANK_COLORS: Record<string, string> = {
  bank_a: '#3b82f6',
  bank_b: '#8b5cf6',
  bank_c: '#06b6d4',
  bank_d: '#ec4899',
};
const BANK_LABELS: Record<string, string> = {
  bank_a: 'Bank A', bank_b: 'Bank B', bank_c: 'Bank C', bank_d: 'Bank D',
};

type RunState = 'idle' | 'connecting' | 'running' | 'done' | 'error';

interface ChartPoint {
  round: number;
  weighted_auc: number;
  mean_f1: number;
}

const CustomTooltip = ({ active, payload, label }: any) => {
  if (!active || !payload?.length) return null;
  return (
    <div className="card px-3 py-2 text-xs font-mono">
      <p className="text-slate-400 mb-1">Round {label}</p>
      {payload.map((p: any) => (
        <p key={p.name} style={{ color: p.color }}>{p.name}: {typeof p.value === 'number' ? p.value.toFixed(4) : p.value}</p>
      ))}
    </div>
  );
};

export default function Playground() {
  // ── Config state ─────────────────────────────────────────────────────
  const [method, setMethod] = useState<RunConfig['method']>('dp_fedavg');
  const [rounds, setRounds] = useState(5);
  const [localEpochs, setLocalEpochs] = useState(3);
  const [mu, setMu] = useState(0.01);
  const [clipNorm, setClipNorm] = useState(1.0);
  const [noiseMult, setNoiseMult] = useState(0);

  // ── Run state ────────────────────────────────────────────────────────
  const [runState, setRunState] = useState<RunState>('idle');
  const [currentRound, setCurrentRound] = useState(0);
  const [totalRounds, setTotalRounds] = useState(0);
  const [bankStatus, setBankStatus] = useState<Record<string, string>>({});
  const [chartData, setChartData] = useState<ChartPoint[]>([]);
  const [latestBankMetrics, setLatestBankMetrics] = useState<Record<string, BankMetrics> | null>(null);
  const [privacyBudget, setPrivacyBudget] = useState<{ epsilon: number; interpretation: string } | null>(null);
  const [logs, setLogs] = useState<string[]>([]);
  const [logsOpen, setLogsOpen] = useState(false);
  const [result, setResult] = useState<RunResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [runId, setRunId] = useState<string | null>(null);

  const cleanupRef = useRef<(() => void) | null>(null);
  const logEndRef = useRef<HTMLDivElement>(null);

  const addLog = useCallback((line: string) => {
    setLogs(prev => [...prev, line]);
    setTimeout(() => logEndRef.current?.scrollIntoView({ behavior: 'smooth' }), 50);
  }, []);

  const handleStart = async () => {
    setRunState('connecting');
    setError(null);
    setChartData([]);
    setLatestBankMetrics(null);
    setPrivacyBudget(null);
    setLogs([]);
    setResult(null);
    setCurrentRound(0);
    setBankStatus({});
    setTotalRounds(rounds);

    const config: RunConfig = {
      method,
      rounds,
      local_epochs: localEpochs,
      mu,
      clip_norm: clipNorm,
      noise_multiplier: noiseMult,
      device: 'cpu',
    };

    try {
      addLog(`[init] Starting run: method=${method} rounds=${rounds} local_epochs=${localEpochs}`);
      if (method === 'dp_fedavg' || method === 'custom') {
        addLog(`[init] DP config: clip_norm=${clipNorm} noise_multiplier=${noiseMult}`);
      }

      const id = await startRun(config);
      setRunId(id);
      setRunState('running');
      addLog(`[api] Run started — ID: ${id}`);

      // Mark all banks as "queued"
      setBankStatus({ bank_a: 'queued', bank_b: 'queued', bank_c: 'queued', bank_d: 'queued' });

      const cleanup = streamRun(
        id,
        (evt: RoundEvent) => {
          setCurrentRound(evt.round);
          setTotalRounds(evt.total_rounds);
          setLatestBankMetrics(evt.per_client);
          if (evt.privacy_budget) {
            setPrivacyBudget({ epsilon: evt.privacy_budget.epsilon, interpretation: evt.privacy_budget.interpretation });
          }
          setChartData(prev => [...prev, {
            round: evt.round,
            weighted_auc: evt.test_evaluation.aggregate.weighted_auc,
            mean_f1: evt.test_evaluation.aggregate.mean_f1,
          }]);

          // Bank status
          const newStatus: Record<string, string> = {};
          for (const bid of BANK_IDS) {
            const m = evt.per_client[bid];
            newStatus[bid] = m ? `val_auc=${m.val_auc.toFixed(4)} · w=${m.effective_weight.toFixed(3)}` : 'done';
          }
          setBankStatus(newStatus);

          // Log
          addLog(`[round ${evt.round}/${evt.total_rounds}] AUC=${evt.test_evaluation.aggregate.weighted_auc.toFixed(4)} F1=${evt.test_evaluation.aggregate.mean_f1.toFixed(4)}`);
          for (const bid of BANK_IDS) {
            const m = evt.per_client[bid];
            if (m) addLog(`  ${bid}: auc=${m.val_auc.toFixed(4)} eff_w=${m.effective_weight.toFixed(4)} rel=${m.reliability_score.toFixed(4)} conf=${m.conflict_penalty.toFixed(4)}`);
          }
          if (evt.privacy_budget) {
            addLog(`  [dp] ε=${evt.privacy_budget.epsilon.toFixed(4)} — ${evt.privacy_budget.interpretation}`);
          }
        },
        (evt: CompleteEvent) => {
          setResult(evt.output);
          setRunState('done');
          addLog(`[done] Training complete in ${evt.output.elapsed_seconds.toFixed(1)}s`);
        },
        (msg: string) => {
          setError(msg);
          setRunState('error');
          addLog(`[error] ${msg}`);
        },
      );
      cleanupRef.current = cleanup;
    } catch (e: any) {
      const msg = e?.message ?? 'Failed to connect to backend';
      setError(msg);
      setRunState('error');
      addLog(`[error] ${msg}`);
    }
  };

  const handleDownload = () => {
    if (!result) return;
    const blob = new Blob([JSON.stringify(result, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `fl_results_${runId?.slice(0, 8) ?? 'run'}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const showDP = method === 'dp_fedavg' || method === 'custom';

  // ── Bar data for effective_weight / reliability_score ────────────────
  const bankBarData = latestBankMetrics
    ? BANK_IDS.map(bid => ({
        bank: BANK_LABELS[bid],
        effective_weight: latestBankMetrics[bid]?.effective_weight ?? 0,
        reliability_score: latestBankMetrics[bid]?.reliability_score ?? 0,
      }))
    : [];

  return (
    <main className="max-w-7xl mx-auto px-6 py-12 animate-fade-in">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-white mb-2">Playground</h1>
        <p className="text-slate-400">Configure and launch a live federated training run. Metrics stream per round.</p>
      </div>

      {/* API offline notice */}
      {runState === 'error' && error?.includes('connect') && (
        <div className="card border-amber-500/30 p-5 mb-8">
          <p className="text-amber-400 font-semibold mb-2">Backend not running</p>
          <p className="text-slate-400 text-sm mb-3">Start the FastAPI server from the repo root:</p>
          <pre className="log-panel text-amber-400 text-xs">uvicorn server.api:app --reload --host 0.0.0.0 --port 8000</pre>
          <p className="text-slate-500 text-xs mt-2">Then refresh this page.</p>
        </div>
      )}

      <div className="grid lg:grid-cols-[360px_1fr] gap-8">

        {/* ── Left: Config panel ──────────────────────────────────────── */}
        <aside className="space-y-6">
          <div className="card p-6 space-y-5">
            <h2 className="text-sm font-mono uppercase tracking-widest text-slate-500">Run Configuration</h2>

            {/* Method */}
            <div>
              <label className="text-xs text-slate-400 font-medium uppercase tracking-widest block mb-2">Method</label>
              <div className="grid grid-cols-2 gap-2">
                {(['dp_fedavg', 'fedprox'] as const).map(m => (
                  <button
                    key={m}
                    id={`method-${m}`}
                    onClick={() => setMethod(m as any)}
                    disabled={runState === 'running' || runState === 'connecting'}
                    className={`px-3 py-2 rounded text-xs font-mono font-semibold border transition-colors ${
                      method === m
                        ? 'border-amber-400 bg-amber-500/15 text-amber-400'
                        : 'border-base-600 text-slate-500 hover:border-slate-500 hover:text-slate-300'
                    } disabled:opacity-40`}
                  >
                    {m === 'dp_fedavg' ? 'DP' : 'FedProx'}
                  </button>
                ))}
              </div>
            </div>

            {/* Rounds */}
            <div>
              <label className="text-xs text-slate-400 font-medium uppercase tracking-widest flex justify-between mb-2">
                <span>Rounds</span>
                <span className="font-mono text-amber-400">{rounds}</span>
              </label>
              <input type="range" min={1} max={30} value={rounds}
                onChange={e => setRounds(+e.target.value)}
                disabled={runState === 'running' || runState === 'connecting'}
                aria-label="rounds" />
              <div className="flex justify-between text-xs font-mono text-slate-600 mt-1"><span>1</span><span>30</span></div>
            </div>

            {/* Local epochs */}
            <div>
              <label className="text-xs text-slate-400 font-medium uppercase tracking-widest flex justify-between mb-2">
                <span>Local Epochs</span>
                <span className="font-mono text-amber-400">{localEpochs}</span>
              </label>
              <input type="range" min={1} max={10} value={localEpochs}
                onChange={e => setLocalEpochs(+e.target.value)}
                disabled={runState === 'running' || runState === 'connecting'}
                aria-label="local_epochs" />
            </div>

            {/* Mu */}
            <div>
              <label className="text-xs text-slate-400 font-medium uppercase tracking-widest flex justify-between mb-2">
                <span>FedProx μ</span>
                <span className="font-mono text-amber-400">{mu.toFixed(3)}</span>
              </label>
              <input type="range" min={0} max={0.5} step={0.001} value={mu}
                onChange={e => setMu(parseFloat(e.target.value))}
                disabled={runState === 'running' || runState === 'connecting'}
                aria-label="mu" />
            </div>

            {/* DP params (conditional) */}
            {showDP && (
              <>
                <div>
                  <label className="text-xs text-slate-400 font-medium uppercase tracking-widest flex justify-between mb-2">
                    <span>clip_norm</span>
                    <span className="font-mono text-amber-400">{clipNorm.toFixed(1)}</span>
                  </label>
                  <input type="range" min={0.1} max={5} step={0.1} value={clipNorm}
                    onChange={e => setClipNorm(parseFloat(e.target.value))}
                    disabled={runState === 'running' || runState === 'connecting'}
                    aria-label="clip_norm" />
                </div>
                <div>
                  <label className="text-xs text-slate-400 font-medium uppercase tracking-widest flex justify-between mb-2">
                    <span>noise_multiplier</span>
                    <span className="font-mono text-amber-400">{noiseMult.toFixed(2)}</span>
                  </label>
                  <input type="range" min={0} max={3} step={0.05} value={noiseMult}
                    onChange={e => setNoiseMult(parseFloat(e.target.value))}
                    disabled={runState === 'running' || runState === 'connecting'}
                    aria-label="noise_multiplier" />
                </div>
              </>
            )}

            <button
              id="start-run-btn"
              onClick={handleStart}
              disabled={runState === 'running' || runState === 'connecting'}
              className="w-full py-3 rounded-lg bg-amber-500 text-black font-bold text-sm hover:bg-amber-400 transition-colors disabled:opacity-40 disabled:cursor-not-allowed amber-glow"
            >
              {runState === 'connecting' ? 'Connecting…' :
               runState === 'running'    ? `Round ${currentRound} / ${totalRounds}…` :
               'Start Run'}
            </button>
          </div>

          {/* Error block */}
          {runState === 'error' && error && !error.includes('connect') && (
            <div className="card border-red-500/30 p-4">
              <p className="text-red-400 text-xs font-mono break-words">{error}</p>
            </div>
          )}
        </aside>

        {/* ── Right: Live metrics ──────────────────────────────────────── */}
        <div className="space-y-6">

          {/* Progress header */}
          {(runState === 'running' || runState === 'done') && (
            <div className="card p-4">
              <div className="flex items-center gap-3 mb-4">
                {runState === 'running' && (
                  <span className="w-2 h-2 rounded-full bg-amber-400 animate-pulse" />
                )}
                {runState === 'done' && (
                  <span className="w-2 h-2 rounded-full bg-emerald-400" />
                )}
                <span className="font-mono text-sm text-slate-300">
                  {runState === 'running'
                    ? `Round ${currentRound} / ${totalRounds} running…`
                    : `Complete — ${totalRounds} rounds finished`}
                </span>
              </div>

              {/* Per-bank status */}
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                {BANK_IDS.map(bid => (
                  <div key={bid} className="rounded-lg p-3" style={{ background: `${BANK_COLORS[bid]}12`, border: `1px solid ${BANK_COLORS[bid]}30` }}>
                    <p className="text-xs font-mono font-bold mb-1" style={{ color: BANK_COLORS[bid] }}>
                      {BANK_LABELS[bid]}
                    </p>
                    <p className="text-xs text-slate-400 font-mono leading-relaxed break-words">
                      {bankStatus[bid] ?? '—'}
                    </p>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Line chart */}
          {chartData.length > 0 && (
            <div className="card p-5">
              <h3 className="text-xs font-mono uppercase tracking-widest text-slate-500 mb-4">
                Global Metrics per Round
              </h3>
              <ResponsiveContainer width="100%" height={200}>
                <LineChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1c2333" />
                  <XAxis dataKey="round" tick={{ fill: '#64748b', fontSize: 11, fontFamily: 'JetBrains Mono' }} axisLine={false} tickLine={false} />
                  <YAxis domain={[0, 1]} tick={{ fill: '#64748b', fontSize: 11, fontFamily: 'JetBrains Mono' }} axisLine={false} tickLine={false} />
                  <Tooltip content={<CustomTooltip />} />
                  <Legend wrapperStyle={{ fontSize: '12px', fontFamily: 'JetBrains Mono', color: '#94a3b8' }} />
                  <Line type="monotone" dataKey="weighted_auc" name="weighted_auc" stroke="#f59e0b" strokeWidth={2} dot={{ fill: '#f59e0b', r: 3 }} isAnimationActive />
                  <Line type="monotone" dataKey="mean_f1" name="mean_f1" stroke="#10b981" strokeWidth={2} dot={{ fill: '#10b981', r: 3 }} isAnimationActive />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}

          {/* Adaptive weights bar chart */}
          {bankBarData.length > 0 && (
            <div className="card p-5">
              <h3 className="text-xs font-mono uppercase tracking-widest text-slate-500 mb-1">
                Adaptive Weights — Round {currentRound}
              </h3>
              <p className="text-xs text-slate-600 mb-4">
                How much each bank's update influenced the global model this round.
              </p>
              <ResponsiveContainer width="100%" height={180}>
                <BarChart data={bankBarData} barCategoryGap="25%">
                  <XAxis dataKey="bank" tick={{ fill: '#64748b', fontSize: 11, fontFamily: 'JetBrains Mono' }} axisLine={false} tickLine={false} />
                  <YAxis domain={[0, 1]} tick={{ fill: '#64748b', fontSize: 11, fontFamily: 'JetBrains Mono' }} axisLine={false} tickLine={false} />
                  <Tooltip content={<CustomTooltip />} />
                  <Legend wrapperStyle={{ fontSize: '12px', fontFamily: 'JetBrains Mono', color: '#94a3b8' }} />
                  <Bar dataKey="effective_weight" name="eff. weight" radius={[4,4,0,0]}>
                    {bankBarData.map((_, i) => (
                      <Cell key={i} fill={BANK_COLORS[BANK_IDS[i]]} />
                    ))}
                  </Bar>
                  <Bar dataKey="reliability_score" name="reliability" fill="#2d3748" radius={[4,4,0,0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}

          {/* DP gauge */}
          {showDP && privacyBudget && (
            <div className="flex items-center gap-4">
              <PrivacyGauge epsilon={privacyBudget.epsilon} interpretation={privacyBudget.interpretation} />
              <div className="text-xs text-slate-500 font-mono">
                <p className="mb-1">Cumulative ε after round {currentRound}</p>
                <p className="text-slate-600">Budget limit reference: ε = 10</p>
                <p className="text-slate-600 mt-1">{privacyBudget.interpretation}</p>
              </div>
            </div>
          )}

          {/* Raw log */}
          {logs.length > 0 && (
            <div className="card overflow-hidden">
              <button
                id="toggle-log-btn"
                onClick={() => setLogsOpen(o => !o)}
                className="w-full flex items-center justify-between px-4 py-3 text-xs font-mono text-slate-500 hover:text-slate-300 transition-colors"
              >
                <span>Raw Log ({logs.length} lines)</span>
                <span>{logsOpen ? '▲' : '▼'}</span>
              </button>
              {logsOpen && (
                <div className="log-panel rounded-none border-t border-base-700">
                  {logs.map((line, i) => <div key={i}>{line}</div>)}
                  <div ref={logEndRef} />
                </div>
              )}
            </div>
          )}

          {/* Completion summary */}
          {runState === 'done' && result && (
            <div className="card border-emerald-400/20 p-6 space-y-5 animate-slide-up">
              <div className="flex items-center justify-between flex-wrap gap-3">
                <h3 className="text-emerald-400 font-semibold">Run Complete</h3>
                <button
                  id="download-results-btn"
                  onClick={handleDownload}
                  className="flex items-center gap-2 text-xs font-mono text-slate-400 border border-slate-600 rounded px-3 py-1.5 hover:text-white hover:border-slate-400 transition-colors"
                >
                  ↓ Download fl_results.json
                </button>
              </div>

              {/* Final aggregate */}
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
                {Object.entries(result.final_aggregate).map(([k, v]) => (
                  <div key={k} className="text-center">
                    <p className="text-xs text-slate-500 uppercase tracking-widest mb-1">{k.replace('_', ' ')}</p>
                    <p className="metric-value text-xl font-bold text-white">
                      {typeof v === 'number' ? v.toFixed(4) : v}
                    </p>
                  </div>
                ))}
              </div>

              {/* Per-bank table */}
              <div className="overflow-x-auto">
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>Bank</th>
                      <th>AUC</th>
                      <th>F1</th>
                      <th>FNR</th>
                      <th>FPR</th>
                      <th>Accuracy</th>
                      <th>Samples</th>
                    </tr>
                  </thead>
                  <tbody>
                    {BANK_IDS.map(bid => {
                      const r = result.final_per_bank[bid];
                      return r ? (
                        <tr key={bid}>
                          <td style={{ color: BANK_COLORS[bid] }}>{BANK_LABELS[bid]}</td>
                          <td>{r.auc_roc?.toFixed(4)}</td>
                          <td>{r.f1_score?.toFixed(4)}</td>
                          <td>{r.false_negative_rate?.toFixed(4)}</td>
                          <td>{r.false_positive_rate?.toFixed(4)}</td>
                          <td>{r.accuracy?.toFixed(4)}</td>
                          <td>{r.num_test_samples?.toLocaleString()}</td>
                        </tr>
                      ) : null;
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* Idle state */}
          {runState === 'idle' && (
            <div className="card p-8 flex flex-col items-center justify-center text-center gap-3 border-dashed">
              <p className="text-slate-600 text-sm font-mono">Configure a run on the left and click Start Run.</p>
              <p className="text-slate-700 text-xs">Metrics will appear here round-by-round as training progresses.</p>
            </div>
          )}
        </div>
      </div>
    </main>
  );
}
