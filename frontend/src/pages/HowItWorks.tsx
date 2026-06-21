import { useState, Fragment } from 'react';
import {
  LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid,
} from 'recharts';

// ── Bank Profiles ─────────────────────────────────────────────────────────────
const BANK_PROFILES = [
  { bank: 'Bank A', type: 'Domestic Retail', rows: '50,000', fraud: '1.94%', diff: 'US-only addresses, low transaction amounts, mostly consumer products' },
  { bank: 'Bank B', type: 'International Corporate', rows: '10,000', fraud: '13.46%', diff: 'Non-US addresses, high-value transactions only (top 40%)' },
  { bank: 'Bank C', type: 'HighFraud Region', rows: '47,326', fraud: '15.00%', diff: 'Deliberately oversampled fraud to 15% (simulates a high-risk region)' },
  { bank: 'Bank D', type: 'Ecommerce (Card Not Present)', rows: '50,000', fraud: '2.79%', diff: 'Desktop-heavy, Visa+credit oversampled, high feature variance' },
];

// ── Step data ─────────────────────────────────────────────────────────────────
const STEPS = [
  {
    id: 'broadcast',
    label: 'Broadcast',
    icon: '↓',
    summary: 'Server distributes current global base-layer weights to all 4 banks.',
    detail: `The central server holds the shared "base layers" of the model — the feature extractor trained collaboratively across all banks. At the start of each round, it serialises this weight dict and sends it to every participating client. No transaction data is ever stored on or sent to the server.`,
  },
  {
    id: 'local-opt',
    label: 'Local Optimization',
    icon: '⚙',
    summary: 'Each bank trains on its private data using FedProx for 1–10 epochs.',
    detail: `Each FLClient loads the global base weights, attaches its private personalised head, then runs FedProx training. The FedProx objective adds a proximal term μ‖w − w_global‖² to the loss, anchoring local training close to the global consensus. This mitigates client drift on non-IID data. After training, only the updated base-layer weights — never raw transactions — are packaged for upload.`,
  },
  {
    id: 'extraction',
    label: 'Weight Extraction',
    icon: '📦',
    summary: 'Clients extract base-layer weights and attach validation metadata.',
    detail: `weight_extractor.py pulls only the base_layers.* parameters from the trained model. The package also contains metadata: val_auc (validation AUC on the bank's held-out set), val_loss, proximal_term magnitude, and sample count. These metadata fields drive reliability scoring on the server — but they are computed locally on private data.`,
  },
  {
    id: 'aggregation',
    label: 'Server Aggregation',
    icon: '⚖',
    summary: 'Server applies DP noise, reliability scoring, conflict resolution, then FedAvg.',
    detail: `The full "custom" pipeline runs in order:\n1. DP clipping + Gaussian noise (privacy)\n2. Reliability scoring: weight each bank by val_auc\n3. Conflict penalties: cosine similarity between each update and the consensus direction\n4. FedProx drift weights: penalise large proximal terms\n5. Weighted average using the combined effective_weight_k\nA minimum weight floor (5%) ensures no bank is silenced by a single bad round.`,
  },
  {
    id: 'evaluation',
    label: 'Global Evaluation',
    icon: '📊',
    summary: 'The new model is evaluated on every bank\'s validation split.',
    detail: `evaluate_global_model() injects the new global base weights into each bank's local model and runs inference on the held-out validation set. It reports weighted_auc (weighted by sample count), mean_f1, mean_fnr (False Negative Rate — missed fraud), and mean_fpr (False Positive Rate — false alarms). These metrics are streamed to the dashboard as each round completes.`,
  },
];

// ── Aggregation methods table ─────────────────────────────────────────────────
const METHODS = [
  {
    name: 'fedavg',
    adds: 'Sample-weighted average of all client updates.',
    tradeoff: 'Simple and fast, but ignores data quality differences between banks.',
  },
  {
    name: 'fedprox',
    adds: 'FedAvg + proximal term to limit client drift on non-IID data.',
    tradeoff: 'More stable convergence; small overhead from μ tuning.',
  },
  {
    name: 'dp_fedavg',
    adds: 'FedAvg + L2 clipping + Gaussian noise for formal (ε,δ)-DP.',
    tradeoff: 'Privacy guaranteed; accuracy decreases with noise multiplier.',
  },
  {
    name: 'custom',
    adds: 'DP + FedProx + reliability scoring + conflict resolution combined.',
    tradeoff: 'Best accuracy and full privacy; highest compute per round.',
  },
];

// ── DP illustrative curve (precomputed, noise_multiplier → approx ε at 10 rounds) ─
function computeEpsilon(nm: number): number {
  if (nm === 0) return 999;
  return (1 / nm) * Math.sqrt(2 * 10 * Math.log(1 / 1e-5));
}

const CustomTooltip = ({ active, payload, label }: any) => {
  if (!active || !payload?.length) return null;
  return (
    <div className="card px-3 py-2 text-xs font-mono">
      <p className="text-slate-400 mb-1">noise_multiplier = {label}</p>
      {payload.map((p: any) => (
        <p key={p.name} style={{ color: p.color }}>
          {p.name}: {p.value === 999 ? 'Infinity' : p.value}
        </p>
      ))}
    </div>
  );
};

export default function HowItWorks() {
  const [nm, setNm] = useState(0);

  const curveData = Array.from({ length: 31 }, (_, i) => {
    const noise = parseFloat((0.0 + i * 0.1).toFixed(1));
    const eps = computeEpsilon(noise);
    return {
      noise: noise.toFixed(1),
      epsilon: parseFloat((eps > 15 ? 15 : eps).toFixed(3)),
      accuracy: parseFloat((0.92 - noise * 0.022).toFixed(3)),
    };
  });

  const rawEps = computeEpsilon(nm);
  const currentEps = rawEps === 999 ? 'Infinity' : rawEps.toFixed(3);

  return (
    <main className="max-w-6xl mx-auto px-6 py-12 space-y-16 animate-fade-in">

      <div>
        <h1 className="text-3xl font-bold text-white mb-2">How It Works</h1>
        <p className="text-slate-400 max-w-2xl">
          A technical walkthrough of the federated learning pipeline — from weight broadcast to global evaluation.
        </p>
      </div>

      {/* ── Simulated Banks ────────────────────────────────────────────── */}
      <section>
        <h2 className="text-sm font-mono uppercase tracking-widest text-slate-500 mb-4">Simulated Banks Specifications</h2>
        <div className="card overflow-hidden">
          <table className="data-table">
            <thead>
              <tr>
                <th>Bank</th>
                <th>Profile</th>
                <th>Rows</th>
                <th>Fraud Rate</th>
                <th>What Makes It Different</th>
              </tr>
            </thead>
            <tbody>
              {BANK_PROFILES.map((b) => (
                <tr key={b.bank}>
                  <td><span className="text-amber-400 font-bold">{b.bank}</span></td>
                  <td className="text-slate-300 font-sans text-xs">{b.type}</td>
                  <td className="text-slate-300 font-sans text-xs">{b.rows}</td>
                  <td className="text-red-400 font-sans text-xs font-semibold">{b.fraud}</td>
                  <td className="text-slate-500 font-sans text-xs">{b.diff}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      {/* ── Split-NN Architecture HTML Layout ────────────────────────────── */}
      <section>
        <h2 className="text-sm font-mono uppercase tracking-widest text-slate-500 mb-6">Split-NN Architecture</h2>
        <div className="card p-8 flex flex-col items-center gap-6">
          {/* Server */}
          <div className="w-full max-w-lg border border-amber-500/30 bg-amber-500/10 shadow-[0_0_20px_rgba(245,158,11,0.05)] rounded-xl p-4 text-center relative">
            <h3 className="text-amber-400 font-bold font-mono tracking-widest text-sm mb-1">↑ SERVER AGGREGATOR ↑</h3>
            <p className="text-amber-400/80 text-xs">Securely averages the base-layer weights</p>
          </div>
          
          {/* Up/Down connecting lines */}
          <div className="flex items-center justify-center gap-16 py-2">
            <div className="flex flex-col items-center text-amber-500/40">
              <span className="text-2xl leading-none">↕</span>
              <span className="text-[10px] font-mono tracking-widest uppercase mt-2">Global Weights</span>
            </div>
          </div>

          {/* 4 Banks container */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6 w-full">
            {BANK_PROFILES.map((b, i) => {
              const colors = [
                'border-blue-500/30 bg-blue-500/10 text-blue-400',
                'border-purple-500/30 bg-purple-500/10 text-purple-400',
                'border-cyan-500/30 bg-cyan-500/10 text-cyan-400',
                'border-pink-500/30 bg-pink-500/10 text-pink-400'
              ];
              const headColor = colors[i];
              
              return (
                <div key={b.bank} className="flex flex-col gap-3">
                  {/* Bank Label */}
                  <div className="text-center mb-2">
                    <div className="text-slate-300 font-bold font-mono">{b.bank}</div>
                    <div className="text-slate-500 text-[10px] uppercase tracking-wider">{b.type}</div>
                  </div>
                  
                  {/* Base Layers (Shared) */}
                  <div className="border-2 border-amber-500/40 border-dashed bg-amber-500/5 rounded-lg p-5 text-center">
                    <div className="text-amber-400 font-bold text-sm mb-1">BASE LAYERS</div>
                    <div className="text-amber-400/60 text-xs">Federated / Shared</div>
                  </div>
                  
                  {/* Arrow down to private head */}
                  <div className="flex justify-center text-slate-600/50 -my-1">
                    <span className="text-xl leading-none">↓</span>
                  </div>
                  
                  {/* Local Head (Private) */}
                  <div className={`border ${headColor} rounded-lg p-5 text-center relative`}>
                    <div className="absolute -top-3.5 left-1/2 -translate-x-1/2 text-lg bg-base-900 rounded-full w-6 h-6 flex items-center justify-center">🔒</div>
                    <div className="font-bold text-sm mt-1 mb-1">PERS. HEAD</div>
                    <div className="opacity-70 text-xs">Private / Local</div>
                  </div>
                </div>
              )
            })}
          </div>
          
          {/* Legend */}
          <div className="flex flex-wrap items-center justify-center gap-8 mt-10 p-4 border border-base-800 rounded-lg w-full max-w-2xl text-xs">
            <div className="flex items-center gap-3">
              <div className="w-5 h-5 border-2 border-amber-500/40 border-dashed bg-amber-500/5 rounded"></div>
              <span className="text-slate-400">Federated (Shared Base) — travels wire</span>
            </div>
            <div className="flex items-center gap-3">
              <div className="w-5 h-5 border border-blue-500/30 bg-blue-500/10 rounded"></div>
              <span className="text-slate-400">Private (Local Head) — never leaves bank</span>
            </div>
          </div>
        </div>
      </section>

      {/* ── Operational Workflow ───────────────────────────────────────── */}
      <section>
        <h2 className="text-sm font-mono uppercase tracking-widest text-slate-500 mb-6">Operational Workflow</h2>

        <div className="space-y-4">
          {STEPS.map((step, idx) => (
            <div key={step.id} className="card p-6 border-base-800 flex flex-col md:flex-row md:items-start gap-4 hover:border-amber-500/30 transition-colors">
              <div className="shrink-0 w-12 h-12 rounded-full border-2 border-amber-500/30 bg-amber-500/10 flex items-center justify-center text-amber-400 text-xl font-bold">
                {idx + 1}
              </div>
              <div className="space-y-2">
                <h3 className="text-lg font-bold text-white flex items-center gap-2">
                  <span>{step.label}</span>
                  <span className="text-slate-500 text-sm font-normal">{step.icon}</span>
                </h3>
                <p className="text-amber-400/80 text-sm font-medium">{step.summary}</p>
                <p className="text-sm text-slate-400 leading-relaxed whitespace-pre-line pt-2">
                  {step.detail}
                </p>
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* ── Aggregation method table ───────────────────────────────────── */}
      <section>
        <h2 className="text-sm font-mono uppercase tracking-widest text-slate-500 mb-4">Aggregation Methods</h2>
        <div className="card overflow-hidden">
          <table className="data-table">
            <thead>
              <tr>
                <th>Method</th>
                <th>What It Adds</th>
                <th>Key Tradeoff</th>
              </tr>
            </thead>
            <tbody>
              {METHODS.map((m) => (
                <tr key={m.name}>
                  <td>
                    <span className="text-amber-400 font-bold">{m.name}</span>
                  </td>
                  <td className="text-slate-300 font-sans text-xs">{m.adds}</td>
                  <td className="text-slate-500 font-sans text-xs">{m.tradeoff}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      {/* ── DP slider + curve ──────────────────────────────────────────── */}
      <section>
        <h2 className="text-sm font-mono uppercase tracking-widest text-slate-500 mb-2">
          Differential Privacy — Accuracy vs Privacy Tradeoff
        </h2>
        <p className="text-xs text-slate-600 mb-6 font-mono">
          Illustrative only · precomputed approximation · not a real training run
        </p>

        <div className="card p-6 space-y-6">
          <div>
            <label className="text-sm text-slate-400 font-medium flex items-center justify-between mb-3">
              <span>noise_multiplier</span>
              <span className="font-mono text-amber-400">{nm.toFixed(1)}</span>
            </label>
            <input
              type="range"
              min={0}
              max={3.0}
              step={0.1}
              value={nm}
              onChange={(e) => setNm(parseFloat(e.target.value))}
              aria-label="noise_multiplier slider"
            />
            <div className="flex justify-between text-xs font-mono text-slate-600 mt-1">
              <span>0.0 (no privacy)</span>
              <span>3.0 (strong privacy)</span>
            </div>
          </div>

          <div className="grid sm:grid-cols-3 gap-4 text-center">
            <div>
              <p className="text-xs text-slate-500 mb-1">Privacy ε (10 rounds)</p>
              <p className="metric-value text-2xl text-amber-400 font-bold">{currentEps}</p>
            </div>
            <div>
              <p className="text-xs text-slate-500 mb-1">Est. AUC impact</p>
              <p className="metric-value text-2xl text-white font-bold">
                −{(nm * 0.022).toFixed(3)}
              </p>
            </div>
            <div>
              <p className="text-xs text-slate-500 mb-1">Privacy level</p>
              <p className={`text-lg font-semibold ${
                rawEps === 999 ? 'text-red-500' :
                rawEps < 3 ? 'text-emerald-400' :
                rawEps < 7 ? 'text-amber-400' : 'text-red-400'
              }`}>
                {rawEps === 999 ? 'None' : rawEps < 3 ? 'Strong' : rawEps < 7 ? 'Moderate' : 'Weak'}
              </p>
            </div>
          </div>

          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={curveData} margin={{ left: -10, right: 10, top: 10 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1c2333" />
              <XAxis
                dataKey="noise"
                tick={{ fill: '#64748b', fontSize: 11, fontFamily: 'JetBrains Mono' }}
                axisLine={false}
                tickLine={false}
                label={{ value: 'noise_multiplier', position: 'insideBottom', dy: 10, fill: '#64748b', fontSize: 10 }}
              />
              <YAxis
                yAxisId="eps"
                tick={{ fill: '#f59e0b', fontSize: 11, fontFamily: 'JetBrains Mono' }}
                axisLine={false}
                tickLine={false}
                domain={[0, 15]}
              />
              <YAxis
                yAxisId="acc"
                orientation="right"
                tick={{ fill: '#10b981', fontSize: 11, fontFamily: 'JetBrains Mono' }}
                axisLine={false}
                tickLine={false}
                domain={[0.85, 0.95]}
              />
              <Tooltip content={<CustomTooltip />} />
              {/* Vertical reference line at current nm */}
              <Line yAxisId="eps" type="monotone" dataKey="epsilon" stroke="#f59e0b" strokeWidth={2} dot={false} name="ε (epsilon)" />
              <Line yAxisId="acc" type="monotone" dataKey="accuracy" stroke="#10b981" strokeWidth={2} dot={false} name="Est. AUC" />
            </LineChart>
          </ResponsiveContainer>
          <p className="text-center text-xs text-slate-600 font-mono -mt-4">
            amber = ε (left axis, capped at 15) · green = est. AUC (right axis)
          </p>
        </div>
      </section>

    </main>
  );
}
