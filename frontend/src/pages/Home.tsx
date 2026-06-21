import { Link } from 'react-router-dom';

const TECHNIQUE_CARDS = [
  {
    title: 'Non-IID Data',
    tag: 'FedProx',
    color: '#3b82f6',
    body: 'Each bank sees different transaction patterns. A proximal term anchors local training to prevent models from drifting apart.',
  },
  {
    title: 'Privacy Regulation',
    tag: 'Differential Privacy',
    color: '#06b6d4',
    body: 'GDPR and banking secrecy laws prohibit sharing raw data. Gaussian noise on weight updates provides formal (ε, δ)-DP guarantees.',
  },
  {
    title: 'Unreliable Participants',
    tag: 'Reliability Scoring',
    color: '#10b981',
    body: 'Some banks may have stale data or poor signal. Each update is weighted by validation AUC, not just dataset size.',
  },
  {
    title: 'Conflicting Updates',
    tag: 'Conflict Resolution',
    color: '#f59e0b',
    body: 'Updates that diverge too far from the consensus direction are penalised, preventing a single outlier bank from corrupting the global model.',
  },
];

export default function Home() {
  return (
    <main className="max-w-6xl mx-auto px-6 py-12 space-y-16 animate-fade-in">

      {/* ── Hero ─────────────────────────────────────────────────────────── */}
      <section className="space-y-6">
        <div className="inline-flex items-center gap-2 text-xs font-mono text-emerald-500 bg-emerald-500/10 border border-emerald-500/20 rounded-full px-3 py-1 mb-2">
          <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
          Adaptive Federated Learning · Fraud Detection
        </div>
        <h1 className="text-4xl sm:text-5xl md:text-6xl font-extrabold text-white leading-tight max-w-4xl tracking-tight">
          Banks can't share fraud data —<br />
          <span className="text-emerald-400">so fraud hides across silos.</span>
        </h1>
        <p className="text-slate-400 text-base sm:text-lg max-w-2xl leading-relaxed mt-2">
          Fraudsters exploit the boundaries between financial institutions. A stolen credit card might be tested at one bank and maxed out at another. However, strict privacy regulations (like GDPR) and competitive barriers prevent banks from pooling their transaction records to train a unified, highly accurate fraud detection model.
        </p>
        <p className="text-slate-400 text-base sm:text-lg max-w-3xl leading-relaxed">
          Federated learning offers a solution. It allows multiple banks to collaboratively train a single global fraud-detection model without ever exposing their underlying private datasets. Each institution's data stays securely on-premise. Only encrypted model weight updates travel the wire.
        </p>
        <Link
          to="/playground"
          className="inline-flex items-center gap-2 mt-4 px-6 py-3 rounded-lg bg-emerald-500 text-black font-semibold hover:bg-emerald-400 transition-colors"
        >
          Run a live training round
          <svg width="16" height="16" viewBox="0 0 14 14" fill="none">
            <path d="M3 7h8M7 3l4 4-4 4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        </Link>
      </section>

      {/* ── Problem Statement Details ────────────────────────────────────── */}
      <section className="grid md:grid-cols-2 gap-8 items-center border-t border-base-800 pt-16">
        <div className="space-y-6">
          <h2 className="text-3xl sm:text-4xl font-bold text-white tracking-tight">The Challenge: Non-IID Data</h2>
          <p className="text-slate-400 leading-relaxed text-base sm:text-lg">
            Standard Federated Learning struggles in the real world because real-world data is notoriously messy. In banking, data is <strong>Non-IID</strong> (Not Independent and Identically Distributed).
          </p>
          <ul className="space-y-4 text-slate-400">
            <li className="flex items-start gap-3">
              <span className="text-amber-500 mt-1">●</span>
              <span><strong>Varying Sizes:</strong> A global tier-1 bank might have 100x the transaction volume of a regional credit union.</span>
            </li>
            <li className="flex items-start gap-3">
              <span className="text-amber-500 mt-1">●</span>
              <span><strong>Different Profiles:</strong> An e-commerce focused bank sees entirely different fraud patterns (card-not-present) compared to a domestic retail bank (point-of-sale skimming).</span>
            </li>
            <li className="flex items-start gap-3">
              <span className="text-amber-500 mt-1">●</span>
              <span><strong>Imbalanced Fraud Rates:</strong> Fraud prevalence fluctuates wildly across regions and sectors.</span>
            </li>
          </ul>
          <p className="text-slate-400 leading-relaxed">
            If we naively average the models (like standard FedAvg), the disparate updates will conflict, causing the global model to drift, diverge, and ultimately fail. We need an <em>adaptive</em> approach.
          </p>
        </div>
        <div className="card p-8 bg-base-900 border-base-800">
          <div className="space-y-4">
            <div className="flex items-center justify-between border-b border-base-800 pb-4">
              <span className="text-slate-300 font-mono text-sm">Standard FL (FedAvg)</span>
              <span className="text-red-400 font-bold text-sm">Fails on Non-IID</span>
            </div>
            <div className="flex items-center justify-between border-b border-base-800 pb-4">
              <span className="text-slate-300 font-mono text-sm">Local Only Training</span>
              <span className="text-amber-400 font-bold text-sm">Limited Insight</span>
            </div>
            <div className="flex items-center justify-between pt-2">
              <span className="text-emerald-400 font-mono text-sm font-bold">Adaptive Federated Learning</span>
              <span className="text-emerald-400 font-bold text-sm">Global Synergy</span>
            </div>
          </div>
        </div>
      </section>

      {/* ── Why this is hard ─────────────────────────────────────────────── */}
      <section className="pt-8">
        <h2 className="text-3xl sm:text-4xl font-bold text-white mb-8 tracking-tight">How We Solve It</h2>
        <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-6">
          {TECHNIQUE_CARDS.map((c) => (
            <div key={c.title} className="card card-hover p-6 relative overflow-hidden group">
              <div
                className="absolute top-0 left-0 w-full h-1 opacity-70"
                style={{ background: c.color }}
              />
              <span
                className="inline-block text-[10px] font-mono font-bold tracking-widest uppercase mb-4 px-2 py-0.5 rounded-full"
                style={{ color: c.color, background: `${c.color}18`, border: `1px solid ${c.color}30` }}
              >
                {c.tag}
              </span>
              <h3 className="text-base font-semibold text-white mb-3">{c.title}</h3>
              <p className="text-sm text-slate-400 leading-relaxed">{c.body}</p>
            </div>
          ))}
        </div>
      </section>

      {/* ── CTA ──────────────────────────────────────────────────────────── */}
      <section className="border border-emerald-500/20 bg-emerald-500/5 rounded-2xl p-10 flex flex-col sm:flex-row items-center justify-between gap-8 mt-12">
        <div>
          <h2 className="text-2xl sm:text-3xl font-bold text-white mb-3 tracking-tight">Ready to see it run?</h2>
          <p className="text-base sm:text-lg text-slate-400 max-w-xl">
            Head to the Playground to configure differential privacy, choose your aggregation method, and watch the collaborative training complete live, round by round.
          </p>
        </div>
        <Link
          to="/playground"
          className="shrink-0 inline-flex items-center gap-2 px-8 py-4 rounded-lg bg-emerald-500 text-black font-bold hover:bg-emerald-400 transition-colors shadow-[0_0_20px_rgba(16,185,129,0.3)]"
        >
          Open Playground
        </Link>
      </section>

    </main>
  );
}

