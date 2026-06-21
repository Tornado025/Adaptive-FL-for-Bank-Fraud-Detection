import { useState } from 'react';

const TECH_STACK = [
  { name: 'PyTorch ≥ 2.0',    role: 'Local training, FedProx loss, model architecture' },
  { name: 'FastAPI + Uvicorn', role: 'REST API wrapper — SSE streaming, CORS, run management' },
  { name: 'React 18 + Vite',  role: 'Frontend framework and build toolchain' },
  { name: 'TypeScript',       role: 'Type-safe API layer and component props' },
  { name: 'Tailwind CSS',     role: 'Utility-first dark-mode styling' },
  { name: 'Recharts',         role: 'Composable charting for live metrics' },
  { name: 'SQLite',           role: 'Per-bank transaction databases (data/ directory)' },
  { name: 'NumPy + Scikit-learn', role: 'DP calculations, AUC/F1 metrics' },
];

export default function About() {
  return (
    <main className="max-w-4xl mx-auto px-6 py-12 space-y-12 animate-fade-in">

      <div>
        <h1 className="text-4xl sm:text-5xl font-extrabold text-white mb-6 tracking-tight">About</h1>
        <p className="text-slate-400 text-base sm:text-lg max-w-3xl leading-relaxed">
          This project implements an Adaptive Federated Learning system for fraud detection across
          four simulated banks. The central problem it addresses: financial institutions hold rich
          fraud signal in their transaction histories, but privacy regulations and competitive concerns
          prevent them from pooling that data into a single dataset.
        </p>
        <p className="text-slate-400 text-base sm:text-lg max-w-3xl leading-relaxed mt-4">
          The system uses a Split Neural Network architecture — the shared base layers act as a
          collaborative feature extractor, while each bank's personalized classification head remains
          entirely local. Differential Privacy, FedProx regularization, reliability-based weighting,
          and conflict resolution are layered together to handle the challenges of heterogeneous,
          non-IID data across institutions.
        </p>
      </div>

      {/* Tech stack */}
      <section>
        <h2 className="text-sm font-mono uppercase tracking-widest text-slate-500 mb-4">Technology Stack</h2>
        <div className="card overflow-hidden">
          <table className="data-table">
            <thead>
              <tr>
                <th>Component</th>
                <th>Role</th>
              </tr>
            </thead>
            <tbody>
              {TECH_STACK.map(t => (
                <tr key={t.name}>
                  <td className="text-amber-400">{t.name}</td>
                  <td className="text-slate-400 font-sans text-xs">{t.role}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      {/* Developers */}
      <section>
        <h2 className="text-sm font-mono uppercase tracking-widest text-slate-500 mb-4">Developers</h2>
        <div className="grid sm:grid-cols-2 gap-6">
          {/* Rishon */}
          <div className="card p-6 border-emerald-500/20 bg-emerald-500/5 hover:border-emerald-500/40 transition-colors flex items-center gap-5">
            <div className="w-16 h-16 rounded-full bg-emerald-500/20 border-2 border-emerald-500/50 flex items-center justify-center text-emerald-400 text-xl font-bold uppercase tracking-widest shrink-0 shadow-[0_0_15px_rgba(16,185,129,0.2)]">
              R
            </div>
            <div>
              <h3 className="text-lg font-bold text-white">Rishon</h3>
              <p className="text-sm text-emerald-400 font-mono mb-2">Lead Developer</p>
              <p className="text-xs text-slate-400 leading-relaxed">
                Made the FedProx, DP mechanisms, and frontend architecture.
              </p>
            </div>
          </div>

          {/* Ajay */}
          <div className="card p-6 border-amber-500/20 bg-amber-500/5 hover:border-amber-500/40 transition-colors flex items-center gap-5">
            <div className="w-16 h-16 rounded-full bg-amber-500/20 border-2 border-amber-500/50 flex items-center justify-center text-amber-400 text-xl font-bold uppercase tracking-widest shrink-0 shadow-[0_0_15px_rgba(245,158,11,0.2)]">
              A
            </div>
            <div>
              <h3 className="text-lg font-bold text-white">Ajay Girish</h3>
              <p className="text-sm text-amber-400 font-mono mb-2">Researcher</p>
              <p className="text-xs text-slate-400 leading-relaxed">
                Lead research on adaptive federated learning and non-IID data distribution.
              </p>
            </div>
          </div>

          {/* Suhayb */}
          <div className="card p-6 border-cyan-500/20 bg-cyan-500/5 hover:border-cyan-500/40 transition-colors flex items-center gap-5">
            <div className="w-16 h-16 rounded-full bg-cyan-500/20 border-2 border-cyan-500/50 flex items-center justify-center text-cyan-400 text-xl font-bold uppercase tracking-widest shrink-0 shadow-[0_0_15px_rgba(6,182,212,0.2)]">
              S
            </div>
            <div>
              <h3 className="text-lg font-bold text-white">Mohammed Suhayb</h3>
              <p className="text-sm text-cyan-400 font-mono mb-2">Data Processor</p>
              <p className="text-xs text-slate-400 leading-relaxed">
                Processed, formatted, and simulated the partitioned bank transaction databases.
              </p>
            </div>
          </div>

          {/* Jerome */}
          <div className="card p-6 border-purple-500/20 bg-purple-500/5 hover:border-purple-500/40 transition-colors flex items-center gap-5">
            <div className="w-16 h-16 rounded-full bg-purple-500/20 border-2 border-purple-500/50 flex items-center justify-center text-purple-400 text-xl font-bold uppercase tracking-widest shrink-0 shadow-[0_0_15px_rgba(168,85,247,0.2)]">
              J
            </div>
            <div>
              <h3 className="text-lg font-bold text-white">Jerome Antony</h3>
              <p className="text-sm text-purple-400 font-mono mb-2">Neural Network Architect</p>
              <p className="text-xs text-slate-400 leading-relaxed">
                Designed and trained the underlying Split-NN fraud detection model.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Disclaimer */}
      <section className="card border-slate-700 p-6 space-y-2">
        <h2 className="text-sm font-mono uppercase tracking-widest text-slate-500 mb-3">Disclaimer</h2>
        <p className="text-xs text-slate-500 leading-relaxed">
          This system operates on <strong className="text-slate-400">simulated bank databases</strong> generated
          for research and coursework purposes. No real financial institution, customer, or transaction data is
          used or implied. The simulated banks (bank_a through bank_d) are synthetic environments designed to
          exhibit non-IID data distributions representative of real-world federated learning challenges.
        </p>
        <p className="text-xs text-slate-500 leading-relaxed">
          The differential privacy guarantees, model metrics, and aggregation behaviours shown in this
          dashboard are produced by the actual federated learning pipeline, but should be interpreted
          as <strong className="text-slate-400">research outputs</strong>, not production fraud-detection
          capabilities. This is not a production system and should not be deployed in a financial context
          without extensive independent evaluation.
        </p>
        <p className="text-xs text-slate-600 mt-3 font-mono">
          Built as a research prototype · GitHub:{' '}
          <a
            href="https://github.com/Tornado025/Adaptive-FL-for-Bank-Fraud-Detection"
            target="_blank"
            rel="noopener noreferrer"
            className="text-amber-500 hover:underline"
          >
            Tornado025/Adaptive-FL-for-Bank-Fraud-Detection
          </a>
        </p>
      </section>

    </main>
  );
}
