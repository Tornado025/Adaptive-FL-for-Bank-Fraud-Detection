import React from 'react';

interface StatCardProps {
  label: string;
  value: string | number;
  unit?: string;
  sample?: boolean;
  accent?: boolean;
  icon?: React.ReactNode;
}

export default function StatCard({ label, value, unit, sample, accent, icon }: StatCardProps) {
  return (
    <div
      className={`card card-hover p-5 relative overflow-hidden ${accent ? 'border-amber-500/30' : ''}`}
    >
      {/* Subtle gradient orb */}
      {accent && (
        <div
          className="absolute -top-6 -right-6 w-24 h-24 rounded-full pointer-events-none"
          style={{ background: 'radial-gradient(circle, rgba(245,158,11,0.08) 0%, transparent 70%)' }}
        />
      )}

      {sample && (
        <span className="absolute top-3 right-3 text-[10px] font-mono font-semibold tracking-widest uppercase text-slate-600 border border-slate-700 rounded px-1.5 py-0.5">
          SAMPLE
        </span>
      )}

      <div className="flex items-start gap-3">
        {icon && (
          <div className="mt-0.5 text-slate-500 shrink-0">{icon}</div>
        )}
        <div>
          <p className="text-xs font-medium text-slate-500 uppercase tracking-widest mb-1">{label}</p>
          <div className="flex items-baseline gap-1.5">
            <span className={`metric-value text-3xl font-bold ${accent ? 'text-amber-400' : 'text-white'}`}>
              {value}
            </span>
            {unit && <span className="text-sm text-slate-500 font-mono">{unit}</span>}
          </div>
        </div>
      </div>
    </div>
  );
}
