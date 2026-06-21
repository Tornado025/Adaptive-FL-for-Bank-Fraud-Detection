
interface PrivacyGaugeProps {
  epsilon: number;
  /** Reference maximum epsilon (default 10) */
  maxEpsilon?: number;
  interpretation?: string;
}

export default function PrivacyGauge({ epsilon, maxEpsilon = 10, interpretation }: PrivacyGaugeProps) {
  const pct = Math.min(epsilon / maxEpsilon, 1);
  const radius = 42;
  const circumference = 2 * Math.PI * radius;
  // We use a 240° arc
  const arcLength = circumference * (240 / 360);

  const color =
    pct < 0.3 ? '#10b981' :   // green — strong privacy
    pct < 0.6 ? '#f59e0b' :   // amber — moderate
    '#ef4444';                  // red — weak

  return (
    <div className="card p-4 flex flex-col items-center gap-2">
      <p className="text-xs font-mono uppercase tracking-widest text-slate-500 mb-1">Privacy Budget (ε)</p>

      <div className="relative w-28 h-24">
        <svg viewBox="0 0 100 85" className="w-full h-full" aria-label={`Privacy epsilon: ${epsilon.toFixed(2)}`}>
          {/* Background arc */}
          <path
            d="M 10 75 A 40 40 0 1 1 90 75"
            fill="none"
            stroke="#1c2333"
            strokeWidth="8"
            strokeLinecap="round"
          />
          {/* Filled arc */}
          <path
            d="M 10 75 A 40 40 0 1 1 90 75"
            fill="none"
            stroke={color}
            strokeWidth="8"
            strokeLinecap="round"
            strokeDasharray={`${arcLength * pct} ${arcLength * (1 - pct) + circumference * (120 / 360)}`}
            strokeDashoffset="0"
            style={{ transition: 'stroke-dasharray 0.6s ease, stroke 0.4s ease' }}
          />
        </svg>

        {/* Epsilon value */}
        <div className="absolute inset-0 flex flex-col items-center justify-end pb-3">
          <span className="metric-value text-2xl font-bold" style={{ color }}>
            {epsilon.toFixed(2)}
          </span>
        </div>
      </div>

      <div className="text-center">
        <p className="text-xs font-mono text-slate-400">
          ε = {epsilon.toFixed(4)} / δ = 1e-5
        </p>
        {interpretation && (
          <p className="text-xs text-slate-500 mt-1 max-w-[160px] text-center">
            {interpretation}
          </p>
        )}
      </div>
    </div>
  );
}
