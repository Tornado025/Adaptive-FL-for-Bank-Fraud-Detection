import { useEffect, useRef, useState } from 'react';

interface SplashProps {
  onDone?: () => void;
  /** If true, loops indefinitely as a loading overlay (no auto-dismiss) */
  loop?: boolean;
}

const BANK_COLORS = ['#3b82f6', '#8b5cf6', '#06b6d4', '#ec4899'];

// Bank positions (diamond layout around centre server)
const BANKS = [
  { cx: 100, cy: 60 },   // top
  { cx: 220, cy: 140 },  // right
  { cx: 100, cy: 220 },  // bottom
  { cx: -20, cy: 140 },  // left
];
const SERVER = { cx: 100, cy: 140 };

export default function Splash({ onDone, loop = false }: SplashProps) {
  const [visible, setVisible] = useState(true);
  const dismissed = useRef(false);

  useEffect(() => {
    if (loop) return;
    const t = setTimeout(() => {
      if (!dismissed.current) {
        dismissed.current = true;
        setVisible(false);
        onDone?.();
      }
    }, 2000);
    return () => clearTimeout(t);
  }, [loop, onDone]);

  if (!visible) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex flex-col items-center justify-center bg-base-900"
      style={{ backgroundColor: '#0d1117' }}
    >
      {/* Animated federation graph */}
      <svg
        viewBox="-40 20 280 240"
        width="240"
        height="200"
        aria-hidden="true"
      >
        {/* Flow lines from banks to server */}
        {BANKS.map((b, i) => (
          <line
            key={`line-${i}`}
            x1={b.cx}
            y1={b.cy}
            x2={SERVER.cx}
            y2={SERVER.cy}
            stroke={BANK_COLORS[i]}
            strokeWidth="1.5"
            className="flow-line"
            style={{ animationDelay: `${i * 0.4}s` }}
          />
        ))}

        {/* Bank nodes */}
        {BANKS.map((b, i) => (
          <g key={`bank-${i}`}>
            <circle
              cx={b.cx}
              cy={b.cy}
              r="16"
              fill={BANK_COLORS[i]}
              fillOpacity="0.12"
              stroke={BANK_COLORS[i]}
              strokeWidth="1.5"
              className="node-pulse"
              style={{ animationDelay: `${i * 0.35}s` }}
            />
            <circle
              cx={b.cx}
              cy={b.cy}
              r="7"
              fill={BANK_COLORS[i]}
              opacity="0.9"
            />
          </g>
        ))}

        {/* Central server node */}
        <circle cx={SERVER.cx} cy={SERVER.cy} r="22" fill="#f59e0b" fillOpacity="0.1" stroke="#f59e0b" strokeWidth="1.5" />
        <circle cx={SERVER.cx} cy={SERVER.cy} r="10" fill="#f59e0b" />
      </svg>

      {/* Labels */}
      <div className="mt-6 text-center">
        <p className="text-xs font-mono tracking-widest text-slate-500 uppercase mb-2">
          Adaptive Federated Learning
        </p>
        <h1 className="text-xl font-semibold text-white tracking-tight">
          Bank Fraud Detection
        </h1>
      </div>

      {loop && (
        <p className="mt-4 text-xs font-mono text-amber-500 animate-pulse">
          Connecting to backend…
        </p>
      )}
    </div>
  );
}
