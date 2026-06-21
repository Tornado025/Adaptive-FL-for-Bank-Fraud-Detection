import { useEffect, useState } from 'react';
import { NavLink } from 'react-router-dom';
import { checkHealth } from '../lib/api';

const LINKS = [
  { to: '/',              label: 'Home' },
  { to: '/how-it-works', label: 'How It Works' },
  { to: '/playground',   label: 'Playground' },
  { to: '/about',        label: 'About' },
];

export default function Nav() {
  const [online, setOnline] = useState<boolean | null>(null);

  useEffect(() => {
    let cancelled = false;

    const poll = async () => {
      const ok = await checkHealth();
      if (!cancelled) setOnline(ok);
    };

    poll();
    const interval = setInterval(poll, 10_000);
    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, []);

  return (
    <nav
      className="sticky top-0 z-40 border-b border-base-700"
      style={{ background: 'rgba(13,17,23,0.92)', backdropFilter: 'blur(12px)' }}
    >
      <div className="max-w-7xl mx-auto px-6 h-14 flex items-center gap-8">
        {/* Logo */}
        <div className="flex items-center gap-2 shrink-0">
          <svg width="22" height="22" viewBox="0 0 22 22" aria-hidden="true">
            {/* Mini federation icon */}
            {[
              { cx: 11, cy: 3 },
              { cx: 19, cy: 11 },
              { cx: 11, cy: 19 },
              { cx: 3, cy: 11 },
            ].map((p, i) => (
              <g key={i}>
                <line x1={p.cx} y1={p.cy} x2={11} y2={11} stroke="#f59e0b" strokeWidth="1" opacity="0.6" />
                <circle cx={p.cx} cy={p.cy} r="2.5" fill="#f59e0b" opacity="0.85" />
              </g>
            ))}
            <circle cx={11} cy={11} r="3" fill="#f59e0b" />
          </svg>
          <span className="font-semibold text-sm text-white tracking-tight hidden sm:block">
            Adaptive FL
          </span>
        </div>

        {/* Divider */}
        <div className="h-5 w-px bg-base-700 hidden sm:block" />

        {/* Nav links */}
        <div className="flex items-center gap-1 flex-1">
          {LINKS.map(({ to, label }) => (
            <NavLink
              key={to}
              to={to}
              end={to === '/'}
              className={({ isActive }) =>
                `px-3 py-1.5 rounded text-sm font-medium transition-colors ${
                  isActive
                    ? 'text-amber-400 bg-amber-500/10'
                    : 'text-slate-400 hover:text-white hover:bg-white/5'
                }`
              }
            >
              {label}
            </NavLink>
          ))}
        </div>

        {/* System status pill */}
        <div className="flex items-center gap-2 shrink-0">
          <span
            className="w-2 h-2 rounded-full"
            style={{
              background:
                online === null
                  ? '#64748b'
                  : online
                  ? '#10b981'
                  : '#f59e0b',
              boxShadow:
                online === null
                  ? 'none'
                  : online
                  ? '0 0 6px #10b981'
                  : '0 0 6px #f59e0b',
            }}
          />
          <span className="text-xs font-mono text-slate-500">
            {online === null ? 'checking…' : online ? 'API online' : 'API offline'}
          </span>
        </div>
      </div>
    </nav>
  );
}
