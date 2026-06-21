/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // Base palette
        base: {
          950: '#080c12',
          900: '#0d1117',
          800: '#161b22',
          700: '#1c2333',
          600: '#2d3748',
          500: '#3d4f63',
        },
        // Accent — amber, reserved for live/active states only
        amber: {
          400: '#fbbf24',
          500: '#f59e0b',
          600: '#d97706',
        },
        // Success / online
        emerald: {
          400: '#34d399',
          500: '#10b981',
        },
        // Muted text
        slate: {
          400: '#94a3b8',
          500: '#64748b',
          600: '#475569',
        },
        // Bank node colors
        bank: {
          a: '#3b82f6',  // blue
          b: '#8b5cf6',  // violet
          c: '#06b6d4',  // cyan
          d: '#ec4899',  // pink
        },
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'Fira Code', 'monospace'],
      },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'fade-in': 'fadeIn 0.4s ease-out forwards',
        'slide-up': 'slideUp 0.3s ease-out forwards',
        'flow': 'flow 2s ease-in-out infinite',
      },
      keyframes: {
        fadeIn: {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
        slideUp: {
          '0%': { opacity: '0', transform: 'translateY(12px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        flow: {
          '0%, 100%': { strokeDashoffset: '60', opacity: '0.3' },
          '50%': { strokeDashoffset: '0', opacity: '1' },
        },
      },
    },
  },
  plugins: [],
}
