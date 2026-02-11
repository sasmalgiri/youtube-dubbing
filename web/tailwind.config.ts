import type { Config } from 'tailwindcss';

export default {
    content: [
        './src/**/*.{ts,tsx}',
    ],
    theme: {
        extend: {
            colors: {
                background: '#0a0a0f',
                card: {
                    DEFAULT: '#1a1a2e',
                    hover: '#222244',
                },
                border: '#2d2d44',
                primary: {
                    DEFAULT: '#7c3aed',
                    hover: '#6d28d9',
                    light: '#a78bfa',
                },
                accent: {
                    DEFAULT: '#06b6d4',
                    light: '#67e8f9',
                },
                success: '#22c55e',
                error: '#ef4444',
                warning: '#f59e0b',
                text: {
                    primary: '#e2e8f0',
                    secondary: '#94a3b8',
                    muted: '#64748b',
                },
            },
            fontFamily: {
                sans: ['Inter', 'system-ui', 'sans-serif'],
            },
            animation: {
                'pulse-glow': 'pulse-glow 2s ease-in-out infinite',
                'slide-up': 'slide-up 0.3s ease-out',
            },
            keyframes: {
                'pulse-glow': {
                    '0%, 100%': { boxShadow: '0 0 5px rgba(124, 58, 237, 0.3)' },
                    '50%': { boxShadow: '0 0 20px rgba(124, 58, 237, 0.6)' },
                },
                'slide-up': {
                    '0%': { transform: 'translateY(10px)', opacity: '0' },
                    '100%': { transform: 'translateY(0)', opacity: '1' },
                },
            },
        },
    },
    plugins: [],
} satisfies Config;
