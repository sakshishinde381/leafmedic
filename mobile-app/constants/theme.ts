export const colors = {
  primary: '#1F6B4B',
  primaryLight: '#2D8C63',
  primaryDark: '#114A33',
  accent: '#89B24D',
  background: '#EEF4EF',
  surface: '#FFFFFF',
  cardBg: 'rgba(255,255,255,0.96)',
  surfaceSoft: '#F7FBF7',
  text: '#0F2D1F',
  textSecondary: '#446358',
  muted: '#6F8C81',
  error: '#C0524D',
  warning: '#B3822B',
  success: '#2D8C63',
  border: '#CFE2D6',
  borderStrong: '#AACBB9',
  gradientStart: '#0F4F39',
  gradientMid: '#1F6B4B',
  gradientEnd: '#5BAA6A',
};

export const spacing = {
  xs: 4,
  sm: 8,
  md: 16,
  lg: 24,
  xl: 32,
  xxl: 48,
};

export const borderRadius = {
  sm: 8,
  md: 12,
  lg: 18,
  xl: 28,
  full: 9999,
};

export const shadows = {
  sm: {
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.05,
    shadowRadius: 4,
    elevation: 2,
  },
  md: {
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.08,
    shadowRadius: 10,
    elevation: 5,
  },
  lg: {
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 8 },
    shadowOpacity: 0.12,
    shadowRadius: 18,
    elevation: 9,
  },
};

export const gradients = {
  primary: ['#0F4F39', '#1F6B4B', '#5BAA6A'] as const,
  secondary: ['#1C5A44', '#6FAE5F'] as const,
  card: ['rgba(255,255,255,0.98)', 'rgba(245,251,247,0.98)'] as const,
};
