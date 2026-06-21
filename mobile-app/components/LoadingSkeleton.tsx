import { useEffect, useRef } from 'react';
import { View, StyleSheet, Animated } from 'react-native';
import { colors, spacing, borderRadius } from '../constants/theme';

type LoadingSkeletonProps = {
  width?: number | string;
  height?: number;
  style?: object;
};

export function LoadingSkeleton({ width = '100%', height = 24, style }: LoadingSkeletonProps) {
  const opacity = useRef(new Animated.Value(0.35)).current;

  useEffect(() => {
    const loop = Animated.loop(
      Animated.sequence([
        Animated.timing(opacity, { toValue: 0.75, useNativeDriver: true, duration: 700 }),
        Animated.timing(opacity, { toValue: 0.35, useNativeDriver: true, duration: 700 }),
      ])
    );
    loop.start();
    return () => loop.stop();
  }, [opacity]);

  return <Animated.View style={[styles.skeleton, { width, height, opacity }, style]} />;
}

export function ResultLoadingSkeleton() {
  return (
    <View style={styles.card}>
      <LoadingSkeleton height={18} width="40%" style={{ marginBottom: spacing.md }} />
      <LoadingSkeleton height={32} width="70%" style={{ marginBottom: spacing.md }} />
      <LoadingSkeleton height={10} style={{ marginBottom: spacing.lg }} />
      <LoadingSkeleton height={16} width="50%" style={{ marginBottom: spacing.sm }} />
      <LoadingSkeleton height={62} />
    </View>
  );
}

const styles = StyleSheet.create({
  skeleton: {
    backgroundColor: '#D6E5DC',
    borderRadius: borderRadius.sm,
  },
  card: {
    backgroundColor: colors.cardBg,
    borderRadius: borderRadius.xl,
    padding: spacing.lg,
    marginBottom: spacing.lg,
    borderWidth: 1,
    borderColor: colors.border,
  },
});
