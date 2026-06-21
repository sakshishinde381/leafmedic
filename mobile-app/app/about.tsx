import { View, Text, StyleSheet, ScrollView } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { LinearGradient } from 'expo-linear-gradient';
import { colors, spacing, borderRadius, shadows, gradients } from '../constants/theme';

const features = [
  {
    icon: 'leaf-outline' as const,
    title: 'AI Classification',
    desc: 'Identifies medicinal leaves using a MobileNet-based model served by Flask.',
  },
  {
    icon: 'flash-outline' as const,
    title: 'Fast Workflow',
    desc: 'Capture or upload, analyze, then view medicinal details instantly.',
  },
  {
    icon: 'shield-checkmark-outline' as const,
    title: 'Practical Use',
    desc: 'Built to support field use and educational demonstrations with clean reliability.',
  },
];

export default function AboutScreen() {
  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.content}>
      <LinearGradient colors={gradients.primary as any} style={[styles.hero, shadows.lg]}>
        <Text style={styles.heroLabel}>About LeafMedic</Text>
        <Text style={styles.heroTitle}>Medicinal Leaf Identification Assistant</Text>
        <Text style={styles.heroSubtitle}>
          A cross-platform app powered by TensorFlow, Flask, and Expo for practical plant recognition.
        </Text>
      </LinearGradient>

      {features.map((feature) => (
        <View key={feature.title} style={[styles.featureCard, shadows.sm]}>
          <View style={styles.featureIconWrap}>
            <Ionicons name={feature.icon} size={20} color={colors.primary} />
          </View>
          <View style={styles.featureBody}>
            <Text style={styles.featureTitle}>{feature.title}</Text>
            <Text style={styles.featureDesc}>{feature.desc}</Text>
          </View>
        </View>
      ))}

      <View style={[styles.noteCard, shadows.sm]}>
        <Text style={styles.noteTitle}>Best Practices</Text>
        <Text style={styles.noteText}>Use close-up, well-lit images and keep the leaf centered for reliable predictions.</Text>
      </View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: colors.background },
  content: { padding: spacing.lg, paddingBottom: spacing.xxl },

  hero: { borderRadius: borderRadius.xl, padding: spacing.lg, marginBottom: spacing.lg },
  heroLabel: {
    color: '#EAF8EF',
    fontSize: 12,
    letterSpacing: 0.6,
    textTransform: 'uppercase',
    fontWeight: '700',
    marginBottom: spacing.sm,
  },
  heroTitle: { color: '#FFFFFF', fontSize: 28, lineHeight: 33, fontWeight: '800', fontFamily: 'serif' },
  heroSubtitle: { color: 'rgba(255,255,255,0.9)', marginTop: spacing.sm, fontSize: 14, lineHeight: 21 },

  featureCard: {
    backgroundColor: colors.cardBg,
    borderRadius: borderRadius.lg,
    borderWidth: 1,
    borderColor: colors.border,
    padding: spacing.md,
    marginBottom: spacing.sm,
    flexDirection: 'row',
    alignItems: 'flex-start',
    gap: spacing.sm,
  },
  featureIconWrap: {
    width: 36,
    height: 36,
    borderRadius: borderRadius.md,
    backgroundColor: '#E8F4EE',
    justifyContent: 'center',
    alignItems: 'center',
    marginTop: 2,
  },
  featureBody: { flex: 1 },
  featureTitle: { fontSize: 16, fontWeight: '700', color: colors.text, marginBottom: 2 },
  featureDesc: { fontSize: 13, lineHeight: 19, color: colors.textSecondary },

  noteCard: {
    backgroundColor: '#FFF9E8',
    borderRadius: borderRadius.lg,
    borderWidth: 1,
    borderColor: '#F1DFA9',
    padding: spacing.md,
    marginTop: spacing.sm,
  },
  noteTitle: { fontSize: 15, fontWeight: '700', color: '#6C5A2A', marginBottom: spacing.xs },
  noteText: { fontSize: 13, lineHeight: 19, color: '#6C5A2A' },
});
