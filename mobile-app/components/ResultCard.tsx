import { View, Text, StyleSheet } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { LinearGradient } from 'expo-linear-gradient';
import { colors, spacing, borderRadius, shadows, gradients } from '../constants/theme';

type ResultCardProps = {
  plant: string;
  info: string;
};

export function ResultCard({ plant, info }: ResultCardProps) {
  const isUnknown = plant.trim().toLowerCase() === 'unknown';

  return (
    <View style={[styles.card, shadows.lg]}>
      <LinearGradient colors={gradients.card as any} style={styles.cardGradient}>
        <View style={styles.header}>
          <View style={styles.iconWrap}>
            <Ionicons
              name={isUnknown ? 'help-circle-outline' : 'leaf'}
              size={24}
              color={colors.primary}
            />
          </View>
          <View style={styles.titleWrap}>
            <Text style={styles.label}>{isUnknown ? 'Classification result' : 'Detected plant'}</Text>
            <Text style={styles.plantName}>{plant}</Text>
          </View>
        </View>

        <View style={styles.infoSection}>
          <Text style={styles.infoLabel}>{isUnknown ? 'Status' : 'Medicinal benefits'}</Text>
          <Text style={styles.infoText}>
            {info || (isUnknown ? 'This leaf is outside the supported classes.' : 'No description available for this class.')}
          </Text>
        </View>
      </LinearGradient>
    </View>
  );
}

const styles = StyleSheet.create({
  card: {
    borderRadius: borderRadius.xl,
    overflow: 'hidden',
    marginBottom: spacing.lg,
  },
  cardGradient: {
    padding: spacing.lg,
    borderWidth: 1,
    borderColor: colors.border,
    borderRadius: borderRadius.xl,
  },
  header: { flexDirection: 'row', alignItems: 'center', marginBottom: spacing.md, gap: spacing.sm },
  iconWrap: {
    width: 46,
    height: 46,
    borderRadius: borderRadius.md,
    backgroundColor: '#EAF6EE',
    alignItems: 'center',
    justifyContent: 'center',
  },
  titleWrap: { flex: 1 },
  label: { fontSize: 12, color: colors.textSecondary, marginBottom: 2 },
  plantName: { fontSize: 26, lineHeight: 30, color: colors.text, fontWeight: '800', fontFamily: 'serif' },

  infoSection: {
    borderTopWidth: 1,
    borderTopColor: colors.border,
    paddingTop: spacing.md,
  },
  infoLabel: {
    fontSize: 12,
    textTransform: 'uppercase',
    letterSpacing: 0.5,
    color: colors.muted,
    marginBottom: spacing.xs,
    fontWeight: '700',
  },
  infoText: { fontSize: 15, lineHeight: 22, color: colors.textSecondary },
});
