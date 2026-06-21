import { router, useLocalSearchParams } from 'expo-router';
import { View, Text, StyleSheet, Image, ScrollView, TouchableOpacity } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { ResultCard } from '../components/ResultCard';
import { ErrorState } from '../components/ErrorState';
import { colors, spacing, borderRadius, shadows } from '../constants/theme';

export default function ResultScreen() {
  const params = useLocalSearchParams<{
    plant?: string;
    info?: string;
    error?: string;
    imageUri?: string;
  }>();

  const plant = Array.isArray(params.plant) ? params.plant[0] : params.plant;
  const info = Array.isArray(params.info) ? params.info[0] : params.info;
  const error = Array.isArray(params.error) ? params.error[0] : params.error;
  const imageUri = Array.isArray(params.imageUri) ? params.imageUri[0] : params.imageUri;

  const isError = !!error;

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.content}>
      <View style={styles.headerRow}>
        <Text style={styles.pageLabel}>Analysis summary</Text>
        {!isError && <Text style={styles.pageStatus}>Completed</Text>}
      </View>

      {imageUri && (
        <View style={[styles.imageCard, shadows.md]}>
          <Image source={{ uri: imageUri }} style={styles.previewImage} />
        </View>
      )}

      {isError ? (
        <ErrorState message={error || 'Something went wrong.'} onRetry={() => router.back()} />
      ) : (
        <ResultCard
          plant={plant || 'Unknown'}
          info={info || ''}
        />
      )}

      <View style={styles.actionRow}>
        <TouchableOpacity style={styles.ghostBtn} onPress={() => router.back()}>
          <Ionicons name="refresh-outline" size={20} color={colors.primaryDark} />
          <Text style={styles.ghostBtnText}>New Scan</Text>
        </TouchableOpacity>

        <TouchableOpacity style={styles.homeBtn} onPress={() => router.replace('/')}>
          <Ionicons name="home-outline" size={20} color="#FFFFFF" />
          <Text style={styles.homeBtnText}>Home</Text>
        </TouchableOpacity>
      </View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: colors.background },
  content: { padding: spacing.lg, paddingBottom: spacing.xxl },

  headerRow: {
    marginBottom: spacing.md,
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  pageLabel: { color: colors.text, fontWeight: '800', fontSize: 17 },
  pageStatus: {
    backgroundColor: '#EAF6EE',
    color: colors.primary,
    paddingVertical: 5,
    paddingHorizontal: spacing.sm,
    borderRadius: borderRadius.full,
    fontSize: 11,
    fontWeight: '700',
  },

  imageCard: {
    backgroundColor: colors.cardBg,
    borderRadius: borderRadius.xl,
    overflow: 'hidden',
    marginBottom: spacing.lg,
    borderWidth: 1,
    borderColor: colors.border,
  },
  previewImage: { width: '100%', aspectRatio: 1, backgroundColor: colors.surfaceSoft },

  actionRow: { flexDirection: 'row', gap: spacing.sm, marginTop: spacing.xs },
  ghostBtn: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: spacing.xs,
    backgroundColor: colors.surface,
    borderWidth: 1,
    borderColor: colors.borderStrong,
    borderRadius: borderRadius.md,
    paddingVertical: spacing.md,
  },
  ghostBtnText: { color: colors.primaryDark, fontWeight: '700' },

  homeBtn: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: spacing.xs,
    backgroundColor: colors.primary,
    borderRadius: borderRadius.md,
    paddingVertical: spacing.md,
  },
  homeBtnText: { color: '#FFFFFF', fontWeight: '700' },
});
