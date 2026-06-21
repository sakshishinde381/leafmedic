import { View, Text, StyleSheet, ScrollView } from 'react-native';
import { useFocusEffect } from 'expo-router';
import { useCallback, useState } from 'react';
import { Ionicons } from '@expo/vector-icons';
import { getHistory, type HistoryItem } from '../utils/historyStorage';
import { colors, spacing, borderRadius, shadows } from '../constants/theme';

export default function HistoryScreen() {
  const [items, setItems] = useState<HistoryItem[]>([]);

  useFocusEffect(
    useCallback(() => {
      getHistory().then(setItems);
    }, [])
  );

  if (items.length === 0) {
    return (
      <View style={styles.emptyWrap}>
        <View style={styles.emptyIconWrap}>
          <Ionicons name="time-outline" size={34} color={colors.primary} />
        </View>
        <Text style={styles.emptyTitle}>No scans yet</Text>
        <Text style={styles.emptySubtitle}>Your most recent leaf analyses will appear here.</Text>
      </View>
    );
  }

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.content}>
      <Text style={styles.heading}>Recent Identifications</Text>

      {items.map((item, i) => {
        return (
          <View key={i} style={[styles.card, shadows.sm]}>
            <View style={styles.rowTop}>
              <View style={styles.nameWrap}>
                <Ionicons name="leaf-outline" size={16} color={colors.primary} />
                <Text style={styles.plantName}>{item.plant}</Text>
              </View>
            </View>

            <Text style={styles.date}>{item.date}</Text>
          </View>
        );
      })}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: colors.background },
  content: { padding: spacing.lg, paddingBottom: spacing.xxl },

  heading: {
    fontSize: 20,
    fontWeight: '800',
    color: colors.text,
    marginBottom: spacing.md,
  },

  card: {
    backgroundColor: colors.cardBg,
    borderRadius: borderRadius.lg,
    borderWidth: 1,
    borderColor: colors.border,
    padding: spacing.md,
    marginBottom: spacing.md,
  },
  rowTop: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: spacing.sm },
  nameWrap: { flexDirection: 'row', alignItems: 'center', gap: spacing.xs },
  plantName: { fontSize: 17, fontWeight: '700', color: colors.text },
  date: { fontSize: 12, color: colors.muted },

  emptyWrap: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: colors.background,
    padding: spacing.xl,
  },
  emptyIconWrap: {
    width: 72,
    height: 72,
    borderRadius: borderRadius.full,
    backgroundColor: '#EAF6EE',
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: spacing.md,
  },
  emptyTitle: { fontSize: 22, fontWeight: '800', color: colors.text },
  emptySubtitle: { fontSize: 14, color: colors.textSecondary, textAlign: 'center', marginTop: spacing.xs },
});
