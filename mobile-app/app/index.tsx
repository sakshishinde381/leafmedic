import { router } from 'expo-router';
import * as ImagePicker from 'expo-image-picker';
import Constants from 'expo-constants';
import { useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  Image,
  ScrollView,
  Alert,
  ActivityIndicator,
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';
import { colors, spacing, borderRadius, shadows, gradients } from '../constants/theme';
import { addToHistory } from '../utils/historyStorage';

function resolveApiBase() {
  const envBase = process.env.EXPO_PUBLIC_API_URL?.trim();
  if (envBase) return envBase.replace(/\/$/, '');

  const configBase = (Constants.expoConfig?.extra?.apiUrl as string | undefined)?.trim();
  if (configBase) return configBase.replace(/\/$/, '');

  const hostUri = Constants.expoConfig?.hostUri || '';
  const host = hostUri.split(':')[0];
  if (host) return `http://${host}:5000`;

  return 'http://localhost:5000';
}

const API_BASE = resolveApiBase();

export default function HomeScreen() {
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const requestCameraPermission = async () => {
    const { status } = await ImagePicker.requestCameraPermissionsAsync();
    if (status !== 'granted') {
      Alert.alert('Permission needed', 'Camera access is required to scan leaves.');
      return;
    }
    return true;
  };

  const openCamera = async () => {
    const ok = await requestCameraPermission();
    if (!ok) return;
    const result = await ImagePicker.launchCameraAsync({
      mediaTypes: ['images'] as any,
      allowsEditing: true,
      aspect: [1, 1],
      quality: 0.85,
    });
    if (!result.canceled && result.assets[0]) {
      setSelectedImage(result.assets[0].uri);
    }
  };

  const openGallery = async () => {
    const { status } = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (status !== 'granted') {
      Alert.alert('Permission needed', 'Gallery access is required to pick a leaf image.');
      return;
    }
    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ['images'] as any,
      allowsEditing: true,
      aspect: [1, 1],
      quality: 0.85,
    });
    if (!result.canceled && result.assets[0]) {
      setSelectedImage(result.assets[0].uri);
    }
  };

  const predict = async () => {
    if (!selectedImage || loading) return;
    setLoading(true);
    try {
      const formData = new FormData();
      const filename = selectedImage.split('/').pop() || 'image.jpg';
      const match = /\.(\w+)$/.exec(filename);
      const type = match ? `image/${match[1]}` : 'image/jpeg';
      (formData as any).append('image', {
        uri: selectedImage,
        name: filename,
        type,
      });

      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), 30000);

      const res = await fetch(`${API_BASE}/predict`, {
        method: 'POST',
        body: formData,
        headers: { Accept: 'application/json' },
        signal: controller.signal,
      });
      clearTimeout(timeout);

      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.error || `Server error ${res.status}`);
      }

      const data = await res.json();
      await addToHistory(data.plant, data.confidence, data.info || '');
      router.push({
        pathname: '/result',
        params: {
          plant: data.plant,
          info: data.info || '',
          imageUri: selectedImage,
        },
      });
    } catch (e: any) {
      const msg =
        e.name === 'AbortError'
          ? `Request timed out at ${API_BASE}.`
          : e.message || `Network error at ${API_BASE}.`;
      router.push({
        pathname: '/result',
        params: { error: msg, imageUri: selectedImage },
      });
    } finally {
      setLoading(false);
    }
  };

  const retake = () => setSelectedImage(null);

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.content}>
      <View style={styles.topActions}>
        <TouchableOpacity style={styles.topActionBtn} onPress={() => router.push('/history')}>
          <Ionicons name="time-outline" size={18} color={colors.primaryDark} />
          <Text style={styles.topActionText}>History</Text>
        </TouchableOpacity>
        <TouchableOpacity style={styles.topActionBtn} onPress={() => router.push('/about')}>
          <Ionicons name="information-circle-outline" size={18} color={colors.primaryDark} />
          <Text style={styles.topActionText}>About</Text>
        </TouchableOpacity>
      </View>

      <LinearGradient colors={gradients.primary as any} style={[styles.hero, shadows.lg]}>
        <View style={styles.heroBadge}>
          <Ionicons name="sparkles-outline" size={16} color="#E9F8EE" />
        </View>
        <Text style={styles.heroTitle}>LeafMedic</Text>
        <Text style={styles.heroSubtitle}>Identify medicinal leaves quickly and reliably.</Text>
      </LinearGradient>

      {!selectedImage ? (
        <View style={styles.actionGrid}>
          <TouchableOpacity style={[styles.actionCard, shadows.md]} onPress={openCamera} activeOpacity={0.9}>
            <LinearGradient colors={gradients.secondary as any} style={styles.actionCardBg}>
              <View style={styles.actionIconWrap}>
                <Ionicons name="camera" size={30} color="#FFFFFF" />
              </View>
              <Text style={styles.actionTitle}>Scan With Camera</Text>
              <Text style={styles.actionSubtitle}>Capture a fresh photo of the leaf.</Text>
            </LinearGradient>
          </TouchableOpacity>

          <TouchableOpacity style={[styles.actionCard, shadows.md]} onPress={openGallery} activeOpacity={0.9}>
            <LinearGradient colors={['#2A6C59', '#398D72'] as any} style={styles.actionCardBg}>
              <View style={styles.actionIconWrap}>
                <Ionicons name="images" size={30} color="#FFFFFF" />
              </View>
              <Text style={styles.actionTitle}>Upload From Gallery</Text>
              <Text style={styles.actionSubtitle}>Use an existing image from your phone.</Text>
            </LinearGradient>
          </TouchableOpacity>
        </View>
      ) : (
        <View style={[styles.previewCard, shadows.md]}>
          <Text style={styles.previewLabel}>Selected image</Text>
          <Image source={{ uri: selectedImage }} style={styles.previewImage} />

          <View style={styles.previewActions}>
            <TouchableOpacity style={styles.secondaryBtn} onPress={retake} disabled={loading}>
              <Ionicons name="camera-reverse-outline" size={20} color={colors.primaryDark} />
              <Text style={styles.secondaryBtnText}>Retake</Text>
            </TouchableOpacity>

            <TouchableOpacity
              style={[styles.primaryBtn, loading && styles.primaryBtnDisabled]}
              onPress={predict}
              disabled={loading}
              activeOpacity={0.9}
            >
              <LinearGradient colors={gradients.primary as any} style={StyleSheet.absoluteFill} />
              {loading ? (
                <>
                  <ActivityIndicator size="small" color="#FFFFFF" />
                  <Text style={styles.primaryBtnText}>Analyzing...</Text>
                </>
              ) : (
                <>
                  <Ionicons name="leaf-outline" size={20} color="#FFFFFF" />
                  <Text style={styles.primaryBtnText}>Identify Leaf</Text>
                </>
              )}
            </TouchableOpacity>
          </View>
        </View>
      )}

      <View style={[styles.tipCard, shadows.sm]}>
        <Ionicons name="bulb-outline" size={20} color={colors.warning} />
        <Text style={styles.tipText}>Use a close, well-lit image with one leaf centered for best accuracy.</Text>
      </View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: colors.background },
  content: { padding: spacing.lg, paddingBottom: spacing.xxl },

  topActions: { flexDirection: 'row', justifyContent: 'flex-end', gap: spacing.sm, marginBottom: spacing.md },
  topActionBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.xs,
    backgroundColor: colors.surface,
    borderRadius: borderRadius.full,
    borderWidth: 1,
    borderColor: colors.border,
    paddingHorizontal: spacing.md,
    paddingVertical: 7,
  },
  topActionText: { fontSize: 12, fontWeight: '700', color: colors.primaryDark },

  hero: { borderRadius: borderRadius.xl, padding: spacing.lg, marginBottom: spacing.lg },
  heroBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    alignSelf: 'flex-start',
    gap: spacing.xs,
    backgroundColor: 'rgba(255,255,255,0.16)',
    borderRadius: borderRadius.full,
    paddingHorizontal: spacing.sm,
    paddingVertical: 6,
    marginBottom: spacing.md,
  },
  heroTitle: { color: '#FFFFFF', fontSize: 32, fontFamily: 'serif', fontWeight: '700' },
  heroSubtitle: { color: 'rgba(255,255,255,0.9)', marginTop: spacing.xs, fontSize: 14, lineHeight: 20 },

  actionGrid: { gap: spacing.md },
  actionCard: { borderRadius: borderRadius.xl, overflow: 'hidden' },
  actionCardBg: { padding: spacing.lg, minHeight: 138 },
  actionIconWrap: {
    width: 48,
    height: 48,
    borderRadius: borderRadius.md,
    backgroundColor: 'rgba(255,255,255,0.18)',
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: spacing.md,
  },
  actionTitle: { color: '#FFFFFF', fontSize: 19, fontWeight: '700' },
  actionSubtitle: { color: 'rgba(255,255,255,0.9)', marginTop: spacing.xs, fontSize: 13, lineHeight: 18 },

  previewCard: {
    backgroundColor: colors.cardBg,
    borderRadius: borderRadius.xl,
    borderWidth: 1,
    borderColor: colors.border,
    padding: spacing.md,
    marginBottom: spacing.md,
  },
  previewLabel: { color: colors.textSecondary, fontSize: 12, marginBottom: spacing.sm, fontWeight: '600' },
  previewImage: {
    width: '100%',
    aspectRatio: 1,
    borderRadius: borderRadius.lg,
    backgroundColor: colors.surfaceSoft,
    marginBottom: spacing.md,
  },
  previewActions: { flexDirection: 'row', gap: spacing.sm },

  secondaryBtn: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: spacing.xs,
    borderWidth: 1,
    borderColor: colors.borderStrong,
    borderRadius: borderRadius.md,
    backgroundColor: colors.surface,
    paddingVertical: spacing.md,
  },
  secondaryBtnText: { color: colors.primaryDark, fontWeight: '700' },
  primaryBtn: {
    flex: 1.5,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: spacing.xs,
    borderRadius: borderRadius.md,
    overflow: 'hidden',
    paddingVertical: spacing.md,
  },
  primaryBtnText: { color: '#FFFFFF', fontWeight: '700' },
  primaryBtnDisabled: { opacity: 0.72 },

  tipCard: {
    marginTop: spacing.sm,
    backgroundColor: '#FFF9E8',
    borderRadius: borderRadius.md,
    borderWidth: 1,
    borderColor: '#F1DFA9',
    flexDirection: 'row',
    alignItems: 'flex-start',
    gap: spacing.sm,
    padding: spacing.md,
  },
  tipText: { flex: 1, color: '#6C5A2A', fontSize: 13, lineHeight: 18 },
});
