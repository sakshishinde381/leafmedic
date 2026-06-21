const HISTORY_KEY = '@leafmedic_history';
const MAX_ITEMS = 5;

export type HistoryItem = {
  plant: string;
  confidence: number;
  info: string;
  date: string;
};

let memoryFallback: HistoryItem[] = [];

type StorageLike = {
  getItem: (key: string) => Promise<string | null>;
  setItem: (key: string, value: string) => Promise<void>;
};

async function getStorage(): Promise<StorageLike | null> {
  try {
    return require('@react-native-async-storage/async-storage').default as StorageLike;
  } catch {
    return null;
  }
}

export async function getHistory(): Promise<HistoryItem[]> {
  const AsyncStorage = await getStorage();
  if (AsyncStorage) {
    try {
      const raw = await AsyncStorage.getItem(HISTORY_KEY);
      if (raw) return JSON.parse(raw);
    } catch (_) {}
  }
  return memoryFallback;
}

export async function addToHistory(plant: string, confidence: number, info: string): Promise<void> {
  const list = await getHistory();
  const entry: HistoryItem = {
    plant,
    confidence,
    info,
    date: new Date().toLocaleString(),
  };
  const next = [entry, ...list].slice(0, MAX_ITEMS);
  memoryFallback = next;
  const AsyncStorage = await getStorage();
  if (AsyncStorage) {
    try {
      await AsyncStorage.setItem(HISTORY_KEY, JSON.stringify(next));
    } catch (_) {}
  }
}
