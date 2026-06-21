import { Stack } from 'expo-router';
import { StatusBar } from 'expo-status-bar';
import { SafeAreaProvider } from 'react-native-safe-area-context';
import { colors } from '../constants/theme';

export default function RootLayout() {
  return (
    <SafeAreaProvider>
      <StatusBar style="dark" />
      <Stack
        screenOptions={{
          headerStyle: { backgroundColor: colors.background },
          headerTintColor: colors.text,
          headerShadowVisible: false,
          headerTitleStyle: { fontWeight: '800' },
          contentStyle: { backgroundColor: colors.background },
        }}
      >
        <Stack.Screen name="index" options={{ title: 'LeafMedic' }} />
        <Stack.Screen name="result" options={{ title: 'Result' }} />
        <Stack.Screen name="history" options={{ title: 'History' }} />
        <Stack.Screen name="about" options={{ title: 'About' }} />
      </Stack>
    </SafeAreaProvider>
  );
}
