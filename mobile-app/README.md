# LeafMedic Mobile App

Expo (React Native) + Expo Router, TypeScript. Green nature theme.

## Setup

1. Install dependencies: `npm install`
2. Set backend URL: create `.env` with `EXPO_PUBLIC_API_URL=https://your-hosted-backend-url`.
3. Run:
   - Web: `npm run web`
   - Android: `npm run android`
   - iOS: `npx expo run:ios`

## API

App calls `POST /predict` with multipart form key `image`. Ensure backend is running and CORS allows your origin.
