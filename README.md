# movie-recommender

## Run locally on the iOS Simulator

For the usual development workflow, one command starts the API and Flutter
app, and opens the Simulator:

```bash
just mobile-ios
```

When you quit Flutter, the recipe also stops the API process it started.

To run each part separately:

1. Start the iOS Simulator (replace the device name if you prefer another
   installed simulator):

   ```bash
   open -a Simulator
   xcrun simctl boot "iPhone 16 Pro"
   ```

2. In a separate terminal, start the local API. `just api` serves it on port
   `8000`:

   ```bash
   just api
   ```

3. In another terminal, start the Flutter app from the `mobile` directory:

   ```bash
   cd mobile
   flutter run --dart-define=API_BASE_URL=http://localhost:8000
   ```

The iOS Simulator can reach the API at `localhost`. Once Flutter is running,
use `r` in its terminal for a hot reload and `q` to stop it.
