import 'package:flutter/material.dart';

/// Material 3 theme: one seed colour, one custom [TextTheme], generous
/// spacing and 16px rounded cards, applied in one place so both platforms
/// look pixel-identical.
ThemeData buildAppTheme(Brightness brightness) {
  const seedColor = Color(0xFF6750A4);
  final colorScheme = ColorScheme.fromSeed(
    seedColor: seedColor,
    brightness: brightness,
  );

  final textTheme = const TextTheme(
    headlineLarge: TextStyle(fontWeight: FontWeight.w700, letterSpacing: -0.5),
    headlineMedium: TextStyle(fontWeight: FontWeight.w700, letterSpacing: -0.3),
    titleLarge: TextStyle(fontWeight: FontWeight.w600),
    titleMedium: TextStyle(fontWeight: FontWeight.w600),
    bodyLarge: TextStyle(height: 1.4),
    bodyMedium: TextStyle(height: 1.4),
  );

  return ThemeData(
    useMaterial3: true,
    colorScheme: colorScheme,
    textTheme: textTheme,
    scaffoldBackgroundColor: colorScheme.surface,
    cardTheme: CardThemeData(
      elevation: 0,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
      margin: const EdgeInsets.all(8),
    ),
    inputDecorationTheme: InputDecorationTheme(
      filled: true,
      border: OutlineInputBorder(
        borderRadius: BorderRadius.circular(16),
        borderSide: BorderSide.none,
      ),
      contentPadding: const EdgeInsets.symmetric(horizontal: 20, vertical: 16),
    ),
    filledButtonTheme: FilledButtonThemeData(
      style: FilledButton.styleFrom(
        padding: const EdgeInsets.symmetric(vertical: 16),
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
      ),
    ),
    visualDensity: VisualDensity.comfortable,
  );
}
