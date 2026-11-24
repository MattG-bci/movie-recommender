import 'package:flutter/material.dart';

/// The app deliberately uses one dark, film-focused Material 3 palette on
/// every platform. Lapis blue carries interactive states while blue-black
/// surfaces keep artwork as the visual focus.
ThemeData buildAppTheme(Brightness brightness) {
  const lapis = Color(0xFF2D70B7);
  final colorScheme =
      ColorScheme.fromSeed(seedColor: lapis, brightness: brightness).copyWith(
    primary: const Color(0xFF78B7FF),
    onPrimary: const Color(0xFF002E5B),
    primaryContainer: const Color(0xFF07477F),
    onPrimaryContainer: const Color(0xFFD8EAFF),
    secondary: const Color(0xFF9BCBFF),
    surface: const Color(0xFF0B1220),
    onSurface: const Color(0xFFE3EAF5),
    surfaceContainer: const Color(0xFF141F32),
    surfaceContainerHigh: const Color(0xFF1B2A42),
    onSurfaceVariant: const Color(0xFFBBC8DB),
    outline: const Color(0xFF60728E),
    error: const Color(0xFFFFB4AB),
  );

  const textTheme = TextTheme(
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
    appBarTheme: AppBarTheme(
      backgroundColor: colorScheme.surface,
      foregroundColor: colorScheme.onSurface,
      centerTitle: false,
      elevation: 0,
      scrolledUnderElevation: 0,
    ),
    cardTheme: CardThemeData(
      elevation: 0,
      color: colorScheme.surfaceContainer,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      margin: EdgeInsets.zero,
    ),
    inputDecorationTheme: InputDecorationTheme(
      filled: true,
      fillColor: colorScheme.surfaceContainer,
      border: OutlineInputBorder(
        borderRadius: BorderRadius.circular(12),
        borderSide: BorderSide.none,
      ),
      focusedBorder: OutlineInputBorder(
        borderRadius: BorderRadius.circular(12),
        borderSide: BorderSide(color: colorScheme.primary, width: 1.5),
      ),
      contentPadding: const EdgeInsets.symmetric(horizontal: 20, vertical: 16),
    ),
    filledButtonTheme: FilledButtonThemeData(
      style: FilledButton.styleFrom(
        padding: const EdgeInsets.symmetric(vertical: 16),
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      ),
    ),
    chipTheme: ChipThemeData(
      backgroundColor: colorScheme.surfaceContainerHigh,
      labelStyle: TextStyle(color: colorScheme.onSurfaceVariant),
      side: BorderSide.none,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
    ),
    navigationBarTheme: NavigationBarThemeData(
      backgroundColor: colorScheme.surfaceContainer,
      indicatorColor: colorScheme.primaryContainer,
      labelTextStyle: WidgetStatePropertyAll(
        TextStyle(color: colorScheme.onSurfaceVariant),
      ),
    ),
    visualDensity: VisualDensity.comfortable,
  );
}
