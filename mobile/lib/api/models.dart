// Wire models mirroring the FastAPI response shapes from `MovieOut`,
// `MoviePage`, `User` and `RecommendationItem`.

class Movie {
  const Movie({
    required this.id,
    required this.title,
    required this.releaseYear,
    required this.genres,
    required this.director,
    required this.country,
    required this.actors,
    this.description,
    this.posterUrl,
  });

  /// Backend identifiers are UUIDs. Keep them opaque: the app only needs to
  /// pass them back to the API and use them for stable widget identity.
  final String id;
  final String title;
  final int releaseYear;
  final List<String> genres;
  final String director;
  final String country;
  final List<String> actors;
  final String? description;
  // Nullable throughout the UI: OMDb will miss some films and may be
  // keyless or over quota.
  final String? posterUrl;

  factory Movie.fromJson(Map<String, dynamic> json) {
    return Movie(
      id: json['id'] as String,
      title: json['title'] as String,
      releaseYear: json['release_year'] as int,
      genres: List<String>.from(json['genres'] as List? ?? const []),
      director: json['director'] as String,
      country: json['country'] as String,
      actors: List<String>.from(json['actors'] as List? ?? const []),
      description: json['description'] as String?,
      posterUrl: json['poster_url'] as String?,
    );
  }
}

class MoviePage {
  const MoviePage({
    required this.items,
    required this.total,
    required this.limit,
    required this.offset,
  });

  final List<Movie> items;
  final int total;
  final int limit;
  final int offset;

  factory MoviePage.fromJson(Map<String, dynamic> json) {
    return MoviePage(
      items: (json['items'] as List? ?? const [])
          .map((item) => Movie.fromJson(item as Map<String, dynamic>))
          .toList(),
      total: json['total'] as int,
      limit: json['limit'] as int,
      offset: json['offset'] as int,
    );
  }
}

class AppUser {
  const AppUser({required this.id, required this.username});

  final String id;
  final String username;

  factory AppUser.fromJson(Map<String, dynamic> json) {
    return AppUser(
        id: json['id'] as String, username: json['username'] as String);
  }
}

class Recommendation {
  const Recommendation({required this.movie, this.reason, this.matchScore});

  final Movie movie;
  final String? reason;
  final double? matchScore;

  factory Recommendation.fromJson(Map<String, dynamic> json) {
    final rawScore = json['match_score'];
    return Recommendation(
      movie: Movie.fromJson(json['movie'] as Map<String, dynamic>),
      reason: json['reason'] as String?,
      matchScore: rawScore == null ? null : (rawScore as num).toDouble(),
    );
  }
}
