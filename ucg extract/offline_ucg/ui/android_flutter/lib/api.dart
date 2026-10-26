import 'dart:async';
import 'dart:convert';
import 'package:http/http.dart' as http;
import 'package:shared_preferences/shared_preferences.dart';

/// Talks to the clinic's FastAPI server over the local network (no internet needed).
class Api {
  Api(this.baseUrl);
  final String baseUrl; // e.g. http://192.168.1.20:8000

  static Future<Api> load() async {
    final p = await SharedPreferences.getInstance();
    return Api(p.getString('baseUrl') ?? 'http://192.168.1.20:8000');
  }

  static Future<void> save(String url) async =>
      (await SharedPreferences.getInstance()).setString('baseUrl', url);

  Uri _u(String path, [Map<String, String>? q]) =>
      Uri.parse(baseUrl).replace(path: path, queryParameters: q);

  Future<dynamic> _get(Uri u) async {
    final r = await http.get(u).timeout(const Duration(seconds: 15));
    if (r.statusCode != 200) throw Exception('HTTP ${r.statusCode}: ${r.body}');
    return jsonDecode(utf8.decode(r.bodyBytes));
  }

  Future<Map<String, dynamic>> health() async => await _get(_u('/api/health'));

  Future<List<dynamic>> suggest(String q) async =>
      (await _get(_u('/api/suggest', {'q': q})))['suggestions'];

  Future<List<dynamic>> search(String q) async =>
      (await _get(_u('/api/search', {'q': q, 'k': '10'})))['results'];

  Future<Map<String, dynamic>> disease(String name) async =>
      await _get(_u('/api/disease/${Uri.encodeComponent(name)}'));

  /// LLM-backed triage can take a while on a CPU server, hence the long timeout.
  Future<Map<String, dynamic>> triage(String text, Map<String, dynamic>? overrides) async {
    final r = await http
        .post(_u('/api/triage'),
            headers: {'Content-Type': 'application/json'},
            body: jsonEncode({'text': text, 'overrides': overrides}))
        .timeout(const Duration(seconds: 180));
    if (r.statusCode != 200) throw Exception('HTTP ${r.statusCode}: ${r.body}');
    return jsonDecode(utf8.decode(r.bodyBytes));
  }

  Future<void> feedback(String id, String verdict) async {
    await http.post(_u('/api/audit/$id/feedback'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({'verdict': verdict, 'reviewer': 'android'}));
  }
}

/// Runs [fn] only after the user stops typing for [ms] milliseconds (search-as-you-type).
class Debouncer {
  Debouncer([this.ms = 200]);
  final int ms;
  Timer? _t;
  void run(void Function() fn) {
    _t?.cancel();
    _t = Timer(Duration(milliseconds: ms), fn);
  }
  void dispose() => _t?.cancel();
}
