import 'package:flutter/material.dart';
import 'api.dart';

class SearchScreen extends StatefulWidget {
  const SearchScreen({super.key, required this.api});
  final Api api;
  @override
  State<SearchScreen> createState() => _SearchScreenState();
}

class _SearchScreenState extends State<SearchScreen> {
  final _c = TextEditingController();
  final _deb = Debouncer(200);
  List<dynamic> _results = [];
  String _error = '';
  bool _busy = false;
  int _seq = 0; // ignore out-of-order responses

  void _onChanged(String q) {
    if (q.trim().isEmpty) {
      setState(() => _results = []);
      return;
    }
    _deb.run(() async {
      final my = ++_seq;
      setState(() => _busy = true);
      try {
        final r = await widget.api.search(q.trim());
        if (my == _seq) setState(() { _results = r; _error = ''; });
      } catch (e) {
        if (my == _seq) setState(() => _error = 'Cannot reach the server: $e');
      } finally {
        if (my == _seq) setState(() => _busy = false);
      }
    });
  }

  @override
  void dispose() { _deb.dispose(); _c.dispose(); super.dispose(); }

  @override
  Widget build(BuildContext context) => Column(children: [
        Padding(
          padding: const EdgeInsets.all(12),
          child: TextField(
            controller: _c,
            onChanged: _onChanged,
            autofocus: true,
            decoration: InputDecoration(
              hintText: 'Condition, ICD-10, symptom or drug',
              prefixIcon: const Icon(Icons.search),
              suffixIcon: _busy ? const Padding(padding: EdgeInsets.all(12), child: SizedBox(width: 16, height: 16, child: CircularProgressIndicator(strokeWidth: 2))) : null,
              border: const OutlineInputBorder(),
            ),
          ),
        ),
        if (_error.isNotEmpty) Padding(padding: const EdgeInsets.all(8), child: Text(_error, style: const TextStyle(color: Colors.red))),
        Expanded(
          child: ListView.separated(
            itemCount: _results.length,
            separatorBuilder: (_, __) => const Divider(height: 1),
            itemBuilder: (_, i) {
              final r = _results[i];
              return ListTile(
                title: Text(r['disease']),
                subtitle: Text((r['snippet'] ?? '').toString().replaceAll('\n', ' '), maxLines: 2, overflow: TextOverflow.ellipsis),
                trailing: Text(r['icd10'] ?? ''),
                onTap: () => Navigator.push(context, MaterialPageRoute(builder: (_) => DiseaseScreen(api: widget.api, name: r['disease']))),
              );
            },
          ),
        ),
      ]);
}

class DiseaseScreen extends StatelessWidget {
  const DiseaseScreen({super.key, required this.api, required this.name});
  final Api api;
  final String name;
  static const sections = {
    'clinical_features': 'Clinical features',
    'investigations': 'Investigations',
    'differential_diagnosis': 'Differential diagnosis',
    'management': 'Management',
  };

  @override
  Widget build(BuildContext context) => Scaffold(
        appBar: AppBar(title: Text(name, overflow: TextOverflow.ellipsis)),
        body: FutureBuilder<Map<String, dynamic>>(
          future: api.disease(name),
          builder: (_, s) {
            if (s.hasError) return Center(child: Text('${s.error}'));
            if (!s.hasData) return const Center(child: CircularProgressIndicator());
            final d = s.data!;
            return ListView(padding: const EdgeInsets.all(16), children: [
              Text('UCG section ${d['section_number']} · PDF page ${d['start_pdf_page']}', style: Theme.of(context).textTheme.bodySmall),
              for (final e in sections.entries)
                if ((d[e.key] ?? '').toString().isNotEmpty) ...[
                  const SizedBox(height: 16),
                  Text(e.value, style: Theme.of(context).textTheme.titleMedium),
                  const SizedBox(height: 4),
                  SelectableText(d[e.key]),
                ],
            ]);
          },
        ),
      );
}
