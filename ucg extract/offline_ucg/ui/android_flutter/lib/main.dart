import 'package:flutter/material.dart';
import 'api.dart';
import 'search_screen.dart';
import 'triage_screen.dart';

void main() => runApp(const App());

class App extends StatelessWidget {
  const App({super.key});
  @override
  Widget build(BuildContext context) => MaterialApp(
        title: 'Offline UCG',
        theme: ThemeData(colorSchemeSeed: const Color(0xFF0F766E), useMaterial3: true),
        home: FutureBuilder<Api>(
          future: Api.load(),
          builder: (_, s) => s.hasData ? Home(api: s.data!) : const Scaffold(body: Center(child: CircularProgressIndicator())),
        ),
      );
}

class Home extends StatefulWidget {
  const Home({super.key, required this.api});
  final Api api;
  @override
  State<Home> createState() => _HomeState();
}

class _HomeState extends State<Home> {
  int _i = 0;
  late Api _api = widget.api;

  Future<void> _settings() async {
    final c = TextEditingController(text: _api.baseUrl);
    final ok = await showDialog<bool>(context: context, builder: (_) => AlertDialog(
      title: const Text('Server address'),
      content: Column(mainAxisSize: MainAxisSize.min, children: [
        TextField(controller: c, decoration: const InputDecoration(hintText: 'http://192.168.1.20:8000')),
        const SizedBox(height: 8),
        const Text('The clinic computer running the Offline UCG service. Same Wi-Fi/hotspot; no internet needed.', style: TextStyle(fontSize: 12)),
      ]),
      actions: [TextButton(onPressed: () => Navigator.pop(context, false), child: const Text('Cancel')),
                FilledButton(onPressed: () => Navigator.pop(context, true), child: const Text('Save & test'))]));
    if (ok != true) return;
    await Api.save(c.text.trim());
    setState(() => _api = Api(c.text.trim()));
    try {
      final h = await _api.health();
      if (mounted) ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('Connected: ${h['diseases']} conditions, LLM ${h['llm']}')));
    } catch (e) {
      if (mounted) ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('Cannot connect: $e')));
    }
  }

  @override
  Widget build(BuildContext context) => Scaffold(
        appBar: AppBar(title: const Text('Offline UCG'), actions: [IconButton(onPressed: _settings, icon: const Icon(Icons.settings_ethernet))]),
        body: IndexedStack(index: _i, children: [SearchScreen(api: _api), TriageScreen(api: _api)]),
        bottomNavigationBar: NavigationBar(selectedIndex: _i, onDestinationSelected: (i) => setState(() => _i = i), destinations: const [
          NavigationDestination(icon: Icon(Icons.search), label: 'Search'),
          NavigationDestination(icon: Icon(Icons.medical_services_outlined), label: 'Assess'),
        ]),
      );
}
