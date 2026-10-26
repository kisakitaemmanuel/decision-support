import 'package:flutter/material.dart';
import 'api.dart';

class TriageScreen extends StatefulWidget {
  const TriageScreen({super.key, required this.api});
  final Api api;
  @override
  State<TriageScreen> createState() => _TriageScreenState();
}

class _TriageScreenState extends State<TriageScreen> {
  final _note = TextEditingController();
  final _weight = TextEditingController();
  bool _pregnant = false, _busy = false;
  Map<String, dynamic>? _r;
  String _error = '';

  Color _sev(String s) => switch (s) {
        'emergency' || 'contraindicated' => Colors.red.shade700,
        'urgent' || 'avoid' => Colors.orange.shade800,
        'ok' => Colors.green.shade700,
        _ => Colors.amber.shade800,
      };

  Future<void> _go() async {
    if (_note.text.trim().length < 3) return;
    setState(() { _busy = true; _error = ''; });
    final ov = <String, dynamic>{
      if (double.tryParse(_weight.text) != null) 'weight_kg': double.parse(_weight.text),
      if (_pregnant) 'pregnant': true,
    };
    try {
      final r = await widget.api.triage(_note.text.trim(), ov.isEmpty ? null : ov);
      setState(() => _r = r);
    } catch (e) {
      setState(() => _error = '$e');
    } finally {
      setState(() => _busy = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final r = _r;
    return ListView(padding: const EdgeInsets.all(12), children: [
      TextField(controller: _note, minLines: 4, maxLines: 8,
          decoration: const InputDecoration(labelText: 'Patient note', hintText: '3 year old, 14 kg, fever, convulsion...', border: OutlineInputBorder())),
      const SizedBox(height: 8),
      Row(children: [
        Expanded(child: TextField(controller: _weight, keyboardType: TextInputType.number,
            decoration: const InputDecoration(labelText: 'Weight kg (optional)', border: OutlineInputBorder()))),
        const SizedBox(width: 12),
        FilterChip(label: const Text('Pregnant'), selected: _pregnant, onSelected: (v) => setState(() => _pregnant = v)),
      ]),
      const SizedBox(height: 8),
      FilledButton(onPressed: _busy ? null : _go, child: Text(_busy ? 'Working…' : 'Assess')),
      if (_error.isNotEmpty) Padding(padding: const EdgeInsets.all(8), child: Text(_error, style: const TextStyle(color: Colors.red))),
      if (r != null) ...[
        const SizedBox(height: 12),
        for (final h in (r['red_flags'] as List))
          Card(color: _sev(h['severity']).withOpacity(.12), child: ListTile(
            leading: Icon(Icons.warning_amber, color: _sev(h['severity'])),
            title: Text(h['message']), subtitle: Text('${h['action']}\n[${h['rule_id']}${h['reviewed'] ? '' : ', unreviewed'}]'))),
        Text('Referral: ${r['assessment']['referral']?['decision'] ?? '-'}', style: Theme.of(context).textTheme.titleMedium),
        const SizedBox(height: 8),
        Text('Likely conditions', style: Theme.of(context).textTheme.titleMedium),
        for (final d in (r['assessment']['differential'] as List))
          ListTile(dense: true, title: Text(d['disease_name']), subtitle: Text('matches: ${(d['matched_features'] as List).join(', ')}'), trailing: Text('p.${d['page']}')),
        Text('Medicines named in the guideline', style: Theme.of(context).textTheme.titleMedium),
        for (final m in (r['medications'] as List))
          ListTile(dense: true,
            leading: CircleAvatar(radius: 6, backgroundColor: _sev(m['status'])),
            title: Text('${m['drug']} · ${m['status']}'),
            subtitle: Text([
              if (m['dose'] != null) 'Calculated: ${m['dose']['dose_min']}-${m['dose']['dose_max']} ${m['dose']['unit']} ${m['dose']['route']}',
              ...(m['messages'] as List),
            ].join('\n'))),
        ExpansionTile(title: const Text('Full report'), children: [Padding(padding: const EdgeInsets.all(12), child: SelectableText(r['explanation']['text']))]),
        Row(mainAxisAlignment: MainAxisAlignment.spaceEvenly, children: [
          for (final v in ['agree', 'disagree', 'unsure'])
            OutlinedButton(onPressed: () async { await widget.api.feedback(r['decision_id'], v);
              if (mounted) ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('Saved: $v'))); }, child: Text(v)),
        ]),
      ],
    ]);
  }
}
