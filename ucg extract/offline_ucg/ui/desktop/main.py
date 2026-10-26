"""
ui/desktop/main.py - PySide6 desktop client.

    python -m ui.desktop.main                         # in-process (no server needed)
    python -m ui.desktop.main --api http://clinic-pc:8000

Layout: left = patient note + optional structured fields + Assess button; right = tabs
(Report | Alerts | Conditions | Medicines | Search | Trace).  Slow work (LLM, retrieval) runs in a worker thread so the
window never freezes; typing in the search box shows suggestions after a 150 ms pause.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from PySide6.QtGui import QColor
from PySide6.QtCore import QObject, QRunnable, Qt, QThreadPool, QTimer, Signal, Slot
from PySide6.QtWidgets import (QApplication, QCheckBox, QCompleter, QDoubleSpinBox, QFormLayout, QGroupBox,
                               QHBoxLayout, QLabel, QLineEdit, QListWidget, QMainWindow, QMessageBox, QPlainTextEdit,
                               QPushButton, QSplitter, QTableWidget, QTableWidgetItem, QTabWidget, QTextEdit,
                               QVBoxLayout, QWidget)

from ui.desktop.backend import Backend, HttpBackend, InProcessBackend

SEV_COLOUR = {"emergency": "#b00020", "urgent": "#d96c00", "warning": "#8a6d00",
              "contraindicated": "#b00020", "avoid": "#d96c00", "caution": "#8a6d00", "ok": "#1b7f3b"}


class Signals(QObject):
    done = Signal(object)
    failed = Signal(str)


class Job(QRunnable):
    def __init__(self, fn, *args):
        super().__init__()
        self.fn, self.args, self.sig = fn, args, Signals()

    @Slot()
    def run(self):
        try:
            self.sig.done.emit(self.fn(*self.args))
        except Exception as e:                          # noqa: BLE001 - show every failure in the UI
            self.sig.failed.emit(f"{type(e).__name__}: {e}")


class MainWindow(QMainWindow):
    def __init__(self, backend: Backend):
        super().__init__()
        self.be, self.pool, self.last = backend, QThreadPool.globalInstance(), None
        self.setWindowTitle("Offline UCG - clinical decision support")
        self.resize(1250, 780)

        # ---------------- left: input
        self.note = QPlainTextEdit(placeholderText="Describe the patient, e.g.\n3 year old girl, 14 kg, fever 39.2 C, "
                                                   "convulsion at home, not able to drink")
        self.age = self._spin(0, 120, "years"); self.weight = self._spin(0, 250, "kg"); self.spo2 = self._spin(0, 100, "%")
        self.preg = QCheckBox("Pregnant")
        form = QFormLayout()
        for label, w in (("Age (blank = from text)", self.age), ("Weight", self.weight), ("SpO2", self.spo2)):
            form.addRow(label, w)
        form.addRow(self.preg)
        box = QGroupBox("Optional structured fields (override the text)"); box.setLayout(form)
        self.btn = QPushButton("Assess"); self.btn.setStyleSheet("font-weight:bold;padding:8px")
        self.btn.clicked.connect(self.assess)
        self.status = QLabel(""); self.status.setWordWrap(True)
        left = QWidget(); lv = QVBoxLayout(left)
        lv.addWidget(QLabel("<b>Patient note</b>")); lv.addWidget(self.note, 1); lv.addWidget(box)
        lv.addWidget(self.btn); lv.addWidget(self.status)
        fb = QHBoxLayout()
        for v in ("agree", "disagree", "unsure"):
            b = QPushButton(v.capitalize()); b.clicked.connect(lambda _=False, v=v: self.send_feedback(v)); fb.addWidget(b)
        lv.addWidget(QLabel("Clinician verdict on this result:")); lv.addLayout(fb)

        # ---------------- right: tabs
        self.report = QTextEdit(readOnly=True)
        self.alerts = QTextEdit(readOnly=True)
        self.cond = QTableWidget(0, 5); self.cond.setHorizontalHeaderLabels(["Condition", "Score", "Matches", "Page", "Note"])
        self.meds = QTableWidget(0, 4); self.meds.setHorizontalHeaderLabels(["Medicine", "Status", "Calculated dose", "Notes"])
        self.trace = QTextEdit(readOnly=True)
        self.tabs = QTabWidget()
        for w, name in ((self.report, "Report"), (self.alerts, "Alerts"), (self.cond, "Conditions"),
                        (self.meds, "Medicines"), (self._search_tab(), "Search guideline"), (self.trace, "Trace / debug")):
            self.tabs.addTab(w, name)

        sp = QSplitter(); sp.addWidget(left); sp.addWidget(self.tabs); sp.setSizes([420, 830])
        self.setCentralWidget(sp)
        self.refresh_health()

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _spin(lo, hi, suffix):
        s = QDoubleSpinBox(); s.setRange(lo - 1 if lo == 0 else lo, hi); s.setSpecialValueText(" "); s.setValue(lo - 1 if lo == 0 else lo)
        s.setSuffix(" " + suffix); s.setDecimals(1)
        return s

    def _val(self, w):
        return None if w.value() < 0 else w.value() if w.value() > 0 else None

    def _run(self, fn, ok, *args):
        job = Job(fn, *args)
        job.sig.done.connect(ok)
        job.sig.failed.connect(lambda m: (self.status.setText("Error: " + m), self.btn.setEnabled(True)))
        self.pool.start(job)

    def refresh_health(self):
        self._run(self.be.health, lambda h: self.setWindowTitle(
            f"Offline UCG - {h['diseases']} conditions | LLM: {h['llm']} | search: {h['semantic']}"))

    # ---------------------------------------------------------------- assess
    def assess(self):
        text = self.note.toPlainText().strip()
        if len(text) < 3:
            QMessageBox.information(self, "Offline UCG", "Please describe the patient first.")
            return
        ov = {k: v for k, v in {"age_years": self._val(self.age), "weight_kg": self._val(self.weight),
                                "spo2": self._val(self.spo2), "pregnant": True if self.preg.isChecked() else None}.items()
              if v is not None}
        self.btn.setEnabled(False); self.status.setText("Working...")
        self._run(self.be.triage, self.show_result, text, ov or None)

    def show_result(self, r: dict):
        self.last = r
        self.btn.setEnabled(True)
        self.status.setText(f"Decision {r['decision_id'][:8]} | {r['latency_ms']:.0f} ms | mode: {r['assessment']['mode']}")
        self.report.setPlainText(r["explanation"]["text"])
        html = []
        for h in r["red_flags"]:
            html.append(f"<p><b style='color:{SEV_COLOUR[h['severity']]}'>{h['severity'].upper()}</b> {h['message']} "
                        f"<i>{h['action']}</i> <small>[{h['rule_id']}{'' if h['reviewed'] else ', unreviewed'}]</small></p>")
        self.alerts.setHtml("".join(html) or "<p>No red flags fired.</p>")
        d = r["assessment"]["differential"]
        self.cond.setRowCount(len(d))
        for i, x in enumerate(d):
            for j, v in enumerate([x["disease_name"], f"{x['score']:.2f}", ", ".join(x["matched_features"]),
                                   x["page"], x["demographic_note"]]):
                self.cond.setItem(i, j, QTableWidgetItem(str(v if v is not None else "")))
        self.cond.resizeColumnsToContents()
        m = r["medications"]
        self.meds.setRowCount(len(m))
        for i, x in enumerate(m):
            dose = x["dose"]
            dose_t = f"{dose['dose_min']:g}-{dose['dose_max']:g} {dose['unit']} {dose['route']}" if dose else ("needs " + ", ".join(x["needs"]) if x["needs"] else "see guideline")
            item = QTableWidgetItem(x["status"].upper()); item.setForeground(Qt.GlobalColor.white)
            item.setBackground(QColor(SEV_COLOUR[x["status"]]))
            for j, v in enumerate([QTableWidgetItem(x["drug"]), item, QTableWidgetItem(dose_t), QTableWidgetItem(" | ".join(x["messages"]))]):
                self.meds.setItem(i, j, v)
        self.meds.resizeColumnsToContents()
        self.trace.setPlainText(json.dumps({"trace": r["trace"], "tool_trace": r["assessment"]["tool_trace"],
                                            "versions": r["versions"], "retrieval": r["retrieval"],
                                            "case": r["case"]}, indent=2, default=str))

    def send_feedback(self, verdict: str):
        if not self.last:
            return
        self._run(lambda: self.be.feedback(self.last["decision_id"], verdict), lambda _: self.status.setText(f"Feedback saved: {verdict}"))

    # ---------------------------------------------------------------- search-as-you-type tab
    def _search_tab(self) -> QWidget:
        w = QWidget(); v = QVBoxLayout(w)
        self.q = QLineEdit(placeholderText="Type a condition, ICD-10 code, symptom or drug...")
        self.hits = QListWidget(); self.detail = QTextEdit(readOnly=True)
        self.completer = QCompleter([]); self.completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self.completer.setFilterMode(Qt.MatchFlag.MatchContains); self.q.setCompleter(self.completer)
        self.timer = QTimer(singleShot=True, interval=150)
        self.timer.timeout.connect(self._typed)
        self.q.textEdited.connect(lambda _: self.timer.start())
        self.q.returnPressed.connect(self._typed)
        self.hits.itemClicked.connect(lambda it: self._open(it.data(Qt.ItemDataRole.UserRole)))
        v.addWidget(self.q); v.addWidget(self.hits, 1); v.addWidget(self.detail, 2)
        return w

    def _typed(self):
        t = self.q.text().strip()
        if not t:
            return
        self._run(self.be.suggest, lambda s: self.completer.model().setStringList([x["disease_name"] for x in s]) if hasattr(self.completer.model(), "setStringList") else None, t)
        self._run(self.be.search, self._show_hits, t)

    def _show_hits(self, rows):
        self.hits.clear()
        for r in rows:
            it = self.hits.addItem(f"{r['disease']}   ({r['icd10'] or '-'})")
            self.hits.item(self.hits.count() - 1).setData(Qt.ItemDataRole.UserRole, r["disease"])

    def _open(self, name):
        def show(d):
            if not d:
                return
            parts = [f"<h2>{d['disease_name']}</h2><p><small>UCG section {d['section_number']}, PDF page {d['start_pdf_page']}</small></p>"]
            for k in ("clinical_features", "investigations", "differential_diagnosis", "management"):
                if d.get(k):
                    parts.append(f"<h4>{k.replace('_', ' ').title()}</h4><pre style='white-space:pre-wrap'>{d[k]}</pre>")
            self.detail.setHtml("".join(parts))
        self._run(self.be.disease, show, name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", help="base URL of the FastAPI service; omit to run in-process")
    a = ap.parse_args()
    app = QApplication(sys.argv)
    be = HttpBackend(a.api) if a.api else InProcessBackend()
    w = MainWindow(be); w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
