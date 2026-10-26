"""
Run:  python -m unittest discover -s tests -v      (or pytest, if installed)
Needs database/ucg.db (python -m etl.run_etl ...).  No network, no model files: the LLM is scripted.
"""
import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import db as dbmod
from agents.base import ToolCallLoop, ToolRegistry, Tool, extract_json
from agents.explanation_agent import verify_grounded
from agents.llm import LLM
from agents.orchestrator import Pipeline, Services
from agents.patient_agent import PatientAgent
from config import get_settings
from retrieval.hybrid import HybridRetriever
from rules import contraindications, dosing, red_flags, referral
from rules.engine import evaluate
from schemas import PatientCase


class ScriptedLLM(LLM):
    """Returns canned replies in order; records what it was sent."""
    name, available = "scripted", True

    def __init__(self, replies):
        self.replies, self.seen = list(replies), []

    def chat(self, messages, **kw):
        self.seen.append(messages)
        return self.replies.pop(0) if self.replies else "{}"


def tmp_settings():
    s = replace(get_settings())
    s.audit_db_path = Path(tempfile.mkdtemp()) / "audit.db"
    return s


class Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.s = tmp_settings()
        cls.con = dbmod.connect(cls.s.db_path, readonly=True)


class TestPatientParsing(Base):
    def test_child_case(self):
        c = PatientAgent().parse("3 year old girl, 14kg, temp 39.2 C, had a convulsion. no cough. allergic to penicillin")
        self.assertEqual(c.age_months, 36)
        self.assertEqual(c.weight_kg, 14)
        self.assertEqual(c.temp_c, 39.2)
        self.assertIn("convulsions", c.symptoms)
        self.assertIn("cough", c.negated_symptoms)
        self.assertNotIn("cough", c.symptoms)
        self.assertIn("penicillin", c.allergies)
        self.assertIs(c.pregnant, False)

    def test_implausible_values_rejected(self):
        c = PatientAgent().parse("adult, 900 kg, SpO2 250%")
        self.assertIsNone(c.weight_kg)
        self.assertIsNone(c.spo2)
        self.assertTrue(c.extraction_notes)

    def test_overrides_win(self):
        c = PatientAgent().parse("child with fever", {"weight_kg": 12, "age_years": 2})
        self.assertEqual((c.weight_kg, c.age_months), (12, 24))


class TestRules(Base):
    def case(self, **kw):
        return PatientCase(**kw)

    def test_unknown_input_never_fires(self):
        hits, und = red_flags.evaluate_all(self.con, self.case(symptoms=["headache"]))
        self.assertFalse([h for h in hits if h.rule_id == "RF-HYPOXIA"])
        self.assertIn("RF-HYPOXIA", [u.rule_id for u in und])

    def test_severe_bp_in_pregnancy(self):
        hits, _ = red_flags.evaluate_all(self.con, self.case(age_months=336, sex="F", pregnant=True, sbp=170, dbp=100))
        self.assertIn("RF-SEVERE-PREECLAMPSIA", [h.rule_id for h in hits])
        self.assertEqual(hits[0].severity, "emergency")

    def test_same_bp_not_pregnant_does_not_fire(self):
        hits, _ = red_flags.evaluate_all(self.con, self.case(age_months=480, sex="M", pregnant=False, sbp=170, dbp=100))
        self.assertNotIn("RF-SEVERE-PREECLAMPSIA", [h.rule_id for h in hits])

    def test_fast_breathing_by_age_band(self):
        f = lambda m, rr: "RF-FAST-BREATHING" in [h.rule_id for h in red_flags.evaluate_all(
            self.con, self.case(age_months=m, rr=rr))[0]]
        self.assertTrue(f(1, 60)); self.assertFalse(f(1, 59))
        self.assertTrue(f(6, 50)); self.assertFalse(f(6, 49))
        self.assertTrue(f(24, 40)); self.assertFalse(f(24, 39))
        self.assertFalse(f(100, 40))                                  # outside any band

    def test_unknown_condition_key_is_loud(self):
        with self.assertRaises(ValueError):
            evaluate({"bogus": 1}, PatientCase())

    def test_penicillin_allergy_blocks_amoxicillin_not_ceftriaxone(self):
        c = self.case(allergies=["penicillin"])
        self.assertEqual(contraindications.check_drug(self.con, c, "amoxicillin")[0].severity, "contraindicated")
        self.assertEqual(contraindications.check_drug(self.con, c, "ceftriaxone"), [])

    def test_ace_inhibitor_pregnancy(self):
        self.assertTrue(contraindications.check_drug(self.con, self.case(pregnant=True), "enalapril"))

    def test_dose_weight_based_and_cap(self):
        d, _ = dosing.calculate(self.con, self.case(weight_kg=14, age_months=36), "paracetamol", "febrile")
        self.assertEqual((d[0].dose_min, d[0].dose_max), (210, 210))
        d, _ = dosing.calculate(self.con, self.case(weight_kg=30, age_months=100), "diazepam")
        self.assertEqual(d[0].dose_max, 10); self.assertTrue(d[0].capped)

    def test_dose_needs_weight_and_unknown_drug(self):
        d, needs = dosing.calculate(self.con, self.case(), "diazepam")
        self.assertEqual((d, needs), ([], ["weight_kg"]))
        self.assertEqual(dosing.calculate(self.con, self.case(weight_kg=10), "amoxicillin"), ([], []))

    def test_adult_not_given_child_only_dose(self):
        d, _ = dosing.calculate(self.con, self.case(weight_kg=60, age_months=360), "dextrose 10%")
        self.assertEqual(d, [])

    def test_referral(self):
        hits, _ = red_flags.evaluate_all(self.con, self.case(spo2=85))
        self.assertEqual(referral.advise(self.con, "HC3", hits, "HC2").decision, "refer_urgent")
        self.assertEqual(referral.advise(self.con, "HC3", [], "HC2, HC4").higher_level_components, ["HC4"])
        self.assertEqual(referral.advise(self.con, "HC2", [], "HC4").decision, "refer_routine")
        self.assertEqual(referral.advise(self.con, "H", [], "HC4").decision, "manage_here")


class TestRetrieval(Base):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.h = HybridRetriever(cls.con, cls.s)

    def top(self, q, mode="hybrid", n=3):
        return [d.disease_name for d in self.h.retrieve(q, mode, k_diseases=n).diseases]

    def test_fts_exact_topics(self):
        self.assertEqual(self.top("rheumatic fever prophylaxis", "fts", 1), ["Rheumatic Fever"])
        self.assertIn("Paracetamol Poisoning", self.top("paracetamol overdose", "fts"))

    def test_hybrid_finds_febrile_convulsions(self):
        self.assertIn("Febrile Convulsions", self.top("child with fever and convulsions"))

    def test_autocomplete(self):
        self.assertEqual(self.h.fts.suggest("hypogly")[0]["disease_name"], "Hypoglycaemia")

    def test_injection_like_input_is_safe(self):
        self.h.retrieve('x"); DROP TABLE diseases;--', "fts")
        self.assertGreater(self.con.execute("select count(*) from diseases").fetchone()[0], 300)

    def test_semantic_failure_degrades_not_crashes(self):
        h = HybridRetriever(self.con, self.s)
        h.semantic = None; h.semantic_error = "simulated"
        r = h.retrieve("malaria")
        self.assertTrue(r.diseases); self.assertTrue(r.degraded)


class TestToolLoop(unittest.TestCase):
    def reg(self):
        r = ToolRegistry()
        r.register(Tool("add", "add numbers", {"properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                                               "required": ["a", "b"]}, lambda a, b: a + b))
        return r

    def test_call_then_final(self):
        llm = ScriptedLLM(['<tool_call>{"name":"add","arguments":{"a":2,"b":3}}</tool_call>', "The answer is 5"])
        res = ToolCallLoop(llm, self.reg()).run("sys", "2+3?")
        self.assertEqual(res.final_text, "The answer is 5")
        self.assertEqual(res.steps[0]["tool"], "add"); self.assertTrue(res.steps[0]["ok"])
        self.assertIn("<tool_response", llm.seen[1][-1]["content"])

    def test_bad_args_and_unknown_tool_go_back_to_model(self):
        llm = ScriptedLLM(['<tool_call>{"name":"add","arguments":{"a":"x"}}</tool_call>',
                           '<tool_call>{"name":"nope","arguments":{}}</tool_call>', "done"])
        res = ToolCallLoop(llm, self.reg()).run("s", "u")
        self.assertEqual([s["ok"] for s in res.steps if s["type"] == "tool"], [False, False])

    def test_runaway_model_is_stopped(self):
        call = '<tool_call>{"name":"add","arguments":{"a":1,"b":1}}</tool_call>'
        res = ToolCallLoop(ScriptedLLM([call] * 10 + ["final"]), self.reg(), max_steps=3).run("s", "u")
        self.assertEqual(res.stopped, "max_steps")

    def test_extract_json_from_prose(self):
        self.assertEqual(extract_json('Sure! ```json\n{"a": {"b": 1}}\n``` thanks'), {"a": {"b": 1}})


class TestGrounding(unittest.TestCase):
    CTX = "paracetamol 210 mg every 6 hours. amoxicillin is contraindicated."

    def test_grounded(self):
        self.assertEqual(verify_grounded("Give paracetamol 210 mg every 6 hours.", self.CTX), [])

    def test_invented_number_or_drug_detected(self):
        self.assertIn("500", verify_grounded("Give 500 mg", self.CTX))
        self.assertIn("ceftriaxone", verify_grounded("Use ceftriaxone", self.CTX))


class TestPipeline(Base):
    def pipe(self, llm=None):
        s = tmp_settings()
        return Pipeline(Services.build(s, llm=llm if llm is not None else LLM()))

    CASE = ("3 year old girl, 14kg, fever 39.2C, had a convulsion at home. allergic to penicillin. "
            "she is not able to drink")

    def test_end_to_end_deterministic(self):
        r = self.pipe().run(self.CASE)
        self.assertEqual(r["assessment"]["mode"], "deterministic")
        self.assertIn("RF-CHILD-NO-DRINK", [h["rule_id"] for h in r["red_flags"]])
        self.assertEqual(r["assessment"]["referral"]["decision"], "refer_urgent")
        names = [d["disease_name"] for d in r["assessment"]["differential"]]
        self.assertIn("Febrile Convulsions", names)
        blocked = {m["drug"] for m in r["medications"] if m["status"] == "contraindicated"}
        self.assertTrue(any(m["drug"] == "amoxicillin" for m in r["medications"]) <= ("amoxicillin" in blocked))
        self.assertIn("ALERTS", r["explanation"]["text"])
        self.assertTrue(r["decision_id"])

    def test_llm_loop_hallucinated_disease_ignored(self):
        # reply order = order of LLM calls in the pipeline: 1 patient symptom extraction, 2-3 clinical tool loop, 4 summary
        replies = ['{"symptoms": []}',
                   '<tool_call>{"name":"check_red_flags","arguments":{}}</tool_call>',
                   json.dumps({"nominated": ["Febrile Convulsions", "Imaginary Syndrome"], "questions": ["Any neck stiffness?"]}),
                   "The child has danger signs and needs urgent referral as listed in the alerts above."]
        p = self.pipe(ScriptedLLM(replies))
        r = p.run(self.CASE)
        a = r["assessment"]
        self.assertEqual(a["mode"], "llm+tools")
        self.assertTrue(any("Imaginary Syndrome" in n for n in a["notes"]))
        self.assertTrue(next(d for d in a["differential"] if d["disease_name"] == "Febrile Convulsions")["llm_nominated"])
        self.assertEqual(a["questions"], ["Any neck stiffness?"])
        self.assertEqual(a["tool_trace"][0]["tool"], "check_red_flags")

    def test_ungrounded_llm_summary_is_rejected(self):
        replies = ['{"symptoms": []}', json.dumps({"nominated": [], "questions": []}), "Give 999 mg of ceftriaxone now."]
        r = self.pipe(ScriptedLLM(replies)).run(self.CASE)
        self.assertEqual(r["explanation"]["mode"], "template")
        self.assertTrue(r["explanation"]["fallback_reason"].startswith("ungrounded"))
        self.assertNotIn("999", r["explanation"]["text"])

    def test_llm_failure_does_not_break_pipeline(self):
        class Boom(ScriptedLLM):
            def chat(self, *a, **k): raise RuntimeError("model crashed")
        r = self.pipe(Boom([])).run(self.CASE)
        self.assertTrue(r["assessment"]["differential"])

    def test_audit_roundtrip_and_feedback(self):
        p = self.pipe()
        r = p.run(self.CASE)
        rec = p.sv.audit.get(r["decision_id"])
        self.assertEqual(rec["versions"]["rules"], r["versions"]["rules"])
        self.assertIn("red_flags", rec)
        p.sv.audit.add_feedback(r["decision_id"], "agree", "ok", "tester")
        self.assertEqual(p.sv.audit.get(r["decision_id"])["feedback"][0]["verdict"], "agree")

    def test_raw_text_can_be_withheld(self):
        s = tmp_settings(); s.audit_store_raw_text = False
        p = Pipeline(Services.build(s, llm=LLM()))
        r = p.run(self.CASE)
        self.assertIsNone(p.sv.audit.get(r["decision_id"])["input_text"])


if __name__ == "__main__":
    unittest.main()
