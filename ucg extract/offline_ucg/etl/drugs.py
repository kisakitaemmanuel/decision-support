"""etl/drugs.py - drug lexicon + extraction of drug mentions from a disease's management text."""
from __future__ import annotations

import re

DRUGS = """paracetamol ibuprofen diclofenac aspirin acetylsalicylic_acid morphine codeine tramadol pethidine diazepam
phenobarbital phenytoin carbamazepine valproate haloperidol chlorpromazine amitriptyline fluoxetine amoxicillin
ampicillin benzylpenicillin phenoxymethylpenicillin benzathine_benzylpenicillin procaine_benzylpenicillin cloxacillin
flucloxacillin ceftriaxone cefotaxime ceftazidime cefalexin cefixime gentamicin amikacin azithromycin erythromycin
clarithromycin doxycycline tetracycline ciprofloxacin ofloxacin levofloxacin metronidazole cotrimoxazole
co-trimoxazole nitrofurantoin chloramphenicol clindamycin vancomycin meropenem rifampicin isoniazid pyrazinamide
ethambutol streptomycin fluconazole amphotericin_b flucytosine nystatin clotrimazole griseofulvin ketoconazole
albendazole mebendazole praziquantel ivermectin artemether lumefantrine artesunate quinine amodiaquine sulfadoxine
pyrimethamine primaquine chloroquine dihydroartemisinin piperaquine tenofovir lamivudine dolutegravir efavirenz
zidovudine abacavir nevirapine lopinavir ritonavir atazanavir aciclovir acyclovir oseltamivir salbutamol
prednisolone hydrocortisone dexamethasone beclomethasone budesonide adrenaline epinephrine atropine furosemide
spironolactone hydrochlorothiazide amlodipine nifedipine enalapril captopril lisinopril atenolol propranolol
methyldopa hydralazine labetalol digoxin glyceryl_trinitrate isosorbide simvastatin atorvastatin warfarin heparin
enoxaparin clopidogrel metformin glibenclamide gliclazide insulin glucagon dextrose oxytocin misoprostol
ergometrine magnesium_sulphate magnesium_sulfate tranexamic_acid folic_acid ferrous_sulphate ferrous_sulfate
vitamin_a vitamin_k zinc oral_rehydration_salts normal_saline ringer's_lactate sodium_bicarbonate potassium_chloride
omeprazole ranitidine metoclopramide ondansetron hyoscine loperamide bisacodyl lactulose cetirizine chlorphenamine
promethazine levonorgestrel medroxyprogesterone norethisterone ethinylestradiol tamoxifen methotrexate vincristine
cyclophosphamide lidocaine ketamine bupivacaine naloxone flumazenil acetylcysteine activated_charcoal""".split()

DRUG_NAMES = sorted({d.replace("_", " ") for d in DRUGS}, key=len, reverse=True)
_DRUG_RE = re.compile(r"\b(" + "|".join(re.escape(d) for d in DRUG_NAMES) + r")\b", re.IGNORECASE)


def find_drugs(text: str) -> dict[str, str]:
    """Return {drug_lowercase: snippet (the line where it first appears)}."""
    found: dict[str, str] = {}
    for line in (text or "").split("\n"):
        for m in _DRUG_RE.finditer(line):
            name = m.group(1).lower()
            if name not in found:
                found[name] = line.strip()[:240]
    return found
