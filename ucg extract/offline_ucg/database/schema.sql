-- offline_ucg/database/schema.sql  (applied by etl/run_etl.py; safe to read as documentation)
PRAGMA foreign_keys = ON;

CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);

CREATE TABLE etl_runs (
    run_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    stage       TEXT NOT NULL,
    started_at  TEXT NOT NULL,
    finished_at TEXT,
    rows        INTEGER,
    status      TEXT NOT NULL,                 -- ok | failed
    detail      TEXT
);

CREATE TABLE chapters (
    chapter_number INTEGER PRIMARY KEY,
    chapter_title  TEXT NOT NULL
);

CREATE TABLE diseases (
    disease_name           TEXT PRIMARY KEY NOT NULL CHECK (length(trim(disease_name)) > 0),
    entry_id               TEXT NOT NULL UNIQUE,
    section_number         TEXT NOT NULL,
    icd10_code             TEXT,
    chapter_number         INTEGER REFERENCES chapters(chapter_number),
    category               TEXT,
    parent_condition       TEXT,
    description            TEXT,
    causes                 TEXT,
    risk_factors           TEXT,
    clinical_features      TEXT,
    differential_diagnosis TEXT,
    investigations         TEXT,
    management             TEXT,
    prevention             TEXT,
    complications          TEXT,
    levels_of_care         TEXT,
    start_pdf_page         INTEGER,
    end_pdf_page           INTEGER
);
CREATE INDEX idx_diseases_chapter ON diseases(chapter_number);
CREATE INDEX idx_diseases_icd10   ON diseases(icd10_code);
CREATE INDEX idx_diseases_name_nc ON diseases(disease_name COLLATE NOCASE);

-- retrieval unit: one chunk = a slice of one section of one disease
CREATE TABLE chunks (
    chunk_id     INTEGER PRIMARY KEY,
    disease_name TEXT NOT NULL REFERENCES diseases(disease_name) ON DELETE CASCADE,
    section      TEXT NOT NULL,                -- clinical_features | investigations | ...
    chunk_idx    INTEGER NOT NULL,
    text         TEXT NOT NULL,
    page         INTEGER,
    UNIQUE (disease_name, section, chunk_idx)
);
CREATE INDEX idx_chunks_disease ON chunks(disease_name);

CREATE VIRTUAL TABLE chunks_fts USING fts5(
    disease_name, section, text,
    content='chunks', content_rowid='chunk_id',
    tokenize='porter unicode61 remove_diacritics 2', prefix='2 3 4'
);
CREATE VIRTUAL TABLE disease_names_fts USING fts5(
    disease_name, icd10_code,
    content='diseases', content_rowid='rowid',
    tokenize='unicode61 remove_diacritics 2', prefix='1 2 3 4'
);

-- drugs mentioned in each disease's management text (ETL: etl/run_etl.py :: stage_drugs)
CREATE TABLE disease_drugs (
    disease_name TEXT NOT NULL REFERENCES diseases(disease_name) ON DELETE CASCADE,
    drug         TEXT NOT NULL,
    snippet      TEXT NOT NULL,
    PRIMARY KEY (disease_name, drug)
);
CREATE INDEX idx_disease_drugs_drug ON disease_drugs(drug);

-- ---------------------------------------------------------------- rules (python + SQLite)
-- condition_json grammar is documented in rules/engine.py
CREATE TABLE red_flag_rules (
    rule_id        TEXT PRIMARY KEY,
    name           TEXT NOT NULL,
    condition_json TEXT NOT NULL,
    severity       TEXT NOT NULL CHECK (severity IN ('emergency','urgent','warning')),
    refer          INTEGER NOT NULL DEFAULT 0,
    message        TEXT NOT NULL,
    action         TEXT NOT NULL DEFAULT '',
    source_ref     TEXT NOT NULL DEFAULT '',
    reviewed       INTEGER NOT NULL DEFAULT 0,   -- 1 only after a clinician signed the rule off
    enabled        INTEGER NOT NULL DEFAULT 1
);
CREATE TABLE contraindication_rules (
    rule_id        TEXT PRIMARY KEY,
    drug_regex     TEXT NOT NULL,
    condition_json TEXT NOT NULL,
    severity       TEXT NOT NULL CHECK (severity IN ('contraindicated','avoid','caution')),
    message        TEXT NOT NULL,
    source_ref     TEXT NOT NULL DEFAULT '',
    reviewed       INTEGER NOT NULL DEFAULT 0,
    enabled        INTEGER NOT NULL DEFAULT 1
);
CREATE TABLE dose_rules (
    rule_id          TEXT PRIMARY KEY,
    drug             TEXT NOT NULL,
    indication       TEXT NOT NULL,
    route            TEXT NOT NULL,
    dose_per_kg_min  REAL NOT NULL,
    dose_per_kg_max  REAL NOT NULL,
    dose_unit        TEXT NOT NULL CHECK (dose_unit IN ('mg','ml','micrograms')),
    max_single_dose  REAL,                       -- same unit as dose_unit
    interval_hours   REAL,
    notes            TEXT NOT NULL DEFAULT '',
    min_age_months   REAL,
    max_age_months   REAL,
    source_ref       TEXT NOT NULL DEFAULT '',
    reviewed         INTEGER NOT NULL DEFAULT 0,
    enabled          INTEGER NOT NULL DEFAULT 1,
    CHECK (dose_per_kg_max >= dose_per_kg_min)
);
CREATE INDEX idx_dose_rules_drug ON dose_rules(drug COLLATE NOCASE);

CREATE TABLE referral_levels (
    code TEXT PRIMARY KEY,
    rank INTEGER NOT NULL UNIQUE,
    name TEXT NOT NULL
);
INSERT INTO referral_levels(code, rank, name) VALUES
  ('VHT',1,'Village health team'),('HC2',2,'Health centre II'),('HC3',3,'Health centre III'),
  ('HC4',4,'Health centre IV'),('H',5,'General hospital'),('RR',6,'Regional referral hospital'),
  ('NR',7,'National referral hospital');

CREATE TABLE vector_index_meta (
    id          INTEGER PRIMARY KEY CHECK (id = 1),
    embedder    TEXT NOT NULL,
    dim         INTEGER NOT NULL,
    n_vectors   INTEGER NOT NULL,
    built_at    TEXT NOT NULL,
    npz_file    TEXT NOT NULL,
    faiss_file  TEXT
);
