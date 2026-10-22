//! Multi-checkpoint routing: a port of the official Python SDK Router
//! (laya/router.py + laya/lang.py), reduced to the two checkpoints this
//! crate can run.
//!
//! Checkpoints of the official SDK:
//!   english          ModernBERT-large, 512 tokens  (implemented here)
//!   typed-decisions  ModernBERT-large, 1024 tokens (implemented here)
//!   multilingual     mmBERT-base, 1024 tokens      (STUB: different encoder
//!                    architecture that src/model.rs does not implement)
//!
//! Detection fidelity vs the official laya/lang.py:
//! - The script-range table and script counting are an exact port
//!   (including the "other" bucket for unlisted alphabetic scripts and the
//!   Latin boundary at 0x02B0 / Latin Extended Additional / fullwidth).
//! - The Latin-script language guess is a REDUCED port: diacritic rate
//!   (same 0.02 threshold, same non-English diacritic set) plus function
//!   word lists for en/fr/de/es/pt/it/nl/sv/ro. The official lists are
//!   longer and also cover romanized Bangla (bn) and Azerbaijani (az).
//! - Not ported: per-word occurrence/collision weighting (distinct word
//!   hits count once), the acronym / all-caps veto, the segment scan for a
//!   foreign line inside mostly-English text, the per-leaf analysis of
//!   nested state values, and the word-run rules of the mixed-script
//!   promotion. Latin text left undecided routes to the english checkpoint
//!   unless its diacritic rate clears the non-English floor, mirroring the
//!   official tie-break ("undecided is not English" only with evidence).
//! - The official Router also routes on explicit lang hints, hub revision
//!   pinning and lazy eviction; none of that applies to a local-directory
//!   loader.

use std::cell::RefCell;
use std::collections::HashMap;

use burn::tensor::Device;
use crate::inference::{Answer, QuestionInput, RlAgent};
use serde_json::Value;

/// Which checkpoint a request is routed to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Route {
    English,
    TypedDecisions,
    /// Not implemented: the mmBERT-base encoder. Predicting on this route
    /// returns [RouterError::MultilingualUnimplemented].
    Multilingual,
}

impl Route {
    pub fn name(self) -> &'static str {
        match self {
            Route::English => "english",
            Route::TypedDecisions => "typed-decisions",
            Route::Multilingual => "multilingual",
        }
    }
}

/// What went wrong on a routed prediction.
#[derive(Debug)]
#[derive(Clone)]
pub enum RouterError {
    /// The request needs the mmBERT-base multilingual checkpoint, which this
    /// crate does not implement (only the ModernBERT-large pipeline in
    /// src/model.rs exists). The message is suitable for display.
    MultilingualUnimplemented(String),
    /// A checkpoint directory the router wanted is missing or unreadable.
    Load(String),
}

/// Script ranges the english (ModernBERT-large, 50k English BPE) checkpoint
/// cannot read. Exact port of the official _SCRIPT_RANGES.
const SCRIPT_RANGES: &[(&str, &[(u32, u32)])] = &[
    ("greek", &[(0x0370, 0x03FF), (0x1F00, 0x1FFF)]),
    ("cyrillic", &[(0x0400, 0x052F), (0x2DE0, 0x2DFF), (0xA640, 0xA69F)]),
    ("armenian", &[(0x0530, 0x058F)]),
    ("hebrew", &[(0x0590, 0x05FF)]),
    (
        "arabic",
        &[(0x0600, 0x06FF), (0x0750, 0x077F), (0x08A0, 0x08FF), (0xFB50, 0xFDFF), (0xFE70, 0xFEFF)],
    ),
    ("devanagari", &[(0x0900, 0x097F), (0xA8E0, 0xA8FF)]),
    ("bengali", &[(0x0980, 0x09FF)]),
    ("gurmukhi", &[(0x0A00, 0x0A7F)]),
    ("gujarati", &[(0x0A80, 0x0AFF)]),
    ("oriya", &[(0x0B00, 0x0B7F)]),
    ("tamil", &[(0x0B80, 0x0BFF)]),
    ("telugu", &[(0x0C00, 0x0C7F)]),
    ("kannada", &[(0x0C80, 0x0CFF)]),
    ("malayalam", &[(0x0D00, 0x0D7F)]),
    ("sinhala", &[(0x0D80, 0x0DFF)]),
    ("thai", &[(0x0E00, 0x0E7F)]),
    ("lao", &[(0x0E80, 0x0EFF)]),
    ("tibetan", &[(0x0F00, 0x0FFF)]),
    ("myanmar", &[(0x1000, 0x109F)]),
    ("georgian", &[(0x10A0, 0x10FF)]),
    ("ethiopic", &[(0x1200, 0x137F)]),
    ("khmer", &[(0x1780, 0x17FF)]),
    ("hangul", &[(0x1100, 0x11FF), (0x3130, 0x318F), (0xAC00, 0xD7AF)]),
    ("kana", &[(0x3040, 0x309F), (0x30A0, 0x30FF), (0x31F0, 0x31FF)]),
    ("han", &[(0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0xF900, 0xFAFF)]),
];

/// Letters ordinary English does not use (port of the official
/// _NON_EN_DIACRITICS, already lowercase as the matcher runs on lowered text).
const NON_EN_DIACRITICS: &str = "àâäãáåçéèêëíìîïñóòôöõøúùûüýÿßæœăâîșțşţąćęłńśźżčďěňřšťůžőűğıāēģīķļņūžđə";

/// A diacritic rate above this is evidence the text is not English.
const NON_EN_DIACRITIC_RATE: f64 = 0.02;
/// Non-Latin letters carry the state at this fraction even when Latin is the
/// plurality, or at NON_LATIN_MIN_FRACTION with NON_LATIN_MIN_LETTERS letters.
const NON_LATIN_FRACTION: f64 = 0.2;
const NON_LATIN_MIN_FRACTION: f64 = 0.1;
const NON_LATIN_MIN_LETTERS: usize = 10;

/// English function words (exact port of the official _STOP["en"]).
const STOP_EN: &[&str] = &[
    "the", "and", "is", "are", "was", "were", "to", "of", "in", "for", "with", "that", "this",
    "it", "you", "have", "has", "not", "but", "on", "at", "be", "as", "from", "will", "can",
    "would", "there", "their", "what", "which", "please", "we", "i",
];

const _PLACEHOLDER: () = ();

/// Reduced non-English function-word lists, curated so no word appears in
/// two lists and none appears in STOP_EN (the official _SHARED_WORDS
/// exclusion applied by hand). The full official lists are longer and cover
/// more languages (bn, az, ...).
const STOP_NON_EN: &[(&str, &[&str])] = &[
    (
        "fr",
        &[
            "les", "des", "une", "est", "pour", "dans", "que", "qui", "avec", "sur", "pas",
            "plus", "nous", "vous", "être", "cette", "mais", "sont", "aux", "elle", "elles",
            "merci", "bonjour", "très", "tout", "tous", "veux", "peux", "dois",
        ],
    ),
    (
        "de",
        &[
            "der", "die", "das", "und", "ist", "ein", "eine", "den", "dem", "nicht", "mit",
            "für", "auf", "von", "zu", "sich", "auch", "werden", "wurde", "haben", "sind",
            "oder", "aber", "ich", "wir", "diese", "bitte", "heute", "kann",
        ],
    ),
    (
        "es",
        &[
            "los", "las", "por", "con", "para", "una", "se", "del", "como", "pero", "son",
            "está", "este", "esta", "más", "muy", "sus", "quiero", "necesito", "gracias",
            "porque", "también", "nada",
        ],
    ),
    (
        "pt",
        &[
            "os", "as", "em", "um", "uma", "com", "não", "é", "do", "da", "dos", "das",
            "muito", "pelo", "pela", "você", "voce", "obrigado", "então", "entao", "depois",
            "estou", "gostaria", "também", "tambem",
        ],
    ),
    (
        "it",
        &[
            "il", "lo", "gli", "che", "di", "per", "non", "è", "si", "del", "della", "sono",
            "questo", "questa", "anche", "come", "più", "nella", "alla", "grazie", "ciao",
        ],
    ),
    (
        "nl",
        &[
            "het", "een", "van", "op", "te", "dat", "niet", "voor", "zijn", "aan", "door",
            "maar", "ook", "deze", "naar", "wordt",
        ],
    ),
    (
        "sv",
        &[
            "jag", "är", "och", "inte", "att", "från", "till", "behöver", "skulle", "vill",
            "måste", "också", "säger", "utan", "kommer", "hjälp", "lösenord", "faktura",
        ],
    ),
    (
        "ro",
        &[
            "și", "să", "este", "sunt", "care", "pentru", "din", "dar", "după", "fără", "lui",
            "fost", "acum", "vreau", "trebuie", "foarte",
        ],
    ),
];

/// Detection result for one state (reduced port of laya.lang.analyse).
#[derive(Debug, Clone)]
pub struct Detection {
    /// Dominant script: "latin", "cyrillic", "han", ... or "unknown".
    pub script: String,
    /// Best-effort language code for Latin text; None when undecided.
    pub language: Option<&'static str>,
    /// Whether the state is safe to hand to the english checkpoint.
    pub is_english: bool,
    /// Fraction of alphabetic characters that are not Latin.
    pub non_latin_fraction: f64,
    /// Diacritic rate of the lowered text (Latin script only).
    pub diacritic_rate: f64,
}

fn is_latin_cp(cp: u32) -> bool {
    cp < 0x02B0
        || (0x1E00..=0x1EFF).contains(&cp)
        || (0xFF21..=0xFF3A).contains(&cp)
        || (0xFF41..=0xFF5A).contains(&cp)
}

fn script_of_cp(cp: u32) -> Option<&'static str> {
    SCRIPT_RANGES
        .iter()
        .find(|(_, ranges)| ranges.iter().any(|&(lo, hi)| lo <= cp && cp <= hi))
        .map(|(name, _)| *name)
}

/// Count alphabetic characters per script (exact port of the official
/// _script_counts, including the "other" bucket for scripts no range claims).
fn script_counts(text: &str) -> HashMap<&'static str, usize> {
    let mut counts: HashMap<&'static str, usize> = HashMap::new();
    let mut latin = 0usize;
    for ch in text.chars() {
        if !ch.is_alphabetic() {
            continue;
        }
        let cp = ch as u32;
        if is_latin_cp(cp) {
            latin += 1;
        } else {
            let name = script_of_cp(cp).unwrap_or("other");
            *counts.entry(name).or_default() += 1;
        }
    }
    counts.insert("latin", latin);
    counts
}

/// Maximal runs of letters (approximates the official _WORD regex, which
/// also excludes digit-only and identifier-shaped runs; not ported).
fn words(text: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut start: Option<usize> = None;
    for (idx, ch) in text.char_indices() {
        if ch.is_alphabetic() {
            if start.is_none() {
                start = Some(idx);
            }
        } else if let Some(s) = start.take() {
            out.push(&text[s..idx]);
        }
    }
    if let Some(s) = start {
        out.push(&text[s..]);
    }
    out
}

fn diacritic_count_and_len(lowered: &str) -> (usize, usize) {
    (
        lowered.chars().filter(|c| NON_EN_DIACRITICS.contains(*c)).count(),
        lowered.chars().count(),
    )
}

fn diacritic_rate(lowered: &str) -> f64 {
    let (diac, len) = diacritic_count_and_len(lowered);
    diac as f64 / len.max(1) as f64
}

/// Reduced port of latin_profile: (language, english_hits, looks_non_english).
/// A non-English language is only named when it matched at least two
/// function words and beats English by the official margin, or matches two
/// with non-English diacritics present.
fn latin_profile(lowered: &str) -> (Option<&'static str>, usize, bool) {
    let rate = diacritic_rate(lowered);
    let non_english = rate >= NON_EN_DIACRITIC_RATE;
    let ws = words(lowered);
    if ws.len() < 4 {
        return (None, 0, non_english);
    }
    let distinct: std::collections::HashSet<&str> = ws.iter().copied().collect();
    let en = distinct.iter().filter(|w| STOP_EN.contains(w)).count();
    let mut best: Option<(&'static str, usize)> = None;
    for (lg, list) in STOP_NON_EN {
        let hits = distinct.iter().filter(|w| list.contains(w)).count();
        if hits > 0 && best.map(|(_, b)| hits > b).unwrap_or(true) {
            best = Some((lg, hits));
        }
    }
    let lang = match best {
        Some((lg, hits)) if hits >= 2.max(en + 2) => Some(lg),
        Some((lg, hits)) if non_english && hits >= 2.max(en) => Some(lg),
        _ if en > 0 && !non_english => Some("en"),
        _ => None,
    };
    (lang, en, non_english)
}

/// Full detection for one already-flattened string (reduced port of
/// _analyse_text).
pub fn analyse_text(text: &str) -> Detection {
    let counts = script_counts(text);
    let total: usize = counts.values().sum();
    if total == 0 {
        return Detection {
            script: "unknown".into(),
            language: None,
            // No letters at all: the official analysis treats "unknown" as English.
            is_english: true,
            non_latin_fraction: 0.0,
            diacritic_rate: 0.0,
        };
    }
    // Dominant script, a named script winning ties against Latin.
    let mut script = *counts
        .iter()
        .max_by_key(|(k, v)| (**v, **k != "latin"))
        .unwrap()
        .0;
    let non_latin = 1.0 - *counts.get("latin").unwrap_or(&0) as f64 / total as f64;
    if script == "latin" {
        // Reduced port of the mixed-script promotion: non-Latin letters that
        // carry the state outvote the Latin plurality. The official version
        // additionally requires non-Latin word runs that are not proper
        // names; not ported.
        let n_non_latin = total - counts.get("latin").copied().unwrap_or(0);
        let carries = non_latin >= NON_LATIN_FRACTION
            || (non_latin >= NON_LATIN_MIN_FRACTION && n_non_latin >= NON_LATIN_MIN_LETTERS);
        if carries {
            if let Some((&named, _)) = counts.iter().filter(|(k, _)| **k != "latin").max_by_key(|(_, v)| **v) {
                script = named;
            }
        }
    }
    if script != "latin" {
        return Detection {
            script: script.into(),
            language: None,
            is_english: false,
            non_latin_fraction: non_latin,
            diacritic_rate: 0.0,
        };
    }
    let lowered = text.to_lowercase();
    let (lang, _en, looks_non_english) = latin_profile(&lowered);
    let is_english = lang == Some("en") || (lang.is_none() && !looks_non_english);
    Detection {
        script: "latin".into(),
        language: lang,
        is_english,
        non_latin_fraction: non_latin,
        diacritic_rate: diacritic_rate(&lowered),
    }
}

/// Flatten a JSON state to its string leaf values, capped at 4000 characters
/// (port of state_text; numbers and booleans contribute nothing).
pub fn state_text(state: &Value, mut max_chars: usize) -> String {
    let mut out = String::new();
    fn walk(v: &Value, out: &mut String, budget: &mut usize) {
        if *budget == 0 {
            return;
        }
        match v {
            Value::String(s) => {
                if s.len() > *budget {
                    let mut cut = (*budget).min(s.len());
                    while cut > 0 && !s.is_char_boundary(cut) {
                        cut -= 1;
                    }
                    out.push_str(&s[..cut]);
                    *budget = 0;
                } else {
                    out.push_str(s);
                    out.push(' ');
                    *budget -= s.len() + 1;
                }
            }
            Value::Array(items) => items.iter().for_each(|i| walk(i, out, budget)),
            Value::Object(map) => map.values().for_each(|i| walk(i, out, budget)),
            _ => {}
        }
    }
    walk(state, &mut out, &mut max_chars);
    out
}

/// Full detection for a state.
pub fn analyse(state: &Value) -> Detection {
    analyse_text(&state_text(state, 4000))
}

/// Routes requests to checkpoints, lazily loading each agent on first use of
/// its route and keeping it alive afterwards (an RlAgent holds the full f32
/// checkpoint in memory, roughly 1.6 GB for ModernBERT-large).
pub struct Router {
    device: Device,
    english_dir: Option<String>,
    typed_dir: Option<String>,
    multilingual_dir: Option<String>,
    english: RefCell<Option<RlAgent>>,
    typed: RefCell<Option<RlAgent>>,
    multilingual: RefCell<Option<RlAgent>>,
    forced: Option<Route>,
}

impl Router {
    /// Build a router over the local checkpoint directories ("model-english"
    /// and "model"); a missing directory disables that route. LAYA_MODEL=
    /// english|typed-decisions|auto (default auto) forces a route.
    pub fn new(device: &Device) -> Self {
        let forced = match std::env::var("LAYA_MODEL").as_deref() {
            Ok("english") | Ok("en") => Some(Route::English),
            Ok("typed-decisions") | Ok("typed_decisions") | Ok("typed") => Some(Route::TypedDecisions),
            Ok("multilingual") | Ok("multi") | Ok("ml") => Some(Route::Multilingual),
            Ok("auto") | Err(_) => None,
            Ok(other) => {
                eprintln!("laya: ignoring unknown LAYA_MODEL={other:?} (expected english|typed-decisions|auto)");
                None
            }
        };
        Self {
            device: device.clone(),
            english_dir: std::path::Path::new("model-english/rl_agent_config.json")
                .is_file()
                .then(|| "model-english".to_string()),
            typed_dir: std::path::Path::new("model/rl_agent_config.json")
                .is_file()
                .then(|| "model".to_string()),
            multilingual_dir: std::path::Path::new("model-multilingual/rl_agent_config.json")
                .is_file()
                .then(|| "model-multilingual".to_string()),
            english: RefCell::new(None),
            typed: RefCell::new(None),
            multilingual: RefCell::new(None),
            forced,
        }
    }

    /// Whether at least one loadable checkpoint directory was found.
    pub fn available(&self) -> bool {
        self.english_dir.is_some() || self.typed_dir.is_some() || self.multilingual_dir.is_some()
    }

    /// The route a state would take. Detection runs even when a route is
    /// forced, so callers can report what auto routing would have chosen.
    pub fn route(&self, state: &Value) -> (Route, Detection) {
        let det = analyse(state);
        if let Some(forced) = self.forced {
            return (forced, det);
        }
        let route = if det.is_english { Route::English } else { Route::Multilingual };
        // A missing directory falls back to the other checkpoint, so a
        // partial download still answers (with a notice).
        match route {
            Route::English if self.english_dir.is_none() => {
                eprintln!(
                    "laya: english checkpoint missing, falling back to {}",
                    Route::TypedDecisions.name()
                );
                (Route::TypedDecisions, det)
            }
            Route::TypedDecisions if self.typed_dir.is_none() => {
                eprintln!(
                    "laya: typed-decisions checkpoint missing, falling back to {}",
                    Route::English.name()
                );
                (Route::English, det)
            }
            r => (r, det),
        }
    }

    fn load(dir: &str, device: &Device) -> RlAgent {
        // LAYA_BURNPACK=1 (or burnpack present and safetensors absent) loads
        // the burn-native export: ~0.9s vs ~9s, identical answers.
        let has_burnpack = std::path::Path::new(&format!("{dir}/model.mpk")).is_file()
            && std::path::Path::new(&format!("{dir}/model.mpk.json")).is_file();
        let force_burnpack = std::env::var("LAYA_BURNPACK").as_deref() == Ok("1");
        let no_safetensors = !std::path::Path::new(&format!("{dir}/model.safetensors")).is_file();
        let mut agent = if has_burnpack && (force_burnpack || no_safetensors) {
            eprintln!("laya: loading {dir} from burnpack");
            RlAgent::load_burnpack(dir, device)
        } else {
            RlAgent::load(dir, device)
        };
        // LAYA_QUANT=1: Q8 MinMax weight quantization (~4x less weight memory).
        if std::env::var("LAYA_QUANT").as_deref() == Ok("1") {
            use burn::module::{Module, Quantizer};
            let scheme = device
                .settings()
                .quantization
                .scheme
                .with_value(burn::tensor::quantization::QuantValue::Q8S)
                .with_store(burn::tensor::quantization::QuantStore::Native);
            let t0 = std::time::Instant::now();
            let q = agent.model.clone().quantize_weights(&mut Quantizer::new(
                burn::tensor::quantization::Calibration::MinMax,
                scheme,
            ));
            agent.model = q;
            eprintln!("laya: Q8 quantized weights in {:?}", t0.elapsed());
        }
        agent
    }

    /// Route one request and run it on the chosen checkpoint. The agent for
    /// a route is loaded on first use and kept alive for later requests.
    pub fn predict(
        &self,
        state: &Value,
        questions: &[QuestionInput],
    ) -> Result<(Route, Vec<Answer>), RouterError> {
        let (route, det) = self.route(state);
        let cell = match route {
            Route::English => &self.english,
            Route::TypedDecisions => &self.typed,
            Route::Multilingual => &self.multilingual,
        };
        if cell.borrow().is_none() {
            let dir = match route {
                Route::English => self.english_dir.clone(),
                Route::TypedDecisions => self.typed_dir.clone(),
                Route::Multilingual => self.multilingual_dir.clone(),
            }
            .ok_or_else(|| {
                RouterError::Load(format!(
                    "state routed to the multilingual checkpoint (script={}) but model-multilingual/ was not found; download it per README, or force a route with LAYA_MODEL=english|typed-decisions.",
                    det.script
                ))
            })?;
            eprintln!("laya: loading {} checkpoint from {dir} ...", route.name());
            *cell.borrow_mut() = Some(Self::load(&dir, &self.device));
        }
        let guard = cell.borrow();
        let agent = guard.as_ref().unwrap();
        Ok((route, agent.system_one(&self.device, state, questions, None)))
    }

    /// Batched predict for the server: items each carry their own state and
    /// questions; routing happens per item (cheap CPU detection), items are
    /// grouped by route, and each group runs as ONE batched forward. Returns
    /// per-item (route, answers) in input order.
    pub fn predict_many(
        &self,
        items: &[(Value, Vec<QuestionInput>)],
    ) -> Vec<Result<(Route, Vec<Answer>), RouterError>> {
        use std::cell::RefCell;

        // Group item indices by detected route (forced route overrides).
        let mut groups: Vec<(Route, Vec<usize>)> = Vec::new();
        let mut routed: Vec<Result<Route, RouterError>> = Vec::with_capacity(items.len());
        for (i, (state, _)) in items.iter().enumerate() {
            // self.route returns (Route, Detection), not a Result.
            let (detected, _det) = self.route(state);
            let route = self.forced.unwrap_or(detected);
            routed.push(Ok(route));
            match groups.iter_mut().find(|(g, _)| *g == route) {
                Some((_, v)) => v.push(i),
                None => groups.push((route, vec![i])),
            }
        }

        // One batched forward per route group.
        let mut results: Vec<Option<Vec<Answer>>> = vec![None; items.len()];
        for (route, idxs) in &groups {
            let group: Vec<(Value, Vec<QuestionInput>)> = idxs
                .iter()
                .map(|i| items[*i].clone())
                .collect();
            let outcome = (|| -> Result<Vec<Vec<Answer>>, RouterError> {
                let cell = match route {
                    Route::English => &self.english,
                    Route::TypedDecisions => &self.typed,
                    Route::Multilingual => &self.multilingual,
                };
                if cell.borrow().is_none() {
                    let dir = match route {
                        Route::English => self.english_dir.clone(),
                        Route::TypedDecisions => self.typed_dir.clone(),
                        Route::Multilingual => self.multilingual_dir.clone(),
                    }
                    .ok_or_else(|| {
                        RouterError::Load(format!(
                            "state routed to {} but its checkpoint directory was not found",
                            route.name()
                        ))
                    })?;
                    eprintln!("laya: loading {} checkpoint from {dir} ...", route.name());
                    let loaded = Self::load(&dir, &self.device);
                    let _ = RefCell::new(()); // keep RefCell import used on all cfgs
                    *cell.borrow_mut() = Some(loaded);
                }
                let guard = cell.borrow();
                let agent = guard.as_ref().unwrap();
                Ok(agent.system_one_many(&self.device, &group, None))
            })();
            match outcome {
                Ok(per_item) => {
                    for (j, i) in idxs.iter().enumerate() {
                        results[*i] = Some(per_item[j].clone());
                    }
                }
                Err(e) => {
                    for i in idxs {
                        routed[*i] = Err(e.clone());
                    }
                }
            }
        }

        routed.into_iter()
            .enumerate()
            .map(|(i, r)| match (r, results[i].take()) {
                (Ok(route), Some(ans)) => Ok((route, ans)),
                (Ok(_), None) => Err(RouterError::Load("internal: missing batched result".into())),
                (Err(e), _) => Err(e),
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn det(s: &str) -> Detection {
        analyse_text(s)
    }

    #[test]
    fn english_prose_routes_english() {
        let d = det("The user was charged twice and wants a refund for the order they placed last week.");
        assert_eq!(d.script, "latin");
        assert_eq!(d.language, Some("en"));
        assert!(d.is_english);
    }

    #[test]
    fn chinese_routes_multilingual() {
        let d = det("用户询问如何修改收货地址并申请退款");
        assert!(!d.is_english);
        assert_eq!(d.script, "han");
    }

    #[test]
    fn cyrillic_routes_multilingual() {
        let d = det("Пользователь дважды списали деньги и хочет вернуть сумму");
        assert!(!d.is_english);
        assert_eq!(d.script, "cyrillic");
    }

    #[test]
    fn french_stopwords_beat_english_margin() {
        let d = det("Bonjour, je veux une facture pour cette commande et merci beaucoup pour votre aide");
        assert!(!d.is_english);
        assert_eq!(d.language, Some("fr"));
    }

    #[test]
    fn one_loanword_stays_english() {
        // A single accented proper noun must not pull English off the english route.
        let d = det("Please send the summary to José about the meeting we had");
        assert!(d.is_english);
    }

    #[test]
    fn no_letters_is_english() {
        let d = det("1234 ... !!!");
        assert!(d.is_english);
        assert_eq!(d.script, "unknown");
    }

    #[test]
    fn mixed_latin_majority_with_cjk_block_routes_multilingual() {
        let d = det("The order 请修改收货地址并申请退款谢谢 was placed last week with the card");
        assert!(!d.is_english);
    }

    #[test]
    fn state_json_string_leaves_are_used() {
        let v: Value = serde_json::json!({"user": {"msg": "用户要求退款"}, "meta": {"id": 7}});
        let d = analyse(&v);
        assert!(!d.is_english);
    }
}
