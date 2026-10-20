//! Calibration math: tempered softmax, confidence, rounding. Ported from the official
//! Laya Rust SDK (laya-rust/src/calibration.rs, Apache-2.0,
//! https://github.com/NandhaKishorM/laya), adapted to this crate's config shape.

/// Namespace for calibration math.
pub struct Calibration;

/// Question types in qtype-tensor order.
pub const QTYPES: [&str; 3] = ["choice", "score", "noul"];

impl Calibration {
    /// Lowest temperature that will be applied. A fitted temperature below 1 sharpens logits
    /// instead of softening them; the shipped `choice:11+` bucket of 0.1006 multiplies them
    /// roughly tenfold, publishing a 0.24 top probability as 0.99. A caller gating on confidence
    /// would be told a coin flip is a certainty, so such a value is refused rather than applied.
    pub const TEMP_MIN: f64 = 0.5;

    /// Highest temperature that will be applied.
    pub const TEMP_MAX: f64 = 5.0;

    /// A usable temperature: confined to [TEMP_MIN, TEMP_MAX], non-finite falling back to 1.0.
    pub fn clamp_temperature(t: f64) -> f64 {
        if !t.is_finite() {
            1.0
        } else {
            t.clamp(Self::TEMP_MIN, Self::TEMP_MAX)
        }
    }

    /// The calibration bucket key for a question type and option count, e.g. "choice:3-5".
    pub fn temp_bucket(qtype: usize, option_count: usize) -> String {
        let size = if option_count <= 2 {
            "2"
        } else if option_count <= 5 {
            "3-5"
        } else if option_count <= 10 {
            "6-10"
        } else {
            "11+"
        };
        format!("{}:{size}", QTYPES[qtype])
    }

    /// Tempered softmax over the first `option_count` logits of one row.
    pub fn probabilities(logits: &[f32], option_count: usize, temperature: f64) -> Vec<f64> {
        let k = option_count.min(logits.len());
        if k == 0 {
            return Vec::new();
        }
        let mut p: Vec<f64> = logits[..k]
            .iter()
            .map(|&z| f64::from(z) / temperature)
            .collect();
        let max = p.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let mut sum = 0.0;
        for v in &mut p {
            *v = (*v - max).exp();
            sum += *v;
        }
        for v in &mut p {
            *v /= sum;
        }
        p
    }

    /// Softmax over a whole row, used for the action head. No temperature scaling.
    pub fn softmax(logits: &[f32]) -> Vec<f64> {
        if logits.is_empty() {
            return Vec::new();
        }
        let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut p: Vec<f64> = logits.iter().map(|&z| f64::from(z - max).exp()).collect();
        let sum: f64 = p.iter().sum();
        for v in &mut p {
            *v /= sum;
        }
        p
    }

    /// Normalized Shannon entropy confidence, 1 - H(p) / log k, clipped to [0, 1].
    pub fn confidence_from_probs(p: &[f64], option_count: usize) -> f64 {
        if option_count < 2 {
            return 1.0;
        }
        let k = option_count.min(p.len());
        let mut entropy = 0.0;
        for &pi in &p[..k] {
            let clipped = pi.clamp(1e-12, 1.0);
            entropy -= pi * clipped.ln();
        }
        (1.0 - entropy / (k as f64).ln()).clamp(0.0, 1.0)
    }

    /// Probability mass on the answer being reported: max(p[:option_count]). This is the
    /// quantity temperature scaling fits and ECE figures are computed on.
    pub fn answer_confidence(p: &[f64], option_count: usize) -> f64 {
        if option_count < 1 {
            return 1.0;
        }
        let k = option_count.min(p.len());
        let max = p[..k].iter().copied().fold(f64::NEG_INFINITY, f64::max);
        max.clamp(0.0, 1.0)
    }

    /// The expected level of a score answer: probability-weighted mean of level indices.
    pub fn expected_score(p: &[f64]) -> f64 {
        p.iter().enumerate().map(|(i, &pi)| i as f64 * pi).sum()
    }

    /// Index of the largest probability, ties to the lowest index (NumPy argmax).
    pub fn arg_max(p: &[f64]) -> usize {
        let mut best = 0;
        for (i, &v) in p.iter().enumerate().skip(1) {
            if v > p[best] {
                best = i;
            }
        }
        best
    }

    /// Python's round(x, 4): round-half-to-even at 4 decimal places.
    pub fn round4(value: f64) -> f64 {
        if !value.is_finite() {
            return value;
        }
        (value * 10_000.0).round_ties_even() / 10_000.0
    }
}
