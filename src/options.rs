//! Explicit load-time configuration.
//!
//! The loaders previously sampled a handful of env vars at unpredictable
//! times (clamp mode mid-config-parse, theta override mid-geometry, pack
//! choice inside the router, dump dir per forward). All of that now resolves
//! ONCE into [LoadOptions] at the binary edge - [LoadOptions::from_env] is
//! the only env adapter - and is threaded explicitly.

/// Which artifact a directory load uses.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PackChoice {
    /// `model.q8.mpk` when present (plus sidecar), then `model.mpk`, then
    /// safetensors. The production default.
    #[default]
    Auto,
    /// Force the f32 burnpack.
    F32Pack,
    /// Force the Q8 pack.
    Q8Pack,
    /// Force safetensors (ignore packs).
    Safetensors,
}

/// Load-time knobs. `Default` equals today's env-unset behavior.
#[derive(Clone, Debug, Default)]
pub struct LoadOptions {
    pub pack: PackChoice,
    /// Quantize at load with the export scheme (old `LAYA_QUANT=1`).
    pub quant_at_load: bool,
    /// Keep raw fitted temperatures instead of the [0.5, 5] clamp - the
    /// official harness mode (old `LAYA_UNCLAMPED=1`).
    pub unclamped: bool,
    /// Sliding RoPE theta override (old `LAYA_THETA_SLIDE`); None keeps the
    /// shipped-checkpoint 10_000.
    pub theta_sliding: Option<f32>,
    /// Parity dump directory (old `LAYA_DUMP_DIR`), resolved once here
    /// instead of on every forward.
    pub dump_dir: Option<String>,
}

impl LoadOptions {
    /// The env adapter. Call ONCE at the binary edge (main / laya-serve);
    /// library code receives the result explicitly.
    pub fn from_env() -> Self {
        let pack = match std::env::var("LAYA_BURNPACK").as_deref() {
            Ok("q8") => PackChoice::Q8Pack,
            Ok("1") | Ok("f32") => PackChoice::F32Pack,
            _ => PackChoice::Auto,
        };
        Self {
            pack,
            quant_at_load: std::env::var("LAYA_QUANT").as_deref() == Ok("1"),
            unclamped: std::env::var("LAYA_UNCLAMPED").as_deref() == Ok("1"),
            theta_sliding: std::env::var("LAYA_THETA_SLIDE")
                .ok()
                .and_then(|v| v.parse::<f32>().ok()),
            dump_dir: std::env::var("LAYA_DUMP_DIR").ok().filter(|s| !s.is_empty()),
        }
    }
}
