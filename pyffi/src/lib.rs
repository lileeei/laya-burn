//! Python bindings for laya-burn.
use burn::tensor::Device;
use laya_burn::inference::{Answer, QuestionInput, RlAgent};
use laya_burn::options::LoadOptions;
use pyo3::exceptions::{PyRuntimeError, PyTypeError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use serde_json::Value;

fn err(e: impl Into<String>) -> PyErr {
    PyRuntimeError::new_err(e.into())
}

fn py_to_json(obj: &Bound<'_, PyAny>) -> PyResult<Value> {
    if obj.is_none() { Ok(Value::Null) }
    else if let Ok(s) = obj.extract::<String>() { Ok(Value::String(s)) }
    else if let Ok(b) = obj.extract::<bool>() { Ok(Value::Bool(b)) }
    else if let Ok(i) = obj.extract::<i64>() { Ok(serde_json::json!(i)) }
    else if let Ok(f) = obj.extract::<f64>() { Ok(serde_json::json!(f)) }
    else if let Ok(d) = obj.downcast::<PyDict>() {
        let mut m = serde_json::Map::new();
        for (k, v) in d.iter() { m.insert(k.extract::<String>()?, py_to_json(&v)?); }
        Ok(Value::Object(m))
    } else if let Ok(l) = obj.downcast::<pyo3::types::PyList>() {
        let mut items = Vec::new();
        for item in l.iter() { items.push(py_to_json(&item)?); }
        Ok(Value::Array(items))
    } else if let Ok(l) = obj.downcast::<pyo3::types::PyTuple>() {
        let mut items = Vec::new();
        for item in l.iter() { items.push(py_to_json(&item)?); }
        Ok(Value::Array(items))
    } else {
        Err(PyTypeError::new_err(format!("cannot convert {}", obj.get_type())))
    }
}

fn answers_to_pydict(
    py: Python,
    ids: &[String],
    inputs: &[QuestionInput],
    answers: &[Answer],
) -> PyResult<PyObject> {
    let dict = PyDict::new_bound(py);
    for (i, answer) in answers.iter().enumerate() {
        let json_val = laya_burn::jev::answer_to_json(&inputs[i], answer);
        // Convert the serde_json Value to a Python object
        let obj = json_to_pyobj(&json_val, py);
        dict.set_item(&ids[i], obj)?;
    }
    Ok(dict.into())
}

fn json_to_pyobj(v: &Value, py: Python) -> PyObject {
    use pyo3::types::PyList;
    match v {
        Value::Null => py.None(),
        Value::Bool(b) => b.to_object(py),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() { i.to_object(py) }
            else { n.as_f64().unwrap_or(0.0).to_object(py) }
        }
        Value::String(s) => s.to_object(py),
        Value::Array(items) => {
            let list = PyList::new_bound(py, items.iter().map(|i| json_to_pyobj(i, py)));
            list.into()
        }
        Value::Object(map) => {
            let dict = PyDict::new_bound(py);
            for (k, v) in map { let _ = dict.set_item(k, json_to_pyobj(v, py)); }
            dict.into()
        }
    }
}

/// The Laya decision agent.
#[pyclass]
struct Agent {
    agent: RlAgent,
    device: Device,
}

#[pymethods]
impl Agent {
    /// Load a checkpoint from a directory.
    #[new]
    fn new(model_dir: &str) -> PyResult<Self> {
        let device = Device::flex();
        let opts = LoadOptions::default();
        let agent = RlAgent::load_with(model_dir, &opts, &device);
        Ok(Self { agent, device })
    }

    /// Ask typed questions.
    /// agent.predict("state text", {
    ///     "team": {"type": "choice", "instructions": "Which team?",
    ///              "criteria": {"billing": None, "support": None}},
    ///     "urgent": {"type": "noul", "instructions": "Is it urgent?"},
    /// })
    fn predict(&self, py: Python, state: &str, questions: &Bound<'_, PyDict>) -> PyResult<PyObject> {
        let state_val = Value::String(state.to_string());
        let qs_val = py_to_json(questions)?;
        let qs_obj = qs_val.as_object().ok_or_else(|| err("questions must be a dict"))?;
        let mut inputs = Vec::new();
        let mut ids = Vec::new();
        for (qid, qdef) in qs_obj {
            let input = laya_burn::jev::question_from_jev(qdef)
                .ok_or_else(|| err(format!("invalid question {qid}")))?;
            ids.push(qid.clone());
            inputs.push(input);
        }
        let answers = self.agent.system_one(&self.device, &state_val, &inputs, None);
        answers_to_pydict(py, &ids, &inputs, &answers)
    }

    /// Convenience: typed answers (choice -> str, noul -> bool, score -> int).
    fn decide(&self, py: Python, state: &str, questions: &Bound<'_, PyDict>) -> PyResult<PyObject> {
        let state_val = Value::String(state.to_string());
        let qs_val = py_to_json(questions)?;
        let qs_obj = qs_val.as_object().ok_or_else(|| err("questions must be a dict"))?;
        let mut inputs = Vec::new();
        let mut ids = Vec::new();
        for (qid, qdef) in qs_obj {
            let input = laya_burn::jev::question_from_jev(qdef)
                .ok_or_else(|| err(format!("invalid question {qid}")))?;
            ids.push(qid.clone());
            inputs.push(input);
        }
        let answers = self.agent.system_one(&self.device, &state_val, &inputs, None);
        let dict = PyDict::new_bound(py);
        for (i, answer) in answers.iter().enumerate() {
            match answer {
                Answer::Choice { index, .. } => {
                    dict.set_item(&ids[i], inputs[i].keys[*index].clone())?;
                }
                Answer::Noul { prob, .. } => {
                    dict.set_item(&ids[i], *prob >= 0.5)?;
                }
                Answer::Score { score, .. } => {
                    dict.set_item(&ids[i], score.round() as i32)?;
                }
            }
        }
        Ok(dict.into())
    }

    fn __repr__(&self) -> String {
        format!("laya_burn.Agent(max_len={})", self.agent.max_len)
    }
}

/// Fine-tune a checkpoint on a JSONL dataset. Returns a report dict.
#[pyfunction]
#[pyo3(signature = (model_dir, data_path, out_dir, epochs=4, freeze_encoder=false))]
fn finetune(
    py: Python,
    model_dir: &str,
    data_path: &str,
    out_dir: &str,
    epochs: usize,
    freeze_encoder: bool,
) -> PyResult<Py<PyAny>> {
    let cfg = laya_burn::train::TrainConfig {
        epochs,
        freeze_encoder,
        ..Default::default()
    };
    let opts = LoadOptions::default();
    let report = laya_burn::train::finetune(
        model_dir, data_path, out_dir, &cfg, &opts, &Device::flex(),
    ).map_err(err)?;
    let dict = PyDict::new_bound(py);
    let _ = dict.set_item("items", report.items);
    let _ = dict.set_item("first_loss", report.first_loss);
    let _ = dict.set_item("last_loss", report.last_loss);
    let _ = dict.set_item("steps", report.steps);
    Ok(dict.into_any().unbind())
}

#[pymodule]
#[pyo3(name = "laya_burn")]
fn laya_burn_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Agent>()?;
    m.add_function(wrap_pyfunction!(finetune, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
