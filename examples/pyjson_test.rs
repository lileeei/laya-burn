use laya_burn::python_json::PythonJson;
use serde_json::json;

fn main() {
    let v = json!({
        "task": "démo-测试",
        "ratio": 2.5,
        "big": 1e18,
        "tiny": 3.3333333333333e-7,
        "flags": [true, null, -0.0],
        "nested": {"k": [1, 2.0]}
    });
    println!("{}", PythonJson::state(&v));
}
