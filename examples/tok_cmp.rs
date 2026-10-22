fn main() {
    let tok = laya_burn::tokenizer::LayaTokenizer::load("model-multilingual/tokenizer/tokenizer.json");
    // stdin: one text per line; stdout: comma-separated ids per line
    let mut buf = String::new();
    use std::io::Read;
    std::io::stdin().read_to_string(&mut buf).unwrap();
    for line in buf.lines() {
        let ids = tok.encode(line);
        println!("{}", ids.iter().map(|i| i.to_string()).collect::<Vec<_>>().join(","));
    }
}
