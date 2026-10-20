fn main() {
    let t = tokenizers::Tokenizer::from_file("model/tokenizer/tokenizer.json").unwrap();
    for s in ["hello world", "The user is deciding"] {
        let e = t.encode(s, false).unwrap();
        let ids: Vec<u32> = e.get_ids().to_vec();
        println!("{s:?} -> {} tokens ids={ids:?}", ids.len());
    }
}
