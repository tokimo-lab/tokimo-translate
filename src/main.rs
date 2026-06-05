/// Minimal zh→en translation demo.
/// Mirrors tokimo.io/packages/rust-models: ort sessions + named tensor inputs.
///
/// Model     : onnx-community/opus-mt-zh-en (MarianMT, int8 ONNX)
/// Tokenizer : HF `tokenizers` crate (tokenizer.json, pure Rust)
/// Pipeline  : tokenize → encoder → greedy decoder loop → decode
use anyhow::{Context, Result};
use ort::{session::Session, value::Tensor};
use std::path::Path;
use tokenizers::Tokenizer;

const HF_BASE: &str = "https://huggingface.co/onnx-community/opus-mt-zh-en/resolve/main";

struct ModelFile {
    url_path: &'static str,
    local: &'static str,
}
const FILES: &[ModelFile] = &[
    ModelFile {
        url_path: "onnx/encoder_model_int8.onnx",
        local: "models/encoder.onnx",
    },
    ModelFile {
        url_path: "onnx/decoder_model_int8.onnx",
        local: "models/decoder.onnx",
    },
    ModelFile {
        url_path: "tokenizer.json",
        local: "models/tokenizer.json",
    },
];

const EOS_ID: u32 = 0;
const DEC_START: i64 = 65000; // MarianMT decoder_start_token_id (not 0/EOS)
const MAX_NEW: usize = 128;

// ── Download ─────────────────────────────────────────────────────────────────

fn download_models() -> Result<()> {
    std::fs::create_dir_all("models")?;
    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(300))
        .build()?;
    for f in FILES {
        let dest = Path::new(f.local);
        if dest.exists() {
            eprintln!("  ✓ {} (cached)", f.local);
            continue;
        }
        let url = format!("{HF_BASE}/{}", f.url_path);
        eprintln!("  ↓ {url}");
        let bytes = client.get(&url).send()?.bytes()?;
        std::fs::write(dest, &bytes)?;
        eprintln!("    {} KB", bytes.len() / 1024);
    }
    Ok(())
}

// ── Session loader — same pattern as tokimo.io build_session ────────────────

fn load_session(path: &str) -> Result<Session> {
    Session::builder()
        .context("ort Session::builder")?
        .commit_from_file(path)
        .with_context(|| format!("load ONNX: {path}"))
}

// ── Encoder: input_ids → last_hidden_state ───────────────────────────────────

fn run_encoder(session: &mut Session, input_ids: &[i64]) -> Result<(Vec<f32>, Vec<i64>)> {
    let seq = input_ids.len() as i64;
    let ids_t = Tensor::from_array(([1i64, seq], input_ids.to_vec())).context("encoder ids")?;
    let mask_t =
        Tensor::from_array(([1i64, seq], vec![1i64; input_ids.len()])).context("encoder mask")?;

    let outputs = session
        .run(ort::inputs![ids_t, mask_t])
        .context("encoder run")?;

    // tokimo.io pattern: let (_shape, data) = outputs[N].try_extract_tensor()?
    let (shape, data) = outputs[0]
        .try_extract_tensor::<f32>()
        .context("encoder extract")?;
    Ok((data.to_vec(), shape.to_vec()))
}

// ── Greedy decoder: feeds all tokens each step (stateless, no KV cache) ──────
// decoder_model_int8.onnx inputs: encoder_attention_mask, input_ids, encoder_hidden_states

fn greedy_decode(
    session: &mut Session,
    enc_hidden: &[f32],
    enc_shape: &[i64], // [1, src_len, 512]
    src_len: usize,
) -> Result<Vec<u32>> {
    let hidden_size = enc_shape[2];
    let mut tokens: Vec<i64> = vec![DEC_START]; // MarianMT decoder start token

    for _ in 0..MAX_NEW {
        let dec_len = tokens.len() as i64;

        let mask_t =
            Tensor::from_array(([1i64, src_len as i64], vec![1i64; src_len])).context("mask")?;
        let ids_t = Tensor::from_array(([1i64, dec_len], tokens.clone())).context("ids")?;
        let hid_t = Tensor::from_array(([1i64, src_len as i64, hidden_size], enc_hidden.to_vec()))
            .context("hidden")?;

        let outputs = session
            .run(ort::inputs![
                "encoder_attention_mask" => mask_t,
                "input_ids"              => ids_t,
                "encoder_hidden_states"  => hid_t
            ])
            .context("decoder run")?;

        // logits: [1, dec_len, vocab_size] → argmax at last position
        let (shape, data) = outputs["logits"]
            .try_extract_tensor::<f32>()
            .context("logits")?;
        let vocab = shape[2] as usize;
        let offset = (tokens.len() - 1) * vocab;
        let last = &data[offset..offset + vocab];

        let next_id = last
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i as i64)
            .unwrap_or(0);

        if next_id as u32 == EOS_ID {
            break;
        }
        tokens.push(next_id);
    }
    Ok(tokens[1..].iter().map(|&x| x as u32).collect())
}

// ── Main ─────────────────────────────────────────────────────────────────────

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let sentences = [
        "人工智能正在改变世界。",
        "今天天气很好，我想出去散步。",
        "这个项目的目标是构建一个高效的翻译系统。",
        "中华文明拥有五千年的历史。",
    ];

    println!("=== zh→en  |  ort + opus-mt-zh-en int8 ONNX ===\n");

    eprintln!("[1/4] Checking model files …");
    download_models()?;

    eprintln!("[2/4] Loading tokenizer …");
    let tokenizer =
        Tokenizer::from_file("models/tokenizer.json").map_err(|e| anyhow::anyhow!("{e}"))?;

    eprintln!("[3/4] Loading ONNX sessions …");
    let mut enc_session = load_session("models/encoder.onnx")?;
    let mut dec_session = load_session("models/decoder.onnx")?;

    eprintln!("[4/4] Translating …\n");
    for src in &sentences {
        let t0 = std::time::Instant::now();

        let enc = tokenizer
            .encode(*src, false)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        let input_ids: Vec<i64> = enc.get_ids().iter().map(|&x| x as i64).collect();

        let (enc_hidden, enc_shape) = run_encoder(&mut enc_session, &input_ids)?;
        let out_ids = greedy_decode(&mut dec_session, &enc_hidden, &enc_shape, input_ids.len())?;
        let result = tokenizer
            .decode(&out_ids, true)
            .map_err(|e| anyhow::anyhow!("{e}"))?;

        println!("ZH : {src}");
        println!("EN : {result}  ({:.0}ms)\n", t0.elapsed().as_millis());
    }
    Ok(())
}
