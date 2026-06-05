//! HY-MT1.5 GGUF translation using llama-cpp-2.
//!
//! Compiled-in GPU backends (set by build.rs based on Cargo features):
//!   --features cuda          NVIDIA CUDA
//!   --features cuda,vulkan   NVIDIA + Vulkan (AMD/Intel fallback)
//!   --features metal         Apple Silicon
//!   --features rocm          AMD ROCm
//!   (no feature)             CPU only
//!
//! At runtime, llama.cpp automatically selects the best available backend.
//! n_gpu_layers = u32::MAX → use all GPU layers if GPU present, else CPU.

use anyhow::{Context, Result};
use llama_cpp_2::{
    context::params::LlamaContextParams,
    llama_backend::LlamaBackend,
    llama_batch::LlamaBatch,
    model::{AddBos, LlamaChatMessage, LlamaModel, params::LlamaModelParams},
    sampling::LlamaSampler,
};
use std::{num::NonZeroU32, path::Path, time::Instant};

/// All GPU layers offloaded when any GPU backend is compiled in.
/// llama.cpp falls back to CPU automatically if no GPU is available at runtime.
fn gpu_layers() -> u32 {
    #[cfg(any(
        gguf_backend_cuda,
        gguf_backend_metal,
        gguf_backend_vulkan,
        gguf_backend_rocm
    ))]
    return u32::MAX;
    #[cfg(gguf_backend_cpu)]
    return 0;
}

/// Compiled-in backends (one or more may be active).
fn compiled_backends() -> &'static str {
    #[cfg(all(gguf_backend_cuda, gguf_backend_vulkan))]
    return "CUDA + Vulkan (runtime auto-select)";
    #[cfg(all(gguf_backend_cuda, not(gguf_backend_vulkan)))]
    return "NVIDIA CUDA";
    #[cfg(gguf_backend_metal)]
    return "Apple Metal";
    #[cfg(all(gguf_backend_vulkan, not(gguf_backend_cuda)))]
    return "Vulkan";
    #[cfg(gguf_backend_rocm)]
    return "AMD ROCm";
    #[allow(unreachable_code)]
    "CPU only"
}

/// Detect the actual GPU in use at runtime (best-effort).
fn detect_runtime_gpu() -> String {
    // Try NVIDIA first
    if let Ok(out) = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=name", "--format=csv,noheader"])
        .output()
        && out.status.success()
    {
        let name = String::from_utf8_lossy(&out.stdout).trim().to_string();
        if !name.is_empty() {
            return format!("NVIDIA {name}");
        }
    }
    // macOS — system_profiler for GPU name
    #[cfg(target_os = "macos")]
    if let Ok(out) = std::process::Command::new("system_profiler")
        .args(["SPDisplaysDataType", "-json"])
        .output()
    {
        if out.status.success() {
            let s = String::from_utf8_lossy(&out.stdout);
            // Quick parse: find "sppci_model"
            if let Some(pos) = s.find("sppci_model") {
                let rest = &s[pos + 14..];
                if let Some(end) = rest.find('"') {
                    return rest[..end].trim().to_string();
                }
            }
        }
    }
    // Fallback to compiled-in backend name
    compiled_backends().to_string()
}

const GGUF_1B: &str = "/home/william/.cache/huggingface/hub/models--tencent--HY-MT1.5-1.8B-GGUF/snapshots/265b2e615a7dc9b06c435dc878829ad99a512ba2/HY-MT1.5-1.8B-Q4_K_M.gguf";
const GGUF_7B: &str = "/home/william/.cache/huggingface/hub/models--tencent--HY-MT1.5-7B-GGUF/snapshots/126325496bc8e3575f1d8615b8ca951d8483f206/HY-MT1.5-7B-Q4_K_M.gguf";

/// Find a GGUF model file in the HuggingFace cache or via environment variable.
///
/// Resolution order:
///   1. `$HF_HOME/hub/models--{repo}/snapshots/*/{filename}`
///   2. `$HUGGINGFACE_HUB_CACHE/hub/...` (same layout)
///   3. `~/.cache/huggingface/hub/...` (default)
///   4. `env_override` env var (e.g. `HY_MT_1B_PATH=/path/to/file.gguf`)
pub fn find_gguf_model(
    repo: &str,
    filename: &str,
    env_override: &str,
) -> Result<std::path::PathBuf> {
    // 1. Explicit env var override
    if let Ok(path) = std::env::var(env_override) {
        let p = std::path::PathBuf::from(&path);
        if p.exists() {
            return Ok(p);
        }
        anyhow::bail!("${env_override}={path} — file not found");
    }

    // 2. Auto-detect from HuggingFace cache
    let hf_home = std::env::var("HF_HOME")
        .or_else(|_| std::env::var("HUGGINGFACE_HUB_CACHE"))
        .unwrap_or_else(|_| {
            let home = std::env::var("HOME").unwrap_or_default();
            format!("{home}/.cache/huggingface")
        });

    let folder = format!("models--{}", repo.replace('/', "--"));
    let snapshots = std::path::PathBuf::from(&hf_home)
        .join("hub")
        .join(folder)
        .join("snapshots");

    if snapshots.exists() {
        for entry in std::fs::read_dir(&snapshots)? {
            let candidate = entry?.path().join(filename);
            if candidate.exists() {
                return Ok(candidate);
            }
        }
    }

    anyhow::bail!(
        "Model not found: {repo}/{filename}\n\
         Download with:\n  \
         huggingface-cli download {repo} {filename}\n\
         Or set env var: {env_override}=/path/to/{filename}"
    )
}

fn zh2en_content(text: &str) -> String {
    format!("将以下文本翻译为英语，注意只需要输出翻译后的结果，不要额外解释：\n\n{text}")
}

fn en2zh_content(text: &str) -> String {
    format!("将以下文本翻译为中文，注意只需要输出翻译后的结果，不要额外解释：\n\n{text}")
}

struct TranslateResult {
    translation: String,
    elapsed_ms: u64,
    tokens_per_sec: f32,
}

fn translate(
    model: &LlamaModel,
    backend: &LlamaBackend,
    content: &str,
    max_tokens: i32,
) -> Result<TranslateResult> {
    // Use the model's built-in chat template
    let tmpl = model.chat_template(None).context("get chat template")?;
    let messages = [LlamaChatMessage::new("user".into(), content.into())?];
    let prompt = model.apply_chat_template(&tmpl, &messages, true)?;

    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(NonZeroU32::new(2048))
        .with_n_threads(4)
        .with_n_threads_batch(4);

    let mut ctx = model
        .new_context(backend, ctx_params)
        .context("create context")?;

    let tokens = model
        .str_to_token(&prompt, AddBos::Never)
        .context("tokenize")?;

    let mut batch = LlamaBatch::new(tokens.len() + max_tokens as usize, 1);
    let last = (tokens.len() - 1) as i32;
    for (i, &tok) in tokens.iter().enumerate() {
        batch.add(tok, i as i32, &[0], i as i32 == last)?;
    }
    ctx.decode(&mut batch).context("decode prompt")?;

    let mut sampler = LlamaSampler::chain_simple([
        LlamaSampler::temp(0.7),
        LlamaSampler::top_k(20),
        LlamaSampler::top_p(0.6, 1),
        LlamaSampler::penalties(64, 1.05, 0.0, 0.0),
        LlamaSampler::dist(42),
    ]);

    let mut decoder = encoding_rs::UTF_8.new_decoder();
    let mut output = String::new();
    let mut n_cur = batch.n_tokens();
    let mut n_decode = 0u32;

    let t0 = Instant::now();

    while n_cur < tokens.len() as i32 + max_tokens {
        let token = sampler.sample(&ctx, batch.n_tokens() - 1);
        sampler.accept(token);

        if model.is_eog_token(token) {
            break;
        }

        let piece = model.token_to_piece(token, &mut decoder, true, None)?;
        output.push_str(&piece);

        batch.clear();
        batch.add(token, n_cur, &[0], true)?;
        n_cur += 1;
        ctx.decode(&mut batch).context("decode token")?;
        n_decode += 1;
    }

    let elapsed = t0.elapsed();

    Ok(TranslateResult {
        translation: output.trim().to_string(),
        elapsed_ms: elapsed.as_millis() as u64,
        tokens_per_sec: n_decode as f32 / elapsed.as_secs_f32(),
    })
}

fn run_model(backend: &LlamaBackend, path: &str, label: &str) -> Result<()> {
    println!("\n{}", "=".repeat(65));
    println!("  {label}");
    println!("{}", "=".repeat(65));

    let model_params = LlamaModelParams::default().with_n_gpu_layers(gpu_layers());
    let t_load = Instant::now();
    let model = LlamaModel::load_from_file(backend, Path::new(path), &model_params)
        .with_context(|| format!("load {path}"))?;
    println!("  加载: {}ms\n", t_load.elapsed().as_millis());

    let zh_sentences = [
        "人工智能正在改变世界。",
        "机器学习模型需要大量的训练数据才能达到良好的性能。",
        "我们的团队在过去三年中开发了多项创新技术。",
    ];
    let en_sentences = [
        "Artificial intelligence is transforming the world.",
        "Machine learning models require large amounts of training data to achieve good performance.",
        "Our team has developed multiple innovative technologies over the past three years.",
    ];

    // warm up
    let _ = translate(&model, backend, &zh2en_content(zh_sentences[0]), 64)?;

    println!("  [zh → en]");
    let mut total_ms = 0u64;
    let mut count = 0u64;
    for src in &zh_sentences {
        let r = translate(&model, backend, &zh2en_content(src), 128)?;
        println!("  ZH: {src}");
        println!(
            "  EN: {}  ({}ms, {:.1}t/s)\n",
            r.translation, r.elapsed_ms, r.tokens_per_sec
        );
        total_ms += r.elapsed_ms;
        count += 1;
    }

    println!("  [en → zh]");
    for src in &en_sentences {
        let r = translate(&model, backend, &en2zh_content(src), 128)?;
        println!("  EN: {src}");
        println!(
            "  ZH: {}  ({}ms, {:.1}t/s)\n",
            r.translation, r.elapsed_ms, r.tokens_per_sec
        );
        total_ms += r.elapsed_ms;
        count += 1;
    }

    println!("  avg latency: {}ms", total_ms / count);
    Ok(())
}

fn main() -> Result<()> {
    let backend = LlamaBackend::init()?;
    llama_cpp_2::send_logs_to_tracing(llama_cpp_2::LogOptions::default().with_logs_enabled(false));

    let runtime_gpu = detect_runtime_gpu();
    println!("\n  编译后端: {}", compiled_backends());
    println!("  运行时GPU: {runtime_gpu}");

    let path_1b = find_gguf_model(
        "tencent/HY-MT1.5-1.8B-GGUF",
        "HY-MT1.5-1.8B-Q4_K_M.gguf",
        "HY_MT_1B_PATH",
    )
    .unwrap_or_else(|_| std::path::PathBuf::from(GGUF_1B));
    let path_7b = find_gguf_model(
        "tencent/HY-MT1.5-7B-GGUF",
        "HY-MT1.5-7B-Q4_K_M.gguf",
        "HY_MT_7B_PATH",
    )
    .unwrap_or_else(|_| std::path::PathBuf::from(GGUF_7B));

    run_model(
        &backend,
        path_1b.to_str().unwrap(),
        "HY-MT1.5-1.8B  GGUF Q4_K_M  (~1.1GB)",
    )?;
    run_model(
        &backend,
        path_7b.to_str().unwrap(),
        "HY-MT1.5-7B    GGUF Q4_K_M  (~4.6GB)",
    )?;

    Ok(())
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Real benchmark texts ─────────────────────────────────────────────

    const ZH_100: &str = "人工智能技术正在深刻改变各行各业的工作方式。机器学习算法通过分析海量数据，能够自动识别复杂模式并做出精准预测。大型语言模型的出现，使自然语言处理达到了前所未有的水平，为人机交互带来了全新的可能性。";

    const ZH_200: &str = "近年来，深度学习技术取得了突破性进展，尤其是在计算机视觉和自然语言处理领域。卷积神经网络能够自动提取图像特征，在图像识别、目标检测和图像分割等任务上表现出色。与此同时，基于Transformer架构的大型语言模型彻底改变了文本理解和生成领域，GPT、BERT等模型在机器翻译、文本摘要、问答系统等众多任务上达到了接近甚至超越人类的水平。这些技术的快速发展，正在推动人工智能从专用系统向通用智能方向迈进。";

    const ZH_500: &str = "随着计算能力的持续提升和数据规模的不断扩大，人工智能技术正经历前所未有的快速发展阶段。深度学习模型的参数规模从最初的数百万增长到如今的数千亿，这一量变引发了质变——模型不仅在特定任务上表现优异，更展现出令人惊讶的涌现能力，如复杂推理、代码生成和跨领域知识迁移。\n\n在工业应用方面，人工智能已深度融入制造、医疗、金融、教育等各个领域。智能工厂借助机器视觉和预测性维护技术，显著提高了生产效率并降低了运营成本。医疗影像诊断系统能够在数秒内分析CT和MRI图像，辅助医生发现早期病变。金融机构利用机器学习模型进行实时风险控制和欺诈检测，每年避免数十亿元的损失。\n\n然而，人工智能的快速发展也带来了诸多挑战。模型的可解释性问题使得在高风险决策场景中的应用受到限制。训练大型模型所需的巨额能耗和碳排放引发了环保争议。算法偏见和数据隐私保护也成为监管机构和社会公众高度关注的议题。如何在推动技术创新的同时确保人工智能的安全、公平和可信，是当前学术界和产业界共同面临的重要课题。";

    const EN_100: &str = "Artificial intelligence is fundamentally transforming the way industries operate. Machine learning algorithms analyze vast datasets to automatically identify complex patterns and make accurate predictions. The emergence of large language models has brought natural language processing to unprecedented levels.";

    const EN_200: &str = "In recent years, deep learning has achieved remarkable breakthroughs, particularly in computer vision and natural language processing. Convolutional neural networks excel at automatically extracting image features for recognition and detection tasks. Meanwhile, Transformer-based large language models have revolutionized text understanding and generation. Models like GPT and BERT achieve near-human performance on machine translation, summarization, and question answering. These rapid advances are driving AI from narrow systems toward more general intelligence capabilities.";

    const EN_500: &str = "As computational power continues to grow and data scales expand, artificial intelligence is experiencing an unprecedented period of rapid development. The parameter count of deep learning models has grown from millions to hundreds of billions, and this quantitative change has triggered qualitative leaps—models now exhibit surprising emergent capabilities such as complex reasoning, code generation, and cross-domain knowledge transfer.\n\nIn industrial applications, AI has become deeply integrated into manufacturing, healthcare, finance, and education. Smart factories leverage machine vision and predictive maintenance to significantly improve production efficiency and reduce operational costs. Medical imaging systems analyze CT and MRI scans within seconds, assisting physicians in detecting early-stage abnormalities. Financial institutions use machine learning models for real-time risk control and fraud detection, preventing billions in annual losses.\n\nHowever, the rapid advancement of AI also presents significant challenges. The lack of model interpretability limits deployment in high-stakes decision-making scenarios. The enormous energy consumption and carbon emissions required to train large models have sparked environmental controversy. Algorithmic bias and data privacy protection have also become issues of intense scrutiny from regulators and the public alike. Balancing technological innovation with the safety, fairness, and trustworthiness of AI systems remains a critical challenge for both academia and industry.";

    // ── Unit tests (no model required) ───────────────────────────────────

    #[test]
    fn test_zh2en_prompt_contains_target_lang() {
        let p = zh2en_content("你好世界");
        assert!(p.contains("英语"), "prompt must specify target language");
        assert!(p.contains("你好世界"), "prompt must include source text");
        assert!(!p.contains("中文"), "zh→en prompt should not say 中文");
    }

    #[test]
    fn test_en2zh_prompt_contains_target_lang() {
        let p = en2zh_content("Hello world");
        assert!(p.contains("中文"), "prompt must specify target language");
        assert!(p.contains("Hello world"), "prompt must include source text");
    }

    #[test]
    fn test_prompt_no_extra_instructions() {
        // Prompt must tell model to output translation only
        let p = zh2en_content("test");
        assert!(
            p.contains("只需要输出翻译后的结果"),
            "prompt must suppress extra output"
        );
    }

    #[test]
    fn test_find_model_missing_returns_err() {
        // Should fail gracefully for a non-existent repo
        let r = find_gguf_model("nonexistent-org/no-model", "no.gguf", "NO_SUCH_ENV_VAR_XYZ");
        assert!(r.is_err(), "missing model should return Err");
        let msg = r.unwrap_err().to_string();
        assert!(
            msg.contains("huggingface-cli download") || msg.contains("not found"),
            "error should contain download instructions"
        );
    }

    #[test]
    fn test_find_model_env_override() {
        // If env var points to a real file, it should be returned
        let tmp = std::env::temp_dir().join("dummy_model.gguf");
        std::fs::write(&tmp, b"dummy").unwrap();
        unsafe {
            std::env::set_var("TEST_TOKIMO_TRANSLATE_PATH", tmp.to_str().unwrap());
        }
        let r = find_gguf_model("any/repo", "dummy_model.gguf", "TEST_TOKIMO_TRANSLATE_PATH");
        unsafe {
            std::env::remove_var("TEST_TOKIMO_TRANSLATE_PATH");
        }
        std::fs::remove_file(&tmp).ok();
        assert!(r.is_ok());
        assert_eq!(r.unwrap(), tmp);
    }

    #[test]
    fn test_test_texts_have_expected_lengths() {
        assert!(ZH_100.chars().count() >= 80, "ZH_100 should be ~100 chars");
        assert!(ZH_200.chars().count() >= 160, "ZH_200 should be ~200 chars");
        assert!(ZH_500.chars().count() >= 400, "ZH_500 should be ~500 chars");
        assert!(EN_100.len() >= 200, "EN_100 should be ~100 words");
        assert!(EN_200.len() >= 400, "EN_200 should be ~200 words");
        assert!(EN_500.len() >= 1000, "EN_500 should be ~500 words");
    }

    // ── Integration tests (require models, skip in CI) ────────────────
    // Run with: cargo test -- --ignored
    // Or single test: cargo test test_1b_zh2en_100 -- --ignored

    fn load_1b_model(backend: &LlamaBackend) -> Option<LlamaModel> {
        let path = find_gguf_model(
            "tencent/HY-MT1.5-1.8B-GGUF",
            "HY-MT1.5-1.8B-Q4_K_M.gguf",
            "HY_MT_1B_PATH",
        )
        .ok()?;
        let params = LlamaModelParams::default().with_n_gpu_layers(gpu_layers());
        LlamaModel::load_from_file(backend, &path, &params).ok()
    }

    fn load_7b_model(backend: &LlamaBackend) -> Option<LlamaModel> {
        let path = find_gguf_model(
            "tencent/HY-MT1.5-7B-GGUF",
            "HY-MT1.5-7B-Q4_K_M.gguf",
            "HY_MT_7B_PATH",
        )
        .ok()?;
        let params = LlamaModelParams::default().with_n_gpu_layers(gpu_layers());
        LlamaModel::load_from_file(backend, &path, &params).ok()
    }

    macro_rules! translation_test {
        ($name:ident, $load:ident, $src:expr, $content_fn:ident, $expect:expr) => {
            #[test]
            #[ignore]
            fn $name() {
                let backend = LlamaBackend::init().unwrap();
                let model = match $load(&backend) {
                    Some(m) => m,
                    None => {
                        eprintln!("SKIP: model not found");
                        return;
                    }
                };
                let r = translate(&model, &backend, &$content_fn($src), 512).unwrap();
                let out = r.translation.to_lowercase();
                eprintln!("  {}ms  {:.1}t/s", r.elapsed_ms, r.tokens_per_sec);
                eprintln!("  OUTPUT: {}", r.translation);
                // Basic sanity: output must contain at least one expected keyword
                assert!(
                    $expect.iter().any(|kw: &&str| r.translation.contains(kw)),
                    "translation missing expected content: {:?}\ngot: {}",
                    $expect,
                    r.translation
                );
                let _ = out;
            }
        };
    }

    // 1.8B model — zh→en
    translation_test!(
        test_1b_zh2en_100,
        load_1b_model,
        ZH_100,
        zh2en_content,
        &[
            "artificial intelligence",
            "machine learning",
            "language model"
        ]
    );
    translation_test!(
        test_1b_zh2en_200,
        load_1b_model,
        ZH_200,
        zh2en_content,
        &["deep learning", "neural network", "Transformer", "GPT"]
    );
    translation_test!(
        test_1b_zh2en_500,
        load_1b_model,
        ZH_500,
        zh2en_content,
        &[
            "artificial intelligence",
            "parameters",
            "healthcare",
            "finance"
        ]
    );

    // 1.8B model — en→zh
    translation_test!(
        test_1b_en2zh_100,
        load_1b_model,
        EN_100,
        en2zh_content,
        &["人工智能", "机器学习", "语言模型"]
    );
    translation_test!(
        test_1b_en2zh_200,
        load_1b_model,
        EN_200,
        en2zh_content,
        &["深度学习", "神经网络", "Transformer"]
    );
    translation_test!(
        test_1b_en2zh_500,
        load_1b_model,
        EN_500,
        en2zh_content,
        &["人工智能", "参数", "医疗", "金融"]
    );

    // 7B model — zh→en
    translation_test!(
        test_7b_zh2en_100,
        load_7b_model,
        ZH_100,
        zh2en_content,
        &[
            "artificial intelligence",
            "machine learning",
            "language model"
        ]
    );
    translation_test!(
        test_7b_zh2en_200,
        load_7b_model,
        ZH_200,
        zh2en_content,
        &["deep learning", "neural network", "Transformer"]
    );
    translation_test!(
        test_7b_zh2en_500,
        load_7b_model,
        ZH_500,
        zh2en_content,
        &["artificial intelligence", "parameters", "healthcare"]
    );

    // 7B model — en→zh
    translation_test!(
        test_7b_en2zh_100,
        load_7b_model,
        EN_100,
        en2zh_content,
        &["人工智能", "机器学习"]
    );
    translation_test!(
        test_7b_en2zh_200,
        load_7b_model,
        EN_200,
        en2zh_content,
        &["深度学习", "神经网络"]
    );
    translation_test!(
        test_7b_en2zh_500,
        load_7b_model,
        EN_500,
        en2zh_content,
        &["人工智能", "参数", "医疗"]
    );
}
