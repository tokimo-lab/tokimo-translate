# tokimo-translate

High-performance **Chinese ↔ English** translation powered by Tencent's [HY-MT1.5](https://huggingface.co/tencent) GGUF models via [llama.cpp](https://github.com/ggerganov/llama.cpp).

- 🏆 **WMT25 SOTA** — HY-MT1.5 ranks #1 on multiple zh↔en translation benchmarks
- ⚡ **Fast** — ~200 t/s on RTX 4080 with 1.8B model, 54 ms average latency
- 🖥️ **Multi-platform** — NVIDIA CUDA / Apple Metal / Vulkan / AMD ROCm / CPU
- 🔍 **Runtime auto-detect** — compiled binary selects best available GPU automatically

---

## Models

| Model | Size | Format | VRAM |
|-------|------|--------|------|
| HY-MT1.5-1.8B-GGUF | Q4_K_M ~1.1 GB | GGUF | 1.7 GB |
| HY-MT1.5-7B-GGUF | Q4_K_M ~4.6 GB | GGUF | 4.9 GB |

Download:
```bash
huggingface-cli download tencent/HY-MT1.5-1.8B-GGUF HY-MT1.5-1.8B-Q4_K_M.gguf
huggingface-cli download tencent/HY-MT1.5-7B-GGUF   HY-MT1.5-7B-Q4_K_M.gguf
```

---

## Benchmark Results

> Hardware: NVIDIA GeForce RTX 4080 (16 GB) · Driver 570 · CUDA 13.0

### HY-MT1.5-1.8B GGUF Q4_K_M — Load: 928 ms · VRAM: 1,696 MB

#### 中文 → English (zh→en)

| Input | 输入字符 | 输出字符 | 时间 | Tokens | 速度 |
|-------|----------|----------|------|--------|------|
| 100字 | 98 | 422 | **561 ms** | 62 | 110 t/s |
| 200字 | 201 | 863 | **648 ms** | 138 | 213 t/s |
| 500字 | 432 | 1,777 | **1,296 ms** | 269 | 208 t/s |

<details>
<summary>100字 翻译示例</summary>

**输入：**
> 人工智能技术正在深刻改变各行各业的工作方式。机器学习算法通过分析海量数据，能够自动识别复杂模式并做出精准预测。大型语言模型的出现，使自然语言处理达到了前所未有的水平，为人机交互带来了全新的可能性。

**输出：**
> Artificial intelligence technology is transforming the way work is done in various industries. Machine learning algorithms can analyze massive amounts of data, thereby enabling them to automatically identify complex patterns and make accurate predictions. The emergence of large language models has elevated natural language processing to an unprecedented level, opening up new possibilities for human-machine interaction.
</details>

<details>
<summary>200字 翻译示例</summary>

**输入：**
> 近年来，深度学习技术取得了突破性进展，尤其是在计算机视觉和自然语言处理领域。卷积神经网络能够自动提取图像特征，在图像识别、目标检测和图像分割等任务上表现出色。与此同时，基于Transformer架构的大型语言模型彻底改变了文本理解和生成领域，GPT、BERT等模型在机器翻译、文本摘要、问答系统等众多任务上达到了接近甚至超越人类的水平。这些技术的快速发展，正在推动人工智能从专用系统向通用智能方向迈进。

**输出：**
> In recent years, deep learning technology has made significant progress, particularly in the fields of computer vision and natural language processing. Convolutional neural networks are capable of automatically extracting features from images, and they perform exceptionally well in tasks such as image recognition, object detection, and image segmentation. At the same time, large language models based on Transformer architectures have revolutionized the field of text understanding and generation. Models like GPT and BERT have achieved levels of performance that are close to or even surpass those of humans in various tasks such as machine translation, text summarization, and question answering. The rapid development of these technologies is driving artificial intelligence towards a direction of general intelligence, rather than just specialized systems.
</details>

<details>
<summary>500字 翻译示例</summary>

**输入：**
> 随着计算能力的持续提升和数据规模的不断扩大，人工智能技术正经历前所未有的快速发展阶段。深度学习模型的参数规模从最初的数百万增长到如今的数千亿，这一量变引发了质变——模型不仅在特定任务上表现优异，更展现出令人惊讶的涌现能力，如复杂推理、代码生成和跨领域知识迁移。
>
> 在工业应用方面，人工智能已深度融入制造、医疗、金融、教育等各个领域。智能工厂借助机器视觉和预测性维护技术，显著提高了生产效率并降低了运营成本。医疗影像诊断系统能够在数秒内分析CT和MRI图像，辅助医生发现早期病变。金融机构利用机器学习模型进行实时风险控制和欺诈检测，每年避免数十亿元的损失。
>
> 然而，人工智能的快速发展也带来了诸多挑战。模型的可解释性问题使得在高风险决策场景中的应用受到限制。训练大型模型所需的巨额能耗和碳排放引发了环保争议。算法偏见和数据隐私保护也成为监管机构和社会公众高度关注的议题。

**输出：**
> With the continuous improvement of computing capabilities and the expansion of data sizes, artificial intelligence technology is experiencing an unprecedented period of rapid development. The size of parameters in deep learning models has increased from millions at the beginning to hundreds of billions today. This quantitative change has led to qualitative improvements—models not only perform exceptionally well on specific tasks but also exhibit surprising abilities such as complex reasoning, code generation, and cross-domain knowledge transfer.
>
> In industrial applications, artificial intelligence has been fully integrated into various fields such as manufacturing, healthcare, finance, and education. Intelligent factories use machine vision and predictive maintenance techniques to significantly improve production efficiency and reduce operating costs. Medical imaging systems can analyze CT and MRI images within seconds, helping doctors detect early-stage diseases. Financial institutions use machine learning models for real-time risk management and fraud detection, thereby avoiding billions of dollars in losses each year.
>
> However, the rapid development of artificial intelligence also brings many challenges. The issue of model interpretability limits its application in high-risk decision-making scenarios. The huge energy consumption and carbon emissions required to train large models have sparked debates regarding environmental sustainability. Algorithmic biases and data privacy issues are also topics of significant concern for regulators and the public.
</details>

---

#### English → 中文 (en→zh)

| Input | 输入字符 | 输出字符 | 时间 | Tokens | 速度 |
|-------|----------|----------|------|--------|------|
| 100 words | 309 | 87 | **185 ms** | 37 | 200 t/s |
| 200 words | 578 | 190 | **407 ms** | 85 | 209 t/s |
| 500 words | 1,540 | 447 | **1,080 ms** | 210 | 194 t/s |

<details>
<summary>100 words 翻译示例</summary>

**Input:**
> Artificial intelligence is fundamentally transforming the way industries operate. Machine learning algorithms analyze vast datasets to automatically identify complex patterns and make accurate predictions. The emergence of large language models has brought natural language processing to unprecedented levels.

**输出：**
> 人工智能正在从根本上改变各个行业的运作方式。机器学习算法能够分析庞大的数据集，从而自动识别复杂的模式并做出准确的预测。大型语言模型的出现使得自然语言处理达到了前所未有的水平。
</details>

<details>
<summary>500 words 翻译示例</summary>

**Input:**
> As computational power continues to grow and data scales expand, artificial intelligence is experiencing an unprecedented period of rapid development. The parameter count of deep learning models has grown from millions to hundreds of billions, and this quantitative change has triggered qualitative leaps—models now exhibit surprising emergent capabilities such as complex reasoning, code generation, and cross-domain knowledge transfer.

**输出：**
> 随着计算能力的不断提升以及数据规模的不断扩大，人工智能正经历着前所未有的快速发展阶段。深度学习模型的参数数量从数百万个增长到了数千亿个，这一量的飞跃也引发了质的提升——现在的模型展现出令人惊讶的能力，比如复杂的推理能力、代码生成能力以及跨领域知识的应用能力。
</details>

---

### HY-MT1.5-7B GGUF Q4_K_M — Load: 3,321 ms · VRAM: 4,904 MB

#### 中文 → English (zh→en)

| Input | 输入字符 | 输出字符 | 时间 | Tokens | 速度 |
|-------|----------|----------|------|--------|------|
| 100字 | 98 | 445 | **961 ms** | 66 | 69 t/s |
| 200字 | 201 | 865 | **1,692 ms** | 140 | 83 t/s |
| 500字 | 432 | 1,879 | **3,475 ms** | 283 | 81 t/s |

#### English → 中文 (en→zh)

| Input | 输入字符 | 输出字符 | 时间 | Tokens | 速度 |
|-------|----------|----------|------|--------|------|
| 100 words | 309 | 87 | **486 ms** | 41 | 84 t/s |
| 200 words | 578 | 189 | **1,085 ms** | 92 | 85 t/s |
| 500 words | 1,540 | 456 | **2,706 ms** | 223 | 82 t/s |

---

### Model Comparison

| | 1.8B | 7B |
|---|---|---|
| VRAM | **1.7 GB** | 4.9 GB |
| Load time | **928 ms** | 3,321 ms |
| zh→en 100字 | **561 ms** | 961 ms |
| zh→en 500字 | **1,296 ms** | 3,475 ms |
| en→zh 100词 | **185 ms** | 486 ms |
| en→zh 500词 | **1,080 ms** | 2,706 ms |
| Speed | **~200 t/s** | ~82 t/s |
| Quality | Good | **Better** |

**推荐：** 速度优先选 1.8B，质量优先选 7B。

---

## Platform Support

| Platform | Backend | Build Flag | Status |
|----------|---------|------------|--------|
| NVIDIA GPU (Linux/Windows) | CUDA | `--features cuda` | ✅ |
| Apple Silicon (M1/M2/M3) | Metal | `--features metal` | ✅ |
| AMD/Intel GPU | Vulkan | `--features vulkan` | ✅ |
| AMD GPU (Linux) | ROCm | `--features rocm` | ✅ |
| Any (no GPU) | CPU | *(default)* | ✅ |

The compiled binary **auto-detects** the best available GPU at runtime — no configuration needed.  
For a CUDA+Vulkan binary, `make release-linux` compiles both backends; the binary uses whichever GPU it finds.

---

## Quick Start

### Prerequisites

- Rust 1.80+
- CUDA Toolkit 12+ (for NVIDIA GPU), or nothing (for CPU)
- Models downloaded (see above)

### Build

```bash
git clone https://github.com/tokimo-lab/tokimo-translate
cd tokimo-translate

make              # auto-detect GPU backend, build debug
make release      # auto-detect, build release (optimized)
make release-linux   # Linux: CUDA + Vulkan (runtime auto-select)
make release-mac     # macOS Apple Silicon: Metal
```

Or manually:

```bash
cargo build --release --features cuda     # NVIDIA
cargo build --release --features metal    # Apple Silicon
cargo build --release --features vulkan   # Vulkan
cargo build --release                     # CPU fallback
```

### Run

```bash
./target/release/tokimo-translate
```

Output:
```
  编译后端: NVIDIA CUDA
  运行时GPU: NVIDIA GeForce RTX 4080

=================================================================
  HY-MT1.5-1.8B  GGUF Q4_K_M  (~1.1GB)
=================================================================
  加载: 928ms

  [zh → en]
  ZH: 人工智能正在改变世界。
  EN: Artificial intelligence is changing the world.  (40ms, 171.6t/s)
  ...
```

---

## Configuration

Model paths are auto-detected from the HuggingFace cache (`~/.cache/huggingface`).  
Override with environment variables:

```bash
export HY_MT_1B_PATH=/path/to/HY-MT1.5-1.8B-Q4_K_M.gguf
export HY_MT_7B_PATH=/path/to/HY-MT1.5-7B-Q4_K_M.gguf
export HF_HOME=/custom/hf/cache   # alternative: custom HF cache dir
```

---

## Testing

```bash
# Unit tests (no model required, fast)
cargo test

# Integration tests with real models (requires downloaded GGUF files)
cargo test -- --ignored

# Single test
cargo test test_1b_zh2en_100 -- --ignored
```

---

## License

MIT
