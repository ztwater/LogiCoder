# LogiCoder

> **Replication Package for LLM See, LLM Do: Enhancing Repository-Aware Code Generation in LLMs Through Logical Context and Usage Knowledge**

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)

**LogiCoder** is a novel Retrieval-Augmented Generation (RAG) approach that leverages **logical relationships** within code repositories to retrieve **usage knowledge** for LLMs during repository-aware code generation. Instead of simply retrieving similar code snippets, LogiCoder identifies candidate callee functions, traces their callers as usage examples, and integrates them with semantic search results via re-ranking — enabling LLMs to imitate established coding patterns rather than learning dependencies from scratch.

<img src="./figures/framework.pdf" alt="LogiCoder Framework" width="800"/>

## Key Results

On the challenging **DevEval-cf** benchmark (471 cross-file code generation tasks across 67 real-world Python repositories), LogiCoder achieves:

| Metric | Best Result | Relative Improvement |
|--------|------------|---------------------|
| **Pass@1** | **44.58%** | up to **27.26%** ↑ |
| **Recall@1** | **46.39%** | up to **14.71%** ↑ |

Evaluated across five LLMs: Kimi-K2, DeepSeek-V3, GPT-4o, Qwen2.5-Coder-14B, and Llama-3.1-8B.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Quick Start](#quick-start)
- [Step-by-Step Reproduction](#step-by-step-reproduction)
  - [Dataset Preparation](#dataset-preparation)
  - [Build Knowledge Base](#build-knowledge-base)
  - [Run LogiCoder Pipeline](#run-logicoder-pipeline)
  - [Run Baselines](#run-baselines)
  - [LLM Inference](#llm-inference)
- [Configuration](#configuration)
- [License](#license)
- [Contact](#contact)

---

## Overview

LogiCoder employs a **four-module pipeline** built on top of a repository dependency graph. The pipeline consists of **four stages**:

1. **Dependency Graph Construction (DGC)**: Uses `pyan` to parse repositories and extract code elements with four relation types (`def`, `use`, `import`, `belong_to`). Virtual edges are also resolved to handle imprecise namespace matches.
2. **Logical Context Extraction (LCE)**: Identifies all candidate callee functions accessible to the target function from intra-file, associated (via call-graph `use` and `virtual` edges), and imported sources.
3. **Usage Example Retrieval (UER)**: Traces callers of each candidate callee and ranks them by context similarity (Jaccard Index on token-level adjacent context).
4. **Semantic Search Fusion (SSF)**: Integrates logical-context-based retrieval with semantic similarity search via Reciprocal Rank Fusion (RRF). Both function-body and context similarities are computed using BoW (Jaccard) over tokenized code, with RRF merging of the two ranked lists.
The fused results form the prompt for augmented generation with LLMs.

---

## Project Structure

```
LogiCoder/
├── logicoder.py                     # Main pipeline: UsageExampleRetriever + prompt building
├── build_call_graph.py              # Dependency graph construction (pyan-based)
├── extract_function.py              # LCE: candidate callee extraction
├── semantic_search_fusion.py        # SSF: BoW retrieval + RRF re-ranking
├── build_prompt.py                  # Prompt builders (RAG, Direct, CoT, RepoCoder, Draco, etc.)
├── templates.py                     # Prompt templates (RAG, CoT, Direct, etc.)
├── inference.py                     # LLM inference client (OpenAI-compatible API)
├── evaluation.py                    # Evaluators for RQ2
│
├── build_vector.py                  # Vector embedding construction (UniXcoder, BoW)
├── build_window.py                  # Sliding window chunking for baselines
├── retriever.py                     # Sparse/dense retrieval for function/baseline pipelines
├── unixcoder.py                     # UniXcoder embedding model wrapper
│
├── repository_cache.py              # Repository data caching layer
├── myutils.py                       # Utility functions (I/O, analysis, tokenization)
├── globals.py                       # Global path configurations (MODIFY THIS FIRST)
│
├── cot_pipeline.py                  # Chain-of-Thought baseline pipeline
├── no_rag_pipeline.py               # Non-RAG baselines (Direct, In-File)
├── repocoder_pipeline.py            # RepoCoder baseline pipeline
├── function_retrieval_pipeline.py   # Function-level sparse/dense retrieval
│
├── extract_completion.py            # Completion extraction from LLM outputs
├── extract_context.py               # Context extraction utilities
│
├── mini-swe-agent_interface.py      # Mini-SWE-agent baseline adapter
└── opencode_interface.py            # OpenCode baseline adapter
```

---

## Requirements

### Hardware
- **GPU**: Required only for local model inference or UniXcoder embeddings (24GB+ VRAM recommended). The retrieval pipeline runs on CPU.
- **Storage**: ~80GB for repositories, embeddings, and cached results
- **RAM**: 32GB+ recommended

### Software
- **Python**: 3.9 or higher
- **OS**: Ubuntu 22.04 (recommended)

---

## Quick Start

```bash
# 1. Install DevEval/CoderEval Environment

# 2. Clone and copy `LogiCoder` to the root directory of the project, e.g., DevEval
cd LogiCoder

# 3. Install required dependencies
pip install -r requirements.txt

# 4. Configure paths (edit globals.py)
# e.g., Set ROOT_DIR to your DevEval dataset root

# 5. Run the full pipeline
python logicoder.py

# 6. Run inference (example with DeepSeek-V3)
python inference.py \
    --prompt_path prompts/logicoder/logicoder.jsonl \
    --output_dir predictions/ \
    --model_str deepseek-v3-241226 \
    --T 0 --N 1 --mode rag

# 7. Evaluate results (according to DevEval/CoderEval)
```

---

## Step-by-Step Reproduction

### Dataset Preparation

LogiCoder is mainly evaluated on the [DevEval](https://github.com/seketeam/DevEval) benchmark. Follow these steps to prepare the data:

```bash
# Download DevEval from the official repository
# Place it in your chosen ROOT_DIR with this structure:
#
# ROOT_DIR/
# ├── Source_Code/           # Repository source code
# │   ├── Database/
# │   ├── Communications/
# │   └── ...
# ├── data.jsonl             # Full benchmark metadata
# └── our_metadata.jsonl     # DevEval-cf subset (cross-file tasks only)
```

**DevEval-cf subset**: The paper's primary experiments use only tasks requiring at least one cross-file call. This filtered subset contains **471 tasks across 67 repositories**. Prepare `our_metadata.jsonl` by filtering `data.jsonl` for tasks with cross-file dependencies.

For the generalization study (RQ5), also prepare:
- `full_metadata.jsonl`: All DevEval tasks (mixed dependencies)
- `codereval_metadata.jsonl`: CoderEval Python tasks

### Build Knowledge Base

The knowledge base construction runs automatically as part of the pipeline, but can also be done separately:

```bash
# Build call graphs for all repositories
python build_call_graph.py

# Build vectors for semantic search (UniXcoder + BoW)
python build_vector.py

# Build sliding windows for baseline methods
python build_window.py
```

**What gets built** (cached in `ROOT_DIR/knowledge_base/`):
- `visitor/`: pyan call graph visitors (`.pkl`)
- `unit/`: Extracted function metadata (`.json`)
- `call_graph/`: Bi-directional call graphs (`_call_graph.json`, `_called_graph.json`)
- `mapping/`: Usage mappings for candidates
- `vector/`: Dense (UniXcoder) and sparse (BoW) embeddings

### Run LogiCoder Pipeline

The main pipeline is executed by `logicoder.py`:

```bash
python logicoder.py
```

This performs **three phases**:

**Phase 1 — Dependency Graph Construction**: Parses each repository with `pyan`, building call graph visitors and extracting function-level knowledge units.

**Phase 2 — Usage Example Retrieval**: For each task:
1. Extracts logical context (candidate callees) via LCE
2. Builds usage mappings (callee → callers)
3. Retrieves and ranks usage examples by context similarity
4. Output: `prompts/logicoder/selected_examples.json`

**Phase 3 — Semantic Search Fusion**: 
1. Runs semantic similarity search over all repository functions
2. Fuses results with usage examples via RRF re-ranking
3. Output: `prompts/logicoder/reranked_examples.json`

Finally, builds the prompt file at `prompts/logicoder/logicoder.jsonl`.

**Key options**:
```bash
# Use ground truth dependencies (for oracle experiments)
python logicoder.py --with_gt_deps

# Use callees directly instead of usage examples (ablation)
python logicoder.py --only_callees

# Random selection instead of similarity ranking (ablation)
python logicoder.py --random_pick
```

### Run Baselines

All baselines are implemented for fair comparison:

```bash
# Non-RAG baselines
python no_rag_pipeline.py              # Direct, In-File Context

# CoT baseline
python cot_pipeline.py

# Similarity-based RAG baselines
python repocoder_pipeline.py           # Vanilla RAG, Shifted RAG, RepoCoder
python function_retrieval_pipeline.py  # Sparse Retrieval, Dense Retrieval

# Static-analysis-enhanced baselines (prompts from external build pipelines)
# DraCo, GraphCoder, RepoScope: each has its own prompt preparation process; 
# see their respective papers/repos for detail.

# Agent-based baselines
# CodeAgent has its own prompt process, see its their respective papers/repos for detail.
python mini-swe-agent_interface.py     # Mini-SWE-Agent interface
python opencode_interface.py           # OpenCode interface

```

### LLM Inference

Use `inference.py` with an OpenAI-compatible API:

```bash
python inference.py \
    --prompt_path prompts/logicoder/logicoder.jsonl \
    --output_dir predictions/ \
    --model_str deepseek-v3-241226 \
    --T 0 \
    --N 1 \
    --mode rag
```

**Supported models** (configure API keys in `config/`):

| Model | `--model_str` | API Provider |
|-------|---------------|-------------|
| Kimi-K2 | `kimi-k2-250711` | Moonshot |
| DeepSeek-V3 | `deepseek-v3-241226` | DeepSeek |
| GPT-4o | `gpt-4o` | OpenAI-compatible proxy |
| Qwen2.5-Coder-14B | `qwen2.5-coder-14b-instruct` | Local |
| Llama-3.1-8B | `llama-3.1-8b` | Local |

**API Key Configuration**:
Place API keys in `ROOT_DIR/config/`:
- `api_keys.txt`

For **local models** (Qwen2.5-Coder, Llama-3.1), deploy with vLLM or similar and set the local endpoint in `inference.py`.

**Key parameters**:
- `--T 0`: Greedy decoding (used for Pass@1)
- `--N 1`: Number of samples per task
- `--mode rag`: RAG prompt mode (use `rag` for all RAG approaches)

---

## Configuration

All paths are centralized in [globals.py](globals.py). **You must modify this file** before running:

```python
class Globals:
    # Set this to your DevEval dataset root directory
    ROOT_DIR = '/path/to/your/DevEval'

    # Data paths (auto-derived from ROOT_DIR)
    BENCHMARK_METADATA_PATH = os.path.join(ROOT_DIR, 'data.jsonl')
    OUR_BENCHMARK_METADATA_PATH = os.path.join(ROOT_DIR, 'our_metadata.jsonl')
    FULL_BENCHMARK_METADATA_PATH = os.path.join(ROOT_DIR, 'full_metadata.jsonl')

    # Repository source code
    REPO_BASE_DIR = os.path.join(ROOT_DIR, 'Source_Code')

    # UniXcoder model path
    UNIXCODER_PATH = os.path.join(ROOT_DIR, 'models', 'microsoft/unixcoder-base')
```
---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Contact

For questions, issues, or collaboration inquiries, please contact:

- **Tanghaoran Zhang** — [zhangthr@nudt.edu.cn](mailto:zhangthr@nudt.edu.cn)

---
