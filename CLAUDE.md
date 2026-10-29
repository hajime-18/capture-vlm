# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Pipeline that converts Japanese archaeological excavation report PDFs (発掘調査報告書) into database-friendly Markdown. NDLOCR-Lite (git submodule at `ndlocr-lite/`) OCRs the PDFs; the OCR result plus page images are fed to a Qwen VLM (Hugging Face transformers, bitsandbytes-quantized) to produce Markdown. `README.md` (Japanese) is the detailed user documentation — keep it in sync when changing config options, steps, or output formats.

Code comments, log messages, error messages, and docs are written in Japanese; follow that convention.

## Running

There is no test suite, linter, or build step. Everything runs through `main.py`, which refuses to run outside the venv.

```bash
source venv/bin/activate
CUDA_VISIBLE_DEVICES=1 python main.py                                   # all steps, all PDFs in pdf/
python main.py --pdf "eriseki_005*" --only pdf ocr plan                 # stop before VLM, then inspect chunks/<report>/plan.md
python main.py --pdf "eriseki_005*" --only vlm merge
python main.py --pdf "eriseki_005*" --only plan vlm merge --overwrite   # redo after changing boundary/chunk settings
python main.py --mode page --pdf NAME                                   # per-page mode
```

- Shared GPU server: always pick a free GPU via `CUDA_VISIBLE_DEVICES` (check `nvidia-smi`). Target GPU is Quadro GP100 16GB (Pascal: no bf16, so fp16 is used automatically; driver is CUDA 12.2 → PyTorch must be a cu126 build).
- Quote PDF names containing parentheses or wildcards.
- `plan` also loads the VLM unless `boundary_llm == "off"`; set it to `"off"` to iterate on heading/chunking logic quickly without a GPU.
- Every step skips outputs that already exist unless `--overwrite`, so reruns resume where they stopped. When testing a code change, remember to pass `--overwrite` (or delete the relevant outputs) or nothing will be regenerated.

## Architecture

**Steps and modes.** `pipeline/pipeline.py` defines `MODES`: an ordered dict of step name → `Step` subclass (`pipeline/base.py`) for each mode. `--only` choices are derived from it automatically. To add a step, subclass `Step` (implement `name` and `run()`) and register it in `MODES`. Raise `PipelineError` for user-actionable failures — `main.py` prints only the message for `PipelineError`/`FileNotFoundError`/`ValueError`.

- `section` mode (default): `pdf → ocr → plan → vlm → merge`, operating per *report* (book). Code lives in `pipeline/section/`.
- `page` mode: `pdf → ocr → vlm → merge`, one page per VLM call (`vlm_converter.py`, `markdown_merger.py`, `prompt.py`).

**All settings** are in the `Config` dataclass in `pipeline/config.py` (no CLI flags beyond those in `main.py`).

**Targets and reports.** `TargetDocuments` (`target.py`) resolves `--pdf` arguments (name, stem, glob, or path). In section mode, `section/report.py` groups split PDFs into one report using `Config.report_patterns` (regex with `report` and `order` groups), and `expand_to_reports()` pulls in sibling PDFs of a partially specified report. Per-page artifacts are named `{pdf_stem}_p{NNNN}` (`Config.page_stem` / `page_name_re`).

**pdf / ocr.** `pdf_converter.py` renders PNGs used *only* as VLM input. `ocr_runner.py` runs NDLOCR-Lite's `src/ocr.py` as a subprocess with `sys.executable` directly on the PDF (requires NDLOCR-Lite ≥ 1.3.0), then enriches the JSON with line type/reading order/blocks parsed from NDLOCR's XML, and reads running heads (柱) and page numbers (ノンブル) via a second subprocess (`ndl_region_ocr.py`) because NDLOCR detects but doesn't recognize those regions. `NdlOcrRunner._is_current()` detects old-format JSON so it is regenerated.

**plan (section mode, the core logic).** `section/planner.py` orchestrates:
1. `ocr_page.py` loads OCR JSON per page.
2. `analyzer.py` scores heading candidates by combining `toc.py` (TOC entries matched to body lines, which also give hierarchy), `headings.py` (format patterns like 第3章/第2節/Ⅱ/(1), OCR type, font size, line length, no trailing punctuation), running-head changes, and `page_map.py` (printed page number ↔ sequential page). Lines scoring between `heading_candidate_score` and `heading_accept_score` are sent to `llm_judge.py`, which shows the VLM a downscaled page image + OCR and asks only "is this a heading, and which level" (and its corrected text). VLM answers cannot by themselves promote long/punctuated lines to headings.
3. `chunker.py` decides chunk boundaries purely by rules (not the VLM): split at `section_level`; a page whose next heading starts mid-page goes into both chunks with line-range scoping; plate (図版) runs become separate chunks; oversize sections split by lower headings then by page (`max_pages_per_chunk`), with breadcrumbs of parent headings.
4. `figures.py` assigns report-wide figure IDs (`F0001`…) and crops them to `output_md/{report}/figures/`.
5. Writes `chunks/{report}/chunk_NNNN.json`, `structure.json`, `plan.md` (human review), and finally `plan.json` (its presence marks the plan as complete).

**vlm / merge (section mode).** `section/vlm.py` first runs `validator.py` across all target reports and aborts if any chunk/page/figure is missing. Each chunk sends multiple page images + in-scope OCR lines (text and type only) with prompts from `section/prompts.py`; missing/unknown figure IDs are warned and marked with `<!-- warning: missing figures: ... -->`. `section/merger.py` concatenates chunk Markdown into `output_md/{report}/{report}.md` with `<!-- chunk: ... -->` markers.

**VLM client.** `QwenVlmClient` in `vlm_converter.py` is shared by both modes and by `llm_judge.py`; `get_client()` caches a single instance per process so the model loads once. It handles quantization, fp16 fallback, multi-GPU `max_memory` from the `gpu_*` config values, image downscaling to a pixel budget, and clamping `max_new_tokens` to the context length. `main.py` sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` before torch is imported.

**Data retention.** Coordinates and confidences from OCR are preserved all the way into chunk JSON (for later training use); only the VLM prompt is stripped down to text and type. Don't drop these fields when refactoring.
