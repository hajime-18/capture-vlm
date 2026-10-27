"""NDLOCR-Liteが領域だけ検出して文字を読まない「柱」「ノンブル」を、NDLOCR-Liteの文字認識モデルで読む

ocr_runner.py から別プロセスとして実行する(NDLOCR-Liteのモジュール名がこのプログラムと衝突しないように)。
  python ndl_region_ocr.py <NDLOCR-Liteのsrc> <ジョブJSON> <結果JSON> <cpu|cuda>
ジョブJSON: {"dpi": 200, "jobs": [{"key": "...", "pdf": "...", "page_index": 0, "boxes": [[x0, y0, x1, y1], ...]}]}
結果JSON  : {"<key>": ["読み取った文字列", ...]}
"""

import json
import sys
from pathlib import Path
from types import SimpleNamespace


def main(src: str, job_path: str, out_path: str, device: str) -> None:
    sys.path.insert(0, src)
    import numpy as np
    import pypdfium2 as pdfium
    import ocr as ndl  # NDLOCR-Lite の ocr.py

    base = Path(src)
    args = SimpleNamespace(
        rec_weights=str(base / "model" / "parseq-ndl-24x768-100-tiny-153epoch-tegaki3-r8data-202604.onnx"),
        rec_classes=str(base / "config" / "NDLmoji.yaml"), device=device, enable_tcy=False,
    )
    if not Path(args.rec_weights).exists():  # 版によってモデルのファイル名が違う場合に備える
        candidates = sorted((base / "model").glob("parseq-*768*.onnx")) or sorted((base / "model").glob("parseq-*.onnx"))
        args.rec_weights = str(candidates[-1])
    recognizer = ndl.get_recognizer(args)

    job = json.loads(Path(job_path).read_text(encoding="utf-8"))
    scale = float(job["dpi"]) / 72.0
    results, docs = {}, {}
    for item in job["jobs"]:
        doc = docs.setdefault(item["pdf"], pdfium.PdfDocument(item["pdf"]))
        img = doc[item["page_index"]].render(scale=scale).to_pil().convert("RGB")
        texts = []
        for x0, y0, x1, y1 in item["boxes"]:
            pad = max(4, int((y1 - y0) * 0.3))
            crop = img.crop((max(0, x0 - pad), max(0, y0 - pad), min(img.width, x1 + pad), min(img.height, y1 + pad)))
            texts.append(recognizer.read(np.array(crop)) if crop.width > 2 and crop.height > 2 else "")
        results[item["key"]] = texts
    for d in docs.values():
        d.close()
    Path(out_path).write_text(json.dumps(results, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main(*sys.argv[1:5])