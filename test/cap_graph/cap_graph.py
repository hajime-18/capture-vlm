"""OCR結果(ocr_json/)のブロックのうち、種別が「表組」のものだけを取り出し、
元のPDFからその領域の画像を切り出して保存する(表の検出結果を目で確認するため)

OCRの座標はOCRしたときのページ画像の大きさ(JSONの page.width / height)が基準なので、
PDFを --dpi で描画した画像の大きさに合わせて換算してから切り出す。
OCRが同じ表を何度も検出している場合は、それぞれを別の画像として保存する(重複の確認用)。

GPU・VLMは不要(先に ocr を実行しておくこと)。
  python test/cap_graph/cap_graph.py                       # ocr_json/ のすべて
  python test/cap_graph/cap_graph.py --pdf "013_001_055"   # PDF名(拡張子なし、ワイルドカード可)で絞る
  python test/cap_graph/cap_graph.py --dpi 300 --pad 8

出力: test/cap_graph/picture/{PDF名}_p{ページ:04d}_t{ページ内の番号:02d}.png
      test/cap_graph/picture/index.tsv(画像ごとのページ・座標・確信度の一覧)
"""

import argparse
import fnmatch
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import pypdfium2 as pdfium

ROOT = Path(__file__).resolve().parents[2]
JSON_DIR = ROOT / "ocr_json"
PDF_DIR = ROOT / "pdf"
OUT_DIR = Path(__file__).resolve().parent / "picture"
TABLE_TYPES = {"表組"}
NAME_RE = re.compile(r"^(?P<stem>.+)_p(?P<page>\d{4})\.json$")


def box(bbox) -> tuple[int, int, int, int]:
    xs = [p[0] for p in bbox]
    ys = [p[1] for p in bbox]
    return min(xs), min(ys), max(xs), max(ys)


def collect(pattern: str) -> dict[str, list[tuple[int, dict]]]:
    """PDF名 → [(ページ番号, OCRのJSON)](表組のブロックがあるページのみ)"""
    result: dict[str, list[tuple[int, dict]]] = defaultdict(list)
    for path in sorted(JSON_DIR.glob("*.json")):
        m = NAME_RE.match(path.name)
        if not m or not fnmatch.fnmatch(m["stem"], pattern):
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        if any(b.get("type") in TABLE_TYPES for b in data.get("blocks", [])):
            result[m["stem"]].append((int(m["page"]), data))
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="OCRで「表組」とされた領域を元のPDFから切り出して保存する")
    ap.add_argument("--pdf", default="*", help="PDF名(拡張子なし)。ワイルドカード可")
    ap.add_argument("--dpi", type=int, default=300, help="PDFを描画する解像度(既定 300)")
    ap.add_argument("--pad", type=int, default=4, help="切り出す範囲の余白(描画後の画素数)")
    args = ap.parse_args()

    targets = collect(args.pdf)
    if not targets:
        sys.exit(f"表組のブロックがあるOCR結果が見つかりません: {JSON_DIR}/{args.pdf}_p*.json")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = ["file\tpdf\tpage\tblock_index\tconfidence\tx0\ty0\tx1\ty1"]
    total = 0
    for stem, pages in targets.items():
        pdf_path = PDF_DIR / f"{stem}.pdf"
        if not pdf_path.exists():
            print(f"[警告] PDFがありません: {pdf_path}")
            continue
        pdf = pdfium.PdfDocument(pdf_path)
        try:
            for page_no, data in pages:
                index = (data.get("imginfo") or {}).get("page_index", page_no - 1)
                page = pdf[index]
                img = page.render(scale=args.dpi / 72).to_pil().convert("RGB")
                page.close()
                size = data.get("page") or {}
                info = data.get("imginfo") or {}
                ocr_w = int(size.get("width") or info.get("img_width") or img.width)
                ocr_h = int(size.get("height") or info.get("img_height") or img.height)
                sx, sy = img.width / ocr_w, img.height / ocr_h

                n = 0
                for i, block in enumerate(data.get("blocks", [])):
                    if block.get("type") not in TABLE_TYPES:
                        continue
                    x0, y0, x1, y1 = box(block["boundingBox"])
                    crop = (max(0, int(x0 * sx) - args.pad), max(0, int(y0 * sy) - args.pad),
                            min(img.width, int(x1 * sx) + args.pad), min(img.height, int(y1 * sy) + args.pad))
                    if crop[2] <= crop[0] or crop[3] <= crop[1]:
                        print(f"[警告] {stem} p.{page_no} ブロック{i}: 領域が空のため飛ばします")
                        continue
                    n += 1
                    name = f"{stem}_p{page_no:04d}_t{n:02d}.png"
                    img.crop(crop).save(OUT_DIR / name)
                    conf = float(block.get("confidence", 0) or 0)
                    rows.append(f"{name}\t{stem}\t{page_no}\t{i}\t{conf:.2f}\t{x0}\t{y0}\t{x1}\t{y1}")
                    print(f"{name}  確信度{conf:.2f}  OCR座標({x0},{y0})-({x1},{y1})")
                total += n
        finally:
            pdf.close()

    (OUT_DIR / "index.tsv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"\n{total}個の表を保存しました: {OUT_DIR}")


if __name__ == "__main__":
    main()
