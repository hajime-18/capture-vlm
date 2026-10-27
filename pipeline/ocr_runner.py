"""[2] PDF → OCR JSON (capture_vlm/ndlocr-lite に置いたNDLOCR-LiteにPDFを直接入力する)

ページごとのJSONは、VLM用の画像と対応づけるため {PDF名}_p0001.json の名前で ocr_json/ に保存する。

NDLOCR-LiteのJSONは行の種別(class_index)が正しく入らない(ほぼすべて「本文」になる)ため、
同時に出力されるXMLから次の情報を補って保存する:
  - 各行の "type"  : 本文 / タイトル本文 / キャプション / 柱 / ノンブル など
  - ページの "blocks": 図版・表組・柱・ノンブルなどの領域
  - ページの "page" : OCR座標系での幅・高さ
また、NDLOCR-Liteは柱・ノンブルの領域を検出しても文字を読まないため、
その領域だけをNDLOCR-Liteの文字認識モデルで読み、ブロックの "text" に加える(ndl_region_ocr.py)。
元のJSONの項目(座標・確信度など)はすべてそのまま残す。
"""

import json
import shutil
import xml.etree.ElementTree as ET
import subprocess
import sys
import tempfile
from pathlib import Path

import pypdfium2 as pdfium

from .base import Step


OCR_FORMAT = "capture_vlm.ocr/3"
REGION_TYPES = {"柱", "ノンブル"}  # 文字を読み足すブロック  # XMLの情報を統合した形式。これが無いJSONは作り直す


class NdlOcrRunner(Step):
    name = "OCR (NDLOCR-Lite)"

    def run(self) -> None:
        command = self._build_command()
        self.config.json_dir.mkdir(parents=True, exist_ok=True)

        pending = self._pending_pdfs()
        if not pending:
            self.log("OCR待ちのPDFはありません")
            return
        device = self._resolve_device()
        total = sum(n for _, n in pending)
        self.log(f"{len(pending)}冊 / {total}ページをOCRします(device: {device})")
        self._run_ocr(command, pending, device)

    # ---------- 準備 ----------
    def _build_command(self) -> list[str]:
        """このプログラムと同じPython(venv)で ocr.py を実行する"""
        src = self.config.ndlocr_script.parent
        if not self.config.ndlocr_script.exists():
            raise FileNotFoundError(
                f"NDLOCR-Liteが見つかりません: {self.config.ndlocr_script}\n"
                "  git submodule update --init を実行してください(README の環境構築を参照)。"
            )
        if not (src / "pdf_output_utils.py").exists():  # PDF入力は1.3.0で追加された機能
            raise FileNotFoundError(
                "このNDLOCR-LiteはPDFの直接入力に対応していません(1.3.0以降が必要です)。\n"
                "  README の「NDLOCR-Liteのバージョンを変える」を参照してください。"
            )
        return [sys.executable, str(self.config.ndlocr_script)]

    def _pending_pdfs(self) -> list[tuple[Path, int]]:
        """ページのJSONが1つでも欠けているPDFを (パス, ページ数) で返す"""
        pending = []
        for pdf in self.targets.pdfs:
            n = self._page_count(pdf)
            done = all(self._is_current(self._json_path(pdf.stem, i)) for i in range(1, n + 1))
            if self.config.overwrite or not done:
                pending.append((pdf, n))
        return pending

    @staticmethod
    def _is_current(path: Path) -> bool:
        """現在の形式で保存されたJSONか(古い形式なら作り直す)"""
        if not path.exists():
            return False
        try:
            return json.loads(path.read_text(encoding="utf-8")).get("format") == OCR_FORMAT
        except (OSError, json.JSONDecodeError):
            return False

    @staticmethod
    def _page_count(pdf: Path) -> int:
        doc = pdfium.PdfDocument(pdf)
        try:
            return len(doc)
        finally:
            doc.close()

    def _json_path(self, doc_name: str, page_no: int) -> Path:
        return self.config.json_dir / f"{self.config.page_stem(doc_name, page_no)}.json"

    def _resolve_device(self) -> str:
        """cudaが指定されていても、onnxruntimeがGPUを使えない場合はcpuに切り替える"""
        if self.config.ocr_device != "cuda":
            return "cpu"
        try:
            import onnxruntime as ort
            if "CUDAExecutionProvider" in ort.get_available_providers():
                return "cuda"
        except ImportError:
            pass
        self.log("[警告] onnxruntimeでCUDAが使えないためCPUで実行します(onnxruntime-gpuを確認してください)")
        return "cpu"

    # ---------- 実行 ----------
    def _run_ocr(self, command: list[str], pdfs: list[tuple[Path, int]], device: str) -> None:
        """
        対象のPDFだけを一時ディレクトリに集めて1回で処理する(モデルの読み込みが1回で済む)。
        NDLOCR-Liteは常にテキスト付きPDF({PDF名}_text.pdf)も出力するため、
        出力も一時ディレクトリに受けて、必要なJSON・XMLだけを取り出す。
        """
        cfg = self.config
        with tempfile.TemporaryDirectory(dir=cfg.json_dir.parent) as tmp:
            stage, out = Path(tmp) / "in", Path(tmp) / "out"
            stage.mkdir()
            out.mkdir()
            for pdf, _ in pdfs:
                self._link_or_copy(pdf, stage / pdf.name)

            cmd = [*command,
                   "--sourcedir", str(stage.resolve()),
                   "--output", str(out.resolve()),
                   "--device", device,
                   "--pdf-render-dpi", str(cfg.ocr_pdf_dpi)]  # XMLも使うので --json-only は付けない
            result = subprocess.run(cmd, cwd=cfg.ndlocr_script.parent)
            if result.returncode != 0:
                self.log(f"[警告] ocr.py が終了コード {result.returncode} で終了しました")

            collected = {}
            for pdf, n in pdfs:
                collected.update(self._collect(out, pdf, n))
            self._read_regions(collected, pdfs, device, Path(tmp))
            for (doc_name, page_no), page in collected.items():
                self._json_path(doc_name, page_no).write_text(
                    json.dumps(page, ensure_ascii=False, indent=2), encoding="utf-8")

    def _collect(self, out: Path, pdf: Path, n_pages: int) -> dict[tuple[str, int], dict]:
        """NDLOCR-Liteの出力を、VLM用画像と同じ {PDF名}_p0001 のページ単位の形式に揃える"""
        doc_name = pdf.stem
        missing, collected = [], {}
        for i in range(1, n_pages + 1):
            page = self._read_page(out, doc_name, i, n_pages)
            if page is None:
                missing.append(i)
                continue
            # 一時ディレクトリのパスではなく、元のPDFの場所とページ番号を記録する
            page["imginfo"].update({"img_path": str(pdf.resolve()), "page_index": i - 1})
            collected[(doc_name, i)] = page

        if missing:
            pages = ", ".join(map(str, missing[:10])) + (" ..." if len(missing) > 10 else "")
            self.log(f"[警告] {doc_name}: JSONが出力されなかったページ {len(missing)}件 (p. {pages})")
        else:
            self.log(f"{doc_name}: {n_pages}ページ分のOCR結果を取得")
        return collected

    def _read_regions(self, collected: dict, pdfs: list[tuple[Path, int]], device: str, tmp: Path) -> None:
        """柱・ノンブルの領域の文字を読み、ブロックに "text" として加える"""
        paths = {pdf.stem: pdf for pdf, _ in pdfs}
        jobs, targets = [], {}
        for (doc_name, page_no), page in collected.items():
            blocks = [b for b in page.get("blocks", []) if b["type"] in REGION_TYPES]
            if not blocks:
                continue
            key = f"{doc_name}\t{page_no}"
            boxes = []
            for b in blocks:
                xs = [p[0] for p in b["boundingBox"]]
                ys = [p[1] for p in b["boundingBox"]]
                boxes.append([min(xs), min(ys), max(xs), max(ys)])
            jobs.append({"key": key, "pdf": str(paths[doc_name].resolve()), "page_index": page_no - 1, "boxes": boxes})
            targets[key] = blocks
        if not jobs:
            return
        job_path, out_path = tmp / "regions_job.json", tmp / "regions_out.json"
        job_path.write_text(json.dumps({"dpi": self.config.ocr_pdf_dpi, "jobs": jobs}, ensure_ascii=False), encoding="utf-8")
        script = Path(__file__).with_name("ndl_region_ocr.py")
        result = subprocess.run([sys.executable, str(script), str(self.config.ndlocr_script.parent),
                                 str(job_path), str(out_path), device])
        if result.returncode != 0 or not out_path.exists():
            self.log("[警告] 柱・ノンブルの文字を読み取れませんでした(区切りの判定の精度が下がります)")
            return
        texts = json.loads(out_path.read_text(encoding="utf-8"))
        for key, blocks in targets.items():
            for block, text in zip(blocks, texts.get(key, [])):
                block["text"] = text
        self.log(f"柱・ノンブルの文字を読み取り: {sum(len(b) for b in targets.values())}箇所")

    @classmethod
    def _read_page(cls, out: Path, doc_name: str, page_no: int, n_pages: int) -> dict | None:
        """
        NDLOCR-Liteの出力形式はページ数で異なる:
          複数ページ: {PDF名}_00001.json/.xml … ページごと (contents, imginfo)
          1ページ  : {PDF名}.json/.xml       … PDF全体の形式 (contents, pdfinfo, pages)
        """
        stem = f"{doc_name}_{page_no:05}" if n_pages > 1 else doc_name
        src = out / f"{stem}.json"
        if not src.exists():
            return None
        data = json.loads(src.read_text(encoding="utf-8"))
        if n_pages > 1:
            page = data
            page.setdefault("imginfo", {})
        else:
            page = {"contents": data.get("contents", []), "imginfo": dict((data.get("pages") or [{}])[0])}
        cls._merge_xml(page, out / f"{stem}.xml")
        page["format"] = OCR_FORMAT
        return page

    @staticmethod
    def _merge_xml(page: dict, xml_path: Path) -> None:
        """XMLの行種別・ブロック領域をJSONに加える(JSONの行とXMLのLINEは同じ順番で対応する)"""
        lines = page["contents"][0] if page.get("contents") else []
        page["blocks"] = []
        if not xml_path.exists():
            return
        root = ET.parse(xml_path).getroot()
        xml_page = root.find("PAGE")
        if xml_page is not None:
            page["page"] = {"width": int(xml_page.get("WIDTH", 0)), "height": int(xml_page.get("HEIGHT", 0))}
        for line, xml_line in zip(lines, root.iter("LINE")):
            line["type"] = xml_line.get("TYPE", "")
            if xml_line.get("ORDER") is not None:
                line["order"] = int(xml_line.get("ORDER"))
        for block in root.iter("BLOCK"):
            x, y = int(block.get("X", 0)), int(block.get("Y", 0))
            w, h = int(block.get("WIDTH", 0)), int(block.get("HEIGHT", 0))
            page["blocks"].append({
                "type": block.get("TYPE", ""),
                "boundingBox": [[x, y], [x, y + h], [x + w, y], [x + w, y + h]],
                "confidence": float(block.get("CONF", 0) or 0),
            })

    @staticmethod
    def _link_or_copy(src: Path, dst: Path) -> None:
        """シンボリックリンクを作れない環境(Windowsの一般ユーザーなど)ではコピーする"""
        try:
            dst.symlink_to(src.resolve())
        except OSError:
            shutil.copy2(src, dst)