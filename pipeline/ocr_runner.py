"""[2] PDF → OCR JSON (capture_vlm/ndlocr-lite に置いたNDLOCR-LiteにPDFを直接入力する)

ページごとのJSONは、VLM用の画像と対応づけるため {PDF名}_p0001.json の名前で ocr_json/ に保存する。
"""

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pypdfium2 as pdfium

from .base import Step


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
            done = all(self._json_path(pdf.stem, i).exists() for i in range(1, n + 1))
            if self.config.overwrite or not done:
                pending.append((pdf, n))
        return pending

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
        出力も一時ディレクトリに受けて、必要なJSONだけを取り出す。
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
                   "--pdf-render-dpi", str(cfg.ocr_pdf_dpi),
                   "--json-only"]
            result = subprocess.run(cmd, cwd=cfg.ndlocr_script.parent)
            if result.returncode != 0:
                self.log(f"[警告] ocr.py が終了コード {result.returncode} で終了しました")

            for pdf, n in pdfs:
                self._collect(out, pdf, n)

    def _collect(self, out: Path, pdf: Path, n_pages: int) -> None:
        """NDLOCR-Liteの出力を、VLM用画像と同じ {PDF名}_p0001.json の名前・形式に揃えて保存する"""
        doc_name = pdf.stem
        missing = []
        for i in range(1, n_pages + 1):
            page = self._read_page(out, doc_name, i, n_pages)
            if page is None:
                missing.append(i)
                continue
            # 一時ディレクトリのパスではなく、元のPDFの場所とページ番号を記録する
            page["imginfo"].update({"img_path": str(pdf.resolve()), "page_index": i - 1})
            self._json_path(doc_name, i).write_text(
                json.dumps(page, ensure_ascii=False, indent=2), encoding="utf-8")

        if missing:
            pages = ", ".join(map(str, missing[:10])) + (" ..." if len(missing) > 10 else "")
            self.log(f"[警告] {doc_name}: JSONが出力されなかったページ {len(missing)}件 (p. {pages})")
        else:
            self.log(f"{doc_name}: {n_pages}ページ分のJSONを保存")

    @staticmethod
    def _read_page(out: Path, doc_name: str, page_no: int, n_pages: int) -> dict | None:
        """
        NDLOCR-Liteの出力形式はページ数で異なる:
          複数ページ: {PDF名}_00001.json … ページごと (contents, imginfo)
          1ページ  : {PDF名}.json       … PDF全体の形式 (contents, pdfinfo, pages)
        """
        if n_pages > 1:
            src = out / f"{doc_name}_{page_no:05}.json"
            if not src.exists():
                return None
            data = json.loads(src.read_text(encoding="utf-8"))
            data.setdefault("imginfo", {})
            return data
        src = out / f"{doc_name}.json"
        if not src.exists():
            return None
        data = json.loads(src.read_text(encoding="utf-8"))
        return {"contents": data.get("contents", []), "imginfo": dict((data.get("pages") or [{}])[0])}

    @staticmethod
    def _link_or_copy(src: Path, dst: Path) -> None:
        """シンボリックリンクを作れない環境(Windowsの一般ユーザーなど)ではコピーする"""
        try:
            dst.symlink_to(src.resolve())
        except OSError:
            shutil.copy2(src, dst)