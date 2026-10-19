"""[2] ページ画像 → OCR JSON (capture_vlm/ndlocr-lite に置いたNDLOCR-Liteを使用)"""

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from .base import Step


class NdlOcrRunner(Step):
    name = "OCR (NDLOCR-Lite)"

    def run(self) -> None:
        cfg = self.config
        command = self._build_command()
        cfg.json_dir.mkdir(parents=True, exist_ok=True)  # NDLOCR-Liteは出力先が無いと何もせずに終了する

        pending = self._pending_images()
        if not pending:
            self.log("OCR待ちの画像はありません")
            return
        device = self._resolve_device()
        self.log(f"{len(pending)}枚をOCRします(device: {device})")
        self._run_ocr(command, pending, device)
        self._report_missing(pending)

    def _build_command(self) -> list[str]:
        """このプログラムと同じPython(venv)で ocr.py を実行する"""
        script = self.config.ndlocr_script
        if not script.exists():
            raise FileNotFoundError(
                f"NDLOCR-Liteが見つかりません: {script}\n"
                "  git submodule update --init を実行してください(README の環境構築を参照)。"
            )
        return [sys.executable, str(script)]

    def _pending_images(self) -> list[Path]:
        return [p for p in sorted(self.config.image_dir.glob("*.png"))
                if self.targets.includes(p.stem)
                and (self.config.overwrite or not (self.config.json_dir / f"{p.stem}.json").exists())]

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

    def _run_ocr(self, command: list[str], images: list[Path], device: str) -> None:
        """未処理の画像だけを一時ディレクトリに集め、モデルの読み込みを1回で済ませる"""
        cfg = self.config
        with tempfile.TemporaryDirectory(dir=cfg.image_dir.parent) as stage:
            for p in images:
                self._link_or_copy(p, Path(stage) / p.name)
            cmd = [*command,
                   "--sourcedir", str(Path(stage).resolve()),
                   "--output", str(cfg.json_dir.resolve()),
                   "--device", device,
                   "--json-only"]
            result = subprocess.run(cmd, cwd=cfg.ndlocr_script.parent)
            if result.returncode != 0:
                self.log(f"[警告] ocr.py が終了コード {result.returncode} で終了しました")

    @staticmethod
    def _link_or_copy(src: Path, dst: Path) -> None:
        """シンボリックリンクを作れない環境(Windowsの一般ユーザーなど)ではコピーする"""
        try:
            dst.symlink_to(src.resolve())
        except OSError:
            shutil.copy2(src, dst)

    def _report_missing(self, images: list[Path]) -> None:
        missing = [p.name for p in images if not (self.config.json_dir / f"{p.stem}.json").exists()]
        if missing:
            self.log(f"[警告] JSONが出力されなかった画像: {len(missing)}件")
            for name in missing[:10]:
                self.log(f"  - {name}")