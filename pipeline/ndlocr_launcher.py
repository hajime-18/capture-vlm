"""NDLOCR-Lite の ocr.py を、onnxruntime の準備(ort_setup.py)をしてから実行する

ocr_runner.py から別プロセスとして実行する。引数は ocr.py のパスに続けて ocr.py の引数をそのまま渡す。
  python ndlocr_launcher.py <ocr.pyのパス> --sourcedir ... --device cuda ...
"""

import runpy
import sys
from pathlib import Path

from ort_setup import install_session_hook


def main() -> None:
    script, *args = sys.argv[1:]
    device = args[args.index("--device") + 1] if "--device" in args else "cpu"
    install_session_hook(device)
    sys.argv = [script, *args]
    sys.path[0] = str(Path(script).resolve().parent)  # このフォルダのモジュール名と衝突しないよう置き換える
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
