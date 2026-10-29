"""生成物(images/ ocr_json/ chunks/ output_md/)を削除する。pdf/ には触れない

  --pdf を指定しない場合: 4つのフォルダの中身をすべて削除する(フォルダ自体は残す)
  --pdf を指定した場合 : そのPDFを含む報告書の生成物だけを削除する
                         (分割されたPDFの一部を指定しても、同じ報告書の残りのPDFの分も削除する)
"""

import shutil
from collections import defaultdict
from pathlib import Path

from .config import Config
from .target import TargetDocuments


class OutputCleaner:
    def __init__(self, config: Config, targets: TargetDocuments):
        self.config = config
        self.targets = targets
        self.roots = [config.image_dir, config.json_dir, config.chunk_dir, config.output_dir]

    def run(self) -> None:
        paths = self._collect()
        if not paths:
            print("削除する生成物はありません。")
            return
        self._show(paths)
        if not self._confirm():
            print("中止しました。")
            return
        for path in paths:
            self._delete(path)
        print(f"{len(paths)}件を削除しました。")

    # --- 削除対象の収集 ---

    def _collect(self) -> list[Path]:
        if self.targets.select_all:
            paths = [p for root in self.roots if root.is_dir() for p in root.iterdir()]
        else:
            paths = self._collect_targets()
        return sorted(p for p in dict.fromkeys(paths) if self._is_safe(p))

    def _collect_targets(self) -> list[Path]:
        cfg = self.config
        docs = self.targets.doc_names
        paths = []
        # ページ単位のファイル: 画像・OCR結果・page モードのページごとのMarkdown
        for folder in (cfg.image_dir, cfg.json_dir, cfg.page_md_dir):
            if folder.is_dir():
                paths += [p for p in folder.iterdir() if self._page_doc(p) in docs]
        # 報告書単位のフォルダ(section モード)
        for report in self.targets.reports:
            paths += [cfg.chunk_dir / report.name, cfg.output_dir / report.name]
        # page モードの結合結果
        paths += [cfg.output_dir / f"{doc}.md" for doc in docs]
        return [p for p in paths if p.exists() or p.is_symlink()]

    def _page_doc(self, path: Path) -> str | None:
        m = self.config.page_name_re.match(path.stem)
        return m["doc"] if m else None

    def _is_safe(self, path: Path) -> bool:
        """4つのフォルダの直下・配下にあり、pdf/ と重ならないものだけを削除対象にする"""
        resolved = path.parent.resolve() / path.name  # シンボリックリンク自体は辿らない
        pdf_dir = self.config.pdf_dir.resolve()
        if resolved == pdf_dir or pdf_dir.is_relative_to(resolved) or resolved.is_relative_to(pdf_dir):
            return False
        return any(resolved.is_relative_to(r.resolve()) and resolved != r.resolve() for r in self.roots)

    # --- 一覧の表示と確認 ---

    def _show(self, paths: list[Path]) -> None:
        """ページごとのファイルは PDF 単位にまとめて表示する(数百件になるため)"""
        base = self.config.pdf_dir.parent
        grouped: dict[str, int] = defaultdict(int)
        lines = []
        for path in paths:
            rel = path.relative_to(base) if path.is_relative_to(base) else path
            doc = self._page_doc(path)
            if path.is_file() and doc is not None:
                grouped[f"{rel.parent}/{doc}_p*{path.suffix}"] += 1
            elif path.is_dir() and not path.is_symlink():
                count = sum(1 for p in path.rglob("*") if p.is_file())
                lines.append(f"{rel}/ ({count}ファイル)")
            else:
                lines.append(str(rel))
        lines += [f"{pattern} ({n}ファイル)" for pattern, n in grouped.items()]
        print("次の生成物を削除します(pdf/ は削除しません):")
        for line in sorted(lines):
            print(f"  {line}")

    @staticmethod
    def _confirm() -> bool:
        try:
            answer = input("削除してよろしいですか? [y/N]: ")
        except EOFError:  # 対話できない環境では削除しない
            print()
            return False
        return answer.strip().lower() in ("y", "yes")

    @staticmethod
    def _delete(path: Path) -> None:
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink(missing_ok=True)
