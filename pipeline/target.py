"""処理対象のPDF(報告書)を決める"""

from fnmatch import fnmatch
from pathlib import Path

from .config import Config


class TargetDocuments:
    """
    names を指定しない場合は pdf/ 内のすべてのPDFが対象。
    names には次のいずれも指定できる:
      - ファイル名      eriseki_005(1).pdf
      - 拡張子なし      eriseki_005(1)
      - ワイルドカード  "eriseki_005*"   (シェルに展開されないよう引用符で囲む)
      - パス            ~/Downloads/report.pdf  (pdf/ 以外の場所でも可)
    """

    def __init__(self, config: Config, names: list[str] | None = None):
        self.config = config
        self.select_all = not names
        self.pdfs = self._all_pdfs() if self.select_all else self._resolve(names)
        self.doc_names = {p.stem for p in self.pdfs}

    def _all_pdfs(self) -> list[Path]:
        return sorted(self.config.pdf_dir.glob("*.pdf"))

    def _resolve(self, names: list[str]) -> list[Path]:
        found: dict[Path, None] = {}  # 順序を保ったまま重複を除く
        not_found = []
        for name in names:
            matches = self._match(name)
            if matches:
                found.update(dict.fromkeys(matches))
            else:
                not_found.append(name)
        if not_found:
            available = "\n".join(f"    {p.name}" for p in self._all_pdfs()) or "    (なし)"
            raise FileNotFoundError(
                f"PDFが見つかりません: {', '.join(not_found)}\n  {self.config.pdf_dir} 内のPDF:\n{available}"
            )
        return list(found)

    def _match(self, name: str) -> list[Path]:
        path = Path(name).expanduser()
        if path.is_file():
            return [path.resolve()]
        for candidate in (self.config.pdf_dir / name, self.config.pdf_dir / f"{name}.pdf"):
            if candidate.is_file():
                return [candidate]
        pattern = name if name.lower().endswith(".pdf") else f"{name}.pdf"
        return [p for p in self._all_pdfs() if fnmatch(p.name, pattern)]

    @property
    def reports(self):
        """対象のPDFを含む報告書(section モード用)。分割されたPDFは1冊にまとめる"""
        if not hasattr(self, "_reports"):
            from .section.report import ReportCatalog
            catalog = ReportCatalog(self.config, list(dict.fromkeys(self._all_pdfs() + self.pdfs)))
            self._reports = catalog.reports if self.select_all else catalog.containing(self.pdfs)
        return self._reports

    def expand_to_reports(self) -> list[str]:
        """1冊の一部のPDFだけが指定された場合、同じ報告書の残りのPDFも対象に加える。加えたPDF名を返す"""
        if self.select_all:
            return []
        current = {p.stem for p in self.pdfs}
        added = [pdf for r in self.reports for pdf in r.pdfs if pdf.stem not in current]
        self.pdfs += added
        self.doc_names |= {p.stem for p in added}
        return [p.name for p in added]

    def includes(self, page_stem: str) -> bool:
        """ページ画像などのファイル名(拡張子なし)が対象のPDFに属するか"""
        if self.select_all:
            return True
        m = self.config.page_name_re.match(page_stem)
        return bool(m) and m["doc"] in self.doc_names