"""[4] ページごとのMarkdown → 1冊分のMarkdown"""

from collections import defaultdict
from pathlib import Path

from .base import Step


class MarkdownMerger(Step):
    name = "Markdownの結合"

    def run(self) -> None:
        docs = self._group_by_document()
        if not docs:
            self.log("結合するMarkdownがありません")
            return
        for doc, pages in sorted(docs.items()):
            self._write(doc, sorted(pages))

    def _group_by_document(self) -> dict[str, list[tuple[int, Path]]]:
        docs = defaultdict(list)
        for md in self.config.page_md_dir.glob("*.md"):
            m = self.config.page_name_re.match(md.stem)
            if m and self.targets.includes(md.stem):
                docs[m["doc"]].append((int(m["page"]), md))
        return docs

    def _write(self, doc: str, pages: list[tuple[int, Path]]) -> None:
        # ページの印はHTMLコメントなのでMarkdownの表示には影響しない
        parts = [f"<!-- page: {n} -->\n{p.read_text(encoding='utf-8').strip()}\n" for n, p in pages]
        (self.config.output_dir / f"{doc}.md").write_text("\n".join(parts), encoding="utf-8")
        self.log(f"{doc}.md ({len(pages)}ページ)")