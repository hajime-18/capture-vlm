"""[merge] チャンクのMarkdownを報告書1冊分に結合する(section モード)

出力: output_md/{報告書名}/{報告書名}.md  (図は同じフォルダの figures/ を参照する)
"""

import json

from ..base import Step
from .validator import PlanValidator


class SectionMerger(Step):
    name = "Markdownの結合(報告書ごと)"

    def run(self) -> None:
        validator = PlanValidator(self.config)
        for report in self.targets.reports:
            plan_path = validator.plan_dir(report) / "plan.json"
            if not plan_path.exists():
                self.log(f"[スキップ] {report.name}: 区切りの判定がまだです")
                continue
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            out_dir = self.config.output_dir / report.name
            parts, missing = [], []
            for item in plan["chunks"]:
                md_path = out_dir / "chunks" / f"chunk_{item['chunk_id']}.md"
                s, e = item["pages"]
                marker = f"<!-- chunk: {item['chunk_id']} | pages: {s}-{e} | {item['title']} -->"
                if md_path.exists():
                    parts.append(f"{marker}\n{md_path.read_text(encoding='utf-8').strip()}\n")
                else:
                    missing.append(item["chunk_id"])
                    parts.append(f"{marker}\n<!-- warning: not converted -->\n")
            out = out_dir / f"{report.name}.md"
            out_dir.mkdir(parents=True, exist_ok=True)
            out.write_text("\n".join(parts), encoding="utf-8")
            note = f"(未変換のチャンク {len(missing)}個)" if missing else ""
            self.log(f"{out.relative_to(self.config.output_dir)}: {len(plan['chunks'])}チャンク {note}")