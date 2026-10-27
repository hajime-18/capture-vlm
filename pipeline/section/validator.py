"""VLMに進む前に、すべてのチャンク(JSONと画像の組み合わせ)が揃っているかを確認する"""

import json
from pathlib import Path

from ..config import Config
from .report import Report

PLAN_FORMAT = "capture_vlm.plan/1"


class PlanValidator:
    def __init__(self, config: Config):
        self.config = config

    def plan_dir(self, report: Report) -> Path:
        return self.config.chunk_dir / report.name

    def problems(self, report: Report) -> list[str]:
        d = self.plan_dir(report)
        plan_path = d / "plan.json"
        if not plan_path.exists():
            return ["区切りの判定(plan)がまだ行われていません"]
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        if plan.get("format") != PLAN_FORMAT or plan.get("status") != "complete":
            return ["区切りの判定が途中で終わっています。plan をやり直してください"]
        problems = []
        if plan.get("page_count") != len(report.pages):
            problems.append(f"ページ数が判定時({plan.get('page_count')})と違います({len(report.pages)})。PDFが変わった可能性があります")
        covered: set[int] = set()
        for item in plan.get("chunks", []):
            path = d / item["file"]
            if not path.exists():
                problems.append(f"チャンクのファイルがありません: {item['file']}")
                continue
            chunk = json.loads(path.read_text(encoding="utf-8"))
            for page in chunk["pages"]:
                covered.add(page["global_page"])
                if not (self.config.image_dir / page["image"]).exists():
                    problems.append(f"画像がありません: {page['image']} (チャンク {chunk['chunk_id']})")
            for fig in chunk.get("figures", []):
                if not (self.config.output_dir / report.name / fig["image"]).exists():
                    problems.append(f"図の画像がありません: {fig['image']} (チャンク {chunk['chunk_id']})")
        missing = sorted(set(range(1, len(report.pages) + 1)) - covered)
        if missing:
            problems.append(f"どのチャンクにも含まれないページがあります: 通し{missing[:10]}")
        return problems