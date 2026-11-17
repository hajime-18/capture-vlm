"""plan の結果(chunks/{報告書名}/)を確認する: 図・表の重複検出、図と表の重なり

(どのチャンクにも入らない図・表は plan のログに警告として出る)

GPU・VLMは不要(先に plan を実行しておくこと)。結果は画面と test/results/plan_{報告書名}_{日時}.txt に出力する。
  python test/check_plan.py 013
  python test/check_plan.py 013 --chunk-dir /別の/chunks   # chunks/ 以外に出力した plan を確認する
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OVERLAP = 0.8  # 小さい方の面積に対する重なりの割合がこれ以上なら重複とみなす


def rect(item: dict) -> tuple[float, float, float, float]:
    xs = [p[0] for p in item["boundingBox"]]
    ys = [p[1] for p in item["boundingBox"]]
    return min(xs), min(ys), max(xs), max(ys)


def overlap(a: dict, b: dict) -> float:
    ra, rb = rect(a), rect(b)
    w = max(0, min(ra[2], rb[2]) - max(ra[0], rb[0]))
    h = max(0, min(ra[3], rb[3]) - max(ra[1], rb[1]))
    area = min((ra[2] - ra[0]) * (ra[3] - ra[1]), (rb[2] - rb[0]) * (rb[3] - rb[1]))
    return w * h / area if area > 0 else 0.0


def pairs(xs: list[dict], ys: list[dict] | None = None) -> list[tuple[dict, dict, float]]:
    out = []
    for i, a in enumerate(xs):
        for b in (xs[i + 1:] if ys is None else ys):
            if a["page"] == b["page"] and (r := overlap(a, b)) >= OVERLAP:
                out.append((a, b, r))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("report", help="報告書名(chunks/ の下のフォルダ名)")
    parser.add_argument("--chunk-dir", type=Path, default=ROOT / "chunks")
    args = parser.parse_args()

    plan_dir = args.chunk_dir / args.report
    if not (plan_dir / "plan.json").exists():
        sys.exit(f"plan.json がありません: {plan_dir}(先に --only plan を実行してください)")
    plan = json.loads((plan_dir / "plan.json").read_text(encoding="utf-8"))
    figs, tables = {}, {}
    for item in plan["chunks"]:
        chunk = json.loads((plan_dir / item["file"]).read_text(encoding="utf-8"))
        figs.update({f["id"]: f for f in chunk.get("figures", [])})
        tables.update({t["id"]: t for t in chunk.get("tables", [])})
    figs, tables = list(figs.values()), list(tables.values())

    lines = [f"plan の確認 {datetime.now():%Y-%m-%d %H:%M:%S} / {plan_dir}",
             f"チャンク {len(plan['chunks'])}個 / 図 {len(figs)}個 / 表 {len(tables)}個", ""]
    problems = 0
    for title, found in (("図どうしの重複", pairs(figs)), ("表どうしの重複", pairs(tables)),
                         ("図と表の重なり", pairs(figs, tables))):
        lines.append(f"## {title}(重なり{OVERLAP:.0%}以上): {len(found)}組")
        lines += [f"  p.{a['page']} {a['id']}(確信度{a['confidence']:.2f}) - {b['id']}(確信度{b['confidence']:.2f})"
                  f" 重なり{r:.0%}" for a, b, r in found]
        problems += len(found)
    lines +=["", f"問題 {problems}件"]

    text = "\n".join(lines) + "\n"
    print(text, end="")
    out = Path(__file__).parent / "results" / f"plan_{args.report}_{datetime.now():%Y%m%d_%H%M%S}.txt"
    out.parent.mkdir(exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"結果: {out}")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
