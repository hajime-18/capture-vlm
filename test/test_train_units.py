"""train モードの単体テスト(構造HTMLの正規化・TEDS-Struct・自作データの分割)

GPUは不要。結果は画面と test/results/train_units_{日時}.txt に出力する。
  python test/test_train_units.py
"""

import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.config import Config  # noqa: E402
from pipeline.train.data import TableDatasets  # noqa: E402
from pipeline.train.structure import table_shape, teds_struct, to_structure_html  # noqa: E402

GT = ('<table><thead><tr><td rowspan="2"></td><td colspan="2"></td></tr><tr><td></td><td></td></tr></thead>'
      '<tbody><tr><td></td><td></td><td></td></tr></tbody></table>')


def run() -> list[tuple[str, bool, str]]:
    results = []

    def check(name, cond, detail=""):
        results.append((name, bool(cond), detail))

    # 正規化: 文字・余計な属性・th・改行・前置きを取り除いて同じ形になる
    noisy = ('以下です\n```html\n<table border="1">\n<thead><tr><th rowspan=2>番号</th><th colspan="2">法量</th></tr>'
             '<tr><th>口径</th><th>器高</th></tr></thead>\n<tbody><tr><td>1</td><td>12.0</td><td>3.1</td></tr>'
             '</tbody></table>\n```')
    check("文字入りHTMLの正規化", to_structure_html(noisy) == GT, str(to_structure_html(noisy)))
    check("正規化は冪等", to_structure_html(GT) == GT)
    check("thead無し・th行は見出し", to_structure_html("<table><tr><th>a</th></tr><tr><td>1</td></tr></table>")
          == "<table><thead><tr><td></td></tr></thead><tbody><tr><td></td></tr></tbody></table>")
    check("表が無ければ None", to_structure_html("表はありません") is None)
    check("行数・列数(結合を展開)", table_shape(GT) == (3, 3), str(table_shape(GT)))

    # TEDS-Struct
    check("完全一致は 1", teds_struct(GT, GT) == 1.0)
    check("読めない出力は 0", teds_struct(None, GT) == 0.0)
    one_off = GT.replace(' colspan="2"', "", 1)
    s = teds_struct(one_off, GT)
    # 節点: table, thead, tr, td×2, tr, td×2, tbody, tr, td×3 = 13
    check("結合の違い1か所は 1 - 1/節点数", abs(s - (1 - 1 / 13)) < 1e-9, f"{s}")
    check("無関係な表ほど低い", teds_struct("<table><tbody><tr><td></td></tr></tbody></table>", GT) < s)

    # 自作データの分割: 同じページの表は同じ側、分割は種で決まる
    cfg = Config()
    ds = TableDatasets(cfg)
    same_page = {ds.own_split_of(f"eriseki_025_p0069_t0{i}") for i in range(1, 6)}
    check("同じページの表は同じ側", len(same_page) == 1, str(same_page))
    ids = [f"r_p{p:04d}_t01" for p in range(1, 401)]
    ratio = sum(ds.own_split_of(i) == "test" for i in ids) / len(ids)
    check("評価用の割合が own_test_ratio に近い", abs(ratio - cfg.own_test_ratio) < 0.06, f"{ratio:.3f}")
    other = TableDatasets(replace(cfg, train_seed=cfg.train_seed + 1))
    check("種を変えると分割が変わる", [ds.own_split_of(i) for i in ids] != [other.own_split_of(i) for i in ids])
    return results


def main():
    results = run()
    lines = [f"{'OK ' if ok else 'NG '} {name}" + (f"  ({detail})" if detail and not ok else "")
             for name, ok, detail in results]
    lines.append(f"\n{sum(ok for _, ok, _ in results)}/{len(results)} 件成功")
    text = "\n".join(lines)
    print(text)
    out = ROOT / "test" / "results" / f"train_units_{datetime.now():%Y%m%d_%H%M%S}.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n", encoding="utf-8")
    print(f"結果: {out}")


if __name__ == "__main__":
    main()
