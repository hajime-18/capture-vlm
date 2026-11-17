# capture_vlm

発掘調査報告書のPDFを、データベース化しやすいMarkdownに変換するパイプラインです。NDLOCR-LiteでPDFから文字を読み取り、そのOCR結果とページ画像をVLM(Qwen)に渡してMarkdownを生成します。

処理方式は2つあり、`config.py` の `mode` または `--mode` で切り替えます。

| 方式 | 内容 |
|---|---|
| `section`(初期値) | 報告書1冊をまとめて扱い、章・節の区切りごとに複数ページをVLMへ入れる。分割されたPDFは1冊にまとめる |
| `page` | 1ページずつVLMへ入れる |

## 処理の流れ(section モード)

```
                       pdf/*.pdf(1冊が複数のPDFに分かれていてもよい)
              ┌────────────────┴────────────────┐
              │ [pdf] PDF → 画像                │ [ocr] OCR (NDLOCR-Lite)
              │       (VLM入力専用)             │       PDFを直接読み取る
              ▼                                 ▼
   images/{PDF名}_p0001.png        ocr_json/{PDF名}_p0001.json
              └────────────────┬────────────────┘
                               │ [plan] 区切りの判定
                               │   目次・見出しの書式・柱・VLM を総合して章・節を判定し、
                               │   VLMに1回で入れる範囲(チャンク)を決めて保存する
                               ▼
              chunks/{報告書名}/chunk_0001.json …(+ plan.md で確認)
                               │ [vlm] すべてのチャンクが揃っていることを確認してから、
                               │       チャンクごとに画像(複数ページ)+OCR結果をVLMへ
                               │       表は1つずつ、切り出した画像+表の中のOCR結果をVLMへ
                               ▼
              output_md/{報告書名}/chunks/chunk_0001.md … (+ tables/T0001.html …)
                               │ [merge] 結合
                               ▼
              output_md/{報告書名}/{報告書名}.md (+ figures/F0001.png …, tables/T0001.html …)
```

page モードでは `plan` が無く、ページごとに `output_md/pages/{PDF名}_p0001.md` を作って `output_md/{PDF名}.md` に結合します。

各段階とも出力済みのものはスキップするため、途中で止まっても再実行すれば続きから処理されます。

## ディレクトリ構成

```
capture_vlm/
├── main.py                     # 実行用スクリプト
├── requirements.txt
├── pipeline/
│   ├── config.py               # 設定(すべての設定項目はここ)
│   ├── base.py                 # 各段階の基底クラス Step
│   ├── target.py               # 処理対象PDFの選択
│   ├── pipeline.py             # 処理方式ごとの段階の並びと実行
│   ├── pdf_converter.py        # [pdf] PDF → VLM用画像
│   ├── ocr_runner.py           # [ocr] NDLOCR-Lite の実行と結果の整形
│   ├── ndl_region_ocr.py       #       柱・ノンブルの文字の読み取り(ocr_runner から呼ぶ)
│   ├── vlm_converter.py        # [vlm] page モード / VLMクライアント(両モード共通)
│   ├── markdown_merger.py      # [merge] page モード
│   ├── prompt.py               # page モードの指示文
│   └── section/                # section モード
│       ├── report.py           #   分割されたPDFを1冊にまとめる
│       ├── ocr_page.py         #   OCR結果(1ページ)の扱い
│       ├── toc.py              #   目次の読み取り
│       ├── headings.py         #   見出しの書式による判定
│       ├── page_map.py         #   印刷ページ番号と通し番号の対応
│       ├── llm_judge.py        #   VLMによる見出しの判定(ページ画像も使う)
│       ├── analyzer.py         #   上記を総合して見出しを確定
│       ├── figures.py          #   図・表の番号付けと切り出し
│       ├── table_html.py       #   VLMが出力した表(HTML)を整えて保存
│       ├── chunker.py          #   チャンク(VLMに入れる範囲)の決定
│       ├── planner.py          # [plan]
│       ├── validator.py        #   チャンクが揃っているかの確認
│       ├── prompts.py          #   section モードの指示文
│       ├── vlm.py              # [vlm]
│       └── merger.py           # [merge]
│   └── train/                  # train モード(表の構造認識の学習と評価)
│       ├── structure.py        #   構造HTMLの正規化と TEDS-Struct
│       ├── data.py             #   自作データ・PubTabNet の読み込みと分割
│       ├── experiment.py       #   学習・評価の記録
│       ├── trainer.py          # [train] QLoRA による学習
│       └── evaluator.py        # [data] [eval] [report]
├── ndlocr-lite/                # NDLOCR-Lite(git submodule)
├── pdf/                        # ★ 処理したいPDFをここに置く
├── images/                     # (自動作成) ページ画像(VLM入力専用)
├── ocr_json/                   # (自動作成) OCR結果
├── chunks/                     # (自動作成) 区切りの判定結果
├── output_md/                  # (自動作成) Markdown
├── train_data/pubtabnet/       # (自動作成) 取得・変換した PubTabNet
└── experiments/                # (自動作成) 学習・評価の記録
```

OCRには国立国会図書館の [NDLOCR-Lite](https://github.com/ndl-lab/ndlocr-lite) を使用しています。NDLOCR-Liteは `ndlocr-lite/` に git submodule として配置し、その `src/ocr.py` をこのプログラムと同じPython(venv)で実行します。

## 動作環境

開発時の環境は次のとおりです。

| 項目 | 内容 |
|---|---|
| OS | Linux |
| Python | 3.12(pyenv) + venv |
| GPU | Quadro GP100 (16GB) を使用 |
| NVIDIAドライバ | CUDA 12.2 対応 |

注意点が2つあります。まず、ドライバがCUDA 12.2対応のため、PyTorchは **CUDA 12系向けのビルド(cu126)** を使います。CUDA 13向けのビルド(cu130)はこのドライバでは動きません。次に、GP100(Pascal世代)はbf16に対応していないため、プログラム側で自動的にfp16で計算します。

## 環境構築

### 0. リポジトリを取得する

NDLOCR-Liteをsubmoduleとして含むため、`--recursive` を付けてcloneします。

```bash
git clone --recursive <このリポジトリのURL>
cd capture_vlm
```

`--recursive` を付けずにcloneした場合は、あとから次のコマンドで取得できます。

```bash
git submodule update --init
```

### 1. venvを作成して有効にする

以降の作業はすべてvenvを有効にした状態で行います。

```bash
python -m venv venv
source venv/bin/activate
which python    # .../capture_vlm/venv/bin/python と表示されればOK
```

`main.py` はvenvの外で実行されると、エラーメッセージを出して停止します。

### 2. PyTorchをインストールする

PyTorchとtorchvisionは、必ず **同じCUDAバージョン向けのもの** を同時に入れます。バージョンが食い違うと、torchvisionの読み込み時にエラーになります。

```bash
pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu126
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__, torch.cuda.is_available())"
```

両方に `+cu126` が付き、最後が `True` になれば成功です。上のバージョンは動作確認済みのものです。CUDAのバージョンが異なる環境では、`cu126` の部分を合わせたうえで、[PyTorchの公式サイト](https://pytorch.org/get-started/locally/)で対応する組み合わせを確認してください。

### 3. その他のライブラリをインストールする

```bash
pip install -r requirements.txt
pip install -r ndlocr-lite/requirements.txt
```

### 4. (任意)OCRをGPUで動かす

NDLOCR-LiteはCPUでも動きますが、onnxruntimeをGPU版に入れ替えると高速になります。GPU対応はNDLOCR-Lite側ではベータ機能の扱いです。

```bash
pip uninstall -y onnxruntime
pip install onnxruntime-gpu
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

表示に `CUDAExecutionProvider` が含まれていればGPUで動きます。含まれていなくても、パイプラインは警告を出してCPUで処理を続けます。

### 5. モデルのダウンロード

初回実行時に、Hugging Faceから自動でダウンロードされます(Qwen3.5-9Bで約20GB)。保存先は `~/.cache/huggingface/` です。ホームディレクトリの容量が足りない場合は、環境変数 `HF_HOME` で保存先を変更してください。

## 使い方

`pdf/` にPDFを置いて実行します。共用サーバーでは、他の人が使っていないGPUを `CUDA_VISIBLE_DEVICES` で指定してください(番号は `nvidia-smi` で確認できます)。

```bash
CUDA_VISIBLE_DEVICES=1 python main.py
```

### おすすめの進め方(section モード)

区切りの判定結果を確認してからVLMを回すと、無駄な処理を避けられます。

```bash
# 1. 画像化・OCR・区切りの判定まで
CUDA_VISIBLE_DEVICES=1 python main.py --pdf "eriseki_005*" --only pdf ocr plan

# 2. 判定結果を確認する
less chunks/eriseki_005/plan.md

# 3. VLMでMarkdownにして結合する
CUDA_VISIBLE_DEVICES=1 python main.py --pdf "eriseki_005*" --only vlm merge
```

`plan` でVLMを使う設定(`boundary_llm` が `"off"` 以外)のときは、手順1でもGPUを使います。

### オプション

| オプション | 説明 |
|---|---|
| `--pdf NAME ...` | 処理するPDFを指定します。省略すると `pdf/` 内のすべてが対象です。section モードでは、1冊の一部を指定すると残りのPDFも自動で対象になります。 |
| `--only STEP ...` | 指定した段階だけ実行します。section: `pdf` `ocr` `plan` `vlm` `merge` / page: `pdf` `ocr` `vlm` `merge` |
| `--mode {section,page}` | 処理方式です(省略時は `config.py` の `mode`)。 |
| `--overwrite` | 出力済みのものも作り直します。 |
| `--ocr-device {cuda,cpu}` | OCRに使うデバイスです(初期値は `cuda`)。 |
| `--clean` | `images/` `ocr_json/` `chunks/` `output_md/` の生成物を、一覧を表示して確認したうえで削除します(`pdf/` は削除しません)。`--pdf` と組み合わせると、そのPDFを含む報告書の分だけを削除します。パイプラインは実行しません。 |

### 実行例

```bash
# PDFを指定する(拡張子は省略可、複数指定可)
python main.py --pdf "eriseki_005(1).pdf" 014_001_004

# ワイルドカードでまとめて指定する
python main.py --pdf "eriseki_005*"

# 1ページずつ処理する
python main.py --mode page --pdf arcou_vlm_test

# 設定を変えたので、区切りの判定からやり直す
python main.py --pdf "eriseki_005*" --only plan vlm merge --overwrite

# 生成物を削除する(すべて / 指定した報告書だけ)
python main.py --clean
python main.py --clean --pdf "eriseki_005*"
```

括弧を含むファイル名とワイルドカードは、必ず引用符で囲んでください。囲まないと、シェルが括弧を構文として解釈したり、ワイルドカードを別のディレクトリで展開したりします。

## section モードの詳細

### 分割されたPDFを1冊にまとめる

ファイル名が次の規則に当てはまるPDFは、同じ報告書として番号順に連結します。規則は `config.py` の `report_patterns` で追加・変更できます(`report` が報告書名、`order` が並び順になる正規表現)。

| ファイル名の例 | 報告書名 | 並び順 |
|---|---|---|
| `eriseki_005(1).pdf`, `eriseki_005(2).pdf` | `eriseki_005` | 括弧内の番号 |
| `014_001_004.pdf`, `014_005_008.pdf` | `014` | 開始ページの番号 |

規則に当てはまらないPDFは、1つのPDFを1冊として扱います。

### 区切りの判定

報告書ごとに書き方が違うため、次の手がかりを点数にして総合的に判定します。

| 手がかり | 内容 |
|---|---|
| 目次 | 目次ページから見出しと印刷上のページ番号を読み取り、本文の行と照合する。一致した見出しは確実性が高く、目次での字下げから階層(章・節)も決まる |
| 書式 | 「第3章」「第2節」「Ⅱ」「1.」「(1)」などの書き方、OCRの種別(見出し)、文字の大きさ、行の長さ、句読点で終わらないこと |
| 柱 | ページ上部の章題が変わったページは、章の始まりである可能性が高い |
| VLM | 上の手がかりで判断が分かれるページを、ページ画像(縮小したもの)とOCR結果からVLMに判定させる |

点数が `heading_accept_score` 以上の行を見出しとし、`heading_candidate_score` との間の行は、VLMで確認します。VLMが「見出しだ」と判断しても、長い文や句読点で終わる行は書式の点数が低いため見出しになりません(VLMの誤答だけで区切られないようにするため)。

**VLMに任せているのは「どの行が見出しで、章・節・項のどれか」の判定だけ** です。どこで区切るか(節単位、ページ途中の重複、図版のまとめ、ページ数の上限)は、すべてプログラム側(`chunker.py`)のルールで決めます。

VLMにはページ画像も入力するため、文字の大きさ・太さ・中央寄せ・余白などの見た目を判断に使えます。また、見出しの正しい表記も答えさせるので、OCRが見出しを読み誤った場合(例:「序」が「17」と読まれた)も、画像のとおりに直した表記で見出しの種類と階層を決め直します。直した表記はチャンクの見出しやVLMへの指示文に使い、`plan.md` にはOCRの読み取りと並べて表示します。

`boundary_llm` で、VLMを使う範囲を選べます。

| 値 | 内容 |
|---|---|
| `"off"` | VLMを使わない(GPU不要で最も速い) |
| `"ambiguous"`(初期値) | 判断が分かれる行のあるページ、見出しらしいのに階層が決まらない行(OCRの読み誤りの可能性)のあるページ、柱が変わったのに見出しが見つからないページ、目次にあるのに本文で見つからない見出しの付近だけ |
| `"all"` | 本文のすべてのページ(時間はかかるが、書式が独特な報告書に強い) |

`boundary_use_image` を `False` にすると、画像を使わずOCR結果だけで判定します(速いが、OCRの読み誤りは直せない)。

**ノンブル(印刷上のページ番号)について**: NDLOCR-Liteは柱とノンブルの領域を検出しても文字を読まないため、その領域だけをNDLOCR-Liteの文字認識モデルで読み足しています。ただし1桁の数字などは読み誤りが多いため、前後のページと番号が連続しているものだけを使います。ノンブルが使えない場合も、目次の見出しを本文と文字で照合し、そのずれから次の見出しの位置を予想します。

### チャンク(VLMに1回で入れる範囲)の決め方

- `section_level`(初期値 2 = 節)以上の見出しで区切ります。序文・例言・目次・報告書抄録などは、それぞれ独立したチャンクになります。
- 次の見出しがページの途中にある場合、**そのページは前後両方のチャンクに入れます**(取りこぼし防止)。VLMには「1枚目は○○から」「最後の画像は△△の手前まで」と範囲を指示し、OCR結果も範囲内の行だけを渡すので、同じ文章が二重に出力されることはありません。
- 図版ページ(図・写真が中心のページ)が `min_plate_run` 枚以上続く部分は、図版だけのチャンクにします。
- `max_pages_per_chunk` を超える節は、下位の見出し(項など)で分割し、それも無ければページ単位で分割します。分割された後半には「○○(続き)」と表示され、VLMには見出しを繰り返さないよう指示します。
- 2つ目以降のチャンクには、上位の見出し(例:「第3章 調査の成果 > 第2節 遺物」)を指示文に含め、見出しの階層が揃うようにしています。

### 図の番号

OCRが検出した図・写真に、報告書の中での通し番号(`F0001`, `F0002`, …)を振り、`output_md/{報告書名}/figures/F0001.png` として切り出します。VLMには、元の図があった位置に次のように書かせます。

```markdown
![第3図 SK01実測図](figures/F0003.png)
```

これで、Markdownのどの位置に何の図があったかを、切り出した画像と対応づけて確認できます。VLMが図を書き忘れた場合や、一覧に無いIDを書いた場合は、ログに警告を出し、そのチャンクの末尾に `<!-- warning: missing figures: F0005 -->` のような印を残します。

### 表の番号とHTML

OCRが検出した表(NDLOCR-Liteの「表組」の領域)にも、図と同じように報告書の中での通し番号(`T0001`, `T0002`, …)を振り、`output_md/{報告書名}/tables/T0001.png` として切り出します。本文のMarkdownでは表の中身を書かせず、元の表があった位置に次の1行だけを書かせます(表の中のOCRの行は、本文用の指示文から除き、`[表 T0001]` の印に置き換えます)。

```markdown
[表3 出土遺物観察表](tables/T0003.html)
```

表の中身は `vlm` の中で、チャンクの変換の後に表1つずつ別にVLMへ入れます。入力は切り出した表の画像と、表の中のOCRの行(高さの揃ったものごとに1行にまとめ、表の左端からの位置を付けたもの)です。VLMにはHTMLの `<table>`(結合セルは `rowspan`・`colspan`)で答えさせ、それを読み直して整えたものを、ブラウザでそのまま開ける次のHTMLとして `tables/T0001.html` に保存します。

```html
<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="utf-8">
<title>表3 出土遺物観察表</title>
</head>
<body>
<table id="T0015" data-report="013" data-rows="7" data-cols="8" data-status="ok" data-page="47" data-pdf="013_001_055" data-page-no="47" data-printed-page="39" data-image="T0015.png" data-x0="188" data-y0="626" data-x1="1239" data-y1="809" data-confidence="0.95">
<caption>表3 出土遺物観察表</caption>
<tr>
  <th data-row="1" data-col="1" rowspan="2">番号</th>
  <th data-row="1" data-col="3" colspan="3">法量</th>
  …
</tr>
</table>
</body>
</html>
```

- 見出しのセルは `<th>`、それ以外は `<td>` です。セル内の改行は `<br>` になります。
- `data-row`・`data-col` は結合セルを展開したときの位置(1始まり)で、`data-rows`・`data-cols` は表全体の大きさです。
- `data-page` などは元のページ(通し番号・PDF名・PDF内のページ番号・印刷ページ番号)、`data-x0`〜`data-y1` はOCRの座標での表の位置です。
- VLMの出力を表として読めなかった場合は `data-status="parse_error"` とし、出力をそのまま表の後の `<pre class="raw">` に残します。
- 本文のMarkdownに書かれなかった表のID・一覧に無い表のIDは、図と同じく `<!-- warning: missing tables: T0005 -->` のような印を残します。
- NDLOCR-Liteが表として検出しなかった表は、これまでどおり本文の中にMarkdownの表として書かせます。
- ページをまたぐ表は、ページごとに別の番号になります。
- `plan` をやり直して表の番号が変わった場合は、`vlm` も `--overwrite` で作り直してください(出力済みのHTMLはスキップされるため)。

### 判定結果の確認

`plan` の後、`chunks/{報告書名}/plan.md` に次の内容が書き出されます。VLMを回す前に確認してください。

- チャンクの一覧(ページの範囲、ページ途中の区切りの位置、上位の見出し、図・表の数、区切りの理由)
- 判定した見出しの一覧(階層、根拠、点数)
- 本文で位置を特定できなかった目次の項目
- ページの種類(本文・目次・図版など)

区切りの理由は、`plan` の実行中にもチャンクごとにログへ表示されます。理由は次のいずれかです。

| 理由 | 例 |
|---|---|
| 報告書の先頭 | 最初の見出しより前の部分(表紙など) |
| 区切りの見出し | `section_level` 以内の見出し。階層・根拠(目次・書式・柱・VLM)・点数も表示する。直後の見出しとの間に本文が無い場合は、まとめたことも表示する |
| 上限による分割 | `max_pages_per_chunk` を超えたため、下位の見出し、またはページの境目で区切った |
| 図版 | 図版ページが `min_plate_run` 枚以上続いた、または図版の後に本文が続いた |

`vlm` は、最初に対象のすべての報告書でチャンクが揃っていること(すべてのページがいずれかのチャンクに含まれること、画像・図・表の切り出しがあること)を確認し、1つでも欠けていれば処理を始めずに止まります。

### 保存されるデータ

| ファイル | 内容 |
|---|---|
| `ocr_json/{PDF名}_p0001.json` | NDLOCR-LiteのJSON(座標・確信度などすべて)に、XMLから行の種別(`type`)・読み順(`order`)・図や表などの領域(`blocks`)、柱・ノンブルの文字を加えたもの |
| `chunks/{報告書名}/chunk_0001.json` | チャンク1つ分。範囲(`scope`)、その位置で区切った理由(`reason`)、上位の見出し(`breadcrumb`)、範囲内の見出しと図・表(表は中のOCRの行ID `line_ids` も)、各ページの範囲内の行ID、各ページのOCR結果の全項目 |
| `chunks/{報告書名}/structure.json` | 判定した見出し、目次の項目と本文での位置、ページの種類など |
| `chunks/{報告書名}/plan.json` | チャンクの一覧(判定が最後まで終わると書き出される) |
| `output_md/{報告書名}/chunks/chunk_0001.md` | チャンクごとのMarkdown |
| `output_md/{報告書名}/figures/F0001.png` | 切り出した図 |
| `output_md/{報告書名}/tables/T0001.png`, `T0001.html` | 切り出した表と、その中身(HTML) |
| `output_md/{報告書名}/{報告書名}.md` | 1冊分のMarkdown。各チャンクの先頭に `<!-- chunk: 0003 \| pages: 5-6 \| 第1節 … -->` の印が入る |

座標や確信度は、後の学習などに使えるよう、OCRの結果からチャンクのJSONまで削らずに保存しています。VLMには、トークン数を抑えるため文字列と種別だけを渡します。

## 表の構造認識の学習(train モード)

表の画像から、文字を含まない構造HTML(行・列・結合セル)を出力するように、VLMをQLoRA(4bitに量子化したモデルにLoRAを付けて学習)で学習します。学習前(`base`)と学習後のモデルを同じ評価用データ・同じ設定で評価し、比較表を作ります。PDFは使いません。

```
data   → 学習・評価データの準備(PubTabNet の取得と変換、自作データの分割)
train  → 学習(experiments/runs/{実行名}/ に重みと記録を保存)
eval   → 評価(base と学習した実行。評価済みのものはスキップ)
report → 比較表(experiments/results.md)
```

### 学習データ

| 名前 | 内容 | 学習用 / 評価用 |
|---|---|---|
| `own` | `study/make_dataset/dataset`(表構造アノテーションツールの出力。`own_dataset_dir`) | ページ単位のハッシュで2割を評価用に分ける(`own_test_ratio`・`own_split_group`)。同じページの表は同じ側に入る。アノテーションが増えても、既存の表の振り分けは変わらない |
| `pubtabnet` | PubTabNet(Hugging Face の `apoidea/pubtabnet-html`。リビジョンを固定) | train の先頭 `pubtabnet_train_shards` ファイルから `pubtabnet_train_samples` 件、val から `pubtabnet_test_samples` 件を、乱数の種を固定して抜き出す |

PubTabNet の正解(文字入りのHTML)は、自作データと同じ決まり(`table thead tbody tr td` と `rowspan` `colspan` だけ、セルは空)に変換します。VLMの出力も同じ形にそろえてから採点します。

### 実行例

```bash
# 1. PubTabNet で学習し、学習前(base)と比べる
CUDA_VISIBLE_DEVICES=1 python main.py --mode train --train-data pubtabnet --run ptn

# 2. 学習済みの ptn を初期値にして、自作データで続けて学習する
CUDA_VISIBLE_DEVICES=1 python main.py --mode train --train-data own --init ptn --run ptn_own

# 3. 学習前の重みから、自作データだけで学習する(2 との比較用)
CUDA_VISIBLE_DEVICES=1 python main.py --mode train --train-data own --run own

# 評価と比較表だけ作り直す(評価済みのものはスキップ)
CUDA_VISIBLE_DEVICES=1 python main.py --mode train --only eval report --models base ptn ptn_own own
```

| オプション | 説明 |
|---|---|
| `--run NAME` | 学習の実行名。省略時は `20261117_153000_own_from-base` のように自動で付ける |
| `--init base\|実行名` | 重みの初期値。`base` は学習前のモデル、実行名を指定するとそのLoRAから続けて学習する |
| `--train-data` | 学習に使うデータ(`pubtabnet` `own`、複数可。混ぜてシャッフルする) |
| `--eval-data` | 評価に使うデータ(省略時は両方) |
| `--models` | 評価するモデル(`base` と実行名)。省略時は `base` と今回の実行 |

同じ実行名で学習済みならスキップします。途中で止まった実行や、作り直す場合は `--overwrite` を付けます。評価は表ごとに `predictions.jsonl` に追記するので、止まっても続きから再開できます。

### 記録されるもの

```
experiments/
  runs/{実行名}/
    run.json          重みの初期値(ベースモデル・その版・量子化、続きから学習した場合は元の実行名とLoRAのハッシュ、
                      さらにその元の初期値と学習データ)、学習データ(データセットごとの件数・中身のハッシュ・取得元の設定)、
                      長すぎて除外した表、指示文、ハイパーパラメータ、乱数の種、ライブラリとGPU、gitのコミット、損失
    train_ids.json    学習に使った表のID(データセットごと)
    train_log.jsonl   更新ごとの損失・学習率・勾配のノルム
    adapter/          学習したLoRAの重み
  evals/{モデル}/{データセット}_{ハッシュ}/
    eval.json         評価したモデル・評価データのハッシュ・評価設定・集計した指標
    predictions.jsonl 表ごとの出力(生の出力と正規化した構造HTML)・正解・点数
  results.csv         すべての評価の一覧
  results.md          比較表(評価データと評価設定が同じものだけを並べ、base との差 Δ を表示)
```

評価データの中身(IDと正解)と評価設定(ベースモデル・量子化・指示文・画素数・最大出力)のハッシュを記録しているので、アノテーションを追加して自作データの評価用が変わった場合は、比較表で別の表として分かれます。

### 指標

| 指標 | 内容 |
|---|---|
| `teds_struct` | TEDS-Struct(PubTabNet の論文の TEDS を、セルの文字を無視して計算したもの。1が完全一致) |
| `teds_struct_simple` / `_complex` | 結合セルの無い表 / ある表だけの TEDS-Struct |
| `exact_match` | 構造が完全に一致した割合 |
| `shape_match` | 行数・列数が一致した割合 |
| `parse_ok` | 出力を表として読めた割合 |
| `truncated` | 最大出力に達して打ち切られた件数 |

### 学習の設定

| 項目 | 初期値 | 説明 |
|---|---|---|
| `lora_r` / `lora_alpha` / `lora_dropout` | `16` / `32` / `0.05` | LoRAの設定 |
| `lora_target_modules` | 言語モデルの線形層 | LoRAを付ける層(画像エンコーダは学習しない) |
| `train_epochs` | `1` | エポック数 |
| `learning_rate` / `warmup_ratio` | `1e-4` / `0.03` | 学習率(コサインで減衰) |
| `grad_accum_steps` | `8` | 1表ずつ処理し、この件数ごとに重みを更新する |
| `train_max_image_pixels` | `1_000_000` | 学習・評価での表の画像の最大画素数 |
| `train_max_length` | `6144` | 画像+正解がこれより長い表は学習に使わない(run.json に記録) |
| `train_gpu_reserve_gib` | `6.0` | 学習時に各GPUに残す作業領域 |
| `fp16_loss_scale` | `1.0` | fp16(GP100・GV100)で学習するときの損失の倍率。PyTorch版の線形注意の層は、倍率を大きくすると勾配が溢れて更新が飛ばされる(1024倍で発生を確認) |
| `eval_max_new_tokens` | `4096` | 評価での最大出力トークン数 |
| `train_seed` | `42` | データの分割・抜き出し・LoRAの初期値・学習順の乱数の種 |

学習したLoRAを section モードなどの推論で使う場合は、`lora_adapter` に `experiments/runs/{実行名}/adapter` を指定します(ただし、学習は「文字を含まない構造」の出力なので、文字入りの表を出力させる現在の表の指示文とは目的が異なります)。

## 設定

設定はすべて `pipeline/config.py` の `Config` クラスにまとまっています。

| 項目 | 初期値 | 説明 |
|---|---|---|
| `mode` | `"section"` | 処理方式(`"section"` / `"page"`) |
| `render_dpi` | `200` | VLM用にPDFを画像化するときの解像度 |
| `ndlocr_dir` | `ndlocr-lite` | NDLOCR-Liteを置いた場所 |
| `ocr_pdf_dpi` | `200` | NDLOCR-LiteがPDFを読み取るときの解像度 |
| `model_id` | `Qwen/Qwen3.5-9B` | 使用するVLM(区切りの判定も同じモデル) |
| `quantization` | `"4bit"` | `"4bit"` / `"8bit"` / `None`(量子化なし) |
| `max_new_tokens` | `8192` | 1回(1ページまたは1チャンク)あたりの最大出力トークン数 |
| `repetition_penalty` | `1.0` | 同じ行が繰り返し出力される場合は `1.05` 程度にする |
| `max_image_pixels` | `1_600_000` | page モードでVLMに渡す画像の最大画素数 |
| `section_max_image_pixels` | `1_200_000` | section モードでの1枚あたりの最大画素数(複数ページを入れるため小さめ) |
| `table_max_image_pixels` | `1_600_000` | 表をHTMLにするときの、切り出した表の画像の最大画素数 |
| `section_level` | `2` | 区切る見出しの階層(1=章, 2=節) |
| `max_pages_per_chunk` | `3` | 1回にVLMへ入れる最大ページ数(重複ページを含む) |
| `min_plate_run` | `2` | 図版ページがこの枚数以上続けば図版だけでまとめる |
| `boundary_llm` | `"ambiguous"` | 区切りの判定でのVLMの使い方(`"off"` / `"ambiguous"` / `"all"`) |
| `boundary_use_image` | `True` | 区切りの判定でVLMにページ画像も入力する |
| `boundary_image_pixels` | `800_000` | 区切りの判定に使う画像の最大画素数(見出しの見た目が分かれば十分なので小さめ) |
| `heading_accept_score` | `4.0` | 見出しと認める点数 |
| `heading_candidate_score` | `2.0` | VLMで確認する候補の点数 |
| `report_patterns` | (上記) | 分割されたPDFを1冊にまとめるファイル名の規則 |
| `gpu_memory_limit_gib` | `14.0` | GPU1枚あたりに載せるモデルの上限 |
| `gpu_reserve_gib` | `2.0` | 推論時の作業領域として各GPUに残す量 |
| `min_gpu_free_gib` | `6.0` | 空きがこれ未満のGPUは使わない |

VLMへの指示文は、section モードは `pipeline/section/prompts.py`、page モードは `pipeline/prompt.py` で変更できます。

### 1回に入れるページ数の決め方

`max_pages_per_chunk` は、GPUのメモリに合わせて実測で決めてください。まず `3` で処理し、VLMの段階でOOMが出るなら `2` に下げるか `section_max_image_pixels` を小さくします。ログに表示される「画像の枚数 / 入力トークン数」が目安になります。32GBのGPUが使える場合は、`4`〜`5` に増やすと、節の途中での分割が減ります。

### モデルの選び方

Qwen3.8-27Bは4bitに量子化しても20GB前後あり、16GBのGP100 1枚には収まりません。32GBのGV100がまるごと空いている場合は、次のように切り替えて使えます。

```python
model_id: str = "Qwen/Qwen3.8-27B"
```

```bash
CUDA_VISIBLE_DEVICES=0 python main.py --only vlm merge
```

まず9Bで出力の質を確認し、遺構番号や数値の読み違いが目立つようであれば27Bを検討してください。

## トラブルシューティング

**`venvが有効になっていません` と表示される / `bitsandbytes` が見つからないと言われる**
venvが有効になっていません。`source venv/bin/activate` を実行してから再実行してください。トレースバックのパスが `~/.pyenv/versions/...` になっていれば、venvの外で実行されています。

**`PyTorch and torchvision were compiled with different CUDA major versions`**
torchとtorchvisionのCUDAバージョンが食い違っています。「環境構築」の手順2のとおり、同じ `--index-url` から両方を入れ直してください。

**`The NVIDIA driver on your system is too old`**
PyTorchのCUDAバージョンがドライバより新しすぎます。cu126版のPyTorchを入れ直してください。

**モデル読み込み中に `CUDA out of memory` / `Some modules are dispatched on the CPU or the disk`**
GPUのメモリが足りません。`nvidia-smi` で空いているGPUを確認し、`CUDA_VISIBLE_DEVICES` で指定してください。それでも足りない場合は、`quantization` を `"4bit"` にするか、より小さいモデルに切り替えてください。

**推論中に `CUDA out of memory`**
入力が大きすぎます。section モードでは `max_pages_per_chunk` か `section_max_image_pixels` を、page モードでは `max_image_pixels` を下げてください。

**`チャンクが揃っていないため、VLMの処理を始められません`**
表示された報告書について `python main.py --only plan` を実行してください。`ページ数が判定時と違います` と出る場合は、PDFを差し替えた可能性があるので `--only pdf ocr plan --overwrite` でやり直してください。

**`plan` で `画像またはOCR結果が揃っていません`**
`--only pdf ocr` を先に実行してください。以前の版で作ったOCR結果は形式が古いため、`ocr` の段階で自動的に作り直されます。

**区切りがおかしい**
`chunks/{報告書名}/plan.md` の「見出し」の表で、根拠と点数を確認してください。見出しの取りこぼしが多い場合は `boundary_llm` を `"all"` に、本文の箇条書きが見出しと判定される場合は `heading_accept_score` を上げてから、`--only plan --overwrite` でやり直します。

**`このNDLOCR-LiteはPDFの直接入力に対応していません`**
submoduleのNDLOCR-Liteが古い版です。「NDLOCR-Liteのバージョンを変える」の手順で1.3.1以降に切り替えてください。

**`PDFが見つかりません`**
`--pdf` で指定した名前が見つかりません。表示される `pdf/` 内の一覧と見比べてください。括弧を含む名前は引用符で囲む必要があります。

## NDLOCR-Liteのバージョンを変える

PDFを直接OCRするため、NDLOCR-Liteは **1.3.0以降** が必要です(動作確認は1.3.1)。submoduleは特定のコミットに固定されています。別のバージョンに変える場合は、次のようにします。

```bash
cd ndlocr-lite
git fetch --tags
git checkout 1.3.1          # 使いたいタグ
cd ..
pip install -r ndlocr-lite/requirements.txt
git add ndlocr-lite
git commit -m "Update NDLOCR-Lite to 1.3.1"
```

## ライセンス・クレジット

OCRには国立国会図書館の [NDLOCR-Lite](https://github.com/ndl-lab/ndlocr-lite) を使用しています。NDLOCR-Liteのライセンスは、同リポジトリの `LICENCE` と `LICENCE_DEPENDENCEIES` を参照してください。使用するVLM(Qwen)のライセンスは、Hugging Faceの各モデルページで確認してください。

## 処理段階を追加するには

`pipeline/base.py` の `Step` を継承したクラスを作り、`pipeline/pipeline.py` の `MODES` の該当する処理方式に登録します。登録した名前は、自動的に `--only` の選択肢にも追加されます。

```python
from .base import Step

class DatabaseImporter(Step):
    name = "データベース登録"

    def run(self) -> None:
        for report in self.targets.reports:      # section モードでは報告書単位で扱える
            md = self.config.output_dir / report.name / f"{report.name}.md"
            ...
```