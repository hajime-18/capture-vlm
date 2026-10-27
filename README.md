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
                               │   目次・見出しの書式・柱・LLM を総合して章・節を判定し、
                               │   VLMに1回で入れる範囲(チャンク)を決めて保存する
                               ▼
              chunks/{報告書名}/chunk_0001.json …(+ plan.md で確認)
                               │ [vlm] すべてのチャンクが揃っていることを確認してから、
                               │       チャンクごとに画像(複数ページ)+OCR結果をVLMへ
                               ▼
              output_md/{報告書名}/chunks/chunk_0001.md …
                               │ [merge] 結合
                               ▼
              output_md/{報告書名}/{報告書名}.md (+ figures/F0001.png …)
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
│       ├── llm_judge.py        #   LLMによる見出しの判定
│       ├── analyzer.py         #   上記を総合して見出しを確定
│       ├── figures.py          #   図の番号付けと切り出し
│       ├── chunker.py          #   チャンク(VLMに入れる範囲)の決定
│       ├── planner.py          # [plan]
│       ├── validator.py        #   チャンクが揃っているかの確認
│       ├── prompts.py          #   section モードの指示文
│       ├── vlm.py              # [vlm]
│       └── merger.py           # [merge]
├── ndlocr-lite/                # NDLOCR-Lite(git submodule)
├── pdf/                        # ★ 処理したいPDFをここに置く
├── images/                     # (自動作成) ページ画像(VLM入力専用)
├── ocr_json/                   # (自動作成) OCR結果
├── chunks/                     # (自動作成) 区切りの判定結果
└── output_md/                  # (自動作成) Markdown
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

`plan` でLLMを使う設定(`boundary_llm`)のときは、手順1でもGPUを使います。

### オプション

| オプション | 説明 |
|---|---|
| `--pdf NAME ...` | 処理するPDFを指定します。省略すると `pdf/` 内のすべてが対象です。section モードでは、1冊の一部を指定すると残りのPDFも自動で対象になります。 |
| `--only STEP ...` | 指定した段階だけ実行します。section: `pdf` `ocr` `plan` `vlm` `merge` / page: `pdf` `ocr` `vlm` `merge` |
| `--mode {section,page}` | 処理方式です(省略時は `config.py` の `mode`)。 |
| `--overwrite` | 出力済みのものも作り直します。 |
| `--ocr-device {cuda,cpu}` | OCRに使うデバイスです(初期値は `cuda`)。 |

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
| LLM | 上の手がかりで判断が分かれる行を、OCRの文字列と文字の大きさからLLMに判定させる(画像は使わない) |

点数が `heading_accept_score` 以上の行を見出しとし、`heading_candidate_score` との間の行は、LLMで確認します。LLMが「見出しだ」と判断しても、長い文や句読点で終わる行は書式の点数が低いため見出しになりません(LLMの誤答だけで区切られないようにするため)。

`boundary_llm` で、LLMの使い方を選べます。

| 値 | 内容 |
|---|---|
| `"off"` | LLMを使わない(GPU不要で最も速い) |
| `"ambiguous"`(初期値) | 判断が分かれる行のあるページ、柱が変わったのに見出しが見つからないページ、目次にあるのに本文で見つからない見出しの付近だけ |
| `"all"` | 本文のすべてのページ(時間はかかるが、書式が独特な報告書に強い) |

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

### 判定結果の確認

`plan` の後、`chunks/{報告書名}/plan.md` に次の内容が書き出されます。VLMを回す前に確認してください。

- チャンクの一覧(ページの範囲、ページ途中の区切りの位置、上位の見出し、図の数)
- 判定した見出しの一覧(階層、根拠、点数)
- 本文で位置を特定できなかった目次の項目
- ページの種類(本文・目次・図版など)

`vlm` は、最初に対象のすべての報告書でチャンクが揃っていること(すべてのページがいずれかのチャンクに含まれること、画像・図の切り出しがあること)を確認し、1つでも欠けていれば処理を始めずに止まります。

### 保存されるデータ

| ファイル | 内容 |
|---|---|
| `ocr_json/{PDF名}_p0001.json` | NDLOCR-LiteのJSON(座標・確信度などすべて)に、XMLから行の種別(`type`)・読み順(`order`)・図や表などの領域(`blocks`)、柱・ノンブルの文字を加えたもの |
| `chunks/{報告書名}/chunk_0001.json` | チャンク1つ分。範囲(`scope`)、上位の見出し(`breadcrumb`)、範囲内の見出しと図、各ページの範囲内の行ID、各ページのOCR結果の全項目 |
| `chunks/{報告書名}/structure.json` | 判定した見出し、目次の項目と本文での位置、ページの種類など |
| `chunks/{報告書名}/plan.json` | チャンクの一覧(判定が最後まで終わると書き出される) |
| `output_md/{報告書名}/chunks/chunk_0001.md` | チャンクごとのMarkdown |
| `output_md/{報告書名}/{報告書名}.md` | 1冊分のMarkdown。各チャンクの先頭に `<!-- chunk: 0003 \| pages: 5-6 \| 第1節 … -->` の印が入る |

座標や確信度は、後の学習などに使えるよう、OCRの結果からチャンクのJSONまで削らずに保存しています。VLMには、トークン数を抑えるため文字列と種別だけを渡します。

## 設定

設定はすべて `pipeline/config.py` の `Config` クラスにまとまっています。

| 項目 | 初期値 | 説明 |
|---|---|---|
| `mode` | `"section"` | 処理方式(`"section"` / `"page"`) |
| `render_dpi` | `200` | VLM用にPDFを画像化するときの解像度 |
| `ndlocr_dir` | `ndlocr-lite` | NDLOCR-Liteを置いた場所 |
| `ocr_pdf_dpi` | `200` | NDLOCR-LiteがPDFを読み取るときの解像度 |
| `model_id` | `Qwen/Qwen3.5-9B` | 使用するVLM(区切りの判定のLLMも同じモデル) |
| `quantization` | `"4bit"` | `"4bit"` / `"8bit"` / `None`(量子化なし) |
| `max_new_tokens` | `8192` | 1回(1ページまたは1チャンク)あたりの最大出力トークン数 |
| `repetition_penalty` | `1.0` | 同じ行が繰り返し出力される場合は `1.05` 程度にする |
| `max_image_pixels` | `1_600_000` | page モードでVLMに渡す画像の最大画素数 |
| `section_max_image_pixels` | `1_200_000` | section モードでの1枚あたりの最大画素数(複数ページを入れるため小さめ) |
| `section_level` | `2` | 区切る見出しの階層(1=章, 2=節) |
| `max_pages_per_chunk` | `3` | 1回にVLMへ入れる最大ページ数(重複ページを含む) |
| `min_plate_run` | `2` | 図版ページがこの枚数以上続けば図版だけでまとめる |
| `boundary_llm` | `"ambiguous"` | 区切りの判定でのLLMの使い方(`"off"` / `"ambiguous"` / `"all"`) |
| `heading_accept_score` | `4.0` | 見出しと認める点数 |
| `heading_candidate_score` | `2.0` | LLMで確認する候補の点数 |
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