# capture_vlm

発掘調査報告書のPDFを、データベース化しやすいMarkdownに変換するパイプラインです。NDLOCR-LiteでPDFから文字を読み取り、そのOCR結果とページ画像をVLM(Qwen)に渡してMarkdownを生成します。

```
                   pdf/*.pdf
          ┌────────────┴────────────┐
          │ [1] PDF → 画像          │ [2] OCR (NDLOCR-Lite)
          │     (VLM入力専用)       │     PDFを直接読み取る
          ▼                         ▼
images/{PDF名}_p0001.png   ocr_json/{PDF名}_p0001.json
          └────────────┬────────────┘
                       │ [3] VLM → Markdown (画像 + OCR結果を入力)
                       ▼
          output_md/pages/{PDF名}_p0001.md
                       │ [4] 結合
                       ▼
             output_md/{PDF名}.md
```

画像化とOCRはどちらもPDFを入力とし、互いに依存しません。画像はVLMに渡すためだけに作るもので、OCRには使いません。両者はファイル名のページ番号(`_p0001`)で対応づけられます。

各段階とも出力済みのファイルはスキップするため、途中で止まっても再実行すれば続きから処理されます。

## ディレクトリ構成

```
capture_vlm/
├── main.py                 # 実行用スクリプト
├── requirements.txt
├── README.md
├── pipeline/
│   ├── config.py           # 設定(パス・モデル・メモリ上限など)
│   ├── base.py             # 各段階の基底クラス Step
│   ├── target.py           # 処理対象PDFの選択 TargetDocuments
│   ├── pdf_converter.py    # [1] PdfToImageConverter
│   ├── ocr_runner.py       # [2] NdlOcrRunner
│   ├── prompt.py           # VLMへの指示文
│   ├── vlm_converter.py    # [3] QwenVlmClient / MarkdownCleaner / VlmMarkdownConverter
│   ├── markdown_merger.py  # [4] MarkdownMerger
│   └── pipeline.py         # 各段階を順に実行する Pipeline
├── ndlocr-lite/            # NDLOCR-Lite(git submodule)
├── pdf/                    # ★ 処理したいPDFをここに置く
├── images/                 # (自動作成) ページ画像(VLM入力専用)
├── ocr_json/               # (自動作成) OCR結果のJSON
├── output_md/              # (自動作成) Markdown
└── venv/
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

### オプション

| オプション | 説明 |
|---|---|
| `--pdf NAME ...` | 処理するPDFを指定します。省略すると `pdf/` 内のすべてが対象です。 |
| `--only STEP ...` | 指定した段階だけ実行します。`pdf` / `ocr` / `vlm` / `merge` から選びます。 |
| `--overwrite` | 出力済みのファイルも作り直します。 |
| `--ocr-device {cuda,cpu}` | OCRに使うデバイスです(初期値は `cuda`)。 |

### 実行例

```bash
# PDFを指定する(拡張子は省略可、複数指定可)
python main.py --pdf arcou_vlm_test.pdf
python main.py --pdf "eriseki_005(1).pdf" 014_001_004

# ワイルドカードでまとめて指定する
python main.py --pdf "eriseki_005*"

# pdf/ 以外の場所にあるPDFを指定する
python main.py --pdf ~/Downloads/eriseki_010.pdf

# OCRまでを先に全件済ませ、あとでVLMを回す
python main.py --only pdf ocr
CUDA_VISIBLE_DEVICES=1 python main.py --only vlm merge

# 指示文を変えたので、Markdownだけ作り直す
CUDA_VISIBLE_DEVICES=1 python main.py --pdf arcou_vlm_test --only vlm merge --overwrite
```

括弧を含むファイル名とワイルドカードは、必ず引用符で囲んでください。囲まないと、シェルが括弧を構文として解釈したり、ワイルドカードを別のディレクトリで展開したりします。

## 出力

ページごとのMarkdownは `output_md/pages/` に出力されます。PDF1冊分を結合したものは `output_md/{PDF名}.md` です。結合したファイルには、各ページの先頭に次のような印が入ります。

```markdown
<!-- page: 12 -->
## 第3章 調査の成果
...
```

この印はHTMLコメントなのでMarkdownの表示には影響しません。データベースに取り込む際に、記述が何ページのものかを追うために使えます。

## 設定

設定はすべて `pipeline/config.py` の `Config` クラスにまとまっています。主な項目は次のとおりです。

| 項目 | 初期値 | 説明 |
|---|---|---|
| `render_dpi` | `200` | VLM用にPDFを画像化するときの解像度です。 |
| `ndlocr_dir` | `ndlocr-lite` | NDLOCR-Liteを置いた場所です。 |
| `ocr_pdf_dpi` | `200` | NDLOCR-LiteがPDFを読み取るときの解像度です。 |
| `model_id` | `Qwen/Qwen3.5-9B` | 使用するVLMです。 |
| `quantization` | `"4bit"` | `"4bit"` / `"8bit"` / `None`(量子化なし)から選びます。 |
| `max_image_pixels` | `1_600_000` | VLMに渡す画像の最大画素数です。超える画像は縮小して渡します。 |
| `max_new_tokens` | `8192` | 1ページあたりの最大出力トークン数です。 |
| `repetition_penalty` | `1.0` | 同じ行が繰り返し出力される場合は `1.05` 程度にします。 |
| `gpu_memory_limit_gib` | `14.0` | GPU1枚あたりに載せるモデルの上限です。 |
| `gpu_reserve_gib` | `2.0` | 推論時の作業領域として各GPUに残す量です。 |
| `min_gpu_free_gib` | `6.0` | 空きがこれ未満のGPUは使いません。 |

VLMへの指示文は `pipeline/prompt.py` で変更できます。

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

**`Qwen3VLVideoProcessor requires the Torchvision library`**
torchvisionが入っていません。手順2を実行してください。

**モデル読み込み中に `CUDA out of memory`**
GPUのメモリが足りません。`nvidia-smi` で空いているGPUを確認し、`CUDA_VISIBLE_DEVICES` で指定してください。それでも足りない場合は、`quantization` を `"4bit"` にするか、より小さいモデルに切り替えてください。

**`Some modules are dispatched on the CPU or the disk`**
量子化したモデルがGPUに収まり切っていません。上と同じ対処をしてください。

**推論中(`[1/1] ...` の後)に `CUDA out of memory`**
画像が大きすぎます。`max_image_pixels` を `1_000_000` 程度に下げてください。処理中に表示される「画像サイズ / 入力トークン数」が目安になります。

**`PDFが見つかりません`**
`--pdf` で指定した名前が見つかりません。表示される `pdf/` 内の一覧と見比べてください。括弧を含む名前は引用符で囲む必要があります。

**`NDLOCR-Liteが見つかりません`**
submoduleが取得されていません。`git submodule update --init` を実行してください。

**`このNDLOCR-LiteはPDFの直接入力に対応していません`**
submoduleのNDLOCR-Liteが古い版です。PDFの直接入力は1.3.0以降の機能なので、「NDLOCR-Liteのバージョンを変える」の手順で1.3.1以降に切り替えてください。

**OCRで `JSONが出力されなかったページ` と表示される**
直前に `ocr.py` が出力したエラーを確認してください。`--ocr-device cpu` で改善する場合は、onnxruntime-gpuの環境に問題があります。

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

`pipeline/base.py` の `Step` を継承したクラスを作り、`pipeline/pipeline.py` の `Pipeline.STEP_CLASSES` に登録します。

```python
from .base import Step

class DatabaseImporter(Step):
    name = "データベース登録"

    def run(self) -> None:
        for md in sorted(self.config.output_dir.glob("*.md")):
            # --pdf の指定に従う
            if not (self.targets.select_all or md.stem in self.targets.doc_names):
                continue
            ...
```

登録したキーは、自動的に `--only` の選択肢にも追加されます。