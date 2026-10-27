"""節ごとの処理(section モード)

  plan  : OCR結果から章・節の区切りを判定し、VLMに入れる範囲(チャンク)を作る
  vlm   : チャンクごとに複数ページの画像とOCR結果をVLMに入れてMarkdownにする
  merge : チャンクのMarkdownを報告書1冊分に結合する
"""