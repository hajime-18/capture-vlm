"""[1] PDF → ページ画像 (pypdfium2)

ここで作る画像はVLMへの入力専用。OCRはPDFから直接行う(ocr_runner.py)。
"""

import pypdfium2 as pdfium

from .base import Step


class PdfToImageConverter(Step):
    name = "PDF → 画像(VLM用)"

    def run(self) -> None:
        pdfs = self.targets.pdfs
        if not pdfs:
            self.log(f"PDFがありません: {self.config.pdf_dir}")
            return
        self.config.image_dir.mkdir(parents=True, exist_ok=True)
        for pdf_path in pdfs:
            self._convert(pdf_path)

    def _convert(self, pdf_path) -> None:
        scale = self.config.render_dpi / 72  # PDFの基準は72dpi
        pdf = pdfium.PdfDocument(pdf_path)
        try:
            n_new = 0
            for i in range(len(pdf)):
                out = self.config.image_dir / f"{self.config.page_stem(pdf_path.stem, i + 1)}.png"
                if out.exists() and not self.config.overwrite:
                    continue
                page = pdf[i]
                page.render(scale=scale).to_pil().save(out)
                page.close()
                n_new += 1
            self.log(f"{pdf_path.name}: {len(pdf)}ページ中 {n_new}ページを画像化")
        finally:
            pdf.close()