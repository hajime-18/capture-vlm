"""各段階を順番に実行する"""

from .base import Step
from .config import Config
from .markdown_merger import MarkdownMerger
from .ocr_runner import NdlOcrRunner
from .pdf_converter import PdfToImageConverter
from .target import TargetDocuments
from .vlm_converter import VlmMarkdownConverter


class Pipeline:
    STEP_CLASSES: dict[str, type[Step]] = {
        "pdf": PdfToImageConverter,
        "ocr": NdlOcrRunner,
        "vlm": VlmMarkdownConverter,
        "merge": MarkdownMerger,
    }

    def __init__(self, config: Config, steps: list[str] | None = None, pdf_names: list[str] | None = None):
        self.config = config
        self.targets = TargetDocuments(config, pdf_names)
        keys = steps or list(self.STEP_CLASSES)
        self.steps = [self.STEP_CLASSES[k](config, self.targets) for k in keys]

    def run(self) -> None:
        if not self.targets.select_all:
            print("対象PDF: " + ", ".join(p.name for p in self.targets.pdfs))
        for step in self.steps:
            print(f"=== [{step.name}] ===")
            step.run()
        print("完了しました。")