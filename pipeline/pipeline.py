"""各段階を順番に実行する"""

from .base import Step
from .config import Config
from .markdown_merger import MarkdownMerger
from .ocr_runner import NdlOcrRunner
from .pdf_converter import PdfToImageConverter
from .section.merger import SectionMerger
from .section.planner import SectionPlanner
from .section.vlm import SectionVlmConverter
from .target import TargetDocuments
from .vlm_converter import VlmMarkdownConverter

# 処理方式ごとの段階(上から順に実行する)
MODES: dict[str, dict[str, type[Step]]] = {
    "page": {
        "pdf": PdfToImageConverter,
        "ocr": NdlOcrRunner,
        "vlm": VlmMarkdownConverter,
        "merge": MarkdownMerger,
    },
    "section": {
        "pdf": PdfToImageConverter,
        "ocr": NdlOcrRunner,
        "plan": SectionPlanner,
        "vlm": SectionVlmConverter,
        "merge": SectionMerger,
    },
}
ALL_STEP_NAMES = list(dict.fromkeys(name for steps in MODES.values() for name in steps))


class Pipeline:
    def __init__(self, config: Config, steps: list[str] | None = None, pdf_names: list[str] | None = None):
        if config.mode not in MODES:
            raise ValueError(f"mode は {list(MODES)} のいずれかです: {config.mode}")
        self.config = config
        self.step_classes = MODES[config.mode]
        unknown = [s for s in (steps or []) if s not in self.step_classes]
        if unknown:
            raise ValueError(f"{config.mode} モードに無い段階です: {unknown}(使える段階: {list(self.step_classes)})")
        self.targets = TargetDocuments(config, pdf_names)
        if config.mode == "section":
            added = self.targets.expand_to_reports()
            if added:
                print("同じ報告書の残りのPDFも対象にします: " + ", ".join(added))
        keys = steps or list(self.step_classes)
        self.steps = [self.step_classes[k](config, self.targets) for k in keys]

    def run(self) -> None:
        print(f"処理方式: {self.config.mode}")
        if not self.targets.select_all:
            print("対象PDF: " + ", ".join(p.name for p in self.targets.pdfs))
        for step in self.steps:
            print(f"=== [{step.name}] ===")
            step.run()
        print("完了しました。")