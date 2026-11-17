"""onnxruntime(NDLOCR-Lite)でGPUを使うための準備

onnxruntime-gpu は cuDNN・CUDA のライブラリを自分では探さないため、venv に pip 版
(nvidia-cudnn-cu12 など)が入っていても、LD_LIBRARY_PATH に無ければ読み込めず
「libcudnn.so.9: cannot open shared object file」と出て CPU に切り替わる。
そこで InferenceSession を作る前に onnxruntime.preload_dlls() で読み込んでおく。

get_available_providers() はビルドに含まれる provider を返すだけで、実際に使えるかは
分からない。CUDA provider のライブラリを読み込めるかで判定し、使えなければ最初から
CPUExecutionProvider だけを指定する(読み込みに失敗したときのエラーログを出さないため)。

NDLOCR-Lite(submodule)は書き換えずに、その onnxruntime.InferenceSession を
install_session_hook() で差し替えて provider の選択とログを行う。
"""

import contextlib
import ctypes
import io
import os
from functools import cache

CUDA = "CUDAExecutionProvider"
CPU = "CPUExecutionProvider"


@cache
def cuda_ready() -> bool:
    """onnxruntime の CUDA provider が実際に使えるか(cuDNN などの読み込みも済ませる)"""
    try:
        import onnxruntime as ort
    except ImportError:
        return False
    if CUDA not in ort.get_available_providers():  # CPU版の onnxruntime
        return False
    if hasattr(ort, "preload_dlls"):  # 1.21以降。venv 内の nvidia-* パッケージから読み込む
        with contextlib.redirect_stdout(io.StringIO()):  # 見つからないライブラリの一覧が大量に出るため
            ort.preload_dlls(cuda=True, cudnn=True, msvc=False)
    capi = os.path.join(os.path.dirname(ort.__file__), "capi")
    try:
        ctypes.CDLL(os.path.join(capi, "libonnxruntime_providers_shared.so"), mode=ctypes.RTLD_GLOBAL)
        ctypes.CDLL(os.path.join(capi, "libonnxruntime_providers_cuda.so"))
    except OSError:
        return False
    return True


def usable_providers(device: str) -> list[str]:
    return [CUDA, CPU] if device == "cuda" and cuda_ready() else [CPU]


def install_session_hook(device: str) -> None:
    """
    以降に作られる onnxruntime.InferenceSession について、
      - 使えない provider(CUDA が使えない場合の CUDAExecutionProvider など)を除く
      - 実際に使われた provider を、プロセスの中で1回だけ表示する
    """
    import onnxruntime as ort

    usable = set(usable_providers(device))
    base = ort.InferenceSession
    logged = False

    class Session(base):
        def __init__(self, path_or_bytes, sess_options=None, providers=None, provider_options=None, **kwargs):
            nonlocal logged
            if providers:
                keep = [i for i, p in enumerate(providers) if (p[0] if isinstance(p, tuple) else p) in usable]
                if provider_options is not None:
                    provider_options = [provider_options[i] for i in keep] or None
                providers = [providers[i] for i in keep] or [CPU]
            super().__init__(path_or_bytes, sess_options, providers, provider_options, **kwargs)
            if not logged:
                logged = True
                print(f"  onnxruntime の provider: {self.get_providers()}", flush=True)

    ort.InferenceSession = Session
