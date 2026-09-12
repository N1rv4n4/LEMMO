from __future__ import annotations

import os
from pathlib import Path


def configure_runtime() -> None:
    """Configure the validated TileLang path before FLA modules are constructed."""
    toolchain = os.environ.get("LEMMO_TOOLCHAIN")
    if toolchain:
        root = Path(toolchain).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"LEMMO_TOOLCHAIN does not exist: {root}")
        os.environ.setdefault("CUDA_HOME", str(root))
        compiler = root / "bin/x86_64-conda-linux-gnu-gcc"
        compiler_cxx = root / "bin/x86_64-conda-linux-gnu-g++"
        if compiler.is_file():
            os.environ.setdefault("CC", str(compiler))
        if compiler_cxx.is_file():
            os.environ.setdefault("CXX", str(compiler_cxx))
    os.environ.setdefault("FLA_TILELANG", "1")
    os.environ.setdefault("TILELANG_DEFAULT_TARGET", '{"kind":"cuda","arch":"sm_90"}')
    os.environ.setdefault("TILELANG_EXECUTION_BACKEND", "nvrtc")

    import tilelang
    from tvm import tirx
    from tilelang.jit.adapter.nvrtc import NVRTCKernelAdapter

    def dynamic_symbols(self):
        mapping = {}
        self._dynamic_symbolic_name_map = {}
        for index, parameter in enumerate(self.prim_func.params):
            try:
                buffer = self.prim_func.buffer_map[parameter]
            except KeyError:
                continue
            for dimension, shape in enumerate(buffer.shape):
                if isinstance(shape, tirx.Var) and shape not in mapping:
                    location = (index, dimension)
                    mapping[shape] = location
                    self._dynamic_symbolic_name_map[shape.name] = location
        return mapping

    NVRTCKernelAdapter._process_dynamic_symbolic = dynamic_symbols

