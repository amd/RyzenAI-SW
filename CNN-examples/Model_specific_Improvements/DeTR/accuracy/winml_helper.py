# -------------------------------------------------------------------------
# Portions Copyright (c) Microsoft Corporation. Licensed under the MIT License.
# Adapted from the Windows ML Python sample in the Microsoft Learn documentation
# (https://learn.microsoft.com/en-us/windows/ai/new-windows-ml/get-started?tabs=python,
# source: https://github.com/MicrosoftDocs/windows-ai-docs, code license LICENSE-CODE).
# --------------------------------------------------------------------------
"""
WinML helper – register Windows ML execution providers with ONNX Runtime.

This module contains the WinML singleton class adapted from the official
Microsoft documentation:

    https://learn.microsoft.com/en-us/windows/ai/new-windows-ml/get-started?tabs=python

Prerequisites (pip):
    pip install wasdk-Microsoft.Windows.AI.MachineLearning[all] \
                wasdk-Microsoft.Windows.ApplicationModel.DynamicDependency.Bootstrap \
                onnxruntime-windowsml

The class discovers execution providers installed on the system via the
Windows App SDK ``ExecutionProviderCatalog`` API, downloads any that are
missing (``ensure_ready_async``), and registers their native libraries with
ONNX Runtime so they appear in ``ort.get_available_providers()`` /
``ort.get_ep_devices()``.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

_winml_instance = None


class WinML:
    """Singleton that registers Windows ML execution providers with ORT.

    Usage::

        from winml_helper import WinML
        registered = WinML().register_execution_providers_to_ort()
    """

    def __new__(cls, *args, **kwargs):
        global _winml_instance
        if _winml_instance is None:
            _winml_instance = super(WinML, cls).__new__(cls, *args, **kwargs)
            _winml_instance._initialized = False
        return _winml_instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True

        self._fix_winrt_runtime()

        from winui3.microsoft.windows.applicationmodel.dynamicdependency.bootstrap import (
            InitializeOptions,
            initialize,
        )
        import winui3.microsoft.windows.ai.machinelearning as winml

        self._win_app_sdk_handle = initialize(
            options=InitializeOptions.ON_NO_MATCH_SHOW_UI
        )
        self._win_app_sdk_handle.__enter__()

        catalog = winml.ExecutionProviderCatalog.get_default()
        self._providers = catalog.find_all_providers()
        self._ep_paths: dict[str, str] = {}
        for provider in self._providers:
            provider.ensure_ready_async().get()
            if provider.library_path == "":
                print(f"Skipping {provider.name} as it has no library path.")
                continue
            # The catalog can list one EP name twice (old and new package); a bare
            # DLL name resolves against onnxruntime\capi and fails, so keep the full path.
            prev = self._ep_paths.get(provider.name)
            if prev and Path(prev).is_absolute() and not Path(provider.library_path).is_absolute():
                continue
            self._ep_paths[provider.name] = provider.library_path
        self._registered_eps: list[str] = []

    def __del__(self):
        self._providers = None
        if hasattr(self, "_win_app_sdk_handle") and self._win_app_sdk_handle:
            self._win_app_sdk_handle.__exit__(None, None, None)

    # ------------------------------------------------------------------
    @staticmethod
    def _fix_winrt_runtime():
        """Remove the msvcp140.dll shipped inside the winrt-runtime package.

        That DLL can conflict with other libraries on the system.
        """
        from importlib import metadata

        site_packages_path = Path(
            str(metadata.distribution("winrt-runtime").locate_file(""))
        )
        dll_path = site_packages_path / "winrt" / "msvcp140.dll"
        if dll_path.exists():
            dll_path.unlink()

    # ------------------------------------------------------------------
    def register_execution_providers_to_ort(self) -> list[str]:
        """Register every discovered EP library with ONNX Runtime.

        Returns the list of successfully registered EP names.
        """
        import onnxruntime as ort

        for name, path in self._ep_paths.items():
            if name not in self._registered_eps:
                try:
                    ort.register_execution_provider_library(name, path)
                    self._registered_eps.append(name)
                except Exception as e:
                    print(
                        f"Failed to register execution provider {name}: {e}",
                        file=sys.stderr,
                    )
                    traceback.print_exc()
        return self._registered_eps
