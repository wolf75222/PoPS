"""Expose the already selected ``_pops`` C++ specialization to runtime adapters.

The "production" DSL backend loads a ``.so`` loader via dlopen ; that loader resolves C++
symbols exported by the ``_pops`` extension (System::install_block, grid_context,
ensure_aux_width, etc.). CPython normally loads extensions with RTLD_LOCAL on Unix/macOS ;
the symbols then stay invisible to the loader and add_native_block fails at dlopen ("symbol
not found in flat namespace"). So we load ``_pops`` with RTLD_GLOBAL, then restore the flags
for the following imports. The already-loaded module keeps its global scope.

The native selector alone performs the authenticated RTLD_GLOBAL load after the resolved domain
has fixed Dim.  Importing this module before that cut is an error; afterwards it binds the C++
config/system types as attributes consumed by runtime adapters.
"""

from ._native_selector import selected_native_module as _selected_native_module
from ._version import authenticate_native_version as _authenticate_native_version


_native_module = _selected_native_module(required=True)
SystemConfig = _native_module.SystemConfig
ModelSpec = _native_module.ModelSpec
_System = _native_module.System
AmrSystemConfig = _native_module.AmrSystemConfig
_AmrSystem = _native_module.AmrSystem
StepAttemptRejected = _native_module.StepAttemptRejected
abi_key = _native_module.abi_key

_authenticate_native_version(_native_module)

del _authenticate_native_version, _native_module, _selected_native_module
