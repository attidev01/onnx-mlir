# Gemmini Hardware ABI Headers

This directory vendors the Gemmini hardware ABI headers needed by the
ONNX-MLIR Gemmini runtime. It is intentionally not a full copy of
`gemmini-rocc-tests`, but it is self-contained for the runtime build.

The most important files are:

```text
include/gemmini.h
include/gemmini_params.h
rocc-software/src/xcustom.h
```

`include/gemmini.h` defines the Gemmini software API used by the runtime.
`include/gemmini_params.h` defines the hardware configuration used when
compiling the runtime: array dimension, scratchpad shape, accumulator shape,
element types, scaling types, and enabled Gemmini features.

The runtime includes:

```c
#include "gemmini.h"
```

`include/gemmini.h` then includes the generated hardware parameters and RoCC
instruction helpers from this directory:

```text
include/gemmini_params.h
rocc-software/src/xcustom.h
```

By default, `Runtime/CMakeLists.txt` uses this bundled directory first so a
fresh checkout can build without machine-local paths such as
`/home/.../toolchains`. Developers who need a different Gemmini hardware
configuration can override the bundled headers with:

```bash
-DGEMMINI_ROCC_TESTS_DIR=/path/to/gemmini-rocc-tests
```

or:

```bash
-DGEMMINI_ROCC_INCLUDE_DIR=/path/to/gemmini-rocc-tests/include
```

When changing Gemmini hardware parameters, update this bundle from the matching
generated Gemmini header set so `gemmini.h`, `gemmini_params.h`, and
`rocc-software/src/xcustom.h` stay consistent.
