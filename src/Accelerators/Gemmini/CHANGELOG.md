# Gemmini Backend — Source Changelog

For the full development history and version notes see
[`gemmini/Gemmini_Docs/CHANGELOG.md`](../../../gemmini/Gemmini_Docs/CHANGELOG.md).

## v3.0.0 — 09 June 2026

- Added `GEMMINI_VERSION "3.0.0"` cache variable to `CMakeLists.txt`
- Added one-line constraint comments to all 18 `isSupportedGemmini*` predicate
  functions in `Conversion/ONNXToGemmini/ONNXToGemmini.cpp`
- All 41 `om_gemmini_*` public functions in
  `Runtime/OMRuntimeGemmini.cpp` have `/** ... */` Doxygen doc comments
- All ODS op classes in `Dialect/GemminiOps.td` and
  `Dialect/GemminiLowOps.td` have `summary` fields
