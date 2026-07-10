# SPDX-License-Identifier: Apache-2.0

if (ONNX_MLIR_ACCELERATORS)
  foreach(t ${ONNX_MLIR_ACCELERATORS})
    if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${t}/CMakeLists.txt)
      add_subdirectory(${t})
    else()
      string(TOLOWER ${t} t_lower)
      if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${t_lower}/CMakeLists.txt)
        add_subdirectory(${t_lower})
      endif()
    endif()
  endforeach(t)
endif(ONNX_MLIR_ACCELERATORS)
