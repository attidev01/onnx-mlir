# SPDX-License-Identifier: Apache-2.0

# Keep the active test tree minimal while Gemmini integration is being brought
# up. Restoring the MLIR lit suite gives the branch a real regression entry
# point again without depending on the larger backend/numerical test setup that
# was removed from this worktree.
foreach(tool FileCheck count not split-file llvm-dis)
  if (NOT TARGET ${tool})
    find_program(${tool}_EXECUTABLE ${tool}
      PATHS ${LLVM_TOOLS_BINARY_DIR} ${LLVM_TOOLS_DIR}
      NO_DEFAULT_PATH)
    if (${tool}_EXECUTABLE)
      add_custom_target(${tool})
    endif()
  endif()
endforeach()

add_subdirectory(mlir)
add_subdirectory(unit)
add_subdirectory(accelerators)
