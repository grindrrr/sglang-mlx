cmake_minimum_required(VERSION 3.26)
project(paged_attention_mlx LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_OSX_DEPLOYMENT_TARGET "13.0" CACHE STRING "Minimum macOS version")

# ── Python ────────────────────────────────────────────────────────────────────
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)

# ── nanobind ──────────────────────────────────────────────────────────────────
execute_process(
    COMMAND "${Python_EXECUTABLE}" -c
        "import nanobind, os; print(os.path.dirname(nanobind.__file__) + '/cmake')"
    OUTPUT_VARIABLE NB_DIR OUTPUT_STRIP_TRAILING_WHITESPACE
)
list(APPEND CMAKE_PREFIX_PATH "${NB_DIR}")
find_package(nanobind CONFIG REQUIRED)

# ── MLX ───────────────────────────────────────────────────────────────────────
execute_process(
    COMMAND "${Python_EXECUTABLE}" -c
        "import mlx; print(mlx.__path__[0])"
    OUTPUT_VARIABLE MLX_PYTHON_DIR OUTPUT_STRIP_TRAILING_WHITESPACE
)
list(APPEND CMAKE_PREFIX_PATH "${MLX_PYTHON_DIR}/share/cmake/MLX")
find_package(MLX CONFIG REQUIRED)

# ── Metal shader compilation ───────────────────────────────────────────────────
set(METALLIB_NAME "paged_attention")
set(METAL_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/paged_attention/metal")
set(METALLIB_OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${METALLIB_NAME}.metallib")
set(METALLIB_RUNTIME_OUTPUT
    "${CMAKE_CURRENT_SOURCE_DIR}/sglang_mlx/${METALLIB_NAME}.metallib")

set(METAL_SOURCES
    "${METAL_SRC_DIR}/attention/paged_attention_inst.metal"
    "${METAL_SRC_DIR}/cache/copy_blocks.metal"
    "${METAL_SRC_DIR}/cache/reshape_and_cache.metal"
    "${METAL_SRC_DIR}/convert_fp8.metal"
    "${METAL_SRC_DIR}/float8.metal"
    "${METAL_SRC_DIR}/utils.metal"
)

add_custom_command(
    OUTPUT "${METALLIB_OUTPUT}" "${METALLIB_RUNTIME_OUTPUT}"
    COMMAND xcrun -sdk macosx metal
        -fmodules-cache-path="${CMAKE_CURRENT_BINARY_DIR}/metal-module-cache"
        -I "${METAL_SRC_DIR}"
        -Wall -Wextra -fno-fast-math -Wno-c++17-extensions
        -gline-tables-only -frecord-sources=flat
        ${METAL_SOURCES}
        -o "${METALLIB_OUTPUT}"
    COMMAND ${CMAKE_COMMAND} -E copy
        "${METALLIB_OUTPUT}"
        "${METALLIB_RUNTIME_OUTPUT}"
    DEPENDS ${METAL_SOURCES}
    COMMENT "Building ${METALLIB_NAME}.metallib"
    VERBATIM
)
add_custom_target(
    metal_shaders
    DEPENDS "${METALLIB_OUTPUT}" "${METALLIB_RUNTIME_OUTPUT}"
)

# ── nanobind extension ─────────────────────────────────────────────────────────
nanobind_add_module(_ext NB_STATIC LTO
    bindings.cpp
    paged_attention/mlx_paged_attention.cpp
    paged_attention/mlx_cache.cpp
    paged_attention/mlx_device.cpp
)

add_dependencies(_ext metal_shaders)

target_include_directories(_ext PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/paged_attention"
    "${MLX_PYTHON_DIR}/include"
)

target_link_libraries(_ext PRIVATE mlx)

target_compile_definitions(_ext PRIVATE
    METALLIB_PATH="${METALLIB_NAME}.metallib"
)

# Default output to sglang_mlx/ for manual cmake builds.
# pip install -e . overrides this via -DCMAKE_LIBRARY_OUTPUT_DIRECTORY.
if(NOT CMAKE_LIBRARY_OUTPUT_DIRECTORY)
    set_target_properties(_ext PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/sglang_mlx"
    )
endif()

# Copy metallib alongside the .so at build time
add_custom_command(TARGET _ext POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
        "${CMAKE_CURRENT_BINARY_DIR}/${METALLIB_NAME}.metallib"
        "$<TARGET_FILE_DIR:_ext>/${METALLIB_NAME}.metallib"
    COMMENT "Copying ${METALLIB_NAME}.metallib to output directory"
)
