cmake_minimum_required(VERSION 3.24)
project(stage-positionner VERSION 0.0.1)

find_package(pipeline-core REQUIRED)
find_package(Vulkan REQUIRED)

# ── Eigen3 ────────────────────────────────────────────────────────────────────
find_package(Eigen3 REQUIRED NO_MODULE)

# ── libigl (header-only, fetched if not present) ─────────────────────────────
find_package(libigl QUIET)
if(NOT libigl_FOUND)
    include(FetchContent)
    FetchContent_Declare(
        libigl
        GIT_REPOSITORY https://github.com/libigl/libigl.git
        GIT_TAG        v2.5.0
    )
    set(LIBIGL_USE_STATIC_LIBRARY OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(libigl)
endif()

# ── GLSL → SPIR-V compilation ─────────────────────────────────────────────────
find_program(GLSLC glslc REQUIRED)
set(SHADER_OUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/shaders/")
file(MAKE_DIRECTORY "${SHADER_OUT_DIR}")

foreach(SHADER suction.vert suction.frag pack_droptest.comp)
    add_custom_command(
        OUTPUT  "${SHADER_OUT_DIR}${SHADER}.spv"
        COMMAND ${GLSLC} -o "${SHADER_OUT_DIR}${SHADER}.spv"
                         "${CMAKE_CURRENT_SOURCE_DIR}/shaders/${SHADER}"
        DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/shaders/${SHADER}"
        COMMENT "Compiling ${SHADER} → ${SHADER}.spv"
    )
endforeach()
add_custom_target(suction-shaders DEPENDS
    "${SHADER_OUT_DIR}suction.vert.spv"
    "${SHADER_OUT_DIR}suction.frag.spv"
    "${SHADER_OUT_DIR}pack_droptest.comp.spv")

# ── Sources ───────────────────────────────────────────────────────────────────
set(STAGE_SOURCES
    src/PositionnerStage.cpp
    src/OrientationOptimizer.cpp
    src/SuctionPass.cpp
    src/PackingPass.cpp
)

# ── Plugin .dylib — loaded by the C++ runner at runtime ──────────────────────
add_library(stage-positionner SHARED ${STAGE_SOURCES})
add_dependencies(stage-positionner suction-shaders)
target_include_directories(stage-positionner PRIVATE include src)
target_link_libraries(stage-positionner PRIVATE
    pipeline-core Eigen3::Eigen igl::core Vulkan::Vulkan)
target_compile_definitions(stage-positionner PRIVATE
    SUCTION_SHADER_DIR="${SHADER_OUT_DIR}")
set_target_properties(stage-positionner PROPERTIES PREFIX "")

# ── Python extension — only built via scikit-build-core (pip install .) ───────
if(DEFINED SKBUILD)
    find_package(pybind11 CONFIG REQUIRED)
    pybind11_add_module(_stage_positionner python/bindings.cpp ${STAGE_SOURCES})
    add_dependencies(_stage_positionner suction-shaders)
    target_include_directories(_stage_positionner PRIVATE include src)
    target_link_libraries(_stage_positionner PRIVATE
        pipeline-core Eigen3::Eigen igl::core Vulkan::Vulkan)
    target_compile_definitions(_stage_positionner PRIVATE
        SUCTION_SHADER_DIR="${SHADER_OUT_DIR}")
    install(TARGETS _stage_positionner DESTINATION stage_positionner)
endif()

# ── Tests ─────────────────────────────────────────────────────────────────────
if(BUILD_TESTING AND NOT DEFINED SKBUILD)
    enable_testing()

    add_executable(test_half_edge tests/test_half_edge.cpp)
    target_include_directories(test_half_edge PRIVATE include src)
    target_link_libraries(test_half_edge PRIVATE pipeline-core Eigen3::Eigen)
    add_test(NAME test_half_edge COMMAND test_half_edge)

    add_executable(test_orientation tests/test_orientation.cpp src/OrientationOptimizer.cpp)
    target_include_directories(test_orientation PRIVATE include src)
    target_link_libraries(test_orientation PRIVATE pipeline-core Eigen3::Eigen igl::core)
    add_test(NAME test_orientation COMMAND test_orientation)

    add_executable(test_packing tests/test_packing.cpp src/PackingPass.cpp)
    target_include_directories(test_packing PRIVATE include src)
    target_link_libraries(test_packing PRIVATE pipeline-core Eigen3::Eigen Vulkan::Vulkan)
    add_test(NAME test_packing COMMAND test_packing)

    # ── GPU packing test — needs LavaCake for createVulkanContext() ───────────
    include(FetchContent)
    FetchContent_Declare(LavaCake
        GIT_REPOSITORY https://github.com/ThibaultTricard/LavaCake.git
        GIT_TAG        e54f0a8)
    set(LAVACAKE_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(LavaCake)

    add_executable(test_packing_gpu
        tests/test_packing_gpu.cpp
        src/PackingPass.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../Runner/src/GpuRuntime.cpp)
    add_dependencies(test_packing_gpu suction-shaders)
    target_include_directories(test_packing_gpu PRIVATE
        include src
        ${CMAKE_CURRENT_SOURCE_DIR}/../Runner/src
        ${CMAKE_CURRENT_SOURCE_DIR}/../Runner/include)
    target_link_libraries(test_packing_gpu PRIVATE
        pipeline-core Eigen3::Eigen Vulkan::Vulkan LavaCake)
    target_compile_definitions(test_packing_gpu PRIVATE
        PACK_SHADER_DIR="${SHADER_OUT_DIR}")
    add_test(NAME test_packing_gpu COMMAND test_packing_gpu)
endif()
