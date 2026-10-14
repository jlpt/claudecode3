#
# Hybrid ray tracing test harness.
#
# Builds RT64's HybridRaytracer together with plume and runs it against a synthetic scene on any Vulkan device with ray
# tracing pipeline support (Mesa's lavapipe works, so no GPU is required). Only the files needed by the pass are built.
#
#   cmake -S tools/hybrid_rt_test -B build-hybrid-rt-test -G Ninja -DRT64_DIR=<path to the patched rt64 checkout>
#   cmake --build build-hybrid-rt-test
#   ./build-hybrid-rt-test/hybrid_rt_test output_directory
#

cmake_minimum_required(VERSION 3.20)
project(hybrid_rt_test LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)

set(RT64_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../BanjoRecomp/lib/rt64" CACHE PATH "Path to the RT64 source tree with the hybrid ray tracing patch applied")
if (NOT EXISTS "${RT64_DIR}/src/render/rt64_hybrid_raytracing.cpp")
    message(FATAL_ERROR "RT64_DIR (${RT64_DIR}) doesn't contain the hybrid ray tracing patch. Run scripts/setup.sh first or set RT64_DIR.")
endif()

if (NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(FATAL_ERROR "The test harness only supports Linux. It runs on any Vulkan driver with ray tracing support, including lavapipe.")
endif()

if (CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|amd64|AMD64")
    set(DXC "LD_LIBRARY_PATH=${RT64_DIR}/src/contrib/dxc/lib/x64" "${RT64_DIR}/src/contrib/dxc/bin/x64/dxc-linux")
else()
    set(DXC "LD_LIBRARY_PATH=${RT64_DIR}/src/contrib/dxc/lib/arm64" "${RT64_DIR}/src/contrib/dxc/bin/arm64/dxc-linux")
endif()

add_subdirectory("${RT64_DIR}/src/tools/file_to_c" "${CMAKE_BINARY_DIR}/file_to_c")
add_subdirectory("${RT64_DIR}/src/contrib/plume" "${CMAKE_BINARY_DIR}/plume")

add_executable(hybrid_rt_test
    "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp"
    "${RT64_DIR}/src/render/rt64_hybrid_raytracing.cpp"
)

target_compile_definitions(hybrid_rt_test PRIVATE HLSL_CPU RT64_HYBRID_RT_ENABLED=1)
target_include_directories(hybrid_rt_test PRIVATE
    "${RT64_DIR}/src"
    "${RT64_DIR}/src/contrib"
    "${RT64_DIR}/src/contrib/plume"
    "${RT64_DIR}/src/contrib/hlslpp/include"
    "${CMAKE_BINARY_DIR}/src"
)

# Plume's Vulkan backend supports X11 windows on Linux even though the harness renders offscreen.
find_package(X11 REQUIRED)
target_link_libraries(hybrid_rt_test PRIVATE plume ${X11_LIBRARIES} ${CMAKE_DL_LIBS})

# Compiles a shader to SPIR-V and embeds it as a C array named <Name>BlobSPIRV, matching RT64's conventions.
function(add_spirv_shader SOURCE OUTPUT_NAME)
    set(OUTPUT_BASE "${CMAKE_BINARY_DIR}/src/shaders/${OUTPUT_NAME}")
    cmake_path(GET OUTPUT_BASE STEM BLOB_NAME)
    file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/src/shaders")
    add_custom_command(OUTPUT "${OUTPUT_BASE}.spv"
        COMMAND ${DXC} -spirv -fspv-target-env=vulkan1.0 -fvk-use-dx-layout "-I${RT64_DIR}/src" ${ARGN} "${SOURCE}" -Fo "${OUTPUT_BASE}.spv"
        DEPENDS "${SOURCE}")
    add_custom_command(OUTPUT "${OUTPUT_BASE}.spirv.c" "${OUTPUT_BASE}.spirv.h"
        COMMAND file_to_c "${OUTPUT_BASE}.spv" "${BLOB_NAME}BlobSPIRV" "${OUTPUT_BASE}.spirv.c" "${OUTPUT_BASE}.spirv.h"
        DEPENDS "${OUTPUT_BASE}.spv" file_to_c)
    target_sources(hybrid_rt_test PRIVATE "${OUTPUT_BASE}.spirv.c")
endfunction()

set(RT64_SHADERS "${RT64_DIR}/src/shaders")
set(RT_OPTS -T lib_6_3 -D RT_SHADER -fspv-target-env=vulkan1.1spirv1.4 -fspv-extension=SPV_KHR_ray_tracing -fspv-extension=SPV_EXT_descriptor_indexing)
add_spirv_shader("${RT64_SHADERS}/FullScreenVS.hlsl" "FullScreenVS.hlsl" -E VSMain -T vs_6_3 -fvk-invert-y)
add_spirv_shader("${RT64_SHADERS}/HybridRtWorldCS.hlsl" "HybridRtWorldCS.hlsl" -E CSMain -T cs_6_3)
add_spirv_shader("${RT64_SHADERS}/HybridRtDepthCopyCS.hlsl" "HybridRtDepthCopyCS.hlsl" -E CSMain -T cs_6_3)
add_spirv_shader("${RT64_SHADERS}/HybridRtDepthCopyCS.hlsl" "HybridRtDepthCopyCSMS.hlsl" -E CSMain -T cs_6_3 -D MULTISAMPLING)
add_spirv_shader("${RT64_SHADERS}/HybridRtTraceRT.hlsl" "HybridRtTraceRT.hlsl" ${RT_OPTS})
add_spirv_shader("${RT64_SHADERS}/HybridRtDenoiseCS.hlsl" "HybridRtDenoiseCS.hlsl" -E CSMain -T cs_6_3)
add_spirv_shader("${RT64_SHADERS}/HybridRtStatsCS.hlsl" "HybridRtStatsCS.hlsl" -E CSMain -T cs_6_3)
add_spirv_shader("${RT64_SHADERS}/HybridRtCompositePS.hlsl" "HybridRtCompositePS.hlsl" -E PSMain -T ps_6_3)
add_spirv_shader("${CMAKE_CURRENT_SOURCE_DIR}/shaders/HarnessRasterVS.hlsl" "HarnessRasterVS.hlsl" -E VSMain -T vs_6_3 -fvk-invert-y)
add_spirv_shader("${CMAKE_CURRENT_SOURCE_DIR}/shaders/HarnessRasterPS.hlsl" "HarnessRasterPS.hlsl" -E PSMain -T ps_6_3)
add_spirv_shader("${CMAKE_CURRENT_SOURCE_DIR}/shaders/HarnessReadbackCS.hlsl" "HarnessReadbackCS.hlsl" -E CSMain -T cs_6_3)
