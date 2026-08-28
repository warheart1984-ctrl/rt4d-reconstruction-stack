cmake_minimum_required(VERSION 3.20)
project(mandala_rasterize LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Vulkan REQUIRED)
find_package(glfw3 3.3 QUIET)

if(NOT glfw3_FOUND)
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_check_modules(GLFW IMPORTED_TARGET glfw3)
    endif()
endif()

set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR})

set(SOURCES
    ${SRC_DIR}/raster_pipeline.cpp
    ${SRC_DIR}/swapchain_manager.cpp
    ${SRC_DIR}/mandala_raster_renderer.cpp
    ${SRC_DIR}/scene_data.cpp
    ${SRC_DIR}/main.cpp
    ${SRC_DIR}/gltf_mesh.cpp
    ${SRC_DIR}/dlss45/reprojection_pipeline.cpp
    ${SRC_DIR}/dlss45/denoiser_pipeline.cpp
    ${SRC_DIR}/dlss45/sr_pipeline.cpp
    ${SRC_DIR}/dlss45/tone_map_pipeline.cpp
    ${SRC_DIR}/dlss45/dlss45_recon.cpp
)

add_executable(mandala_rasterize ${SOURCES})

if(MSVC)
    target_compile_options(mandala_rasterize PRIVATE /W4 /WX)
else()
    target_compile_options(mandala_rasterize PRIVATE -Wall -Wextra -Wpedantic -Werror)
endif()

target_include_directories(mandala_rasterize PRIVATE ${SRC_DIR} ${SRC_DIR}/third_party/cgltf)
target_link_libraries(mandala_rasterize PRIVATE Vulkan::Vulkan)

if(glfw3_FOUND)
    target_link_libraries(mandala_rasterize PRIVATE glfw)
elseif(TARGET PkgConfig::GLFW)
    target_link_libraries(mandala_rasterize PRIVATE PkgConfig::GLFW)
else()
    message(STATUS "GLFW not found — main.cpp will not compile (use -DGLFW_FOUND=OFF to skip)")
    set_target_properties(mandala_rasterize PROPERTIES EXCLUDE_FROM_ALL TRUE)
endif()

if(UNIX AND NOT APPLE)
    target_link_libraries(mandala_rasterize PRIVATE dl pthread)
endif()

add_custom_target(compile_shaders
    COMMAND bash ${SRC_DIR}/compile_shaders.sh
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${SRC_DIR}/spirv_rasterize
        ${CMAKE_CURRENT_BINARY_DIR}/spirv_rasterize
    WORKING_DIRECTORY ${SRC_DIR}
    COMMENT "Compiling rasterization shaders to SPIR-V"
)

add_dependencies(mandala_rasterize compile_shaders)

include(CTest)
if(BUILD_TESTING)
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    add_executable(rt4d_gltf_mesh_tests
        ${SRC_DIR}/tests/gltf_mesh_test.cpp
        ${SRC_DIR}/gltf_mesh.cpp
    )
    target_include_directories(rt4d_gltf_mesh_tests PRIVATE
        ${SRC_DIR}
        ${SRC_DIR}/third_party/cgltf
    )
    if(MSVC)
        target_compile_options(rt4d_gltf_mesh_tests PRIVATE /W4 /WX)
    else()
        target_compile_options(rt4d_gltf_mesh_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)
    endif()
    add_test(
        NAME rt4d_gltf_mesh_fixture
        COMMAND rt4d_gltf_mesh_tests
            ${SRC_DIR}/armored-sentinel-v1.glb
            ${SRC_DIR}/fixtures/armored-sentinel-textured-v0.3.glb
            ${SRC_DIR}/fixtures/armored-sentinel-textured-missing-uv-v0.3.glb
    )
    add_test(
        NAME rt4d_material_fixture_reproducibility
        COMMAND ${Python3_EXECUTABLE}
            ${SRC_DIR}/tools/generate_material_fixtures.py --check
    )
endif()
