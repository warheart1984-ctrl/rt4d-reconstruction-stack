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
    ${SRC_DIR}/vertex_buffer_manager.cpp
    ${SRC_DIR}/mandala_raster_renderer.cpp
    ${SRC_DIR}/scene_data.cpp
    ${SRC_DIR}/main.cpp
    ${SRC_DIR}/texture_manager.cpp
    ${SRC_DIR}/gltf_mesh.cpp
    ${SRC_DIR}/dlss45/reprojection_pipeline.cpp
    ${SRC_DIR}/dlss45/denoiser_pipeline.cpp
    ${SRC_DIR}/dlss45/sr_pipeline.cpp
    ${SRC_DIR}/dlss45/tone_map_pipeline.cpp
    ${SRC_DIR}/dlss45/ml_weights_loader.cpp
    ${SRC_DIR}/dlss45/dlss45_recon.cpp
)

add_executable(mandala_rasterize ${SOURCES})

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
