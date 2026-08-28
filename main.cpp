#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include "mandala_raster_renderer.h"
#include <algorithm>
#include <filesystem>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <vector>
#include <string>

static constexpr int WIDTH = 1280;
static constexpr int HEIGHT = 720;
static constexpr int MAX_FRAMES = 2;

static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT severity,
    VkDebugUtilsMessageTypeFlagsEXT type,
    const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
    void* pUserData) {
    (void)type;
    (void)pUserData;
    if (severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT)
        fprintf(stderr, "[VULKAN] %s\n", pCallbackData->pMessage);
    return VK_FALSE;
}

static void destroyDebugMessenger(VkInstance instance, VkDebugUtilsMessengerEXT messenger) {
    if (messenger == VK_NULL_HANDLE) return;
    auto destroy = (PFN_vkDestroyDebugUtilsMessengerEXT)
        vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
    if (destroy) destroy(instance, messenger, nullptr);
}

static bool checkValidationLayerSupport(const std::vector<const char*>& layers) {
    uint32_t count;
    vkEnumerateInstanceLayerProperties(&count, nullptr);
    std::vector<VkLayerProperties> available(count);
    vkEnumerateInstanceLayerProperties(&count, available.data());
    for (auto* name : layers) {
        bool found = false;
        for (auto& l : available) {
            if (strcmp(name, l.layerName) == 0) { found = true; break; }
        }
        if (!found) return false;
    }
    return true;
}

static std::string frameDirectoryName(uint32_t frameIndex) {
    std::ostringstream name;
    name << "frame-" << std::setw(3) << std::setfill('0') << frameIndex;
    return name.str();
}

static bool writeMetricsReceipt(
    const std::string& path,
    const RenderConfig& config,
    uint32_t frameCount,
    const std::vector<RT4DCameraSample>& cameraSamples,
    const std::vector<RT4DFrameMetrics>& frameMetrics,
    const std::vector<double>& gpuTimes,
    uint64_t historyResetCount) {
    if (path.empty() || frameMetrics.size() != frameCount ||
        cameraSamples.size() != frameCount || gpuTimes.size() != frameCount)
        return false;
    const std::filesystem::path outputPath(path);
    if (outputPath.has_parent_path()) {
        std::error_code error;
        std::filesystem::create_directories(outputPath.parent_path(), error);
        if (error) return false;
    }

    std::vector<double> validConfidence;
    std::vector<double> validAcceptedRatio;
    std::vector<double> validResidual;
    std::vector<double> frameMeanMotion;
    std::vector<double> compositeDelta;
    const size_t warmupExcluded = gpuTimes.size() > 2 ? 2u : 0u;
    const std::vector<double> postWarmupTimes(
        gpuTimes.begin() + static_cast<std::ptrdiff_t>(warmupExcluded),
        gpuTimes.end());
    uint32_t historyValidFrames = 0;
    for (const RT4DFrameMetrics& metrics : frameMetrics) {
        frameMeanMotion.push_back(metrics.meanMotionPixels);
        if (metrics.frameIndex > 0)
            compositeDelta.push_back(metrics.compositeLumaMaeFromPrevious);
        if (metrics.historyValid) {
            ++historyValidFrames;
            validConfidence.push_back(metrics.meanReprojectionConfidence);
            validAcceptedRatio.push_back(metrics.acceptedHistoryRatio);
            validResidual.push_back(metrics.meanAcceptedLumaResidual);
        }
    }

    std::ofstream output(outputPath);
    if (!output) return false;
    output << std::setprecision(9)
           << "{\n"
           << "  \"schema\": \"rt4d-temporal-metrics/0.4\",\n"
           << "  \"cameraSequence\": \""
           << rt4dCameraSequenceName(config.cameraSequence) << "\",\n"
           << "  \"frameCount\": " << frameCount << ",\n"
           << "  \"cameraCutFrame\": ";
    if (config.cameraCutFrame == std::numeric_limits<uint32_t>::max())
        output << "null";
    else
        output << config.cameraCutFrame;
    output << ",\n"
           << "  \"historyResetCount\": " << historyResetCount << ",\n"
           << "  \"historyValidFrames\": " << historyValidFrames << ",\n"
           << "  \"metricBoundary\": "
              "\"observed diagnostics, not a perceptual-quality score\",\n"
           << "  \"summary\": {\n"
           << "    \"meanFrameMotionPixels\": " << rt4dMean(frameMeanMotion) << ",\n"
           << "    \"p95FrameMotionPixels\": "
           << rt4dPercentile(frameMeanMotion, 0.95) << ",\n"
           << "    \"meanReprojectionConfidenceHistoryValidFrames\": "
           << rt4dMean(validConfidence) << ",\n"
           << "    \"meanAcceptedHistoryRatioHistoryValidFrames\": "
           << rt4dMean(validAcceptedRatio) << ",\n"
           << "    \"meanAcceptedLumaResidualHistoryValidFrames\": "
           << rt4dMean(validResidual) << ",\n"
           << "    \"meanCompositeLumaMae\": " << rt4dMean(compositeDelta) << ",\n"
           << "    \"p95CompositeLumaMae\": "
           << rt4dPercentile(compositeDelta, 0.95) << "\n"
           << "  },\n"
           << "  \"gpuTimingMs\": {\n"
           << "    \"sampleCount\": " << gpuTimes.size() << ",\n"
           << "    \"min\": " << *std::min_element(gpuTimes.begin(), gpuTimes.end()) << ",\n"
           << "    \"p50\": " << rt4dPercentile(gpuTimes, 0.50) << ",\n"
           << "    \"p95\": " << rt4dPercentile(gpuTimes, 0.95) << ",\n"
           << "    \"p99\": " << rt4dPercentile(gpuTimes, 0.99) << ",\n"
           << "    \"max\": " << *std::max_element(gpuTimes.begin(), gpuTimes.end()) << ",\n"
           << "    \"samples\": [";
    for (size_t i = 0; i < gpuTimes.size(); ++i) {
        if (i) output << ", ";
        output << gpuTimes[i];
    }
    output << "]\n"
           << "  },\n"
           << "  \"gpuTimingPostWarmupMs\": {\n"
           << "    \"warmupFramesExcluded\": " << warmupExcluded << ",\n"
           << "    \"sampleCount\": " << postWarmupTimes.size() << ",\n"
           << "    \"p50\": " << rt4dPercentile(postWarmupTimes, 0.50) << ",\n"
           << "    \"p95\": " << rt4dPercentile(postWarmupTimes, 0.95) << ",\n"
           << "    \"p99\": " << rt4dPercentile(postWarmupTimes, 0.99) << ",\n"
           << "    \"max\": "
           << *std::max_element(postWarmupTimes.begin(), postWarmupTimes.end()) << "\n"
           << "  },\n"
           << "  \"frames\": [\n";
    for (size_t i = 0; i < frameMetrics.size(); ++i) {
        const RT4DFrameMetrics& metrics = frameMetrics[i];
        output << "    {\"frameIndex\": " << metrics.frameIndex
               << ", \"historyValid\": "
               << (metrics.historyValid ? "true" : "false")
               << ", \"historyResetReason\": \""
               << rt4dHistoryResetReasonName(metrics.resetReason)
               << "\", \"foregroundPixels\": " << metrics.foregroundPixels
               << ", \"meanMotionPixels\": " << metrics.meanMotionPixels
               << ", \"p95MotionPixels\": " << metrics.p95MotionPixels
               << ", \"maxMotionPixels\": " << metrics.maxMotionPixels
               << ", \"meanReprojectionConfidence\": "
               << metrics.meanReprojectionConfidence
               << ", \"acceptedHistoryRatio\": " << metrics.acceptedHistoryRatio
               << ", \"meanAcceptedLumaResidual\": "
               << metrics.meanAcceptedLumaResidual
               << ", \"compositeLumaMaeFromPrevious\": "
               << metrics.compositeLumaMaeFromPrevious << "}";
        output << (i + 1 == frameMetrics.size() ? "\n" : ",\n");
    }
    output << "  ],\n"
           << "  \"cameraSamples\": [\n";
    for (size_t i = 0; i < cameraSamples.size(); ++i) {
        const RT4DCameraSample& sample = cameraSamples[i];
        output << "    {\"frameIndex\": " << sample.frameIndex
               << ", \"segment\": " << sample.segment
               << ", \"cameraCut\": " << (sample.cameraCut ? "true" : "false")
               << ", \"eye\": [" << sample.pose.eye[0] << ", "
               << sample.pose.eye[1] << ", " << sample.pose.eye[2]
               << "], \"target\": [" << sample.pose.target[0] << ", "
               << sample.pose.target[1] << ", " << sample.pose.target[2] << "]}";
        output << (i + 1 == cameraSamples.size() ? "\n" : ",\n");
    }
    output << "  ]\n"
           << "}\n";
    return output.good();
}

static void printUsage(const char* executable) {
    fprintf(stderr,
            "Usage: %s [living-map|taco|battle|dragon|sentinel|recon] "
            "[--asset=PATH] [--capture=PATH] [--frames=1..600] "
            "[--missing-uv=reject|generate-planar-labeled] "
            "[--camera-sequence=static|orbit] [--camera-cut-frame=N] "
            "[--sequence-dir=PATH] [--observability-dir=PATH] "
            "[--metrics=PATH] "
            "[--debug=gpu-timer]\n"
            "\n"
            "Only gpu-timer is a supported debug capability. Motion, observability, "
            "and metrics are supported only in recon mode.\n",
            executable);
}

int main(int argc, char** argv) {
    RenderScene startScene = RenderScene::LIVING_MAP;
    std::string capturePath;
    std::string sequenceDirectory;
    std::string observabilityDirectory;
    std::string metricsPath;
    std::string assetPath = "armored-sentinel-v1.glb";
    GLTFMissingUvPolicy missingUvPolicy =
        GLTFMissingUvPolicy::RejectTexturedPrimitive;
    RT4DCameraSequenceMode cameraSequence = RT4DCameraSequenceMode::Static;
    uint32_t cameraCutFrame = std::numeric_limits<uint32_t>::max();
    uint32_t captureFrames = 1;
    struct DebugConfig {
        bool gpuTimer = false;
    } debug;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            printUsage(argv[0]);
            return 0;
        }
        else if (arg == "living-map") startScene = RenderScene::LIVING_MAP;
        else if (arg == "taco") startScene = RenderScene::TACO;
        else if (arg == "battle") startScene = RenderScene::BATTLE;
        else if (arg == "dragon") startScene = RenderScene::DRAGON_HATCH;
        else if (arg == "sentinel") startScene = RenderScene::SENTINEL;
        else if (arg == "recon") startScene = RenderScene::RECON;
        else if (arg.rfind("--asset=", 0) == 0) {
            assetPath = arg.substr(8);
            if (assetPath.empty()) {
                fprintf(stderr, "--asset requires a non-empty path\n");
                return 2;
            }
        }
        else if (arg.rfind("--missing-uv=", 0) == 0) {
            const std::string policy = arg.substr(13);
            if (policy == "reject") {
                missingUvPolicy = GLTFMissingUvPolicy::RejectTexturedPrimitive;
            } else if (policy == "generate-planar-labeled") {
                missingUvPolicy = GLTFMissingUvPolicy::GeneratePlanarLabeled;
            } else {
                fprintf(stderr, "Invalid --missing-uv policy; expected reject or "
                                "generate-planar-labeled\n");
                return 2;
            }
        }
        else if (arg.rfind("--camera-sequence=", 0) == 0) {
            const std::string sequence = arg.substr(18);
            if (sequence == "static")
                cameraSequence = RT4DCameraSequenceMode::Static;
            else if (sequence == "orbit")
                cameraSequence = RT4DCameraSequenceMode::DeterministicOrbit;
            else {
                fprintf(stderr, "Invalid --camera-sequence; expected static or orbit\n");
                return 2;
            }
        }
        else if (arg.rfind("--camera-cut-frame=", 0) == 0) {
            try {
                size_t parsed = 0;
                const std::string value = arg.substr(19);
                const unsigned long frame = std::stoul(value, &parsed);
                if (parsed != value.size() || frame > 599)
                    throw std::out_of_range("camera cut frame");
                cameraCutFrame = static_cast<uint32_t>(frame);
            } catch (...) {
                fprintf(stderr, "Invalid --camera-cut-frame value; expected 0..599\n");
                return 2;
            }
        }
        else if (arg.rfind("--sequence-dir=", 0) == 0) {
            sequenceDirectory = arg.substr(15);
            if (sequenceDirectory.empty()) {
                fprintf(stderr, "--sequence-dir requires a non-empty path\n");
                return 2;
            }
        }
        else if (arg.rfind("--observability-dir=", 0) == 0) {
            observabilityDirectory = arg.substr(20);
            if (observabilityDirectory.empty()) {
                fprintf(stderr, "--observability-dir requires a non-empty path\n");
                return 2;
            }
        }
        else if (arg.rfind("--metrics=", 0) == 0) {
            metricsPath = arg.substr(10);
            if (metricsPath.empty()) {
                fprintf(stderr, "--metrics requires a non-empty path\n");
                return 2;
            }
        }
        else if (arg.rfind("--capture=", 0) == 0) capturePath = arg.substr(10);
        else if (arg.rfind("--frames=", 0) == 0) {
            try {
                size_t parsed = 0;
                const std::string value = arg.substr(9);
                const unsigned long count = std::stoul(value, &parsed);
                if (parsed != value.size() || count < 1 || count > 600)
                    throw std::out_of_range("capture frame count");
                captureFrames = static_cast<uint32_t>(count);
            } catch (...) {
                fprintf(stderr, "Invalid --frames value; expected 1..600\n");
                return 2;
            }
        }
        else if (arg.rfind("--debug=", 0) == 0) {
            std::string sub = arg.substr(8);
            if (sub == "gpu-timer") debug.gpuTimer = true;
            else {
                fprintf(stderr, "Unsupported --debug capability: %s (available: gpu-timer)\n",
                        sub.c_str());
                return 2;
            }
        }
        else {
            fprintf(stderr, "Unknown argument: %s\n", arg.c_str());
            printUsage(argv[0]);
            return 2;
        }
    }

    const bool v04CaptureRequested = !sequenceDirectory.empty() ||
        !observabilityDirectory.empty() || !metricsPath.empty();
    if (captureFrames != 1 && capturePath.empty() && !v04CaptureRequested) {
        fprintf(stderr,
                "--frames requires --capture, --sequence-dir, --observability-dir, "
                "or --metrics\n");
        return 2;
    }
    if (v04CaptureRequested && startScene != RenderScene::RECON) {
        fprintf(stderr, "v0.4 observability and motion options require recon mode\n");
        return 2;
    }
    if (!metricsPath.empty() && sequenceDirectory.empty()) {
        fprintf(stderr,
                "--metrics requires --sequence-dir so composite deltas are observed\n");
        return 2;
    }
    if (cameraSequence != RT4DCameraSequenceMode::Static &&
        startScene != RenderScene::RECON) {
        fprintf(stderr, "camera motion sequences require recon mode\n");
        return 2;
    }
    if (cameraCutFrame != std::numeric_limits<uint32_t>::max()) {
        if (cameraSequence != RT4DCameraSequenceMode::DeterministicOrbit) {
            fprintf(stderr, "--camera-cut-frame requires --camera-sequence=orbit\n");
            return 2;
        }
        if (cameraCutFrame == 0 || cameraCutFrame >= captureFrames) {
            fprintf(stderr, "--camera-cut-frame must be within 1..frames-1\n");
            return 2;
        }
    }
    if (!metricsPath.empty()) debug.gpuTimer = true;

    if (!glfwInit()) {
        fprintf(stderr, "Failed to initialize GLFW\n");
        return 1;
    }
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    // Resize is intentionally disabled until swapchain, render-pass, pipeline,
    // and temporal-history recreation can be performed as one transaction.
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

    GLFWwindow* window = glfwCreateWindow(WIDTH, HEIGHT,
        "Mandala Rendering Software — Raster", nullptr, nullptr);
    if (!window) {
        fprintf(stderr, "Failed to create window\n");
        glfwTerminate();
        return 1;
    }

    uint32_t glfwExtCount = 0;
    const char** glfwExts = glfwGetRequiredInstanceExtensions(&glfwExtCount);
    std::vector<const char*> extensions(glfwExts, glfwExts + glfwExtCount);

    std::vector<const char*> layers;
    bool useValidation = false;
    if (checkValidationLayerSupport({"VK_LAYER_KHRONOS_validation"})) {
        layers.push_back("VK_LAYER_KHRONOS_validation");
        extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        useValidation = true;
    }

    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "Mandala Raster";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "MRS";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 6, 0);
    appInfo.apiVersion = VK_API_VERSION_1_2;

    VkInstanceCreateInfo instCI{};
    instCI.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instCI.pApplicationInfo = &appInfo;
    instCI.enabledExtensionCount = (uint32_t)extensions.size();
    instCI.ppEnabledExtensionNames = extensions.data();
    instCI.enabledLayerCount = (uint32_t)layers.size();
    instCI.ppEnabledLayerNames = layers.data();

    VkInstance instance;
    if (vkCreateInstance(&instCI, nullptr, &instance) != VK_SUCCESS) {
        fprintf(stderr, "Failed to create Vulkan instance\n");
        return 1;
    }

    VkDebugUtilsMessengerEXT debugMessenger = VK_NULL_HANDLE;
    if (useValidation) {
        auto func = (PFN_vkCreateDebugUtilsMessengerEXT)
            vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
        if (func) {
            VkDebugUtilsMessengerCreateInfoEXT dci{};
            dci.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
            dci.messageSeverity =
                VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
            dci.messageType =
                VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
            dci.pfnUserCallback = debugCallback;
            func(instance, &dci, nullptr, &debugMessenger);
        }
    }

    VkSurfaceKHR surface;
    if (glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS) {
        fprintf(stderr, "Failed to create surface\n");
        return 1;
    }

    uint32_t physCount = 0;
    vkEnumeratePhysicalDevices(instance, &physCount, nullptr);
    if (physCount == 0) {
        fprintf(stderr, "No GPU found\n");
        return 1;
    }
    std::vector<VkPhysicalDevice> physDevs(physCount);
    vkEnumeratePhysicalDevices(instance, &physCount, physDevs.data());

    VkPhysicalDevice phys = physDevs[0];
    VkPhysicalDeviceProperties physProps;
    vkGetPhysicalDeviceProperties(phys, &physProps);
    fprintf(stderr, "GPU: %s\n", physProps.deviceName);

    float queuePriority = 1.0f;
    VkDeviceQueueCreateInfo qci{};
    qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qci.queueFamilyIndex = 0;
    qci.queueCount = 1;
    qci.pQueuePriorities = &queuePriority;

    const char* deviceExts[] = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};
    VkDeviceCreateInfo devCI{};
    devCI.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    devCI.queueCreateInfoCount = 1;
    devCI.pQueueCreateInfos = &qci;
    devCI.enabledExtensionCount = 1;
    devCI.ppEnabledExtensionNames = deviceExts;

    VkDevice device;
    if (vkCreateDevice(phys, &devCI, nullptr, &device) != VK_SUCCESS) {
        fprintf(stderr, "Failed to create device\n");
        return 1;
    }

    RenderConfig cfg{};
    cfg.width = WIDTH;
    cfg.height = HEIGHT;
    cfg.fovDegrees = 60.0f;
    cfg.scene = startScene;
    cfg.assetPath = assetPath;
    cfg.missingUvPolicy = missingUvPolicy;
    cfg.cameraSequence = cameraSequence;
    cfg.cameraCutFrame = cameraCutFrame;

    MandalaRasterRenderer renderer;
    renderer.setDebugFlags(debug.gpuTimer);
    if (!renderer.init(instance, phys, device, surface, cfg)) {
        fprintf(stderr, "Failed to init renderer\n");
        renderer.shutdown();
        vkDestroySurfaceKHR(instance, surface, nullptr);
        vkDestroyDevice(device, nullptr);
        destroyDebugMessenger(instance, debugMessenger);
        vkDestroyInstance(instance, nullptr);
        glfwDestroyWindow(window);
        glfwTerminate();
        return 1;
    }
    renderer.setScene(startScene);

    // Bounded capture mode: render one or more frames, then write the last
    // presented frame. Multi-frame capture is required to exercise history.
    if (!capturePath.empty() || v04CaptureRequested) {
        fprintf(stderr,
                "[capture] rendering %u frame(s), sequence=%s, cut=",
                captureFrames, rt4dCameraSequenceName(cameraSequence));
        if (cameraCutFrame == std::numeric_limits<uint32_t>::max())
            fprintf(stderr, "none\n");
        else
            fprintf(stderr, "%u\n", cameraCutFrame);
        bool renderOk = true;
        if (!sequenceDirectory.empty()) {
            std::error_code directoryError;
            std::filesystem::create_directories(sequenceDirectory, directoryError);
            if (directoryError) {
                fprintf(stderr, "Failed to create sequence directory\n");
                renderOk = false;
            }
        }
        std::vector<uint8_t> previousComposite;
        std::vector<RT4DFrameMetrics> temporalMetrics;
        temporalMetrics.reserve(captureFrames);
        for (uint32_t frame = 0; renderOk && frame < captureFrames; ++frame) {
            if (!renderer.renderFrame(1.0f / 60.0f)) {
                renderOk = false;
                break;
            }
            std::vector<uint8_t> currentComposite;
            if (!sequenceDirectory.empty()) {
                const std::filesystem::path framePath =
                    std::filesystem::path(sequenceDirectory) /
                    (frameDirectoryName(frame) + ".png");
                if (!renderer.captureScreenshot(framePath.string(),
                                                &currentComposite)) {
                    renderOk = false;
                    break;
                }
            }

            if (!metricsPath.empty() || !observabilityDirectory.empty()) {
                const bool writeImages = !observabilityDirectory.empty() &&
                    (frame + 1 == captureFrames || frame == cameraCutFrame);
                std::string exportPath;
                if (writeImages) {
                    exportPath = (std::filesystem::path(observabilityDirectory) /
                                  frameDirectoryName(frame)).string();
                }
                RT4DFrameMetrics metrics{};
                if (!renderer.exportReconObservability(exportPath, frame,
                                                       writeImages, metrics)) {
                    renderOk = false;
                    break;
                }
                if (!previousComposite.empty() && !currentComposite.empty()) {
                    metrics.compositeLumaMaeFromPrevious =
                        rt4dCompositeLumaMae(previousComposite, currentComposite);
                }
                temporalMetrics.push_back(metrics);
            }
            if (!currentComposite.empty())
                previousComposite = std::move(currentComposite);
        }
        const bool waitOk = vkDeviceWaitIdle(device) == VK_SUCCESS;
        bool captureOk = renderOk && waitOk;
        if (captureOk && !capturePath.empty())
            captureOk = renderer.captureScreenshot(capturePath);
        if (captureOk && debug.gpuTimer)
            captureOk = renderer.finalizeGpuTimings();
        if (captureOk && !metricsPath.empty()) {
            captureOk = writeMetricsReceipt(
                metricsPath, cfg, captureFrames, renderer.cameraSamples(),
                temporalMetrics, renderer.gpuFrameTimesMs(),
                renderer.dlss45.historyResetCount());
        }
        fprintf(stderr, "[capture] %s\n", captureOk ? "OK" : "FAILED");
        renderer.shutdown();
        vkDestroySurfaceKHR(instance, surface, nullptr);
        vkDestroyDevice(device, nullptr);
        destroyDebugMessenger(instance, debugMessenger);
        vkDestroyInstance(instance, nullptr);
        glfwDestroyWindow(window);
        glfwTerminate();
        return captureOk ? 0 : 1;
    }

    fprintf(stderr, "Rendering. Keys: 1=LivingMap 2=Taco 3=Battle 4=Dragon 5=Sentinel 6=RT4D Reconstruction Stack  P=screenshot  ESC=quit\n");

    double lastTime = glfwGetTime();
    uint64_t frameCount = 0;
    double fpsAccum = 0.0;
    double lastFpsPrint = lastTime;

    bool runtimeOk = true;
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        double now = glfwGetTime();
        float dt = (float)(now - lastTime);
        lastTime = now;

        if (glfwGetKey(window, GLFW_KEY_1) == GLFW_PRESS)
            renderer.setScene(RenderScene::LIVING_MAP);
        if (glfwGetKey(window, GLFW_KEY_2) == GLFW_PRESS)
            renderer.setScene(RenderScene::TACO);
        if (glfwGetKey(window, GLFW_KEY_3) == GLFW_PRESS)
            renderer.setScene(RenderScene::BATTLE);
        if (glfwGetKey(window, GLFW_KEY_4) == GLFW_PRESS)
            renderer.setScene(RenderScene::DRAGON_HATCH);
        if (glfwGetKey(window, GLFW_KEY_5) == GLFW_PRESS)
            renderer.setScene(RenderScene::SENTINEL);
        if (glfwGetKey(window, GLFW_KEY_6) == GLFW_PRESS)
            renderer.setScene(RenderScene::RECON);
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
            glfwSetWindowShouldClose(window, GLFW_TRUE);
        if (glfwGetKey(window, GLFW_KEY_P) == GLFW_PRESS) {
            static int shotNum = 0;
            char path[64];
            snprintf(path, sizeof(path), "screenshot_%04d.png", shotNum++);
            fprintf(stderr, "[capture] saving %s ... ", path);
            if (renderer.captureScreenshot(path))
                fprintf(stderr, "OK\n");
            else
                fprintf(stderr, "FAILED\n");
        }

        if (!renderer.renderFrame(dt)) {
            runtimeOk = false;
            break;
        }

        // FPS / frame-time benchmark
        frameCount++;
        double frameMs = dt * 1000.0;
        fpsAccum += frameMs;
        if (now - lastFpsPrint >= 0.5) {
            double avgMs = fpsAccum / frameCount;
            double fps = 1000.0 / avgMs;
            fprintf(stderr, "\r[bench] %llu frames | %.1f ms/frame (%.1f FPS)     ",
                    (unsigned long long)frameCount, avgMs, fps);
            frameCount = 0;
            fpsAccum = 0.0;
            lastFpsPrint = now;
        }
    }

    vkDeviceWaitIdle(device);

    // Final benchmark summary
    fprintf(stderr, "\n[bench] Done. Total frames rendered: %llu\n",
            (unsigned long long)frameCount);

    renderer.shutdown();
    vkDestroySurfaceKHR(instance, surface, nullptr);

    vkDestroyDevice(device, nullptr);
    destroyDebugMessenger(instance, debugMessenger);
    vkDestroyInstance(instance, nullptr);
    glfwDestroyWindow(window);
    glfwTerminate();
    return runtimeOk ? 0 : 1;
}
