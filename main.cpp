#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include "mandala_raster_renderer.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

static void printUsage(const char* executable) {
    fprintf(stderr,
            "Usage: %s [living-map|taco|battle|dragon|sentinel|recon] "
            "[--capture=PATH] [--frames=1..600] [--debug=gpu-timer]\n"
            "\n"
            "Only gpu-timer is a supported debug capability. Multi-frame counts "
            "require --capture.\n",
            executable);
}

int main(int argc, char** argv) {
    RenderScene startScene = RenderScene::LIVING_MAP;
    std::string capturePath;
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

    if (captureFrames != 1 && capturePath.empty()) {
        fprintf(stderr, "--frames requires --capture=PATH\n");
        return 2;
    }

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

    MandalaRasterRenderer renderer;
    renderer.setDebugFlags(debug.gpuTimer);
    if (!renderer.init(instance, phys, device, surface, cfg)) {
        fprintf(stderr, "Failed to init renderer\n");
        return 1;
    }
    renderer.setScene(startScene);

    // Bounded capture mode: render one or more frames, then write the last
    // presented frame. Multi-frame capture is required to exercise history.
    if (!capturePath.empty()) {
        fprintf(stderr, "[capture] rendering %u frame(s) to %s ... ",
                captureFrames, capturePath.c_str());
        bool renderOk = true;
        for (uint32_t frame = 0; frame < captureFrames; ++frame) {
            if (!renderer.renderFrame(1.0f / 60.0f)) {
                renderOk = false;
                break;
            }
        }
        vkDeviceWaitIdle(device);
        const bool captureOk = renderOk && renderer.captureScreenshot(capturePath);
        if (captureOk)
            fprintf(stderr, "OK\n");
        else
            fprintf(stderr, "FAILED\n");
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
