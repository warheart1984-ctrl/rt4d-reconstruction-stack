#pragma once

#include <vulkan/vulkan.h>
#include <vector>
#include <cstdint>

struct SwapchainSupport {
    VkSurfaceCapabilitiesKHR caps;
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR> presentModes;
};

class SwapchainManager {
public:
    bool init(VkPhysicalDevice phys, VkDevice device, VkSurfaceKHR surface,
              uint32_t width, uint32_t height);
    void shutdown(VkDevice device);

    VkResult acquireNextImage(VkDevice device, VkSemaphore signalSemaphore,
                              uint32_t& imageIndex);
    VkResult present(VkQueue queue, uint32_t imageIndex, VkSemaphore waitSemaphore);

    VkFormat colorFormat() const { return colorFormat_; }
    VkExtent2D extent() const { return extent_; }
    uint32_t imageCount() const { return (uint32_t)images_.size(); }
    VkImageView imageView(uint32_t i) const { return imageViews_[i]; }
    VkImage image(uint32_t i) const { return images_[i]; }

private:
    void querySupport(VkPhysicalDevice phys, VkSurfaceKHR surface);
    bool chooseFormat();
    bool choosePresentMode();
    bool createSwapchain(VkDevice device, VkSurfaceKHR surface,
                          uint32_t w, uint32_t h);
    void destroyInternal(VkDevice device);

    VkPhysicalDevice phys_ = VK_NULL_HANDLE;
    VkSwapchainKHR swapchain_ = VK_NULL_HANDLE;
    VkSurfaceFormatKHR chosenFormat_{};
    VkPresentModeKHR chosenMode_ = VK_PRESENT_MODE_FIFO_KHR;
    VkExtent2D extent_{};
    VkFormat colorFormat_ = VK_FORMAT_B8G8R8A8_UNORM;

    std::vector<VkImage> images_;
    std::vector<VkImageView> imageViews_;

    SwapchainSupport support_{};
};
