#include "swapchain_manager.h"
#include <algorithm>
#include <limits>

void SwapchainManager::querySupport(VkPhysicalDevice phys, VkSurfaceKHR surface) {
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(phys, surface, &support_.caps);

    uint32_t fmtCount;
    vkGetPhysicalDeviceSurfaceFormatsKHR(phys, surface, &fmtCount, nullptr);
    support_.formats.resize(fmtCount);
    vkGetPhysicalDeviceSurfaceFormatsKHR(phys, surface, &fmtCount, support_.formats.data());

    uint32_t pmCount;
    vkGetPhysicalDeviceSurfacePresentModesKHR(phys, surface, &pmCount, nullptr);
    support_.presentModes.resize(pmCount);
    vkGetPhysicalDeviceSurfacePresentModesKHR(phys, surface, &pmCount, support_.presentModes.data());
}

bool SwapchainManager::chooseFormat() {
    for (auto& f : support_.formats) {
        if (f.format == VK_FORMAT_B8G8R8A8_UNORM &&
            f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            chosenFormat_ = f;
            colorFormat_ = VK_FORMAT_B8G8R8A8_UNORM;
            return true;
        }
    }
    if (!support_.formats.empty()) {
        chosenFormat_ = support_.formats[0];
        colorFormat_ = chosenFormat_.format;
        return true;
    }
    return false;
}

bool SwapchainManager::choosePresentMode() {
    for (auto m : support_.presentModes) {
        if (m == VK_PRESENT_MODE_MAILBOX_KHR) {
            chosenMode_ = m;
            return true;
        }
    }
    for (auto m : support_.presentModes) {
        if (m == VK_PRESENT_MODE_FIFO_RELAXED_KHR) {
            chosenMode_ = m;
            return true;
        }
    }
    chosenMode_ = VK_PRESENT_MODE_FIFO_KHR;
    return true;
}

bool SwapchainManager::createSwapchain(VkDevice device, VkSurfaceKHR surface,
                                         uint32_t w, uint32_t h) {
    VkSurfaceCapabilitiesKHR& caps = support_.caps;
    extent_.width = std::clamp(w, caps.minImageExtent.width, caps.maxImageExtent.width);
    extent_.height = std::clamp(h, caps.minImageExtent.height, caps.maxImageExtent.height);

    uint32_t imageCount = caps.minImageCount + 1;
    if (caps.maxImageCount > 0 && imageCount > caps.maxImageCount)
        imageCount = caps.maxImageCount;

    VkSwapchainCreateInfoKHR ci{};
    ci.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    ci.surface = surface;
    ci.minImageCount = imageCount;
    ci.imageFormat = chosenFormat_.format;
    ci.imageColorSpace = chosenFormat_.colorSpace;
    ci.imageExtent = extent_;
    ci.imageArrayLayers = 1;
    ci.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;

    ci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    ci.preTransform = caps.currentTransform;
    ci.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    ci.presentMode = chosenMode_;
    ci.clipped = VK_TRUE;
    ci.oldSwapchain = VK_NULL_HANDLE;

    if (vkCreateSwapchainKHR(device, &ci, nullptr, &swapchain_) != VK_SUCCESS)
        return false;

    uint32_t count;
    vkGetSwapchainImagesKHR(device, swapchain_, &count, nullptr);
    images_.resize(count);
    vkGetSwapchainImagesKHR(device, swapchain_, &count, images_.data());

    imageViews_.resize(count);
    for (uint32_t i = 0; i < count; i++) {
        VkImageViewCreateInfo vi{};
        vi.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        vi.image = images_[i];
        vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vi.format = chosenFormat_.format;
        vi.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        vi.subresourceRange.levelCount = 1;
        vi.subresourceRange.layerCount = 1;
        vkCreateImageView(device, &vi, nullptr, &imageViews_[i]);
    }

    return true;
}

void SwapchainManager::destroyInternal(VkDevice device) {
    for (auto v : imageViews_) if (v) vkDestroyImageView(device, v, nullptr);
    imageViews_.clear();
    images_.clear();
    if (swapchain_) {
        vkDestroySwapchainKHR(device, swapchain_, nullptr);
        swapchain_ = VK_NULL_HANDLE;
    }
}

bool SwapchainManager::init(VkPhysicalDevice phys, VkDevice device,
                              VkSurfaceKHR surface, uint32_t w, uint32_t h) {
    phys_ = phys;
    querySupport(phys, surface);
    if (support_.formats.empty() || support_.presentModes.empty())
        return false;
    if (!chooseFormat() || !choosePresentMode())
        return false;
    return createSwapchain(device, surface, w, h);
}

void SwapchainManager::shutdown(VkDevice device) {
    destroyInternal(device);
}

bool SwapchainManager::recreate(VkDevice device, uint32_t w, uint32_t h) {
    destroyInternal(device);
    return createSwapchain(device, VK_NULL_HANDLE, w, h);
}

uint32_t SwapchainManager::acquireNextImage(VkDevice device, VkSemaphore signalSemaphore) {
    uint32_t index;
    vkAcquireNextImageKHR(device, swapchain_, UINT64_MAX,
                           signalSemaphore, VK_NULL_HANDLE, &index);
    return index;
}

VkResult SwapchainManager::present(VkQueue queue, uint32_t imageIndex,
                                     VkSemaphore waitSemaphore) {
    VkPresentInfoKHR pi{};
    pi.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    pi.waitSemaphoreCount = 1;
    pi.pWaitSemaphores = &waitSemaphore;
    pi.swapchainCount = 1;
    pi.pSwapchains = &swapchain_;
    pi.pImageIndices = &imageIndex;
    return vkQueuePresentKHR(queue, &pi);
}
