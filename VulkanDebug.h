#pragma once

#include <iostream>
#include <vector>
#include <vulkan/vulkan.h>

namespace vk_debug
{

// Default debug callback
VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
    VkDebugUtilsMessageTypeFlagsEXT messageType,
    const VkDebugUtilsMessengerCallbackDataEXT *pCallbackData,
    void *pUserData);

void populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT &createInfo);

void setupDebugMessenger(VkInstance instance);

void freeDebugMessenger(VkInstance instance);

} // namespace vk_debug