#pragma once

#include <fstream>
#include <stdexcept>
#include <vulkan/vulkan.h>

#include "VulkanDevice.h"

namespace VulkanUtils
{
inline constexpr std::uint32_t width{ 800 };
inline constexpr std::uint32_t height{ 600 };

inline const int maxFramesInFlight{ 2 };
inline const int maxRTFrames{ 1000 };

std::vector<char> readFile(const std::string& filename);

VkShaderModule createShaderModule(VkDevice logicalDevice, const std::vector<char>& code);

VkDeviceAddress getBufferDeviceAddress(VkDevice logicalDevice, VkBuffer buffer);
} // namespace VulkanUtils