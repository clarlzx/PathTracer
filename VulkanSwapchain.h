#pragma once

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <algorithm>
#include <cassert>
#include <stdexcept>
#include <vector>

#include "VulkanDevice.h"

struct SurfaceSupportDetails
{
	VkSurfaceCapabilitiesKHR capabilities{};
	std::vector<VkSurfaceFormatKHR> formats{};
	std::vector<VkPresentModeKHR> presentModes{};
};

class VulkanSwapchain
{
private:
	VkInstance m_instance{};
	VulkanDevice *m_vulkanDevice{};
	GLFWwindow *m_window{};

public:
	VkSurfaceKHR m_surface{};
	VkFormat m_format{};
	VkExtent2D m_extent{};
	VkSwapchainKHR m_swapchain{VK_NULL_HANDLE};
	std::vector<VkImage> m_images{};
	std::vector<VkImageView> m_imageViews{};
	uint32_t m_imageCount{};

	void createSurface(VkInstance instance, GLFWwindow *window);

	// Set context before creating swapchain for the first time
	void setContext(VkInstance instance, VulkanDevice *vulkanDevice, GLFWwindow *window);

	void createSwapchain();

	SurfaceSupportDetails querySurfaceSupport(VkPhysicalDevice physicalDevice);

	void cleanup();
};