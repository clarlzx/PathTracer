#include "VulkanSwapchain.h"

VkSurfaceFormatKHR chooseSwapSurfaceFormat(
    const std::vector<VkSurfaceFormatKHR> &availableFormats)
{
	for (const auto &availableFormat : availableFormats)
	{
		if (availableFormat.format == VK_FORMAT_B8G8R8A8_SRGB && availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
		{
			return availableFormat;
		}
	}

	throw std::runtime_error("failed to support SRGB colorspace");

	// TODO: Can also return other image formats but need to do manual gamma correction
	// return availableFormats[0];
}

VkPresentModeKHR chooseSwapPresentMode(
    const std::vector<VkPresentModeKHR> &availablePresentModes)
{
	for (const auto &availablePresentMode : availablePresentModes)
	{
		if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR)
		{
			return availablePresentMode;
		}
	}

	return VK_PRESENT_MODE_FIFO_KHR;
}

VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR &capabilities, GLFWwindow *window)
{
	if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max())
	{
		return capabilities.currentExtent;
	}
	else
	{
		int width, height;
		glfwGetFramebufferSize(window, &width, &height);

		VkExtent2D actualExtent{static_cast<uint32_t>(width), static_cast<uint32_t>(height)};

		actualExtent.width = std::clamp(actualExtent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
		actualExtent.height = std::clamp(actualExtent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);

		return actualExtent;
	}
}

/*
 * VulkanSwapchain functions
 */

void VulkanSwapchain::setContext(VkInstance instance, VulkanDevice *vulkanDevice, GLFWwindow *window)
{
	m_instance = instance;
	m_window = window;
	m_vulkanDevice = vulkanDevice;
}

void VulkanSwapchain::createSurface(VkInstance instance, GLFWwindow *window)
{
	if (glfwCreateWindowSurface(instance, window, nullptr, &m_surface) != VK_SUCCESS)
	{
		throw std::runtime_error("failed to create window surface!");
	}
}

void VulkanSwapchain::createSwapchain()
{
	assert(m_instance);
	assert(m_window);
	assert(m_vulkanDevice);
	assert(m_surface);

	// Store the current swapchain handle so we can use it later on to ease up recreation
	// VkSwapchainKHR oldSwapchain{m_swapchain};
	if (m_swapchain != VK_NULL_HANDLE) {
		for (auto imageView : m_imageViews)
		{
			vkDestroyImageView(m_vulkanDevice->logicalDevice, imageView, nullptr);
		}

		vkDestroySwapchainKHR(m_vulkanDevice->logicalDevice, m_swapchain, nullptr);
	}

	SurfaceSupportDetails swapChainSupport{querySurfaceSupport(m_vulkanDevice->physicalDevice)};

	VkSurfaceFormatKHR surfaceFormat{chooseSwapSurfaceFormat(swapChainSupport.formats)};
	VkPresentModeKHR presentMode{chooseSwapPresentMode(swapChainSupport.presentModes)};
	VkExtent2D extent{chooseSwapExtent(swapChainSupport.capabilities, m_window)};

	uint32_t imageCount{swapChainSupport.capabilities.minImageCount + 1};
	if (swapChainSupport.capabilities.maxImageCount > 0 && imageCount > swapChainSupport.capabilities.maxImageCount)
	{
		imageCount = swapChainSupport.capabilities.maxImageCount;
	}

	VkSwapchainCreateInfoKHR createInfo{};
	createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
	createInfo.surface = m_surface;

	createInfo.minImageCount = imageCount;
	createInfo.imageFormat = surfaceFormat.format;
	createInfo.imageColorSpace = surfaceFormat.colorSpace;
	createInfo.imageExtent = extent;
	createInfo.imageArrayLayers = 1;
	createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

	uint32_t queueFamilyIndices[]{m_vulkanDevice->queueFamilyIndices.graphics.value(),
	                              m_vulkanDevice->queueFamilyIndices.present.value()};

	if (queueFamilyIndices[0] != queueFamilyIndices[1])
	{
		createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
		createInfo.queueFamilyIndexCount = 2;
		createInfo.pQueueFamilyIndices = queueFamilyIndices;
	}
	else
	{
		// If same queue families, use exclusive mode since concurrent mode
		// needs at least 2 distinct queue families Requires ownership
		// transfer for exclusive mode, if used for >2 distinct queue
		// families
		createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
	}

	createInfo.preTransform = swapChainSupport.capabilities.currentTransform;
	createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
	createInfo.presentMode = presentMode;
	createInfo.clipped = VK_TRUE;

	// Setting oldSwapchain to the saved handle of previous swapchain aids in resource reuse
	// and makes sure that we can still present already acquired images
	// createInfo.oldSwapchain = oldSwapchain;

	if (vkCreateSwapchainKHR(m_vulkanDevice->logicalDevice, &createInfo, nullptr, &m_swapchain) != VK_SUCCESS)
	{
		throw std::runtime_error("failed to create swap chain!");
	}

	// If an existing swapchain is re-created, destroy the old swapchain.
	// This also cleans up all presentable images.
	/*if (oldSwapchain != VK_NULL_HANDLE)
	{
		for (auto imageView : m_imageViews)
		{
			vkDestroyImageView(m_vulkanDevice->logicalDevice, imageView, nullptr);
		}

		vkDestroySwapchainKHR(m_vulkanDevice->logicalDevice, oldSwapchain, nullptr);
	}*/

	// Get number of presentable images first, since we only specified minImageCount earlier on, there could be more presentable images
	vkGetSwapchainImagesKHR(m_vulkanDevice->logicalDevice, m_swapchain, &imageCount, nullptr);
	m_images.resize(imageCount);
	vkGetSwapchainImagesKHR(m_vulkanDevice->logicalDevice, m_swapchain, &imageCount, m_images.data());

	m_format = surfaceFormat.format;
	m_extent = extent;

	m_imageViews.resize(imageCount);
	for (uint32_t i{0}; i < imageCount; ++i)
	{
		VkImageViewCreateInfo viewInfo{};
		viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		viewInfo.image = m_images[i];
		viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
		viewInfo.format = m_format;
		viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		viewInfo.subresourceRange.baseMipLevel = 0;
		viewInfo.subresourceRange.levelCount = 1;
		viewInfo.subresourceRange.baseArrayLayer = 0;
		viewInfo.subresourceRange.layerCount = 1;

		if (vkCreateImageView(m_vulkanDevice->logicalDevice, &viewInfo, nullptr, &m_imageViews[i]) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to create image view!");
		}
	}
}

SurfaceSupportDetails VulkanSwapchain::querySurfaceSupport(VkPhysicalDevice physicalDevice)
{
	SurfaceSupportDetails details{};

	vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physicalDevice, m_surface, &details.capabilities);

	uint32_t formatCount{};
	vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, m_surface, &formatCount, nullptr);

	if (formatCount != 0)
	{
		details.formats.resize(formatCount);
		vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, m_surface, &formatCount, details.formats.data());
	}
	else
	{
		throw std::runtime_error("failed to find any supported image formats by surface");
	}

	uint32_t presentModeCount{};
	vkGetPhysicalDeviceSurfacePresentModesKHR(physicalDevice, m_surface, &presentModeCount, nullptr);

	if (presentModeCount != 0)
	{
		details.presentModes.resize(presentModeCount);
		vkGetPhysicalDeviceSurfacePresentModesKHR(physicalDevice, m_surface, &presentModeCount, details.presentModes.data());
	}
	else
	{
		throw std::runtime_error("failed to find any supported presentation modes by surface");
	}

	return details;
}

void VulkanSwapchain::cleanup()
{
	for (auto imageView : m_imageViews)
	{
		vkDestroyImageView(m_vulkanDevice->logicalDevice, imageView, nullptr);
	}

	vkDestroySwapchainKHR(m_vulkanDevice->logicalDevice, m_swapchain, nullptr);
	vkDestroySurfaceKHR(m_instance, m_surface, nullptr);
}
