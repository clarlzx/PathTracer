#pragma once

#include <cassert>
#include <iostream>
#include <optional>
#include <set>
#include <stdexcept>
#include <vector>
#include <vulkan/vulkan.h>

struct VulkanDevice
{
	// Physical device representation
	VkPhysicalDevice physicalDevice{};
	// Logical device representation (application's interface of physical device)
	VkDevice logicalDevice{};
	// Properties of physical device including limits that the application can check against
	VkPhysicalDeviceProperties properties{};
	// Memory types and heaps of the physical device
	VkPhysicalDeviceMemoryProperties memProperties{};
	// Max usable samples
	VkSampleCountFlagBits msaaSamples{};
	// Supported depth format for physical device
	VkFormat depthFormat{};
	// Default command pool for the graphics queue family index
	VkCommandPool commandPool{VK_NULL_HANDLE};
	// Command buffers will be destroyed when commandPool is destroyed
	std::vector<VkCommandBuffer> commandBuffers{}; 

	struct
	{
		// Stores index of queue family
		// Index is possible to be the different/same queue family for different kinds of operations
		std::optional<uint32_t> graphics;
		std::optional<uint32_t> present;
		std::optional<uint32_t> compute;
		std::optional<uint32_t> transfer;

		bool isComplete()
		{
			// TODO: Can change to prefer physical device that supports drawing and
			// presentation in the same queue for improved performance.
			return graphics.has_value() && present.has_value() && compute.has_value() && transfer.has_value();
		}

		void reset()
		{
			graphics.reset();
			present.reset();
			compute.reset();
			transfer.reset();
		}
	} queueFamilyIndices;

	VkPhysicalDevice pickPhysicalDevice(VkInstance instance, VkSurfaceKHR surface, const std::vector<const char *> &enabledExtensions, VkPhysicalDeviceFeatures enabledFeatures);

	VkDevice createLogicalDevice(bool enableValidationLayers, const std::vector<const char *> &validationLayers, const std::vector<const char *> &enabledExtensions, VkPhysicalDeviceFeatures enabledFeatures, void *pNextChain);

	bool checkDeviceExtensionSupport(const std::vector<const char *> &enabledExtensions);

	bool checkDeviceFeaturesSupport(VkPhysicalDeviceFeatures enabledFeatures);

	void queryQueueFamilyIndices(VkSurfaceKHR surface);

	VkSampleCountFlagBits getMaxUsableSampleCount();

	uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties);

	VkFormat findSupportedFormat(const std::vector<VkFormat> &candidates, VkImageTiling tiling, VkFormatFeatureFlags features);

	VkFormat findDepthFormat();

	VkCommandBuffer beginSingleTimeCommands();
	void endSingleTimeCommands(VkCommandBuffer commandBuffer, VkQueue queue);

	void createCommandPool();
	void createCommandBuffers(const uint32_t maxFramesInFlight);

	void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer &buffer, VkDeviceMemory &bufferMemory);

	void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size, VkQueue queue);

	void cleanup();
};