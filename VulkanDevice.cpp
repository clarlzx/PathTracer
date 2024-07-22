#include "VulkanDevice.h"

void VulkanDevice::pickPhysicalDevice(VkInstance instance, VkSurfaceKHR surface, const std::vector<const char *> &enabledExtensions, VkPhysicalDeviceFeatures enabledFeatures)
{
	uint32_t deviceCount{};
	vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);

	if (deviceCount == 0)
	{
		throw std::runtime_error("failed to find GPUs with Vulkan support!");
	}

	std::vector<VkPhysicalDevice> devices(deviceCount);
	vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

	for (const auto &device : devices)
	{
		physicalDevice = device;

		queryQueueFamilyIndices(surface);
		if (!queueFamilyIndices.isComplete())
			continue;

		if (!checkDeviceExtensionSupport(enabledExtensions))
			continue;

		if (!checkDeviceFeaturesSupport(enabledFeatures))
			continue;

		// Else device is adequate

		vkGetPhysicalDeviceProperties(physicalDevice, &properties);

		vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);
		msaaSamples = getMaxUsableSampleCount();
		depthFormat = findDepthFormat();
		// std::cout << properties.deviceName << '\n';

		// Favor a dedicated graphics card if available
		if (properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
			break;
		}
	}

	if (!msaaSamples)
	{
		throw std::runtime_error("MSAA: failed to find GPUs with Vulkan support!");
	}
}

void VulkanDevice::createLogicalDevice(bool enableValidationLayers, const std::vector<const char *> &validationLayers, const std::vector<const char *> &enabledExtensions, VkPhysicalDeviceFeatures enabledFeatures, void *pNextChain)
{
	assert(physicalDevice);

	std::vector<VkDeviceQueueCreateInfo> queueCreateInfos{};
	// Set is used such that duplicate queue family index are removed
	std::set<uint32_t> uniqueQueueFamilies{queueFamilyIndices.graphics.value(),
	                                       queueFamilyIndices.present.value()};

	float queuePriority{1.0f};
	for (uint32_t queueFamily : uniqueQueueFamilies)
	{
		VkDeviceQueueCreateInfo queueCreateInfo{};
		queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
		queueCreateInfo.queueFamilyIndex = queueFamily;
		queueCreateInfo.queueCount = 1;
		queueCreateInfo.pQueuePriorities = &queuePriority;
		queueCreateInfos.push_back(queueCreateInfo);
	}

	VkDeviceCreateInfo createInfo{};
	createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;

	createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
	createInfo.pQueueCreateInfos = queueCreateInfos.data();

	createInfo.pEnabledFeatures = &enabledFeatures;

	createInfo.enabledExtensionCount = static_cast<uint32_t>(enabledExtensions.size());
	createInfo.ppEnabledExtensionNames = enabledExtensions.data();

	if (enableValidationLayers)
	{
		createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
		createInfo.ppEnabledLayerNames = validationLayers.data();
	}
	else
	{
		createInfo.enabledLayerCount = 0;
	}

	VkPhysicalDeviceFeatures2 physicalDeviceFeatures2{};
	if (pNextChain)
	{
		physicalDeviceFeatures2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
		physicalDeviceFeatures2.features = enabledFeatures;
		physicalDeviceFeatures2.pNext = pNextChain;
		createInfo.pEnabledFeatures = nullptr;
		createInfo.pNext = &physicalDeviceFeatures2;
	}

	if (vkCreateDevice(physicalDevice, &createInfo, nullptr, &logicalDevice) != VK_SUCCESS)
	{
		throw std::runtime_error("failed to create logical device!");
	}
}

bool VulkanDevice::checkDeviceExtensionSupport(const std::vector<const char *> &enabledExtensions)
{
	uint32_t extensionCount{};
	vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &extensionCount, nullptr);

	std::vector<VkExtensionProperties> availableExtensions(extensionCount);
	vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &extensionCount, availableExtensions.data());

	std::set<std::string> requiredExtensions(enabledExtensions.begin(), enabledExtensions.end());

	for (const auto &extension : availableExtensions)
	{
		requiredExtensions.erase(extension.extensionName);
	}

	return requiredExtensions.empty();
}

bool VulkanDevice::checkDeviceFeaturesSupport(VkPhysicalDeviceFeatures enabledFeatures)
{
	VkPhysicalDeviceFeatures features;
	vkGetPhysicalDeviceFeatures(physicalDevice, &features);

	VkBool32 *baseAddr = reinterpret_cast<VkBool32 *>(&features);
	VkBool32 *enabledBaseAddr = reinterpret_cast<VkBool32 *>(&enabledFeatures);

	int numOfFeatures{sizeof(VkPhysicalDeviceFeatures) / sizeof(VkBool32)};
	for (int i{0}; i < numOfFeatures; ++i)
	{
		// If the feature that we want to enable is not supported then device is not adequate
		if ((*(enabledBaseAddr + i) == VK_TRUE) && (*(baseAddr + i) == VK_FALSE))
		{
			return false;
		}
	}

	return true;
}

void VulkanDevice::queryQueueFamilyIndices(VkSurfaceKHR surface)
{
	uint32_t queueFamilyCount;
	vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, nullptr);
	std::vector<VkQueueFamilyProperties> queueFamilyProperties(queueFamilyCount);
	vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, queueFamilyProperties.data());

	for (uint32_t i{0}; i < static_cast<uint32_t>(queueFamilyProperties.size()); ++i)
	{
		// Take the first queue family that supports graphics
		if (!queueFamilyIndices.graphics.has_value() && (queueFamilyProperties[i].queueFlags & VK_QUEUE_GRAPHICS_BIT))
		{
			queueFamilyIndices.graphics = i;
		}

		// Take the first queue family that supports present
		if (!queueFamilyIndices.present.has_value())
		{
			VkBool32 presentSupport{false};
			vkGetPhysicalDeviceSurfaceSupportKHR(physicalDevice, i, surface, &presentSupport);
			if (presentSupport)
			{
				queueFamilyIndices.present = i;
			}
		}

		// Find dedicated queue for compute
		if ((queueFamilyProperties[i].queueFlags & VK_QUEUE_COMPUTE_BIT) && ((queueFamilyProperties[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) == 0))
		{
			queueFamilyIndices.compute = i;
		}

		// If no separate compute queue is found yet, take any queue that supports compute
		if (!queueFamilyIndices.compute.has_value() && (queueFamilyProperties[i].queueFlags & VK_QUEUE_COMPUTE_BIT))
		{
			queueFamilyIndices.compute = i;
		}

		// Find dedicated queue for transfer
		if ((queueFamilyProperties[i].queueFlags & VK_QUEUE_TRANSFER_BIT) && ((queueFamilyProperties[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) == 0) && ((queueFamilyProperties[i].queueFlags & VK_QUEUE_COMPUTE_BIT) == 0))
		{
			queueFamilyIndices.transfer = i;
		}

		// If no separate transfer queue is found yet, take any queue that supports transfer
		if (!queueFamilyIndices.transfer.has_value() && (queueFamilyProperties[i].queueFlags & VK_QUEUE_TRANSFER_BIT))
		{
			queueFamilyIndices.transfer = i;
		}
	}
}

VkSampleCountFlagBits VulkanDevice::getMaxUsableSampleCount()
{
	VkSampleCountFlags counts{properties.limits.framebufferColorSampleCounts &
	                          properties.limits.framebufferDepthSampleCounts};

	if (counts & VK_SAMPLE_COUNT_64_BIT)
	{
		return VK_SAMPLE_COUNT_64_BIT;
	}
	if (counts & VK_SAMPLE_COUNT_32_BIT)
	{
		return VK_SAMPLE_COUNT_32_BIT;
	}
	if (counts & VK_SAMPLE_COUNT_16_BIT)
	{
		return VK_SAMPLE_COUNT_16_BIT;
	}
	if (counts & VK_SAMPLE_COUNT_8_BIT)
	{
		return VK_SAMPLE_COUNT_8_BIT;
	}
	if (counts & VK_SAMPLE_COUNT_4_BIT)
	{
		return VK_SAMPLE_COUNT_4_BIT;
	}
	if (counts & VK_SAMPLE_COUNT_2_BIT)
	{
		return VK_SAMPLE_COUNT_2_BIT;
	}

	return VK_SAMPLE_COUNT_1_BIT;
}

uint32_t VulkanDevice::findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags memPropertyFlags)
{
	for (uint32_t i{0}; i < memProperties.memoryTypeCount; i++)
	{
		if ((typeFilter & (1 << i)) &&
		    (memProperties.memoryTypes[i].propertyFlags & memPropertyFlags) == memPropertyFlags)
		{
			return i;
		}
	}

	throw std::runtime_error("failed to find suitable memory type!");
}

VkFormat VulkanDevice::findSupportedFormat(const std::vector<VkFormat> &candidates, VkImageTiling tiling, VkFormatFeatureFlags features)
{
	for (VkFormat format : candidates)
	{
		VkFormatProperties props{};
		vkGetPhysicalDeviceFormatProperties(physicalDevice, format, &props);
		if (tiling == VK_IMAGE_TILING_LINEAR && (props.linearTilingFeatures & features) == features)
		{
			return format;
		}
		else if (tiling == VK_IMAGE_TILING_OPTIMAL && (props.optimalTilingFeatures & features) == features)
		{
			return format;
		}
	}

	throw std::runtime_error("failed to find supported format!");
}

VkFormat VulkanDevice::findDepthFormat()
{
	if (depthFormat)
		return depthFormat;

	return findSupportedFormat(
	    {VK_FORMAT_X8_D24_UNORM_PACK32, VK_FORMAT_D24_UNORM_S8_UINT, VK_FORMAT_D32_SFLOAT,
	     VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D16_UNORM, VK_FORMAT_D16_UNORM_S8_UINT},
	    VK_IMAGE_TILING_OPTIMAL, VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT);
}
