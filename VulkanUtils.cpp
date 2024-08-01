#include "VulkanUtils.h"

namespace VulkanUtils
{
std::vector<char> readFile(const std::string& filename)
{
	std::ifstream file(filename, std::ios::ate | std::ios::binary);

	if (!file.is_open())
	{
		throw std::runtime_error("failed to open file!");
	}

	size_t fileSize = (size_t) file.tellg();
	std::vector<char> buffer(fileSize);

	file.seekg(0); // seek back to beginning of file
	file.read(buffer.data(), fileSize);

	file.close();

	return buffer;
}

VkShaderModule createShaderModule(VkDevice logicalDevice, const std::vector<char>& code)
{
	VkShaderModuleCreateInfo createInfo{};
	createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
	createInfo.codeSize = code.size();
	createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

	VkShaderModule shaderModule;
	if (vkCreateShaderModule(logicalDevice, &createInfo, nullptr, &shaderModule) != VK_SUCCESS)
	{
		throw std::runtime_error("failed to create shader module!");
	}

	return shaderModule;
}

VkDeviceAddress getBufferDeviceAddress(VkDevice logicalDevice, VkBuffer buffer)
{
	VkBufferDeviceAddressInfo bufferDeviceAddressInfo{};
	bufferDeviceAddressInfo.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
	bufferDeviceAddressInfo.buffer = buffer;
	return vkGetBufferDeviceAddress(logicalDevice, &bufferDeviceAddressInfo);
}

} // namespace VulkanUtils
