#pragma once

#include <cassert>
#include <vulkan/vulkan.h>
#include <vector>
#include <glm/glm.hpp>

class VulkanRaytracing
{
private:
	VkDevice logicalDevice;

public:
	struct BlasInput {
		// Per Geometry 
		std::vector<VkAccelerationStructureGeometryKHR> geometry{};
		std::vector<VkAccelerationStructureBuildRangeInfoKHR> buildRangeInfo{};

		// Per BLAS
		const VkAccelerationStructureBuildRangeInfoKHR* pBuildRangeInfo{};
		VkAccelerationStructureBuildGeometryInfoKHR buildGeomInfo{};
		VkAccelerationStructureBuildSizesInfoKHR buildSizeInfo{};
	};

	struct Blas {
		VkAccelerationStructureKHR as{};
		VkBuffer buffer{};
		VkDeviceMemory bufferMemory{};
	};

	struct Tlas {
		VkAccelerationStructureKHR as{};
		VkBuffer buffer{};
		VkDeviceMemory bufferMemory{};
	};

	// TODO: Might want to change this later
	struct PushConstantRay {
		int frame{-1};
	};

	void setContext(VkDevice device);

	PFN_vkCreateAccelerationStructureKHR vkCreateAccelerationStructureKHR;
	PFN_vkDestroyAccelerationStructureKHR vkDestroyAccelerationStructureKHR;
	PFN_vkCmdBuildAccelerationStructuresKHR vkCmdBuildAccelerationStructuresKHR;
	PFN_vkCmdCopyAccelerationStructureKHR vkCmdCopyAccelerationStructureKHR;
	PFN_vkCmdWriteAccelerationStructuresPropertiesKHR vkCmdWriteAccelerationStructuresPropertiesKHR;
	PFN_vkGetAccelerationStructureBuildSizesKHR vkGetAccelerationStructureBuildSizesKHR;
	PFN_vkGetAccelerationStructureDeviceAddressKHR vkGetAccelerationStructureDeviceAddressKHR;
	PFN_vkCreateRayTracingPipelinesKHR vkCreateRayTracingPipelinesKHR;
	PFN_vkGetRayTracingShaderGroupHandlesKHR vkGetRayTracingShaderGroupHandlesKHR;
	PFN_vkCmdTraceRaysKHR vkCmdTraceRaysKHR;
};