#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <vulkan/vulkan.h>
#include <vector>
#include <glm/glm.hpp>
#include <stdexcept>

#include "VulkanUtils.h"

class VulkanRaytracing
{
private:
	// Holds data for building BLAS
	struct BlasInput
	{
		// Per Geometry
		std::vector<VkAccelerationStructureGeometryKHR> geometry{};
		std::vector<VkAccelerationStructureBuildRangeInfoKHR> buildRangeInfo{};

		// Per BLAS
		const VkAccelerationStructureBuildRangeInfoKHR *pBuildRangeInfo{};
		VkAccelerationStructureBuildGeometryInfoKHR buildGeomInfo{};
		VkAccelerationStructureBuildSizesInfoKHR buildSizeInfo{};
	};

	struct Blas {
		VkAccelerationStructureKHR as{};
		VkBuffer buffer{};
		VkDeviceMemory bufferMemory{};
	};

	struct PushConstantRay
	{
		int frame{ -1 };
	} m_pcRay;

	VulkanDevice *m_vulkanDevice{};

	PFN_vkCreateAccelerationStructureKHR vkCreateAccelerationStructureKHR{};
	PFN_vkDestroyAccelerationStructureKHR vkDestroyAccelerationStructureKHR{};
	PFN_vkCmdBuildAccelerationStructuresKHR vkCmdBuildAccelerationStructuresKHR{};
	PFN_vkGetAccelerationStructureBuildSizesKHR vkGetAccelerationStructureBuildSizesKHR{};

	VkPhysicalDeviceShaderClockFeaturesKHR m_shaderClockFeatures{};
	VkPhysicalDeviceScalarBlockLayoutFeatures m_scalarBlockLayoutFeatures{};
	VkPhysicalDeviceHostQueryResetFeatures m_hostQueryResetFeatures{};
	VkPhysicalDeviceBufferDeviceAddressFeatures m_bufferDeviceAddressFeatures{};
	VkPhysicalDeviceAccelerationStructureFeaturesKHR m_asFeatures{};
	VkPhysicalDeviceRayTracingPipelineFeaturesKHR m_rtPipelineFeatures{};

	VkPhysicalDeviceRayTracingPipelinePropertiesKHR m_pipelineProperties{};

	// For bottom level acceleration structure (BLAS)
	std::vector<Blas> m_blases{};

	VkDescriptorPool m_descriptorPool{};
	std::vector<VkDescriptorSet> m_descriptorSets{};
	VkDescriptorSetLayout m_descriptorSetLayout{};

	VkBuffer m_SBTBuffer{};
	VkDeviceMemory m_SBTBufferMemory{};

	VkStridedDeviceAddressRegionKHR m_rgenRegion{};
	VkStridedDeviceAddressRegionKHR m_missRegion{};
	VkStridedDeviceAddressRegionKHR m_hitRegion{};
	VkStridedDeviceAddressRegionKHR m_callRegion{};

	std::vector<VkRayTracingShaderGroupCreateInfoKHR> m_shaderGroups{};
	VkPipelineLayout m_pipelineLayout{};
	VkPipeline m_pipeline{};

	// Assign vkCreateAccelerationStructureKHR and other variables to their functions
	void setExtensionFunctions();

	// For creating BLAS
	BlasInput objectToVkGeometryKHR(uint32_t vertexCount, uint32_t indexCount, VkDeviceAddress vertexAddress, VkDeviceAddress indexAddress, VkDeviceSize vertexStride);
	void cmdCreateBlas(std::vector<BlasInput> &blasInputs, std::vector<std::size_t> indices, VkDeviceAddress scratchAddress, VkQueryPool queryPool, VkQueue graphicsQueue);
	void cmdCompactBlas(std::vector<BlasInput> &blasInputs, std::vector<std::size_t> indices, VkQueryPool queryPool, VkQueue graphicsQueue);

	// For building TLAS
	void createInstancesBuffer(VkBuffer &instancesBuffer, VkDeviceMemory &instancesBufferMemory, VkQueue graphicsQueue);
	
	VkDeviceAddress getBlasDeviceAddress(VkAccelerationStructureKHR blas);

public:
	struct Tlas {
		VkAccelerationStructureKHR as{};
		VkBuffer buffer{};
		VkDeviceMemory bufferMemory{};
	};

	// For top level acceleration structure (TLAS)
	Tlas m_tlas{};

	void setContext(VulkanDevice* vulkanDevice);
	void* getEnabledFeatures();
	void getPipelineProperties();

	void buildBlas(VkQueue graphicsQueue, uint32_t vertexCount, uint32_t indexCount, VkDeviceAddress vertexAddress, VkDeviceAddress indexAddress, VkDeviceSize vertexStride, bool allowCompaction = true);
	void buildTlas(VkQueue graphicsQueue);

	void createDescriptorSetLayout();
	void createDescriptorPool();
	void createDescriptorSets(VkImageView imageView);
	void updateDescriptorSets(VkImageView imageView);
	
	void createPipeline(VkDescriptorSetLayout rasterDescriptorSetLayout);
	void createShaderBindingTable();

	void raytrace(VkCommandBuffer commandBuffer, uint32_t currentFrame, const std::vector<VkDescriptorSet>& rasterDescriptorSets, uint32_t width, uint32_t height);
	void updateFrame();
	void resetFrame();

	void cleanup();
};