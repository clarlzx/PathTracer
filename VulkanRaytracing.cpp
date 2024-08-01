#include "VulkanRaytracing.h"

// Return new alignment (that >= size, hence fitting size), that is a multiple of alignment
uint32_t alignedSize(uint32_t size, uint32_t alignment)
{
	return (size + alignment - 1) & ~(alignment - 1);
}

void VulkanRaytracing::setContext(VulkanDevice* vulkanDevice)
{
	m_vulkanDevice = vulkanDevice;
}

// Assign vkCreateAccelerationStructureKHR and other variables to their functions
void VulkanRaytracing::setExtensionFunctions()
{
	vkCreateAccelerationStructureKHR = reinterpret_cast<PFN_vkCreateAccelerationStructureKHR>(vkGetDeviceProcAddr(m_vulkanDevice->logicalDevice, "vkCreateAccelerationStructureKHR"));
	vkDestroyAccelerationStructureKHR = reinterpret_cast<PFN_vkDestroyAccelerationStructureKHR>(vkGetDeviceProcAddr(m_vulkanDevice->logicalDevice, "vkDestroyAccelerationStructureKHR"));
	vkCmdBuildAccelerationStructuresKHR = reinterpret_cast<PFN_vkCmdBuildAccelerationStructuresKHR>(vkGetDeviceProcAddr(m_vulkanDevice->logicalDevice, "vkCmdBuildAccelerationStructuresKHR"));
	vkGetAccelerationStructureBuildSizesKHR = reinterpret_cast<PFN_vkGetAccelerationStructureBuildSizesKHR>(vkGetDeviceProcAddr(m_vulkanDevice->logicalDevice, "vkGetAccelerationStructureBuildSizesKHR"));
}

/*
Checks if required features for raytracing are supported, if so store it in pNext to be enabled during logical device creation
*/
void* VulkanRaytracing::getEnabledFeatures()
{
	assert(m_vulkanDevice->physicalDevice);

	VkPhysicalDeviceFeatures2 deviceFeatures2{};
	deviceFeatures2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;

	m_shaderClockFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_CLOCK_FEATURES_KHR;
	deviceFeatures2.pNext = &m_shaderClockFeatures;
	vkGetPhysicalDeviceFeatures2(m_vulkanDevice->physicalDevice, &deviceFeatures2);
	if (m_shaderClockFeatures.shaderDeviceClock != VK_TRUE && m_shaderClockFeatures.shaderSubgroupClock != VK_TRUE)
	{
		throw std::runtime_error("shader clock is not supported!");
	}

	m_scalarBlockLayoutFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SCALAR_BLOCK_LAYOUT_FEATURES;
	deviceFeatures2.pNext = &m_scalarBlockLayoutFeatures;
	vkGetPhysicalDeviceFeatures2(m_vulkanDevice->physicalDevice, &deviceFeatures2);
	if (m_scalarBlockLayoutFeatures.scalarBlockLayout != VK_TRUE)
	{
		throw std::runtime_error("scalar block layout not supported!");
	}

	m_hostQueryResetFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_HOST_QUERY_RESET_FEATURES;
	deviceFeatures2.pNext = &m_hostQueryResetFeatures;
	vkGetPhysicalDeviceFeatures2(m_vulkanDevice->physicalDevice, &deviceFeatures2);
	if (m_hostQueryResetFeatures.hostQueryReset != VK_TRUE)
	{
		throw std::runtime_error("host query reset not supported!");
	}

	m_bufferDeviceAddressFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES;
	deviceFeatures2.pNext = &m_bufferDeviceAddressFeatures;
	vkGetPhysicalDeviceFeatures2(m_vulkanDevice->physicalDevice, &deviceFeatures2);
	if (m_bufferDeviceAddressFeatures.bufferDeviceAddress != VK_TRUE)
	{
		throw std::runtime_error("buffer device address not supported!");
	}

	m_rtPipelineFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR;
	deviceFeatures2.pNext = &m_rtPipelineFeatures;
	vkGetPhysicalDeviceFeatures2(m_vulkanDevice->physicalDevice, &deviceFeatures2);
	if (m_rtPipelineFeatures.rayTracingPipeline != VK_TRUE)
	{
		throw std::runtime_error("ray tracing pipeline not supported");
	}

	m_asFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR;
	deviceFeatures2.pNext = &m_asFeatures;
	vkGetPhysicalDeviceFeatures2(m_vulkanDevice->physicalDevice, &deviceFeatures2);
	if (m_asFeatures.accelerationStructure != VK_TRUE)
	{
		throw std::runtime_error("acceleration structure not supported");
	}

	m_scalarBlockLayoutFeatures.pNext = &m_shaderClockFeatures;
	m_hostQueryResetFeatures.pNext = &m_scalarBlockLayoutFeatures;
	m_bufferDeviceAddressFeatures.pNext = &m_hostQueryResetFeatures;
	m_rtPipelineFeatures.pNext = &m_bufferDeviceAddressFeatures;
	m_asFeatures.pNext = &m_rtPipelineFeatures;
	return &m_asFeatures;
}

void VulkanRaytracing::getPipelineProperties()
{
	assert(m_vulkanDevice->physicalDevice);

	// Request ray tracing properties
	VkPhysicalDeviceProperties2 deviceProperties2{};
	deviceProperties2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;

	m_pipelineProperties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR;
	deviceProperties2.pNext = &m_pipelineProperties;
	vkGetPhysicalDeviceProperties2(m_vulkanDevice->physicalDevice, &deviceProperties2);
}

VulkanRaytracing::BlasInput VulkanRaytracing::objectToVkGeometryKHR(uint32_t vertexCount, uint32_t indexCount,
                                                                    VkDeviceAddress vertexAddress, VkDeviceAddress indexAddress,
                                                                    VkDeviceSize vertexStride)
{
	// Number of triangles, each triangle is treated as 3 vertices
	/*uint32_t maxPrimitiveCount{ static_cast<uint32_t>(m_indices.size()) / 3 };*/

	// Describe buffer as array of Vertex objects
	VkAccelerationStructureGeometryTrianglesDataKHR triangles{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR };
	triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
	//triangles.vertexData.deviceAddress = m_model.vertexAddress;
	//triangles.vertexStride = sizeof(Vertex);
	triangles.vertexData.deviceAddress = vertexAddress;
	triangles.vertexStride = vertexStride;
	// Describe index data (32-bit unsigned int, so max possible indices in 2^32)
	triangles.indexType = VK_INDEX_TYPE_UINT32;
	//triangles.indexData.deviceAddress = m_model.indexAddress;
	triangles.indexData.deviceAddress = indexAddress;
	// Indicate identity transform by setting transformData to null device pointer
	triangles.transformData = {};
	//triangles.maxVertex = static_cast<uint32_t>(m_vertices.size() - 1);
	triangles.maxVertex = vertexCount - 1;

	// Identify the above data as containing opaque triangles
	VkAccelerationStructureGeometryKHR asGeometry{};
	asGeometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
	asGeometry.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
	asGeometry.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
	asGeometry.geometry.triangles = triangles;

	// The entire array will be used to build the BLAS
	VkAccelerationStructureBuildRangeInfoKHR buildRangeInfo{};
	buildRangeInfo.primitiveCount = indexCount / 3;
	buildRangeInfo.primitiveOffset = 0;
	buildRangeInfo.firstVertex = 0;
	buildRangeInfo.transformOffset = 0;

	// Could add more geometry in each Blas, but only one for now
	BlasInput input{};
	input.geometry.emplace_back(asGeometry);
	input.buildRangeInfo.emplace_back(buildRangeInfo);

	return input;
}

void VulkanRaytracing::buildBlas(VkQueue graphicsQueue,
                                 uint32_t vertexCount, uint32_t indexCount,
                                 VkDeviceAddress vertexAddress, VkDeviceAddress indexAddress,
                                 VkDeviceSize vertexStride, bool allowCompaction /*= true*/)
{
	setExtensionFunctions();

	VkDeviceSize asTotalSize{ 0 };    // Memory size of all allocated BLAS
	VkDeviceSize maxScratchSize{ 0 }; // Largest required scratch size

	// Convert all OBJ models into ray tracing geometry used to build BLAS
	const size_t numBlas{ 1 };
	std::vector<BlasInput> blasInputs{ objectToVkGeometryKHR(vertexCount, indexCount, vertexAddress, indexAddress, vertexStride) };

	m_blases.resize(numBlas);

	for (size_t blasIdx{ 0 }; blasIdx < numBlas; ++blasIdx)
	{
		blasInputs[blasIdx].buildGeomInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
		blasInputs[blasIdx].buildGeomInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
		blasInputs[blasIdx].buildGeomInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
		// TODO: Check if with compaction is better or not
		// Note that there cannot be a mix of compaction & no compaction
		if (allowCompaction)
		{
			// Compaction will limit memory allocation required
			// Add flag VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_COMPACTION_BIT_KHR to allow compaction, so that BLAS will use less memory
			blasInputs[blasIdx].buildGeomInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR | VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_COMPACTION_BIT_KHR;
		}
		else
		{
			blasInputs[blasIdx].buildGeomInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
		}
		blasInputs[blasIdx].buildGeomInfo.geometryCount = static_cast<uint32_t>(blasInputs[blasIdx].geometry.size());
		blasInputs[blasIdx].buildGeomInfo.pGeometries = blasInputs[blasIdx].geometry.data();

		blasInputs[blasIdx].pBuildRangeInfo = blasInputs[blasIdx].buildRangeInfo.data();

		// Find sizes of each geometry in the BLAS to create acceleration structure and scratch buffer
		std::vector<uint32_t> maxPrimitiveCounts(blasInputs[blasIdx].buildRangeInfo.size());
		for (size_t geomIndex{ 0 }; geomIndex < blasInputs[blasIdx].buildRangeInfo.size(); ++geomIndex)
		{
			maxPrimitiveCounts[geomIndex] = blasInputs[blasIdx].buildRangeInfo[geomIndex].primitiveCount; // Total number of primitives for one BLAS
		}
		blasInputs[blasIdx].buildSizeInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
		vkGetAccelerationStructureBuildSizesKHR(m_vulkanDevice->logicalDevice, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
		                                                 &blasInputs[blasIdx].buildGeomInfo, maxPrimitiveCounts.data(), &blasInputs[blasIdx].buildSizeInfo);

		asTotalSize += blasInputs[blasIdx].buildSizeInfo.accelerationStructureSize;
		maxScratchSize = std::max(maxScratchSize, blasInputs[blasIdx].buildSizeInfo.buildScratchSize);
	}

	// Create scratch buffer
	// Will be reused when building each BLAS, so scratch buffer created with maximum scratch memory needed
	VkBuffer scratchBuffer{};
	VkDeviceMemory scratchBufferMemory{};
	m_vulkanDevice->createBuffer(maxScratchSize,
	                             VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
	                             VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
	                             scratchBuffer, scratchBufferMemory);
	VkDeviceAddress scratchAddress{ VulkanUtils::getBufferDeviceAddress(m_vulkanDevice->logicalDevice, scratchBuffer) };

	// Allocate a query pool for storing the needed size for every acceleration structure compaction
	VkQueryPool queryPool{ VK_NULL_HANDLE };
	if (allowCompaction)
	{
		VkQueryPoolCreateInfo queryPoolCreateInfo{};
		queryPoolCreateInfo.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
		queryPoolCreateInfo.queryCount = numBlas;
		queryPoolCreateInfo.queryType = VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR;
		vkCreateQueryPool(m_vulkanDevice->logicalDevice, &queryPoolCreateInfo, nullptr, &queryPool);
	}

	// Batching creation/compaction of BLAS to allow staying in restricted amount of memory (set to 256 MB chunks)
	// Otherwise creating all BLAS in single command buffer could stall pipeline and potentially create problems
	std::vector<size_t> blasIndices{};
	VkDeviceSize batchSize{ 0 };
	VkDeviceSize batchLimit{ 256'000'000 };

	for (size_t blasIdx{ 0 }; blasIdx < numBlas; ++blasIdx)
	{
		blasIndices.push_back(blasIdx);
		batchSize += blasInputs[blasIdx].buildSizeInfo.accelerationStructureSize;

		// Build BLAS when over the limit or last BLAS element
		if (batchSize >= batchLimit || blasIdx == numBlas - 1)
		{
			cmdCreateBlas(blasInputs, blasIndices, scratchAddress, queryPool, graphicsQueue);

			if (allowCompaction)
			{
				cmdCompactBlas(blasInputs, blasIndices, queryPool, graphicsQueue);
			}

			batchSize = 0;
			blasIndices.clear();
		}
	}

	// Logging reduction
	if (allowCompaction)
	{
		VkDeviceSize compactSize{ 0 };
		for (size_t blasIdx{ 0 }; blasIdx < numBlas; ++blasIdx)
		{
			compactSize += blasInputs[blasIdx].buildSizeInfo.accelerationStructureSize;
		}
		const float fractionSmaller{ asTotalSize == 0 ? 0 : (asTotalSize - compactSize) / float(asTotalSize) };
		std::cout << "Reduced size of BLAS by " << fractionSmaller * 100.f << "% from " << asTotalSize << " to " << compactSize << '\n';
	}

	// Clean up temporary resources
	vkDestroyQueryPool(m_vulkanDevice->logicalDevice, queryPool, nullptr);
	vkDestroyBuffer(m_vulkanDevice->logicalDevice, scratchBuffer, nullptr);
	vkFreeMemory(m_vulkanDevice->logicalDevice, scratchBufferMemory, nullptr);
}

void VulkanRaytracing::cmdCreateBlas(std::vector<VulkanRaytracing::BlasInput> &blasInputs,
                                     std::vector<size_t> indices, VkDeviceAddress scratchAddress,
                                     VkQueryPool queryPool, VkQueue graphicsQueue)
{
	// Reset the query to know real size of the BLAS (for compaction)
	if (queryPool)
	{
		vkResetQueryPool(m_vulkanDevice->logicalDevice, queryPool, 0, static_cast<uint32_t>(indices.size()));
	}

	for (size_t i : indices)
	{
		// Create acceleration structure buffer
		m_vulkanDevice->createBuffer(blasInputs[i].buildSizeInfo.accelerationStructureSize,
		                             VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
		                             VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
		                             m_blases[i].buffer,
		                             m_blases[i].bufferMemory);

		VkAccelerationStructureCreateInfoKHR asCreateInfo{};
		asCreateInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
		asCreateInfo.buffer = m_blases[i].buffer;
		asCreateInfo.size = blasInputs[i].buildSizeInfo.accelerationStructureSize;
		asCreateInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;

		vkCreateAccelerationStructureKHR(m_vulkanDevice->logicalDevice, &asCreateInfo, nullptr, &m_blases[i].as);

		blasInputs[i].buildGeomInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
		blasInputs[i].buildGeomInfo.dstAccelerationStructure = m_blases[i].as;  // Setting where the build lands
		blasInputs[i].buildGeomInfo.scratchData.deviceAddress = scratchAddress; // All builds are using the same scratch buffer

		VkCommandBuffer commandBuffer{ m_vulkanDevice->beginSingleTimeCommands() };

		// Build the Bottom Level Acceleration Structure (BLAS)
		vkCmdBuildAccelerationStructuresKHR(commandBuffer, 1, &blasInputs[i].buildGeomInfo, &blasInputs[i].pBuildRangeInfo);

		// Since the scratch buffer is reused across builds for multiple BLAS,
		// need a barrier to ensure one build is finished before starting the next one
		// TODO: Ideally should have multiple regions of scratch buffer so multiple BLAS can be built simultaneously
		VkMemoryBarrier barrier{};
		barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
		barrier.srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
		barrier.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
		vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
		                     VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, 0, 1, &barrier, 0, nullptr, 0, nullptr);

		// Query the amount of memory needed for compaction
		if (queryPool)
		{
			PFN_vkCmdWriteAccelerationStructuresPropertiesKHR vkCmdWriteAccelerationStructuresPropertiesKHR = reinterpret_cast<PFN_vkCmdWriteAccelerationStructuresPropertiesKHR>(vkGetDeviceProcAddr(m_vulkanDevice->logicalDevice, "vkCmdWriteAccelerationStructuresPropertiesKHR"));
			vkCmdWriteAccelerationStructuresPropertiesKHR(commandBuffer, 1, &blasInputs[i].buildGeomInfo.dstAccelerationStructure,
			                                                   VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR, queryPool, static_cast<uint32_t>(i - indices[0]));
		}

		m_vulkanDevice->endSingleTimeCommands(commandBuffer, graphicsQueue);
	}
}

// Create and replace a new acceleration structure and buffer
// based on the size retrieved in cmdCreateBlas()
void VulkanRaytracing::cmdCompactBlas(std::vector<VulkanRaytracing::BlasInput> &blasInputs,
                                      std::vector<size_t> indices, VkQueryPool queryPool,
                                      VkQueue graphicsQueue)
{
	// Get compacted size result written in cmdCreateBlas()`
	uint32_t numBlas{ static_cast<uint32_t>(indices.size()) };
	std::vector<VkDeviceSize> compactSizes(numBlas);
	vkGetQueryPoolResults(m_vulkanDevice->logicalDevice, queryPool, 0, numBlas, numBlas * sizeof(VkDeviceSize),
	                      compactSizes.data(), sizeof(VkDeviceSize), VK_QUERY_RESULT_WAIT_BIT);

	for (size_t i : indices)
	{
		// New reduced size
		blasInputs[i].buildSizeInfo.accelerationStructureSize = compactSizes[i - indices[0]];

		// Create a compact version of the acceleration structure
		VkAccelerationStructureKHR compactBlas{};
		VkBuffer compactBlasBuffer{};
		VkDeviceMemory compactBlasBufferMemory{};

		// Create compact acceleration structure buffer
		m_vulkanDevice->createBuffer(blasInputs[i].buildSizeInfo.accelerationStructureSize,
		                             VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
		                             VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
		                             compactBlasBuffer,
		                             compactBlasBufferMemory);

		VkAccelerationStructureCreateInfoKHR compactASCreateInfo{};
		compactASCreateInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
		compactASCreateInfo.buffer = compactBlasBuffer;
		compactASCreateInfo.size = blasInputs[i].buildSizeInfo.accelerationStructureSize;
		compactASCreateInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
		vkCreateAccelerationStructureKHR(m_vulkanDevice->logicalDevice, &compactASCreateInfo, nullptr, &compactBlas);

		VkCommandBuffer compactCommandBuffer{ m_vulkanDevice->beginSingleTimeCommands() };

		// Copy the original BLAS to a compact version
		VkCopyAccelerationStructureInfoKHR copyInfo{};
		copyInfo.sType = VK_STRUCTURE_TYPE_COPY_ACCELERATION_STRUCTURE_INFO_KHR;
		copyInfo.src = blasInputs[i].buildGeomInfo.dstAccelerationStructure;
		copyInfo.dst = compactBlas;
		copyInfo.mode = VK_COPY_ACCELERATION_STRUCTURE_MODE_COMPACT_KHR;
		PFN_vkCmdCopyAccelerationStructureKHR vkCmdCopyAccelerationStructureKHR = reinterpret_cast<PFN_vkCmdCopyAccelerationStructureKHR>(vkGetDeviceProcAddr(m_vulkanDevice->logicalDevice, "vkCmdCopyAccelerationStructureKHR"));
		vkCmdCopyAccelerationStructureKHR(compactCommandBuffer, &copyInfo);

		m_vulkanDevice->endSingleTimeCommands(compactCommandBuffer, graphicsQueue);

		// Destroy original BLAS
		vkDestroyAccelerationStructureKHR(m_vulkanDevice->logicalDevice, m_blases[i].as, nullptr);
		vkDestroyBuffer(m_vulkanDevice->logicalDevice, m_blases[i].buffer, nullptr);
		vkFreeMemory(m_vulkanDevice->logicalDevice, m_blases[i].bufferMemory, nullptr);

		m_blases[i].as = compactBlas;
		m_blases[i].buffer = compactBlasBuffer;
		m_blases[i].bufferMemory = compactBlasBufferMemory;
	}
}

void VulkanRaytracing::createInstancesBuffer(VkBuffer &instancesBuffer, VkDeviceMemory &instancesBufferMemory, VkQueue graphicsQueue)
{
	// Could have more instances, but only one for now
	const size_t numTlas{ 1 };
	std::vector<VkAccelerationStructureInstanceKHR> instances{};
	instances.reserve(numTlas);

	// Use identity matrix, and gl_InstanceCustomIndex = i = 0 for now,
	// since there is only one instance
	VkTransformMatrixKHR transformMatrix = { 1.0f, 0.0f, 0.0f, 0.0f,
		                                     0.0f, 1.0f, 0.0f, 0.0f,
		                                     0.0f, 0.0f, 1.0f, 0.0f };

	for (size_t i{ 0 }; i < numTlas; ++i)
	{
		VkAccelerationStructureInstanceKHR instance{};
		// Set the instance transform to the identity matrix
		instance.transform = transformMatrix;
		// gl_InstanceCustomIndex
		instance.instanceCustomIndex = i;
		// Reference to corresponding BLAS
		instance.accelerationStructureReference = getBlasDeviceAddress(m_blases[i].as);
		// Disable backface culling for simplicity and independence on winding of input models
		instance.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
		// Only be hit if rayMask in traceRayEXT & instance.mask != 0, 0xFF = 1 for all bits
		instance.mask = 0xFF;
		// Set to use same hit group for all objects
		instance.instanceShaderBindingTableRecordOffset = 0; // hitGroupId
		instances.emplace_back(instance);
	}

	VkDeviceSize bufferSize{ sizeof(VkAccelerationStructureInstanceKHR) * numTlas };

	// Create staging buffer to write instance data to device local instance buffer
	VkBuffer stagingBuffer{};
	VkDeviceMemory stagingBufferMemory{};
	m_vulkanDevice->createBuffer(bufferSize,
	                             VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
	                             VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
	                             stagingBuffer,
	                             stagingBufferMemory);

	void *data{};
	vkMapMemory(m_vulkanDevice->logicalDevice, stagingBufferMemory, 0, bufferSize, 0, &data);
	memcpy(data, instances.data(), static_cast<size_t>(bufferSize));
	vkUnmapMemory(m_vulkanDevice->logicalDevice, stagingBufferMemory);

	// Create instances buffer
	m_vulkanDevice->createBuffer(bufferSize,
	                             VK_BUFFER_USAGE_TRANSFER_DST_BIT |
	                                 VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
	                             VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
	                             instancesBuffer,
	                             instancesBufferMemory);

	// vkQueueWaitIdle() in vulkanDevice.endSingleTimeCommands() ensures that instance buffer is copied
	// before we move onto acceleration structure build
	m_vulkanDevice->copyBuffer(stagingBuffer, instancesBuffer, bufferSize, graphicsQueue);

	// Clear staging buffer resources
	vkDestroyBuffer(m_vulkanDevice->logicalDevice, stagingBuffer, nullptr);
	vkFreeMemory(m_vulkanDevice->logicalDevice, stagingBufferMemory, nullptr);
}

// TODO: Call function again when updating Tlas with updated matrices,
// need to have VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR flag
void VulkanRaytracing::buildTlas(VkQueue graphicsQueue)
{
	uint32_t numTlas{ 1 };

	// Create a buffer to hold real instance data
	VkBuffer instancesBuffer{};
	VkDeviceMemory instancesBufferMemory{};
	createInstancesBuffer(instancesBuffer, instancesBufferMemory, graphicsQueue);

	VkAccelerationStructureGeometryInstancesDataKHR instancesData{};
	instancesData.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
	instancesData.data.deviceAddress = VulkanUtils::getBufferDeviceAddress(m_vulkanDevice->logicalDevice, instancesBuffer);

	VkAccelerationStructureGeometryKHR geometry{};
	geometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
	geometry.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
	geometry.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
	geometry.geometry.instances = instancesData;

	VkAccelerationStructureBuildGeometryInfoKHR buildGeometryInfo{};
	buildGeometryInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
	buildGeometryInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
	buildGeometryInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
	buildGeometryInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
	buildGeometryInfo.geometryCount = 1;
	buildGeometryInfo.pGeometries = &geometry;
	buildGeometryInfo.srcAccelerationStructure = VK_NULL_HANDLE;

	VkAccelerationStructureBuildSizesInfoKHR buildSizeInfo{};
	buildSizeInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
	vkGetAccelerationStructureBuildSizesKHR(m_vulkanDevice->logicalDevice, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
	                                        &buildGeometryInfo, &numTlas, &buildSizeInfo);

	// Create acceleration structure buffer
	m_vulkanDevice->createBuffer(buildSizeInfo.accelerationStructureSize,
	                             VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
	                             VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
	                             m_tlas.buffer,
	                             m_tlas.bufferMemory);

	VkAccelerationStructureCreateInfoKHR asCreateInfo{};
	asCreateInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
	asCreateInfo.buffer = m_tlas.buffer;
	asCreateInfo.size = buildSizeInfo.accelerationStructureSize;
	asCreateInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;

	vkCreateAccelerationStructureKHR(m_vulkanDevice->logicalDevice, &asCreateInfo, nullptr, &m_tlas.as);

	// Allocate the scratch buffers holding the temporary data used during build of acceleration structure
	VkBuffer scratchBuffer{};
	VkDeviceMemory scratchBufferMemory{};
	m_vulkanDevice->createBuffer(buildSizeInfo.buildScratchSize,
	                             VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
	                             VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
	                             scratchBuffer,
	                             scratchBufferMemory);
	VkDeviceAddress scratchAddress{ VulkanUtils::getBufferDeviceAddress(m_vulkanDevice->logicalDevice, scratchBuffer) };

	buildGeometryInfo.srcAccelerationStructure = VK_NULL_HANDLE;
	buildGeometryInfo.dstAccelerationStructure = m_tlas.as;
	buildGeometryInfo.scratchData.deviceAddress = scratchAddress;

	VkAccelerationStructureBuildRangeInfoKHR buildRangeInfo{};
	buildRangeInfo.primitiveCount = numTlas;
	buildRangeInfo.primitiveOffset = 0;
	buildRangeInfo.firstVertex = 0;
	buildRangeInfo.transformOffset = 0;

	VkCommandBuffer commandBuffer{ m_vulkanDevice->beginSingleTimeCommands() };

	const VkAccelerationStructureBuildRangeInfoKHR *pBuildRangeInfo = &buildRangeInfo;
	vkCmdBuildAccelerationStructuresKHR(commandBuffer, 1, &buildGeometryInfo, &pBuildRangeInfo);

	m_vulkanDevice->endSingleTimeCommands(commandBuffer, graphicsQueue);

	vkDestroyBuffer(m_vulkanDevice->logicalDevice, instancesBuffer, nullptr);
	vkFreeMemory(m_vulkanDevice->logicalDevice, instancesBufferMemory, nullptr);
	vkDestroyBuffer(m_vulkanDevice->logicalDevice, scratchBuffer, nullptr);
	vkFreeMemory(m_vulkanDevice->logicalDevice, scratchBufferMemory, nullptr);
}

VkDeviceAddress VulkanRaytracing::getBlasDeviceAddress(VkAccelerationStructureKHR blas)
{
	VkAccelerationStructureDeviceAddressInfoKHR accelerationDeviceAddrInfo{};
	accelerationDeviceAddrInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
	accelerationDeviceAddrInfo.accelerationStructure = blas;
	PFN_vkGetAccelerationStructureDeviceAddressKHR vkGetAccelerationStructureDeviceAddressKHR = reinterpret_cast<PFN_vkGetAccelerationStructureDeviceAddressKHR>(vkGetDeviceProcAddr(m_vulkanDevice->logicalDevice, "vkGetAccelerationStructureDeviceAddressKHR"));
	return vkGetAccelerationStructureDeviceAddressKHR(m_vulkanDevice->logicalDevice, &accelerationDeviceAddrInfo);
}

void VulkanRaytracing::createDescriptorSetLayout()
{
	// For TLAS
	VkDescriptorSetLayoutBinding tlasLayoutBinding{};
	tlasLayoutBinding.binding = 0;
	tlasLayoutBinding.descriptorCount = 1;
	tlasLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
	// TLAS is usable by both ray generation, and closest hit (to shoot secondary rays)
	tlasLayoutBinding.stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;

	// For output image written by RayGen shader
	VkDescriptorSetLayoutBinding outImageLayoutBinding{};
	outImageLayoutBinding.binding = 1;
	outImageLayoutBinding.descriptorCount = 1;
	outImageLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
	outImageLayoutBinding.stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR;

	std::array<VkDescriptorSetLayoutBinding, 2> bindings{ tlasLayoutBinding,
		                                                  outImageLayoutBinding };

	VkDescriptorSetLayoutCreateInfo layoutInfo{};
	layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
	layoutInfo.pBindings = bindings.data();

	if (vkCreateDescriptorSetLayout(m_vulkanDevice->logicalDevice, &layoutInfo, nullptr,
	                                &m_descriptorSetLayout) != VK_SUCCESS)
	{
		throw std::runtime_error("failed to create ray tracing descriptor set layout!");
	}
}

void VulkanRaytracing::createDescriptorPool()
{
	std::array<VkDescriptorPoolSize, 2> poolSizes{};
	poolSizes[0].type = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
	poolSizes[0].descriptorCount = VulkanUtils::maxFramesInFlight;
	poolSizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
	poolSizes[1].descriptorCount = VulkanUtils::maxFramesInFlight;

	VkDescriptorPoolCreateInfo poolInfo{};
	poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
	poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
	poolInfo.pPoolSizes = poolSizes.data();
	poolInfo.maxSets = VulkanUtils::maxFramesInFlight;

	if (vkCreateDescriptorPool(m_vulkanDevice->logicalDevice, &poolInfo, nullptr, &m_descriptorPool) != VK_SUCCESS)
	{
		throw std::runtime_error("failed to create ray tracing descriptor pool!");
	}
}

void VulkanRaytracing::createDescriptorSets(VkImageView imageView)
{
	std::vector<VkDescriptorSetLayout> layouts{ VulkanUtils::maxFramesInFlight, m_descriptorSetLayout };

	VkDescriptorSetAllocateInfo allocInfo{};
	allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
	allocInfo.descriptorPool = m_descriptorPool;
	allocInfo.descriptorSetCount = VulkanUtils::maxFramesInFlight;
	allocInfo.pSetLayouts = layouts.data();

	m_descriptorSets.resize(VulkanUtils::maxFramesInFlight);
	if (vkAllocateDescriptorSets(m_vulkanDevice->logicalDevice, &allocInfo, m_descriptorSets.data()) != VK_SUCCESS)
	{
		throw std::runtime_error("failed to allocate ray tracing descriptor sets!");
	}

	// Configure descriptors in descriptor sets
	for (size_t i{ 0 }; i < VulkanUtils::maxFramesInFlight; ++i)
	{
		// For TLAS
		VkWriteDescriptorSetAccelerationStructureKHR asInfo{};
		asInfo.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR;
		asInfo.accelerationStructureCount = 1;
		asInfo.pAccelerationStructures = &m_tlas.as;

		// For output image written by RayGen shader
		VkDescriptorImageInfo imageInfo{};
		imageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
		imageInfo.imageView = imageView;

		std::array<VkWriteDescriptorSet, 2> descriptorWrites{};

		descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		descriptorWrites[0].dstSet = m_descriptorSets[i];
		descriptorWrites[0].dstBinding = 0;
		descriptorWrites[0].dstArrayElement = 0;
		descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
		descriptorWrites[0].descriptorCount = 1;
		descriptorWrites[0].pNext = &asInfo;

		descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		descriptorWrites[1].dstSet = m_descriptorSets[i];
		descriptorWrites[1].dstBinding = 1;
		descriptorWrites[1].dstArrayElement = 0;
		descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
		descriptorWrites[1].descriptorCount = 1;
		descriptorWrites[1].pImageInfo = &imageInfo;

		vkUpdateDescriptorSets(m_vulkanDevice->logicalDevice, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
	}
}

void VulkanRaytracing::updateDescriptorSets(VkImageView imageView)
{
	// Only need to update output image reference
	for (size_t i{ 0 }; i < VulkanUtils::maxFramesInFlight; ++i)
	{
		// For output image written by RayGen shader
		VkDescriptorImageInfo imageInfo{};
		// TODO: Not sure why choose VK_IMAGE_LAYOUT_GENERAL
		imageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
		imageInfo.imageView = imageView;

		VkWriteDescriptorSet descriptorWrite{};

		descriptorWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		descriptorWrite.dstSet = m_descriptorSets[i];
		descriptorWrite.dstBinding = 1;
		descriptorWrite.dstArrayElement = 0;
		descriptorWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
		descriptorWrite.descriptorCount = 1;
		descriptorWrite.pImageInfo = &imageInfo;

		vkUpdateDescriptorSets(m_vulkanDevice->logicalDevice, 1, &descriptorWrite, 0, nullptr);
	}
}

void VulkanRaytracing::createPipeline(VkDescriptorSetLayout rasterDescriptorSetLayout)
{
	// TODO: Maybe don't need this, especially if have multiple of one type of shader type
	enum StageIndices
	{
		eRaygen,
		eMiss,
		eClosestHit,
		eShaderGroupCount
	};

	std::array<VkPipelineShaderStageCreateInfo, eShaderGroupCount> shaderStages{};

	VkPipelineShaderStageCreateInfo shaderStageInfo{};
	shaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	shaderStageInfo.pName = "main"; // All the same entry point

	// Raygen
	shaderStageInfo.module = VulkanUtils::createShaderModule(m_vulkanDevice->logicalDevice, VulkanUtils::readFile("spv/raytrace.rgen.spv"));
	shaderStageInfo.stage = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
	shaderStages[eRaygen] = shaderStageInfo;

	// Miss
	shaderStageInfo.module = VulkanUtils::createShaderModule(m_vulkanDevice->logicalDevice, VulkanUtils::readFile("spv/raytrace.rmiss.spv"));
	shaderStageInfo.stage = VK_SHADER_STAGE_MISS_BIT_KHR;
	shaderStages[eMiss] = shaderStageInfo;

	// Closest Hit
	shaderStageInfo.module = VulkanUtils::createShaderModule(m_vulkanDevice->logicalDevice, VulkanUtils::readFile("spv/raytrace.rchit.spv"));
	shaderStageInfo.stage = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
	shaderStages[eClosestHit] = shaderStageInfo;

	// Shader groups
	VkRayTracingShaderGroupCreateInfoKHR group{};
	group.sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
	group.anyHitShader = VK_SHADER_UNUSED_KHR;
	group.closestHitShader = VK_SHADER_UNUSED_KHR;
	group.generalShader = VK_SHADER_UNUSED_KHR;
	group.intersectionShader = VK_SHADER_UNUSED_KHR;

	// Note that a raygen or miss shader is one group on its own
	// Raygen
	group.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
	group.generalShader = eRaygen;
	m_shaderGroups.push_back(group);

	// Miss
	group.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
	group.generalShader = eMiss;
	m_shaderGroups.push_back(group);

	// Hit Group - Closest hit
	// Each hit group can comprise 1 - 3 shaders (intersection, any hit, closest hit)
	// Since triangles, can use built-in ray-triangle intersection test
	// If not triangles, will need to define an intersection shader and set type to VK_*_PROCEDURAL_HIT_GROUP_KHR
	group.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
	group.generalShader = VK_SHADER_UNUSED_KHR;
	group.closestHitShader = eClosestHit;
	m_shaderGroups.push_back(group);

	VkPushConstantRange pushConstant{};
	pushConstant.stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR;
	pushConstant.offset = 0;
	pushConstant.size = sizeof(PushConstantRay);

	VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
	pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
	pipelineLayoutInfo.pushConstantRangeCount = 1;
	pipelineLayoutInfo.pPushConstantRanges = &pushConstant;

	// Two descriptor sets
	// set = 0, specific to ray tracing pipeline (TlAS and output image)
	// set = 1, shared by rt and rasterization (i.e. scene data)
	std::vector<VkDescriptorSetLayout> layouts{ m_descriptorSetLayout,
		                                        rasterDescriptorSetLayout };
	pipelineLayoutInfo.setLayoutCount = static_cast<uint32_t>(layouts.size());
	pipelineLayoutInfo.pSetLayouts = layouts.data();

	if (vkCreatePipelineLayout(m_vulkanDevice->logicalDevice, &pipelineLayoutInfo, nullptr, &m_pipelineLayout) != VK_SUCCESS)
	{
		throw std::runtime_error("failed to create ray tracing pipeline layout!");
	}

	// Assemble the shader stages and recursion depth info into ray tracing pipeline
	VkRayTracingPipelineCreateInfoKHR rtPipelineInfo{};
	rtPipelineInfo.sType = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR;
	rtPipelineInfo.stageCount = static_cast<uint32_t>(shaderStages.size());
	rtPipelineInfo.pStages = shaderStages.data();
	// We have one raygen group, one miss group, and one hit group
	rtPipelineInfo.groupCount = static_cast<uint32_t>(m_shaderGroups.size());
	rtPipelineInfo.pGroups = m_shaderGroups.data();
	// Ray depth, 1 = no recursion (i.e. a hit shader calling traceRayEXT())
	// Note that a recursion check at runtime is not guaranteed, exceeding this depth or
	// physical device recursion limit results in undefined behavior.
	// Also, ray depth should be kept as low as possible, recursive ray tracing should be flattened into a loop
	// in the ray generation to avoid deep recursion
	rtPipelineInfo.maxPipelineRayRecursionDepth = 1;
	rtPipelineInfo.layout = m_pipelineLayout;

	if (m_pipelineProperties.maxRayRecursionDepth < rtPipelineInfo.maxPipelineRayRecursionDepth)
	{
		throw std::runtime_error("device fails to support specified ray recursion depth!");
	}
	
	PFN_vkCreateRayTracingPipelinesKHR vkCreateRayTracingPipelinesKHR = reinterpret_cast<PFN_vkCreateRayTracingPipelinesKHR>(vkGetDeviceProcAddr(m_vulkanDevice->logicalDevice, "vkCreateRayTracingPipelinesKHR"));

	if (vkCreateRayTracingPipelinesKHR(m_vulkanDevice->logicalDevice, VK_NULL_HANDLE, VK_NULL_HANDLE, 1,
	                                            &rtPipelineInfo, nullptr, &m_pipeline) != VK_SUCCESS)
	{
		throw std::runtime_error("failed to create ray tracing pipeline!");
	}

	for (auto &shaderStage : shaderStages)
	{
		vkDestroyShaderModule(m_vulkanDevice->logicalDevice, shaderStage.module, nullptr);
	}
}

void VulkanRaytracing::createShaderBindingTable()
{
	const uint32_t rgenCount{ 1 }; // Always only one raygen
	const uint32_t missCount{ 1 };
	const uint32_t hitCount{ 1 };
	auto handleCount{ rgenCount + missCount + hitCount };

	// Size in bytes of shader header
	const uint32_t handleSize{ m_pipelineProperties.shaderGroupHandleSize };
	// Required alignment in bytes for each entry in a SBT
	const uint32_t handleAlignment{ m_pipelineProperties.shaderGroupHandleAlignment };
	// Required alignment in bytes for the base of the SBT
	const uint32_t baseAlignment{ m_pipelineProperties.shaderGroupBaseAlignment };

	// size of handle aligned to shaderGroupHandleAlignment
	uint32_t handleSizeAligned{ alignedSize(handleSize, handleAlignment) };

	// stride is the byte stride between consecutive elements, each group can have more than one element
	m_rgenRegion.stride = alignedSize(handleSizeAligned, baseAlignment);
	// the size of pRayGenShaderBindingTable must be equal to its stride
	m_rgenRegion.size = m_rgenRegion.stride;

	// size of each group (except for raygen) is number of elements in the group aligned to baseAlignment
	m_missRegion.stride = handleSizeAligned;
	m_missRegion.size = alignedSize(missCount * handleSizeAligned, baseAlignment);

	m_hitRegion.stride = handleSizeAligned;
	m_hitRegion.size = alignedSize(hitCount * handleSizeAligned, baseAlignment);

	// Get all shader handles
	const uint32_t dataSize{ handleCount * handleSize };
	std::vector<uint8_t> handles(dataSize);
	// Fetch handles to shader groups of pipeline, hence must be done after createPipeline()
	PFN_vkGetRayTracingShaderGroupHandlesKHR vkGetRayTracingShaderGroupHandlesKHR = reinterpret_cast<PFN_vkGetRayTracingShaderGroupHandlesKHR>(vkGetDeviceProcAddr(m_vulkanDevice->logicalDevice, "vkGetRayTracingShaderGroupHandlesKHR"));

	if (vkGetRayTracingShaderGroupHandlesKHR(m_vulkanDevice->logicalDevice, m_pipeline, 0, handleCount, dataSize, handles.data()) != VK_SUCCESS)
	{
		throw std::runtime_error("failed to get ray tracing shader group handles");
	}

	// Note that it is also possible to se
	// parate SBT into several buffers, one for each type: raygen, miss, hit group, call
	VkDeviceSize sbtSize{ m_rgenRegion.size + m_missRegion.size + m_hitRegion.size + m_callRegion.size };
	// TODO: Not sure if memory in host is best option
	m_vulkanDevice->createBuffer(sbtSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR,
	                          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
	                          m_SBTBuffer, m_SBTBufferMemory);

	// Find SBT addresses of each group
	VkDeviceAddress sbtAddress{ VulkanUtils::getBufferDeviceAddress(m_vulkanDevice->logicalDevice, m_SBTBuffer) };
	m_rgenRegion.deviceAddress = sbtAddress;
	m_missRegion.deviceAddress = sbtAddress + m_rgenRegion.size;
	m_hitRegion.deviceAddress = sbtAddress + m_rgenRegion.size + m_missRegion.size;

	auto getHandlePtr = [&](int i) { return handles.data() + i * handleSize; };

	// Map SBT Buffer and write in the handles
	void *pSBTBuffer{};
	vkMapMemory(m_vulkanDevice->logicalDevice, m_SBTBufferMemory, 0, dataSize, 0, &pSBTBuffer);
	uint8_t *pData{ reinterpret_cast<uint8_t *>(pSBTBuffer) };
	uint32_t handleIdx{ 0 };

	// Raygen
	memcpy(pData, getHandlePtr(handleIdx++), handleSize);
	// Miss
	pData += m_rgenRegion.size;
	for (uint32_t i{ 0 }; i < missCount; ++i)
	{
		memcpy(pData, getHandlePtr(handleIdx++), handleSize);
		pData += m_missRegion.stride;
	}
	// Hit
	pData = reinterpret_cast<uint8_t *>(pSBTBuffer) + m_rgenRegion.size + m_missRegion.size;
	for (uint32_t i{ 0 }; i < hitCount; ++i)
	{
		memcpy(pData, getHandlePtr(handleIdx++), handleSize);
		pData += m_hitRegion.stride;
	}

	// Unmap memory
	vkUnmapMemory(m_vulkanDevice->logicalDevice, m_SBTBufferMemory);
}

void VulkanRaytracing::raytrace(VkCommandBuffer commandBuffer, uint32_t currentFrame,
                                const std::vector<VkDescriptorSet>& rasterDescriptorSets,
                                uint32_t width, uint32_t height)
{
	updateFrame();

	// If max sampling has reached, output should be good enough hence stop accumulating further samples.
	// This helps to reduce GPU usage too.
	// Note that total samples = VulkanUtils::maxRTFrames * MAX_SAMPLES (in rgen shader)
	static bool printOnce{ true };
	if (m_pcRay.frame >= VulkanUtils::maxRTFrames)
	{
		if (printOnce)
		{
			std::cout << "Max frames reached, ray tracing terminated!" << '\n';
			printOnce = false;
		}
		return;
	}

	std::vector<VkDescriptorSet> descriptorSets{ m_descriptorSets[currentFrame],
		                                         rasterDescriptorSets[currentFrame] };

	vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_pipeline);
	vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_pipelineLayout, 0,
	                        static_cast<uint32_t>(descriptorSets.size()), descriptorSets.data(), 0, nullptr);
	vkCmdPushConstants(commandBuffer, m_pipelineLayout,
	                   VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR,
	                   0, sizeof(VulkanRaytracing::PushConstantRay), &m_pcRay);

	PFN_vkCmdTraceRaysKHR vkCmdTraceRaysKHR = reinterpret_cast<PFN_vkCmdTraceRaysKHR>(vkGetDeviceProcAddr(m_vulkanDevice->logicalDevice, "vkCmdTraceRaysKHR"));
	vkCmdTraceRaysKHR(commandBuffer, &m_rgenRegion, &m_missRegion, &m_hitRegion, &m_callRegion,
	                  width, height, 1);
}

// Resets the frame counter if the camera has changed
void VulkanRaytracing::updateFrame()
{
	// TODO: Add resetRtFrame for when camera changes
	m_pcRay.frame++;
}

void VulkanRaytracing::resetFrame()
{
	m_pcRay.frame = -1;
}

void VulkanRaytracing::cleanup()
{
	for (const auto &blas : m_blases)
	{
		vkDestroyAccelerationStructureKHR(m_vulkanDevice->logicalDevice, blas.as, nullptr);
		vkDestroyBuffer(m_vulkanDevice->logicalDevice, blas.buffer, nullptr);
		vkFreeMemory(m_vulkanDevice->logicalDevice, blas.bufferMemory, nullptr);
	}

	vkDestroyAccelerationStructureKHR(m_vulkanDevice->logicalDevice, m_tlas.as, nullptr);
	vkDestroyBuffer(m_vulkanDevice->logicalDevice, m_tlas.buffer, nullptr);
	vkFreeMemory(m_vulkanDevice->logicalDevice, m_tlas.bufferMemory, nullptr);

	vkDestroyDescriptorPool(m_vulkanDevice->logicalDevice, m_descriptorPool, nullptr);
	vkDestroyDescriptorSetLayout(m_vulkanDevice->logicalDevice, m_descriptorSetLayout, nullptr);

	vkDestroyPipeline(m_vulkanDevice->logicalDevice, m_pipeline, nullptr);
	vkDestroyPipelineLayout(m_vulkanDevice->logicalDevice, m_pipelineLayout, nullptr);
	vkDestroyBuffer(m_vulkanDevice->logicalDevice, m_SBTBuffer, nullptr);
	vkFreeMemory(m_vulkanDevice->logicalDevice, m_SBTBufferMemory, nullptr);
}