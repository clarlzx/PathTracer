#include <GLFW/glfw3.h>
#include <vulkan/vulkan.h>

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/hash.hpp>

#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "VulkanDebug.h"
#include "VulkanDevice.h"
#include "VulkanRaytracing.h"
#include "VulkanSwapchain.h"
#include "VulkanUtils.h"

const std::string modelPath{ "models/cornell.obj" };

#ifdef NDEBUG
const bool enableValidationLayers{ false };
#else
const bool enableValidationLayers{ true };
#endif

struct Model
{
	VkDeviceAddress vertexAddress{};
	VkDeviceAddress indexAddress{};
	VkDeviceAddress matAddress{};
	VkDeviceAddress matIndexAddress{};
};

struct Material
{
	glm::vec3 diffuse{};
	glm::vec3 emission{};
};

struct Vertex
{
	glm::vec3 pos{};
	glm::vec3 normal{};
	glm::vec3 color{};
	glm::vec2 texCoord{};

	static VkVertexInputBindingDescription getBindingDescription()
	{
		VkVertexInputBindingDescription bindingDescription{};
		bindingDescription.binding = 0;
		bindingDescription.stride = sizeof(Vertex);
		bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

		return bindingDescription;
	}

	static std::array<VkVertexInputAttributeDescription, 4> getAttributeDescriptions()
	{
		std::array<VkVertexInputAttributeDescription, 4>
		    attributeDescriptions{};

		// For position
		attributeDescriptions[0].binding = 0;
		attributeDescriptions[0].location = 0;
		attributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
		attributeDescriptions[0].offset = offsetof(Vertex, pos);

		// For normal
		attributeDescriptions[1].binding = 0;
		attributeDescriptions[1].location = 1;
		attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
		attributeDescriptions[1].offset = offsetof(Vertex, normal);

		// For color
		attributeDescriptions[3].binding = 0;
		attributeDescriptions[3].location = 2;
		attributeDescriptions[3].format = VK_FORMAT_R32G32B32_SFLOAT;
		attributeDescriptions[3].offset = offsetof(Vertex, color);

		// For texcoord
		attributeDescriptions[2].binding = 0;
		attributeDescriptions[2].location = 3;
		attributeDescriptions[2].format = VK_FORMAT_R32G32_SFLOAT;
		attributeDescriptions[2].offset = offsetof(Vertex, texCoord);

		return attributeDescriptions;
	}

	bool operator==(const Vertex &other) const
	{
		return pos == other.pos && normal == other.normal && color == other.color && texCoord == other.texCoord;
	}
};

template <>
struct std::hash<Vertex>
{
	size_t operator()(const Vertex &vertex) const
	{
		size_t h1{ std::hash<glm::vec3>()(vertex.pos) ^ (std::hash<glm::vec3>()(vertex.normal) << 1) };
		size_t h2{ (h1 >> 1) ^ (std::hash<glm::vec3>()(vertex.color) << 1) };
		return (h2 >> 1) ^ (std::hash<glm::vec2>()(vertex.texCoord) << 1);
	}
};

struct UniformBufferObject
{
	alignas(16) glm::mat4 model;
	alignas(16) glm::mat4 view;
	alignas(16) glm::mat4 proj;
};

class PathTracer
{
public:
	void run()
	{
		initWindow();
		initVulkan();
		mainLoop();
		cleanup();
	}

private:
	// For updating FPS
	uint32_t fpsCounter{ 0 };
	std::chrono::time_point<std::chrono::high_resolution_clock> lastTimestamp{};

	GLFWwindow *m_window{};
	VkInstance m_instance{};

	const std::vector<const char *> m_validationLayers{
		"VK_LAYER_KHRONOS_validation",
	};

	const std::vector<const char *> m_deviceExtensions{
		VK_KHR_SWAPCHAIN_EXTENSION_NAME,
		// Required to build acceleration structures
		VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
		// Required to use vkCmdTraceRaysKHR
		VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME,
		// Required by ray tracing pipelines
		VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
		VK_KHR_SHADER_CLOCK_EXTENSION_NAME
	};

	VkPhysicalDeviceFeatures m_deviceFeatures{};

	VulkanDevice vulkanDevice{};
	VkPhysicalDevice m_physicalDevice{};
	VkDevice m_logicalDevice{};

	VkQueue m_graphicsQueue{};
	VkQueue m_presentQueue{};

	VulkanSwapchain vulkanSwapchain{};
	std::vector<VkFramebuffer> m_swapchainFramebuffers{};

	VulkanRaytracing vulkanRT{};

	std::vector<VkSemaphore> m_imageAvailableSemaphores{};
	std::vector<VkSemaphore> m_renderFinishedSemaphores{};
	std::vector<VkFence> m_inFlightFences{};

	std::vector<Vertex> m_vertices{};
	VkBuffer m_vertexBuffer{};
	VkDeviceMemory m_vertexBufferMemory{};

	std::vector<uint32_t> m_indices{};
	VkBuffer m_indexBuffer{};
	VkDeviceMemory m_indexBufferMemory{};

	std::vector<Material> m_materials{};
	VkBuffer m_matBuffer{};
	VkDeviceMemory m_matBufferMemory{};

	std::vector<uint32_t> m_matIndices{};
	VkBuffer m_matIndexBuffer{};
	VkDeviceMemory m_matIndexBufferMemory{};

	Model m_model{};
	VkBuffer m_modelBuffer{};
	VkDeviceMemory m_modelBufferMemory{};

	std::vector<VkBuffer> m_uniformBuffers{};
	std::vector<VkDeviceMemory> m_uniformBuffersMemory{};
	std::vector<void *> m_uniformBuffersMapped{};

	VkDescriptorSetLayout m_descriptorSetLayout{};
	VkDescriptorPool m_descriptorPool{};
	std::vector<VkDescriptorSet> m_descriptorSets{};

	bool m_resized{ false };
	uint32_t m_currentFrame{ 0 };

	// Offscreen image
	VkImage m_depthImage{};
	VkDeviceMemory m_depthImageMemory{};
	VkImageView m_depthImageView{};

	VkImage m_colorImage{};
	VkDeviceMemory m_colorImageMemory{};
	VkImageView m_colorImageView{};

	VkSampler m_offscreenResultSampler{};

	// Post renderpass resources
	VkImage m_postDepthImage{};
	VkDeviceMemory m_postDepthImageMemory{};
	VkImageView m_postDepthImageView{};

	VkRenderPass m_postRenderPass{};
	VkDescriptorSetLayout m_postDescriptorSetLayout{};
	VkDescriptorPool m_postDescriptorPool{};
	std::vector<VkDescriptorSet> m_postDescriptorSets{};
	VkPipelineLayout m_postPipelineLayout{};
	VkPipeline m_postGraphicsPipeline{};

	void initWindow()
	{
		glfwInit();

		// Tell GLFW to not create OpenGL context
		glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

		m_window = glfwCreateWindow(VulkanUtils::width, VulkanUtils::height, "Vulkan", nullptr, nullptr);
		glfwSetWindowUserPointer(m_window, this);
		glfwSetFramebufferSizeCallback(m_window, framebufferResizeCallback);
	}

	static void framebufferResizeCallback(GLFWwindow *window,
	                                      [[maybe_unused]] int width,
	                                      [[maybe_unused]] int height)
	{
		auto app = reinterpret_cast<PathTracer *>(glfwGetWindowUserPointer(window));
		app->m_resized = true;
	}

	void initVulkan()
	{
		createInstance();

		if (enableValidationLayers)
		{
			vk_debug::setupDebugMessenger(m_instance);
		}

		vulkanSwapchain.createSurface(m_instance, m_window);
		setDeviceFeatures();
		m_physicalDevice = vulkanDevice.pickPhysicalDevice(m_instance, vulkanSwapchain.m_surface, m_deviceExtensions, m_deviceFeatures);
		 
		vulkanRT.setContext(&vulkanDevice);
		void *deviceCreatepNextChain{ vulkanRT.getEnabledFeatures() };
		m_logicalDevice = vulkanDevice.createLogicalDevice(enableValidationLayers, m_validationLayers, m_deviceExtensions, m_deviceFeatures, deviceCreatepNextChain);
		
		createQueues();

		vulkanSwapchain.setContext(m_instance, &vulkanDevice, m_window);
		vulkanSwapchain.createSwapchain();

		vulkanDevice.createCommandPool();

		createOffscreenResultSampler();
		createColorResources();
		createDepthResources();

		loadModel();
		createModelBuffers();

		createUniformBuffers();

		createDescriptorSetLayout();
		createDescriptorPool();
		createDescriptorSets();

		createPostRenderPass();
		createPostDepthResources();
		createPostFramebuffers();

		createPostDescriptorSetLayout();
		createPostDescriptorPool();
		createPostDescriptorSets();
		createPostGraphicsPipeline();

		vulkanDevice.createCommandBuffers(VulkanUtils::maxFramesInFlight);
		createSyncObjects();

		vulkanRT.getPipelineProperties();
		vulkanRT.buildBlas(m_graphicsQueue, static_cast<uint32_t>(m_vertices.size()), static_cast<uint32_t>(m_indices.size()),
		                   m_model.vertexAddress, m_model.indexAddress, sizeof(Vertex));
		vulkanRT.buildTlas(m_graphicsQueue);

		vulkanRT.createDescriptorSetLayout();
		vulkanRT.createDescriptorPool();
		vulkanRT.createDescriptorSets(m_colorImageView);
		vulkanRT.createPipeline(m_descriptorSetLayout);
		vulkanRT.createShaderBindingTable();
	}

	void mainLoop()
	{
		while (!glfwWindowShouldClose(m_window))
		{
			glfwPollEvents();
			drawFrame();
		}
		// Wait for logical device to finish operations before exiting mainLoop()
		vkDeviceWaitIdle(m_logicalDevice);
	}

	void cleanup()
	{
		vulkanSwapchain.cleanup();

		vkDestroyImageView(m_logicalDevice, m_colorImageView, nullptr);
		vkDestroyImage(m_logicalDevice, m_colorImage, nullptr);
		vkFreeMemory(m_logicalDevice, m_colorImageMemory, nullptr);

		vkDestroySampler(m_logicalDevice, m_offscreenResultSampler, nullptr);

		vkDestroyImageView(m_logicalDevice, m_depthImageView, nullptr);
		vkDestroyImage(m_logicalDevice, m_depthImage, nullptr);
		vkFreeMemory(m_logicalDevice, m_depthImageMemory, nullptr);

		vkDestroyImageView(m_logicalDevice, m_postDepthImageView, nullptr);
		vkDestroyImage(m_logicalDevice, m_postDepthImage, nullptr);
		vkFreeMemory(m_logicalDevice, m_postDepthImageMemory, nullptr);

		for (auto framebuffer : m_swapchainFramebuffers)
		{
			vkDestroyFramebuffer(m_logicalDevice, framebuffer, nullptr);
		}

		vkDestroyDescriptorPool(m_logicalDevice, m_descriptorPool, nullptr);
		vkDestroyDescriptorSetLayout(m_logicalDevice, m_descriptorSetLayout, nullptr);

		for (size_t i{ 0 }; i < VulkanUtils::maxFramesInFlight; ++i)
		{
			vkDestroyBuffer(m_logicalDevice, m_uniformBuffers[i], nullptr);
			vkFreeMemory(m_logicalDevice, m_uniformBuffersMemory[i], nullptr);
		}

		vkDestroyBuffer(m_logicalDevice, m_modelBuffer, nullptr);
		vkFreeMemory(m_logicalDevice, m_modelBufferMemory, nullptr);

		vkDestroyBuffer(m_logicalDevice, m_vertexBuffer, nullptr);
		vkFreeMemory(m_logicalDevice, m_vertexBufferMemory, nullptr);

		vkDestroyBuffer(m_logicalDevice, m_indexBuffer, nullptr);
		vkFreeMemory(m_logicalDevice, m_indexBufferMemory, nullptr);

		vkDestroyBuffer(m_logicalDevice, m_matBuffer, nullptr);
		vkFreeMemory(m_logicalDevice, m_matBufferMemory, nullptr);

		vkDestroyBuffer(m_logicalDevice, m_matIndexBuffer, nullptr);
		vkFreeMemory(m_logicalDevice, m_matIndexBufferMemory, nullptr);

		vulkanRT.cleanup();

		vkDestroyPipeline(m_logicalDevice, m_postGraphicsPipeline, nullptr);
		vkDestroyPipelineLayout(m_logicalDevice, m_postPipelineLayout, nullptr);
		vkDestroyRenderPass(m_logicalDevice, m_postRenderPass, nullptr);
		vkDestroyDescriptorPool(m_logicalDevice, m_postDescriptorPool, nullptr);
		vkDestroyDescriptorSetLayout(m_logicalDevice, m_postDescriptorSetLayout, nullptr);

		for (size_t i{ 0 }; i < VulkanUtils::maxFramesInFlight; ++i)
		{
			vkDestroySemaphore(m_logicalDevice, m_imageAvailableSemaphores[i],
			                   nullptr);
			vkDestroySemaphore(m_logicalDevice, m_renderFinishedSemaphores[i],
			                   nullptr);
			vkDestroyFence(m_logicalDevice, m_inFlightFences[i], nullptr);
		}

		vulkanDevice.cleanup();

		if (enableValidationLayers)
		{
			vk_debug::freeDebugMessenger(m_instance);
		}

		vkDestroyInstance(m_instance, nullptr);

		glfwDestroyWindow(m_window);

		glfwTerminate();
	}

	void recreateSwapChain()
	{
		int width{};
		int height{};

		glfwGetFramebufferSize(m_window, &width, &height); // If window is not minimized, nothing to wait on
		while (width == 0 || height == 0)
		{
			// If window is minimized such that framebuffer size is 0, pause until window is in foreground again
			glfwGetFramebufferSize(m_window, &width, &height);
			glfwWaitEvents();
		}

		vkDeviceWaitIdle(m_logicalDevice);

		vulkanSwapchain.createSwapchain();

		createColorResources();
		createDepthResources();

		createPostDepthResources();
		createPostFramebuffers();
		updatePostDescriptorSets();

		vulkanRT.updateDescriptorSets(m_colorImageView);

		vulkanRT.resetFrame();
	}

	void createInstance()
	{
		if (enableValidationLayers && !checkValidationLayerSupport())
		{
			throw std::runtime_error("validation layers requested, but not available!");
		}

		VkApplicationInfo appInfo{};
		appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
		appInfo.pApplicationName = "Hello Triangle";
		appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
		appInfo.pEngineName = "No Engine";
		appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
		appInfo.apiVersion = VK_API_VERSION_1_2;

		VkInstanceCreateInfo createInfo{};
		createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
		createInfo.pApplicationInfo = &appInfo;

		auto extensions{ getRequiredExtensions() };
		createInfo.enabledExtensionCount =
		    static_cast<uint32_t>(extensions.size());
		createInfo.ppEnabledExtensionNames = extensions.data();

		VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo{};
		if (enableValidationLayers)
		{
			createInfo.enabledLayerCount = static_cast<uint32_t>(m_validationLayers.size());
			createInfo.ppEnabledLayerNames = m_validationLayers.data();

			vk_debug::populateDebugMessengerCreateInfo(debugCreateInfo);
			createInfo.pNext =
			    (VkDebugUtilsMessengerCreateInfoEXT *) &debugCreateInfo;
		}
		else
		{
			createInfo.enabledLayerCount = 0;

			createInfo.pNext = nullptr;
		}

		if (vkCreateInstance(&createInfo, nullptr, &m_instance) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to create instance!");
		}
	}

	void setDeviceFeatures()
	{
		m_deviceFeatures.samplerAnisotropy = VK_TRUE;
		m_deviceFeatures.sampleRateShading = VK_TRUE;
		m_deviceFeatures.shaderInt64 = VK_TRUE;
	}

	VkImageView createImageView(VkImage image, VkFormat format, VkImageAspectFlags aspectFlags, uint32_t mipLevels)
	{
		VkImageViewCreateInfo viewInfo{};
		viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		viewInfo.image = image;
		viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
		viewInfo.format = format;
		viewInfo.subresourceRange.aspectMask = aspectFlags;
		viewInfo.subresourceRange.baseMipLevel = 0;
		viewInfo.subresourceRange.levelCount = mipLevels;
		viewInfo.subresourceRange.baseArrayLayer = 0;
		viewInfo.subresourceRange.layerCount = 1;

		VkImageView imageView;
		if (vkCreateImageView(m_logicalDevice, &viewInfo, nullptr,
		                      &imageView) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to create image view!");
		}

		return imageView;
	}

	void createQueues()
	{
		vkGetDeviceQueue(m_logicalDevice, vulkanDevice.queueFamilyIndices.graphics.value(), 0, &m_graphicsQueue);
		vkGetDeviceQueue(m_logicalDevice, vulkanDevice.queueFamilyIndices.present.value(), 0, &m_presentQueue);
	}

	void createPostRenderPass()
	{
		VkAttachmentDescription colorAttachment{};
		colorAttachment.format = vulkanSwapchain.m_format;
		colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
		colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

		VkAttachmentDescription depthAttachment{};
		depthAttachment.format = vulkanDevice.depthFormat;
		depthAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
		depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		depthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		depthAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		depthAttachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

		VkAttachmentReference colorAttachmentRef{};
		colorAttachmentRef.attachment = 0;
		colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

		VkAttachmentReference depthAttachmentRef{};
		depthAttachmentRef.attachment = 1;
		depthAttachmentRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

		VkSubpassDescription subpass{};
		subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
		subpass.colorAttachmentCount = 1;
		subpass.pColorAttachments = &colorAttachmentRef;
		subpass.pDepthStencilAttachment = &depthAttachmentRef;

		std::array<VkSubpassDependency, 2> dependencies{};

		dependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
		dependencies[0].dstSubpass = 0;
		dependencies[0].srcStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
		dependencies[0].dstStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
		dependencies[0].srcAccessMask = 0;
		dependencies[0].dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

		dependencies[1].srcSubpass = VK_SUBPASS_EXTERNAL;
		dependencies[1].dstSubpass = 0;
		dependencies[1].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		dependencies[1].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		dependencies[1].srcAccessMask = 0;
		dependencies[1].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

		std::array<VkAttachmentDescription, 2> attachments{ colorAttachment, depthAttachment };

		VkRenderPassCreateInfo renderPassInfo{};
		renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
		renderPassInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
		renderPassInfo.pAttachments = attachments.data();
		renderPassInfo.subpassCount = 1;
		renderPassInfo.pSubpasses = &subpass;
		renderPassInfo.dependencyCount = static_cast<uint32_t>(dependencies.size());
		renderPassInfo.pDependencies = dependencies.data();

		if (vkCreateRenderPass(m_logicalDevice, &renderPassInfo, nullptr, &m_postRenderPass) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to create render pass!");
		}
	}

	void createDescriptorSetLayout()
	{
		VkDescriptorSetLayoutBinding uboLayoutBinding{};
		uboLayoutBinding.binding = 0;
		uboLayoutBinding.descriptorCount = 1;
		uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		uboLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_RAYGEN_BIT_KHR;

		VkDescriptorSetLayoutBinding modelLayoutBinding{};
		modelLayoutBinding.binding = 1;
		modelLayoutBinding.descriptorCount = 1;
		modelLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		modelLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;

		std::array<VkDescriptorSetLayoutBinding, 2> bindings{ uboLayoutBinding,
			                                                  modelLayoutBinding };

		VkDescriptorSetLayoutCreateInfo layoutInfo{};
		layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
		layoutInfo.pBindings = bindings.data();

		if (vkCreateDescriptorSetLayout(m_logicalDevice, &layoutInfo, nullptr,
		                                &m_descriptorSetLayout) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to create descriptor set layout!");
		}
	}

	void createPostDescriptorSetLayout()
	{
		VkDescriptorSetLayoutBinding samplerLayoutBinding{};
		samplerLayoutBinding.binding = 0;
		samplerLayoutBinding.descriptorCount = 1;
		samplerLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		samplerLayoutBinding.pImmutableSamplers = nullptr;
		samplerLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

		VkDescriptorSetLayoutCreateInfo layoutInfo{};
		layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		layoutInfo.bindingCount = 1;
		layoutInfo.pBindings = &samplerLayoutBinding;

		if (vkCreateDescriptorSetLayout(m_logicalDevice, &layoutInfo, nullptr, &m_postDescriptorSetLayout) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to create post descriptor set layout!");
		}
	}

	void createPostDescriptorPool()
	{
		VkDescriptorPoolSize descriptorPoolSize{};
		descriptorPoolSize.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		descriptorPoolSize.descriptorCount = static_cast<uint32_t>(VulkanUtils::maxFramesInFlight);

		VkDescriptorPoolCreateInfo poolInfo{};
		poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
		poolInfo.poolSizeCount = 1;
		poolInfo.pPoolSizes = &descriptorPoolSize;
		poolInfo.maxSets = static_cast<uint32_t>(VulkanUtils::maxFramesInFlight);

		if (vkCreateDescriptorPool(m_logicalDevice, &poolInfo, nullptr, &m_postDescriptorPool) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to create post descriptor pool!");
		}
	}

	void createPostDescriptorSets()
	{
		std::vector<VkDescriptorSetLayout> layouts{ VulkanUtils::maxFramesInFlight, m_postDescriptorSetLayout };

		VkDescriptorSetAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
		allocInfo.descriptorPool = m_postDescriptorPool;
		allocInfo.descriptorSetCount = static_cast<uint32_t>(VulkanUtils::maxFramesInFlight);
		allocInfo.pSetLayouts = layouts.data();

		m_postDescriptorSets.resize(VulkanUtils::maxFramesInFlight);
		if (vkAllocateDescriptorSets(m_logicalDevice, &allocInfo, m_postDescriptorSets.data()) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to allocate post descriptor sets!");
		}

		// Configure descriptors in descriptor sets
		updatePostDescriptorSets();
	}

	void updatePostDescriptorSets()
	{
		for (size_t i{ 0 }; i < VulkanUtils::maxFramesInFlight; ++i)
		{
			VkDescriptorImageInfo imageInfo{};
			imageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
			imageInfo.imageView = m_colorImageView;
			imageInfo.sampler = m_offscreenResultSampler;

			VkWriteDescriptorSet descriptorWrite{};
			descriptorWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			descriptorWrite.dstSet = m_postDescriptorSets[i];
			descriptorWrite.dstBinding = 0;
			descriptorWrite.dstArrayElement = 0;
			descriptorWrite.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
			descriptorWrite.descriptorCount = 1;
			descriptorWrite.pImageInfo = &imageInfo;

			vkUpdateDescriptorSets(m_logicalDevice, 1, &descriptorWrite, 0, nullptr);
		}
	}

	void createPostGraphicsPipeline()
	{
		auto vertShaderCode{ VulkanUtils::readFile("spv/post.vert.spv") };
		auto fragShaderCode{ VulkanUtils::readFile("spv/post.frag.spv") };

		VkShaderModule vertShaderModule{ VulkanUtils::createShaderModule(m_logicalDevice, vertShaderCode) };
		VkShaderModule fragShaderModule{ VulkanUtils::createShaderModule(m_logicalDevice, fragShaderCode) };

		VkPipelineShaderStageCreateInfo vertShaderStageInfo{};
		vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
		vertShaderStageInfo.module = vertShaderModule;
		vertShaderStageInfo.pName = "main";

		VkPipelineShaderStageCreateInfo fragShaderStageInfo{};
		fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
		fragShaderStageInfo.module = fragShaderModule;
		fragShaderStageInfo.pName = "main";

		VkPipelineShaderStageCreateInfo shaderStages[]{ vertShaderStageInfo,
			                                            fragShaderStageInfo };

		// Vertex Input
		// Is an empty vertex input state, since we are not passing any vertices to the shaders
		VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
		vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
		vertexInputInfo.vertexBindingDescriptionCount = 0;
		vertexInputInfo.vertexAttributeDescriptionCount = 0;

		VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
		inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
		inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
		inputAssembly.primitiveRestartEnable = VK_FALSE;

		// Viewport
		VkPipelineViewportStateCreateInfo viewportState{};
		viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
		viewportState.viewportCount = 1;
		viewportState.scissorCount = 1;

		// Rasterizer
		VkPipelineRasterizationStateCreateInfo rasterizer{};
		rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
		rasterizer.depthClampEnable = VK_FALSE;
		rasterizer.rasterizerDiscardEnable = VK_FALSE;
		rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
		rasterizer.lineWidth = 1.0f;
		rasterizer.cullMode = VK_CULL_MODE_NONE;
		rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;
		rasterizer.depthBiasEnable = VK_FALSE;

		// Multisampling (anti-aliasing)
		VkPipelineMultisampleStateCreateInfo multisampling{};
		multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
		multisampling.sampleShadingEnable = VK_FALSE;
		multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

		// Depth and stencil testing
		VkPipelineDepthStencilStateCreateInfo depthStencil{};
		depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
		depthStencil.depthTestEnable = VK_TRUE;
		depthStencil.depthWriteEnable = VK_TRUE;
		depthStencil.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
		depthStencil.depthBoundsTestEnable = VK_FALSE; // Depth bound test that allows you to keep fragments that fall in specified depth range
		depthStencil.stencilTestEnable = VK_FALSE;

		// Color blending
		VkPipelineColorBlendAttachmentState colorBlendAttachment{};
		colorBlendAttachment.colorWriteMask =
		    VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
		    VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
		colorBlendAttachment.blendEnable = VK_FALSE; // Set to false, since only have one framebuffer

		VkPipelineColorBlendStateCreateInfo colorBlending{};
		colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
		colorBlending.logicOpEnable = VK_FALSE;
		colorBlending.logicOp = VK_LOGIC_OP_COPY;
		colorBlending.attachmentCount = 1;
		colorBlending.pAttachments = &colorBlendAttachment;
		colorBlending.blendConstants[0] = 0.0f;
		colorBlending.blendConstants[1] = 0.0f;
		colorBlending.blendConstants[2] = 0.0f;
		colorBlending.blendConstants[3] = 0.0f;

		// Dynamic state for dynamic viewport, scissor
		std::vector<VkDynamicState> dynamicStates{ VK_DYNAMIC_STATE_VIEWPORT,
			                                       VK_DYNAMIC_STATE_SCISSOR };

		VkPipelineDynamicStateCreateInfo dynamicState{};
		dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
		dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
		dynamicState.pDynamicStates = dynamicStates.data();

		VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
		pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		pipelineLayoutInfo.setLayoutCount = 1;
		pipelineLayoutInfo.pSetLayouts = &m_postDescriptorSetLayout;

		if (vkCreatePipelineLayout(m_logicalDevice, &pipelineLayoutInfo, nullptr, &m_postPipelineLayout) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to create post pipeline layout!");
		}

		// Create graphics pipeline
		VkGraphicsPipelineCreateInfo graphicsPipelineInfo{};
		graphicsPipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
		graphicsPipelineInfo.stageCount = 2;
		graphicsPipelineInfo.pStages = shaderStages;
		graphicsPipelineInfo.pVertexInputState = &vertexInputInfo;
		graphicsPipelineInfo.pInputAssemblyState = &inputAssembly;
		graphicsPipelineInfo.pViewportState = &viewportState;
		graphicsPipelineInfo.pRasterizationState = &rasterizer;
		graphicsPipelineInfo.pMultisampleState = &multisampling;
		graphicsPipelineInfo.pDepthStencilState = &depthStencil;
		graphicsPipelineInfo.pColorBlendState = &colorBlending;
		graphicsPipelineInfo.pDynamicState = &dynamicState;

		graphicsPipelineInfo.layout = m_postPipelineLayout;

		graphicsPipelineInfo.renderPass = m_postRenderPass;
		graphicsPipelineInfo.subpass = 0;

		graphicsPipelineInfo.basePipelineHandle = VK_NULL_HANDLE;

		if (vkCreateGraphicsPipelines(m_logicalDevice, VK_NULL_HANDLE, 1, &graphicsPipelineInfo,
		                              nullptr, &m_postGraphicsPipeline) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to create post graphics pipeline!");
		}

		vkDestroyShaderModule(m_logicalDevice, fragShaderModule, nullptr);
		vkDestroyShaderModule(m_logicalDevice, vertShaderModule, nullptr);
	}

	void createPostFramebuffers()
	{
		if (m_swapchainFramebuffers.empty())
		{
			m_swapchainFramebuffers.resize(vulkanSwapchain.m_imageViews.size());
		}
		else
		{
			// Release resources if buffers need to be recreated
			for (auto framebuffer : m_swapchainFramebuffers)
			{
				vkDestroyFramebuffer(m_logicalDevice, framebuffer, nullptr);
			}
		}

		for (size_t i{ 0 }; i < vulkanSwapchain.m_imageViews.size(); ++i)
		{
			std::array<VkImageView, 2> attachments{
				vulkanSwapchain.m_imageViews[i], m_postDepthImageView
			};

			VkFramebufferCreateInfo framebufferInfo{};
			framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
			framebufferInfo.renderPass = m_postRenderPass;
			framebufferInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
			framebufferInfo.pAttachments = attachments.data();
			framebufferInfo.width = vulkanSwapchain.m_extent.width;
			framebufferInfo.height = vulkanSwapchain.m_extent.height;
			framebufferInfo.layers = 1;

			if (vkCreateFramebuffer(m_logicalDevice, &framebufferInfo, nullptr, &m_swapchainFramebuffers[i]) != VK_SUCCESS)
			{
				throw std::runtime_error("failed to create framebuffer!");
			}
		}
	}

	void createPostDepthResources()
	{
		// Release resources if image needs to be recreated
		if (m_postDepthImage != VK_NULL_HANDLE)
		{
			vkDestroyImageView(m_logicalDevice, m_postDepthImageView, nullptr);
			vkDestroyImage(m_logicalDevice, m_postDepthImage, nullptr);
			vkFreeMemory(m_logicalDevice, m_postDepthImageMemory, nullptr);
		}

		createImage(vulkanSwapchain.m_extent.width, vulkanSwapchain.m_extent.height, 1,
		            VK_SAMPLE_COUNT_1_BIT, vulkanDevice.depthFormat, VK_IMAGE_TILING_OPTIMAL,
		            VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
		            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, m_postDepthImage,
		            m_postDepthImageMemory);
		m_postDepthImageView = createImageView(m_postDepthImage, vulkanDevice.depthFormat, VK_IMAGE_ASPECT_DEPTH_BIT, 1);
	}

	bool hasStencilComponent(VkFormat format)
	{
		return format == VK_FORMAT_D32_SFLOAT_S8_UINT ||
		       format == VK_FORMAT_D24_UNORM_S8_UINT ||
		       format == VK_FORMAT_D16_UNORM_S8_UINT;
	}

	void createOffscreenResultSampler()
	{
		// Create sampler
		VkSamplerCreateInfo samplerInfo{};
		samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;

		if (vkCreateSampler(m_logicalDevice, &samplerInfo, nullptr, &m_offscreenResultSampler) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to create offscreen color sampler!");
		}
	}

	void createColorResources()
	{
		// Release resources if image needs to be recreated
		if (m_colorImage != VK_NULL_HANDLE)
		{
			vkDestroyImageView(m_logicalDevice, m_colorImageView, nullptr);
			vkDestroyImage(m_logicalDevice, m_colorImage, nullptr);
			vkFreeMemory(m_logicalDevice, m_colorImageMemory, nullptr);
		}

		VkFormat offscreenColorFormat{ VK_FORMAT_R32G32B32A32_SFLOAT };

		// Create image to be multisampled
		createImage(vulkanSwapchain.m_extent.width, vulkanSwapchain.m_extent.height, 1,
		            VK_SAMPLE_COUNT_1_BIT, offscreenColorFormat, VK_IMAGE_TILING_OPTIMAL,
		            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT,
		            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, m_colorImage, m_colorImageMemory);

		m_colorImageView = createImageView(m_colorImage, offscreenColorFormat, VK_IMAGE_ASPECT_COLOR_BIT, 1);

		transitionImageLayout(m_colorImage, offscreenColorFormat,
		                      VK_IMAGE_LAYOUT_UNDEFINED,
		                      VK_IMAGE_LAYOUT_GENERAL, 1);
	}

	void createDepthResources()
	{
		// Release resources if image needs to be recreated
		if (m_depthImage != VK_NULL_HANDLE)
		{
			vkDestroyImageView(m_logicalDevice, m_depthImageView, nullptr);
			vkDestroyImage(m_logicalDevice, m_depthImage, nullptr);
			vkFreeMemory(m_logicalDevice, m_depthImageMemory, nullptr);
		}

		createImage(vulkanSwapchain.m_extent.width, vulkanSwapchain.m_extent.height, 1,
		            VK_SAMPLE_COUNT_1_BIT, vulkanDevice.depthFormat, VK_IMAGE_TILING_OPTIMAL,
		            VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
		            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, m_depthImage, m_depthImageMemory);

		m_depthImageView = createImageView(m_depthImage, vulkanDevice.depthFormat, VK_IMAGE_ASPECT_DEPTH_BIT, 1);

		transitionImageLayout(m_depthImage, vulkanDevice.depthFormat,
		                      VK_IMAGE_LAYOUT_UNDEFINED,
		                      VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, 1);
	}

	void createImage(uint32_t width, uint32_t height, uint32_t mipLevels,
	                 VkSampleCountFlagBits numSamples, VkFormat format,
	                 VkImageTiling tiling, VkImageUsageFlags usage,
	                 VkMemoryPropertyFlags properties, VkImage &image,
	                 VkDeviceMemory &imageMemory)
	{
		VkImageCreateInfo imageInfo{};
		imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
		imageInfo.imageType = VK_IMAGE_TYPE_2D;
		imageInfo.extent.width = width;
		imageInfo.extent.height = height;
		imageInfo.extent.depth = 1;
		imageInfo.mipLevels = mipLevels;
		imageInfo.arrayLayers = 1;
		imageInfo.format = format;
		imageInfo.tiling = tiling;
		imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		imageInfo.usage = usage;
		imageInfo.samples = numSamples;
		imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

		if (vkCreateImage(m_logicalDevice, &imageInfo, nullptr, &image) !=
		    VK_SUCCESS)
		{
			throw std::runtime_error("failed to create image!");
		}

		VkMemoryRequirements memRequirememts{};
		vkGetImageMemoryRequirements(m_logicalDevice, image, &memRequirememts);

		VkMemoryAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		allocInfo.allocationSize = memRequirememts.size;
		allocInfo.memoryTypeIndex = vulkanDevice.findMemoryType(memRequirememts.memoryTypeBits, properties);

		if (vkAllocateMemory(m_logicalDevice, &allocInfo, nullptr, &imageMemory) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to allocate image memory!");
		}

		vkBindImageMemory(m_logicalDevice, image, imageMemory, 0);
	}

	void transitionImageLayout(VkImage image, VkFormat format,
	                           VkImageLayout oldLayout, VkImageLayout newLayout,
	                           uint32_t mipLevels)
	{
		VkCommandBuffer commandBuffer{ vulkanDevice.beginSingleTimeCommands() };

		VkImageMemoryBarrier barrier{};
		barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
		barrier.oldLayout = oldLayout;
		barrier.newLayout = newLayout;
		barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.image = image;

		if (newLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
		{
			barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;

			if (hasStencilComponent(format))
			{
				barrier.subresourceRange.aspectMask |= VK_IMAGE_ASPECT_STENCIL_BIT;
			}
		}
		else
		{
			barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		}

		barrier.subresourceRange.baseMipLevel = 0;
		barrier.subresourceRange.levelCount = mipLevels;
		barrier.subresourceRange.baseArrayLayer = 0;
		barrier.subresourceRange.layerCount = 1;

		VkPipelineStageFlags sourceStage{};
		VkPipelineStageFlags destinationStage{};

		switch (oldLayout)
		{
			case VK_IMAGE_LAYOUT_UNDEFINED:
				barrier.srcAccessMask = 0;
				sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
				break;
			case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL:
				barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
				sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
				break;
			default:
				throw std::invalid_argument("unsupported layout transition from this old layout!");
		}

		switch (newLayout)
		{
			case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL:
				barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
				destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
				break;
			case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
				barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
				destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
				break;
			case VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL:
				barrier.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
				destinationStage = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
				break;
			case VK_IMAGE_LAYOUT_GENERAL:
				barrier.dstAccessMask = 0;
				destinationStage = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
				break;
			default:
				throw std::invalid_argument("unsupported layout transition to this new layout!");
		}

		vkCmdPipelineBarrier(commandBuffer, sourceStage, destinationStage, 0, 0,
		                     nullptr, 0, nullptr, 1, &barrier);

		vulkanDevice.endSingleTimeCommands(commandBuffer, m_graphicsQueue);
	}

	void copyBufferToImage(VkBuffer buffer, VkImage image, uint32_t width,
	                       uint32_t height)
	{
		VkCommandBuffer commandBuffer{ vulkanDevice.beginSingleTimeCommands() };

		VkBufferImageCopy region{};
		region.bufferOffset = 0;
		region.bufferRowLength = 0;
		region.bufferImageHeight = 0;

		region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		region.imageSubresource.mipLevel = 0;
		region.imageSubresource.baseArrayLayer = 0;
		region.imageSubresource.layerCount = 1;

		region.imageOffset = { 0, 0, 0 };
		region.imageExtent = { width, height, 1 };

		vkCmdCopyBufferToImage(commandBuffer, buffer, image,
		                       VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1,
		                       &region);

		vulkanDevice.endSingleTimeCommands(commandBuffer, m_graphicsQueue);
	}

	void loadModel()
	{
		tinyobj::attrib_t attrib;
		std::vector<tinyobj::shape_t> shapes;
		std::vector<tinyobj::material_t> materials;
		std::string err{};

		// Get directory to mtl file, assumes that obj and mtl are in the same directory
		std::string mtlDir{};
		size_t pos = modelPath.find_last_of("/\\");
		if (pos != std::string::npos)
		{
			mtlDir = modelPath.substr(0, pos);
		}
		mtlDir += "/";

		if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &err, modelPath.c_str(), mtlDir.c_str()))
		{
			throw std::runtime_error(err);
		}

		for (const auto &material : materials)
		{
			Material matObj{};
			matObj.diffuse = glm::vec3(material.diffuse[0], material.diffuse[1], material.diffuse[2]);
			matObj.emission = glm::vec3(material.emission[0], material.emission[1], material.emission[2]);
			m_materials.emplace_back(matObj);
		}

		const uint32_t matCount{ static_cast<uint32_t>(m_materials.size()) };

		Material grey{};
		grey.diffuse = glm::vec3(0.5);
		grey.emission = glm::vec3(0.0);
		m_materials.emplace_back(grey);

		// Maps a unique vertex to a vertex index
		std::unordered_map<Vertex, uint32_t> uniqueVertices{};

		for (const auto &shape : shapes)
		{
			for (const auto matIdx : shape.mesh.material_ids)
			{
				if (matIdx == -1)
				{
					m_matIndices.push_back(matCount);
				}
				else
				{
					m_matIndices.push_back(matIdx);
				}
			}

			for (const auto &index : shape.mesh.indices)
			{
				Vertex vertex{};

				vertex.pos = { attrib.vertices[3 * index.vertex_index],
					           attrib.vertices[3 * index.vertex_index + 1],
					           attrib.vertices[3 * index.vertex_index + 2] };

				vertex.normal = { attrib.normals[3 * index.normal_index],
					              attrib.normals[3 * index.normal_index + 1],
					              attrib.normals[3 * index.normal_index + 2] };

				// V coord (equiv. to Y-coord for texture coordinate system) needs to be flipped
				// Since OBJ follows same coordinate system as OpenGL their y-axis is the flipped version of Vulkan's y-axis
				if (index.texcoord_index != -1)
				{
					vertex.texCoord = { attrib.texcoords[2 * index.texcoord_index],
						                1.0f - attrib.texcoords[2 * index.texcoord_index + 1] };
				}
				else
				{
					vertex.texCoord = { 0.0f, 0.0f };
				}

				vertex.color = { 1.0f, 1.0f, 1.0f };

				// If vertex is unique, add to m_vertices
				if (uniqueVertices.count(vertex) == 0)
				{
					uniqueVertices[vertex] = static_cast<uint32_t>(m_vertices.size());
					m_vertices.push_back(vertex);
				}

				m_indices.push_back(uniqueVertices[vertex]);
			}
		}
	}

	void createModelBuffers()
	{
		createVertexBuffer();
		createIndexBuffer();
		createMaterialBuffer();
		createMatIndexBuffer();

		m_model.vertexAddress = VulkanUtils::getBufferDeviceAddress(m_logicalDevice, m_vertexBuffer);
		m_model.indexAddress = VulkanUtils::getBufferDeviceAddress(m_logicalDevice, m_indexBuffer);
		m_model.matAddress = VulkanUtils::getBufferDeviceAddress(m_logicalDevice, m_matBuffer);
		m_model.matIndexAddress = VulkanUtils::getBufferDeviceAddress(m_logicalDevice, m_matIndexBuffer);

		VkDeviceSize bufferSize{ sizeof(m_model) };

		VkBufferUsageFlags usageFlags = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;

		createBufferWithStaging<Model>(bufferSize, &m_model, usageFlags, m_modelBuffer, m_modelBufferMemory);
	}

	void createVertexBuffer()
	{
		VkDeviceSize bufferSize{ sizeof(m_vertices[0]) * m_vertices.size() };

		VkBufferUsageFlags usageFlags = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |
		                                VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
		                                VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;

		createBufferWithStaging<Vertex>(bufferSize, m_vertices.data(), usageFlags, m_vertexBuffer, m_vertexBufferMemory);
	}

	void createIndexBuffer()
	{
		VkDeviceSize bufferSize{ sizeof(m_indices[0]) * m_indices.size() };

		VkBufferUsageFlags usageFlags = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
		                                VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
		                                VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;

		createBufferWithStaging<uint32_t>(bufferSize, m_indices.data(), usageFlags, m_indexBuffer, m_indexBufferMemory);
	}

	void createMaterialBuffer()
	{
		VkDeviceSize bufferSize{ sizeof(m_materials[0]) * m_materials.size() };

		VkBufferUsageFlags usageFlags = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
		                                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;

		createBufferWithStaging<Material>(bufferSize, m_materials.data(), usageFlags, m_matBuffer, m_matBufferMemory);
	}

	void createMatIndexBuffer()
	{
		VkDeviceSize bufferSize{ sizeof(m_matIndices[0]) * m_matIndices.size() };

		VkBufferUsageFlags usageFlags = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
		                                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;

		createBufferWithStaging<uint32_t>(bufferSize, m_matIndices.data(), usageFlags, m_matIndexBuffer, m_matIndexBufferMemory);
	}

	template <typename T>
	void createBufferWithStaging(VkDeviceSize bufferSize, T *source, VkBufferUsageFlags usage,
	                             VkBuffer &buffer, VkDeviceMemory &bufferMemory)
	{
		// Create staging buffer
		VkBuffer stagingBuffer{};
		VkDeviceMemory stagingBufferMemory{};
		vulkanDevice.createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
		                          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
		                          stagingBuffer, stagingBufferMemory);

		void *data{};
		// Maps region of memory heap to host (CPU) accessible memory region
		vkMapMemory(m_logicalDevice, stagingBufferMemory, 0, bufferSize, 0, &data);
		memcpy(data, source, static_cast<size_t>(bufferSize));
		// Unmap memory
		vkUnmapMemory(m_logicalDevice, stagingBufferMemory);

		// Create buffer
		vulkanDevice.createBuffer(bufferSize, usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, buffer, bufferMemory);

		vulkanDevice.copyBuffer(stagingBuffer, buffer, bufferSize, m_graphicsQueue);

		// Destroy staging buffer
		vkDestroyBuffer(m_logicalDevice, stagingBuffer, nullptr);
		vkFreeMemory(m_logicalDevice, stagingBufferMemory, nullptr);
	}

	void createUniformBuffers()
	{
		VkDeviceSize bufferSize{ sizeof(UniformBufferObject) };

		m_uniformBuffers.resize(VulkanUtils::maxFramesInFlight);
		m_uniformBuffersMemory.resize(VulkanUtils::maxFramesInFlight);
		m_uniformBuffersMapped.resize(VulkanUtils::maxFramesInFlight);

		for (size_t i{ 0 }; i < VulkanUtils::maxFramesInFlight; i++)
		{
			// Not on device local memory since need to update uniform buffer every frame
			// see updateUniformBuffer()
			vulkanDevice.createBuffer(bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
			             VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
			             m_uniformBuffers[i], m_uniformBuffersMemory[i]);

			// Persistent mapping where buffer stays map to this pointer for memory to increase performance
			// To avoid mapping the buffer every time we need to update the buffer
			vkMapMemory(m_logicalDevice, m_uniformBuffersMemory[i], 0, bufferSize, 0, &m_uniformBuffersMapped[i]);
		}
	}

	void createDescriptorPool()
	{
		std::array<VkDescriptorPoolSize, 2> poolSizes{};
		poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		poolSizes[0].descriptorCount = static_cast<uint32_t>(VulkanUtils::maxFramesInFlight);
		poolSizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		poolSizes[1].descriptorCount = static_cast<uint32_t>(VulkanUtils::maxFramesInFlight);

		VkDescriptorPoolCreateInfo poolInfo{};
		poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
		poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
		poolInfo.pPoolSizes = poolSizes.data();
		poolInfo.maxSets = static_cast<uint32_t>(VulkanUtils::maxFramesInFlight);

		if (vkCreateDescriptorPool(m_logicalDevice, &poolInfo, nullptr, &m_descriptorPool) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to create descriptor pool!");
		}
	}

	void createDescriptorSets()
	{
		std::vector<VkDescriptorSetLayout> layouts(VulkanUtils::maxFramesInFlight, m_descriptorSetLayout);
		VkDescriptorSetAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
		allocInfo.descriptorPool = m_descriptorPool;
		allocInfo.descriptorSetCount = static_cast<uint32_t>(VulkanUtils::maxFramesInFlight);
		allocInfo.pSetLayouts = layouts.data();

		m_descriptorSets.resize(VulkanUtils::maxFramesInFlight);
		if (vkAllocateDescriptorSets(m_logicalDevice, &allocInfo, m_descriptorSets.data()) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to allocate descriptor sets!");
		}

		// Configure descriptors in descriptor sets
		for (size_t i{ 0 }; i < VulkanUtils::maxFramesInFlight; ++i)
		{
			VkDescriptorBufferInfo uniformBufferInfo{};
			uniformBufferInfo.buffer = m_uniformBuffers[i];
			uniformBufferInfo.offset = 0;
			uniformBufferInfo.range = sizeof(UniformBufferObject);

			VkDescriptorBufferInfo modelBufferInfo{};
			modelBufferInfo.buffer = m_modelBuffer;
			modelBufferInfo.offset = 0;
			modelBufferInfo.range = sizeof(Model);

			std::array<VkWriteDescriptorSet, 2> descriptorWrites{};

			descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			descriptorWrites[0].dstSet = m_descriptorSets[i];
			descriptorWrites[0].dstBinding = 0;
			descriptorWrites[0].dstArrayElement = 0;
			descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
			descriptorWrites[0].descriptorCount = 1;
			descriptorWrites[0].pBufferInfo = &uniformBufferInfo;

			descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			descriptorWrites[1].dstSet = m_descriptorSets[i];
			descriptorWrites[1].dstBinding = 1;
			descriptorWrites[1].dstArrayElement = 0;
			descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
			descriptorWrites[1].descriptorCount = 1;
			descriptorWrites[1].pBufferInfo = &modelBufferInfo;

			vkUpdateDescriptorSets(m_logicalDevice, static_cast<uint32_t>(descriptorWrites.size()),
			                       descriptorWrites.data(), 0, nullptr);
		}
	}

	void setViewport(VkCommandBuffer commandBuffer)
	{
		// As viewport, scissor are set as dynamic need to set them in the
		// command buffer before issuing draw command

		VkViewport viewport{};
		viewport.x = 0.0f;
		viewport.y = 0.0f;
		viewport.width = static_cast<float>(vulkanSwapchain.m_extent.width);
		viewport.height = static_cast<float>(vulkanSwapchain.m_extent.height);
		viewport.minDepth = 0.0f;
		viewport.maxDepth = 1.0f;
		vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

		VkRect2D scissor{};
		scissor.offset = { 0, 0 };
		scissor.extent = vulkanSwapchain.m_extent;
		vkCmdSetScissor(commandBuffer, 0, 1, &scissor);
	}

	void recordCommandBuffer(VkCommandBuffer commandBuffer, uint32_t imageIndex)
	{
		VkCommandBufferBeginInfo beginInfo{};
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

		// Begin recording command buffer
		if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to begin recording command buffer!");
		}

		// The order of clearValues should be identical to the order of attachments (i.e. color, depth)
		// in createRenderPass()
		std::array<VkClearValue, 2> clearValues{};
		clearValues[0].color = { { 0.0f, 0.0f, 0.0f, 1.0f } };
		clearValues[1].depthStencil = { 1.0f, 0 }; // depth and stencil clear values

		vulkanRT.raytrace(commandBuffer, m_currentFrame, m_descriptorSets,
		                  vulkanSwapchain.m_extent.width, vulkanSwapchain.m_extent.height);

		// Begin Post Render Pass
		VkRenderPassBeginInfo postRenderPassInfo{};
		postRenderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
		postRenderPassInfo.renderPass = m_postRenderPass;
		postRenderPassInfo.framebuffer = m_swapchainFramebuffers[imageIndex];
		postRenderPassInfo.renderArea.offset = { 0, 0 };
		postRenderPassInfo.renderArea.extent = vulkanSwapchain.m_extent;
		postRenderPassInfo.clearValueCount = 2;
		postRenderPassInfo.pClearValues = clearValues.data();

		vkCmdBeginRenderPass(commandBuffer, &postRenderPassInfo, VK_SUBPASS_CONTENTS_INLINE);
		vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_postGraphicsPipeline);
		setViewport(commandBuffer);
		vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_postPipelineLayout, 0, 1, &m_postDescriptorSets[m_currentFrame], 0, nullptr);
		// Draw 3 vertices for the full screen triangle
		vkCmdDraw(commandBuffer, 3, 1, 0, 0);

		// End Render Pass
		vkCmdEndRenderPass(commandBuffer);

		// End recording the command buffer
		if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to record command buffer!");
		}
	}

	void createSyncObjects()
	{
		m_imageAvailableSemaphores.resize(VulkanUtils::maxFramesInFlight);
		m_renderFinishedSemaphores.resize(VulkanUtils::maxFramesInFlight);
		m_inFlightFences.resize(VulkanUtils::maxFramesInFlight);

		VkSemaphoreCreateInfo semaphoreInfo{};
		semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

		VkFenceCreateInfo fenceInfo{};
		fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
		fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

		for (size_t i{ 0 }; i < VulkanUtils::maxFramesInFlight; ++i)
		{
			if (vkCreateSemaphore(m_logicalDevice, &semaphoreInfo, nullptr, &m_imageAvailableSemaphores[i]) != VK_SUCCESS ||
			    vkCreateSemaphore(m_logicalDevice, &semaphoreInfo, nullptr, &m_renderFinishedSemaphores[i]) != VK_SUCCESS ||
			    vkCreateFence(m_logicalDevice, &fenceInfo, nullptr, &m_inFlightFences[i]) != VK_SUCCESS)
			{
				throw std::runtime_error("failed to create semaphores!");
			}
		}
	}

	void updateUniformBuffer(uint32_t currentImage)
	{
		UniformBufferObject ubo{};
		ubo.model = glm::mat4(1.0f);
		ubo.view = glm::lookAt(glm::vec3(0.0f, 1.0f, 3.5f),
		                       glm::vec3(0.0f, 1.0f, 0.0f),
		                       glm::vec3(0.0f, 1.0f, 0.0f));
		ubo.proj = glm::perspective(glm::radians(45.0f),
		                            vulkanSwapchain.m_extent.width / static_cast<float>(vulkanSwapchain.m_extent.height),
		                            0.1f,
		                            10.0f);
		// Flip y axis
		ubo.proj[1][1] *= -1;

		memcpy(m_uniformBuffersMapped[currentImage], &ubo, sizeof(ubo));
	}

	void drawFrame()
	{
		vkWaitForFences(m_logicalDevice, 1, &m_inFlightFences[m_currentFrame], VK_TRUE, UINT64_MAX);

		uint32_t imageIndex;
		VkResult result = vkAcquireNextImageKHR(m_logicalDevice, vulkanSwapchain.m_swapchain, UINT64_MAX,
		                                        m_imageAvailableSemaphores[m_currentFrame],
		                                        VK_NULL_HANDLE, &imageIndex);

		if (result == VK_ERROR_OUT_OF_DATE_KHR)
		{
			recreateSwapChain();
			return;
		}
		else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR)
		{
			throw std::runtime_error("failed to acquire swap chain image!");
		}

		updateUniformBuffer(m_currentFrame);

		vkResetFences(m_logicalDevice, 1, &m_inFlightFences[m_currentFrame]);

		vkResetCommandBuffer(vulkanDevice.commandBuffers[m_currentFrame], 0);
		recordCommandBuffer(vulkanDevice.commandBuffers[m_currentFrame], imageIndex);

		VkSubmitInfo submitInfo{};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

		VkSemaphore waitSemaphores[]{ m_imageAvailableSemaphores[m_currentFrame] };
		VkSemaphore signalSemaphores[]{ m_renderFinishedSemaphores[m_currentFrame] };
		VkPipelineStageFlags waitStages[]{ VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
		submitInfo.waitSemaphoreCount = 1;
		submitInfo.pWaitSemaphores = waitSemaphores;
		submitInfo.pWaitDstStageMask = waitStages;
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &vulkanDevice.commandBuffers[m_currentFrame];
		submitInfo.signalSemaphoreCount = 1;
		submitInfo.pSignalSemaphores = signalSemaphores;

		if (vkQueueSubmit(m_graphicsQueue, 1, &submitInfo, m_inFlightFences[m_currentFrame]) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to submit draw command buffer!");
		}

		VkPresentInfoKHR presentInfo{};
		presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
		presentInfo.waitSemaphoreCount = 1;
		presentInfo.pWaitSemaphores = signalSemaphores;

		VkSwapchainKHR swapChains[]{ vulkanSwapchain.m_swapchain };
		presentInfo.swapchainCount = 1;
		presentInfo.pSwapchains = swapChains;
		presentInfo.pImageIndices = &imageIndex;

		result = vkQueuePresentKHR(m_presentQueue, &presentInfo);

		if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || m_resized)
		{
			m_resized = false;
			recreateSwapChain();
		}
		else if (result != VK_SUCCESS)
		{
			throw std::runtime_error("failed to present swap chain image!");
		}

		m_currentFrame = (m_currentFrame + 1) % VulkanUtils::maxFramesInFlight;

		// Update FPS
		fpsCounter++;

		auto currentTime{ std::chrono::high_resolution_clock::now() };

		if (lastTimestamp == std::chrono::time_point<std::chrono::high_resolution_clock>())
		{
			lastTimestamp = std::chrono::high_resolution_clock::now();
		}

		auto timeAfterLastTimestamp{ std::chrono::duration<float, std::milli>(currentTime - lastTimestamp).count() };
		if (timeAfterLastTimestamp > 1000.0f)
		{
			glfwSetWindowTitle(m_window, getWindowTitle().c_str());
			lastTimestamp = currentTime;
			fpsCounter = 0;
		}
	}

	std::string getWindowTitle()
	{
		std::string title{ "Vulkan" };
		return title + " - " + std::to_string(fpsCounter) + " fps";
	}

	std::vector<const char *> getRequiredExtensions()
	{
		uint32_t glfwExtensionCount{};
		const char **glfwExtensions;
		glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

		std::vector<const char *> extensions(
		    glfwExtensions, glfwExtensions + glfwExtensionCount);

		if (enableValidationLayers)
		{
			extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
		}

		return extensions;
	}

	bool checkValidationLayerSupport()
	{
		uint32_t layerCount{};
		vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

		std::vector<VkLayerProperties> availableLayers(layerCount);
		vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

		for (const char *layerName : m_validationLayers)
		{
			bool layerFound{ false };

			for (const auto &layerProperties : availableLayers)
			{
				if (strcmp(layerName, layerProperties.layerName) == 0)
				{
					layerFound = true;
					break;
				}
			}

			if (!layerFound)
			{
				return false;
			}
		}

		return true;
	}
};

int main()
{
	PathTracer pathTracer;

	try
	{
		pathTracer.run();
	}
	catch (const std::exception &e)
	{
		std::cerr << e.what() << std::endl;
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}