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
#include <fstream>
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

const uint32_t WIDTH{ 800 };
const uint32_t HEIGHT{ 600 };
const std::string MODEL_PATH{ "models/cornell.obj" };
const int MAX_FRAMES_IN_FLIGHT{ 2 };
const int MAX_RT_FRAMES{ 1000 };

const std::vector<const char *> g_validationLayers{
	"VK_LAYER_KHRONOS_validation",
};

const std::vector<const char *> g_deviceExtensions{
	VK_KHR_SWAPCHAIN_EXTENSION_NAME,
	// Required to build acceleration structures
	VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
	// Required to use vkCmdTraceRaysKHR
	VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME,
	// Required by ray tracing pipelines
	VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
	VK_KHR_SHADER_CLOCK_EXTENSION_NAME
};

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

class HelloTriangleApplication
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

	VulkanDevice vulkanDevice{};
	VkPhysicalDevice m_physicalDevice{};
	VkDevice m_logicalDevice{};

	VkQueue m_graphicsQueue{};
	VkQueue m_presentQueue{};

	VulkanSwapchain vulkanSwapchain{};
	std::vector<VkFramebuffer> m_swapchainFramebuffers{};

	VulkanRaytracing vkrt{};

	VkCommandPool m_commandPool{};
	std::vector<VkCommandBuffer> m_commandBuffers{}; // will be destroyed when m_commandPool is destroyed

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

	// For raytracing device features
	void *deviceCreatepNextChain{ nullptr };
	VkPhysicalDeviceShaderClockFeaturesKHR m_shaderClockFeatures{};
	VkPhysicalDeviceScalarBlockLayoutFeatures m_scalarBlockLayoutFeatures{};
	VkPhysicalDeviceHostQueryResetFeatures m_hostQueryResetFeatures{};
	VkPhysicalDeviceBufferDeviceAddressFeatures m_bufferDeviceAddressFeatures{};
	VkPhysicalDeviceAccelerationStructureFeaturesKHR m_asFeatures{};
	VkPhysicalDeviceRayTracingPipelineFeaturesKHR m_rtPipelineFeatures{};

	VkPhysicalDeviceRayTracingPipelinePropertiesKHR m_rtPipelineProperties{};

	// For bottom level acceleration structure (BLAS)
	std::vector<VulkanRaytracing::Blas> m_blases{};

	// For top level acceleration structure (TLAS)
	VulkanRaytracing::Tlas m_tlas{};

	VkDescriptorPool m_rtDescriptorPool{};
	VkDescriptorSetLayout m_rtDescriptorSetLayout{};
	std::vector<VkDescriptorSet> m_rtDescriptorSets{};

	std::vector<VkRayTracingShaderGroupCreateInfoKHR> m_rtShaderGroups{};
	VkPipelineLayout m_rtPipelineLayout{};
	VkPipeline m_rtPipeline{};

	VulkanRaytracing::PushConstantRay m_pcRay{};

	VkBuffer m_rtSBTBuffer{};
	VkDeviceMemory m_rtSBTBufferMemory{};
	VkStridedDeviceAddressRegionKHR m_rgenRegion{};
	VkStridedDeviceAddressRegionKHR m_missRegion{};
	VkStridedDeviceAddressRegionKHR m_hitRegion{};
	VkStridedDeviceAddressRegionKHR m_callRegion{};

	void initWindow()
	{
		glfwInit();

		// Tell GLFW to not create OpenGL context
		glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

		m_window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
		glfwSetWindowUserPointer(m_window, this);
		glfwSetFramebufferSizeCallback(m_window, framebufferResizeCallback);
	}

	static void framebufferResizeCallback(GLFWwindow *window,
	                                      [[maybe_unused]] int width,
	                                      [[maybe_unused]] int height)
	{
		auto app = reinterpret_cast<HelloTriangleApplication *>(
		    glfwGetWindowUserPointer(window));
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

		VkPhysicalDeviceFeatures deviceFeatures{};
		deviceFeatures.samplerAnisotropy = VK_TRUE;
		deviceFeatures.sampleRateShading = VK_TRUE;
		deviceFeatures.shaderInt64 = VK_TRUE;
		vulkanDevice.pickPhysicalDevice(m_instance, vulkanSwapchain.m_surface, g_deviceExtensions, deviceFeatures);
		m_physicalDevice = vulkanDevice.physicalDevice;

		prepareRayTracing();

		vulkanDevice.createLogicalDevice(enableValidationLayers, g_validationLayers, g_deviceExtensions, deviceFeatures, deviceCreatepNextChain);
		m_logicalDevice = vulkanDevice.logicalDevice;

		createQueues();

		vulkanSwapchain.setContext(m_instance, &vulkanDevice, m_window);
		vulkanSwapchain.createSwapchain();

		createCommandPool();

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

		createCommandBuffers();
		createSyncObjects();

		vkrt.setContext(m_logicalDevice);
		buildBlas();
		buildTlas();

		createRtDescriptorSetLayout();
		createRtDescriptorPool();
		createRtDescriptorSets();
		createRtPipeline();
		createRtShaderBindingTable();
	}

	void mainLoop()
	{
		while (!glfwWindowShouldClose(m_window))
		{
			glfwPollEvents();
			drawFrame();
		}
		// Wait for logical device to finish operations before exiting mainLoop()
		glfwMakeContextCurrent(m_window);
		vkDeviceWaitIdle(m_logicalDevice);
	}

	void cleanup()
	{
		vulkanSwapchain.cleanup();

		// Clean up for rendering resources
		for (const auto &blas : m_blases)
		{
			vkrt.vkDestroyAccelerationStructureKHR(m_logicalDevice, blas.as, nullptr);
			vkDestroyBuffer(m_logicalDevice, blas.buffer, nullptr);
			vkFreeMemory(m_logicalDevice, blas.bufferMemory, nullptr);
		}

		vkrt.vkDestroyAccelerationStructureKHR(m_logicalDevice, m_tlas.as, nullptr);
		vkDestroyBuffer(m_logicalDevice, m_tlas.buffer, nullptr);
		vkFreeMemory(m_logicalDevice, m_tlas.bufferMemory, nullptr);

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

		for (size_t i{ 0 }; i < MAX_FRAMES_IN_FLIGHT; ++i)
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

		vkDestroyPipeline(m_logicalDevice, m_rtPipeline, nullptr);
		vkDestroyPipelineLayout(m_logicalDevice, m_rtPipelineLayout, nullptr);
		vkDestroyDescriptorPool(m_logicalDevice, m_rtDescriptorPool, nullptr);
		vkDestroyDescriptorSetLayout(m_logicalDevice, m_rtDescriptorSetLayout, nullptr);
		vkDestroyBuffer(m_logicalDevice, m_rtSBTBuffer, nullptr);
		vkFreeMemory(m_logicalDevice, m_rtSBTBufferMemory, nullptr);

		vkDestroyPipeline(m_logicalDevice, m_postGraphicsPipeline, nullptr);
		vkDestroyPipelineLayout(m_logicalDevice, m_postPipelineLayout, nullptr);
		vkDestroyRenderPass(m_logicalDevice, m_postRenderPass, nullptr);
		vkDestroyDescriptorPool(m_logicalDevice, m_postDescriptorPool, nullptr);
		vkDestroyDescriptorSetLayout(m_logicalDevice, m_postDescriptorSetLayout, nullptr);

		for (size_t i{ 0 }; i < MAX_FRAMES_IN_FLIGHT; ++i)
		{
			vkDestroySemaphore(m_logicalDevice, m_imageAvailableSemaphores[i],
			                   nullptr);
			vkDestroySemaphore(m_logicalDevice, m_renderFinishedSemaphores[i],
			                   nullptr);
			vkDestroyFence(m_logicalDevice, m_inFlightFences[i], nullptr);
		}

		vkDestroyCommandPool(m_logicalDevice, m_commandPool, nullptr);

		vkDestroyDevice(m_logicalDevice, nullptr);

		if (enableValidationLayers)
		{
			vk_debug::freeDebugMessenger(m_instance);
		}

		vkDestroyInstance(m_instance, nullptr);

		glfwDestroyWindow(m_window);

		std::cout << "Window destroyed!\n";

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

		updateRtDescriptorSets();

		resetRtFrame();
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
			createInfo.enabledLayerCount =
			    static_cast<uint32_t>(g_validationLayers.size());
			createInfo.ppEnabledLayerNames = g_validationLayers.data();

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
		descriptorPoolSize.descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);

		VkDescriptorPoolCreateInfo poolInfo{};
		poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
		poolInfo.poolSizeCount = 1;
		poolInfo.pPoolSizes = &descriptorPoolSize;
		poolInfo.maxSets = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);

		if (vkCreateDescriptorPool(m_logicalDevice, &poolInfo, nullptr, &m_postDescriptorPool) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to create post descriptor pool!");
		}
	}

	void createPostDescriptorSets()
	{
		std::vector<VkDescriptorSetLayout> layouts{ MAX_FRAMES_IN_FLIGHT, m_postDescriptorSetLayout };

		VkDescriptorSetAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
		allocInfo.descriptorPool = m_postDescriptorPool;
		allocInfo.descriptorSetCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
		allocInfo.pSetLayouts = layouts.data();

		m_postDescriptorSets.resize(MAX_FRAMES_IN_FLIGHT);
		if (vkAllocateDescriptorSets(m_logicalDevice, &allocInfo, m_postDescriptorSets.data()) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to allocate post descriptor sets!");
		}

		// Configure descriptors in descriptor sets
		updatePostDescriptorSets();
	}

	void updatePostDescriptorSets()
	{
		for (size_t i{ 0 }; i < MAX_FRAMES_IN_FLIGHT; ++i)
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
		auto vertShaderCode{ readFile("spv/post.vert.spv") };
		auto fragShaderCode{ readFile("spv/post.frag.spv") };

		VkShaderModule vertShaderModule{ createShaderModule(vertShaderCode) };
		VkShaderModule fragShaderModule{ createShaderModule(fragShaderCode) };

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
		colorBlendAttachment.blendEnable = VK_FALSE; // Set to false, since there is only one framebuffer

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

	void createCommandPool()
	{
		assert(m_physicalDevice);

		VkCommandPoolCreateInfo poolInfo{};
		poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
		poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

		// Command pool can only allocate command buffers that are submitted on single type of queue
		// We only want to record commands for drawing, that's why choose graphics queue family
		poolInfo.queueFamilyIndex = vulkanDevice.queueFamilyIndices.graphics.value();

		if (vkCreateCommandPool(m_logicalDevice, &poolInfo, nullptr, &m_commandPool) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to create command pool!");
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
		/*
		TODO: Change queue idle so that execution will run asynchronously instead of synchronously
		https://vulkan-tutorial.com/Texture_mapping/Images#:~:text=All%20of%20the%20helper,still%20set%20up%20correctly
		*/
		VkCommandBuffer commandBuffer{ beginSingleTimeCommands() };

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

		endSingleTimeCommands(commandBuffer, m_graphicsQueue);
	}

	void copyBufferToImage(VkBuffer buffer, VkImage image, uint32_t width,
	                       uint32_t height)
	{
		VkCommandBuffer commandBuffer{ beginSingleTimeCommands() };

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

		endSingleTimeCommands(commandBuffer, m_graphicsQueue);
	}

	void loadModel()
	{
		tinyobj::attrib_t attrib;
		std::vector<tinyobj::shape_t> shapes;
		std::vector<tinyobj::material_t> materials;
		std::string err{};

		// Get directory to mtl file, assumes that obj and mtl are in the same directory
		std::string mtlDir{};
		size_t pos = MODEL_PATH.find_last_of("/\\");
		if (pos != std::string::npos)
		{
			mtlDir = MODEL_PATH.substr(0, pos);
		}
		mtlDir += "/";

		if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &err, MODEL_PATH.c_str(), mtlDir.c_str()))
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

		m_model.vertexAddress = getBufferDeviceAddress(m_vertexBuffer);
		m_model.indexAddress = getBufferDeviceAddress(m_indexBuffer);
		m_model.matAddress = getBufferDeviceAddress(m_matBuffer);
		m_model.matIndexAddress = getBufferDeviceAddress(m_matIndexBuffer);

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
		createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
		             VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
		             stagingBuffer, stagingBufferMemory);

		void *data{};
		// Maps region of memory heap to host (CPU) accessible memory region
		vkMapMemory(m_logicalDevice, stagingBufferMemory, 0, bufferSize, 0, &data);
		memcpy(data, source, static_cast<size_t>(bufferSize));
		// Unmap memory
		vkUnmapMemory(m_logicalDevice, stagingBufferMemory);

		// Create buffer
		createBuffer(bufferSize, usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, buffer, bufferMemory);

		copyBuffer(stagingBuffer, buffer, bufferSize);

		// Destroy staging buffer
		vkDestroyBuffer(m_logicalDevice, stagingBuffer, nullptr);
		vkFreeMemory(m_logicalDevice, stagingBufferMemory, nullptr);
	}

	void createUniformBuffers()
	{
		VkDeviceSize bufferSize{ sizeof(UniformBufferObject) };

		m_uniformBuffers.resize(MAX_FRAMES_IN_FLIGHT);
		m_uniformBuffersMemory.resize(MAX_FRAMES_IN_FLIGHT);
		m_uniformBuffersMapped.resize(MAX_FRAMES_IN_FLIGHT);

		for (size_t i{ 0 }; i < MAX_FRAMES_IN_FLIGHT; i++)
		{
			// Not on device local memory since we need to update uniform buffer every frame
			// see updateUniformBuffer()
			createBuffer(bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
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
		poolSizes[0].descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
		poolSizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		poolSizes[1].descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);

		VkDescriptorPoolCreateInfo poolInfo{};
		poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
		poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
		poolInfo.pPoolSizes = poolSizes.data();
		poolInfo.maxSets = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);

		if (vkCreateDescriptorPool(m_logicalDevice, &poolInfo, nullptr, &m_descriptorPool) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to create descriptor pool!");
		}
	}

	void createDescriptorSets()
	{
		std::vector<VkDescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, m_descriptorSetLayout);
		VkDescriptorSetAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
		allocInfo.descriptorPool = m_descriptorPool;
		allocInfo.descriptorSetCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
		allocInfo.pSetLayouts = layouts.data();

		m_descriptorSets.resize(MAX_FRAMES_IN_FLIGHT);
		if (vkAllocateDescriptorSets(m_logicalDevice, &allocInfo, m_descriptorSets.data()) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to allocate descriptor sets!");
		}

		// Configure descriptors in descriptor sets
		for (size_t i{ 0 }; i < MAX_FRAMES_IN_FLIGHT; ++i)
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

	void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
	                  VkMemoryPropertyFlags properties, VkBuffer &buffer,
	                  VkDeviceMemory &bufferMemory)
	{
		VkBufferCreateInfo bufferInfo{};
		bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
		bufferInfo.size = size;
		bufferInfo.usage = usage;
		bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

		if (vkCreateBuffer(m_logicalDevice, &bufferInfo, nullptr, &buffer) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to create buffer!");
		}

		VkMemoryRequirements memRequirements;
		vkGetBufferMemoryRequirements(m_logicalDevice, buffer, &memRequirements);

		VkMemoryAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		allocInfo.allocationSize = memRequirements.size;
		allocInfo.memoryTypeIndex = vulkanDevice.findMemoryType(memRequirements.memoryTypeBits, properties);

		VkMemoryAllocateFlagsInfo allocFlagInfo{};
		if (usage & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT)
		{
			allocFlagInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
			allocFlagInfo.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT_KHR;
			allocInfo.pNext = &allocFlagInfo;
		}

		if (vkAllocateMemory(m_logicalDevice, &allocInfo, nullptr, &bufferMemory) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to allocate buffer memory!");
		}

		if (vkBindBufferMemory(m_logicalDevice, buffer, bufferMemory, 0) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to bind buffer memory!");
		}
	}

	VkCommandBuffer beginSingleTimeCommands()
	{
		VkCommandBufferAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		allocInfo.commandPool = m_commandPool;
		allocInfo.commandBufferCount = 1;

		VkCommandBuffer commandBuffer;
		vkAllocateCommandBuffers(m_logicalDevice, &allocInfo, &commandBuffer);

		VkCommandBufferBeginInfo beginInfo{};
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

		vkBeginCommandBuffer(commandBuffer, &beginInfo);

		return commandBuffer;
	}

	void endSingleTimeCommands(VkCommandBuffer commandBuffer, VkQueue queue)
	{
		vkEndCommandBuffer(commandBuffer);

		VkSubmitInfo submitInfo{};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &commandBuffer;

		vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE);

		// TODO: Let CPU and GPU run in parallel instead of making CPU wait for GPU to finish
		//		 after submitting each command buffer
		// Refer to https://nvpro-samples.github.io/vk_mini_path_tracer/index.html#accelerationstructuresandraytracing:~:text=Command%20Buffer%20Techniques%20for%20Production%20Applications
		vkQueueWaitIdle(queue);

		vkFreeCommandBuffers(m_logicalDevice, m_commandPool, 1, &commandBuffer);
	}

	void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size)
	{
		VkCommandBuffer commandBuffer{ beginSingleTimeCommands() };

		VkBufferCopy copyRegion{};
		copyRegion.size = size;
		vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);

		endSingleTimeCommands(commandBuffer, m_graphicsQueue);
	}

	void createCommandBuffers()
	{
		m_commandBuffers.resize(MAX_FRAMES_IN_FLIGHT);

		VkCommandBufferAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		allocInfo.commandPool = m_commandPool;
		allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		allocInfo.commandBufferCount = static_cast<uint32_t>(m_commandBuffers.size());

		if (vkAllocateCommandBuffers(m_logicalDevice, &allocInfo, m_commandBuffers.data()) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to allocate command buffers!");
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

		raytrace(commandBuffer);

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
		m_imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
		m_renderFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
		m_inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);

		VkSemaphoreCreateInfo semaphoreInfo{};
		semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

		VkFenceCreateInfo fenceInfo{};
		fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
		fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

		for (size_t i{ 0 }; i < MAX_FRAMES_IN_FLIGHT; ++i)
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
		// TODO: Allow view to change when user drags cursor
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

		vkResetCommandBuffer(m_commandBuffers[m_currentFrame], 0);
		recordCommandBuffer(m_commandBuffers[m_currentFrame], imageIndex);

		VkSubmitInfo submitInfo{};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

		VkSemaphore waitSemaphores[]{ m_imageAvailableSemaphores[m_currentFrame] };
		VkSemaphore signalSemaphores[]{ m_renderFinishedSemaphores[m_currentFrame] };
		VkPipelineStageFlags waitStages[]{ VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
		submitInfo.waitSemaphoreCount = 1;
		submitInfo.pWaitSemaphores = waitSemaphores;
		submitInfo.pWaitDstStageMask = waitStages;
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &m_commandBuffers[m_currentFrame];
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

		m_currentFrame = (m_currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;

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

	VkShaderModule createShaderModule(const std::vector<char> &code)
	{
		VkShaderModuleCreateInfo createInfo{};
		createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
		createInfo.codeSize = code.size();
		createInfo.pCode = reinterpret_cast<const uint32_t *>(code.data());

		VkShaderModule shaderModule;
		if (vkCreateShaderModule(m_logicalDevice, &createInfo, nullptr,
		                         &shaderModule) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to create shader module!");
		}

		return shaderModule;
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

		for (const char *layerName : g_validationLayers)
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

	static std::vector<char> readFile(const std::string &filename)
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

	/*
	Checks if required features for raytracing are supported, if so store it in pNext to be enabled during logical device creation
	*/
	void prepareRayTracing()
	{
		VkPhysicalDeviceFeatures2 deviceFeatures2{};
		deviceFeatures2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;

		m_shaderClockFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_CLOCK_FEATURES_KHR;
		deviceFeatures2.pNext = &m_shaderClockFeatures;
		vkGetPhysicalDeviceFeatures2(m_physicalDevice, &deviceFeatures2);
		if (m_shaderClockFeatures.shaderDeviceClock != VK_TRUE && m_shaderClockFeatures.shaderSubgroupClock != VK_TRUE)
		{
			throw std::runtime_error("shader clock is not supported!");
		}

		m_scalarBlockLayoutFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SCALAR_BLOCK_LAYOUT_FEATURES;
		deviceFeatures2.pNext = &m_scalarBlockLayoutFeatures;
		vkGetPhysicalDeviceFeatures2(m_physicalDevice, &deviceFeatures2);
		if (m_scalarBlockLayoutFeatures.scalarBlockLayout != VK_TRUE)
		{
			throw std::runtime_error("scalar block layout not supported!");
		}

		m_hostQueryResetFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_HOST_QUERY_RESET_FEATURES;
		deviceFeatures2.pNext = &m_hostQueryResetFeatures;
		vkGetPhysicalDeviceFeatures2(m_physicalDevice, &deviceFeatures2);
		if (m_hostQueryResetFeatures.hostQueryReset != VK_TRUE)
		{
			throw std::runtime_error("host query reset not supported!");
		}

		m_bufferDeviceAddressFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES;
		deviceFeatures2.pNext = &m_bufferDeviceAddressFeatures;
		vkGetPhysicalDeviceFeatures2(m_physicalDevice, &deviceFeatures2);
		if (m_bufferDeviceAddressFeatures.bufferDeviceAddress != VK_TRUE)
		{
			throw std::runtime_error("buffer device address not supported!");
		}

		m_rtPipelineFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR;
		deviceFeatures2.pNext = &m_rtPipelineFeatures;
		vkGetPhysicalDeviceFeatures2(m_physicalDevice, &deviceFeatures2);
		if (m_rtPipelineFeatures.rayTracingPipeline != VK_TRUE)
		{
			throw std::runtime_error("ray tracing pipeline not supported");
		}

		m_asFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR;
		deviceFeatures2.pNext = &m_asFeatures;
		vkGetPhysicalDeviceFeatures2(m_physicalDevice, &deviceFeatures2);
		if (m_asFeatures.accelerationStructure != VK_TRUE)
		{
			throw std::runtime_error("acceleration structure not supported");
		}

		m_scalarBlockLayoutFeatures.pNext = &m_shaderClockFeatures;
		m_hostQueryResetFeatures.pNext = &m_scalarBlockLayoutFeatures;
		m_bufferDeviceAddressFeatures.pNext = &m_hostQueryResetFeatures;
		m_rtPipelineFeatures.pNext = &m_bufferDeviceAddressFeatures;
		m_asFeatures.pNext = &m_rtPipelineFeatures;
		deviceCreatepNextChain = &m_asFeatures;

		// Request ray tracing properties
		VkPhysicalDeviceProperties2 deviceProperties2{};
		deviceProperties2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;

		m_rtPipelineProperties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR;
		deviceProperties2.pNext = &m_rtPipelineProperties;
		vkGetPhysicalDeviceProperties2(m_physicalDevice, &deviceProperties2);
	}

	VulkanRaytracing::BlasInput objectToVkGeometryKHR()
	{
		assert(m_model.vertexAddress);
		assert(m_model.indexAddress);

		// Number of triangles, each triangle is treated as 3 vertices
		uint32_t maxPrimitiveCount{ static_cast<uint32_t>(m_indices.size()) / 3 };

		// Describe buffer as array of Vertex objects
		VkAccelerationStructureGeometryTrianglesDataKHR triangles{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR };
		triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
		triangles.vertexData.deviceAddress = m_model.vertexAddress;
		triangles.vertexStride = sizeof(Vertex);
		// Describe index data (32-bit unsigned int, so max possible indices in 2^32)
		triangles.indexType = VK_INDEX_TYPE_UINT32;
		triangles.indexData.deviceAddress = m_model.indexAddress;
		// Indicate identity transform by setting transformData to null device pointer
		triangles.transformData = {};
		triangles.maxVertex = static_cast<uint32_t>(m_vertices.size() - 1);

		// Identify the above data as containing opaque triangles
		VkAccelerationStructureGeometryKHR asGeometry{};
		asGeometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
		asGeometry.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
		asGeometry.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
		asGeometry.geometry.triangles = triangles;

		// The entire array will be used to build the BLAS
		VkAccelerationStructureBuildRangeInfoKHR buildRangeInfo{};
		buildRangeInfo.primitiveCount = maxPrimitiveCount;
		buildRangeInfo.primitiveOffset = 0;
		buildRangeInfo.firstVertex = 0;
		buildRangeInfo.transformOffset = 0;

		// Could add more geometry in each Blas, but only one for now
		VulkanRaytracing::BlasInput input{};
		input.geometry.emplace_back(asGeometry);
		input.buildRangeInfo.emplace_back(buildRangeInfo);

		return input;
	}

	void buildBlas(bool allowCompaction = true)
	{
		VkDeviceSize asTotalSize{ 0 };    // Memory size of all allocated BLAS
		VkDeviceSize maxScratchSize{ 0 }; // Largest required scratch size

		// Convert all OBJ models into ray tracing geometry used to build BLAS
		const size_t numBlas{ 1 };
		std::vector<VulkanRaytracing::BlasInput> blasInputs{
			objectToVkGeometryKHR()
		};

		m_blases.resize(numBlas);

		for (size_t blasIdx{ 0 }; blasIdx < numBlas; ++blasIdx)
		{
			blasInputs[blasIdx].buildGeomInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
			blasInputs[blasIdx].buildGeomInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
			blasInputs[blasIdx].buildGeomInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;

			// Note that there cannot be a mix of compaction & no compaction
			if (allowCompaction)
			{
				// Compaction will limit memory allocation required
				// Adding flag VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_COMPACTION_BIT_KHR to allow compaction, so that BLAS will use less memory
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
			vkrt.vkGetAccelerationStructureBuildSizesKHR(m_logicalDevice, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
			                                             &blasInputs[blasIdx].buildGeomInfo, maxPrimitiveCounts.data(), &blasInputs[blasIdx].buildSizeInfo);

			asTotalSize += blasInputs[blasIdx].buildSizeInfo.accelerationStructureSize;
			maxScratchSize = std::max(maxScratchSize, blasInputs[blasIdx].buildSizeInfo.buildScratchSize);
		}

		// Create scratch buffer
		// Will be reused when building each BLAS, so scratch buffer created with maximum scratch memory needed
		VkBuffer scratchBuffer{};
		VkDeviceMemory scratchBufferMemory{};
		createBuffer(
		    maxScratchSize,
		    VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
		    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
		    scratchBuffer,
		    scratchBufferMemory);
		VkDeviceAddress scratchAddress{ getBufferDeviceAddress(scratchBuffer) };

		// Allocate a query pool for storing the needed size for every acceleration structure compaction
		VkQueryPool queryPool{ VK_NULL_HANDLE };
		if (allowCompaction)
		{
			VkQueryPoolCreateInfo queryPoolCreateInfo{};
			queryPoolCreateInfo.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
			queryPoolCreateInfo.queryCount = numBlas;
			queryPoolCreateInfo.queryType = VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR;
			vkCreateQueryPool(m_logicalDevice, &queryPoolCreateInfo, nullptr, &queryPool);
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
				cmdCreateBlas(blasInputs, blasIndices, scratchAddress, queryPool);

				if (allowCompaction)
				{
					cmdCompactBlas(blasInputs, blasIndices, queryPool);
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
		vkDestroyQueryPool(m_logicalDevice, queryPool, nullptr);
		vkDestroyBuffer(m_logicalDevice, scratchBuffer, nullptr);
		vkFreeMemory(m_logicalDevice, scratchBufferMemory, nullptr);
	}

	void cmdCreateBlas(std::vector<VulkanRaytracing::BlasInput> &blasInputs,
	                   std::vector<size_t> indices,
	                   VkDeviceAddress scratchAddress,
	                   VkQueryPool queryPool)
	{
		// Reset the query to know real size of the BLAS (for compaction)
		if (queryPool)
		{
			vkResetQueryPool(m_logicalDevice, queryPool, 0, static_cast<uint32_t>(indices.size()));
		}

		for (size_t i : indices)
		{
			// Create acceleration structure buffer
			createBuffer(blasInputs[i].buildSizeInfo.accelerationStructureSize,
			             VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
			             VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
			             m_blases[i].buffer,
			             m_blases[i].bufferMemory);

			VkAccelerationStructureCreateInfoKHR asCreateInfo{};
			asCreateInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
			asCreateInfo.buffer = m_blases[i].buffer;
			asCreateInfo.size = blasInputs[i].buildSizeInfo.accelerationStructureSize;
			asCreateInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;

			vkrt.vkCreateAccelerationStructureKHR(m_logicalDevice, &asCreateInfo, nullptr, &m_blases[i].as);

			blasInputs[i].buildGeomInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
			blasInputs[i].buildGeomInfo.dstAccelerationStructure = m_blases[i].as;  // Setting where the build lands
			blasInputs[i].buildGeomInfo.scratchData.deviceAddress = scratchAddress; // All builds are using the same scratch buffer

			VkCommandBuffer commandBuffer{ beginSingleTimeCommands() };

			// Build the Bottom Level Acceleration Structure (BLAS)
			vkrt.vkCmdBuildAccelerationStructuresKHR(commandBuffer, 1, &blasInputs[i].buildGeomInfo, &blasInputs[i].pBuildRangeInfo);

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
				vkrt.vkCmdWriteAccelerationStructuresPropertiesKHR(commandBuffer, 1, &blasInputs[i].buildGeomInfo.dstAccelerationStructure,
				                                                   VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR, queryPool, static_cast<uint32_t>(i - indices[0]));
			}

			endSingleTimeCommands(commandBuffer, m_graphicsQueue);
		}
	}

	// Create and replace a new acceleration structure and buffer
	// based on the size retrieved in cmdCreateBlas()
	void cmdCompactBlas(std::vector<VulkanRaytracing::BlasInput> &blasInputs,
	                    std::vector<size_t> indices,
	                    VkQueryPool queryPool)
	{
		// Get compacted size result written in cmdCreateBlas()
		uint32_t numBlas{ static_cast<uint32_t>(indices.size()) };
		std::vector<VkDeviceSize> compactSizes(numBlas);
		vkGetQueryPoolResults(m_logicalDevice, queryPool, 0, numBlas, numBlas * sizeof(VkDeviceSize),
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
			createBuffer(blasInputs[i].buildSizeInfo.accelerationStructureSize,
			             VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
			             VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
			             compactBlasBuffer,
			             compactBlasBufferMemory);

			VkAccelerationStructureCreateInfoKHR compactASCreateInfo{};
			compactASCreateInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
			compactASCreateInfo.buffer = compactBlasBuffer;
			compactASCreateInfo.size = blasInputs[i].buildSizeInfo.accelerationStructureSize;
			compactASCreateInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
			vkrt.vkCreateAccelerationStructureKHR(m_logicalDevice, &compactASCreateInfo, nullptr, &compactBlas);

			VkCommandBuffer compactCommandBuffer{ beginSingleTimeCommands() };

			// Copy the original BLAS to a compact version
			VkCopyAccelerationStructureInfoKHR copyInfo{};
			copyInfo.sType = VK_STRUCTURE_TYPE_COPY_ACCELERATION_STRUCTURE_INFO_KHR;
			copyInfo.src = blasInputs[i].buildGeomInfo.dstAccelerationStructure;
			copyInfo.dst = compactBlas;
			copyInfo.mode = VK_COPY_ACCELERATION_STRUCTURE_MODE_COMPACT_KHR;
			vkrt.vkCmdCopyAccelerationStructureKHR(compactCommandBuffer, &copyInfo);

			endSingleTimeCommands(compactCommandBuffer, m_graphicsQueue);

			// Destroy original BLAS
			vkrt.vkDestroyAccelerationStructureKHR(m_logicalDevice, m_blases[i].as, nullptr);
			vkDestroyBuffer(m_logicalDevice, m_blases[i].buffer, nullptr);
			vkFreeMemory(m_logicalDevice, m_blases[i].bufferMemory, nullptr);

			m_blases[i].as = compactBlas;
			m_blases[i].buffer = compactBlasBuffer;
			m_blases[i].bufferMemory = compactBlasBufferMemory;
		}
	}

	void createInstancesBuffer(VkBuffer &instancesBuffer,
	                           VkDeviceMemory &instancesBufferMemory)
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
		createBuffer(bufferSize,
		             VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
		             VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
		             stagingBuffer,
		             stagingBufferMemory);

		void *data{};
		vkMapMemory(m_logicalDevice, stagingBufferMemory, 0, bufferSize, 0, &data);
		memcpy(data, instances.data(), static_cast<size_t>(bufferSize));
		vkUnmapMemory(m_logicalDevice, stagingBufferMemory);

		// Create instances buffer
		createBuffer(bufferSize,
		             VK_BUFFER_USAGE_TRANSFER_DST_BIT |
		                 VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
		             VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
		             instancesBuffer,
		             instancesBufferMemory);

		// vkQueueWaitIdle() in endSingleTimeCommands() ensures that instance buffer is copied
		// before we move onto acceleration structure build
		copyBuffer(stagingBuffer, instancesBuffer, bufferSize);

		// Clear staging buffer resources
		vkDestroyBuffer(m_logicalDevice, stagingBuffer, nullptr);
		vkFreeMemory(m_logicalDevice, stagingBufferMemory, nullptr);
	}

	// TODO: Call function again when updating Tlas with updated matrices,
	// need to have VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR flag
	void buildTlas()
	{
		uint32_t numTlas{ 1 };

		// Create a buffer to hold real instance data
		VkBuffer instancesBuffer{};
		VkDeviceMemory instancesBufferMemory{};
		createInstancesBuffer(instancesBuffer, instancesBufferMemory);

		VkAccelerationStructureGeometryInstancesDataKHR instancesData{};
		instancesData.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
		instancesData.data.deviceAddress = getBufferDeviceAddress(instancesBuffer);

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
		vkrt.vkGetAccelerationStructureBuildSizesKHR(m_logicalDevice, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
		                                             &buildGeometryInfo, &numTlas, &buildSizeInfo);

		// Create acceleration structure buffer
		createBuffer(
		    buildSizeInfo.accelerationStructureSize,
		    VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
		    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
		    m_tlas.buffer,
		    m_tlas.bufferMemory);

		VkAccelerationStructureCreateInfoKHR asCreateInfo{};
		asCreateInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
		asCreateInfo.buffer = m_tlas.buffer;
		asCreateInfo.size = buildSizeInfo.accelerationStructureSize;
		asCreateInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;

		vkrt.vkCreateAccelerationStructureKHR(m_logicalDevice, &asCreateInfo, nullptr, &m_tlas.as);

		// Allocate the scratch buffers holding the temporary data used during build of acceleration structure
		VkBuffer scratchBuffer{};
		VkDeviceMemory scratchBufferMemory{};
		createBuffer(
		    buildSizeInfo.buildScratchSize,
		    VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
		    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
		    scratchBuffer,
		    scratchBufferMemory);
		VkDeviceAddress scratchAddress{ getBufferDeviceAddress(scratchBuffer) };

		buildGeometryInfo.srcAccelerationStructure = VK_NULL_HANDLE;
		buildGeometryInfo.dstAccelerationStructure = m_tlas.as;
		buildGeometryInfo.scratchData.deviceAddress = scratchAddress;

		VkAccelerationStructureBuildRangeInfoKHR buildRangeInfo{};
		buildRangeInfo.primitiveCount = numTlas;
		buildRangeInfo.primitiveOffset = 0;
		buildRangeInfo.firstVertex = 0;
		buildRangeInfo.transformOffset = 0;

		VkCommandBuffer commandBuffer{ beginSingleTimeCommands() };

		const VkAccelerationStructureBuildRangeInfoKHR *pBuildRangeInfo = &buildRangeInfo;
		vkrt.vkCmdBuildAccelerationStructuresKHR(commandBuffer, 1, &buildGeometryInfo, &pBuildRangeInfo);

		endSingleTimeCommands(commandBuffer, m_graphicsQueue);

		vkDestroyBuffer(m_logicalDevice, instancesBuffer, nullptr);
		vkFreeMemory(m_logicalDevice, instancesBufferMemory, nullptr);
		vkDestroyBuffer(m_logicalDevice, scratchBuffer, nullptr);
		vkFreeMemory(m_logicalDevice, scratchBufferMemory, nullptr);
	}

	VkDeviceAddress getBufferDeviceAddress(VkBuffer buffer)
	{
		VkBufferDeviceAddressInfo bufferDeviceAddressInfo{};
		bufferDeviceAddressInfo.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
		bufferDeviceAddressInfo.buffer = buffer;
		return vkGetBufferDeviceAddress(m_logicalDevice, &bufferDeviceAddressInfo);
	}

	VkDeviceAddress getBlasDeviceAddress(VkAccelerationStructureKHR blas)
	{
		VkAccelerationStructureDeviceAddressInfoKHR accelerationDeviceAddrInfo{};
		accelerationDeviceAddrInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
		accelerationDeviceAddrInfo.accelerationStructure = blas;
		return vkrt.vkGetAccelerationStructureDeviceAddressKHR(m_logicalDevice, &accelerationDeviceAddrInfo);
	}

	void createRtDescriptorPool()
	{
		std::array<VkDescriptorPoolSize, 2> poolSizes{};
		poolSizes[0].type = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
		poolSizes[0].descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
		poolSizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
		poolSizes[1].descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);

		VkDescriptorPoolCreateInfo poolInfo{};
		poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
		poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
		poolInfo.pPoolSizes = poolSizes.data();
		poolInfo.maxSets = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);

		if (vkCreateDescriptorPool(m_logicalDevice, &poolInfo, nullptr, &m_rtDescriptorPool) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to create ray tracing descriptor pool!");
		}
	}

	void createRtDescriptorSetLayout()
	{
		// For TLAS
		VkDescriptorSetLayoutBinding tlasLayoutBinding{};
		tlasLayoutBinding.binding = 0;
		tlasLayoutBinding.descriptorCount = 1;
		tlasLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
		// TLAS is usable by both ray generation, and closest hit (to shoot shadow rays)
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

		if (vkCreateDescriptorSetLayout(m_logicalDevice, &layoutInfo, nullptr,
		                                &m_rtDescriptorSetLayout) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to create ray tracing descriptor set layout!");
		}
	}

	void createRtDescriptorSets()
	{
		std::vector<VkDescriptorSetLayout> layouts{ MAX_FRAMES_IN_FLIGHT, m_rtDescriptorSetLayout };

		VkDescriptorSetAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
		allocInfo.descriptorPool = m_rtDescriptorPool;
		allocInfo.descriptorSetCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
		allocInfo.pSetLayouts = layouts.data();

		m_rtDescriptorSets.resize(MAX_FRAMES_IN_FLIGHT);
		if (vkAllocateDescriptorSets(m_logicalDevice, &allocInfo, m_rtDescriptorSets.data()) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to allocate ray tracing descriptor sets!");
		}

		// Configure descriptors in descriptor sets
		for (size_t i{ 0 }; i < MAX_FRAMES_IN_FLIGHT; ++i)
		{
			// For TLAS
			VkWriteDescriptorSetAccelerationStructureKHR asInfo{};
			asInfo.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR;
			asInfo.accelerationStructureCount = 1;
			asInfo.pAccelerationStructures = &m_tlas.as;

			// For output image written by RayGen shader
			VkDescriptorImageInfo imageInfo{};
			imageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
			imageInfo.imageView = m_colorImageView;

			std::array<VkWriteDescriptorSet, 2> descriptorWrites{};

			descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			descriptorWrites[0].dstSet = m_rtDescriptorSets[i];
			descriptorWrites[0].dstBinding = 0;
			descriptorWrites[0].dstArrayElement = 0;
			descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
			descriptorWrites[0].descriptorCount = 1;
			descriptorWrites[0].pNext = &asInfo;

			descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			descriptorWrites[1].dstSet = m_rtDescriptorSets[i];
			descriptorWrites[1].dstBinding = 1;
			descriptorWrites[1].dstArrayElement = 0;
			descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
			descriptorWrites[1].descriptorCount = 1;
			descriptorWrites[1].pImageInfo = &imageInfo;

			vkUpdateDescriptorSets(m_logicalDevice, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
		}
	}

	void updateRtDescriptorSets()
	{
		// Only need to update output image reference
		for (size_t i{ 0 }; i < MAX_FRAMES_IN_FLIGHT; ++i)
		{
			// For output image written by RayGen shader
			VkDescriptorImageInfo imageInfo{};
			imageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
			imageInfo.imageView = m_colorImageView;

			VkWriteDescriptorSet descriptorWrite{};

			descriptorWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			descriptorWrite.dstSet = m_rtDescriptorSets[i];
			descriptorWrite.dstBinding = 1;
			descriptorWrite.dstArrayElement = 0;
			descriptorWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
			descriptorWrite.descriptorCount = 1;
			descriptorWrite.pImageInfo = &imageInfo;

			vkUpdateDescriptorSets(m_logicalDevice, 1, &descriptorWrite, 0, nullptr);
		}
	}

	void createRtPipeline()
	{
		enum StageIndices
		{
			eRaygen,
			eMiss,
			eMiss2,
			eClosestHit,
			eShaderGroupCount
		};

		std::array<VkPipelineShaderStageCreateInfo, eShaderGroupCount> shaderStages{};

		VkPipelineShaderStageCreateInfo shaderStageInfo{};
		shaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		shaderStageInfo.pName = "main"; // All the same entry point

		// Raygen
		shaderStageInfo.module = createShaderModule(readFile("spv/raytrace.rgen.spv"));
		shaderStageInfo.stage = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
		shaderStages[eRaygen] = shaderStageInfo;

		// Miss
		shaderStageInfo.module = createShaderModule(readFile("spv/raytrace.rmiss.spv"));
		shaderStageInfo.stage = VK_SHADER_STAGE_MISS_BIT_KHR;
		shaderStages[eMiss] = shaderStageInfo;

		// Shadow Miss
		shaderStageInfo.module = createShaderModule(readFile("spv/raytraceShadow.rmiss.spv"));
		shaderStageInfo.stage = VK_SHADER_STAGE_MISS_BIT_KHR;
		shaderStages[eMiss2] = shaderStageInfo;

		// Closest Hit
		shaderStageInfo.module = createShaderModule(readFile("spv/raytrace.rchit.spv"));
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
		m_rtShaderGroups.push_back(group);

		// Miss
		group.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
		group.generalShader = eMiss;
		m_rtShaderGroups.push_back(group);

		// Shadow Miss
		group.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
		group.generalShader = eMiss2;
		m_rtShaderGroups.push_back(group);

		// Hit Group - Closest hit
		// Each hit group can comprise 1 - 3 shaders (intersection, any hit, closest hit)
		// Since triangles, can use built-in ray-triangle intersection test
		// If not triangles, will need to define an intersection shader and set type to VK_*_PROCEDURAL_HIT_GROUP_KHR
		group.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
		group.generalShader = VK_SHADER_UNUSED_KHR;
		group.closestHitShader = eClosestHit;
		m_rtShaderGroups.push_back(group);

		VkPushConstantRange pushConstant{};
		pushConstant.stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR;
		pushConstant.offset = 0;
		pushConstant.size = sizeof(VulkanRaytracing::PushConstantRay);

		VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
		pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		pipelineLayoutInfo.pushConstantRangeCount = 1;
		pipelineLayoutInfo.pPushConstantRanges = &pushConstant;

		// Two descriptor sets
		// set = 0, specific to ray tracing pipeline (TlAS and output image)
		// set = 1, shared by rt and rasterization (i.e. scene data)
		std::vector<VkDescriptorSetLayout> layouts{ m_rtDescriptorSetLayout,
			                                        m_descriptorSetLayout };
		pipelineLayoutInfo.setLayoutCount = static_cast<uint32_t>(layouts.size());
		pipelineLayoutInfo.pSetLayouts = layouts.data();

		if (vkCreatePipelineLayout(m_logicalDevice, &pipelineLayoutInfo, nullptr, &m_rtPipelineLayout) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to create ray tracing pipeline layout!");
		}

		// Assemble the shader stages and recursion depth info into ray tracing pipeline
		VkRayTracingPipelineCreateInfoKHR rtPipelineInfo{};
		rtPipelineInfo.sType = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR;
		rtPipelineInfo.stageCount = static_cast<uint32_t>(shaderStages.size());
		rtPipelineInfo.pStages = shaderStages.data();
		// We have one raygen group, one miss group, and one hit group
		rtPipelineInfo.groupCount = static_cast<uint32_t>(m_rtShaderGroups.size());
		rtPipelineInfo.pGroups = m_rtShaderGroups.data();
		rtPipelineInfo.maxPipelineRayRecursionDepth = 1;
		rtPipelineInfo.layout = m_rtPipelineLayout;

		if (m_rtPipelineProperties.maxRayRecursionDepth < rtPipelineInfo.maxPipelineRayRecursionDepth)
		{
			throw std::runtime_error("device fails to support specified ray recursion depth!");
		}

		if (vkrt.vkCreateRayTracingPipelinesKHR(m_logicalDevice, VK_NULL_HANDLE, VK_NULL_HANDLE, 1,
		                                        &rtPipelineInfo, nullptr, &m_rtPipeline) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to create ray tracing pipeline!");
		}

		for (auto &shaderStage : shaderStages)
		{
			vkDestroyShaderModule(m_logicalDevice, shaderStage.module, nullptr);
		}
	}

	// Return new alignment (that >= size, hence fitting size), that is a multiple of alignment
	uint32_t alignedSize(uint32_t size, uint32_t alignment)
	{
		return (size + alignment - 1) & ~(alignment - 1);
	}

	void createRtShaderBindingTable()
	{
		const uint32_t rgenCount{ 1 }; // Always only one raygen
		const uint32_t missCount{ 2 };
		const uint32_t hitCount{ 1 };
		auto handleCount{ rgenCount + missCount + hitCount };

		// Size in bytes of shader header
		const uint32_t handleSize{ m_rtPipelineProperties.shaderGroupHandleSize };
		// Required alignment in bytes for each entry in a SBT
		const uint32_t handleAlignment{ m_rtPipelineProperties.shaderGroupHandleAlignment };
		// Required alignment in bytes for the base of the SBT
		const uint32_t baseAlignment{ m_rtPipelineProperties.shaderGroupBaseAlignment };

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
		// Fetch handles to shader groups of pipeline, hence must be done after createRtPipeline()
		if (vkrt.vkGetRayTracingShaderGroupHandlesKHR(m_logicalDevice, m_rtPipeline, 0, handleCount, dataSize, handles.data()) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to get ray tracing shader group handles");
		}

		// Note that it is also possible to separate SBT into several buffers, one for each type: raygen, miss, hit group, call
		VkDeviceSize sbtSize{ m_rgenRegion.size + m_missRegion.size + m_hitRegion.size + m_callRegion.size };
		// TODO: Not sure if memory in host is best option
		createBuffer(sbtSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR,
		             VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
		             m_rtSBTBuffer, m_rtSBTBufferMemory);

		// Find SBT addresses of each group
		VkDeviceAddress sbtAddress{ getBufferDeviceAddress(m_rtSBTBuffer) };
		m_rgenRegion.deviceAddress = sbtAddress;
		m_missRegion.deviceAddress = sbtAddress + m_rgenRegion.size;
		m_hitRegion.deviceAddress = sbtAddress + m_rgenRegion.size + m_missRegion.size;

		auto getHandlePtr = [&](int i) { return handles.data() + i * handleSize; };

		// Map SBT Buffer and write in the handles
		void *pSBTBuffer{};
		vkMapMemory(m_logicalDevice, m_rtSBTBufferMemory, 0, dataSize, 0, &pSBTBuffer);
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
		vkUnmapMemory(m_logicalDevice, m_rtSBTBufferMemory);
	}

	void raytrace(VkCommandBuffer commandBuffer)
	{
		updateRtFrame();

		// If max sampling has reached, output should be good enough hence stop accumulating further samples.
		// This helps to reduce GPU usage too.
		// Note that total samples = MAX_RT_FRAMES * MAX_SAMPLES (in rgen shader)
		static bool printOnce{ true };
		if (m_pcRay.frame >= MAX_RT_FRAMES)
		{
			if (printOnce)
			{
				std::cout << "Max frames reached, ray tracing terminated!" << '\n';
				printOnce = false;
			}
			return;
		}

		std::vector<VkDescriptorSet> descriptorSets{ m_rtDescriptorSets[m_currentFrame],
			                                         m_descriptorSets[m_currentFrame] };

		vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_rtPipeline);
		vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_rtPipelineLayout, 0,
		                        static_cast<uint32_t>(descriptorSets.size()), descriptorSets.data(), 0, nullptr);
		vkCmdPushConstants(commandBuffer, m_rtPipelineLayout,
		                   VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR,
		                   0, sizeof(VulkanRaytracing::PushConstantRay), &m_pcRay);

		vkrt.vkCmdTraceRaysKHR(commandBuffer, &m_rgenRegion, &m_missRegion, &m_hitRegion, &m_callRegion,
		                       vulkanSwapchain.m_extent.width, vulkanSwapchain.m_extent.height, 1);
	}

	// Resets the frame counter if the camera has changed
	void updateRtFrame()
	{
		// TODO: Add resetRtFrame for when camera changes
		m_pcRay.frame++;
	}

	void resetRtFrame()
	{
		m_pcRay.frame = -1;
	}
};

int main()
{
	HelloTriangleApplication app;

	try
	{
		app.run();
	}
	catch (const std::exception &e)
	{
		std::cerr << e.what() << std::endl;
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}