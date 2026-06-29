#pragma once

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <vector>
#include <optional>
#include <string>

struct QueueFamilyIndices
{
    std::optional<uint32_t> GraphicsFamily;
    std::optional<uint32_t> PresentFamily;
    bool IsComplete() const { return GraphicsFamily.has_value() && PresentFamily.has_value(); }
};

struct SwapchainSupportDetails
{
    VkSurfaceCapabilitiesKHR        Capabilities{};
    std::vector<VkSurfaceFormatKHR> Formats;
    std::vector<VkPresentModeKHR>   PresentModes;
};

class VulkanContext
{
public:
    static constexpr int MAX_FRAMES_IN_FLIGHT = 2;

    void Init(GLFWwindow* InWindow);
    void DrawFrame();
    void WaitIdle();
    void Cleanup();

    void NotifyFramebufferResized() { FramebufferResized = true; }

private:
    GLFWwindow*                  Window = nullptr;

    // Core
    VkInstance                   Instance       = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT     DebugMessenger = VK_NULL_HANDLE;
    VkSurfaceKHR                 Surface        = VK_NULL_HANDLE;
    VkPhysicalDevice             PhysicalDevice = VK_NULL_HANDLE;
    VkDevice                     Device         = VK_NULL_HANDLE;
    VkQueue                      GraphicsQueue  = VK_NULL_HANDLE;
    VkQueue                      PresentQueue   = VK_NULL_HANDLE;

    // Swapchain
    VkSwapchainKHR               Swapchain = VK_NULL_HANDLE;
    VkFormat                     SwapchainImageFormat{};
    VkExtent2D                   SwapchainExtent{};
    std::vector<VkImage>         SwapchainImages;
    std::vector<VkImageView>     SwapchainImageViews;

    // Commands
    VkCommandPool                CommandPool = VK_NULL_HANDLE;
    std::vector<VkCommandBuffer> CommandBuffers;

    // Synchronisation
    std::vector<VkSemaphore>     ImageAvailableSemaphores;
    std::vector<VkSemaphore>     RenderFinishedSemaphores;
    std::vector<VkFence>         InFlightFences;
    uint32_t                     CurrentFrame       = 0;
    bool                         FramebufferResized = false;
    bool                         ValidationEnabled  = false;

    // Pipeline
    VkPipeline                   GraphicsPipeline = VK_NULL_HANDLE;
    VkPipelineLayout             PipelineLayout = VK_NULL_HANDLE;

    // Vertex Buffer
    uint32_t                     VertexCount = 0;
    VkBuffer                     VertexBuffer = VK_NULL_HANDLE;
    VkDeviceMemory               VertexBufferMemory = VK_NULL_HANDLE;

    // Model Pipeline
    VkImage                      DepthImage = VK_NULL_HANDLE;
    VkDeviceMemory               DepthImageMemory = VK_NULL_HANDLE;
    VkImageView                  DepthImageView = VK_NULL_HANDLE;
    uint32_t                     IndexCount = 0;
    VkBuffer                     IndexBuffer = VK_NULL_HANDLE;
    VkDeviceMemory               IndexBufferMemory = VK_NULL_HANDLE;

    VkPipeline                   ModelPipeline = VK_NULL_HANDLE;
    VkPipelineLayout             ModelPipelineLayout = VK_NULL_HANDLE;

    // Texture
    VkImage               TextureImage = VK_NULL_HANDLE;
    VkDeviceMemory        TextureImageMemory = VK_NULL_HANDLE;
    VkImageView           TextureImageView = VK_NULL_HANDLE;
    VkSampler             TextureSampler = VK_NULL_HANDLE;

    VkDescriptorSetLayout TextureDescriptorSetLayout = VK_NULL_HANDLE;
    VkDescriptorPool      TextureDescriptorPool = VK_NULL_HANDLE;
    VkDescriptorSet       TextureDescriptorSet = VK_NULL_HANDLE;

    // Initialisation steps (called in order by Init)
    void CreateInstance();
    void SetupDebugMessenger();
    void CreateSurface();
    void PickPhysicalDevice();
    void CreateLogicalDevice();
    void CreateSwapchain();
    void CreateImageViews();
    void CreateRenderFinishedSemaphores();

    void CreateDepthResources();

    void CreateTextureDescriptorSetLayout();
    void CreateModelPipeline();
    void CreateModelBuffers();

    void CreateCommandPool();
    void CreateDescriptorPool();
    void CreateTextureSampler();

    void CreateTextureImage();
    void CreateTextureImageView();
    void CreateDescriptorSets();

    void AllocateCommandBuffers();
    void CreateSyncObjects();


    VkCommandBuffer BeginSingleTimeCommands();
    void EndSingleTimeCommands(VkCommandBuffer Cmd);
    void TransitionImageLayout(VkCommandBuffer Cmd, VkImage Image, VkImageLayout OldLayout, VkImageLayout NewLayout);
    void CopyBufferToImage(VkCommandBuffer Cmd, VkBuffer Buffer, VkImage Image, uint32_t Width, uint32_t Height);

    // Runtime helpers
    void RecordCommandBuffer(VkCommandBuffer InCmd, uint32_t InImageIndex);
    void RecreateSwapchain();
    void CleanupSwapchain();

    // Query helpers
    QueueFamilyIndices      FindQueueFamilies(VkPhysicalDevice InDevice);
    SwapchainSupportDetails QuerySwapchainSupport(VkPhysicalDevice InDevice);
    bool                    IsDeviceSuitable(VkPhysicalDevice InDevice);
    VkSurfaceFormatKHR      ChooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& InFormats);
    VkPresentModeKHR        ChooseSwapPresentMode(const std::vector<VkPresentModeKHR>& InModes);
    VkExtent2D              ChooseSwapExtent(const VkSurfaceCapabilitiesKHR& InCaps);

    VkShaderModule          CreateShaderModule(const std::string& InPath);
    void                    CreateGraphicsPipeline();

    void                    CreateVertexBuffer();
    uint32_t                FindMemoryType(uint32_t InTypeFilter, VkMemoryPropertyFlags InProps);


};
