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
    void SetVSync(bool InEnabled);

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
    bool                         VSyncEnabled = true;   //my add
    bool                         ValidationEnabled  = false;

    // Pipeline
    VkPipeline                   GraphicsPipeline = VK_NULL_HANDLE;
    VkPipelineLayout             PipelineLayout = VK_NULL_HANDLE;

    // Initialisation steps (called in order by Init)
    void CreateInstance();
    void SetupDebugMessenger();
    void CreateSurface();
    void PickPhysicalDevice();
    void CreateLogicalDevice();
    void CreateSwapchain();
    void CreateImageViews();
    void CreateCommandPool();
    void AllocateCommandBuffers();
    void CreateRenderFinishedSemaphores();
    void CreateSyncObjects();

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
};
