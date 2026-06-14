#include "VulkanContext.h"
#include "Common.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>

// ── Validation-layer configuration ───────────────────────────────────

#ifdef _DEBUG
static constexpr bool kWantValidation = true;
#else
static constexpr bool kWantValidation = false;
#endif

static const std::vector<const char*> kValidationLayers = {
    "VK_LAYER_KHRONOS_validation"
};

static const std::vector<const char*> kDeviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME
};

// ── Debug-messenger helpers ──────────────────────────────────────────

static VKAPI_ATTR VkBool32 VKAPI_CALL DebugCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT      InSeverity,
    VkDebugUtilsMessageTypeFlagsEXT             /*InType*/,
    const VkDebugUtilsMessengerCallbackDataEXT* InData,
    void*                                       /*InUserData*/)
{
    if (InSeverity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT)
        std::cerr << "[Vulkan Validation] " << InData->pMessage << "\n";
    return VK_FALSE;
}

static VkResult CreateDebugUtilsMessengerEXT(
    VkInstance InInstance,
    const VkDebugUtilsMessengerCreateInfoEXT* InInfo,
    const VkAllocationCallbacks*              InAlloc,
    VkDebugUtilsMessengerEXT*                 OutMessenger)
{
    auto Fn = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
        vkGetInstanceProcAddr(InInstance, "vkCreateDebugUtilsMessengerEXT"));
    return Fn ? Fn(InInstance, InInfo, InAlloc, OutMessenger) : VK_ERROR_EXTENSION_NOT_PRESENT;
}

static void DestroyDebugUtilsMessengerEXT(
    VkInstance                   InInstance,
    VkDebugUtilsMessengerEXT     InMessenger,
    const VkAllocationCallbacks* InAlloc)
{
    auto Fn = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
        vkGetInstanceProcAddr(InInstance, "vkDestroyDebugUtilsMessengerEXT"));
    if (Fn) Fn(InInstance, InMessenger, InAlloc);
}

static void PopulateDebugCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& OutInfo)
{
    OutInfo = {};
    OutInfo.sType           = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    OutInfo.messageSeverity  = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT
                             | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT
                             | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    OutInfo.messageType      = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT
                             | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT
                             | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    OutInfo.pfnUserCallback  = DebugCallback;
}

// ── Public interface ─────────────────────────────────────────────────

void VulkanContext::Init(GLFWwindow* InWindow)
{
    Window = InWindow;
    CreateInstance();
    SetupDebugMessenger();
    CreateSurface();
    PickPhysicalDevice();
    CreateLogicalDevice();
    CreateSwapchain();
    CreateImageViews();

    CreateGraphicsPipeline();

    CreateCommandPool();
    AllocateCommandBuffers();
    CreateSyncObjects();
}

void VulkanContext::DrawFrame()
{
    vkWaitForFences(Device, 1, &InFlightFences[CurrentFrame], VK_TRUE, UINT64_MAX);

    uint32_t ImageIndex;
    VkResult Result = vkAcquireNextImageKHR(
        Device, Swapchain, UINT64_MAX,
        ImageAvailableSemaphores[CurrentFrame], VK_NULL_HANDLE, &ImageIndex);

    if (Result == VK_ERROR_OUT_OF_DATE_KHR) { RecreateSwapchain(); return; }
    if (Result != VK_SUCCESS && Result != VK_SUBOPTIMAL_KHR)
        throw std::runtime_error("Failed to acquire swapchain image");

    vkResetFences(Device, 1, &InFlightFences[CurrentFrame]);
    vkResetCommandBuffer(CommandBuffers[CurrentFrame], 0);
    RecordCommandBuffer(CommandBuffers[CurrentFrame], ImageIndex);

    VkSemaphore          WaitSems[]   = { ImageAvailableSemaphores[CurrentFrame] };
    VkPipelineStageFlags WaitStages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
    VkSemaphore          SignalSems[] = { RenderFinishedSemaphores[ImageIndex] };

    VkSubmitInfo SubmitInfo{};
    SubmitInfo.sType                = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    SubmitInfo.waitSemaphoreCount   = 1;
    SubmitInfo.pWaitSemaphores      = WaitSems;
    SubmitInfo.pWaitDstStageMask    = WaitStages;
    SubmitInfo.commandBufferCount   = 1;
    SubmitInfo.pCommandBuffers      = &CommandBuffers[CurrentFrame];
    SubmitInfo.signalSemaphoreCount = 1;
    SubmitInfo.pSignalSemaphores    = SignalSems;

    CHECK_VK(vkQueueSubmit(GraphicsQueue, 1, &SubmitInfo, InFlightFences[CurrentFrame]), "Failed to submit draw command buffer");

    VkSwapchainKHR   Swapchains[] = { Swapchain };
    VkPresentInfoKHR PresentInfo{};
    PresentInfo.sType              = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    PresentInfo.waitSemaphoreCount = 1;
    PresentInfo.pWaitSemaphores    = SignalSems;
    PresentInfo.swapchainCount     = 1;
    PresentInfo.pSwapchains        = Swapchains;
    PresentInfo.pImageIndices      = &ImageIndex;

    Result = vkQueuePresentKHR(PresentQueue, &PresentInfo);

    if (Result == VK_ERROR_OUT_OF_DATE_KHR || Result == VK_SUBOPTIMAL_KHR || FramebufferResized) {
        FramebufferResized = false;
        RecreateSwapchain();
    } else if (Result != VK_SUCCESS) {
        DIE("Failed to present swapchain image");
    }

    CurrentFrame = (CurrentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
}

void VulkanContext::WaitIdle() { vkDeviceWaitIdle(Device); }

void VulkanContext::Cleanup()
{
    CleanupSwapchain();

    for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        vkDestroySemaphore(Device, ImageAvailableSemaphores[i], nullptr);
        vkDestroyFence(Device, InFlightFences[i], nullptr);
    }

    vkDestroyCommandPool(Device, CommandPool, nullptr);
    vkDestroyDevice(Device, nullptr);

    if (ValidationEnabled)
        DestroyDebugUtilsMessengerEXT(Instance, DebugMessenger, nullptr);

    vkDestroySurfaceKHR(Instance, Surface, nullptr);
    vkDestroyInstance(Instance, nullptr);
}

// ═════════════════════════════════════════════════════════════════════
//  Instance
// ═════════════════════════════════════════════════════════════════════

void VulkanContext::CreateInstance()
{
    // Check whether validation layers are available (requires LunarG Vulkan SDK)
    if constexpr (kWantValidation) {
        uint32_t LayerCount;
        vkEnumerateInstanceLayerProperties(&LayerCount, nullptr);
        std::vector<VkLayerProperties> Available(LayerCount);
        vkEnumerateInstanceLayerProperties(&LayerCount, Available.data());

        ValidationEnabled = true;
        for (const char* Name : kValidationLayers) {
            bool Found = false;
            for (const auto& Layer : Available)
                if (std::strcmp(Name, Layer.layerName) == 0) { Found = true; break; }
            if (!Found) {
                std::cerr << "[Warning] Validation layer not available: " << Name
                          << "\n  Install the LunarG Vulkan SDK to enable validation.\n";
                ValidationEnabled = false;
                break;
            }
        }
    }

    VkApplicationInfo AppInfo{};
    AppInfo.sType              = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    AppInfo.pApplicationName   = "Vulkan App";
    AppInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    AppInfo.pEngineName        = "No Engine";
    AppInfo.engineVersion      = VK_MAKE_VERSION(1, 0, 0);
    AppInfo.apiVersion         = VK_API_VERSION_1_3;

    // GLFW tells us which instance extensions it needs for surface creation
    uint32_t     GlfwExtCount = 0;
    const char** GlfwExts     = glfwGetRequiredInstanceExtensions(&GlfwExtCount);
    std::vector<const char*> Extensions(GlfwExts, GlfwExts + GlfwExtCount);

    if (ValidationEnabled)
        Extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);

    VkInstanceCreateInfo CreateInfo{};
    CreateInfo.sType                   = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    CreateInfo.pApplicationInfo        = &AppInfo;
    CreateInfo.enabledExtensionCount   = static_cast<uint32_t>(Extensions.size());
    CreateInfo.ppEnabledExtensionNames = Extensions.data();

    // Chain a debug messenger into instance creation so we catch errors
    // during vkCreateInstance / vkDestroyInstance as well.
    VkDebugUtilsMessengerCreateInfoEXT DebugCI{};
    if (ValidationEnabled) {
        CreateInfo.enabledLayerCount   = static_cast<uint32_t>(kValidationLayers.size());
        CreateInfo.ppEnabledLayerNames = kValidationLayers.data();
        PopulateDebugCreateInfo(DebugCI);
        CreateInfo.pNext = &DebugCI;
    }

    CHECK_VK(vkCreateInstance(&CreateInfo, nullptr, &Instance), "Failed to create Vulkan instance");
}

// ═════════════════════════════════════════════════════════════════════
//  Debug messenger
// ═════════════════════════════════════════════════════════════════════

void VulkanContext::SetupDebugMessenger()
{
    if (!ValidationEnabled) return;

    VkDebugUtilsMessengerCreateInfoEXT CreateInfo{};
    PopulateDebugCreateInfo(CreateInfo);

    CHECK_VK(CreateDebugUtilsMessengerEXT(Instance, &CreateInfo, nullptr, &DebugMessenger), "Failed to set up debug messenger");
}

// ═════════════════════════════════════════════════════════════════════
//  Surface
// ═════════════════════════════════════════════════════════════════════

void VulkanContext::CreateSurface()
{
    CHECK_VK(glfwCreateWindowSurface(Instance, Window, nullptr, &Surface), "Failed to create window surface");
}

// ═════════════════════════════════════════════════════════════════════
//  Physical device selection
// ═════════════════════════════════════════════════════════════════════

QueueFamilyIndices VulkanContext::FindQueueFamilies(VkPhysicalDevice InDevice)
{
    QueueFamilyIndices Indices;

    uint32_t Count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(InDevice, &Count, nullptr);
    std::vector<VkQueueFamilyProperties> Families(Count);
    vkGetPhysicalDeviceQueueFamilyProperties(InDevice, &Count, Families.data());

    for (uint32_t i = 0; i < Count; i++) {
        if (Families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)
            Indices.GraphicsFamily = i;

        VkBool32 PresentSupport = false;
        vkGetPhysicalDeviceSurfaceSupportKHR(InDevice, i, Surface, &PresentSupport);
        if (PresentSupport)
            Indices.PresentFamily = i;

        if (Indices.IsComplete()) break;
    }
    return Indices;
}

SwapchainSupportDetails VulkanContext::QuerySwapchainSupport(VkPhysicalDevice InDevice)
{
    SwapchainSupportDetails Details;
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(InDevice, Surface, &Details.Capabilities);

    uint32_t FormatCount;
    vkGetPhysicalDeviceSurfaceFormatsKHR(InDevice, Surface, &FormatCount, nullptr);
    if (FormatCount) {
        Details.Formats.resize(FormatCount);
        vkGetPhysicalDeviceSurfaceFormatsKHR(InDevice, Surface, &FormatCount, Details.Formats.data());
    }

    uint32_t ModeCount;
    vkGetPhysicalDeviceSurfacePresentModesKHR(InDevice, Surface, &ModeCount, nullptr);
    if (ModeCount) {
        Details.PresentModes.resize(ModeCount);
        vkGetPhysicalDeviceSurfacePresentModesKHR(InDevice, Surface, &ModeCount, Details.PresentModes.data());
    }
    return Details;
}

bool VulkanContext::IsDeviceSuitable(VkPhysicalDevice InDevice)
{
    if (!FindQueueFamilies(InDevice).IsComplete()) return false;

    // Verify required device extensions
    uint32_t ExtCount;
    vkEnumerateDeviceExtensionProperties(InDevice, nullptr, &ExtCount, nullptr);
    std::vector<VkExtensionProperties> Available(ExtCount);
    vkEnumerateDeviceExtensionProperties(InDevice, nullptr, &ExtCount, Available.data());

    std::set<std::string> Required(kDeviceExtensions.begin(), kDeviceExtensions.end());
    for (const auto& Ext : Available) Required.erase(Ext.extensionName);
    if (!Required.empty()) return false;

    // At least one format and one present mode
    auto Support = QuerySwapchainSupport(InDevice);
    return !Support.Formats.empty() && !Support.PresentModes.empty();
}

void VulkanContext::PickPhysicalDevice()
{
    uint32_t Count = 0;
    vkEnumeratePhysicalDevices(Instance, &Count, nullptr);
    CHECK_DIE(Count != 0, "No GPUs with Vulkan support found");
    
    std::vector<VkPhysicalDevice> Devices(Count);
    vkEnumeratePhysicalDevices(Instance, &Count, Devices.data());

    // Pick best suitable device, preferring discrete GPUs
    VkPhysicalDeviceProperties DeviceProps;
    for (auto Dev : Devices) 
    {
        VkPhysicalDeviceProperties CurrentProps;
        vkGetPhysicalDeviceProperties(Dev, &CurrentProps);
        LOG_DEBUG("GPU: {}", CurrentProps.deviceName);

        if (!IsDeviceSuitable(Dev)) continue;

        // Just to print all the devices, 
        // but get just the first in order of preference (discrete, then fallback)
        if (PhysicalDevice == VK_NULL_HANDLE)
        {
            if (CurrentProps.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) 
            {
                PhysicalDevice = Dev;
                DeviceProps = CurrentProps;
            } else // Fallback
            {
                PhysicalDevice = Dev;
                DeviceProps = CurrentProps;
            }
        }
    }

    CHECK_DIE(PhysicalDevice, "No suitable GPU found");

    LOG_DEBUG("GPU SELECTED: {} VK {}.{}",
        DeviceProps.deviceName,
        VK_VERSION_MAJOR(DeviceProps.apiVersion),
        VK_VERSION_MINOR(DeviceProps.apiVersion));
}

// ═════════════════════════════════════════════════════════════════════
//  Logical device & queues
// ═════════════════════════════════════════════════════════════════════

void VulkanContext::CreateLogicalDevice()
{
    QueueFamilyIndices Indices = FindQueueFamilies(PhysicalDevice);

    std::set<uint32_t> UniqueFamilies = {
        Indices.GraphicsFamily.value(),
        Indices.PresentFamily.value()
    };

    float Priority = 1.0f;
    std::vector<VkDeviceQueueCreateInfo> QueueCreateInfos;
    for (uint32_t Family : UniqueFamilies) {
        VkDeviceQueueCreateInfo QueueInfo{};
        QueueInfo.sType            = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        QueueInfo.queueFamilyIndex = Family;
        QueueInfo.queueCount       = 1;
        QueueInfo.pQueuePriorities = &Priority;
        QueueCreateInfos.push_back(QueueInfo);
    }

    VkPhysicalDeviceFeatures Features{};

    VkPhysicalDeviceDynamicRenderingFeatures DynRenderFeature{};
    DynRenderFeature.sType            = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES;
    DynRenderFeature.dynamicRendering = VK_TRUE;

    VkDeviceCreateInfo CreateInfo{};
    CreateInfo.sType                   = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    CreateInfo.pNext                   = &DynRenderFeature;
    CreateInfo.queueCreateInfoCount    = static_cast<uint32_t>(QueueCreateInfos.size());
    CreateInfo.pQueueCreateInfos       = QueueCreateInfos.data();
    CreateInfo.pEnabledFeatures        = &Features;
    CreateInfo.enabledExtensionCount   = static_cast<uint32_t>(kDeviceExtensions.size());
    CreateInfo.ppEnabledExtensionNames = kDeviceExtensions.data();

    if (ValidationEnabled) {
        CreateInfo.enabledLayerCount   = static_cast<uint32_t>(kValidationLayers.size());
        CreateInfo.ppEnabledLayerNames = kValidationLayers.data();
    }

    CHECK_VK(vkCreateDevice(PhysicalDevice, &CreateInfo, nullptr, &Device), "Failed to create logical device");

    vkGetDeviceQueue(Device, Indices.GraphicsFamily.value(), 0, &GraphicsQueue);
    vkGetDeviceQueue(Device, Indices.PresentFamily.value(), 0, &PresentQueue);
}

// ═════════════════════════════════════════════════════════════════════
//  Swapchain
// ═════════════════════════════════════════════════════════════════════

VkSurfaceFormatKHR VulkanContext::ChooseSwapSurfaceFormat(
    const std::vector<VkSurfaceFormatKHR>& InFormats)
{
    
    for (const auto& Format : InFormats)
        if (Format.format == VK_FORMAT_B8G8R8A8_SRGB &&
            Format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
            return Format;
    return InFormats[0];
}

VkPresentModeKHR VulkanContext::ChooseSwapPresentMode(
    const std::vector<VkPresentModeKHR>& InModes)
{
    if (VSyncEnabled)
    {
        return VK_PRESENT_MODE_FIFO_KHR;
    }
    for (auto Mode : InModes)
        if (Mode == VK_PRESENT_MODE_MAILBOX_KHR) return Mode;
    return VK_PRESENT_MODE_FIFO_KHR; // guaranteed available
}

VkExtent2D VulkanContext::ChooseSwapExtent(const VkSurfaceCapabilitiesKHR& InCaps)
{
    if (InCaps.currentExtent.width != (std::numeric_limits<uint32_t>::max)())
        return InCaps.currentExtent;

    int W, H;
    glfwGetFramebufferSize(Window, &W, &H);
    VkExtent2D Extent{ static_cast<uint32_t>(W), static_cast<uint32_t>(H) };
    Extent.width  = std::clamp(Extent.width,  InCaps.minImageExtent.width,  InCaps.maxImageExtent.width);
    Extent.height = std::clamp(Extent.height, InCaps.minImageExtent.height, InCaps.maxImageExtent.height);
    return Extent;
}

void VulkanContext::CreateSwapchain()
{
    auto Support = QuerySwapchainSupport(PhysicalDevice);
    auto Format  = ChooseSwapSurfaceFormat(Support.Formats);
    auto Mode    = ChooseSwapPresentMode(Support.PresentModes);
    auto Extent  = ChooseSwapExtent(Support.Capabilities);

    uint32_t ImageCount = Support.Capabilities.minImageCount + 1;
    if (Support.Capabilities.maxImageCount > 0 && ImageCount > Support.Capabilities.maxImageCount)
        ImageCount = Support.Capabilities.maxImageCount;

    VkSwapchainCreateInfoKHR CreateInfo{};
    CreateInfo.sType            = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    CreateInfo.surface          = Surface;
    CreateInfo.minImageCount    = ImageCount;
    CreateInfo.imageFormat      = Format.format;
    CreateInfo.imageColorSpace  = Format.colorSpace;
    CreateInfo.imageExtent      = Extent;
    CreateInfo.imageArrayLayers = 1;
    CreateInfo.imageUsage       = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

    QueueFamilyIndices Indices   = FindQueueFamilies(PhysicalDevice);
    uint32_t FamilyIndices[]     = { Indices.GraphicsFamily.value(), Indices.PresentFamily.value() };

    if (Indices.GraphicsFamily != Indices.PresentFamily) {
        CreateInfo.imageSharingMode      = VK_SHARING_MODE_CONCURRENT;
        CreateInfo.queueFamilyIndexCount = 2;
        CreateInfo.pQueueFamilyIndices   = FamilyIndices;
    } else {
        CreateInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    }

    CreateInfo.preTransform   = Support.Capabilities.currentTransform;
    CreateInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    CreateInfo.presentMode    = Mode;
    CreateInfo.clipped        = VK_TRUE;
    CreateInfo.oldSwapchain   = VK_NULL_HANDLE;

    CHECK_VK(vkCreateSwapchainKHR(Device, &CreateInfo, nullptr, &Swapchain), "Failed to create swapchain");

    vkGetSwapchainImagesKHR(Device, Swapchain, &ImageCount, nullptr);
    SwapchainImages.resize(ImageCount);
    vkGetSwapchainImagesKHR(Device, Swapchain, &ImageCount, SwapchainImages.data());

    SwapchainImageFormat = Format.format;
    SwapchainExtent      = Extent;
}

// ═════════════════════════════════════════════════════════════════════
//  Image views
// ═════════════════════════════════════════════════════════════════════

void VulkanContext::CreateImageViews()
{
    SwapchainImageViews.resize(SwapchainImages.size());

    for (size_t i = 0; i < SwapchainImages.size(); i++) {
        VkImageViewCreateInfo CreateInfo{};
        CreateInfo.sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        CreateInfo.image    = SwapchainImages[i];
        CreateInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        CreateInfo.format   = SwapchainImageFormat;
        CreateInfo.components = { VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY,
                                  VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY };
        CreateInfo.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
        CreateInfo.subresourceRange.baseMipLevel   = 0;
        CreateInfo.subresourceRange.levelCount     = 1;
        CreateInfo.subresourceRange.baseArrayLayer = 0;
        CreateInfo.subresourceRange.layerCount     = 1;

        CHECK_VK(vkCreateImageView(Device, &CreateInfo, nullptr, &SwapchainImageViews[i]), "Failed to create image view");
    }
}

// ═════════════════════════════════════════════════════════════════════
//  Command pool & buffers
// ═════════════════════════════════════════════════════════════════════

void VulkanContext::CreateCommandPool()
{
    QueueFamilyIndices Indices = FindQueueFamilies(PhysicalDevice);

    VkCommandPoolCreateInfo CreateInfo{};
    CreateInfo.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    CreateInfo.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    CreateInfo.queueFamilyIndex = Indices.GraphicsFamily.value();

    CHECK_VK(vkCreateCommandPool(Device, &CreateInfo, nullptr, &CommandPool), "Failed to create command pool");
}

void VulkanContext::AllocateCommandBuffers()
{
    CommandBuffers.resize(MAX_FRAMES_IN_FLIGHT);

    VkCommandBufferAllocateInfo AllocInfo{};
    AllocInfo.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    AllocInfo.commandPool        = CommandPool;
    AllocInfo.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    AllocInfo.commandBufferCount = static_cast<uint32_t>(CommandBuffers.size());

    CHECK_VK(vkAllocateCommandBuffers(Device, &AllocInfo, CommandBuffers.data()), "Failed to allocate command buffers");
}

// ═════════════════════════════════════════════════════════════════════
//  Synchronisation primitives
// ═════════════════════════════════════════════════════════════════════

void VulkanContext::CreateRenderFinishedSemaphores()
{
    VkSemaphoreCreateInfo SemaphoreInfo{};
    SemaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    RenderFinishedSemaphores.resize(SwapchainImages.size());
    for (size_t i = 0; i < SwapchainImages.size(); i++) {
        CHECK_VK(vkCreateSemaphore(Device, &SemaphoreInfo, nullptr, &RenderFinishedSemaphores[i]), "Failed to create render-finished semaphore");
    }
}

void VulkanContext::CreateSyncObjects()
{
    ImageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
    InFlightFences.resize(MAX_FRAMES_IN_FLIGHT);

    VkSemaphoreCreateInfo SemaphoreInfo{};
    SemaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    VkFenceCreateInfo FenceInfo{};
    FenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    FenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT; // start signalled so first wait succeeds

    for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        CHECK_VK(vkCreateSemaphore(Device, &SemaphoreInfo, nullptr, &ImageAvailableSemaphores[i]), "Failed to create image-available semaphore");
        CHECK_VK(vkCreateFence(Device, &FenceInfo, nullptr, &InFlightFences[i]), "Failed to create in-flight fence");
    }

    // One render-finished semaphore per swapchain image so that a semaphore
    // used in vkQueuePresentKHR is not reused until that image is re-acquired.
    CreateRenderFinishedSemaphores();
}

// ═════════════════════════════════════════════════════════════════════
//  Command buffer recording  (clear screen only for Lesson 1)
// ═════════════════════════════════════════════════════════════════════

void VulkanContext::RecordCommandBuffer(VkCommandBuffer InCmd, uint32_t InImageIndex)
{
    VkCommandBufferBeginInfo BeginInfo{};
    BeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

    CHECK_VK(vkBeginCommandBuffer(InCmd, &BeginInfo), "Failed to begin recording command buffer");

    // Transition: UNDEFINED -> COLOR_ATTACHMENT_OPTIMAL
    VkImageMemoryBarrier ToColourBarrier{};
    ToColourBarrier.sType                           = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    ToColourBarrier.srcAccessMask                    = 0;
    ToColourBarrier.dstAccessMask                    = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    ToColourBarrier.oldLayout                        = VK_IMAGE_LAYOUT_UNDEFINED;
    ToColourBarrier.newLayout                        = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    ToColourBarrier.srcQueueFamilyIndex              = VK_QUEUE_FAMILY_IGNORED;
    ToColourBarrier.dstQueueFamilyIndex              = VK_QUEUE_FAMILY_IGNORED;
    ToColourBarrier.image                            = SwapchainImages[InImageIndex];
    ToColourBarrier.subresourceRange.aspectMask      = VK_IMAGE_ASPECT_COLOR_BIT;
    ToColourBarrier.subresourceRange.baseMipLevel    = 0;
    ToColourBarrier.subresourceRange.levelCount      = 1;
    ToColourBarrier.subresourceRange.baseArrayLayer  = 0;
    ToColourBarrier.subresourceRange.layerCount      = 1;

    vkCmdPipelineBarrier(InCmd,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        0, 0, nullptr, 0, nullptr, 1, &ToColourBarrier);

    // Dynamic rendering — no VkRenderPass or VkFramebuffer needed
    VkClearValue ClearColour = {{{ 0.39f, 0.58f, 0.93f, 1.0f }}}; // cornflower blue

    VkRenderingAttachmentInfo ColourAttachment{};
    ColourAttachment.sType       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    ColourAttachment.imageView   = SwapchainImageViews[InImageIndex];
    ColourAttachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    ColourAttachment.loadOp      = VK_ATTACHMENT_LOAD_OP_CLEAR;
    ColourAttachment.storeOp     = VK_ATTACHMENT_STORE_OP_STORE;
    ColourAttachment.clearValue  = ClearColour;

    VkRenderingInfo RenderInfo{};
    RenderInfo.sType                = VK_STRUCTURE_TYPE_RENDERING_INFO;
    RenderInfo.renderArea.offset    = { 0, 0 };
    RenderInfo.renderArea.extent    = SwapchainExtent;
    RenderInfo.layerCount           = 1;
    RenderInfo.colorAttachmentCount = 1;
    RenderInfo.pColorAttachments    = &ColourAttachment;

    vkCmdBeginRendering(InCmd, &RenderInfo);
    
    vkCmdBindPipeline(InCmd, VK_PIPELINE_BIND_POINT_GRAPHICS, GraphicsPipeline);

    VkViewport Viewport{};
    Viewport.width = static_cast<float>(SwapchainExtent.width);
    Viewport.height = static_cast<float>(SwapchainExtent.height);
    Viewport.minDepth = 0.0f;
    Viewport.maxDepth = 1.0f;
    vkCmdSetViewport(InCmd, 0, 1, &Viewport);

    VkRect2D Scissor{};
    Scissor.extent = SwapchainExtent;
    vkCmdSetScissor(InCmd, 0, 1, &Scissor);

    vkCmdDraw(InCmd, 3, 1, 0, 0);

    vkCmdEndRendering(InCmd);

    // Transition: COLOR_ATTACHMENT_OPTIMAL -> PRESENT_SRC_KHR
    VkImageMemoryBarrier ToPresentBarrier{};
    ToPresentBarrier.sType                           = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    ToPresentBarrier.srcAccessMask                    = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    ToPresentBarrier.dstAccessMask                    = 0;
    ToPresentBarrier.oldLayout                        = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    ToPresentBarrier.newLayout                        = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    ToPresentBarrier.srcQueueFamilyIndex              = VK_QUEUE_FAMILY_IGNORED;
    ToPresentBarrier.dstQueueFamilyIndex              = VK_QUEUE_FAMILY_IGNORED;
    ToPresentBarrier.image                            = SwapchainImages[InImageIndex];
    ToPresentBarrier.subresourceRange.aspectMask      = VK_IMAGE_ASPECT_COLOR_BIT;
    ToPresentBarrier.subresourceRange.baseMipLevel    = 0;
    ToPresentBarrier.subresourceRange.levelCount      = 1;
    ToPresentBarrier.subresourceRange.baseArrayLayer  = 0;
    ToPresentBarrier.subresourceRange.layerCount      = 1;

    vkCmdPipelineBarrier(InCmd,
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
        0, 0, nullptr, 0, nullptr, 1, &ToPresentBarrier);

    CHECK_VK(vkEndCommandBuffer(InCmd), "Failed to record command buffer");
}

// ═════════════════════════════════════════════════════════════════════
//  Swapchain recreation  (handles window resize / minimise)
// ═════════════════════════════════════════════════════════════════════

void VulkanContext::CleanupSwapchain()
{
    vkDestroyPipeline(Device, GraphicsPipeline, nullptr);
    vkDestroyPipelineLayout(Device, PipelineLayout, nullptr);

    for (auto Sem : RenderFinishedSemaphores) vkDestroySemaphore(Device, Sem, nullptr);
    RenderFinishedSemaphores.clear();//my add fix??
    for (auto Iv : SwapchainImageViews)       vkDestroyImageView(Device, Iv, nullptr);
    vkDestroySwapchainKHR(Device, Swapchain, nullptr);
}

void VulkanContext::RecreateSwapchain()
{
    // Handle minimisation — wait until the window has a non-zero size again
    int W = 0, H = 0;
    glfwGetFramebufferSize(Window, &W, &H);
    while (W == 0 || H == 0) {
        glfwGetFramebufferSize(Window, &W, &H);
        glfwWaitEvents();
    }

    vkDeviceWaitIdle(Device);

    CleanupSwapchain();
    CreateSwapchain();
    CreateImageViews();//TODO have to add here new pipeline gr? search on the net
    CreateGraphicsPipeline();//my add source Vulkan to fix the resize error form + gemini doc reading
    CreateRenderFinishedSemaphores();
}

VkShaderModule VulkanContext::CreateShaderModule(const std::string& InPath)
{
    std::ifstream File(InPath, std::ios::ate | std::ios::binary);
    CHECK_DIE(File.is_open(), "Failed to open shader file");

    size_t FileSize = static_cast<size_t>(File.tellg());
    std::vector<char> Code(FileSize);
    File.seekg(0);
    File.read(Code.data(), FileSize);

    VkShaderModuleCreateInfo CreateInfo{};
    CreateInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    CreateInfo.codeSize = Code.size();
    CreateInfo.pCode = reinterpret_cast<const uint32_t*>(Code.data());

    VkShaderModule Module;
    CHECK_VK(vkCreateShaderModule(Device, &CreateInfo, nullptr, &Module), "Failed creating shader module");
    return Module;
}

//my add for Vsync
void VulkanContext::SetVSync(bool InEnabled)
{
    if (VSyncEnabled != InEnabled)
    {
        VSyncEnabled = InEnabled;
        RecreateSwapchain();
    }
}



void VulkanContext::CreateGraphicsPipeline()
{
    VkShaderModule VertModule = CreateShaderModule("resources/shaders/triangle.vert.spv");
    VkShaderModule FragModule = CreateShaderModule("resources/shaders/triangle.frag.spv");

    VkPipelineShaderStageCreateInfo ShaderStages[2]{};
    ShaderStages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    ShaderStages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    ShaderStages[0].module = VertModule;
    ShaderStages[0].pName = "main";
    ShaderStages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    ShaderStages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    ShaderStages[1].module = FragModule;
    ShaderStages[1].pName = "main";

    VkPipelineVertexInputStateCreateInfo VertexInput{};
    VertexInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

    VkPipelineInputAssemblyStateCreateInfo InputAssembly{};
    InputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    InputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkPipelineRasterizationStateCreateInfo Rasterizer{};
    Rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    Rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    Rasterizer.lineWidth = 1.0f;
    Rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
    Rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;

    VkDynamicState DynamicStates[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR }; 
    VkPipelineDynamicStateCreateInfo DynamicState{};
    DynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    DynamicState.dynamicStateCount = 2;
    DynamicState.pDynamicStates = DynamicStates;

    VkPipelineViewportStateCreateInfo ViewportState{};
    ViewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    ViewportState.viewportCount = 1;
    ViewportState.scissorCount = 1;

    VkPipelineMultisampleStateCreateInfo Multisampling{};
    Multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    Multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineColorBlendAttachmentState BlendAttachment{};
    BlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                     VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

    VkPipelineColorBlendStateCreateInfo ColorBlend{};
    ColorBlend.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    ColorBlend.attachmentCount = 1;
    ColorBlend.pAttachments = &BlendAttachment;     

    VkPipelineRenderingCreateInfo RenderingInfo{};
    RenderingInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    RenderingInfo.colorAttachmentCount = 1;
    RenderingInfo.pColorAttachmentFormats = &SwapchainImageFormat;


    VkPipelineLayoutCreateInfo LayoutInfo{};
    LayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    CHECK_VK(vkCreatePipelineLayout(Device, &LayoutInfo, nullptr, &PipelineLayout), "Failed creating layout");

    VkGraphicsPipelineCreateInfo PipelineInfo{};
    PipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    PipelineInfo.pNext = &RenderingInfo;
    PipelineInfo.stageCount = 2;
    PipelineInfo.pStages = ShaderStages;
    PipelineInfo.pVertexInputState = &VertexInput;
    PipelineInfo.pInputAssemblyState = &InputAssembly;
    PipelineInfo.pRasterizationState = &Rasterizer;
    PipelineInfo.pViewportState = &ViewportState;
    PipelineInfo.pDynamicState = &DynamicState;
    PipelineInfo.pMultisampleState = &Multisampling;
    PipelineInfo.pColorBlendState = &ColorBlend;
    PipelineInfo.layout = PipelineLayout;
    PipelineInfo.renderPass = VK_NULL_HANDLE;

    CHECK_VK(vkCreateGraphicsPipelines(Device, VK_NULL_HANDLE, 1, &PipelineInfo, nullptr, &GraphicsPipeline), "Failed creating pipeline");

    vkDestroyShaderModule(Device, VertModule, nullptr);
    vkDestroyShaderModule(Device, FragModule, nullptr);



}
