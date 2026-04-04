#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <vector>

#include "render/renderer.hpp"
#include "render/text_overlay.hpp"
#include "triangulation/triangulator.hpp"
#include "core/logging.h"
#include "core/pipeline_config.h"

// Vertex: position (xyz) + color (rgba) = 28 bytes
struct vertex {
  float x, y, z;
  float r, g, b, a;
};

// Body part color assignments
static void assign_body_colors(float* colors, int num_kp) {
  // default: dim white
  for (int i = 0; i < num_kp; i++) {
    colors[i * 4 + 0] = 0.5f;
    colors[i * 4 + 1] = 0.5f;
    colors[i * 4 + 2] = 0.5f;
    colors[i * 4 + 3] = 1.0f;
  }

  auto set_range = [&](int start, int end, float r, float g, float b) {
    for (int i = start; i <= end && i < num_kp; i++) {
      colors[i * 4 + 0] = r;
      colors[i * 4 + 1] = g;
      colors[i * 4 + 2] = b;
    }
  };

  // COCO WholeBody 133 coloring
  // head (nose, eyes, ears)
  set_range(0, 4, 1.0f, 0.7f, 0.0f);
  // body (shoulders, hips)
  for (int i : {5, 6, 11, 12}) {
    if (i < num_kp) {
      colors[i*4+0] = 0.0f; colors[i*4+1] = 1.0f; colors[i*4+2] = 0.5f;
    }
  }
  // left arm
  for (int i : {7, 9}) {
    if (i < num_kp) {
      colors[i*4+0] = 0.3f; colors[i*4+1] = 1.0f; colors[i*4+2] = 0.3f;
    }
  }
  // right arm
  for (int i : {8, 10}) {
    if (i < num_kp) {
      colors[i*4+0] = 0.3f; colors[i*4+1] = 1.0f; colors[i*4+2] = 0.3f;
    }
  }
  // left hand (91-111)
  set_range(91, 111, 0.3f, 0.7f, 1.0f);
  // right hand (112-132)
  set_range(112, 132, 1.0f, 0.3f, 1.0f);
  // left leg
  for (int i : {13, 15}) {
    if (i < num_kp) {
      colors[i*4+0] = 0.0f; colors[i*4+1] = 1.0f; colors[i*4+2] = 0.8f;
    }
  }
  // right leg
  for (int i : {14, 16}) {
    if (i < num_kp) {
      colors[i*4+0] = 0.0f; colors[i*4+1] = 1.0f; colors[i*4+2] = 0.8f;
    }
  }
  // feet (17-22)
  set_range(17, 22, 1.0f, 0.3f, 0.3f);
  // face (23-90)
  set_range(23, 90, 0.8f, 0.6f, 0.0f);
}

// Simple matrix helpers (column-major for Vulkan/GLM convention)
static void mat4_identity(float* m) {
  memset(m, 0, 64);
  m[0] = m[5] = m[10] = m[15] = 1.0f;
}

static void mat4_perspective(float* m, float fovy, float aspect, float near, float far) {
  memset(m, 0, 64);
  float f = 1.0f / tanf(fovy * 0.5f);
  m[0]  = f / aspect;
  m[5]  = -f; // Vulkan Y is flipped
  m[10] = far / (near - far);
  m[11] = -1.0f;
  m[14] = (near * far) / (near - far);
}

static void mat4_look_at(float* m, const float* eye, const float* center, const float* up) {
  float f[3] = {center[0]-eye[0], center[1]-eye[1], center[2]-eye[2]};
  float len = sqrtf(f[0]*f[0] + f[1]*f[1] + f[2]*f[2]);
  f[0] /= len; f[1] /= len; f[2] /= len;

  float s[3] = {f[1]*up[2]-f[2]*up[1], f[2]*up[0]-f[0]*up[2], f[0]*up[1]-f[1]*up[0]};
  len = sqrtf(s[0]*s[0] + s[1]*s[1] + s[2]*s[2]);
  s[0] /= len; s[1] /= len; s[2] /= len;

  float u[3] = {s[1]*f[2]-s[2]*f[1], s[2]*f[0]-s[0]*f[2], s[0]*f[1]-s[1]*f[0]};

  mat4_identity(m);
  m[0] = s[0]; m[4] = s[1]; m[8]  = s[2];
  m[1] = u[0]; m[5] = u[1]; m[9]  = u[2];
  m[2] = -f[0]; m[6] = -f[1]; m[10] = -f[2];
  m[12] = -(s[0]*eye[0]+s[1]*eye[1]+s[2]*eye[2]);
  m[13] = -(u[0]*eye[0]+u[1]*eye[1]+u[2]*eye[2]);
  m[14] = (f[0]*eye[0]+f[1]*eye[1]+f[2]*eye[2]);
}

Renderer::Renderer(const renderer_config& config, pipeline_config* pipe_config)
  : m_width(config.width), m_height(config.height),
    m_num_keypoints(config.num_keypoints), m_num_edges(config.num_edges),
    m_cam_azimuth(0.0f), m_cam_elevation(30.0f), m_cam_distance(2000.0f),
    m_cam_roll(0.0f), m_mouse_dragging(false), m_selected_axis(-1),
    m_pipe_config(pipe_config)
{
  m_cam_target[0] = 0; m_cam_target[1] = 0; m_cam_target[2] = 1000;
  init_colors();

  glfwInit();
  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
  m_window = glfwCreateWindow(m_width, m_height, "Mocap 3D", nullptr, nullptr);
  glfwSetWindowUserPointer(m_window, this);
  glfwSetMouseButtonCallback(m_window, mouse_button_callback);
  glfwSetCursorPosCallback(m_window, cursor_pos_callback);
  glfwSetScrollCallback(m_window, scroll_callback);

  create_instance();
  create_surface();
  pick_physical_device();
  create_device();
  create_swapchain();
  create_depth_resources();
  create_render_pass();
  create_descriptor_layout();
  create_pipelines();
  create_buffers(config.skeleton_edges, config.num_edges);
  create_descriptor_set();
  create_sync_objects();
  create_command_pool();

  m_text_overlay = new TextOverlay(m_device, m_physical_device, m_render_pass, m_width, m_height);
  create_gizmo();
}

Renderer::~Renderer() {
  vkDeviceWaitIdle(m_device);

  delete m_text_overlay;
  vkDestroyCommandPool(m_device, m_command_pool, nullptr);
  vkDestroySemaphore(m_device, m_image_available, nullptr);
  vkDestroySemaphore(m_device, m_render_finished, nullptr);
  vkDestroyFence(m_device, m_in_flight, nullptr);

  vkDestroyBuffer(m_device, m_vertex_buffer, nullptr);
  vkFreeMemory(m_device, m_vertex_memory, nullptr);
  vkDestroyBuffer(m_device, m_line_index_buffer, nullptr);
  vkFreeMemory(m_device, m_line_index_memory, nullptr);
  vkDestroyBuffer(m_device, m_uniform_buffer, nullptr);
  vkFreeMemory(m_device, m_uniform_memory, nullptr);
  vkDestroyBuffer(m_device, m_gizmo_vb, nullptr);
  vkFreeMemory(m_device, m_gizmo_vb_memory, nullptr);
  vkDestroyBuffer(m_device, m_gizmo_ub, nullptr);
  vkFreeMemory(m_device, m_gizmo_ub_memory, nullptr);

  vkDestroyDescriptorPool(m_device, m_descriptor_pool, nullptr);
  vkDestroyDescriptorSetLayout(m_device, m_descriptor_set_layout, nullptr);

  vkDestroyPipeline(m_device, m_line_pipeline, nullptr);
  vkDestroyPipeline(m_device, m_point_pipeline, nullptr);
  vkDestroyPipelineLayout(m_device, m_pipeline_layout, nullptr);

  for (auto fb : m_framebuffers) vkDestroyFramebuffer(m_device, fb, nullptr);
  vkDestroyRenderPass(m_device, m_render_pass, nullptr);

  vkDestroyImageView(m_device, m_depth_view, nullptr);
  vkDestroyImage(m_device, m_depth_image, nullptr);
  vkFreeMemory(m_device, m_depth_memory, nullptr);

  for (auto iv : m_swapchain_image_views) vkDestroyImageView(m_device, iv, nullptr);
  vkDestroySwapchainKHR(m_device, m_swapchain, nullptr);
  vkDestroySurfaceKHR(m_instance, m_surface, nullptr);
  vkDestroyDevice(m_device, nullptr);
  vkDestroyInstance(m_instance, nullptr);

  glfwDestroyWindow(m_window);
  glfwTerminate();
}

void Renderer::init_colors() {
  assign_body_colors(m_colors, m_num_keypoints);
}

void Renderer::create_instance() {
  VkApplicationInfo app_info{};
  app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  app_info.pApplicationName = "Mocap Renderer";
  app_info.apiVersion = VK_API_VERSION_1_2;

  uint32_t glfw_ext_count;
  const char** glfw_exts = glfwGetRequiredInstanceExtensions(&glfw_ext_count);

  VkInstanceCreateInfo info{};
  info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  info.pApplicationInfo = &app_info;
  info.enabledExtensionCount = glfw_ext_count;
  info.ppEnabledExtensionNames = glfw_exts;

  if (vkCreateInstance(&info, nullptr, &m_instance) != VK_SUCCESS)
    throw std::runtime_error("Failed to create Vulkan instance");
}

void Renderer::create_surface() {
  if (glfwCreateWindowSurface(m_instance, m_window, nullptr, &m_surface) != VK_SUCCESS)
    throw std::runtime_error("Failed to create window surface");
}

void Renderer::pick_physical_device() {
  uint32_t count = 0;
  vkEnumeratePhysicalDevices(m_instance, &count, nullptr);
  std::vector<VkPhysicalDevice> devices(count);
  vkEnumeratePhysicalDevices(m_instance, &count, devices.data());
  m_physical_device = devices[0]; // pick first (should be the 4080 Super)
}

void Renderer::create_device() {
  // find graphics queue family
  uint32_t qcount;
  vkGetPhysicalDeviceQueueFamilyProperties(m_physical_device, &qcount, nullptr);
  std::vector<VkQueueFamilyProperties> qprops(qcount);
  vkGetPhysicalDeviceQueueFamilyProperties(m_physical_device, &qcount, qprops.data());

  m_queue_family = 0;
  for (uint32_t i = 0; i < qcount; i++) {
    VkBool32 present = VK_FALSE;
    vkGetPhysicalDeviceSurfaceSupportKHR(m_physical_device, i, m_surface, &present);
    if ((qprops[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) && present) {
      m_queue_family = i;
      break;
    }
  }

  float priority = 1.0f;
  VkDeviceQueueCreateInfo queue_info{};
  queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
  queue_info.queueFamilyIndex = m_queue_family;
  queue_info.queueCount = 1;
  queue_info.pQueuePriorities = &priority;

  VkPhysicalDeviceFeatures features{};
  features.wideLines = VK_TRUE;
  features.largePoints = VK_TRUE;

  const char* ext = VK_KHR_SWAPCHAIN_EXTENSION_NAME;
  VkDeviceCreateInfo info{};
  info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  info.queueCreateInfoCount = 1;
  info.pQueueCreateInfos = &queue_info;
  info.enabledExtensionCount = 1;
  info.ppEnabledExtensionNames = &ext;
  info.pEnabledFeatures = &features;

  if (vkCreateDevice(m_physical_device, &info, nullptr, &m_device) != VK_SUCCESS)
    throw std::runtime_error("Failed to create Vulkan device");

  vkGetDeviceQueue(m_device, m_queue_family, 0, &m_graphics_queue);
}

void Renderer::create_swapchain() {
  VkSurfaceCapabilitiesKHR caps;
  vkGetPhysicalDeviceSurfaceCapabilitiesKHR(m_physical_device, m_surface, &caps);

  m_swapchain_format = VK_FORMAT_B8G8R8A8_SRGB;
  m_swapchain_extent = {static_cast<uint32_t>(m_width), static_cast<uint32_t>(m_height)};

  VkSwapchainCreateInfoKHR info{};
  info.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
  info.surface = m_surface;
  info.minImageCount = 2;
  info.imageFormat = m_swapchain_format;
  info.imageColorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
  info.imageExtent = m_swapchain_extent;
  info.imageArrayLayers = 1;
  info.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
  info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
  info.preTransform = caps.currentTransform;
  info.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
  info.presentMode = VK_PRESENT_MODE_FIFO_KHR;
  info.clipped = VK_TRUE;

  if (vkCreateSwapchainKHR(m_device, &info, nullptr, &m_swapchain) != VK_SUCCESS)
    throw std::runtime_error("Failed to create swapchain");

  uint32_t count;
  vkGetSwapchainImagesKHR(m_device, m_swapchain, &count, nullptr);
  m_swapchain_images.resize(count);
  vkGetSwapchainImagesKHR(m_device, m_swapchain, &count, m_swapchain_images.data());

  m_swapchain_image_views.resize(count);
  for (uint32_t i = 0; i < count; i++) {
    VkImageViewCreateInfo iv_info{};
    iv_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    iv_info.image = m_swapchain_images[i];
    iv_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
    iv_info.format = m_swapchain_format;
    iv_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    iv_info.subresourceRange.levelCount = 1;
    iv_info.subresourceRange.layerCount = 1;
    vkCreateImageView(m_device, &iv_info, nullptr, &m_swapchain_image_views[i]);
  }
}

void Renderer::create_depth_resources() {
  VkFormat depth_format = VK_FORMAT_D32_SFLOAT;

  VkImageCreateInfo img_info{};
  img_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  img_info.imageType = VK_IMAGE_TYPE_2D;
  img_info.format = depth_format;
  img_info.extent = {m_swapchain_extent.width, m_swapchain_extent.height, 1};
  img_info.mipLevels = 1;
  img_info.arrayLayers = 1;
  img_info.samples = VK_SAMPLE_COUNT_1_BIT;
  img_info.tiling = VK_IMAGE_TILING_OPTIMAL;
  img_info.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;

  vkCreateImage(m_device, &img_info, nullptr, &m_depth_image);

  VkMemoryRequirements mem_req;
  vkGetImageMemoryRequirements(m_device, m_depth_image, &mem_req);

  VkMemoryAllocateInfo alloc_info{};
  alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  alloc_info.allocationSize = mem_req.size;
  alloc_info.memoryTypeIndex = find_memory_type(mem_req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  vkAllocateMemory(m_device, &alloc_info, nullptr, &m_depth_memory);
  vkBindImageMemory(m_device, m_depth_image, m_depth_memory, 0);

  VkImageViewCreateInfo view_info{};
  view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
  view_info.image = m_depth_image;
  view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
  view_info.format = depth_format;
  view_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
  view_info.subresourceRange.levelCount = 1;
  view_info.subresourceRange.layerCount = 1;
  vkCreateImageView(m_device, &view_info, nullptr, &m_depth_view);
}

void Renderer::create_render_pass() {
  VkAttachmentDescription color_att{};
  color_att.format = m_swapchain_format;
  color_att.samples = VK_SAMPLE_COUNT_1_BIT;
  color_att.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
  color_att.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
  color_att.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  color_att.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

  VkAttachmentDescription depth_att{};
  depth_att.format = VK_FORMAT_D32_SFLOAT;
  depth_att.samples = VK_SAMPLE_COUNT_1_BIT;
  depth_att.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
  depth_att.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
  depth_att.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  depth_att.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

  VkAttachmentReference color_ref{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
  VkAttachmentReference depth_ref{1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};

  VkSubpassDescription subpass{};
  subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
  subpass.colorAttachmentCount = 1;
  subpass.pColorAttachments = &color_ref;
  subpass.pDepthStencilAttachment = &depth_ref;

  VkAttachmentDescription attachments[] = {color_att, depth_att};

  VkRenderPassCreateInfo info{};
  info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
  info.attachmentCount = 2;
  info.pAttachments = attachments;
  info.subpassCount = 1;
  info.pSubpasses = &subpass;

  vkCreateRenderPass(m_device, &info, nullptr, &m_render_pass);

  m_framebuffers.resize(m_swapchain_image_views.size());
  for (size_t i = 0; i < m_swapchain_image_views.size(); i++) {
    VkImageView att[] = {m_swapchain_image_views[i], m_depth_view};
    VkFramebufferCreateInfo fb_info{};
    fb_info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    fb_info.renderPass = m_render_pass;
    fb_info.attachmentCount = 2;
    fb_info.pAttachments = att;
    fb_info.width = m_swapchain_extent.width;
    fb_info.height = m_swapchain_extent.height;
    fb_info.layers = 1;
    vkCreateFramebuffer(m_device, &fb_info, nullptr, &m_framebuffers[i]);
  }
}

VkShaderModule Renderer::load_shader(const char* path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file.is_open())
    throw std::runtime_error(std::string("Failed to open shader: ") + path);

  size_t size = file.tellg();
  std::vector<char> code(size);
  file.seekg(0);
  file.read(code.data(), size);

  VkShaderModuleCreateInfo info{};
  info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  info.codeSize = size;
  info.pCode = reinterpret_cast<const uint32_t*>(code.data());

  VkShaderModule mod;
  vkCreateShaderModule(m_device, &info, nullptr, &mod);
  return mod;
}

void Renderer::create_descriptor_layout() {
  VkDescriptorSetLayoutBinding binding{};
  binding.binding = 0;
  binding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
  binding.descriptorCount = 1;
  binding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

  VkDescriptorSetLayoutCreateInfo info{};
  info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  info.bindingCount = 1;
  info.pBindings = &binding;
  vkCreateDescriptorSetLayout(m_device, &info, nullptr, &m_descriptor_set_layout);
}

void Renderer::create_pipelines() {
  VkShaderModule vert = load_shader("mocap-server/src/render/shaders/skeleton.vert.spv");
  VkShaderModule frag = load_shader("mocap-server/src/render/shaders/skeleton.frag.spv");

  VkPipelineShaderStageCreateInfo stages[2]{};
  stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
  stages[0].module = vert;
  stages[0].pName = "main";
  stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
  stages[1].module = frag;
  stages[1].pName = "main";

  VkVertexInputBindingDescription binding{};
  binding.binding = 0;
  binding.stride = sizeof(vertex);
  binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

  VkVertexInputAttributeDescription attrs[2]{};
  attrs[0].location = 0; attrs[0].binding = 0;
  attrs[0].format = VK_FORMAT_R32G32B32_SFLOAT; attrs[0].offset = 0;
  attrs[1].location = 1; attrs[1].binding = 0;
  attrs[1].format = VK_FORMAT_R32G32B32A32_SFLOAT; attrs[1].offset = 12;

  VkPipelineVertexInputStateCreateInfo vertex_input{};
  vertex_input.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
  vertex_input.vertexBindingDescriptionCount = 1;
  vertex_input.pVertexBindingDescriptions = &binding;
  vertex_input.vertexAttributeDescriptionCount = 2;
  vertex_input.pVertexAttributeDescriptions = attrs;

  VkPipelineViewportStateCreateInfo viewport_state{};
  viewport_state.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
  viewport_state.viewportCount = 1;
  viewport_state.scissorCount = 1;

  VkDynamicState dyn_states[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
  VkPipelineDynamicStateCreateInfo dynamic{};
  dynamic.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
  dynamic.dynamicStateCount = 2;
  dynamic.pDynamicStates = dyn_states;

  VkPipelineRasterizationStateCreateInfo raster{};
  raster.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
  raster.polygonMode = VK_POLYGON_MODE_FILL;
  raster.lineWidth = 2.0f;
  raster.cullMode = VK_CULL_MODE_NONE;
  raster.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;

  VkPipelineMultisampleStateCreateInfo multisample{};
  multisample.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
  multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

  VkPipelineDepthStencilStateCreateInfo depth{};
  depth.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
  depth.depthTestEnable = VK_TRUE;
  depth.depthWriteEnable = VK_TRUE;
  depth.depthCompareOp = VK_COMPARE_OP_LESS;

  VkPipelineColorBlendAttachmentState blend_att{};
  blend_att.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                             VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
  blend_att.blendEnable = VK_TRUE;
  blend_att.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
  blend_att.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
  blend_att.colorBlendOp = VK_BLEND_OP_ADD;
  blend_att.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
  blend_att.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
  blend_att.alphaBlendOp = VK_BLEND_OP_ADD;

  VkPipelineColorBlendStateCreateInfo blend{};
  blend.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
  blend.attachmentCount = 1;
  blend.pAttachments = &blend_att;

  VkPipelineLayoutCreateInfo layout_info{};
  layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  layout_info.setLayoutCount = 1;
  layout_info.pSetLayouts = &m_descriptor_set_layout;
  vkCreatePipelineLayout(m_device, &layout_info, nullptr, &m_pipeline_layout);

  // Line pipeline
  VkPipelineInputAssemblyStateCreateInfo line_assembly{};
  line_assembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
  line_assembly.topology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST;

  VkGraphicsPipelineCreateInfo pipe_info{};
  pipe_info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
  pipe_info.stageCount = 2;
  pipe_info.pStages = stages;
  pipe_info.pVertexInputState = &vertex_input;
  pipe_info.pInputAssemblyState = &line_assembly;
  pipe_info.pViewportState = &viewport_state;
  pipe_info.pRasterizationState = &raster;
  pipe_info.pMultisampleState = &multisample;
  pipe_info.pDepthStencilState = &depth;
  pipe_info.pColorBlendState = &blend;
  pipe_info.pDynamicState = &dynamic;
  pipe_info.layout = m_pipeline_layout;
  pipe_info.renderPass = m_render_pass;

  vkCreateGraphicsPipelines(m_device, VK_NULL_HANDLE, 1, &pipe_info, nullptr, &m_line_pipeline);

  // Point pipeline
  VkPipelineInputAssemblyStateCreateInfo point_assembly{};
  point_assembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
  point_assembly.topology = VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
  pipe_info.pInputAssemblyState = &point_assembly;

  vkCreateGraphicsPipelines(m_device, VK_NULL_HANDLE, 1, &pipe_info, nullptr, &m_point_pipeline);

  vkDestroyShaderModule(m_device, vert, nullptr);
  vkDestroyShaderModule(m_device, frag, nullptr);
}

uint32_t Renderer::find_memory_type(uint32_t type_filter, VkMemoryPropertyFlags properties) {
  VkPhysicalDeviceMemoryProperties mem_props;
  vkGetPhysicalDeviceMemoryProperties(m_physical_device, &mem_props);
  for (uint32_t i = 0; i < mem_props.memoryTypeCount; i++) {
    if ((type_filter & (1 << i)) && (mem_props.memoryTypes[i].propertyFlags & properties) == properties)
      return i;
  }
  throw std::runtime_error("Failed to find suitable memory type");
}

void Renderer::create_buffers(const int* skeleton_edges, int num_edges) {
  // Vertex buffer (host-visible for CPU updates — will switch to CUDA interop later)
  VkDeviceSize vb_size = m_num_keypoints * sizeof(vertex);
  VkBufferCreateInfo vb_info{};
  vb_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  vb_info.size = vb_size;
  vb_info.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;

  vkCreateBuffer(m_device, &vb_info, nullptr, &m_vertex_buffer);

  VkMemoryRequirements vb_req;
  vkGetBufferMemoryRequirements(m_device, m_vertex_buffer, &vb_req);

  VkMemoryAllocateInfo vb_alloc{};
  vb_alloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  vb_alloc.allocationSize = vb_req.size;
  vb_alloc.memoryTypeIndex = find_memory_type(vb_req.memoryTypeBits,
    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  vkAllocateMemory(m_device, &vb_alloc, nullptr, &m_vertex_memory);
  vkBindBufferMemory(m_device, m_vertex_buffer, m_vertex_memory, 0);
  vkMapMemory(m_device, m_vertex_memory, 0, vb_size, 0, &m_vertex_mapped);

  // Index buffer for lines
  VkDeviceSize ib_size = num_edges * 2 * sizeof(uint16_t);
  VkBufferCreateInfo ib_info{};
  ib_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  ib_info.size = ib_size;
  ib_info.usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT;

  vkCreateBuffer(m_device, &ib_info, nullptr, &m_line_index_buffer);

  VkMemoryRequirements ib_req;
  vkGetBufferMemoryRequirements(m_device, m_line_index_buffer, &ib_req);

  VkMemoryAllocateInfo ib_alloc{};
  ib_alloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  ib_alloc.allocationSize = ib_req.size;
  ib_alloc.memoryTypeIndex = find_memory_type(ib_req.memoryTypeBits,
    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  vkAllocateMemory(m_device, &ib_alloc, nullptr, &m_line_index_memory);
  vkBindBufferMemory(m_device, m_line_index_buffer, m_line_index_memory, 0);

  void* ib_mapped;
  vkMapMemory(m_device, m_line_index_memory, 0, ib_size, 0, &ib_mapped);
  uint16_t* indices = static_cast<uint16_t*>(ib_mapped);
  for (int i = 0; i < num_edges; i++) {
    indices[i * 2 + 0] = static_cast<uint16_t>(skeleton_edges[i * 2 + 0]);
    indices[i * 2 + 1] = static_cast<uint16_t>(skeleton_edges[i * 2 + 1]);
  }
  vkUnmapMemory(m_device, m_line_index_memory);

  // Uniform buffer
  VkDeviceSize ub_size = sizeof(renderer_ubo);
  VkBufferCreateInfo ub_info{};
  ub_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  ub_info.size = ub_size;
  ub_info.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;

  vkCreateBuffer(m_device, &ub_info, nullptr, &m_uniform_buffer);

  VkMemoryRequirements ub_req;
  vkGetBufferMemoryRequirements(m_device, m_uniform_buffer, &ub_req);

  VkMemoryAllocateInfo ub_alloc{};
  ub_alloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  ub_alloc.allocationSize = ub_req.size;
  ub_alloc.memoryTypeIndex = find_memory_type(ub_req.memoryTypeBits,
    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  vkAllocateMemory(m_device, &ub_alloc, nullptr, &m_uniform_memory);
  vkBindBufferMemory(m_device, m_uniform_buffer, m_uniform_memory, 0);
  vkMapMemory(m_device, m_uniform_memory, 0, ub_size, 0, &m_uniform_mapped);
}

void Renderer::create_descriptor_set() {
  VkDescriptorPoolSize pool_size{};
  pool_size.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
  pool_size.descriptorCount = 2; // main scene + gizmo

  VkDescriptorPoolCreateInfo pool_info{};
  pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  pool_info.maxSets = 2;
  pool_info.poolSizeCount = 1;
  pool_info.pPoolSizes = &pool_size;
  vkCreateDescriptorPool(m_device, &pool_info, nullptr, &m_descriptor_pool);

  VkDescriptorSetAllocateInfo alloc_info{};
  alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  alloc_info.descriptorPool = m_descriptor_pool;
  alloc_info.descriptorSetCount = 1;
  alloc_info.pSetLayouts = &m_descriptor_set_layout;
  vkAllocateDescriptorSets(m_device, &alloc_info, &m_descriptor_set);

  VkDescriptorBufferInfo buf_info{};
  buf_info.buffer = m_uniform_buffer;
  buf_info.offset = 0;
  buf_info.range = sizeof(renderer_ubo);

  VkWriteDescriptorSet write{};
  write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  write.dstSet = m_descriptor_set;
  write.dstBinding = 0;
  write.descriptorCount = 1;
  write.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
  write.pBufferInfo = &buf_info;
  vkUpdateDescriptorSets(m_device, 1, &write, 0, nullptr);
}

void Renderer::create_sync_objects() {
  VkSemaphoreCreateInfo sem_info{};
  sem_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
  vkCreateSemaphore(m_device, &sem_info, nullptr, &m_image_available);
  vkCreateSemaphore(m_device, &sem_info, nullptr, &m_render_finished);

  VkFenceCreateInfo fence_info{};
  fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;
  vkCreateFence(m_device, &fence_info, nullptr, &m_in_flight);
}

void Renderer::create_command_pool() {
  VkCommandPoolCreateInfo pool_info{};
  pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  pool_info.queueFamilyIndex = m_queue_family;
  vkCreateCommandPool(m_device, &pool_info, nullptr, &m_command_pool);

  VkCommandBufferAllocateInfo alloc_info{};
  alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  alloc_info.commandPool = m_command_pool;
  alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  alloc_info.commandBufferCount = 1;
  vkAllocateCommandBuffers(m_device, &alloc_info, &m_command_buffer);
}

void Renderer::create_gizmo() {
  // Generate ring geometry: 3 unit circles (YZ, XZ, XY planes)
  vertex gizmo_verts[GIZMO_VERTS];
  float colors[3][4] = {
    {1.0f, 0.3f, 0.3f, 1.0f}, // X axis - red
    {0.3f, 1.0f, 0.3f, 1.0f}, // Y axis - green
    {0.3f, 0.3f, 1.0f, 1.0f}, // Z axis - blue
  };

  int vi = 0;
  for (int axis = 0; axis < 3; axis++) {
    for (int seg = 0; seg < GIZMO_SEGMENTS; seg++) {
      float a0 = 2.0f * 3.14159265f * seg / GIZMO_SEGMENTS;
      float a1 = 2.0f * 3.14159265f * (seg + 1) / GIZMO_SEGMENTS;
      float c0 = cosf(a0), s0 = sinf(a0);
      float c1 = cosf(a1), s1 = sinf(a1);

      vertex v0{}, v1{};
      v0.r = v1.r = colors[axis][0];
      v0.g = v1.g = colors[axis][1];
      v0.b = v1.b = colors[axis][2];
      v0.a = v1.a = colors[axis][3];

      if (axis == 0) { // X rotation: circle in YZ plane
        v0.x = 0; v0.y = c0; v0.z = s0;
        v1.x = 0; v1.y = c1; v1.z = s1;
      } else if (axis == 1) { // Y rotation: circle in XZ plane
        v0.x = c0; v0.y = 0; v0.z = s0;
        v1.x = c1; v1.y = 0; v1.z = s1;
      } else { // Z rotation: circle in XY plane
        v0.x = c0; v0.y = s0; v0.z = 0;
        v1.x = c1; v1.y = s1; v1.z = 0;
      }

      gizmo_verts[vi++] = v0;
      gizmo_verts[vi++] = v1;
    }
  }

  // Create vertex buffer
  VkDeviceSize vb_size = sizeof(gizmo_verts);
  VkBufferCreateInfo vb_info{};
  vb_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  vb_info.size = vb_size;
  vb_info.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
  vkCreateBuffer(m_device, &vb_info, nullptr, &m_gizmo_vb);

  VkMemoryRequirements vb_req;
  vkGetBufferMemoryRequirements(m_device, m_gizmo_vb, &vb_req);
  VkMemoryAllocateInfo vb_alloc{};
  vb_alloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  vb_alloc.allocationSize = vb_req.size;
  vb_alloc.memoryTypeIndex = find_memory_type(vb_req.memoryTypeBits,
    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  vkAllocateMemory(m_device, &vb_alloc, nullptr, &m_gizmo_vb_memory);
  vkBindBufferMemory(m_device, m_gizmo_vb, m_gizmo_vb_memory, 0);

  void* mapped;
  vkMapMemory(m_device, m_gizmo_vb_memory, 0, vb_size, 0, &mapped);
  memcpy(mapped, gizmo_verts, vb_size);
  vkUnmapMemory(m_device, m_gizmo_vb_memory);

  // Create uniform buffer for gizmo (rotation-only view + ortho projection)
  VkDeviceSize ub_size = sizeof(renderer_ubo);
  VkBufferCreateInfo ub_info{};
  ub_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  ub_info.size = ub_size;
  ub_info.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
  vkCreateBuffer(m_device, &ub_info, nullptr, &m_gizmo_ub);

  VkMemoryRequirements ub_req;
  vkGetBufferMemoryRequirements(m_device, m_gizmo_ub, &ub_req);
  VkMemoryAllocateInfo ub_alloc{};
  ub_alloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  ub_alloc.allocationSize = ub_req.size;
  ub_alloc.memoryTypeIndex = find_memory_type(ub_req.memoryTypeBits,
    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  vkAllocateMemory(m_device, &ub_alloc, nullptr, &m_gizmo_ub_memory);
  vkBindBufferMemory(m_device, m_gizmo_ub, m_gizmo_ub_memory, 0);
  vkMapMemory(m_device, m_gizmo_ub_memory, 0, ub_size, 0, &m_gizmo_ub_mapped);

  // Allocate descriptor set from existing pool (reuses same layout)
  VkDescriptorSetAllocateInfo ds_alloc{};
  ds_alloc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  ds_alloc.descriptorPool = m_descriptor_pool;
  ds_alloc.descriptorSetCount = 1;
  ds_alloc.pSetLayouts = &m_descriptor_set_layout;
  vkAllocateDescriptorSets(m_device, &ds_alloc, &m_gizmo_desc_set);

  VkDescriptorBufferInfo buf_info{};
  buf_info.buffer = m_gizmo_ub;
  buf_info.offset = 0;
  buf_info.range = sizeof(renderer_ubo);

  VkWriteDescriptorSet write{};
  write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  write.dstSet = m_gizmo_desc_set;
  write.dstBinding = 0;
  write.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
  write.descriptorCount = 1;
  write.pBufferInfo = &buf_info;
  vkUpdateDescriptorSets(m_device, 1, &write, 0, nullptr);
}

void Renderer::update_camera() {
  float rad_az = m_cam_azimuth * 3.14159265f / 180.0f;
  float rad_el = m_cam_elevation * 3.14159265f / 180.0f;

  float eye[3];
  eye[0] = m_cam_target[0] + m_cam_distance * cosf(rad_el) * sinf(rad_az);
  eye[1] = m_cam_target[1] + m_cam_distance * sinf(rad_el);
  eye[2] = m_cam_target[2] + m_cam_distance * cosf(rad_el) * cosf(rad_az);

  // compute up vector with roll
  float rad_roll = m_cam_roll * 3.14159265f / 180.0f;
  // right vector (perpendicular to view direction in horizontal plane)
  float right[3] = {cosf(rad_az), 0, -sinf(rad_az)};
  // base up
  float base_up[3] = {0, 1, 0};
  // rotate base_up around view direction by roll angle
  float up[3] = {
    base_up[0] * cosf(rad_roll) + right[0] * sinf(rad_roll),
    base_up[1] * cosf(rad_roll) + right[1] * sinf(rad_roll),
    base_up[2] * cosf(rad_roll) + right[2] * sinf(rad_roll)
  };

  renderer_ubo ubo;
  mat4_look_at(ubo.view, eye, m_cam_target, up);
  mat4_perspective(ubo.projection, 45.0f * 3.14159265f / 180.0f,
                   (float)m_width / (float)m_height, 1.0f, 100000.0f);

  memcpy(m_uniform_mapped, &ubo, sizeof(ubo));

  // update gizmo UBO: same rotation but no translation, orthographic projection
  float gizmo_eye[3] = {
    3.0f * cosf(rad_el) * sinf(rad_az),
    3.0f * sinf(rad_el),
    3.0f * cosf(rad_el) * cosf(rad_az)
  };
  float gizmo_target[3] = {0, 0, 0};

  renderer_ubo gizmo_ubo;
  mat4_look_at(gizmo_ubo.view, gizmo_eye, gizmo_target, up);
  // orthographic projection [-2, 2]
  memset(gizmo_ubo.projection, 0, sizeof(gizmo_ubo.projection));
  gizmo_ubo.projection[0]  =  0.5f;  // 1/2 for [-2,2] range
  gizmo_ubo.projection[5]  =  0.5f;
  gizmo_ubo.projection[10] = -0.1f;  // depth
  gizmo_ubo.projection[15] =  1.0f;

  memcpy(m_gizmo_ub_mapped, &gizmo_ubo, sizeof(gizmo_ubo));
}

bool Renderer::render_frame(const float* keypoints_3d, int num_keypoints, const frame_metrics* metrics) {
  glfwPollEvents();
  if (glfwWindowShouldClose(m_window))
    return false;

  // Update vertex buffer with keypoints + colors
  vertex* verts = static_cast<vertex*>(m_vertex_mapped);
  for (int i = 0; i < num_keypoints && i < m_num_keypoints; i++) {
    float x = keypoints_3d[i * 3 + 0];
    float y = keypoints_3d[i * 3 + 1];
    float z = keypoints_3d[i * 3 + 2];
    // remap camera coords (Z=up from floor cameras) to renderer coords (Y=up)
    verts[i].x = x;
    verts[i].y = z;
    verts[i].z = y;
    if (std::isnan(x)) {
      verts[i].r = verts[i].g = verts[i].b = 0;
      verts[i].a = 0;
    } else {
      verts[i].r = m_colors[i * 4 + 0];
      verts[i].g = m_colors[i * 4 + 1];
      verts[i].b = m_colors[i * 4 + 2];
      verts[i].a = m_colors[i * 4 + 3];
    }
  }

  // Auto-center camera on visible keypoints
  float cx = 0, cy = 0, cz = 0;
  int count = 0;
  for (int i = 0; i < num_keypoints; i++) {
    if (!std::isnan(keypoints_3d[i * 3])) {
      cx += verts[i].x;
      cy += verts[i].y;
      cz += verts[i].z;
      count++;
    }
  }
  if (count > 0) {
    float alpha = 0.1f;
    m_cam_target[0] = m_cam_target[0] * (1 - alpha) + (cx / count) * alpha;
    m_cam_target[1] = m_cam_target[1] * (1 - alpha) + (cy / count) * alpha;
    m_cam_target[2] = m_cam_target[2] * (1 - alpha) + (cz / count) * alpha;
  }

  update_camera();

  // Render
  vkWaitForFences(m_device, 1, &m_in_flight, VK_TRUE, UINT64_MAX);
  vkResetFences(m_device, 1, &m_in_flight);

  uint32_t img_idx;
  vkAcquireNextImageKHR(m_device, m_swapchain, UINT64_MAX, m_image_available, VK_NULL_HANDLE, &img_idx);

  vkResetCommandBuffer(m_command_buffer, 0);

  VkCommandBufferBeginInfo begin{};
  begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  vkBeginCommandBuffer(m_command_buffer, &begin);

  VkClearValue clears[2]{};
  clears[0].color = {{0.05f, 0.05f, 0.08f, 1.0f}};
  clears[1].depthStencil = {1.0f, 0};

  VkRenderPassBeginInfo rp_begin{};
  rp_begin.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
  rp_begin.renderPass = m_render_pass;
  rp_begin.framebuffer = m_framebuffers[img_idx];
  rp_begin.renderArea.extent = m_swapchain_extent;
  rp_begin.clearValueCount = 2;
  rp_begin.pClearValues = clears;
  vkCmdBeginRenderPass(m_command_buffer, &rp_begin, VK_SUBPASS_CONTENTS_INLINE);

  VkViewport viewport{};
  viewport.width = (float)m_width;
  viewport.height = (float)m_height;
  viewport.maxDepth = 1.0f;
  vkCmdSetViewport(m_command_buffer, 0, 1, &viewport);

  VkRect2D scissor{};
  scissor.extent = m_swapchain_extent;
  vkCmdSetScissor(m_command_buffer, 0, 1, &scissor);

  VkDeviceSize offset = 0;
  vkCmdBindVertexBuffers(m_command_buffer, 0, 1, &m_vertex_buffer, &offset);
  vkCmdBindDescriptorSets(m_command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                          m_pipeline_layout, 0, 1, &m_descriptor_set, 0, nullptr);

  // Draw skeleton lines
  vkCmdBindPipeline(m_command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_line_pipeline);
  vkCmdBindIndexBuffer(m_command_buffer, m_line_index_buffer, 0, VK_INDEX_TYPE_UINT16);
  vkCmdDrawIndexed(m_command_buffer, m_num_edges * 2, 1, 0, 0, 0);

  // Draw joint points
  vkCmdBindPipeline(m_command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_point_pipeline);
  vkCmdDraw(m_command_buffer, m_num_keypoints, 1, 0, 0);

  // Draw rotation gizmo in bottom-right corner
  {
    VkViewport gizmo_vp{};
    gizmo_vp.x = (float)(m_swapchain_extent.width - GIZMO_SIZE);
    gizmo_vp.y = (float)(m_swapchain_extent.height - GIZMO_SIZE);
    gizmo_vp.width = (float)GIZMO_SIZE;
    gizmo_vp.height = (float)GIZMO_SIZE;
    gizmo_vp.maxDepth = 1.0f;
    vkCmdSetViewport(m_command_buffer, 0, 1, &gizmo_vp);

    VkRect2D gizmo_scissor{};
    gizmo_scissor.offset = {(int32_t)(m_swapchain_extent.width - GIZMO_SIZE),
                            (int32_t)(m_swapchain_extent.height - GIZMO_SIZE)};
    gizmo_scissor.extent = {(uint32_t)GIZMO_SIZE, (uint32_t)GIZMO_SIZE};
    vkCmdSetScissor(m_command_buffer, 0, 1, &gizmo_scissor);

    VkDeviceSize gizmo_offset = 0;
    vkCmdBindVertexBuffers(m_command_buffer, 0, 1, &m_gizmo_vb, &gizmo_offset);
    vkCmdBindDescriptorSets(m_command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            m_pipeline_layout, 0, 1, &m_gizmo_desc_set, 0, nullptr);
    vkCmdBindPipeline(m_command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_line_pipeline);
    vkCmdDraw(m_command_buffer, GIZMO_VERTS, 1, 0, 0);

    // restore full viewport and scissor
    VkViewport full_vp{};
    full_vp.width = (float)m_width;
    full_vp.height = (float)m_height;
    full_vp.maxDepth = 1.0f;
    vkCmdSetViewport(m_command_buffer, 0, 1, &full_vp);

    VkRect2D full_scissor{};
    full_scissor.extent = m_swapchain_extent;
    vkCmdSetScissor(m_command_buffer, 0, 1, &full_scissor);
  }

  // Draw text overlay with metrics
  if (metrics) {
    int valid_count = 0;
    float total_reproj = 0.0f;
    int reproj_count = 0;
    float total_views = 0.0f;
    for (int k = 0; k < num_keypoints; k++) {
      if (!std::isnan(keypoints_3d[k * 3]))
        valid_count++;
      if (!std::isnan(metrics->reproj_error[k])) {
        total_reproj += metrics->reproj_error[k];
        reproj_count++;
      }
      total_views += metrics->num_views[k];
    }
    float avg_reproj = reproj_count > 0 ? total_reproj / reproj_count : 0.0f;
    float avg_views = num_keypoints > 0 ? total_views / num_keypoints : 0.0f;

    float running_reproj = m_pipe_config ? m_pipe_config->avg_reproj_running.load(std::memory_order_relaxed) : 0.0f;
    int running_frames = m_pipe_config ? m_pipe_config->avg_reproj_frames.load(std::memory_order_relaxed) : 0;
    m_text_overlay->update(metrics, valid_count, num_keypoints, avg_reproj, avg_views,
      running_reproj, running_frames);
    m_text_overlay->record(m_command_buffer);
  }

  vkCmdEndRenderPass(m_command_buffer);
  vkEndCommandBuffer(m_command_buffer);

  VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  VkSubmitInfo submit{};
  submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submit.waitSemaphoreCount = 1;
  submit.pWaitSemaphores = &m_image_available;
  submit.pWaitDstStageMask = &wait_stage;
  submit.commandBufferCount = 1;
  submit.pCommandBuffers = &m_command_buffer;
  submit.signalSemaphoreCount = 1;
  submit.pSignalSemaphores = &m_render_finished;

  vkQueueSubmit(m_graphics_queue, 1, &submit, m_in_flight);

  VkPresentInfoKHR present{};
  present.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
  present.waitSemaphoreCount = 1;
  present.pWaitSemaphores = &m_render_finished;
  present.swapchainCount = 1;
  present.pSwapchains = &m_swapchain;
  present.pImageIndices = &img_idx;

  vkQueuePresentKHR(m_graphics_queue, &present);

  return true;
}

// GLFW callbacks
void Renderer::mouse_button_callback(GLFWwindow* window, int button, int action, int /*mods*/) {
  auto* self = static_cast<Renderer*>(glfwGetWindowUserPointer(window));
  if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS) {
    double mx, my;
    glfwGetCursorPos(window, &mx, &my);

    // scale to render coords for gizmo hit test
    int win_w, win_h;
    glfwGetWindowSize(window, &win_w, &win_h);
    double rx = mx * (double)self->m_width / (double)win_w;
    double ry = my * (double)self->m_height / (double)win_h;

    // check if click is in the gizmo region (bottom-right corner)
    float gx = (float)self->m_width - GIZMO_SIZE;
    float gy = (float)self->m_height - GIZMO_SIZE;
    self->m_selected_axis = -1;

    if (rx >= gx && ry >= gy) {
      // normalize click within gizmo to [-1, 1]
      float gnx = ((float)rx - gx) / GIZMO_SIZE * 2.0f - 1.0f;
      float gny = ((float)ry - gy) / GIZMO_SIZE * 2.0f - 1.0f;

      // project each ring and find closest to click
      // the gizmo view is from the same camera angle looking at origin
      // simplify: use screen-space distance to each axis line through origin
      // X axis (red/pitch): ring in YZ plane — on screen, appears as ellipse
      // Y axis (green/yaw): ring in XZ plane
      // Z axis (blue/roll): ring in XY plane
      // Approximate: distance from click to the projected circle
      // For each axis, compute distance from (gnx, gny) to the unit circle
      // projected through the gizmo view matrix

      float rad_az = self->m_cam_azimuth * 3.14159265f / 180.0f;
      float rad_el = self->m_cam_elevation * 3.14159265f / 180.0f;
      float rad_roll = self->m_cam_roll * 3.14159265f / 180.0f;

      float best_dist = 0.3f; // threshold in normalized gizmo coords
      int best_axis = -1;

      // sample points on each ring and find min distance to click
      for (int axis = 0; axis < 3; axis++) {
        float min_d = 1e9f;
        for (int s = 0; s < GIZMO_SEGMENTS; s++) {
          float a = 2.0f * 3.14159265f * s / GIZMO_SEGMENTS;
          float ca = cosf(a), sa = sinf(a);
          float wx, wy, wz;
          if (axis == 0) { wx = 0; wy = ca; wz = sa; }
          else if (axis == 1) { wx = ca; wy = 0; wz = sa; }
          else { wx = ca; wy = sa; wz = 0; }

          // apply camera rotation (simplified orbit projection)
          // rotate by -azimuth around Y, then -elevation around X
          float x1 = wx * cosf(rad_az) + wz * sinf(rad_az);
          float z1 = -wx * sinf(rad_az) + wz * cosf(rad_az);
          float y1 = wy;
          float y2 = y1 * cosf(rad_el) - z1 * sinf(rad_el);
          float x2 = x1;

          // apply roll
          float xr = x2 * cosf(rad_roll) - y2 * sinf(rad_roll);
          float yr = x2 * sinf(rad_roll) + y2 * cosf(rad_roll);

          // match the 0.5 scale in the gizmo ortho projection
          float sx = xr * 0.5f;
          float sy = yr * 0.5f;

          float d = sqrtf((gnx - sx) * (gnx - sx) + (gny - sy) * (gny - sy));
          if (d < min_d) min_d = d;
        }
        if (min_d < best_dist) {
          best_dist = min_d;
          best_axis = axis;
        }
      }
      self->m_selected_axis = best_axis;
    }

    self->m_mouse_dragging = true;
    self->m_last_mouse_x = mx;
    self->m_last_mouse_y = my;
  } else if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_RELEASE) {
    self->m_mouse_dragging = false;
    self->m_selected_axis = -1;
  }
}

void Renderer::cursor_pos_callback(GLFWwindow* window, double xpos, double ypos) {
  auto* self = static_cast<Renderer*>(glfwGetWindowUserPointer(window));
  if (self->m_mouse_dragging) {
    double dx = xpos - self->m_last_mouse_x;
    double dy = ypos - self->m_last_mouse_y;
    float sensitivity = 0.3f;

    // scale azimuth sensitivity by cos(elevation) to prevent pole spin
    float el_rad = self->m_cam_elevation * 3.14159265f / 180.0f;
    float az_scale = fmaxf(cosf(el_rad), 0.1f);

    switch (self->m_selected_axis) {
      case 0: // X axis (pitch/elevation) — vertical drag only
        self->m_cam_elevation += (float)dy * sensitivity;
        break;
      case 1: // Y axis (yaw/azimuth) — horizontal drag only
        self->m_cam_azimuth += (float)dx * sensitivity / az_scale;
        break;
      case 2: // Z axis (roll) — horizontal drag
        self->m_cam_roll += (float)dx * sensitivity;
        break;
      default: // free rotation
        self->m_cam_azimuth += (float)dx * sensitivity / az_scale;
        self->m_cam_elevation += (float)dy * sensitivity;
        break;
    }
    self->m_cam_elevation = fmaxf(-89.0f, fminf(89.0f, self->m_cam_elevation));
    self->m_last_mouse_x = xpos;
    self->m_last_mouse_y = ypos;
  }
}

void Renderer::scroll_callback(GLFWwindow* window, double /*xoffset*/, double yoffset) {
  auto* self = static_cast<Renderer*>(glfwGetWindowUserPointer(window));
  self->m_cam_distance *= (1.0f - (float)yoffset * 0.1f);
  self->m_cam_distance = fmaxf(100.0f, fminf(50000.0f, self->m_cam_distance));
}
