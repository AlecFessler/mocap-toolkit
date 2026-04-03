#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <vector>

#define STB_TRUETYPE_IMPLEMENTATION
#include "render/stb_truetype.h"
#include "render/text_overlay.hpp"

static const char* FONT_PATH = "/usr/share/fonts/noto/NotoSansMono-Regular.ttf";
static constexpr float FONT_SIZE = 18.0f;
static constexpr int ATLAS_W = 512;
static constexpr int ATLAS_H = 512;
static constexpr int FIRST_CHAR = 32;
static constexpr int NUM_CHARS = 95; // ASCII 32-126

TextOverlay::TextOverlay(
  VkDevice device,
  VkPhysicalDevice physical_device,
  VkRenderPass render_pass,
  int screen_width,
  int screen_height
) : m_device(device),
    m_physical_device(physical_device),
    m_screen_width(screen_width),
    m_screen_height(screen_height),
    m_vertex_count(0)
{
  m_baked_chars = new stbtt_bakedchar[NUM_CHARS];
  create_font_atlas();
  create_vertex_buffer();
  create_descriptor(render_pass);
  create_pipeline(render_pass);
}

TextOverlay::~TextOverlay() {
  vkDestroyPipeline(m_device, m_pipeline, nullptr);
  vkDestroyPipelineLayout(m_device, m_pipeline_layout, nullptr);
  vkDestroyDescriptorPool(m_device, m_desc_pool, nullptr);
  vkDestroyDescriptorSetLayout(m_device, m_desc_layout, nullptr);

  vkDestroyBuffer(m_device, m_vertex_buffer, nullptr);
  vkFreeMemory(m_device, m_vertex_memory, nullptr);

  vkDestroySampler(m_device, m_sampler, nullptr);
  vkDestroyImageView(m_device, m_atlas_view, nullptr);
  vkDestroyImage(m_device, m_atlas_image, nullptr);
  vkFreeMemory(m_device, m_atlas_memory, nullptr);

  delete[] static_cast<stbtt_bakedchar*>(m_baked_chars);
}

uint32_t TextOverlay::find_memory_type(uint32_t type_filter, VkMemoryPropertyFlags props) {
  VkPhysicalDeviceMemoryProperties mem_props;
  vkGetPhysicalDeviceMemoryProperties(m_physical_device, &mem_props);
  for (uint32_t i = 0; i < mem_props.memoryTypeCount; i++) {
    if ((type_filter & (1 << i)) && (mem_props.memoryTypes[i].propertyFlags & props) == props)
      return i;
  }
  throw std::runtime_error("Failed to find suitable memory type for text overlay");
}

void TextOverlay::create_font_atlas() {
  // Load font file
  std::ifstream file(FONT_PATH, std::ios::binary | std::ios::ate);
  if (!file.is_open())
    throw std::runtime_error(std::string("Failed to open font: ") + FONT_PATH);

  size_t file_size = file.tellg();
  std::vector<unsigned char> font_data(file_size);
  file.seekg(0);
  file.read(reinterpret_cast<char*>(font_data.data()), file_size);

  // Bake font atlas bitmap
  std::vector<unsigned char> atlas_bitmap(ATLAS_W * ATLAS_H);
  stbtt_BakeFontBitmap(
    font_data.data(), 0, FONT_SIZE,
    atlas_bitmap.data(), ATLAS_W, ATLAS_H,
    FIRST_CHAR, NUM_CHARS,
    static_cast<stbtt_bakedchar*>(m_baked_chars)
  );

  m_atlas_w = ATLAS_W;
  m_atlas_h = ATLAS_H;

  // Estimate char dimensions from 'M'
  stbtt_bakedchar* bc = static_cast<stbtt_bakedchar*>(m_baked_chars);
  m_char_width = bc['M' - FIRST_CHAR].xadvance;
  m_char_height = FONT_SIZE;

  // Create Vulkan image for atlas (R8_UNORM)
  VkImageCreateInfo img_info{};
  img_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  img_info.imageType = VK_IMAGE_TYPE_2D;
  img_info.format = VK_FORMAT_R8_UNORM;
  img_info.extent = {(uint32_t)ATLAS_W, (uint32_t)ATLAS_H, 1};
  img_info.mipLevels = 1;
  img_info.arrayLayers = 1;
  img_info.samples = VK_SAMPLE_COUNT_1_BIT;
  img_info.tiling = VK_IMAGE_TILING_LINEAR;
  img_info.usage = VK_IMAGE_USAGE_SAMPLED_BIT;
  img_info.initialLayout = VK_IMAGE_LAYOUT_PREINITIALIZED;
  vkCreateImage(m_device, &img_info, nullptr, &m_atlas_image);

  VkMemoryRequirements mem_req;
  vkGetImageMemoryRequirements(m_device, m_atlas_image, &mem_req);

  VkMemoryAllocateInfo alloc_info{};
  alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  alloc_info.allocationSize = mem_req.size;
  alloc_info.memoryTypeIndex = find_memory_type(
    mem_req.memoryTypeBits,
    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
  );
  vkAllocateMemory(m_device, &alloc_info, nullptr, &m_atlas_memory);
  vkBindImageMemory(m_device, m_atlas_image, m_atlas_memory, 0);

  // Copy bitmap data to image
  VkImageSubresource subres{};
  subres.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  VkSubresourceLayout layout;
  vkGetImageSubresourceLayout(m_device, m_atlas_image, &subres, &layout);

  void* mapped;
  vkMapMemory(m_device, m_atlas_memory, 0, mem_req.size, 0, &mapped);
  if (layout.rowPitch == (VkDeviceSize)ATLAS_W) {
    memcpy(mapped, atlas_bitmap.data(), ATLAS_W * ATLAS_H);
  } else {
    for (int y = 0; y < ATLAS_H; y++) {
      memcpy(
        static_cast<char*>(mapped) + y * layout.rowPitch,
        atlas_bitmap.data() + y * ATLAS_W,
        ATLAS_W
      );
    }
  }
  vkUnmapMemory(m_device, m_atlas_memory);

  // Image view
  VkImageViewCreateInfo view_info{};
  view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
  view_info.image = m_atlas_image;
  view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
  view_info.format = VK_FORMAT_R8_UNORM;
  view_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  view_info.subresourceRange.levelCount = 1;
  view_info.subresourceRange.layerCount = 1;
  vkCreateImageView(m_device, &view_info, nullptr, &m_atlas_view);

  // Sampler
  VkSamplerCreateInfo samp_info{};
  samp_info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
  samp_info.magFilter = VK_FILTER_LINEAR;
  samp_info.minFilter = VK_FILTER_LINEAR;
  samp_info.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  samp_info.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  samp_info.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  vkCreateSampler(m_device, &samp_info, nullptr, &m_sampler);
}

void TextOverlay::create_vertex_buffer() {
  VkDeviceSize size = MAX_TEXT_VERTICES * sizeof(text_vertex);

  VkBufferCreateInfo buf_info{};
  buf_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  buf_info.size = size;
  buf_info.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;

  vkCreateBuffer(m_device, &buf_info, nullptr, &m_vertex_buffer);

  VkMemoryRequirements mem_req;
  vkGetBufferMemoryRequirements(m_device, m_vertex_buffer, &mem_req);

  VkMemoryAllocateInfo alloc_info{};
  alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  alloc_info.allocationSize = mem_req.size;
  alloc_info.memoryTypeIndex = find_memory_type(
    mem_req.memoryTypeBits,
    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
  );
  vkAllocateMemory(m_device, &alloc_info, nullptr, &m_vertex_memory);
  vkBindBufferMemory(m_device, m_vertex_buffer, m_vertex_memory, 0);
  vkMapMemory(m_device, m_vertex_memory, 0, size, 0, &m_vertex_mapped);
}

void TextOverlay::create_descriptor(VkRenderPass) {
  // Descriptor set layout: one combined image sampler
  VkDescriptorSetLayoutBinding binding{};
  binding.binding = 0;
  binding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  binding.descriptorCount = 1;
  binding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

  VkDescriptorSetLayoutCreateInfo layout_info{};
  layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  layout_info.bindingCount = 1;
  layout_info.pBindings = &binding;
  vkCreateDescriptorSetLayout(m_device, &layout_info, nullptr, &m_desc_layout);

  // Descriptor pool
  VkDescriptorPoolSize pool_size{};
  pool_size.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  pool_size.descriptorCount = 1;

  VkDescriptorPoolCreateInfo pool_info{};
  pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  pool_info.maxSets = 1;
  pool_info.poolSizeCount = 1;
  pool_info.pPoolSizes = &pool_size;
  vkCreateDescriptorPool(m_device, &pool_info, nullptr, &m_desc_pool);

  // Allocate descriptor set
  VkDescriptorSetAllocateInfo alloc_info{};
  alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  alloc_info.descriptorPool = m_desc_pool;
  alloc_info.descriptorSetCount = 1;
  alloc_info.pSetLayouts = &m_desc_layout;
  vkAllocateDescriptorSets(m_device, &alloc_info, &m_desc_set);

  // Write descriptor
  VkDescriptorImageInfo img_info{};
  img_info.sampler = m_sampler;
  img_info.imageView = m_atlas_view;
  img_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

  VkWriteDescriptorSet write{};
  write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  write.dstSet = m_desc_set;
  write.dstBinding = 0;
  write.descriptorCount = 1;
  write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  write.pImageInfo = &img_info;
  vkUpdateDescriptorSets(m_device, 1, &write, 0, nullptr);
}

void TextOverlay::create_pipeline(VkRenderPass render_pass) {
  // Load shaders
  auto load_shader = [&](const char* path) -> VkShaderModule {
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
  };

  VkShaderModule vert = load_shader("mocap-server/src/render/shaders/text.vert.spv");
  VkShaderModule frag = load_shader("mocap-server/src/render/shaders/text.frag.spv");

  VkPipelineShaderStageCreateInfo stages[2]{};
  stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
  stages[0].module = vert;
  stages[0].pName = "main";
  stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
  stages[1].module = frag;
  stages[1].pName = "main";

  // Vertex input: vec2 pos, vec2 uv
  VkVertexInputBindingDescription binding{};
  binding.binding = 0;
  binding.stride = sizeof(text_vertex);
  binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

  VkVertexInputAttributeDescription attrs[2]{};
  attrs[0].location = 0;
  attrs[0].binding = 0;
  attrs[0].format = VK_FORMAT_R32G32_SFLOAT;
  attrs[0].offset = offsetof(text_vertex, px);
  attrs[1].location = 1;
  attrs[1].binding = 0;
  attrs[1].format = VK_FORMAT_R32G32_SFLOAT;
  attrs[1].offset = offsetof(text_vertex, u);

  VkPipelineVertexInputStateCreateInfo vert_input{};
  vert_input.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
  vert_input.vertexBindingDescriptionCount = 1;
  vert_input.pVertexBindingDescriptions = &binding;
  vert_input.vertexAttributeDescriptionCount = 2;
  vert_input.pVertexAttributeDescriptions = attrs;

  VkPipelineInputAssemblyStateCreateInfo assembly{};
  assembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
  assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

  VkPipelineViewportStateCreateInfo viewport_state{};
  viewport_state.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
  viewport_state.viewportCount = 1;
  viewport_state.scissorCount = 1;

  VkPipelineRasterizationStateCreateInfo raster{};
  raster.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
  raster.polygonMode = VK_POLYGON_MODE_FILL;
  raster.cullMode = VK_CULL_MODE_NONE;
  raster.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
  raster.lineWidth = 1.0f;

  VkPipelineMultisampleStateCreateInfo ms{};
  ms.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
  ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

  VkPipelineDepthStencilStateCreateInfo depth{};
  depth.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
  depth.depthTestEnable = VK_FALSE;
  depth.depthWriteEnable = VK_FALSE;

  VkPipelineColorBlendAttachmentState blend_attach{};
  blend_attach.blendEnable = VK_TRUE;
  blend_attach.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
  blend_attach.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
  blend_attach.colorBlendOp = VK_BLEND_OP_ADD;
  blend_attach.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
  blend_attach.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
  blend_attach.alphaBlendOp = VK_BLEND_OP_ADD;
  blend_attach.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                 VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

  VkPipelineColorBlendStateCreateInfo blend{};
  blend.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
  blend.attachmentCount = 1;
  blend.pAttachments = &blend_attach;

  VkDynamicState dyn_states[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
  VkPipelineDynamicStateCreateInfo dynamic{};
  dynamic.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
  dynamic.dynamicStateCount = 2;
  dynamic.pDynamicStates = dyn_states;

  // Push constant for screen size
  VkPushConstantRange push_range{};
  push_range.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
  push_range.offset = 0;
  push_range.size = sizeof(float) * 2;

  VkPipelineLayoutCreateInfo layout_info{};
  layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  layout_info.setLayoutCount = 1;
  layout_info.pSetLayouts = &m_desc_layout;
  layout_info.pushConstantRangeCount = 1;
  layout_info.pPushConstantRanges = &push_range;
  vkCreatePipelineLayout(m_device, &layout_info, nullptr, &m_pipeline_layout);

  VkGraphicsPipelineCreateInfo pipe_info{};
  pipe_info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
  pipe_info.stageCount = 2;
  pipe_info.pStages = stages;
  pipe_info.pVertexInputState = &vert_input;
  pipe_info.pInputAssemblyState = &assembly;
  pipe_info.pViewportState = &viewport_state;
  pipe_info.pRasterizationState = &raster;
  pipe_info.pMultisampleState = &ms;
  pipe_info.pDepthStencilState = &depth;
  pipe_info.pColorBlendState = &blend;
  pipe_info.pDynamicState = &dynamic;
  pipe_info.layout = m_pipeline_layout;
  pipe_info.renderPass = render_pass;
  pipe_info.subpass = 0;

  vkCreateGraphicsPipelines(m_device, VK_NULL_HANDLE, 1, &pipe_info, nullptr, &m_pipeline);

  vkDestroyShaderModule(m_device, vert, nullptr);
  vkDestroyShaderModule(m_device, frag, nullptr);
}

float TextOverlay::add_text(const char* str, float x, float y, text_vertex* verts, int* count) {
  stbtt_bakedchar* bc = static_cast<stbtt_bakedchar*>(m_baked_chars);

  for (const char* p = str; *p; p++) {
    int ch = *p;
    if (ch < FIRST_CHAR || ch >= FIRST_CHAR + NUM_CHARS)
      continue;
    if (*count + 6 > MAX_TEXT_VERTICES)
      break;

    stbtt_bakedchar* b = &bc[ch - FIRST_CHAR];

    float x0 = x + b->xoff;
    float y0 = y + b->yoff + m_char_height; // offset baseline
    float x1 = x0 + (b->x1 - b->x0);
    float y1 = y0 + (b->y1 - b->y0);

    float u0 = (float)b->x0 / m_atlas_w;
    float v0 = (float)b->y0 / m_atlas_h;
    float u1 = (float)b->x1 / m_atlas_w;
    float v1 = (float)b->y1 / m_atlas_h;

    // Two triangles for the quad
    verts[*count] = {x0, y0, u0, v0}; (*count)++;
    verts[*count] = {x1, y0, u1, v0}; (*count)++;
    verts[*count] = {x1, y1, u1, v1}; (*count)++;
    verts[*count] = {x0, y0, u0, v0}; (*count)++;
    verts[*count] = {x1, y1, u1, v1}; (*count)++;
    verts[*count] = {x0, y1, u0, v1}; (*count)++;

    x += b->xadvance;
  }

  return x;
}

static const char* BONE_NAMES[NUM_BONES] = {
  "L.UArm", "L.LArm", "R.UArm", "R.LArm",
  "L.Torso", "R.Torso", "L.ULeg", "L.LLeg",
  "R.ULeg", "R.LLeg", "Shoulders", "Hips"
};

void TextOverlay::update(
  const frame_metrics* metrics,
  int valid_count,
  int total_keypoints,
  float avg_reproj,
  float avg_views
) {
  text_vertex* verts = static_cast<text_vertex*>(m_vertex_mapped);
  int count = 0;
  char line[128];
  float x_start = 10.0f;
  float y = 10.0f;
  float line_spacing = m_char_height + 4.0f;

  // Line 1: summary
  snprintf(line, sizeof(line), "Valid: %d/%d | Reproj: %.1fpx | Views: %.1f",
    valid_count, total_keypoints, avg_reproj, avg_views);
  add_text(line, x_start, y, verts, &count);
  y += line_spacing;

  // Bone lengths
  if (metrics) {
    add_text("Bones (mm):", x_start, y, verts, &count);
    y += line_spacing;

    // Two bones per line for compactness
    for (int i = 0; i < NUM_BONES; i += 2) {
      if (i + 1 < NUM_BONES) {
        float b0 = std::isnan(metrics->bone_lengths[i]) ? 0.0f : metrics->bone_lengths[i];
        float b1 = std::isnan(metrics->bone_lengths[i+1]) ? 0.0f : metrics->bone_lengths[i+1];
        snprintf(line, sizeof(line), "  %s: %.0f  %s: %.0f",
          BONE_NAMES[i], b0, BONE_NAMES[i+1], b1);
      } else {
        float b0 = std::isnan(metrics->bone_lengths[i]) ? 0.0f : metrics->bone_lengths[i];
        snprintf(line, sizeof(line), "  %s: %.0f", BONE_NAMES[i], b0);
      }
      add_text(line, x_start, y, verts, &count);
      y += line_spacing;
    }
  }

  m_vertex_count = count;
}

void TextOverlay::record(VkCommandBuffer cmd) {
  if (m_vertex_count == 0)
    return;

  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipeline);
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
    m_pipeline_layout, 0, 1, &m_desc_set, 0, nullptr);

  float screen_size[2] = {(float)m_screen_width, (float)m_screen_height};
  vkCmdPushConstants(cmd, m_pipeline_layout, VK_SHADER_STAGE_VERTEX_BIT,
    0, sizeof(screen_size), screen_size);

  VkDeviceSize offset = 0;
  vkCmdBindVertexBuffers(cmd, 0, 1, &m_vertex_buffer, &offset);
  vkCmdDraw(cmd, m_vertex_count, 1, 0, 0);
}
