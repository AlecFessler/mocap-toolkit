#ifndef TEXT_OVERLAY_HPP
#define TEXT_OVERLAY_HPP

#include <vulkan/vulkan.h>
#include "triangulation/triangulator.hpp"

struct text_vertex {
  float px, py;  // screen pixel position
  float u, v;    // atlas UV
};

// Max characters we can render per frame
constexpr int MAX_TEXT_CHARS = 512;
constexpr int MAX_TEXT_VERTICES = MAX_TEXT_CHARS * 6; // 2 tris per char

class TextOverlay {
public:
  TextOverlay(
    VkDevice device,
    VkPhysicalDevice physical_device,
    VkRenderPass render_pass,
    int screen_width,
    int screen_height
  );
  ~TextOverlay();

  // Format metrics into text and generate quad vertices
  void update(
    const frame_metrics* metrics,
    int valid_count,
    int total_keypoints,
    float avg_reproj,
    float avg_views
  );

  // Record draw commands into command buffer
  void record(VkCommandBuffer cmd);

private:
  VkDevice m_device;
  VkPhysicalDevice m_physical_device;
  int m_screen_width;
  int m_screen_height;

  // Font atlas
  VkImage m_atlas_image;
  VkDeviceMemory m_atlas_memory;
  VkImageView m_atlas_view;
  VkSampler m_sampler;

  // Atlas metadata
  int m_atlas_w, m_atlas_h;
  float m_char_width;   // pixels per char at rendered size
  float m_char_height;

  // stb baked char data (ASCII 32-126)
  void* m_baked_chars; // stbtt_bakedchar[96]

  // Text quad vertex buffer
  VkBuffer m_vertex_buffer;
  VkDeviceMemory m_vertex_memory;
  void* m_vertex_mapped;
  int m_vertex_count;

  // Pipeline
  VkPipelineLayout m_pipeline_layout;
  VkPipeline m_pipeline;
  VkDescriptorSetLayout m_desc_layout;
  VkDescriptorPool m_desc_pool;
  VkDescriptorSet m_desc_set;

  void create_font_atlas();
  void create_pipeline(VkRenderPass render_pass);
  void create_vertex_buffer();
  void create_descriptor(VkRenderPass render_pass);

  uint32_t find_memory_type(uint32_t type_filter, VkMemoryPropertyFlags props);

  // Add a string at pixel position, returns updated x
  float add_text(const char* str, float x, float y, text_vertex* verts, int* count);
};

#endif // TEXT_OVERLAY_HPP
