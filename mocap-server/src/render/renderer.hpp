#ifndef RENDERER_HPP
#define RENDERER_HPP

#include <cuda_runtime.h>
#include <vulkan/vulkan.h>
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <vector>
#include <cstdint>

struct frame_metrics;
struct pipeline_config;
class TextOverlay;

struct renderer_config {
  int width = 1280;
  int height = 720;
  int num_keypoints;
  const int* skeleton_edges; // pairs: [a0, b0, a1, b1, ...]
  int num_edges;
};

struct renderer_ubo {
  float view[16];       // 4x4 column-major
  float projection[16]; // 4x4 column-major
};

class Renderer {
public:
  Renderer(const renderer_config& config, pipeline_config* pipe_config);
  ~Renderer();

  // Update keypoints from host memory and render
  // keypoints_3d: [num_keypoints * 3] floats (x,y,z), NaN for invalid
  // Returns false if window closed
  bool render_frame(const float* keypoints_3d, int num_keypoints, const frame_metrics* metrics = nullptr);

private:
  int m_width, m_height;
  int m_num_keypoints;
  int m_num_edges;

  // GLFW
  GLFWwindow* m_window;

  // Vulkan core
  VkInstance m_instance;
  VkPhysicalDevice m_physical_device;
  VkDevice m_device;
  VkQueue m_graphics_queue;
  uint32_t m_queue_family;
  VkSurfaceKHR m_surface;

  // Swapchain
  VkSwapchainKHR m_swapchain;
  VkFormat m_swapchain_format;
  VkExtent2D m_swapchain_extent;
  std::vector<VkImage> m_swapchain_images;
  std::vector<VkImageView> m_swapchain_image_views;

  // Depth buffer
  VkImage m_depth_image;
  VkDeviceMemory m_depth_memory;
  VkImageView m_depth_view;

  // Render pass + framebuffers
  VkRenderPass m_render_pass;
  std::vector<VkFramebuffer> m_framebuffers;

  // Pipelines
  VkPipelineLayout m_pipeline_layout;
  VkPipeline m_line_pipeline;   // for skeleton edges
  VkPipeline m_point_pipeline;  // for joint points
  VkDescriptorSetLayout m_descriptor_set_layout;
  VkDescriptorPool m_descriptor_pool;
  VkDescriptorSet m_descriptor_set;

  // Buffers
  VkBuffer m_vertex_buffer;
  VkDeviceMemory m_vertex_memory;
  void* m_vertex_mapped;

  VkBuffer m_line_index_buffer;
  VkDeviceMemory m_line_index_memory;

  VkBuffer m_uniform_buffer;
  VkDeviceMemory m_uniform_memory;
  void* m_uniform_mapped;

  // Sync
  VkSemaphore m_image_available;
  VkSemaphore m_render_finished;
  VkFence m_in_flight;

  // Command
  VkCommandPool m_command_pool;
  VkCommandBuffer m_command_buffer;

  // Camera
  float m_cam_azimuth;
  float m_cam_elevation;
  float m_cam_distance;
  float m_cam_roll;
  float m_cam_target[3];
  double m_last_mouse_x, m_last_mouse_y;
  bool m_mouse_dragging;
  int m_selected_axis; // -1=free, 0=X(pitch), 1=Y(yaw), 2=Z(roll)

  // Body part colors [keypoint_index] -> rgba
  float m_colors[512 * 4]; // max keypoints * rgba

  // Text overlay
  TextOverlay* m_text_overlay;

  // Pipeline toggles (shared with inference thread)
  pipeline_config* m_pipe_config;

  // Rotation gizmo
  static constexpr int GIZMO_SEGMENTS = 48;
  static constexpr int GIZMO_VERTS = 3 * GIZMO_SEGMENTS * 2;
  static constexpr int GIZMO_SIZE = 150;
  VkBuffer m_gizmo_vb;
  VkDeviceMemory m_gizmo_vb_memory;
  VkBuffer m_gizmo_ub;
  VkDeviceMemory m_gizmo_ub_memory;
  void* m_gizmo_ub_mapped;
  VkDescriptorSet m_gizmo_desc_set;

  void create_gizmo();

  // Setup
  void create_instance();
  void create_surface();
  void pick_physical_device();
  void create_device();
  void create_swapchain();
  void create_depth_resources();
  void create_render_pass();
  void create_descriptor_layout();
  void create_pipelines();
  void create_buffers(const int* skeleton_edges, int num_edges);
  void create_descriptor_set();
  void create_sync_objects();
  void create_command_pool();
  void init_colors();

  void update_camera();
  VkShaderModule load_shader(const char* path);
  uint32_t find_memory_type(uint32_t type_filter, VkMemoryPropertyFlags properties);

  static void mouse_button_callback(GLFWwindow* window, int button, int action, int mods);
  static void cursor_pos_callback(GLFWwindow* window, double xpos, double ypos);
  static void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
};

#endif // RENDERER_HPP
