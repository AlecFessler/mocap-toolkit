#version 450

layout(binding = 0) uniform UBO {
    mat4 view;
    mat4 projection;
} ubo;

layout(location = 0) in vec3 position;
layout(location = 1) in vec4 color;

layout(location = 0) out vec4 frag_color;

void main() {
    gl_Position = ubo.projection * ubo.view * vec4(position, 1.0);
    gl_PointSize = 6.0;
    frag_color = color;
}
