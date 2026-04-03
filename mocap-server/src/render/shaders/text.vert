#version 450

layout(location = 0) in vec2 pos;
layout(location = 1) in vec2 uv;

layout(location = 0) out vec2 frag_uv;

layout(push_constant) uniform PC {
    vec2 screen_size;
} pc;

void main() {
    vec2 ndc = (pos / pc.screen_size) * 2.0 - 1.0;
    gl_Position = vec4(ndc, 0.0, 1.0);
    frag_uv = uv;
}
