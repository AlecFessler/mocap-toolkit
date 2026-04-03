#version 450

layout(location = 0) in vec2 frag_uv;
layout(location = 0) out vec4 out_color;

layout(binding = 0) uniform sampler2D font_atlas;

void main() {
    float a = texture(font_atlas, frag_uv).r;
    if (a < 0.1) discard;
    out_color = vec4(0.9, 0.9, 0.9, a);
}
