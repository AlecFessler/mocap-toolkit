#ifndef SHARED_DEFS_H
#define SHARED_DEFS_H

constexpr size_t IMAGE_WIDTH = 1920;
constexpr size_t IMAGE_HEIGHT = 1080;
constexpr size_t IMAGE_BYTES = IMAGE_WIDTH * IMAGE_HEIGHT;

#define PTR_MATH_CAST(type, ptr, offset) (type*)((char*)ptr + offset)

#endif
