#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include "TextureLoader.h"

#include <cstring>
#include <stdexcept>

TextureData LoadTexture(const std::string& InPath)
{
    TextureData Data{};

    stbi_set_flip_vertically_on_load(true);

    int W = 0, H = 0, C = 0;
    stbi_uc* Pixels = stbi_load(
        InPath.c_str(),
        &W, &H, &C,
        STBI_rgb_alpha);

    if (!Pixels)
        throw std::runtime_error("Failed to load texture: " + InPath);

    Data.Width = W;
    Data.Height = H;
    Data.Channels = 4;
    Data.Pixels.resize(static_cast<size_t>(W) * static_cast<size_t>(H) * 4);

    std::memcpy(Data.Pixels.data(), Pixels, Data.Pixels.size());
    stbi_image_free(Pixels);

    return Data;
}