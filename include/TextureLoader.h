#pragma once

#include <cstdint>
#include <string>
#include <vector>

struct TextureData
{
    int Width = 0;
    int Height = 0;
    int Channels = 0;

    // sempre RGBA8, così semplifichi tutto
    std::vector<uint8_t> Pixels;
};

TextureData LoadTexture(const std::string& InPath);