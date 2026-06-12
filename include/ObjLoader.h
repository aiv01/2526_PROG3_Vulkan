#pragma once

#include <vector>
#include <string>
#include <cstdint>


struct ModelData 
{
    std::vector<float>    Vertices; // interleaved: pos(3) + normal(3) + uv(2) per vertex
    std::vector<uint32_t> Indices;  
    uint32_t              VertexCount = 0;
    uint32_t              IndexCount = 0;
};

ModelData LoadObj(const std::string& InPath);
