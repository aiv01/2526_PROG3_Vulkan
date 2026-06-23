#include "ObjLoader.h"
#include "Common.h"

#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>

#include <map>
#include <tuple>

ModelData LoadObj(const std::string& InPath)
{
    tinyobj::attrib_t Attrib;
    std::vector<tinyobj::shape_t> Shapes;
    std::vector<tinyobj::material_t> Materials;
    std::string Warn, Err;

    CHECK_DIE(tinyobj::LoadObj(&Attrib, &Shapes, &Materials, &Warn, &Err, InPath.c_str()),
        "Failed to load OBJ file");

    ModelData Data;

    using VertexKey = std::tuple<int, int, int>;
    std::map<VertexKey, uint32_t> UniqueVertices;

    for (const auto& Shape : Shapes)
    {
        for (const auto& Idx : Shape.mesh.indices)
        {
            VertexKey Key{ Idx.vertex_index, Idx.normal_index, Idx.texcoord_index };

            auto It = UniqueVertices.find(Key);
            if (It != UniqueVertices.end()) {
                Data.Indices.push_back(It->second);
                continue;
            }

            uint32_t NewIndex = static_cast<uint32_t>(UniqueVertices.size());
            UniqueVertices[Key] = NewIndex;

            // Position (3 floats)
            int Vi = Idx.vertex_index * 3;
            Data.Vertices.push_back(Attrib.vertices[Vi + 0]);
            Data.Vertices.push_back(Attrib.vertices[Vi + 1]);
            Data.Vertices.push_back(Attrib.vertices[Vi + 2]);

            // Normal (3 floats)
            if (Idx.normal_index >= 0) {
                int Ni = Idx.normal_index * 3;
                Data.Vertices.push_back(Attrib.normals[Ni + 0]);
                Data.Vertices.push_back(Attrib.normals[Ni + 1]);
                Data.Vertices.push_back(Attrib.normals[Ni + 2]);
            } else { // fallback or exception
                Data.Vertices.push_back(0.0f);
                Data.Vertices.push_back(1.0f);
                Data.Vertices.push_back(0.0f);
            }

            // TexCoord (2 floats)
            if (Idx.texcoord_index >= 0) {
                int Ti = Idx.texcoord_index * 2;
                Data.Vertices.push_back(Attrib.texcoords[Ti + 0]);
                Data.Vertices.push_back(1.0f - Attrib.texcoords[Ti + 1]); // flip V for Vulkan
            } else { // fallback or exception
                Data.Vertices.push_back(0.0f);
                Data.Vertices.push_back(0.0f);
            }

            Data.Indices.push_back(NewIndex);
        }
    }

    Data.VertexCount = static_cast<uint32_t>(UniqueVertices.size());
    Data.IndexCount  = static_cast<uint32_t>(Data.Indices.size());

    LOG_DEBUG("OBJ loaded: {} vertices, {} indices", Data.VertexCount, Data.IndexCount);

    return Data;
}
