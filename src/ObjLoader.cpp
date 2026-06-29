#include "ObjLoader.h"

#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>

#include <iostream>
#include <stdexcept>
#include <vector>

//i thouht about an usefull struct to compare all vertices info
struct PackedVertex
{
    float Px = 0.0f, Py = 0.0f, Pz = 0.0f;
    float Nx = 0.0f, Ny = 0.0f, Nz = 0.0f;
    float U  = 0.0f, V  = 0.0f;

    bool operator==(const PackedVertex& Other) const
    {
        return Px == Other.Px && Py == Other.Py && Pz == Other.Pz &&
               Nx == Other.Nx && Ny == Other.Ny && Nz == Other.Nz &&
               U  == Other.U  && V  == Other.V;
    }
};

//TODO Vertex exist yet?  implement search function
static uint32_t FindVertex(
    const std::vector<PackedVertex>& UniqueVertices,
    const PackedVertex& Vertex)
{
    for (uint32_t i = 0; i < UniqueVertices.size(); ++i)
    {
        if (UniqueVertices[i] == Vertex)
            return i;
    }

    return UINT32_MAX;//false case
}


ModelData LoadObj(const std::string& InPath)
{
    //declaring all the parameters needed and using tinyobj
    tinyobj::attrib_t Attrib;
    std::vector<tinyobj::shape_t> Shapes;
    std::vector<tinyobj::material_t> Materials;
    std::string Warning;
    std::string Error;
    
    const bool Ok = tinyobj::LoadObj(&Attrib,&Shapes,&Materials,&Warning,&Error,InPath.c_str(),nullptr,true);
    //NOW  all infos are inside and Attrib and indices inside shapes?
    if (!Warning.empty())
    {
        std::cerr << "[ObjLoader Warning] " << Warning << '\n';
    }
    if (!Ok)
    {
        throw std::runtime_error("Failed to load OBJ in : " + InPath );
    }

    ModelData Data{}; //so this have Vertices vector ,indices vector and counter to fill
    std::vector<PackedVertex> UniqueVertices; //my readed vertices array

    //for every index inside every shape we load the info in the support struct and Load it in the vertices array if not present
    for (const auto& Shape : Shapes)
    {
        const auto& Indices = Shape.mesh.indices;

        for (size_t i = 0; i + 2 < Indices.size(); i += 3)
        {
            uint32_t TriangleIndices[3];

            for (int Corner = 0; Corner < 3; ++Corner)
            {
                const auto& Index = Indices[i + Corner];

                PackedVertex Vertex{};

                if (Index.vertex_index >= 0)
                {
                    const size_t P = static_cast<size_t>(Index.vertex_index) * 3;
                    Vertex.Px = Attrib.vertices[P + 0];
                    Vertex.Py = Attrib.vertices[P + 1];
                    Vertex.Pz = Attrib.vertices[P + 2];
                }

                if (Index.normal_index >= 0 && !Attrib.normals.empty())
                {
                    const size_t N = static_cast<size_t>(Index.normal_index) * 3;
                    Vertex.Nx = Attrib.normals[N + 0];
                    Vertex.Ny = Attrib.normals[N + 1];
                    Vertex.Nz = Attrib.normals[N + 2];
                }

                if (Index.texcoord_index >= 0 && !Attrib.texcoords.empty())
                {
                    const size_t T = static_cast<size_t>(Index.texcoord_index) * 2;
                    Vertex.U = Attrib.texcoords[T + 0];
                    Vertex.V = Attrib.texcoords[T + 1];
                }

                uint32_t ExistingIndex = FindVertex(UniqueVertices, Vertex);

                if (ExistingIndex != UINT32_MAX)
                {
                    TriangleIndices[Corner] = ExistingIndex;
                }
                else
                {
                    const uint32_t NewIndex = static_cast<uint32_t>(UniqueVertices.size());
                    UniqueVertices.push_back(Vertex);

                    Data.Vertices.push_back(Vertex.Px);
                    Data.Vertices.push_back(Vertex.Py);
                    Data.Vertices.push_back(Vertex.Pz);
                    Data.Vertices.push_back(Vertex.Nx);
                    Data.Vertices.push_back(Vertex.Ny);
                    Data.Vertices.push_back(Vertex.Nz);
                    Data.Vertices.push_back(Vertex.U);
                    Data.Vertices.push_back(Vertex.V);

                    TriangleIndices[Corner] = NewIndex;
                }
            }

            // winding invertito
            Data.Indices.push_back(TriangleIndices[0]);
            Data.Indices.push_back(TriangleIndices[2]);
            Data.Indices.push_back(TriangleIndices[1]);
        }
    }

    //final counters setting
    Data.VertexCount = static_cast<uint32_t>(UniqueVertices.size());
    Data.IndexCount   = static_cast<uint32_t>(Data.Indices.size());

    return Data;
}