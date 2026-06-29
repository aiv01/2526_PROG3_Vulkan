#version 460

layout(location = 0) in vec3 inPos;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inUV;

layout(location = 0) out vec3 outNormal;
layout(location = 1) out vec2 outUV;

layout(push_constant) uniform PushData
{
    mat4 Mvp;
    mat4 Model;
} pc;

void main()
{
    outNormal = inNormal;
    outUV = inUV;
    gl_Position = pc.Mvp * vec4(inPos, 1.0);
}