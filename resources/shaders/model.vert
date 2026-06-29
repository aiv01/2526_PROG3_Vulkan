#version 460

layout(location = 0) in vec3 inPos;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inUV;

layout(location = 0) out vec3 outWorldPos;
layout(location = 1) out vec3 outWorldNormal;
layout(location = 2) out vec2 outUV;

layout(push_constant) uniform PushData
{
    mat4 Mvp;
    mat4 Model;
    vec4 LightDir;
    vec4 LightColor;
} pc;

void main()
{
    vec4 worldPos = pc.Model * vec4(inPos, 1.0);
    outWorldPos = worldPos.xyz;
    outWorldNormal = normalize(mat3(pc.Model) * inNormal);
    outUV = inUV;
    gl_Position = pc.Mvp * vec4(inPos, 1.0);
}