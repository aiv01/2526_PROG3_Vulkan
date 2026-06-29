#version 460

layout(location = 0) in vec3 FragNormal;
layout(location = 1) in vec2 FragUV;

layout(set = 0, binding = 0) uniform sampler2D texSampler;

layout(location = 0) out vec4 OutColor;

void main()
{
    OutColor = texture(texSampler, FragUV);
}