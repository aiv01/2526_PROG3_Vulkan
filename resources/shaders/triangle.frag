#version 460

layout(set = 0, binding = 0) uniform sampler2D texSampler;

layout(location = 0) in vec3 outNormal;
layout(location = 1) in vec2 outUV;

layout(location = 0) out vec4 outColor;

void main()
{
    outColor = texture(texSampler, outUV);
}