#version 450

layout(location = 0) in vec3 FragNormal;
layout(location = 1) in vec3 FragTexCoord;

layout(location = 0) out vec4 OutColor;

void main()
{
    vec3 Normal = normalize(FragNormal) * 0.5 + 0.5;
    OutColor = vec4(Normal, 1.0);
}
