#version 460

layout(location = 0) in vec3 FragWorldPos;
layout(location = 1) in vec3 FragWorldNormal;
layout(location = 2) in vec2 FragUV;

layout(set = 0, binding = 0) uniform sampler2D texSampler;

layout(push_constant) uniform PushData
{
    mat4 Mvp;
    mat4 Model;
    vec4 LightDir;
    vec4 LightColor;
} pc;

layout(location = 0) out vec4 OutColor;

void main()
{
    vec3 albedo = texture(texSampler, FragUV).rgb;

    vec3 N = normalize(FragWorldNormal);
    vec3 L = normalize(pc.LightDir.xyz);
    vec3 V = normalize(-FragWorldPos);

    float diff = max(dot(N, L), 0.0);

    vec3 ambient = 0.15 * albedo;
    vec3 diffuse = diff * albedo * pc.LightColor.rgb;

    vec3 R = reflect(-L, N);
    float spec = pow(max(dot(V, R), 0.0), 32.0);
    vec3 specular = 0.25 * spec * pc.LightColor.rgb;

    OutColor = vec4(ambient + diffuse + specular, 1.0);
}