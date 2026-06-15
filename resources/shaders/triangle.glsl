#version 460

#ifdef VERTEX_SHADER

layout(location = 0) out vec3 FragColor;

vec2 Positions[3] = vec2[](
    vec2( 0.0, -0.5 ),
    vec2( 0.5,  0.5 ),
    vec2(-0.5,  0.5 )
);

vec3 Colors[3] = vec3[](
    vec3(1.0, 0.0, 0.0),
    vec3(0.0, 1.0, 0.0),
    vec3(0.0, 0.0, 1.0)
);

void main()
{
    gl_Position = vec4(Positions[gl_VertexIndex], 0.0, 1.0);
    FragColor = Colors[gl_VertexIndex];
}

#endif

#ifdef FRAGMENT_SHADER

layout(location = 0) in vec3 FragColor;
layout(location = 0) out vec4 OutColor;

void main()
{
    OutColor = vec4(FragColor, 1.0);
}

#endif