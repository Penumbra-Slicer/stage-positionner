#version 450
layout(location = 0) in vec3 inPos;
layout(push_constant) uniform PC { mat4 mvp; } pc;
layout(location = 0) out vec3 fragWorldPos;
void main() {
    fragWorldPos = inPos;
    gl_Position  = pc.mvp * vec4(inPos, 1.0);
}
