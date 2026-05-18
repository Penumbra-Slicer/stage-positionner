#version 450
layout(location = 0) in  vec3  fragWorldPos;
layout(location = 0) out float outSuction;
void main() {
    vec3 N = normalize(cross(dFdx(fragWorldPos), dFdy(fragWorldPos)));
    outSuction = (N.z > 0.0) ? 1.0 : 0.0;
}
