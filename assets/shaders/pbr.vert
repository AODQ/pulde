#version 460 core

#define HAS_TEXTURE 1
#define HAS_NORMALS 1
#define HAS_COLORS 0
#define HAS_BASE_COLOR 0

in layout(location = 0) vec4 inPosition;
#if HAS_NORMALS
in layout(location = 1) vec3 inNormal;
#endif
#if HAS_TEXTURE
in layout(location = 3) vec2 inUv;
#endif
#if HAS_COLORS
in layout(location = 7) vec3 inColor;
#endif

layout(push_constant) uniform PushConstants {
  mat4 cameraViewProj;
  mat4 transform;
#if HAS_BASE_COLOR
  vec4 baseColor;
#endif
};

#if HAS_NORMALS
out layout(location = 0) vec3 outNormal;
#endif
#if HAS_TEXTURE
out layout(location = 2) vec2 outUv;
#endif
#if HAS_COLORS
out layout(location = 3) vec3 outColor;
#endif
#if HAS_BASE_COLOR
out layout(location = 4) vec4 outBaseColor;
#endif

out gl_PerVertex {
  vec4 gl_Position;
};

mat4 glLookAt(vec3 eye, vec3 center, vec3 up) {
  vec3 f = normalize(center - eye);
  vec3 s = normalize(cross(f, up));
  vec3 u = cross(s, f);

  mat4 result = mat4(
    vec4(s, 0.0f),
    vec4(u, 0.0f),
    vec4(-f, 0.0f),
    vec4(0.0f, 0.0f, 0.0f, 1.0f)
  );

  result = transpose(result);
  result[3] = vec4(-eye, 1.0f);

  return result;
}

mat4 glPerspective(float fovy, float aspect, float near, float far) {
  float f = 1.0f / tan(fovy / 2.0f);
  float nf = 1.0f / (near - far);

  return mat4(
    vec4(f / aspect, 0.0f, 0.0f, 0.0f),
    vec4(0.0f, f, 0.0f, 0.0f),
    vec4(0.0f, 0.0f, (far + near) * nf, -1.0f),
    vec4(0.0f, 0.0f, 2.0f * far * near * nf, 0.0f)
  );
}

void main() {
  gl_Position = cameraViewProj * transform * vec4(inPosition.xyz, 1.0f);
#if HAS_NORMALS
  outNormal = normalize(inNormal);
#endif
#if HAS_TEXTURE
  outUv = inUv;
#endif
#if HAS_COLORS
  outColor = inColor;
#endif
#if HAS_BASE_COLOR
  outBaseColor = baseColor;
#endif
}
