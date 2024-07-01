#version 460 core

#define HAS_TEXTURE 1
#define HAS_NORMALS 1
#define HAS_COLORS 0
#define HAS_BASE_COLOR 0

#if HAS_NORMALS
in layout(location = 0) vec3 inNormal;
#endif
#if HAS_TEXTURE
in layout(location = 2) vec2 inUv;
#endif
#if HAS_COLORS
in layout(location = 3) vec3 inColor;
#endif
#if HAS_BASE_COLOR
in layout(location = 4) vec4 inBaseColor;
#endif

#if HAS_TEXTURE
uniform layout(set = 0, binding = 0) sampler2D baseColorSampler;
#endif

out layout(location = 0) vec4 outColor;

void main() {
  outColor = vec4(1.0, 0.7, 0.7, 1.0);
  // First get colors from any source available, overwrite with more likely
  //   sources
#if HAS_NORMALS
  outColor = vec4(inNormal, 1.0);
  outColor.xyz = (outColor.xyz + 1.0) / 2.0;
#endif
#if HAS_TEXTURE
  outColor = texture(baseColorSampler, inUv);
  outColor.a = 1.0;
#endif
#if HAS_COLORS
  outColor = vec4(inColor, 1.0);
  outColor.a = 1.0;
#endif
#if HAS_BASE_COLOR
  outColor = inBaseColor;
  outColor.a = 1.0;
#endif

  // then compute real fake lighting if possible
#if HAS_NORMALS
  vec3 lightDir = normalize(vec3(1.0, 1.0, 1.0));
  outColor.rgb *= max(dot(inNormal, lightDir), 0.5);
  outColor.rgb *= max(dot(inNormal, -lightDir), 0.5);
  outColor.rgb = pow(outColor.rgb, vec3(1.0/2.2));
#endif
}
