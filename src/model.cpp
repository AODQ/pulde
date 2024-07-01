#include <pulchritude/asset-image.hpp>
#include <pulchritude/asset-model.hpp>
#include <pulchritude/core.hpp>
#include <pulchritude/core.hpp>
#include <pulchritude/error.hpp>
#include <pulchritude/log.hpp>
#include <pulchritude/plugin.hpp>

#include <string>
#include <unordered_map>
#include <vector>

static void assignAttribute(
  PuleAssetModel & model,
  PuleAssetModelMeshPrimitive & primitive,
  int64_t attributeIndex,
  PuleStringView label,
  PuleError * error
) {
  if (puleStringViewEq(label, "NORMAL"_psv)) {
    primitive.attributeAccessors[PuleAssetModelAttribute_normal] = (
      model.accessors + attributeIndex
    );
    return;
  }
  if (puleStringViewEq(label, "POSITION"_psv)) {
    primitive.attributeAccessors[PuleAssetModelAttribute_origin] = (
      model.accessors + attributeIndex
    );
    return;
  }
  if (puleStringViewEq(label, "TANGENT"_psv)) {
    primitive.attributeAccessors[PuleAssetModelAttribute_tangent] = (
      model.accessors + attributeIndex
    );
    return;
  }
  if (puleStringViewEq(label, "TEXCOORD_0"_psv)) {
    primitive.attributeAccessors[PuleAssetModelAttribute_uvcoord_0] = (
      model.accessors + attributeIndex
    );
    return;
  }
  if (puleStringViewEq(label, "TEXCOORD_1"_psv)) {
    primitive.attributeAccessors[PuleAssetModelAttribute_uvcoord_1] = (
      model.accessors + attributeIndex
    );
    return;
  }
  if (puleStringViewEq(label, "COLOR_0"_psv)) {
    primitive.attributeAccessors[PuleAssetModelAttribute_color_0] = (
      model.accessors + attributeIndex
    );
    return;
  }
  if (puleStringViewEq(label, "JOINTS_0"_psv)) {
    primitive.attributeAccessors[PuleAssetModelAttribute_joints_0] = (
      model.accessors + attributeIndex
    );
    return;
  }
  if (puleStringViewEq(label, "WEIGHTS_0"_psv)) {
    primitive.attributeAccessors[PuleAssetModelAttribute_weights_0] = (
      model.accessors + attributeIndex
    );
    return;
  }
  PULE_error(PuleErrorAssetModel_decode, "invalid attribute label");
}

static void assetModelLoadScene(
  PuleAssetModel & model,
  std::string const & basePath,
  bool loadBuffers,
  bool loadImages,
  PuleDsValue modelValue,
  PuleError * error
) {
  auto & allocator = model.allocator;

  auto allocateWithCanary = [&](size_t byteLen) -> void * {
    // allocate memory with canary, first align to 4 bytes
    size_t alignedByteLen = byteLen + (4 - (byteLen % 4));
    void * data = puleMalloc(allocator, alignedByteLen + sizeof(uint32_t));
    memset(data, 0, alignedByteLen);
    uint32_t * canary = (uint32_t *)((uint8_t *)data + alignedByteLen);
    *canary = 0xdeadbeef;
    return data;
  };

  // allocate memory (allocate one extra for dummy data, last elem)
  // also allocate an extra for the canary to check for overflows
  #define modelAlloc(names, nameLen, type, extraLen) \
    PuleDsValueArray names = puleDsMemberAsArray(modelValue, #names ); \
    model. nameLen = names .length + extraLen; \
    model. names = ( \
      (type *) \
      allocateWithCanary(model. nameLen * sizeof(type)) \
    );
  #define modelQuickAlloc(name, type, extraLen) \
    modelAlloc(name ## s, name ## Len, type, extraLen)

  // before allocating need to figure out how many mesh primitives have an
  //   element-index buffer, in which case we need to supply that data
  size_t missingElementIdxBuffers = 0;
  size_t hasMissingAttributes = 0;
  size_t hasMissingElementIdxBuffers = 0;
  {
    auto meshes = puleDsMemberAsArray(modelValue, "meshes");
    for (uint32_t i = 0; i < meshes.length; ++ i) {
      PuleDsValue meshValue = meshes.values[i];
      auto primitives = puleDsMemberAsArray(meshValue, "primitives");
      for (uint32_t j = 0; j < primitives.length; ++ j) {
        PuleDsValue primitiveValue = primitives.values[j];
        if (!puleDsObjectMember(primitiveValue, "indices").id) {
          ++ missingElementIdxBuffers;
          hasMissingElementIdxBuffers = 1;
        }
        for (auto & required : {
          "NORMAL", // later more will be required
        }) {
          if (
            !puleDsObjectMember(
              puleDsObjectMember(primitiveValue, "attributes"), required
            ).id
          ) {
            hasMissingAttributes = 1;
          }
        }
      }
    }
  }

  // for most things the allocation is [ pds data ]
  // however for defaults/dummy-data it will look like  [ pds data | dummy ]
  // and for accessors that are missing element-index buffers it will look like
  //   [ pds data | dummy | missing element-index accessors ]
  // the dummy data will just be a flat list of u8 255s
  // the missing element-index accessors will be a monotonically increasing
  //   list of u16s
  // buffer/buffer-view will also have the buffer data for the missing
  //   element-index accessors

  modelQuickAlloc(
    accessor,
    PuleAssetModelAccessor,
    hasMissingAttributes+missingElementIdxBuffers
  );
  modelQuickAlloc(
    buffer,
    PuleAssetModelBuffer,
    hasMissingAttributes + hasMissingElementIdxBuffers
  );
  modelQuickAlloc(
    bufferView,
    PuleAssetModelBufferView,
    hasMissingAttributes + hasMissingElementIdxBuffers
  );
  modelQuickAlloc(material,   PuleAssetModelMaterial, 0);
  modelQuickAlloc(node,       PuleAssetModelNode, 0);
  modelQuickAlloc(scene,      PuleAssetModelScene, 0);
  modelQuickAlloc(texture,    PuleAssetModelTexture, 0);
  modelQuickAlloc(image,      PuleAssetModelImage, 0);
  modelQuickAlloc(animation,  PuleAssetModelAnimation, 0);
  modelAlloc(meshes, meshLen, PuleAssetModelMesh, 0);

  #undef modelAlloc
  #undef modelQuickAlloc

  auto applyName = [&](PuleDsValue value, std::string const & type, size_t it) {
    PuleStringView name = puleDsMemberAsString(value, "name");
    if (name.contents != nullptr) {
      return puleStringCopy(allocator, name);
    }
    return puleStringFormat(allocator, "%s_%zu", type.c_str(), it);
  };

  // order we load doesn't matter since data was allocated

  // - load buffers
  for (uint32_t i = 0; i < buffers.length; ++ i) {
    PuleDsValue bufferValue = buffers.values[i];
    PuleAssetModelBuffer & buffer = model.buffers[i];
    buffer.name = applyName(bufferValue, "buffer", i);
    buffer.index = i;
    buffer.resourceUri = (
      puleStringCopy(allocator, puleDsMemberAsString(bufferValue, "uri"))
    );
    if (loadBuffers) {
      // load up URI
      PuleError err = puleError();
      std::string uri = (
        basePath + "/" + std::string(buffer.resourceUri.contents)
      );
      PuleBuffer uriContents = (
        puleFileDumpContents(
          allocator,
          puleCStr(uri.c_str()),
          PuleFileDataMode_binary,
          &err
        )
      );
      puleScopeExit { puleBufferDestroy(uriContents); };
      if (puleErrorConsume(&err)) { continue; }
      buffer.data = (
        puleBufferCopyWithData(
          allocator, uriContents.data, uriContents.byteLength
        )
      );
      buffer.data.byteLength = uriContents.byteLength;
    }
  }

  // - load images
  for (uint32_t i = 0; i < images.length; ++ i) {
    PuleDsValue imageValue = images.values[i];
    PuleAssetModelImage & image = model.images[i];
    image.name = applyName(imageValue, "image", i);
    image.index = i;
    image.resourceUri = (
      puleStringCopy(allocator, puleDsMemberAsString(imageValue, "uri"))
    );
    if (loadImages) {
      // load URI
      std::string uri = (
        basePath + "/" + std::string(image.resourceUri.contents)
      );
      PuleFileStream fileStream = (
        puleFileStreamReadOpen(puleCStr(uri.c_str()), PuleFileDataMode_binary)
      );
      puleScopeExit { puleFileStreamClose(fileStream); };
      PuleError err = puleError();
      puleLogDev("loading image %s", uri.c_str());
      image.image = (
        puleAssetImageLoadFromStream(
          allocator,
          puleFileStreamReader(fileStream),
          PuleAssetImageFormat_rgbaU8,
          &err
        )
      );
      if (puleErrorConsume(&err)) { continue; }
      puleLog("loaded image %s", uri.c_str());
    }
  }

  // - load buffer views
  for (uint32_t i = 0; i < bufferViews.length; ++ i) {
    PuleDsValue bufferViewValue = bufferViews.values[i];
    PuleAssetModelBufferView & bufferView = model.bufferViews[i];
    bufferView.buffer = (
      model.buffers + puleDsMemberAsI64(bufferViewValue, "buffer")
    );
    bufferView.name = applyName(bufferViewValue, "bufferView", i);
    bufferView.index = i;
    bufferView.byteLen = puleDsMemberAsI64(bufferViewValue, "byteLength");
    bufferView.byteOffset = puleDsMemberAsI64(bufferViewValue, "byteOffset");
    bufferView.byteStride = puleDsMemberAsI64(bufferViewValue, "byteStride");
    switch (puleDsMemberAsI64(bufferViewValue, "target")) {
      case 0:
        bufferView.usage = PuleAssetModelBufferViewUsage_none;
      break;
      case 34962:
        bufferView.usage = PuleAssetModelBufferViewUsage_attribute;
      break;
      case 34963:
        bufferView.usage = PuleAssetModelBufferViewUsage_elementIdx;
      break;
      default:
        PULE_error(
          PuleErrorAssetModel_decode,
          "invalid element-target: %d",
          puleDsMemberAsI64(bufferViewValue, "target")
        );
        return;
    }
  }

  // - load accessors
  for (uint32_t i = 0; i < accessors.length; ++ i) {
    PuleDsValue accessorValue = accessors.values[i];
    PuleAssetModelAccessor & accessor = model.accessors[i];

    accessor.name = applyName(accessorValue, "accessor", i);
    accessor.index = i;
    accessor.bufferView = (
      model.bufferViews + puleDsMemberAsI64(accessorValue, "bufferView")
    );
    accessor.byteOffset = puleDsMemberAsI64(accessorValue, "byteOffset");
    accessor.elementCount = puleDsMemberAsI64(accessorValue, "count");
    accessor.convertFixedToNormalized = (
      puleDsMemberAsBool(accessorValue, "normalized")
    );
    auto elementType = puleDsMemberAsString(accessorValue, "type");

    if (puleStringViewEq(elementType, "SCALAR"_psv)) {
      accessor.elementType = PuleAssetModelElementType_scalar;
    } else if (puleStringViewEq(elementType, "VEC2"_psv)) {
      accessor.elementType = PuleAssetModelElementType_vec2;
    } else if (puleStringViewEq(elementType, "VEC3"_psv)) {
      accessor.elementType = PuleAssetModelElementType_vec3;
    } else if (puleStringViewEq(elementType, "VEC4"_psv)) {
      accessor.elementType = PuleAssetModelElementType_vec4;
    } else if (puleStringViewEq(elementType, "MAT2"_psv)) {
      accessor.elementType = PuleAssetModelElementType_mat2;
    } else if (puleStringViewEq(elementType, "MAT3"_psv)) {
      accessor.elementType = PuleAssetModelElementType_mat3;
    } else if (puleStringViewEq(elementType, "MAT4"_psv)) {
      accessor.elementType = PuleAssetModelElementType_mat4;
    } else {
      PULE_error(PuleErrorAssetModel_decode, "invalid element type");
      return;
    }

    switch (puleDsMemberAsI64(accessorValue, "componentType")) {
      case 5120:
        PULE_error(PuleErrorAssetModel_decode, "i8 not supported");
        return;
      break;
      case 5121:
        accessor.dataType = PuleGpuAttributeDataType_u8;
        //PULE_error(PuleErrorAssetModel_decode, "u8 not supported");
      break;
      case 5122:
        PULE_error(PuleErrorAssetModel_decode, "i16 not supported");
      break;
      case 5123:
        accessor.dataType = PuleGpuAttributeDataType_u16;
      break;
      case 5124:
        PULE_error(PuleErrorAssetModel_decode, "i32 not supported");
      break;
      case 5125:
        accessor.dataType = PuleGpuAttributeDataType_u32;
      break;
      case 5126:
        accessor.dataType = PuleGpuAttributeDataType_f32;
      break;
      default:
        PULE_error(PuleErrorAssetModel_decode, "invalid component type");
        return;
    }
  }

  // - load meshes
  for (uint32_t i = 0; i < meshes.length; ++ i) {
    PuleDsValue meshValue = meshes.values[i];
    PuleAssetModelMesh & mesh = model.meshes[i];

    mesh.name = applyName(meshValue, "mesh", i);
    mesh.index = i;

    { // mesh weights
      auto weights = puleDsMemberAsArray(meshValue, "weights");
      mesh.weightLen = weights.length;
      mesh.weights = (
        (float *)puleMalloc(allocator, weights.length * sizeof(float))
      );
      for (uint32_t weightIt = 0; weightIt < weights.length; ++ weightIt) {
        mesh.weights[weightIt] = puleDsAsF32(weights.values[weightIt]);
      }
    }

    // mesh primitives
    PuleDsValueArray primitives = puleDsMemberAsArray(meshValue, "primitives");
    mesh.primitives = (
      (PuleAssetModelMeshPrimitive *)
      puleMalloc(
        allocator,
        primitives.length * sizeof(PuleAssetModelMeshPrimitive)
      )
    );
    mesh.primitiveLen = primitives.length;
    for (uint32_t j = 0; j < mesh.primitiveLen; ++ j) {
      PuleDsValue primitiveValue = primitives.values[j];
      auto & primitive = mesh.primitives[j];
      if (puleDsObjectMember(primitiveValue, "material").id == 0) {
        primitive.material = nullptr;
      } else {
        primitive.material = (
          model.materials + puleDsMemberAsI64(primitiveValue, "material")
        );
      }
      auto modeMap = (
        std::unordered_map<int32_t, PuleGpuDrawPrimitive> {
          {0, PuleGpuDrawPrimitive_point},
          {1, PuleGpuDrawPrimitive_line},
          {4, PuleGpuDrawPrimitive_triangle},
          {5, PuleGpuDrawPrimitive_triangleStrip},
        }
      );
      if (puleDsObjectMember(primitiveValue, "mode").id == 0) {
        primitive.topology = PuleGpuDrawPrimitive_triangle; // default
      } else {
        auto mode = puleDsMemberAsI64(primitiveValue, "mode");
        auto modeFind = modeMap.find(mode);
        if (modeFind == modeMap.end()) {
          PULE_error(PuleErrorAssetModel_decode, "invalid mode");
          return;
        }
        primitive.topology = modeFind->second;
      }
      if (puleDsObjectMember(primitiveValue, "indices").id == 0) {
        // this is fine
        primitive.elementIdxAccessor = nullptr;
      } else {
        primitive.elementIdxAccessor = (
          model.accessors + puleDsMemberAsI64(primitiveValue, "indices")
        );
      }
      auto attributes = puleDsMemberAsObject(primitiveValue, "attributes");
      // clear out attributes
      memset(
        primitive.attributeAccessors, 0, sizeof(primitive.attributeAccessors)
      );
      for (uint32_t attrIt = 0; attrIt < attributes.length; ++ attrIt) {
        auto & attribute = attributes.values[attrIt];
        auto & label = attributes.labels[attrIt];
        assignAttribute(model, primitive, puleDsAsI64(attribute), label, error);
        if (puleErrorExists(error)) { return; }
      }
      PULE_assert(attributes.values[0].id && "Must have at least origin");

      { // morph weights
      }

      { // morph targets
        auto targets = puleDsMemberAsArray(primitiveValue, "targets");
        primitive.morphTargetLen = targets.length;
        primitive.morphTargets = (
          (PuleAssetModelMorphTarget *)
          puleMalloc(
            allocator,
            targets.length * sizeof(PuleAssetModelMorphTarget)
          )
        );
        for (size_t targetIt = 0; targetIt < targets.length; ++ targetIt) {
          auto & morphTarget = primitive.morphTargets[targetIt];
          auto const target = targets.values[targetIt];
          auto position = puleDsObjectMember(target, "POSITION");
          auto normal = puleDsObjectMember(target, "NORMAL");
          auto tangent = puleDsObjectMember(target, "TANGENT");
          if (position.id) {
            morphTarget.attributeAccessor[PuleAssetModelAttribute_origin] = (
              model.accessors + puleDsAsI64(position)
            );
          }
          if (normal.id) {
            morphTarget.attributeAccessor[PuleAssetModelAttribute_normal] = (
              model.accessors + puleDsAsI64(normal)
            );
          }
          if (tangent.id) {
            morphTarget.attributeAccessor[PuleAssetModelAttribute_tangent] = (
              model.accessors + puleDsAsI64(tangent)
            );
          }
        }
      }

      // draw element count
      primitive.drawElementCount = (
        primitive.elementIdxAccessor
        ? primitive.elementIdxAccessor->elementCount
        : (
          primitive.attributeAccessors[
            PuleAssetModelAttribute_origin
          ]->elementCount
        )
      );
    }
  }

  // - load materials
  for (uint32_t i = 0; i < materials.length; ++ i) {
    PuleDsValue materialValue = materials.values[i];
    PuleAssetModelMaterial & material = model.materials[i];
    material.name = applyName(materialValue, "material", i);
    material.index = i;

    // assign default values
    auto & pbr = material.pbrMetallicRoughness;
    pbr.baseColorTexture = nullptr;
    pbr.baseColorFactor = { 1.0f, 1.0f, 1.0f, 1.0f, };
    pbr.metallicFactor = 0.0f;
    pbr.roughnessFactor = 0.0f;

    auto pbrValue = puleDsObjectMember(materialValue, "pbrMetallicRoughness");

    // check for textures
    if (puleDsObjectMember(pbrValue, "baseColorTexture").id) {
      auto baseColorTexture = puleDsObjectMember(pbrValue, "baseColorTexture");
      auto const index = puleDsMemberAsI64(baseColorTexture, "index");
      pbr.baseColorTexture = model.textures + index;
    }

    // check for factors
    auto baseColorFactor = puleDsObjectMember(pbrValue, "baseColorFactor");
    if (baseColorFactor.id) {
      pbr.baseColorFactor = puleDsAsF32v4(baseColorFactor);
    }
  }

  // - load nodes
  for (uint32_t i = 0; i < nodes.length; ++ i) {
    PuleDsValue nodeValue = nodes.values[i];
    PuleAssetModelNode & node = model.nodes[i];
    node.name = applyName(nodeValue, "node", i);
    node.index = i;
    // children
    node.children = nullptr;
    node.childrenLen = 0;
    if (puleDsObjectMember(nodeValue, "children").id) {
      auto childrenValue = puleDsMemberAsArray(nodeValue, "children");
      node.childrenLen = childrenValue.length;
      node.children = (
        (PuleAssetModelNode **)
        puleMalloc(allocator, node.childrenLen * sizeof(PuleAssetModelNode *))
      );
      for (uint32_t j = 0; j < node.childrenLen; ++ j) {
        node.children[j] = model.nodes + puleDsAsI64(childrenValue.values[j]);
      }
    }
    // transform
    node.transform = puleF32m44(1.0f);
    node.hasTranslate = node.hasRotate = node.hasScale = false;
    if (puleDsObjectMember(nodeValue, "matrix").id) {
      auto matrix = puleDsMemberAsArray(nodeValue, "matrix");
      for (uint32_t j = 0; j < 16; ++ j) {
        node.transform.elem[j] = puleDsAsF32(matrix.values[j]);
      }
    }
    if (puleDsObjectMember(nodeValue, "translation").id) {
      node.translate = (
        puleDsAsF32v3(puleDsObjectMember(nodeValue, "translation"))
      );
      node.hasTranslate = true;
    }
    if (puleDsObjectMember(nodeValue, "rotation").id) {
      auto r = (
        puleDsAsF32v4(puleDsObjectMember(nodeValue, "rotation"))
      );
      node.rotate = { r.x, r.y, r.z, r.w, };
      node.hasRotate = true;
    }
    if (puleDsObjectMember(nodeValue, "scale").id) {
      node.scale = (
        puleDsAsF32v3(puleDsObjectMember(nodeValue, "scale"))
      );
      node.hasScale = true;
    }
    // mesh
    node.mesh = nullptr;
    if (puleDsObjectMember(nodeValue, "mesh").id) {
      node.mesh = model.meshes + puleDsMemberAsI64(nodeValue, "mesh");
    }
    // TODO camera, light, skin, weights
  }

  // - load scene
  for (uint32_t i = 0; i < scenes.length; ++ i) {
    PuleDsValue sceneValue = scenes.values[i];
    PuleAssetModelScene & scene = model.scenes[i];
    scene.name = applyName(sceneValue, "scene", i);
    scene.index = i;
    auto nodesValue = puleDsMemberAsArray(sceneValue, "nodes");
    scene.nodeLen = nodesValue.length;
    scene.nodes = (
      (PuleAssetModelNode **)
      puleMalloc(allocator, scene.nodeLen * sizeof(PuleAssetModelNode *))
    );
    for (uint32_t j = 0; j < scene.nodeLen; ++ j) {
      scene.nodes[j] = model.nodes + puleDsAsI64(nodesValue.values[j]);
    }
  }

  // - load animation
  for (uint32_t i = 0; i < animations.length; ++ i) {
    PuleDsValue animationValue = animations.values[i];
    PuleAssetModelAnimation & animation = model.animations[i];
    memset(&animation, 0, sizeof(PuleAssetModelAnimation));
    animation.name = applyName(animationValue, "animation", i);
    animation.index = i;
    // - load channels
    auto channels = puleDsMemberAsArray(animationValue, "channels");
    animation.channelLen = channels.length;
    for (size_t chIt = 0; chIt < channels.length; ++ chIt) {
      PuleDsValue channelValue = channels.values[chIt];
      auto & channel = animation.channels[chIt];
      int64_t sampler = puleDsMemberAsI64(channelValue, "sampler");
      PULE_assert(sampler < 16 && "impl only supports 16 samplers");
      channel.sampler = (
        animation.samplers + puleDsMemberAsI64(channelValue, "sampler")
      );
      auto targetValue = puleDsObjectMember(channelValue, "target");
      channel.node = (
        model.nodes + puleDsMemberAsI64(targetValue, "node")
      );
      auto path = puleDsMemberAsString(targetValue, "path");
      if (puleStringViewEq(path, "translation"_psv)) {
        channel.target = PuleAssetModelAnimationTarget_translation;
      } else if (puleStringViewEq(path, "rotation"_psv)) {
        channel.target = PuleAssetModelAnimationTarget_rotation;
      } else if (puleStringViewEq(path, "scale"_psv)) {
        channel.target = PuleAssetModelAnimationTarget_scale;
      } else if (puleStringViewEq(path, "weights"_psv)) {
        channel.target = PuleAssetModelAnimationTarget_weights;
      } else {
        PULE_error(
          PuleErrorAssetModel_decode, "invalid path: '%s'", path.contents
        );
        return;
      }
    }
    // - load sampler
    auto samplers = puleDsMemberAsArray(animationValue, "samplers");
    animation.samplerLen = samplers.length;
    for (size_t saIt = 0; saIt < samplers.length; ++ saIt) {
      PuleDsValue samplerValue = samplers.values[saIt];
      auto & sampler = animation.samplers[saIt];
      sampler.timeline = (
        model.accessors + puleDsMemberAsI64(samplerValue, "input")
      );
      sampler.output = (
        model.accessors + puleDsMemberAsI64(samplerValue, "output")
      );
      auto interpolation = puleDsMemberAsString(samplerValue, "interpolation");
      if (puleStringViewEq(interpolation, "LINEAR"_psv)) {
        sampler.interpolation = PuleAssetModelAnimationInterpolation_linear;
      } else if (puleStringViewEq(interpolation, "STEP"_psv)) {
        sampler.interpolation = PuleAssetModelAnimationInterpolation_step;
      } else if (puleStringViewEq(interpolation, "CUBICSPLINE"_psv)) {
        sampler.interpolation = (
          PuleAssetModelAnimationInterpolation_cubicspline
        );
      } else {
        PULE_error(PuleErrorAssetModel_decode, "invalid interpolation");
        return;
      }
    }
  }

  // --- patch data ------------------------------------------------------------
  // TODO at some point this should be patched as part of data-conversion

  // for now missing accessors have dummy data, NOTE TODO this will change
  //   when proper pipeline support is in place that can accept missing data
  // (check the memory allocation at top of function to see why the indices
  //   are set to -2)
  if (hasMissingAttributes) {
    uint32_t dummyBufferIndex = (
      (model.bufferLen-1) - hasMissingElementIdxBuffers
    );
    uint32_t dummyBufferViewIndex = (
      model.bufferViewLen-1-hasMissingElementIdxBuffers
    );
    // note that in the accessor case need to offset by the accessors needed
    //   to be created by missing element-index buffers
    uint32_t dummyAccessorIndex = (
      model.accessorLen-1-missingElementIdxBuffers
    );
    for (size_t it = 0; it < model.meshLen; ++ it) {
      auto & mesh = model.meshes[it];
      for (size_t primIt = 0; primIt < mesh.primitiveLen; ++ primIt) {
        auto & primitive = mesh.primitives[primIt];
        auto & normal = (
          primitive.attributeAccessors[PuleAssetModelAttribute_normal]
        );
        if (normal == nullptr) {
          normal = model.accessors + dummyAccessorIndex;
          auto & bufferView = model.bufferViews[dummyBufferViewIndex];
          bufferView.byteLen = (
            std::max<uint32_t>(
              bufferView.byteLen,
              primitive.drawElementCount * 3 * sizeof(uint8_t)
            )
          );
        }
      }
    }

    // now create dummy data
    auto & dummyAccessor = model.accessors[dummyAccessorIndex];
    dummyAccessor = {
      .bufferView = model.bufferViews + dummyBufferViewIndex,
      .byteOffset = 0,
      .dataType = PuleGpuAttributeDataType_u8,
      .convertFixedToNormalized = true,
      .elementCount = 1, // this doesn't matter
      .name = puleStringCopy(allocator, "dummyAccessor"_psv),
      .index = dummyAccessorIndex,
    };

    auto & dummyBufferView = model.bufferViews[dummyBufferViewIndex];
    dummyBufferView = {
      .buffer = model.buffers + dummyBufferIndex,
      .byteOffset = 0,
      .byteLen = dummyBufferView.byteLen,
      .byteStride = sizeof(uint8_t),
      .usage = PuleAssetModelBufferViewUsage_attribute,
      .name = puleStringCopy(allocator, "dummyBufferView"_psv),
      .index = dummyBufferViewIndex,
    };

    auto & dummyBuffer = model.buffers[dummyBufferIndex];
    dummyBuffer = {
      .resourceUri = puleStringCopy(allocator, "dummyBuffer"_psv),
      .data = puleBufferCreate(allocator),
      .name = puleStringCopy(allocator, "dummyBuffer"_psv),
      .index = dummyBufferIndex,
    };
    puleBufferResize(&dummyBuffer.data, dummyBufferView.byteLen);
    for (size_t i = 0; i < dummyBufferView.byteLen; ++ i) {
      dummyBuffer.data.data[i] = 255;
    }
  }

  // now need to check if any primitives are missing an element-index buffer,
  //   in that case the data will need to be patched in
  size_t missingElementIdxIt = 0;
  std::vector<uint16_t> missingElementIdxData;
  uint32_t missingElementIdxByteOffset = 0;
  uint32_t missingElementIdxBufferViewIdx = model.bufferViewLen-1;
  uint32_t missingElementIdxBufferIdx = model.bufferLen-1;
  uint32_t missingElementIdxAccessorIdx = (
    model.accessorLen-missingElementIdxBuffers
  );
  for (size_t it = 0; it < model.meshLen; ++ it) {
    auto & mesh = model.meshes[it];
    for (size_t primIt = 0; primIt < mesh.primitiveLen; ++ primIt) {
      auto & primitive = mesh.primitives[primIt];
      if (primitive.elementIdxAccessor != nullptr) {
        continue;
      }
      // need to patch in the accessor
      uint32_t accessorIdx = missingElementIdxAccessorIdx+missingElementIdxIt;
      if (accessorIdx >= model.accessorLen) {
        puleLogError(
          "when patching accessor, idx %d >= accessor len %d",
          accessorIdx, model.accessorLen
        );
        exit(0);
      }
      primitive.elementIdxAccessor = model.accessors + accessorIdx;
      *primitive.elementIdxAccessor = {
        .bufferView = model.bufferViews + missingElementIdxBufferViewIdx,
        .byteOffset = missingElementIdxByteOffset,
        .dataType = PuleGpuAttributeDataType_u16,
        .convertFixedToNormalized = false,
        .elementType = PuleAssetModelElementType_scalar,
        .name = puleStringFormat(allocator, "missingElementIdx_%zu", it),
        .index = accessorIdx,
      };
      // the data is monotonic increasing indices
      for (size_t i = 0; i < primitive.drawElementCount; ++ i) {
        missingElementIdxData.push_back(i);
      }
      missingElementIdxByteOffset += (
        primitive.drawElementCount * sizeof(uint16_t)
      );
      ++ missingElementIdxIt;
    }
  }
  // if there was any found missing data, then create buffer & buffer view
  if (missingElementIdxData.size() > 0) {
    auto & bufferView = model.bufferViews[missingElementIdxBufferViewIdx];
    bufferView = {
      .buffer = model.buffers + missingElementIdxBufferIdx,
      .byteOffset = 0,
      .byteLen = missingElementIdxByteOffset,
      .byteStride = sizeof(uint16_t),
      .usage = PuleAssetModelBufferViewUsage_elementIdx,
      .name = puleStringCopy(allocator, "missingElementIdxBufferView"_psv),
      .index = missingElementIdxBufferViewIdx,
    };
    auto & buffer = model.buffers[missingElementIdxBufferIdx];
    buffer = {
      .resourceUri = puleStringCopy(allocator, "missingElementIdxBuffer"_psv),
      .data = puleBufferCreate(allocator),
      .name = puleStringCopy(allocator, "missingElementIdxBuffer"_psv),
      .index = missingElementIdxBufferIdx,
    };
    puleBufferResize(&buffer.data, missingElementIdxByteOffset);
    memcpy(
      buffer.data.data,
      missingElementIdxData.data(),
      missingElementIdxByteOffset
    );
  }

  // now check if any stride is missing from buffer-view, in that case
  //   find an accessor that uses it and set the stride
  bool hadToGuessStrides = false;
  for (size_t it = 0; it < meshes.length; ++ it) {
    auto & mesh = model.meshes[it];
    for (size_t primIt = 0; primIt < mesh.primitiveLen; ++ primIt) {
      auto & primitive = mesh.primitives[primIt];
      for (size_t attr = 0; attr < PuleAssetModelAttributeSize; ++ attr) {
        auto & accessor = primitive.attributeAccessors[attr];
        if (accessor == nullptr) { continue; }
        auto & bufferView = *primitive.attributeAccessors[attr]->bufferView;
        if (bufferView.byteStride != 0) { continue; }
        switch (accessor->elementType) {
          case PuleAssetModelElementType_scalar: bufferView.byteStride=1; break;
          case PuleAssetModelElementType_vec2:   bufferView.byteStride=2; break;
          case PuleAssetModelElementType_vec3:   bufferView.byteStride=3; break;
          case PuleAssetModelElementType_vec4:   bufferView.byteStride=4; break;
          case PuleAssetModelElementType_mat2:   bufferView.byteStride=4; break;
          case PuleAssetModelElementType_mat3:   bufferView.byteStride=9; break;
          case PuleAssetModelElementType_mat4:   bufferView.byteStride=16;break;
          default:
            puleLogError("invalid element type");
            exit(0);
        }
        // TODO implement puleGpuAttributeDataTypeSize
        switch (primitive.attributeAccessors[0]->dataType) {
          case PuleGpuAttributeDataType_u8:
            bufferView.byteStride *= sizeof(uint8_t);
          break;
          case PuleGpuAttributeDataType_u16:
            bufferView.byteStride *= sizeof(uint16_t);
          break;
          case PuleGpuAttributeDataType_f32:
            bufferView.byteStride *= sizeof(float);
          break;
          default:
            puleLogError("invalid data type");
            exit(0);
        }
        hadToGuessStrides = true;
        puleLogDev(
          "for mesh '%s' primitive %d attr %d, guessed stride %d : %d",
          mesh.name.contents, primIt, attr,
          bufferView.byteStride,
          model.meshes[it].primitives[primIt].attributeAccessors[attr]->bufferView->byteStride
        );
      }
    }
  }

  std::vector<PuleString> modelWarnings;
  if (hasMissingAttributes) {
    modelWarnings.push_back(puleString("missing attributes"));
  }
  if (hasMissingElementIdxBuffers) {
    modelWarnings.push_back(puleString("missing element-index buffers"));
  }
  if (hadToGuessStrides) {
    modelWarnings.push_back(puleString("missing buffer-view strides"));
  }
  if (modelWarnings.size() > 0) {
    model.loadWarnings = (
      (PuleString *)puleMalloc(
        allocator, modelWarnings.size() * sizeof(PuleString)
      )
    );
    memcpy(
      model.loadWarnings, modelWarnings.data(),
      modelWarnings.size() * sizeof(PuleString)
    );
    model.loadWarningLen = modelWarnings.size();
  }

  // dump everything out to stddv
  puleLogSectionBegin({.label="asset-model %s", .tabs=true}, basePath.c_str());
  puleLog("buffers: %d", model.bufferLen);
  for (size_t i = 0; i < model.bufferLen; ++ i) {
    auto & buffer = model.buffers[i];
    puleLog("  buffer %d: %s", i, buffer.name.contents);
  }
  puleLog("buffer-views: %d", model.bufferViewLen);
  for (size_t i = 0; i < model.bufferViewLen; ++ i) {
    auto & bufferView = model.bufferViews[i];
    puleLog(
      "  buffer-view %d: %s, buffer %d, offset %d, len %d, stride %d",
      i, bufferView.name.contents, bufferView.buffer->index,
      bufferView.byteOffset, bufferView.byteLen, bufferView.byteStride
    );
  }
  puleLog("materials: %d", model.materialLen);
  puleLogSectionEnd();
}

static void validateModel(
  PuleAssetModel & model,
  PuleError * error
) {
  // these are all pretty redundant checks just to make sure the engine is
  //   working correctly
  // Majority of errors here shouldn't be produced from user error

  #define CHECK_NONNIL(name) \
    PULE_assert(model. name != nullptr && #name " is null")
  CHECK_NONNIL(accessors);
  CHECK_NONNIL(bufferViews);
  CHECK_NONNIL(buffers);
  CHECK_NONNIL(materials);
  CHECK_NONNIL(nodes);
  CHECK_NONNIL(scenes);
  CHECK_NONNIL(textures);
  CHECK_NONNIL(images);
  CHECK_NONNIL(meshes);
  #undef CHECK_NONNIL

  // check canaries for OOB detection
  auto checkCanary = [](void * data, size_t byteLen) {
    uint32_t * canary = (uint32_t *)((uint8_t *)data + byteLen);
    return *canary == 0xdeadbeef;
  };

  #define CHECK_CANARY(name, len) \
    if (checkCanary(model. name, model. len * sizeof(*model. name))) { \
      puleLogError(#name " canary is invalid"); \
      exit(0); \
    }
  #define CHECK_CANARY_SINGLE(name) CHECK_CANARY(name ## s, name ## Len)

  CHECK_CANARY_SINGLE(accessor);
  CHECK_CANARY_SINGLE(bufferView);
  CHECK_CANARY_SINGLE(buffer);
  CHECK_CANARY_SINGLE(material);
  CHECK_CANARY_SINGLE(node);
  CHECK_CANARY_SINGLE(scene);
  CHECK_CANARY_SINGLE(texture);
  CHECK_CANARY_SINGLE(image);
  CHECK_CANARY(meshes, meshLen);
  #undef CHECK_CANARY

  // -- check pointers inside 'objects' are null
  for (size_t i = 0; i < model.meshLen; ++ i) {
    auto & mesh = model.meshes[i];
    PULE_assert(mesh.primitives != nullptr && "mesh primitives is null");
  }

  for (size_t i = 0; i < model.sceneLen; ++ i) {
    auto & scene = model.scenes[i];
    PULE_assert(scene.nodes != nullptr && "scene nodes is null");
  }

  for (size_t i = 0; i < model.accessorLen; ++ i) {
    auto & accessor = model.accessors[i];
    PULE_assert(
      accessor.bufferView != nullptr && "accessor bufferView is null"
    );
  }

  for (size_t i = 0; i < model.bufferViewLen; ++ i) {
    auto & bufferView = model.bufferViews[i];
    PULE_assert(
      bufferView.buffer != nullptr && "bufferView buffer is null"
    );
  }

  // -- check all buffers have an actual size and data
  for (size_t i = 0; i < model.bufferLen; ++ i) {
    auto & buffer = model.buffers[i];
    PULE_assert(buffer.data.data != nullptr && "buffer data is null");
    if (buffer.data.byteLength == 0) {
      puleLogError("buffer %s has 0 byte length", buffer.name.contents);
      exit(0);
    }
  }

  // -- all index buffers must not have a stride
  //    they must also be u16 or u32
  for (size_t it = 0; it < model.meshLen; ++ it) {
    auto & mesh = model.meshes[it];
    for (size_t primIt = 0; primIt < mesh.primitiveLen; ++ primIt) {
      auto & primitive = mesh.primitives[primIt];
      if (primitive.elementIdxAccessor == nullptr) { continue; }
      PULE_assert(
        primitive.elementIdxAccessor->bufferView->byteStride == 0
        && "element index buffer must not have a stride"
      );
      auto dt = primitive.elementIdxAccessor->dataType;
      if (
           dt != PuleGpuAttributeDataType_u16
        && dt != PuleGpuAttributeDataType_u32
        && dt != PuleGpuAttributeDataType_u8
      ) {
        puleLogError(
          "for mesh %s prim %s,"
          " element index buffer must be u16 or u32, not %s",
          mesh.name.contents, primitive.material->name.contents,
          pule::toStr(primitive.elementIdxAccessor->dataType).data.contents
        );
        exit(0);
      }
    }
  }

  // -- all accessor buffer views must have a stride
  for (size_t it = 0; it < model.meshLen; ++ it) {
    auto & mesh = model.meshes[it];
    for (size_t primIt = 0; primIt < mesh.primitiveLen; ++ primIt) {
      auto & primitive = mesh.primitives[primIt];
      for (size_t attr = 0; attr < PuleAssetModelAttributeSize; ++ attr) {
        auto & accessor = primitive.attributeAccessors[it];
        if (accessor == nullptr) { continue; }
        if (accessor->bufferView->byteStride == 0) {
          puleLogError(
            (
              "in mesh '%s' (%d), primitive '%s' (%d), attribute %d, "
              "buffer-view '%s' (%d), stride must be non-zero"
            ),
            mesh.name.contents, mesh.index,
            primitive.material->name.contents, primitive.material->index,
            attr, accessor->bufferView->name.contents,
            accessor->bufferView->index
          );
          exit(0);
        }
      }
    }
  }
}

extern "C" {
PuleAssetModel puleAssetModelLoad(PuleAssetModelLoadInfo info) {
  auto & error = info.error;
  PuleAssetModel model = {};
  if (info.modelPath.contents == nullptr) {
    PULE_error(PuleErrorAssetModel_decode, "model path is null");
    return model;
  }
  model.allocator = info.allocator;
  PuleDsValue modelValue = info.modelSource;
  assetModelLoadScene(
    model,
    std::string(info.modelPath.contents),
    info.loadBuffers, info.loadImages,
    modelValue,
    error
  );
  if (puleErrorExists(error)) {
    memset(&model, 0, sizeof(model));
    return model;
  }
  validateModel(model, error);
  if (puleErrorExists(error)) {
    memset(&model, 0, sizeof(model));
    return model;
  }
  return model;
}
} // extern C
