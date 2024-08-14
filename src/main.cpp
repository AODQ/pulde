#include <pulchritude/asset-image.hpp>
#include <pulchritude/asset-model.hpp>
#include <pulchritude/asset-pds.hpp>
#include <pulchritude/camera.hpp>
#include <pulchritude/core.hpp>
#include <pulchritude/data-serializer.hpp>
#include <pulchritude/error.hpp>
#include <pulchritude/file.hpp>
#include <pulchritude/gpu.hpp>
#include <pulchritude/gpu-ir.hpp>
#include <pulchritude/log.hpp>
#include <pulchritude/plugin.hpp>
#include <pulchritude/render-graph.hpp>

#include <algorithm>
#include <vector>

// TODO replace this with gui selection etc
#define MDLMIX(a) puleCStr("assets/converted/" a "/" a ".pds")
PuleStringView MODEL = MDLMIX(
  //"Box"
  //"Triangle"
  //"TriangleWithoutIndices"
  //"BoxInterleaved"
  //"BoxTextured"
  //"BoxTexturedNonPowerOfTwo"
  //"OrientationTest"
  //"TextureCoordinateTest" // works, but crashes if you look behind it
  //"SimpleMeshes"
  //"AnimatedTriangle"
  //"Cube"
  //"AnimatedCube"
  //"BoxVertexColors"
  //"Duck"
  //"BoxAnimated" // does not work
  // "ABeautifulGame"
  // "InterpolationTest" // needs cubic spline
  //"NormalTangentMirrorTest" // fails
  // "NegativeScaleTest" // i think it might not pass?
  //"SciFiHelmet"
  //"CesiumMilkTruck"
  //"ToyCar"

  //"Sponza"
  //"Lantern"
  
  "SimpleMorph"
);

static PuleAssetModel loadModel(PuleStringView path) {
  // get the directory
  std::string directory = std::string(path.contents, path.len);
  directory = directory.substr(0, directory.find_last_of("/\\"));
  PuleError err = puleError();
  PuleDsValue modelValue = puleAssetPdsLoadFromFile(
    puleAllocateDefault(), path, &err
  );
  if (puleErrorConsume(&err)) { return {}; }

  PuleAssetModel model = (
    puleAssetModelLoad({
      .allocator = puleAllocateDefault(),
      .modelPath = puleCStr(directory.c_str()),
      .modelSource = modelValue,
      .loadBuffers = true,
      .loadImages = true,
      .error = &err
    })
  );
  if (puleErrorConsume(&err)) { return {}; }
  return model;
}

struct PuleModel3DDraw {
  PuleGpuBuffer elementIdxBuffer;
  size_t elementIdxBufferOffset;
  PuleGpuElementType elementIdxType;
  PuleGpuBuffer buffers[PuleAssetModelAttributeSize];
  // this is needed to distinguish buffer views from 'accessors'
  // note that it is illegal in Vk to have a stride smaller than the attribute's
  //   offset, so the 'buffer' offset must be applied during binding
  // Thus Vk enforces at API level:
  //   %p = buffer + bufferOffset + attributeOffset + stride*vertexIndex
  // To simplify this, all bindings of the buffers are done separately
  //   (attributes bind to unique buffer-slots)
  size_t bufferOffsets[PuleAssetModelAttributeSize];
  size_t strides[PuleAssetModelAttributeSize];
  PuleF32m44 transform;
  PuleGpuPipeline pipeline;
  size_t numElements;
  size_t elementOffset;
  size_t baseVertexOffset;
  int64_t colorTextureIndex = -1;

  bool hasBaseColor = false;
  PuleF32v4 baseColor;

  // animation data

};

// animation components
namespace anim {

struct Output {
  PuleAssetModelAccessor * accessor;
  PuleAssetModelAnimationTarget target;
  std::vector<float> keyframeData;
  std::vector<float> keyframeTimeline;

  PuleF32v3 translation(size_t idx) {
    float * flData = reinterpret_cast<float *>(&keyframeData[idx*3]);
    return PuleF32v3 { flData[0], flData[1], flData[2], };
  }

  PuleF32v3 scale(size_t idx) {
    return translation(idx);
  }

  PuleF32q rotation(size_t idx) {
    float * flData = reinterpret_cast<float *>(&keyframeData[idx*4]);
    return PuleF32q { flData[0], flData[1], flData[2], flData[3], };
  }

  void weights(size_t idx, float * weights, size_t weightLen) {
    float * flData = reinterpret_cast<float *>(keyframeData.data());
    flData += idx*weightLen;
    memcpy(weights, flData, weightLen*sizeof(float));
  }

  static Output create(
    PuleAssetModelAccessor * accessor,
    PuleAssetModelAnimationTarget target,
    PuleAssetModelAccessor * timeline
  ) {
    Output inp;
    inp.accessor = accessor;
    inp.target = target;
    // need to compute stride
    size_t stride;
    switch (target) {
      case PuleAssetModelAnimationTarget_scale: stride = 3; break;
      case PuleAssetModelAnimationTarget_translation: stride = 3; break;
      case PuleAssetModelAnimationTarget_rotation: stride = 4; break;
      case PuleAssetModelAnimationTarget_weights: stride = 1; break;
      default: PULE_assert(false && "unimplemented animation target");
    }
    switch (accessor->dataType) {
      case PuleGpuAttributeDataType_u8: stride *= 1; break;
      case PuleGpuAttributeDataType_u16: stride *= 2; break;
      case PuleGpuAttributeDataType_u32: stride *= 4; break;
      case PuleGpuAttributeDataType_f32: stride *= 4; break;
      default: PULE_assert(false && "unimplemented data type");
    }
    PULE_assert(
         accessor->bufferView->byteStride == 0
      || accessor->bufferView->byteStride == stride
    );

    auto data = [&](size_t idx) {
      return (
          reinterpret_cast<uint8_t *>(accessor->bufferView->buffer->data.data)
        + accessor->bufferView->byteOffset
        + accessor->byteOffset
        + idx*stride
      );
    };

    // copy outputs
    for (size_t it = 0; it < accessor->elementCount; ++ it) {
      switch (inp.target) {
        case PuleAssetModelAnimationTarget_scale: {
          float * flData = reinterpret_cast<float *>(data(it));
          inp.keyframeData.emplace_back(flData[0]);
          inp.keyframeData.emplace_back(flData[1]);
          inp.keyframeData.emplace_back(flData[2]);
        } break;
        case PuleAssetModelAnimationTarget_translation: {
          float * flData = reinterpret_cast<float *>(data(it));
          inp.keyframeData.emplace_back(flData[0]);
          inp.keyframeData.emplace_back(flData[1]);
          inp.keyframeData.emplace_back(flData[2]);
        } break;
        case PuleAssetModelAnimationTarget_rotation: {
          void * byteData = data(it);
          switch (accessor->dataType) {
            case PuleGpuAttributeDataType_f32: {
              float * flData = reinterpret_cast<float *>(byteData);
              inp.keyframeData.emplace_back(flData[0]);
              inp.keyframeData.emplace_back(flData[1]);
              inp.keyframeData.emplace_back(flData[2]);
              inp.keyframeData.emplace_back(flData[3]);
            } break;
            case PuleGpuAttributeDataType_u8: {
              uint8_t * u8Data = reinterpret_cast<uint8_t *>(byteData);
              inp.keyframeData.emplace_back(u8Data[0]/255.0f);
              inp.keyframeData.emplace_back(u8Data[1]/255.0f);
              inp.keyframeData.emplace_back(u8Data[2]/255.0f);
              inp.keyframeData.emplace_back(u8Data[3]/255.0f);
            } break;
            default: PULE_assert(false && "unimplemented data type");
          }
        } break;
        case PuleAssetModelAnimationTarget_weights: {
          float * flData = reinterpret_cast<float *>(data(it));
          inp.keyframeData.emplace_back(flData[0]);
        } break;
        default: PULE_assert(false && "unimplemented animation target");
      }
    }

    // copy timeline
    float * keyframeTimeline = (
      reinterpret_cast<float *>(
          timeline->bufferView->buffer->data.data
        + timeline->bufferView->byteOffset
        + timeline->byteOffset
      )
    );
    for (size_t it = 0; it < timeline->elementCount; ++ it) {
      inp.keyframeTimeline.emplace_back(keyframeTimeline[it]);
    }

    // insert dummy keyframe
    PULE_assert(inp.keyframeTimeline.size() > 0);
    if (inp.keyframeTimeline[0] > 0.0f) {
      inp.keyframeTimeline.insert(inp.keyframeTimeline.begin(), 0.0f);
      switch (inp.target) {
        case PuleAssetModelAnimationTarget_scale:
        case PuleAssetModelAnimationTarget_translation: {
          float cpdata[3] = {
            inp.keyframeData[0], inp.keyframeData[1], inp.keyframeData[2]
          };
          inp.keyframeData.insert(inp.keyframeData.begin(), cpdata, cpdata+3);
        } break;
        case PuleAssetModelAnimationTarget_rotation: {
          float cpdata[4] = {
            inp.keyframeData[0], inp.keyframeData[1],
            inp.keyframeData[2], inp.keyframeData[3]
          };
          inp.keyframeData.insert(inp.keyframeData.begin(), cpdata, cpdata+4);
        } break;
        case PuleAssetModelAnimationTarget_weights: {
          float cpdata[1] = { inp.keyframeData[0] };
          inp.keyframeData.insert(inp.keyframeData.begin(), cpdata, cpdata+1);
        } break;
        default: PULE_assert(false && "unimplemented animation target");
      }
    }

    return inp;
  }
};

void sample(
  PuleAssetModelAnimationChannel & channel,
  float msTime
) {
  auto out = ( // somehow cache this (it allocates memory)
    Output::create(
      channel.sampler->output, channel.target, channel.sampler->timeline
    )
  );

  // find the first keyframe index whose keyframe is greater than msDelta
  int64_t kfIdx = 0;
  for (size_t i = 0; i < out.keyframeTimeline.size(); ++ i) {
    if (out.keyframeTimeline[i] > msTime) {
      kfIdx = i-1;
      break;
    }
  }
  PULE_assert(kfIdx >= 0); // this shouldnt trigger since 0 is 0.0f timeline

  switch (channel.target) {
    case PuleAssetModelAnimationTarget_scale:
      channel.node->hasScale = true;
    break;
    case PuleAssetModelAnimationTarget_translation:
      channel.node->hasTranslate = true;
    break;
    case PuleAssetModelAnimationTarget_rotation:
      channel.node->hasRotate = true;
    break;
    case PuleAssetModelAnimationTarget_weights: // nothing to do
    break;
  }

  switch (channel.sampler->interpolation) {
    default: PULE_assert(false && "unimplemented animation interpolation");
    case PuleAssetModelAnimationInterpolation_step:
      switch (channel.target) {
        default: PULE_assert(false && "unimplemented animation target");
        case PuleAssetModelAnimationTarget_scale:
          channel.node->scale = out.scale(kfIdx);
        break;
        case PuleAssetModelAnimationTarget_translation:
          channel.node->translate = out.translation(kfIdx);
        break;
        case PuleAssetModelAnimationTarget_rotation:
          channel.node->rotate = out.rotation(kfIdx);
        break;
        case PuleAssetModelAnimationTarget_weights:
          out.weights(
            kfIdx, channel.node->mesh->weights, channel.node->mesh->weightLen
          );
        break;
      }
    break;
    case PuleAssetModelAnimationInterpolation_linear: {
      int64_t kfIdxNext = (
        std::min(kfIdx + 1, (int64_t)out.keyframeTimeline.size()-1)
      );
      if (kfIdxNext == kfIdx) break; // hit end
      float t = (
          (msTime - out.keyframeTimeline[kfIdx])
        / (out.keyframeTimeline[kfIdxNext] - out.keyframeTimeline[kfIdx])
      );
      t = std::clamp(t, 0.0f, 1.0f);
      switch (channel.target) {
        default: PULE_assert(false && "unimplemented animation target");
        case PuleAssetModelAnimationTarget_scale:
          channel.node->scale = (
            puleF32v3Mix(out.scale(kfIdx), out.scale(kfIdxNext), t)
          );
        break;
        case PuleAssetModelAnimationTarget_translation:
          channel.node->translate = (
            puleF32v3Mix(out.translation(kfIdx), out.translation(kfIdxNext), t)
          );
        break;
        case PuleAssetModelAnimationTarget_rotation:
          channel.node->rotate = (
            puleF32qSlerp(out.rotation(kfIdx), out.rotation(kfIdxNext), t)
          );
        break;
        case PuleAssetModelAnimationTarget_weights:
          std::vector<float> weights(channel.node->mesh->weightLen);
          out.weights(kfIdx, weights.data(), weights.size());
          std::vector<float> weightsNext(channel.node->mesh->weightLen);
          out.weights(kfIdxNext, weightsNext.data(), weightsNext.size());
          for (size_t i = 0; i < weights.size(); ++ i) {
            channel.node->mesh->weights[i] = (
              puleF32Mix(weights[i], weightsNext[i], t)
            );
          }
        break;
      }
    } break;
    case PuleAssetModelAnimationInterpolation_cubicspline:
      PULE_assert(false && "unimplemented cubic spline interpolation");
  }
}

} // namespace anim


struct PuleModel3D {
  std::vector<PuleModel3DDraw> draws;
  std::vector<PuleGpuBuffer> buffers;
  std::vector<PuleGpuImage> images;
  std::vector<PuleGpuImageView> imageViews;
  std::unordered_map<size_t, std::vector<size_t>> nodeToDraws;
  std::unordered_map<size_t, anim::Output> animationChannelToOutput;
  std::unordered_map<size_t, std::unordered_map<size_t, size_t>>
    meshToPrimitiveToDraw;
  PuleAssetModel asset;
};

void puleModel3DRender(
  PuleModel3D const & model3d,
  PuleGpuCommandListRecorder commandRecorder,
  PuleCamera camera
) {
  auto proj = puleCameraProj(camera);
  auto view = puleCameraView(camera);
  auto projView = puleF32m44Mul(view, proj);
  for (auto & draw : model3d.draws) {
    // bind pipeline
    puleGpuCommandListAppendAction(commandRecorder, PuleGpuCommand {
      .bindPipeline = {
        .action = PuleGpuAction_bindPipeline,
        .pipeline = draw.pipeline,
      },
    });

    // bind element buffer
    puleGpuCommandListAppendAction(commandRecorder, PuleGpuCommand {
      .bindElementBuffer = {
        .buffer = draw.elementIdxBuffer,
        .offset = draw.elementIdxBufferOffset,
        .elementType = draw.elementIdxType,
      },
    });

    // bind attribute buffers
    for (size_t i = 0; i < PuleAssetModelAttributeSize; ++ i) {
      auto & buffer = draw.buffers[i];
      if (buffer.id == 0) { continue; }
      puleGpuCommandListAppendAction(commandRecorder, PuleGpuCommand {
        .bindAttributeBuffer = {
          .bindingIndex = i,
          .buffer = buffer,
          .offset = draw.bufferOffsets[i],
          .stride = draw.strides[i],
        },
      });
    }

    // bind push constants
    struct __attribute__((__packed__)) data {
      PuleF32m44 projView;
      PuleF32m44 transform;
      PuleF32v4 baseColor;
    } data = {
      .projView = projView, .transform = draw.transform,
      .baseColor = draw.baseColor
    };
    puleGpuCommandListAppendAction(commandRecorder, PuleGpuCommand {
      .pushConstants = {
        .stage = PuleGpuDescriptorStage_vertex,
        .byteLength = (
          draw.hasBaseColor ? sizeof(data) : offsetof(struct data, baseColor)
        ),
        .byteOffset = 0,
        .data = &data,
      },
    });

    // bind texture
    if (draw.colorTextureIndex != -1) {
      puleGpuCommandListAppendAction(commandRecorder, PuleGpuCommand {
        .bindTexture = {
          .bindingIndex = 0,
          .imageView = model3d.imageViews[draw.colorTextureIndex],
          .imageLayout = PuleGpuImageLayout_storage,
        },
      });
    }

    // dispatch render elements
    puleGpuCommandListAppendAction(commandRecorder, {
      .dispatchRenderElements = {
        .numElements = draw.numElements,
        .elementOffset = draw.elementOffset,
        .baseVertexOffset = draw.baseVertexOffset,
      }
    });
  }
}

static void prepareModelNode(
  PuleAssetModel model,
  PuleAssetModelNode * node,
  PuleModel3D & model3d,
  PuleF32m44 const & oldTransform
) {
  // apply transforms
  auto transform = puleF32m44Mul(oldTransform, node->transform);

  // create the generic shader
  PuleError err = puleError();
  PuleBuffer moduleVertex = (
    puleFileDumpContents(
      puleAllocateDefault(),
      "assets/shaders/pbr.vert.spv"_psv,
      PuleFileDataMode_binary,
      &err
    )
  );
  moduleVertex.byteLength -= 1; // remove null terminator
  if (puleErrorConsume(&err)) { return; }
  puleScopeExit { puleBufferDestroy(moduleVertex); };
  PuleBuffer moduleFragment = (
    puleFileDumpContents(
      puleAllocateDefault(),
      "assets/shaders/pbr.frag.spv"_psv,
      PuleFileDataMode_binary,
      &err
    )
  );
  moduleFragment.byteLength -= 1; // remove null terminator
  if (puleErrorConsume(&err)) { return; }
  puleScopeExit { puleBufferDestroy(moduleFragment); };

  PuleGpuShaderModule shaderModule = (
    puleGpuShaderModuleCreate(
      puleBufferView(moduleVertex),
      puleBufferView(moduleFragment),
      &err
    )
  );
  if (puleErrorConsume(&err)) { return; }

  // apply draws
  if (node->mesh != nullptr) {
    auto & mesh = *node->mesh;
    for (size_t i = 0; i < mesh.primitiveLen; ++ i) {
      auto & primitive = mesh.primitives[i];
      auto & draw = model3d.draws.emplace_back();
      // keep track of node->draws to update transforms
      model3d.nodeToDraws[node->index].push_back(model3d.draws.size()-1);
      // keep track of mesh->node->draw to update morph targets
      model3d.meshToPrimitiveToDraw[mesh.index][i] = model3d.draws.size()-1;
      // fill out draw data
      draw.transform = transform;
      draw.baseVertexOffset = 0;
      draw.elementOffset = 0;
      draw.numElements = primitive.drawElementCount;
      { // create elementIdx buffer (it must exist)
        auto & elementIdxBufferView = primitive.elementIdxAccessor->bufferView;
        PULE_assert(elementIdxBufferView->buffer->data.data != nullptr);
        auto & accessor = primitive.elementIdxAccessor;
        auto & bufferView = elementIdxBufferView;
        auto & buffer = bufferView->buffer;
        draw.elementIdxBuffer = model3d.buffers[buffer->index];
        draw.elementIdxBufferOffset = (
          bufferView->byteOffset + accessor->byteOffset
        );
        // get element type
        switch (accessor->dataType) {
          default: PULE_assert(false && "Invalid element data type");
          case PuleGpuAttributeDataType_u8:
            draw.elementIdxType = PuleGpuElementType_u8; break;

          case PuleGpuAttributeDataType_u16:
            draw.elementIdxType = PuleGpuElementType_u16; break;
          case PuleGpuAttributeDataType_u32:
            draw.elementIdxType = PuleGpuElementType_u32; break;
        }
      }
      // find attribute buffers
      for (size_t j = 0; j < PuleAssetModelAttributeSize; ++ j) {
        auto & accessor = primitive.attributeAccessors[j];
        if (!accessor) {
          continue;
        }
        PULE_assert(accessor->bufferView != nullptr);
        PULE_assert(accessor->bufferView->buffer != nullptr);
        PULE_assert(accessor->bufferView->buffer->data.data != nullptr);
        auto & bufferView = accessor->bufferView;
        auto & buffer = bufferView->buffer;
        draw.buffers[j] = model3d.buffers[buffer->index];
        draw.strides[j] = bufferView->byteStride;
        draw.bufferOffsets[j] = bufferView->byteOffset + accessor->byteOffset;
      }
      // create pipeline
      PuleGpuPipelineCreateInfo pipelineInfo = {};
      pipelineInfo.config = PuleGpuPipelineConfig {
        .depthTestEnabled = true,
        .blendEnabled = false,
        .scissorTestEnabled = false,
        .viewportMin = {0, 0},
        .viewportMax = {800, 600}, // TODO: get from window
        .scissorMin = {0, 0},
        .scissorMax = {800, 600}, // TODO: get from window
        .drawPrimitive = primitive.topology,
        .colorAttachmentCount = 1,
        .colorAttachmentFormats = {PuleGpuImageByteFormat_rgba8U},
        .depthAttachmentFormat = PuleGpuImageByteFormat_depth16,
      };
      pipelineInfo.layoutDescriptorSet = {};
      memset(
        pipelineInfo.layoutDescriptorSet.attributeBindings,
        0,
        sizeof(pipelineInfo.layoutDescriptorSet.attributeBindings)
      );
      // -- attribute descriptors
      for (size_t k = 0; k < PuleAssetModelAttributeSize; ++ k) {
        auto & attr = primitive.attributeAccessors[k];
        if (attr == nullptr) { continue; }
        pipelineInfo.layoutDescriptorSet.attributeBindings[k] = {
          .dataType = attr->dataType,
          .bufferIndex = k,
          .numComponents = 0, // done below
          .convertFixedDataTypeToNormalizedFloating = (
            attr->convertFixedToNormalized
          ),
          .relativeOffset = 0,
        };
        auto & numComp = (
          pipelineInfo.layoutDescriptorSet.attributeBindings[k].numComponents
        );
        switch (attr->elementType) {
          case PuleAssetModelElementType_scalar: numComp = 1; break;
          case PuleAssetModelElementType_vec2: numComp = 2; break;
          case PuleAssetModelElementType_vec3: numComp = 3; break;
          case PuleAssetModelElementType_vec4: numComp = 4; break;
          case PuleAssetModelElementType_mat2: numComp = 4; break;
          case PuleAssetModelElementType_mat3: numComp = 9; break;
          case PuleAssetModelElementType_mat4: numComp = 16; break;
        }
        pipelineInfo.layoutDescriptorSet.attributeBufferBindings[k] = {
          .stridePerElement = draw.strides[k],
        };
      }

      // -- textures
      if (primitive.material != nullptr) {
        auto & material = *primitive.material;
        if (material.pbrMetallicRoughness.baseColorTexture != nullptr) {
          pipelineInfo.layoutDescriptorSet.textureBindings[0] = {
            PuleGpuDescriptorStage_fragment,
          };
        }
      }

      // -- material push constants
      draw.hasBaseColor = (
           primitive.material
        && primitive.material->pbrMetallicRoughness.baseColorTexture == nullptr
      );
      if (draw.hasBaseColor) {
        memcpy(
          &draw.baseColor,
          &primitive.material->pbrMetallicRoughness.baseColorFactor,
          sizeof(draw.baseColor)
        );
      }

      // -- push constants
      pipelineInfo.shaderModule = shaderModule;
      pipelineInfo.layoutPushConstants = {
        PuleGpuPipelineLayoutPushConstant  {
          .stage = PuleGpuDescriptorStage_vertex,
          .byteLength = sizeof(PuleF32m44)*2,
          .byteOffset = 0,
        }
      };
      if (draw.hasBaseColor) {
        pipelineInfo.layoutPushConstants.byteLength += sizeof(PuleF32v4);
      }
      draw.pipeline = puleGpuPipelineCreate(pipelineInfo, &err);
      if (puleErrorConsume(&err)) { return; }
      if (primitive.material != nullptr) {
        auto & material = *primitive.material;
        if (material.pbrMetallicRoughness.baseColorTexture != nullptr) {
          auto & texture = *material.pbrMetallicRoughness.baseColorTexture;
          draw.colorTextureIndex = texture.index;
        }
      }
    }
  }

  // apply children
  for (size_t i = 0; i < node->childrenLen; ++ i) {
    auto * child = node->children[i];
    prepareModelNode(model, child, model3d, transform);
  }
}

static PuleModel3D prepareModel(PuleAssetModel model) {
  PuleModel3D model3d;

  model3d.asset = model;

  // create buffers
  for (size_t i = 0; i < model.bufferLen; ++ i) {
    auto & buffer = model.buffers[i];
    model3d.buffers.push_back(
      puleGpuBufferCreate(
        puleStringView(buffer.name),
        buffer.data.byteLength,
        static_cast<PuleGpuBufferUsage>(
          PuleGpuBufferUsage_element | PuleGpuBufferUsage_attribute
        ),
        PuleGpuBufferVisibilityFlag_deviceOnly
      )
    );
    puleGpuBufferMemcpy(
      {
        .buffer = model3d.buffers.back(),
        .byteOffset = 0,
        .byteLength = buffer.data.byteLength,
      },
      buffer.data.data
    );
  }

  // create images
  for (size_t i = 0; i < model.imageLen; ++ i) {
    auto & image = model.images[i];
    puleLogDev("image: %zu", i);
    puleLogDev("image: %s", image.name.contents);
    auto sampler = (
      puleGpuSamplerCreate({
        .minify = PuleGpuImageMagnification_nearest,
        .magnify = PuleGpuImageMagnification_nearest,
        .wrapU = PuleGpuImageWrap_clampToEdge,
        .wrapV = PuleGpuImageWrap_clampToEdge,
        .wrapW = PuleGpuImageWrap_clampToEdge,
      })
    );
    model3d.images.push_back(
      puleGpuImageCreate({
        .width = puleAssetImageWidth(image.image),
        .height = puleAssetImageHeight(image.image),
        .target = PuleGpuImageTarget_i2D,
        .byteFormat = PuleGpuImageByteFormat_rgba8U,
        .sampler = sampler,
        .label = puleStringView(image.name),
        .optionalInitialData = puleAssetImageDecodedData(image.image),
      })
    );
    model3d.imageViews.push_back(
      PuleGpuImageView {
        .image = model3d.images.back(),
        .mipmapLevelStart = 0,
        .mipmapLevelCount = 1,
        .arrayLayerStart = 0,
        .arrayLayerCount = 1,
        .byteFormat = PuleGpuImageByteFormat_rgba8U,
      }
    );
  }

  // prepare nodes/meshes
  for (size_t i = 0; i < model.scenes[0].nodeLen; ++ i) {
    auto * node = model.scenes[0].nodes[i];
    prepareModelNode(model, node, model3d, puleF32m44(1.0f));
  }
  return model3d;
}

static void puleAssetModelNodeDump(
  PuleAssetModel model, PuleAssetModelNode node
) {
  puleLogRaw("Node %s\n", node.name.contents);
  puleLogRaw("  Transform\n");
  for (size_t i = 0; i < 4; ++ i) {
    puleLogRaw(
      "    %f %f %f %f\n",
      (double)node.transform.elem[i*4+0],
      (double)node.transform.elem[i*4+1],
      (double)node.transform.elem[i*4+2],
      (double)node.transform.elem[i*4+3]
    );
  }
  if (node.hasTranslate) {
    puleLogRaw(
      "  translate: <%f, %f, %f>",
      node.translate.x, node.translate.y, node.translate.z
    );
  }
  if (node.hasRotate) {
    puleLogRaw(
      "  rotate: <%f, %f, %f, %f>",
      node.rotate.x, node.rotate.y, node.rotate.z, node.rotate.w
    );
  }
  if (node.hasScale) {
    puleLogRaw(
      "  scale: <%f, %f, %f>", node.scale.x, node.scale.y, node.scale.z
    );
  }
  if (node.mesh != nullptr) {
    puleLogRaw("  Mesh %s\n", node.mesh->name.contents);
    for (size_t i = 0; i < node.mesh->primitiveLen; ++ i) {
      auto & primitive = node.mesh->primitives[i];
      puleLogRaw("    Primitive %zu\n", i);
      puleLogRaw(
        "      Element count %zu\n",
        primitive.drawElementCount
      );
      puleLogRaw("      Topology %d\n", primitive.topology);
      puleLogRaw(
        "      index buffer %d\n", primitive.elementIdxAccessor->name.contents
      );
      for (size_t j = 0; j < PuleAssetModelAttributeSize; ++ j) {
        auto & accessor = primitive.attributeAccessors[j];
        if (!accessor || accessor->elementCount == 0) { continue; }
        puleLogRaw("      Attribute %zu\n", j);
        puleLogRaw("        Element count %zu\n", accessor->elementCount);
        puleLogRaw("        Element type %d\n", accessor->elementType);
        puleLogRaw("        Data type %d\n", accessor->dataType);
        puleLogRaw("        Byte offset %zu\n", accessor->byteOffset);
        auto & bufferView = accessor->bufferView;
        puleLogRaw(
          "        Buffer view '%s'\n"
          "          buffer name '%s'\n"
          "          resource uri '%s'\n"
          "          byte length %zu\n"
          "          byte offset %zu\n"
          "          byte stride %zu\n"
          "          usage: %s\n"
          "          buffer pointer %p\n"
          ,
          bufferView->name.contents,
          bufferView->buffer->name.contents,
          bufferView->buffer->resourceUri.contents,
          bufferView->byteLen,
          bufferView->byteOffset,
          bufferView->byteStride,
          pule::toStr(bufferView->usage).data.contents,
          bufferView->buffer->data.data
        );
      }
    }
  }
  for (size_t i = 0; i < node.childrenLen; ++ i) {
    puleAssetModelNodeDump(model, *node.children[i]);
  }
}

static void puleAssetModelDump(PuleAssetModel model) {
  puleLogDev("Model has %zu scenes", model.sceneLen);
  for (size_t i = 0; i < model.sceneLen; ++ i) {
    auto & scene = model.scenes[i];
    puleLogDev("Scene %zu has %zu nodes", i, scene.nodeLen);
    for (size_t j = 0; j < scene.nodeLen; ++ j) {
      auto & node = scene.nodes[j];
      puleLogDev("Node %zu has %zu children", j, node->childrenLen);
      puleAssetModelNodeDump(model, *node);
    }
  }
  puleLogDev("Model has %zu buffers", model.bufferLen);
  for (size_t i = 0; i < model.bufferLen; ++ i) {
    auto & buffer = model.buffers[i];
    puleLogDev("Buffer %zu has %zu bytes:", i, buffer.data.byteLength);
  }
  puleLogDev("Model has %zu images", model.imageLen);
  puleLogDev("Model has %zu textures", model.textureLen);
  puleLogDev("Model has %zu accessors", model.accessorLen);
  puleLogDev("Model has %zu buffer views", model.bufferViewLen);
  puleLogDev("Model has %zu materials", model.materialLen);
  puleLogDev("Model has %zu skins", model.skinLen);
  puleLogDev("Model has %zu meshes", model.meshLen);
  puleLogDev("Model has %zu cameras", model.cameraLen);
  puleLogDev("Model has %zu nodes", model.nodeLen);
  puleLogSectionBegin(
    {"Model has %zu animations:", true, true},
    model.animationLen
  );
  for (size_t i = 0; i < model.animationLen; ++ i) {
    auto & animation = model.animations[i];
    puleLogDev("Animation %zu has %zu channels", i, animation.channelLen);
    for (size_t ch = 0; ch < animation.channelLen; ++ ch) {
      auto & channel = animation.channels[ch];
      puleLogDev(
        "  Channel %zu sampler %zu node %s target %s",
        ch,
        (size_t)(animation.samplers - channel.sampler),
        channel.node->name.contents,
        pule::toStr(channel.target).data.contents
      );
    }
    puleLogDev("Animation %zu has %zu samplers", i, animation.samplerLen);
    for (size_t sa = 0; sa < animation.channelLen; ++ sa) {
      auto & sampler = animation.samplers[sa];
      puleLogDev(
        "  sampler %zu interpolation %s timeline %s output %s",
        sa,
        pule::toStr(sampler.interpolation).data.contents,
        sampler.timeline->name.contents,
        sampler.output->name.contents
      );
    }
  }
  puleLogSectionEnd();
}

static void renderModel(
  PulePluginPayload const payload,
  PuleModel3D const & model3d,
  PuleCamera camera
) {
  auto renderGraph = (
    PuleRenderGraph {
      .id = pulePluginPayloadFetchU64(payload, "pule-render-graph"_psv),
    }
  );
  // -- draw
  PuleRenderGraphNode renderNodeDraw = (
    puleRenderGraphNodeFetch(renderGraph, "draw"_psv)
  );

  PuleGpuCommandListRecorder commandRecorder = (
    puleRenderGraph_commandListRecorder(renderNodeDraw)
  );

  puleRenderGraphNode_renderPassBegin(renderNodeDraw, commandRecorder);
  puleModel3DRender(model3d, commandRecorder, camera);
  puleRenderGraphNode_renderPassEnd(renderNodeDraw, commandRecorder);

  // -- blit
  PuleRenderGraphNode renderNodeBlit = (
    puleRenderGraphNodeFetch(renderGraph, "blit"_psv)
  );
  PuleGpuImage const framebufferImage = (
    puleGpuImageReference_image(
      puleRenderGraph_resource(
        renderGraph, "framebuffer-image"_psv
      ).resource.image.imageReference
    )
  );
  PuleGpuImage const windowImage = (
    puleGpuImageReference_image(
      puleRenderGraph_resource(
        renderGraph, "window-swapchain-image"_psv
      ).resource.image.imageReference
    )
  );
  PuleGpuCommandListRecorder commandRecorderBlit = (
    puleRenderGraph_commandListRecorder(renderNodeBlit)
  );
  puleGpuCommandListAppendAction(commandRecorderBlit, PuleGpuCommand {
    .copyImageToImage = {
      .action = PuleGpuAction_copyImageToImage,
      .srcImage = framebufferImage,
      .dstImage = windowImage,
      .srcOffset = { .x = 0, .y = 0 },
      .dstOffset = { .x = 0, .y = 0 },
      .extent = { .x = 800, .y = 600, .z = 1 }, // TODO
    },
  });
}

static void applyAnimationModel(
  PuleModel3D & model3d,
  float msTime
) {
  for (size_t i = 0; i < model3d.asset.animationLen; ++ i) {
    auto & animation = model3d.asset.animations[i];
    for (size_t ch = 0; ch < animation.channelLen; ++ ch) {
      anim::sample(animation.channels[ch], msTime);
    }
  }
}

static void applyAnimationMorphTargetForAttribute(
  PuleModel3D & model3d,
  PuleAssetModelMesh const & mesh,
  PuleAssetModelMeshPrimitive const & primitive,
  size_t primitiveIndex,
  PuleAssetModelAttribute attribute,
  float * weightData,
  size_t weightLen
) {
  // TODO this is pretty horrendous and should just be done at the shader
  //      level with custom pipeline
  // calculate the interpolated morph data
  std::vector<PuleF32v3> morphTargetData;
  auto & attr0 = primitive.morphTargets[0].attributeAccessor[attribute];
  morphTargetData.resize(attr0->elementCount);
  memset(morphTargetData.data(), 0, morphTargetData.size()*sizeof(float));
  // since memory is 0, can just iteratively interpolate data
  for (size_t weightIt = 0; weightIt < weightLen; ++ weightIt) {
    for (size_t attrIt = 0; attrIt < attr0->elementCount; ++ attrIt) {
      float * flData = (
        reinterpret_cast<float *>(
            attr0->bufferView->buffer->data.data
          + attr0->bufferView->byteOffset
          + attr0->byteOffset
          + attrIt*sizeof(float)
        )
      );
      PuleF32v3 v0 = PuleF32v3{flData[0], flData[1], flData[2]};
      PuleF32v3 & morphTargetInterp = morphTargetData[attrIt];
      morphTargetInterp = (
        puleF32v3Add(
          morphTargetInterp,
          puleF32v3MulScalar(v0, weightData[weightIt])
        )
      );
    }
  }
  // apply to attribute buffer
  PuleGpuBuffer attributeBuffer = (
    model3d.draws[
      model3d.meshToPrimitiveToDraw.at(mesh.index).at(primitiveIndex)
    ].buffers[attribute]
  );
  puleGpuBufferMemcpy(
    {
      .buffer = attributeBuffer,
      .byteOffset = 0,
      .byteLength = morphTargetData.size()*sizeof(PuleF32v3),
    },
    morphTargetData.data()
  );
}

static void applyAnimationMorphTargets(PuleModel3D & model3d) {
  for (size_t meshIt = 0; meshIt < model3d.asset.meshLen; ++ meshIt) {
    auto & mesh = model3d.asset.meshes[meshIt];
    if (mesh.weightLen == 0) { continue; }
    PULE_assert(mesh.weights);
    for (size_t primIt = 0; primIt < mesh.primitiveLen; ++ primIt) {
      auto & primitive = mesh.primitives[primIt];
      PULE_assert(mesh.weightLen == primitive.morphTargetLen);
      PULE_assert(primitive.morphTargets);
      for (size_t attrIt = 0; attrIt < 3; ++ attrIt) {
        if (primitive.morphTargets[0].attributeAccessor[attrIt] == nullptr) {
          continue;
        }
        applyAnimationMorphTargetForAttribute(
          model3d, mesh, primitive, primIt, (PuleAssetModelAttribute)attrIt,
          mesh.weights, mesh.weightLen
        );
      }
    }
  }
}

static void updateTransformsModelIt(
  PuleModel3D & model3d,
  PuleAssetModelNode const & node,
  PuleF32m44 const & parentTransform
) {
  PuleF32m44 transform = node.transform;
  PuleF32m44 trs = puleF32m44(1.0f);
  // apply TRS
  if (node.hasScale) {
    PuleF32m44 scale = {{
      node.scale.x, 0.0f, 0.0f, 0.0f,
        0.0f, node.scale.y, 0.0f, 0.0f,
      0.0f, 0.0f, node.scale.z, 0.0f,
      0.0f, 0.0f, 0.0f, 1.0f,
    }};
    trs = puleF32m44Mul(trs, scale);
  }
  if (node.hasRotate) {
    PuleF32m44 qasm33 = puleF32qAsM44(node.rotate);
    trs = puleF32m44Mul(trs, qasm33);
  }
  if (node.hasTranslate) {
    trs.elem[12] += node.translate.x;
    trs.elem[13] += node.translate.y;
    trs.elem[14] += node.translate.z;
  }

  transform = puleF32m44Mul(transform, trs);
  transform = puleF32m44Mul(transform, parentTransform);

  // apply to draws (TODO check if there's even any draws for this node)
  for (auto drawIdx : model3d.nodeToDraws[node.index]) {
    auto & draw = model3d.draws[drawIdx];
    draw.transform = transform;
    puleLog("new transform:");
    puleF32m44DumpToStdout(draw.transform);
  }

  // apply children
  for (size_t i = 0; i < node.childrenLen; ++ i) {
    updateTransformsModelIt(model3d, *node.children[i], transform);
  }
}

static void updateTransformsModel(
  PuleModel3D & model3d
) {
  auto & scene = model3d.asset.scenes[0]; // TODO select scene
  for (size_t nodes = 0; nodes < scene.nodeLen; ++ nodes) {
    updateTransformsModelIt(
      model3d, *scene.nodes[nodes], puleF32m44(1.0f)
    );
  }
}

extern "C" {

PulePluginType pulcPluginType() { return PulePluginType_component; }

static PuleModel3D model3d = {};
static PuleCamera camera = {};
static PuleCameraController cameraController = {};

void pulcComponentLoad([[maybe_unused]] PulePluginPayload const payload) {
  auto model = loadModel(MODEL);
  puleAssetModelDump(model);
  for (size_t it = 0; it < model.loadWarningLen; ++ it) {
    puleLogWarn("model load warning: %s", model.loadWarnings[it].contents);
  }
  if (model.scenes == nullptr) {
    puleLogError("model has no scenes");
    exit(0);
  }
  model3d = prepareModel(model);

  auto platform = (
    PulePlatform {
      .id = pulePluginPayloadFetchU64(payload, "pule-platform"_psv),
    }
  );

  camera = puleCameraCreate();

  // try to generate material
  PuleGpuIr_Pipeline pipeline = (
    puleGpuIr_pipeline(
      "pbr-test"_psv,
       PuleGpuIr_PipelineType_renderVertexFragment
    )
  );

  std::vector<PuleGpuIr_Parameter> vertParams;
  vertParams.push_back({
    .inputAttribute = {
      .paramType = PuleGpuIr_ParameterType_inputAttribute,
      .location = 0,
      .bufferBinding = 0,
      .byteOffset = 0,
      .byteStride = sizeof(float)*3,
    },
  });

  PuleGpuIr_Shader vert = (
    puleGpuIr_pipelineAddShader(
      pipeline,
      PuleGpuIr_ShaderDescriptor {
        .label = "vertex"_psv,
        .stage = {
          .vertex = {
            .stageType = PuleGpuIr_ShaderStageType_vertex,
            .topology = PuleGpuIr_VertexTopology_triangleStrip,
          },
        },
        .params = vertParams.data(),
        .paramLen = vertParams.size(),
      }
    )
  );

  // types / variables / constants
  PuleGpuIr_Type tVoid = puleGpuIr_opTypeVoid(vert);
  PuleGpuIr_Type tFloat = puleGpuIr_opTypeFloat(vert, 32);
  PuleGpuIr_Type tFloat3 = puleGpuIr_opTypeVector(vert, tFloat, 3);
  PuleGpuIr_Type tFloat4 = puleGpuIr_opTypeVector(vert, tFloat, 4);
  PuleGpuIr_Type tFloat3Iattr = (
    puleGpuIr_opTypePointer(vert, tFloat3, PuleGpuIr_StorageClass_input)
  );
  PuleGpuIr_Type tFloat4Oattr = (
    puleGpuIr_opTypePointer(vert, tFloat4, PuleGpuIr_StorageClass_output)
  );
  PuleGpuIr_Type tFnVert = puleGpuIr_opTypeFunction(vert, tVoid, nullptr, 0);
  puleLogDev("tFnVert: %d", tFnVert.id);

  // entry point
  PuleGpuIr_Value vertMain = (
    puleGpuIr_opFunction(
      vert, tVoid, PuleGpuIr_FunctionControl_none, tFnVert,
      "main"_psv
    )
  );
  PuleGpuIr_Value vertLabel = puleGpuIr_opLabel(vert);

  PuleGpuIr_Value inPos = (
    puleGpuIr_opVariableStorage(
      vert, tFloat3Iattr, PuleGpuIr_StorageClass_input, 0
    )
  );
  PuleGpuIr_Value outPos = (
    puleGpuIr_opVariableStorage(
      vert, tFloat4Oattr, PuleGpuIr_StorageClass_output, 0
    )
  );

  // main function
  // TODO function
  PuleGpuIr_Value pos = puleGpuIr_opLoad(vert, tFloat3, inPos);
  PuleGpuIr_Value pos1 = puleGpuIr_opCompositeExtract(vert, tFloat, pos, 0);
  PuleGpuIr_Value pos2 = puleGpuIr_opCompositeExtract(vert, tFloat, pos, 1);
  PuleGpuIr_Value pos3 = puleGpuIr_opCompositeExtract(vert, tFloat, pos, 2);
  std::vector<PuleGpuIr_Value> args;
  args = {
    pos1, pos2, pos3,
    puleGpuIr_opConstant(
      vert, tFloat, PuleGpuIr_ConstantType_float,
      { .floating = 1.0f, }
    )
  };
  PuleGpuIr_Value p4 = (
    puleGpuIr_opCompositeConstruct(
      vert,
      tFloat4,
      args.data(),
      args.size()
    )
  );

  PuleGpuIr_Value np4 = (
    puleGpuIr_opExtInst(vert, tFloat4, "Normalize"_psv, &p4, 1
    )
  );

  puleGpuIr_opReturn(vert);
  puleGpuIr_opFunctionEnd(vert);

  puleGpuIr_pipelineCompile(pipeline);

  exit(0);

  /*
    get the min and max dimensions of the model TODO (
      I'd probably need to reference the node transforms
    )
  */

  cameraController = (
    puleCameraControllerOrbit(platform, camera, puleF32v3(0.0f), 5.0f)
  );
}

void pulcComponentUpdate([[maybe_unused]] PulePluginPayload const payload) {
  puleCameraControllerPollEvents();
  static size_t frame = 0;
  applyAnimationModel(model3d, frame/60.0f/30.0f);
  applyAnimationMorphTargets(model3d);
  updateTransformsModel(model3d);
  renderModel(payload, model3d, camera);
  ++ frame;
}

void pulcComponentUnload([[maybe_unused]] PulePluginPayload const payload) {
}

} // extern C
