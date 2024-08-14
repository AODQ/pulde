#include <pulchritude/gpu-ir.h>

#include <pulchritude/allocator.hpp>
#include <pulchritude/array.hpp>
#include <pulchritude/core.hpp>

#include <string>
#include <vector>

// -- utilities ----------------------------------------------------------------

namespace priv {

enum struct Opcode {
  typeVoid,
  typeFloat,
  typeVector,
  typePointer,
  typeFunction,
  typeStruct,
  typeImage,
  typeSampledImage,
  imageSampleImplicitLod,
  imageSampleExplicitLod,
  variable,
  load,
  store,
  constant,
  logicalNot,
  logicalAnd,
  logicalOr,
  logicalEq,
  logicalNotEq,
  select,
  intEq,
  intNotEq,
  intGreaterThan,
  intGreaterThanEq,
  intLessThan,
  intLessThanEq,
  floatEq,
  floatNotEq,
  floatGreaterThan,
  floatGreaterThanEq,
  floatLessThan,
  floatLessThanEq,
  transpose,
  intNegate,
  floatNegate,
  intAdd,
  intSubtract,
  intMultiply,
  intDivide,
  intModulo,
  floatAdd,
  floatSubtract,
  floatMultiply,
  floatDivide,
  floatRem,
  vectorMulScalar,
  matrixMulScalar,
  vectorMulMatrix,
  matrixMulVector,
  matrixMulMatrix,
  vectorOuterProduct,
  vectorDotProduct,
  branchJmp,
  branchCond,
  returnVoid,
  returnValue,
  accessChain,
  compositeExtract,
  compositeConstruct,
  vectorShuffle,
  extInst,
  functionCall,
  convertSignedToFloat,
  label,
  function,
  functionEnd,
};

struct Instruction {

  Opcode opcode;
  union {
    struct {
      PuleGpuIr_Type returnType;
    } opTypeVoid;
    struct {
      size_t bits;
      PuleGpuIr_Type returnType;
    } opTypeFloat;
    struct {
      PuleGpuIr_Type elementType;
      size_t elementSize;
      PuleGpuIr_Type returnType;
    } opTypeVector;
    struct {
      PuleGpuIr_Type underlyingType;
      PuleGpuIr_StorageClass storageClass;
      PuleGpuIr_Type returnType;
    } opTypePointer;
    struct {
      PuleGpuIr_Type fnReturnType;
      PuleBuffer params;
      PuleGpuIr_Type returnType;
    } opTypeFunction;
    struct {
      PuleBuffer members;
      PuleGpuIr_Type returnType;
    } opTypeStruct;
    struct {
      PuleGpuIr_Type type;
      PuleGpuIr_ImageDim dim;
      PuleGpuIr_ImageDepth depth;
      bool arrayed;
      bool multisampled;
      PuleGpuIr_Type returnType;
    } opTypeImage;
    struct {
      PuleGpuIr_Type imageType;
      PuleGpuIr_Type returnType;
    } opTypeSampledImage;
    struct {
      PuleGpuIr_Type resultType;
      PuleGpuIr_Value image;
      PuleGpuIr_Value coordinate;
      PuleGpuIr_Value returnValue;
    } opImageSampleImplicitLod;
    struct {
      PuleGpuIr_Type resultType;
      PuleGpuIr_Value image;
      PuleGpuIr_Value coordinate;
      PuleGpuIr_Value lod;
      PuleGpuIr_Value returnValue;
    } opImageSampleExplicitLod;
    struct {
      PuleGpuIr_Type type;
      PuleGpuIr_StorageClass storageClass;
      PuleGpuIr_Value returnValue;
    } opVariable;
    struct {
      PuleGpuIr_Type resultType;
      PuleGpuIr_Value pointer;
      PuleGpuIr_Value returnValue;
    } opLoad;
    struct {
      PuleGpuIr_Value pointer;
      PuleGpuIr_Value value;
    } opStore;
    struct {
      PuleGpuIr_Type type;
      PuleGpuIr_ConstantType constantType;
      PuleGpuIr_Constant constant;
      PuleGpuIr_Value returnValue;
    } opConstant;
    struct {
      PuleGpuIr_Value value;
      PuleGpuIr_Value returnValue;
    } opLogicalNot;
    struct {
      PuleGpuIr_Value value1;
      PuleGpuIr_Value value2;
      PuleGpuIr_Value returnValue;
    } opLogicalAnd;
    struct {
      PuleGpuIr_Value value1;
      PuleGpuIr_Value value2;
      PuleGpuIr_Value returnValue;
    } opLogicalOr;
    struct {
      PuleGpuIr_Value value1;
      PuleGpuIr_Value value2;
      PuleGpuIr_Value returnValue;
    } opLogicalEq;
    struct {
      PuleGpuIr_Value value1;
      PuleGpuIr_Value value2;
      PuleGpuIr_Value returnValue;
    } opLogicalNotEq;
    struct {
      PuleGpuIr_Value condition;
      PuleGpuIr_Value valueTrue;
      PuleGpuIr_Value valueFalse;
      PuleGpuIr_Value returnValue;
    } opSelect;
    struct {
      PuleGpuIr_Value value1;
      PuleGpuIr_Value value2;
      PuleGpuIr_Value returnValue;
    } opIntEq;
    struct {
      PuleGpuIr_Value value1;
      PuleGpuIr_Value value2;
      PuleGpuIr_Value returnValue;
    } opIntNotEq;
    struct {
      bool isSigned;
      PuleGpuIr_Value value1;
      PuleGpuIr_Value value2;
      PuleGpuIr_Value returnValue;
    } opIntGreaterThan;
    struct {
      bool isSigned;
      PuleGpuIr_Value value1;
      PuleGpuIr_Value value2;
      PuleGpuIr_Value returnValue;
    } opIntGreaterThanEq;
    struct {
      bool isSigned;
      PuleGpuIr_Value value1;
      PuleGpuIr_Value value2;
      PuleGpuIr_Value returnValue;
    } opIntLessThan;
    struct {
      bool isSigned;
      PuleGpuIr_Value value1;
      PuleGpuIr_Value value2;
      PuleGpuIr_Value returnValue;
    } opIntLessThanEq;
    struct {
      PuleGpuIr_Value value1;
      PuleGpuIr_Value value2;
      PuleGpuIr_Value returnValue;
    } opFloatEq;
    struct {
      PuleGpuIr_Value value1;
      PuleGpuIr_Value value2;
      PuleGpuIr_Value returnValue;
    } opFloatNotEq;
    struct {
      PuleGpuIr_Value value1;
      PuleGpuIr_Value value2;
      PuleGpuIr_Value returnValue;
    } opFloatGreaterThan;
    struct {
      PuleGpuIr_Value value1;
      PuleGpuIr_Value value2;
      PuleGpuIr_Value returnValue;
    } opFloatGreaterThanEq;
    struct {
      PuleGpuIr_Value value1;
      PuleGpuIr_Value value2;
      PuleGpuIr_Value returnValue;
    } opFloatLessThan;
    struct {
      PuleGpuIr_Value value1;
      PuleGpuIr_Value value2;
      PuleGpuIr_Value returnValue;
    } opFloatLessThanEq;
    struct {
      PuleGpuIr_Value value;
      PuleGpuIr_Value returnValue;
    } opTranspose;
    struct {
      PuleGpuIr_Value value;
      PuleGpuIr_Value returnValue;
    } opIntNegate;
    struct {
      PuleGpuIr_Value value;
      PuleGpuIr_Value returnValue;
    } opFloatNegate;
    struct {
      PuleGpuIr_Value value1;
      PuleGpuIr_Value value2;
      PuleGpuIr_Value returnValue;
    } opIntAdd;
    struct {
      PuleGpuIr_Value value1;
      PuleGpuIr_Value value2;
      PuleGpuIr_Value returnValue;
    } opIntSubtract;
    struct {
      PuleGpuIr_Value value1;
      PuleGpuIr_Value value2;
      PuleGpuIr_Value returnValue;
    } opIntMultiply;
    struct {
      bool isSigned;
      PuleGpuIr_Value value1;
      PuleGpuIr_Value value2;
      PuleGpuIr_Value returnValue;
    } opIntDivide;
    struct {
      bool isSigned;
      PuleGpuIr_Value value1;
      PuleGpuIr_Value value2;
      PuleGpuIr_Value returnValue;
    } opIntModulo;
    struct {
      PuleGpuIr_Value value1;
      PuleGpuIr_Value value2;
      PuleGpuIr_Value returnValue;
    } opFloatAdd;
    struct {
      PuleGpuIr_Value value1;
      PuleGpuIr_Value value2;
      PuleGpuIr_Value returnValue;
    } opFloatSubtract;
    struct {
      PuleGpuIr_Value value1;
      PuleGpuIr_Value value2;
      PuleGpuIr_Value returnValue;
    } opFloatMultiply;
    struct {
      PuleGpuIr_Value value1;
      PuleGpuIr_Value value2;
      PuleGpuIr_Value returnValue;
    } opFloatDivide;
    struct {
      PuleGpuIr_Value value1;
      PuleGpuIr_Value value2;
      PuleGpuIr_Value returnValue;
    } opFloatRem;
    struct {
      PuleGpuIr_Value vector;
      PuleGpuIr_Value scalar;
      PuleGpuIr_Value returnValue;
    } opVectorMulScalar;
    struct {
      PuleGpuIr_Value matrix;
      PuleGpuIr_Value scalar;
      PuleGpuIr_Value returnValue;
    } opMatrixMulScalar;
    struct {
      PuleGpuIr_Value vector;
      PuleGpuIr_Value matrix;
      PuleGpuIr_Value returnValue;
    } opVectorMulMatrix;
    struct {
      PuleGpuIr_Value matrix;
      PuleGpuIr_Value vector;
      PuleGpuIr_Value returnValue;
    } opMatrixMulVector;
    struct {
      PuleGpuIr_Value matrix1;
      PuleGpuIr_Value matrix2;
      PuleGpuIr_Value returnValue;
    } opMatrixMulMatrix;
    struct {
      PuleGpuIr_Value vector1;
      PuleGpuIr_Value vector2;
      PuleGpuIr_Value returnValue;
    } opVectorOuterProduct;
    struct {
      PuleGpuIr_Value vector1;
      PuleGpuIr_Value vector2;
      PuleGpuIr_Value returnValue;
    } opVectorDotProduct;
    struct {
      PuleGpuIr_Type type;
      PuleGpuIr_Value value;
      PuleGpuIr_Value returnValue;
    } opConvertSignedToFloat;
    struct {
      PuleGpuIr_Value target;
    } opBranchJmp;
    struct {
      PuleGpuIr_Value condition;
      PuleGpuIr_Value targetTrue;
      PuleGpuIr_Value targetFalse;
    } opBranchCond;
    struct {
    } opReturnVoid;
    struct {
      PuleGpuIr_Value value;
    } opReturnValue;
    struct {
      PuleGpuIr_Value base;
      PuleBuffer indices;
      PuleGpuIr_Value returnValue;
    } opAccessChain;
    struct {
      PuleGpuIr_Type type;
      PuleGpuIr_Value value;
      size_t index;
      PuleGpuIr_Value returnValue;
    } opCompositeExtract;
    struct {
      PuleGpuIr_Type type;
      PuleGpuIr_Value vec0;
      PuleGpuIr_Value vec1;
      PuleBuffer components;
      PuleGpuIr_Value returnValue;
    } opVectorShuffle;
    struct {
      PuleGpuIr_Type type;
      PuleString instruction;
      PuleBuffer operands;
      PuleGpuIr_Value returnValue;
    } opExtInst;
    struct {
      PuleGpuIr_Type fnType;
      PuleGpuIr_Value fn;
      PuleBuffer arguments;
      PuleGpuIr_Value returnValue;
    } opFunctionCall;
    struct {
      PuleGpuIr_Type type;
      PuleBuffer constituents;
      PuleGpuIr_Value returnValue;
    } opCompositeConstruct;
    struct {
      PuleGpuIr_Value target;
    } opLabel;
    struct {
      PuleGpuIr_Type fnReturnType;
      PuleGpuIr_FunctionControl functionControl;
      PuleGpuIr_Type fnType;
      PuleGpuIr_Value returnValue;
      PuleString label;
    } opFunction;
    struct { } opFunctionEnd;
    struct {
      PuleGpuIr_Value functionEntry;
      PuleBuffer globals;
    } opEntryPoint;
    struct {
      PuleGpuIr_Type type;
      PuleBuffer decorations;
    } opDecorate;
    struct {
      PuleGpuIr_Type type;
      size_t memberIndex;
      PuleBuffer decorations;
    } opDecorateMember;
  };

  static void _copy(Instruction & self, Instruction const & instr) {
    memcpy(&self, &instr, sizeof(Instruction));
    if (instr.opcode == Opcode::accessChain) {
      auto & indices = self.opAccessChain.indices;
      indices = (
        puleBufferCopyWithData(
          instr.opAccessChain.indices.allocator,
          instr.opAccessChain.indices.data,
          instr.opAccessChain.indices.byteLength
        )
      );
    }
    if (instr.opcode == Opcode::compositeConstruct) {
      auto & constituents = self.opCompositeConstruct.constituents;
      constituents = (
        puleBufferCopyWithData(
          instr.opCompositeConstruct.constituents.allocator,
          instr.opCompositeConstruct.constituents.data,
          instr.opCompositeConstruct.constituents.byteLength
        )
      );
    }
    if (instr.opcode == Opcode::vectorShuffle) {
      auto & components = self.opVectorShuffle.components;
      components = (
        puleBufferCopyWithData(
          instr.opVectorShuffle.components.allocator,
          instr.opVectorShuffle.components.data,
          instr.opVectorShuffle.components.byteLength
        )
      );
    }
    if (instr.opcode == Opcode::extInst) {
      auto & instrLbl = self.opExtInst.instruction;
      instrLbl = puleStringCopy(instrLbl.allocator, puleStringView(instrLbl));
    }
    if (instr.opcode == Opcode::functionCall) {
      auto & args = self.opFunctionCall.arguments;
      args = (
        puleBufferCopyWithData(
          instr.opFunctionCall.arguments.allocator,
          instr.opFunctionCall.arguments.data,
          instr.opFunctionCall.arguments.byteLength
        )
      );
    }
    if (instr.opcode == Opcode::typeFunction) {
      auto & params = self.opTypeFunction.params;
      params = (
        puleBufferCopyWithData(
          instr.opTypeFunction.params.allocator,
          instr.opTypeFunction.params.data,
          instr.opTypeFunction.params.byteLength
        )
      );
    }
    if (instr.opcode == Opcode::function) {
      self.opFunction.label = (
        puleStringCopy(
          puleAllocateDefault(), puleStringView(instr.opFunction.label)
        )
      );
    }
  }

  static void _move(Instruction & self, Instruction && instr) {
    memcpy(&self, &instr, sizeof(Instruction));
    if (instr.opcode == Opcode::accessChain) {
      self.opAccessChain.indices = instr.opAccessChain.indices;
      instr.opAccessChain.indices.data = nullptr;
    }
    if (instr.opcode == Opcode::compositeConstruct) {
      self.opCompositeConstruct.constituents = (
        instr.opCompositeConstruct.constituents
      );
      instr.opCompositeConstruct.constituents.data = nullptr;
    }
    if (instr.opcode == Opcode::vectorShuffle) {
      self.opVectorShuffle.components = (
        instr.opVectorShuffle.components
      );
      instr.opVectorShuffle.components.data = nullptr;
    }
    if (instr.opcode == Opcode::extInst) {
      self.opExtInst.instruction = instr.opExtInst.instruction;
      instr.opExtInst.instruction.contents = nullptr;
    }
    if (instr.opcode == Opcode::functionCall) {
      self.opFunctionCall.arguments = instr.opFunctionCall.arguments;
      instr.opFunctionCall.arguments.data = nullptr;
    }
    if (instr.opcode == Opcode::typeFunction) {
      self.opTypeFunction.params = instr.opTypeFunction.params;
      instr.opTypeFunction.params.data = nullptr;
    }
    if (instr.opcode == Opcode::function) {
      self.opFunction.label = instr.opFunction.label;
      instr.opFunction.label.contents = nullptr;
    }
  }

  void _destroy() {
    if (this->opcode == Opcode::accessChain) {
      puleBufferDestroy(this->opAccessChain.indices);
    }
    if (this->opcode == Opcode::compositeConstruct) {
      puleBufferDestroy(this->opCompositeConstruct.constituents);
    }
    if (this->opcode == Opcode::vectorShuffle) {
      puleBufferDestroy(this->opVectorShuffle.components);
    }
    if (this->opcode == Opcode::extInst) {
      puleStringDestroy(this->opExtInst.instruction);
    }
    if (this->opcode == Opcode::functionCall) {
      puleBufferDestroy(this->opFunctionCall.arguments);
    }
    if (this->opcode == Opcode::typeFunction) {
      puleBufferDestroy(this->opTypeFunction.params);
    }
    if (this->opcode == Opcode::function) {
      puleStringDestroy(this->opFunction.label);
    }
  }

  Instruction() {}
  ~Instruction() {
    _destroy();
  }
  Instruction(Instruction const & instr) {
    _copy(*this, instr);
  }
  Instruction(Instruction && instr) {
    _move(*this, std::move(instr));
  }
  Instruction & operator=(Instruction const & instr) {
    _destroy();
    _copy(*this, instr);
    return *this;
  }
  Instruction & operator=(Instruction && instr) {
    _destroy();
    _move(*this, std::move(instr));
    return *this;
  }
};

struct Pipeline;

struct EntryPointValue {
  PuleGpuIr_Value value;
  PuleGpuIr_StorageClass storageClass;
  size_t layoutIndex;
};

struct Shader {
  std::string label;
  size_t irId;
  PuleGpuIr_ShaderStage stage;
  std::vector<PuleGpuIr_Parameter> parameters;
  std::vector<PuleGpuIr_Value> values;
  std::vector<PuleGpuIr_Type> types;

  struct {
    PuleGpuIr_Value entryPointFn;
    std::vector<PuleGpuIr_Value> globals;
  } entryPoint;

  struct Decoration {
    PuleGpuIr_Type type;
    int64_t memberIndex; // -1 if not a member-call
    std::vector<uint32_t> values;
  };

  std::vector<Decoration> decorations;

  std::vector<EntryPointValue> entryPointInOuts;

  std::vector<Instruction> instructions;

  size_t irIdCounter = 2; // 0, 1, 2 and 3 are reserved for header

  PuleGpuIr_Value newValue() {
    values.push_back(PuleGpuIr_Value { .id = irIdCounter, });
    ++ irIdCounter;
    return values.back();
  }

  PuleGpuIr_Type newType() {
    types.push_back(PuleGpuIr_Type { .id = irIdCounter, });
    ++ irIdCounter;
    return types.back();
  }
};

struct Pipeline {
  std::string label;
  PuleGpuIr_PipelineType type;
  std::vector<PuleGpuIr_Shader> entryPoints;
};

pule::ResourceContainer<Pipeline, PuleGpuIr_Pipeline> pipelines;
pule::ResourceContainer<Shader, PuleGpuIr_Shader> shaders;

} // namespace priv

// -- pipeline -----------------------------------------------------------------

namespace priv {

void compilePipelineInstruction(
  priv::Instruction const & instr,
  std::string & data,
  bool filterInFunction
) {
  auto toStr = [](PuleGpuIr_Value const value) -> std::string {
    return "%" + std::to_string(value.id);
  };
  auto toTypeStr = [](PuleGpuIr_Type const value) -> std::string {
    return "%" + std::to_string(value.id);
  };
  auto toStorageClassStr = [](
    PuleGpuIr_StorageClass const storageClass
  ) -> std::string {
    switch (storageClass) {
      default: return "Unknown";
      case PuleGpuIr_StorageClass_generic: return "";
      case PuleGpuIr_StorageClass_function: return "Function";
      case PuleGpuIr_StorageClass_input: return "Input";
      case PuleGpuIr_StorageClass_output: return "Output";
      case PuleGpuIr_StorageClass_uniform: return "Uniform";
      case PuleGpuIr_StorageClass_storageBuffer: return "StorageBuffer";
      case PuleGpuIr_StorageClass_pushConstant: return "PushConstant";
    }
  };

  // apply filterInFunction
  switch (instr.opcode) {
    case priv::Opcode::typeVoid:
    case priv::Opcode::typeFloat:
    case priv::Opcode::typeVector:
    case priv::Opcode::typePointer:
    case priv::Opcode::typeFunction:
    case priv::Opcode::typeStruct:
    case priv::Opcode::constant:
      if (!filterInFunction) { return; }
    break;
    case priv::Opcode::variable: {
      bool isGlobal = (
        instr.opVariable.storageClass != PuleGpuIr_StorageClass_generic
      );
      if (isGlobal != filterInFunction) { return; }
    } break;
    default:
      if (filterInFunction) { return; }
    break;
  }

  data += "\n";
  switch (instr.opcode) {
    case priv::Opcode::typeStruct: {
      auto & op = instr.opTypeStruct;
      data += (
          toTypeStr(op.returnType)
        + " = OpTypeStruct"
      );
      PuleGpuIr_Type * members = (PuleGpuIr_Type *)op.members.data;
      size_t memberLen = op.members.byteLength / sizeof(PuleGpuIr_Type);
      for (size_t it = 0; it < memberLen; ++ it) {
        data += " " + toTypeStr(members[it]);
      }
    } break;
    case priv::Opcode::typeVoid: {
      auto & op = instr.opTypeVoid;
      data += (
          toTypeStr(op.returnType)
        + " = OpTypeVoid"
      );
    } break;
    case priv::Opcode::typeFloat: {
      auto & op = instr.opTypeFloat;
      data += (
          toTypeStr(op.returnType)
        + " = OpTypeFloat " + std::to_string(op.bits)
      );
    } break;
    case priv::Opcode::typeVector: {
      auto & op = instr.opTypeVector;
      data += (
          toTypeStr(op.returnType)
        + " = OpTypeVector "
        + toTypeStr(op.elementType)
        + " " + std::to_string(op.elementSize)
      );
    } break;
    case priv::Opcode::typePointer: {
      auto & op = instr.opTypePointer;
      data += (
          toTypeStr(op.returnType)
        + " = OpTypePointer "
        + toStorageClassStr(op.storageClass)
        + " " + toTypeStr(op.underlyingType)
      );
    } break;
    case priv::Opcode::typeFunction: {
      auto & op = instr.opTypeFunction;
      data += (
        toTypeStr(op.returnType)
      + " = OpTypeFunction "
      + toTypeStr(op.fnReturnType)
      );
      PuleGpuIr_Type * params = (PuleGpuIr_Type *)op.params.data;
      size_t paramLen = op.params.byteLength / sizeof(PuleGpuIr_Type);
      for (size_t it = 0; it < paramLen; ++ it) {
        data += " " + toTypeStr(params[it]);
      }
    } break;
    case priv::Opcode::typeImage: {
      auto & op = instr.opTypeImage;
      data += (
          toTypeStr(op.returnType)
        + " = OpTypeImage "
        + toTypeStr(op.type) + " "
      );
      switch (op.dim) {
        case PuleGpuIr_ImageDim_i1d: data += "1D"; break;
        case PuleGpuIr_ImageDim_i2d: data += "2D"; break;
        case PuleGpuIr_ImageDim_i3d: data += "3D"; break;
        case PuleGpuIr_ImageDim_cube: data += "Cube"; break;
        case PuleGpuIr_ImageDim_rect: data += "Rect"; break;
        case PuleGpuIr_ImageDim_buffer: data += "Buffer"; break;
        case PuleGpuIr_ImageDim_subpassData: data += "SubpassData"; break;
      }
      data += " ";
      switch (op.depth) {
        case PuleGpuIr_ImageDepth_noDepth: data += " 0"; break;
        case PuleGpuIr_ImageDepth_depth: data += " 1"; break;
        case PuleGpuIr_ImageDepth_unknown: data += " 2"; break;
      }
      data += " " + std::string(op.arrayed ? "1" : "0");
      data += " " + std::string(op.multisampled ? "1" : "0");
    } break;
    case priv::Opcode::typeSampledImage: {
      auto & op = instr.opTypeSampledImage;
      data += (
          toTypeStr(op.returnType)
        + " = OpTypeSampledImage "
        + toTypeStr(op.imageType)
      );
    } break;
    case priv::Opcode::imageSampleImplicitLod: {
      auto & op = instr.opImageSampleImplicitLod;
      data += (
          toStr(op.returnValue)
        + " = OpImageSampleImplicitLod "
        + toTypeStr(op.resultType)
        + " " + toStr(op.image)
        + " " + toStr(op.coordinate)
      );
    } break;
    case priv::Opcode::imageSampleExplicitLod: {
      auto & op = instr.opImageSampleExplicitLod;
      data += (
          toStr(op.returnValue)
        + " = OpImageSampleExplicitLod "
        + toTypeStr(op.resultType)
        + " " + toStr(op.image)
        + " " + toStr(op.coordinate)
        + " " + toStr(op.lod)
      );
    } break;
    case priv::Opcode::variable: {
      auto & op = instr.opVariable;
      data += (
          toStr(op.returnValue)
        + " = OpVariable "
        + toTypeStr(op.type)
        + " " + toStorageClassStr(op.storageClass)
      );
    } break;
    case priv::Opcode::load: {
      auto & op = instr.opLoad;
      data += (
          "%" + std::to_string(op.returnValue.id)
        + " = OpLoad "
        + toTypeStr(op.resultType)
        + " " + toStr(op.pointer)
      );
    } break;
    case priv::Opcode::store: {
      auto & op = instr.opStore;
      data += (
        "OpStore " + toStr(op.pointer) + " " + toStr(op.value)
      );
    } break;
    case priv::Opcode::constant: {
      auto & op = instr.opConstant;
      data += (
          toStr(op.returnValue)
        + " = OpConstant "
        + toTypeStr(op.type)
        + " "
      );
      switch (op.constantType) {
        default: PULE_assert(false && "unsupported type");
        case PuleGpuIr_ConstantType_bool: {
          data += (op.constant.boolean ? "true" : "false");
        } break;
        case PuleGpuIr_ConstantType_int:
          data += std::to_string(op.constant.integer);
        break;
        case PuleGpuIr_ConstantType_float:
          data += std::to_string(op.constant.floating);
        break;
      }
    } break;
    case priv::Opcode::logicalNot: {
      auto & op = instr.opLogicalNot;
      data += toStr(op.returnValue) + " = OpLogicalNot " + toStr(op.value);
    } break;
    case priv::Opcode::logicalAnd: {
      auto & op = instr.opLogicalAnd;
      data += (
          toStr(op.returnValue)
        + " = OpLogicalAnd "
        + toStr(op.value1)
        + " " + toStr(op.value2)
      );
    } break;
    case priv::Opcode::logicalOr: {
      auto & op = instr.opLogicalOr;
      data += (
          toStr(op.returnValue)
        + " = OpLogicalOr "
        + toStr(op.value1)
        + " " + toStr(op.value2)
      );
    } break;
    case priv::Opcode::logicalEq: {
      auto & op = instr.opLogicalEq;
      data += (
          toStr(op.returnValue)
        + " = OpLogicalEq "
        + toStr(op.value1)
        + " " + toStr(op.value2)
      );
    } break;
    case priv::Opcode::logicalNotEq: {
      auto & op = instr.opLogicalNotEq;
      data += (
          toStr(op.returnValue)
        + " = OpLogicalNotEq "
        + toStr(op.value1)
        + " " + toStr(op.value2)
      );
    } break;
    case priv::Opcode::select: {
      auto & op = instr.opSelect;
      data += (
          toStr(op.returnValue)
        + " = OpSelect "
        + toStr(op.condition)
        + " " + toStr(op.valueTrue)
        + " " + toStr(op.valueFalse)
      );
    } break;
    case priv::Opcode::intEq: {
      auto & op = instr.opIntEq;
      data += (
          toStr(op.returnValue)
        + " = OpIEqual "
        + toStr(op.value1)
        + " " + toStr(op.value2)
      );
    } break;
    case priv::Opcode::intNotEq: {
      auto & op = instr.opIntNotEq;
      data += (
          toStr(op.returnValue)
        + " = OpINotEqual "
        + toStr(op.value1)
        + " " + toStr(op.value2)
      );
    } break;
    case priv::Opcode::intGreaterThan: {
      auto & op = instr.opIntGreaterThan;
      data += (
          toStr(op.returnValue)
        + " = OpSGreaterThan "
        + toStr(op.value1)
        + " " + toStr(op.value2)
      );
    } break;
    case priv::Opcode::intGreaterThanEq: {
      auto & op = instr.opIntGreaterThanEq;
      data += (
          toStr(op.returnValue)
        + " = OpSGreaterThanEqual "
        + toStr(op.value1)
        + " " + toStr(op.value2)
      );
    } break;
    case priv::Opcode::intLessThan: {
      auto & op = instr.opIntLessThan;
      data += (
          toStr(op.returnValue)
        + " = OpSLessThan "
        + toStr(op.value1)
        + " " + toStr(op.value2)
      );
    } break;
    case priv::Opcode::intLessThanEq: {
      auto & op = instr.opIntLessThanEq;
      data += (
          toStr(op.returnValue)
        + " = OpSLessThanEqual "
        + toStr(op.value1)
        + " " + toStr(op.value2)
      );
    } break;
    case priv::Opcode::floatEq: {
      auto & op = instr.opFloatEq;
      data += (
          toStr(op.returnValue)
        + " = OpFOrdEqual "
        + toStr(op.value1)
        + " " + toStr(op.value2)
      );
    } break;
    case priv::Opcode::floatNotEq: {
      auto & op = instr.opFloatNotEq;
      data += (
          toStr(op.returnValue)
        + " = OpFOrdNotEqual "
        + toStr(op.value1)
        + " " + toStr(op.value2)
      );
    } break;
    case priv::Opcode::floatGreaterThan: {
      auto & op = instr.opFloatGreaterThan;
      data += (
          toStr(op.returnValue)
        + " = OpFOrdGreaterThan "
        + toStr(op.value1)
        + " " + toStr(op.value2)
      );
    } break;
    case priv::Opcode::floatGreaterThanEq: {
      auto & op = instr.opFloatGreaterThanEq;
      data += (
          toStr(op.returnValue)
        + " = OpFOrdGreaterThanEqual "
        + toStr(op.value1)
        + " " + toStr(op.value2)
      );
    } break;
    case priv::Opcode::floatLessThan: {
      auto & op = instr.opFloatLessThan;
      data += (
          toStr(op.returnValue)
        + " = OpFOrdLessThan "
        + toStr(op.value1)
        + " " + toStr(op.value2)
      );
    } break;
    case priv::Opcode::floatLessThanEq: {
      auto & op = instr.opFloatLessThanEq;
      data += (
          toStr(op.returnValue)
        + " = OpFOrdLessThanEqual "
        + toStr(op.value1)
        + " " + toStr(op.value2)
      );
    } break;
    case priv::Opcode::transpose: {
      auto & op = instr.opTranspose;
      data += toStr(op.returnValue) + " = OpTranspose " + toStr(op.value);
    } break;
    case priv::Opcode::intNegate: {
      auto & op = instr.opIntNegate;
      data += toStr(op.returnValue) + " = OpSNegate " + toStr(op.value);
    } break;
    case priv::Opcode::floatNegate: {
      auto & op = instr.opFloatNegate;
      data += toStr(op.returnValue) + " = OpFNegate " + toStr(op.value);
    } break;
    case priv::Opcode::intAdd: {
      auto & op = instr.opIntAdd;
      data += (
          toStr(op.returnValue)
        + " = OpIAdd "
        + toStr(op.value1)
        + " " + toStr(op.value2)
      );
    } break;
    case priv::Opcode::intSubtract: {
      auto & op = instr.opIntSubtract;
      data += (
          toStr(op.returnValue)
        + " = OpISub "
        + toStr(op.value1)
        + " " + toStr(op.value2)
      );
    } break;
    case priv::Opcode::intMultiply: {
      auto & op = instr.opIntMultiply;
      data += (
          toStr(op.returnValue)
        + " = OpIMul "
        + toStr(op.value1)
        + " " + toStr(op.value2)
      );
    } break;
    case priv::Opcode::intDivide: {
      auto & op = instr.opIntDivide;
      data += (
          toStr(op.returnValue)
        + " = OpSDiv "
        + toStr(op.value1)
        + " " + toStr(op.value2)
      );
    } break;
    case priv::Opcode::intModulo: {
      auto & op = instr.opIntModulo;
      data += (
          toStr(op.returnValue)
        + " = OpSMod "
        + toStr(op.value1)
        + " " + toStr(op.value2)
      );
    } break;
    case priv::Opcode::floatAdd: {
      auto & op = instr.opFloatAdd;
      data += (
          toStr(op.returnValue)
        + " = OpFAdd "
        + toStr(op.value1)
        + " " + toStr(op.value2)
      );
    } break;
    case priv::Opcode::floatSubtract: {
      auto & op = instr.opFloatSubtract;
      data += (
          toStr(op.returnValue)
        + " = OpFSub "
        + toStr(op.value1)
        + " " + toStr(op.value2)
      );
    } break;
    case priv::Opcode::floatMultiply: {
      auto & op = instr.opFloatMultiply;
      data += (
          toStr(op.returnValue)
        + " = OpFMul "
        + toStr(op.value1)
        + " " + toStr(op.value2)
      );
    } break;
    case priv::Opcode::floatDivide: {
      auto & op = instr.opFloatDivide;
      data += (
          toStr(op.returnValue)
        + " = OpFDiv "
        + toStr(op.value1)
        + " " + toStr(op.value2)
      );
    } break;
    case priv::Opcode::floatRem: {
      auto & op = instr.opFloatRem;
      data += (
          toStr(op.returnValue)
        + " = OpFRem "
        + toStr(op.value1)
        + " " + toStr(op.value2)
      );
    } break;
    case priv::Opcode::vectorMulScalar: {
      auto & op = instr.opVectorMulScalar;
      data += (
          toStr(op.returnValue)
        + " = OpVectorTimesScalar "
        + toStr(op.vector)
        + " " + toStr(op.scalar)
      );
    } break;
    case priv::Opcode::matrixMulScalar: {
      auto & op = instr.opMatrixMulScalar;
      data += (
          toStr(op.returnValue)
        + " = OpMatrixTimesScalar "
        + toStr(op.matrix)
        + " " + toStr(op.scalar)
      );
    } break;
    case priv::Opcode::vectorMulMatrix: {
      auto & op = instr.opVectorMulMatrix;
      data += (
          toStr(op.returnValue)
        + " = OpVectorTimesMatrix "
        + toStr(op.vector)
        + " " + toStr(op.matrix)
      );
    } break;
    case priv::Opcode::matrixMulVector: {
      auto & op = instr.opMatrixMulVector;
      data += (
          toStr(op.returnValue)
        + " = OpMatrixTimesVector "
        + toStr(op.matrix)
        + " " + toStr(op.vector)
      );
    } break;
    case priv::Opcode::matrixMulMatrix: {
      auto & op = instr.opMatrixMulMatrix;
      data += (
          toStr(op.returnValue)
        + " = OpMatrixTimesMatrix "
        + toStr(op.matrix1)
        + " " + toStr(op.matrix2)
      );
    } break;
    case priv::Opcode::vectorOuterProduct: {
      auto & op = instr.opVectorOuterProduct;
      data += (
          toStr(op.returnValue)
        + " = OpOuterProduct "
        + toStr(op.vector1)
        + " " + toStr(op.vector2)
      );
    } break;
    case priv::Opcode::vectorDotProduct: {
      auto & op = instr.opVectorDotProduct;
      data += (
          toStr(op.returnValue)
        + " = OpDot "
        + toStr(op.vector1)
        + " " + toStr(op.vector2)
      );
    } break;
    case priv::Opcode::convertSignedToFloat: {
      auto & op = instr.opConvertSignedToFloat;
      data += (
          toStr(op.returnValue)
        + " = OpConvertSToF "
        + toTypeStr(op.type)
        + " " + toStr(op.value)
      );
    } break;
    case priv::Opcode::branchJmp: {
      auto & op = instr.opBranchJmp;
      data += "OpBranch " + toStr(op.target);
    } break;
    case priv::Opcode::branchCond: {
      auto & op = instr.opBranchCond;
      data += (
          "OpBranchConditional "
        + toStr(op.condition)
        + " " + toStr(op.targetTrue)
        + " " + toStr(op.targetFalse)
      );
    } break;
    case priv::Opcode::returnVoid: {
      data += "OpReturn";
    } break;
    case priv::Opcode::returnValue: {
      auto & op = instr.opReturnValue;
      data += "OpReturnValue " + toStr(op.value);
    } break;
    case priv::Opcode::accessChain: {
      auto & op = instr.opAccessChain;
      data += toStr(op.returnValue) + " = OpAccessChain " + toStr(op.base);
      size_t const indexCount = op.indices.byteLength / sizeof(PuleGpuIr_Value);
      for (size_t it = 0; it < indexCount; ++ it) {
        data += " " + toStr(((PuleGpuIr_Value *)op.indices.data)[it]);
      }
    } break;
    case priv::Opcode::compositeExtract: {
      auto & op = instr.opCompositeExtract;
      data += (
          toStr(op.returnValue)
        + " = OpCompositeExtract "
        + toTypeStr(op.type)
        + " " + toStr(op.value)
        + " " + std::to_string(op.index)
      );
    } break;
    case priv::Opcode::compositeConstruct: {
      auto & op = instr.opCompositeConstruct;
      data += toStr(op.returnValue) + " = OpCompositeConstruct";
      data += " " + toTypeStr(op.type);
      PuleGpuIr_Value * constituents = (PuleGpuIr_Value *)op.constituents.data;
      size_t const count = op.constituents.byteLength / sizeof(PuleGpuIr_Value);
      for (size_t it = 0; it < count; ++ it) {
        data += " " + toStr(constituents[it]);
      }
    } break;
    case priv::Opcode::vectorShuffle: {
      auto & op = instr.opVectorShuffle;
      data += (
        toStr(op.returnValue)
        + " = OpVectorShuffle "
        + toTypeStr(op.type)
        + " " + toStr(op.vec0)
        + " " + toStr(op.vec1)
      );
      uint32_t * components = (uint32_t *)op.components.data;
      size_t const componentLen = op.components.byteLength / sizeof(uint32_t);
      for (size_t it = 0; it < componentLen; ++ it) {
        data += " " + std::to_string(components[it]);
      }
    } break;
    case priv::Opcode::extInst: {
      auto & op = instr.opExtInst;
      data += (
          toStr(op.returnValue)
        + " = OpExtInst "
        + toTypeStr(op.type)
        + " %1 "
        + std::string(op.instruction.contents)
      );
      PuleGpuIr_Value * operands = (PuleGpuIr_Value *)op.operands.data;
      size_t const operandLen = op.operands.byteLength/sizeof(PuleGpuIr_Value);
      for (size_t it = 0; it < operandLen; ++ it) {
        data += " " + toStr(operands[it]);
      }
    } break;
    case priv::Opcode::functionCall: {
      auto & op = instr.opFunctionCall;
      data += (
          toStr(op.returnValue)
        + " = OpFunctionCall "
        + toTypeStr(op.fnType)
        + " " + toStr(op.fn)
      );
      PuleGpuIr_Value * args = (PuleGpuIr_Value *)op.arguments.data;
      size_t const argLen = op.arguments.byteLength / sizeof(PuleGpuIr_Value);
      for (size_t it = 0; it < argLen; ++ it) {
        data += " " + toStr(args[it]);
      }
    } break;
    case priv::Opcode::label: {
      auto & op = instr.opLabel;
      data += toStr(op.target) + " = OpLabel";
    } break;
    case priv::Opcode::function: {
      auto & op = instr.opFunction;
      data += (
          toStr(op.returnValue)
        + " = OpFunction "
        + toTypeStr(op.fnReturnType)
        + " "
      );
      switch (op.functionControl) {
        default: PULE_assert(false && "unsupported function control");
        case PuleGpuIr_FunctionControl_none: {
          data += "None";
        } break;
        case PuleGpuIr_FunctionControl_inline: {
          data += "Inline";
        } break;
      }
      puleLogDev("function type: %d", op.fnType.id);
      data += " " + toTypeStr(op.fnType);
    } break;
    case priv::Opcode::functionEnd: {
      data += "OpFunctionEnd";
    } break;
  }
}

std::vector<uint32_t> compilePipeline(
  priv::Shader const & shaderStage
) {
  std::string dataSpirvPlaintext;
  // -- header
  dataSpirvPlaintext += "OpCapability Shader\n";
  dataSpirvPlaintext += "%1 = OpExtInstImport \"GLSL.std.450\"\n";
  dataSpirvPlaintext += "OpMemoryModel Logical GLSL450\n";

  { // mode setting
    dataSpirvPlaintext += "OpEntryPoint ";
    switch (shaderStage.stage.stageType) {
      default: PULE_assert(false && "unsupported shader stage");
      case PuleGpuIr_ShaderStageType_vertex: {
        dataSpirvPlaintext += "Vertex";
      } break;
      case PuleGpuIr_ShaderStageType_fragment: {
        dataSpirvPlaintext += "Fragment";
      } break;
    }
    // find the entry point
    for (auto & instr : shaderStage.instructions) {
      if (instr.opcode != priv::Opcode::function) { continue; }
      if (puleStringViewEq(puleStringView(instr.opFunction.label), "main"_psv)){
        dataSpirvPlaintext += (
          " %" + std::to_string(instr.opFunction.returnValue.id)
        );
      }
      break;
    }
    dataSpirvPlaintext += " \"main\"";
    // now have to insert entry point in-outs
    for (auto & param : shaderStage.entryPointInOuts) {
      dataSpirvPlaintext += " %" + std::to_string(param.value.id);
    }
    dataSpirvPlaintext += "\n";
  }

  // debug information & annotations
  switch (shaderStage.stage.stageType) {
    default: PULE_assert(false && "unimplemented shader stage");
    case PuleGpuIr_ShaderStageType_vertex:
      // debug information
      dataSpirvPlaintext += "OpSource GLSL 460\n";
      for (auto & instr : shaderStage.instructions) {
        if (instr.opcode != priv::Opcode::function) { continue; }
        dataSpirvPlaintext += (
          "OpName %" + std::to_string(instr.opFunction.returnValue.id)
          + " \"" + std::string(instr.opFunction.label.contents) + "\"\n"
        );
      }
    break;
  }

  // insert decorations
  for (auto & decoration : shaderStage.decorations) {
    dataSpirvPlaintext += (
      (decoration.memberIndex == -1 ? "OpDecorate " : "OpMemberDecorate ")
    );
    dataSpirvPlaintext += "%" + std::to_string(decoration.type.id) + " ";
    if (decoration.memberIndex != -1) {
      dataSpirvPlaintext += std::to_string(decoration.memberIndex);
    }
    PULE_assert(decoration.values.size() > 0 && "no decoration values");
    switch (decoration.values[0]) {
      default: PULE_assert(false && "unsupported decoration");
      case PuleGpuIr_Decoration_builtin: {
        dataSpirvPlaintext += "BuiltIn ";
        static std::unordered_map<uint32_t, std::string> const builtinMap = {
          { PuleGpuIr_Builtin_origin, "Position" },
          { PuleGpuIr_Builtin_pointSize, "PointSize" },
          { PuleGpuIr_Builtin_clipDistance, "ClipDistance" },
          { PuleGpuIr_Builtin_cullDistance, "CullDistance" },
          { PuleGpuIr_Builtin_vertexId, "VertexID" },
          { PuleGpuIr_Builtin_instanceId, "instanceID" },
        };
        dataSpirvPlaintext += builtinMap.at(decoration.values[1]);
      } break;
      case PuleGpuIr_Decoration_block: {
        dataSpirvPlaintext += "Block";
      } break;
      case PuleGpuIr_Decoration_location: {
        dataSpirvPlaintext += (
          "Location " + std::to_string(decoration.values[1])
        );
      } break;
    }
  }

  // TODO maybe consider sorting instructions so all types come first?

  // -- instructions, there are two passes. First for types and constants,
  //    which can never be in a function, then for the rest
  for (auto & instr : shaderStage.instructions) {
    compilePipelineInstruction(instr, dataSpirvPlaintext, true);
  }
  for (auto & instr : shaderStage.instructions) {
    compilePipelineInstruction(instr, dataSpirvPlaintext, false);
  }

  puleLogDev("generated SPIRV:```\n%s\n```", dataSpirvPlaintext.c_str());

  puleGpuIr_compileSpirv(puleCStr(dataSpirvPlaintext.c_str()));

  return {};
}

} // namespace priv

extern "C" {

PuleGpuIr_Pipeline puleGpuIr_pipeline(
  PuleStringView label,
  PuleGpuIr_PipelineType type
) {
  auto pipeline = priv::Pipeline {
    .label = std::string(label.contents),
    .type = type,
  };
  return priv::pipelines.create(pipeline);
}

void puleGpuIrPipelineDestroy(PuleGpuIr_Pipeline const pipeline) {
  priv::pipelines.destroy(pipeline);
}

PuleGpuPipeline puleGpuIr_pipelineCompile(PuleGpuIr_Pipeline const pipeline) {
  auto & privPipeline = *priv::pipelines.at(pipeline);
  PuleGpuPipeline gpuPipeline;
  switch (privPipeline.type) {
    default: PULE_assert(false && "unsupported pipeline type");
    case PuleGpuIr_PipelineType_renderVertexFragment: {
      std::vector<uint32_t> vertexFnSpirv;
      std::vector<uint32_t> fragmentFnSpirv;
      for (auto & fn : privPipeline.entryPoints) {
        auto const & shader = *priv::shaders.at(fn);
        auto const stageType = shader.stage.stageType;
        switch (stageType) {
          default: PULE_assert(false && "unsupported shader stage");
          case PuleGpuIr_ShaderStageType_vertex: {
            vertexFnSpirv = priv::compilePipeline(shader);
          } break;
          case PuleGpuIr_ShaderStageType_fragment: {
            fragmentFnSpirv = priv::compilePipeline(shader);
          } break;
        }
      }
    } break;
  }
  return gpuPipeline;
}

} // extern "C"

// -- misc ---------------------------------------------------------------------
extern "C" {

PuleGpuIr_Shader puleGpuIr_pipelineAddShader(
  PuleGpuIr_Pipeline pipeline,
  PuleGpuIr_ShaderDescriptor descriptor
) {
  auto & privPipeline = *priv::pipelines.at(pipeline);
  puleLogDev("adding shader '%s' descriptor params %p len %zu",
    descriptor.label.contents, descriptor.params, descriptor.paramLen
  );
  priv::Shader shader {};
  shader.label = std::string(descriptor.label.contents);
  shader.stage = descriptor.stage;
  shader.parameters = std::vector<PuleGpuIr_Parameter>(
    descriptor.params,
    descriptor.params + descriptor.paramLen
  );
  privPipeline.entryPoints.emplace_back(
    priv::shaders.create(shader)
  );
  return privPipeline.entryPoints.back();
}

} // extern "C"

// -- instructions -------------------------------------------------------------

#define PRIV_SHADER(opcode_) \
  auto & privShader = *priv::shaders.at(shader); \
  priv::Instruction instr; \
  instr.opcode = opcode_

#define PRIV_SHADER_VALUE(opcode_) \
  auto & privShader = *priv::shaders.at(shader); \
  auto returnValue = privShader.newValue(); \
  priv::Instruction instr; \
  instr.opcode = opcode_
#define PRIV_SHADER_TYPE(opcode_) \
  auto & privShader = *priv::shaders.at(shader); \
  auto returnType = privShader.newType(); \
  priv::Instruction instr; \
  instr.opcode = opcode_

extern "C" {

PuleGpuIr_Type puleGpuIr_opTypeVoid(PuleGpuIr_Shader shader) {
  PRIV_SHADER_TYPE(priv::Opcode::typeVoid);
  instr.opTypeVoid = {
    .returnType = returnType,
  };
  privShader.instructions.push_back(instr);
  return returnType;
}

PuleGpuIr_Type puleGpuIr_opTypeFloat(PuleGpuIr_Shader shader, size_t bits) {
  PRIV_SHADER_TYPE(priv::Opcode::typeFloat);
  instr.opTypeFloat = {
    .bits = bits,
    .returnType = returnType,
  };
  privShader.instructions.push_back(instr);
  return returnType;
}

PuleGpuIr_Type puleGpuIr_opTypeFunction(
  PuleGpuIr_Shader shader, PuleGpuIr_Type fnReturnType,
  PuleGpuIr_Type const * params, size_t paramLen
) {
  PRIV_SHADER_TYPE(priv::Opcode::typeFunction);
  instr.opTypeFunction = {
    .fnReturnType = fnReturnType,
    .params = (
      puleBufferCopyWithData(
        puleAllocateDefault(),
        (uint8_t const *)params, paramLen * sizeof(PuleGpuIr_Type)
      )
    ),
    .returnType = returnType,
  };
  privShader.instructions.push_back(instr);
  return returnType;
}

PuleGpuIr_Type puleGpuIr_opTypeVector(
  PuleGpuIr_Shader shader,
  PuleGpuIr_Type elementType, size_t elementSize
) {
  PRIV_SHADER_TYPE(priv::Opcode::typeVector);
  instr.opTypeVector = {
    .elementType = elementType,
    .elementSize = elementSize,
    .returnType = returnType,
  };
  privShader.instructions.push_back(instr);
  return returnType;
}

PuleGpuIr_Type puleGpuIr_opTypePointer(
  PuleGpuIr_Shader shader,
  PuleGpuIr_Type underlyingType, PuleGpuIr_StorageClass storageClass
) {
  PRIV_SHADER_TYPE(priv::Opcode::typePointer);
  instr.opTypePointer = {
    .underlyingType = underlyingType,
    .storageClass = storageClass,
    .returnType = returnType,
  };
  privShader.instructions.push_back(instr);
  return returnType;
}

PuleGpuIr_Type puleGpuIr_opTypeStruct(
  PuleGpuIr_Shader shader,
  PuleGpuIr_Type const * members, size_t memberLen
) {
  PRIV_SHADER_TYPE(priv::Opcode::typeStruct);
  instr.opTypeStruct = {
    .members = (
      puleBufferCopyWithData(
        puleAllocateDefault(),
        (uint8_t const *)members, memberLen * sizeof(PuleGpuIr_Type)
      )
    ),
    .returnType = returnType,
  };
  privShader.instructions.push_back(instr);
  return returnType;
}

PuleGpuIr_Type puleGpuIr_opTypeImage(
  PuleGpuIr_Shader shader,
  PuleGpuIr_Type sampledType,
  PuleGpuIr_ImageDim dim,
  PuleGpuIr_ImageDepth depth,
  bool arrayed,
  bool multisampled
) {
  PRIV_SHADER_TYPE(priv::Opcode::typeImage);
  instr.opTypeImage = {
    .type = sampledType,
    .dim = dim,
    .depth = depth,
    .arrayed = arrayed,
    .multisampled = multisampled,
    .returnType = returnType,
  };
  privShader.instructions.push_back(instr);
  return returnType;
}

PuleGpuIr_Type puleGpuIr_opTypeSampledImage(
  PuleGpuIr_Shader shader,
  PuleGpuIr_Type type
) {
  PRIV_SHADER_TYPE(priv::Opcode::typeSampledImage);
  instr.opTypeSampledImage = {
    .imageType = type,
    .returnType = returnType,
  };
  privShader.instructions.push_back(instr);
  return returnType;
}

PuleGpuIr_Value puleGpuIr_opImageSampleImplicitLod(
  PuleGpuIr_Shader shader,
  PuleGpuIr_Type resultType,
  PuleGpuIr_Value sampledImage,
  PuleGpuIr_Value coordinate
) {
  PRIV_SHADER_VALUE(priv::Opcode::imageSampleImplicitLod);
  instr.opImageSampleImplicitLod = {
    .resultType = resultType,
    .image = sampledImage,
    .coordinate = coordinate,
    .returnValue = returnValue,
  };
  privShader.instructions.push_back(instr);
  return returnValue;
}

PuleGpuIr_Value puleGpuIr_opImageSampleExplicitLod(
  PuleGpuIr_Shader shader,
  PuleGpuIr_Type resultType,
  PuleGpuIr_Value sampledImage,
  PuleGpuIr_Value coordinate,
  PuleGpuIr_Value lod
) {
  PRIV_SHADER_VALUE(priv::Opcode::imageSampleExplicitLod);
  instr.opImageSampleExplicitLod = {
    .resultType = resultType,
    .image = sampledImage,
    .coordinate = coordinate,
    .lod = lod,
    .returnValue = returnValue,
  };
  privShader.instructions.push_back(instr);
  return returnValue;
}

PuleGpuIr_Value puleGpuIr_opVariable(
  PuleGpuIr_Shader shader,
  PuleGpuIr_Type type
) {
  PRIV_SHADER_VALUE(priv::Opcode::variable);
  instr.opVariable = {
    .type = type,
    .storageClass = PuleGpuIr_StorageClass_generic,
    .returnValue = returnValue,
  };
  privShader.instructions.push_back(instr);
  return returnValue;
}

PuleGpuIr_Value puleGpuIr_opVariableStorage(
  PuleGpuIr_Shader shader,
  PuleGpuIr_Type type,
  PuleGpuIr_StorageClass storageClass,
  size_t layoutIndex
) {
  PRIV_SHADER_VALUE(priv::Opcode::variable);
  instr.opVariable = {
    .type = type,
    .storageClass = storageClass,
    .returnValue = returnValue,
  };
  switch (instr.opVariable.storageClass) {
    default: break;
    case PuleGpuIr_StorageClass_input:
    case PuleGpuIr_StorageClass_output:
      privShader.entryPointInOuts.push_back({
        .value = returnValue,
        .storageClass = storageClass,
        .layoutIndex = layoutIndex,
      });
    break;
  }
  privShader.instructions.push_back(instr);
  return returnValue;
};

PuleGpuIr_Value puleGpuIr_opLoad(
  PuleGpuIr_Shader shader,
  PuleGpuIr_Type resultType,
  PuleGpuIr_Value pointer
) {
  PRIV_SHADER_VALUE(priv::Opcode::load);
  instr.opLoad = {
    .resultType = resultType,
    .pointer = pointer,
    .returnValue = returnValue,
  };
  privShader.instructions.push_back(instr);
  return returnValue;
}

void puleGpuIr_opStore(
  PuleGpuIr_Shader shader,
  PuleGpuIr_Value pointer,
  PuleGpuIr_Value value
) {
  PRIV_SHADER(priv::Opcode::store);
  instr.opStore = {
    .pointer = pointer,
    .value = value,
  };
  privShader.instructions.push_back(instr);
}

PuleGpuIr_Value puleGpuIr_opConstant(
  PuleGpuIr_Shader shader,
  PuleGpuIr_Type type,
  PuleGpuIr_ConstantType constantType,
  PuleGpuIr_Constant constant
) {
  PRIV_SHADER_VALUE(priv::Opcode::constant);
  instr.opConstant = {
    .type = type,
    .constantType = constantType,
    .constant = constant,
    .returnValue = returnValue,
  };
  privShader.instructions.push_back(instr);
  return returnValue;
}

PuleGpuIr_Value puleGpuIr_opLogicalNot(
  PuleGpuIr_Shader shader,
  PuleGpuIr_Value value
) {
  PRIV_SHADER_VALUE(priv::Opcode::logicalNot);
  instr.opLogicalNot = {
    .value = value,
    .returnValue = returnValue,
  };
  privShader.instructions.push_back(instr);
  return returnValue;
}

PuleGpuIr_Value puleGpuIr_opLogicalAnd(
  PuleGpuIr_Shader shader,
  PuleGpuIr_Value value1,
  PuleGpuIr_Value value2
) {
  PRIV_SHADER_VALUE(priv::Opcode::logicalAnd);
  instr.opLogicalAnd = {
    .value1 = value1,
    .value2 = value2,
    .returnValue = returnValue,
  };
  privShader.instructions.push_back(instr);
  return returnValue;
}

PuleGpuIr_Value puleGpuIr_opLogicalOr(
  PuleGpuIr_Shader shader,
  PuleGpuIr_Value value1,
  PuleGpuIr_Value value2
) {
  PRIV_SHADER_VALUE(priv::Opcode::logicalOr);
  instr.opLogicalOr = {
    .value1 = value1,
    .value2 = value2,
    .returnValue = returnValue,
  };
  privShader.instructions.push_back(instr);
  return returnValue;
}

PuleGpuIr_Value puleGpuIr_opLogicalEq(
  PuleGpuIr_Shader shader,
  PuleGpuIr_Value value1,
  PuleGpuIr_Value value2
) {
  PRIV_SHADER_VALUE(priv::Opcode::logicalEq);
  instr.opLogicalEq = {
    .value1 = value1,
    .value2 = value2,
    .returnValue = returnValue,
  };
  privShader.instructions.push_back(instr);
  return returnValue;
}

PuleGpuIr_Value puleGpuIr_opLogicalNotEq(
  PuleGpuIr_Shader shader,
  PuleGpuIr_Value value1,
  PuleGpuIr_Value value2
) {
  PRIV_SHADER_VALUE(priv::Opcode::logicalNotEq);
  instr.opLogicalNotEq = {
    .value1 = value1,
    .value2 = value2,
    .returnValue = returnValue,
  };
  privShader.instructions.push_back(instr);
  return returnValue;
}

PuleGpuIr_Value puleGpuIr_opSelect(
  PuleGpuIr_Shader shader,
  PuleGpuIr_Value condition,
  PuleGpuIr_Value valueTrue,
  PuleGpuIr_Value valueFalse
) {
  PRIV_SHADER_VALUE(priv::Opcode::select);
  instr.opSelect = {
    .condition = condition,
    .valueTrue = valueTrue,
    .valueFalse = valueFalse,
    .returnValue = returnValue,
  };
  privShader.instructions.push_back(instr);
  return returnValue;
}

PuleGpuIr_Value puleGpuIr_opIntEq(
  PuleGpuIr_Shader shader,
  PuleGpuIr_Value value1,
  PuleGpuIr_Value value2
) {
  PRIV_SHADER_VALUE(priv::Opcode::intEq);
  instr.opIntEq = {
    .value1 = value1,
    .value2 = value2,
    .returnValue = returnValue,
  };
  privShader.instructions.push_back(instr);
  return returnValue;
}

PuleGpuIr_Value puleGpuIr_opIntNotEq(
  PuleGpuIr_Shader shader,
  PuleGpuIr_Value value1,
  PuleGpuIr_Value value2
) {
  PRIV_SHADER_VALUE(priv::Opcode::intNotEq);
  instr.opIntNotEq = {
    .value1 = value1,
    .value2 = value2,
    .returnValue = returnValue,
  };
  privShader.instructions.push_back(instr);
  return returnValue;
}

PuleGpuIr_Value puleGpuIr_opIntGreaterThan(
  PuleGpuIr_Shader shader,
  bool isSigned,
  PuleGpuIr_Value value1,
  PuleGpuIr_Value value2
) {
  PRIV_SHADER_VALUE(priv::Opcode::intGreaterThan);
  instr.opIntGreaterThan = {
    .isSigned = isSigned,
    .value1 = value1,
    .value2 = value2,
    .returnValue = returnValue,
  };
  privShader.instructions.push_back(instr);
  return returnValue;
}

PuleGpuIr_Value puleGpuIr_opIntGreaterThanEq(
  PuleGpuIr_Shader shader,
  bool isSigned,
  PuleGpuIr_Value value1,
  PuleGpuIr_Value value2
) {
  PRIV_SHADER_VALUE(priv::Opcode::intGreaterThanEq);
  instr.opIntGreaterThanEq = {
    .isSigned = isSigned,
    .value1 = value1,
    .value2 = value2,
    .returnValue = returnValue,
  };
  privShader.instructions.push_back(instr);
  return returnValue;
}

PuleGpuIr_Value puleGpuIr_opIntLessThan(
  PuleGpuIr_Shader shader,
  bool isSigned,
  PuleGpuIr_Value value1,
  PuleGpuIr_Value value2
) {
  PRIV_SHADER_VALUE(priv::Opcode::intLessThan);
  instr.opIntLessThan = {
    .isSigned = isSigned,
    .value1 = value1,
    .value2 = value2,
    .returnValue = returnValue,
  };
  privShader.instructions.push_back(instr);
  return returnValue;
}

PuleGpuIr_Value puleGpuIr_opIntLessThanEq(
  PuleGpuIr_Shader shader,
  bool isSigned,
  PuleGpuIr_Value value1,
  PuleGpuIr_Value value2
) {
  PRIV_SHADER_VALUE(priv::Opcode::intLessThanEq);
  instr.opIntLessThanEq = {
    .isSigned = isSigned,
    .value1 = value1,
    .value2 = value2,
    .returnValue = returnValue,
  };
  privShader.instructions.push_back(instr);
  return returnValue;
}

PuleGpuIr_Value puleGpuIr_opFloatEq(
  PuleGpuIr_Shader shader,
  PuleGpuIr_Value value1,
  PuleGpuIr_Value value2
) {
  PRIV_SHADER_VALUE(priv::Opcode::floatEq);
  instr.opFloatEq = {
    .value1 = value1,
    .value2 = value2,
    .returnValue = returnValue,
  };
  privShader.instructions.push_back(instr);
  return returnValue;
}

PuleGpuIr_Value puleGpuIr_opFloatNotEq(
  PuleGpuIr_Shader shader,
  PuleGpuIr_Value value1,
  PuleGpuIr_Value value2
) {
  PRIV_SHADER_VALUE(priv::Opcode::floatNotEq);
  instr.opFloatNotEq = {
    .value1 = value1,
    .value2 = value2,
    .returnValue = returnValue,
  };
  privShader.instructions.push_back(instr);
  return returnValue;
}

PuleGpuIr_Value puleGpuIr_opFloatGreaterThan(
  PuleGpuIr_Shader shader,
  PuleGpuIr_Value value1,
  PuleGpuIr_Value value2
) {
  PRIV_SHADER_VALUE(priv::Opcode::floatGreaterThan);
  instr.opFloatGreaterThan = {
    .value1 = value1,
    .value2 = value2,
    .returnValue = returnValue,
  };
  privShader.instructions.push_back(instr);
  return returnValue;
}

PuleGpuIr_Value puleGpuIr_opFloatGreaterThanEq(
  PuleGpuIr_Shader shader,
  PuleGpuIr_Value value1,
  PuleGpuIr_Value value2
) {
  PRIV_SHADER_VALUE(priv::Opcode::floatGreaterThanEq);
  instr.opFloatGreaterThanEq = {
    .value1 = value1,
    .value2 = value2,
    .returnValue = returnValue,
  };
  privShader.instructions.push_back(instr);
  return returnValue;
}

PuleGpuIr_Value puleGpuIr_opFloatLessThan(
  PuleGpuIr_Shader shader,
  PuleGpuIr_Value value1,
  PuleGpuIr_Value value2
) {
  PRIV_SHADER_VALUE(priv::Opcode::floatLessThan);
  instr.opFloatLessThan = {
    .value1 = value1,
    .value2 = value2,
    .returnValue = returnValue,
  };
  privShader.instructions.push_back(instr);
  return returnValue;
}

PuleGpuIr_Value puleGpuIr_opFloatLessThanEq(
  PuleGpuIr_Shader shader,
  PuleGpuIr_Value value1,
  PuleGpuIr_Value value2
) {
  PRIV_SHADER_VALUE(priv::Opcode::floatLessThanEq);
  instr.opFloatLessThanEq = {
    .value1 = value1,
    .value2 = value2,
    .returnValue = returnValue,
  };
  privShader.instructions.push_back(instr);
  return returnValue;
}

PuleGpuIr_Value puleGpuIr_opTranspose(
  PuleGpuIr_Shader shader,
  PuleGpuIr_Value value
) {
  PRIV_SHADER_VALUE(priv::Opcode::transpose);
  instr.opTranspose = {
    .value = value,
    .returnValue = returnValue,
  };
  privShader.instructions.push_back(instr);
  return returnValue;
}

PuleGpuIr_Value puleGpuIr_opIntNegate(
  PuleGpuIr_Shader shader,
  PuleGpuIr_Value value
) {
  PRIV_SHADER_VALUE(priv::Opcode::intNegate);
  instr.opIntNegate = {
    .value = value,
    .returnValue = returnValue,
  };
  privShader.instructions.push_back(instr);
  return returnValue;
}

PuleGpuIr_Value puleGpuIr_opFloatNegate(
  PuleGpuIr_Shader shader,
  PuleGpuIr_Value value
) {
  PRIV_SHADER_VALUE(priv::Opcode::floatNegate);
  instr.opFloatNegate = {
    .value = value,
    .returnValue = returnValue,
  };
  privShader.instructions.push_back(instr);
  return returnValue;
}

PuleGpuIr_Value puleGpuIr_opIntAdd(
  PuleGpuIr_Shader shader,
  PuleGpuIr_Value value1,
  PuleGpuIr_Value value2
) {
  PRIV_SHADER_VALUE(priv::Opcode::intAdd);
  instr.opIntAdd = {
    .value1 = value1,
    .value2 = value2,
    .returnValue = returnValue,
  };
  privShader.instructions.push_back(instr);
  return returnValue;
}

PuleGpuIr_Value puleGpuIr_opIntSubtract(
  PuleGpuIr_Shader shader,
  PuleGpuIr_Value value1,
  PuleGpuIr_Value value2
) {
  PRIV_SHADER_VALUE(priv::Opcode::intSubtract);
  instr.opIntSubtract = {
    .value1 = value1,
    .value2 = value2,
    .returnValue = returnValue,
  };
  privShader.instructions.push_back(instr);
  return returnValue;
}

PuleGpuIr_Value puleGpuIr_opIntMultiply(
  PuleGpuIr_Shader shader,
  PuleGpuIr_Value value1,
  PuleGpuIr_Value value2
) {
  PRIV_SHADER_VALUE(priv::Opcode::intMultiply);
  instr.opIntMultiply = {
    .value1 = value1,
    .value2 = value2,
    .returnValue = returnValue,
  };
  privShader.instructions.push_back(instr);
  return returnValue;
}

PuleGpuIr_Value puleGpuIr_opIntDivide(
  PuleGpuIr_Shader shader,
  bool isSigned,
  PuleGpuIr_Value value1,
  PuleGpuIr_Value value2
) {
  PRIV_SHADER_VALUE(priv::Opcode::intDivide);
  instr.opIntDivide = {
    .isSigned = isSigned,
    .value1 = value1,
    .value2 = value2,
    .returnValue = returnValue,
  };
  privShader.instructions.push_back(instr);
  return returnValue;
}

PuleGpuIr_Value puleGpuIr_opIntModulo(
  PuleGpuIr_Shader shader,
  bool isSigned,
  PuleGpuIr_Value value1,
  PuleGpuIr_Value value2
) {
  PRIV_SHADER_VALUE(priv::Opcode::intModulo);
  instr.opIntModulo = {
    .isSigned = isSigned,
    .value1 = value1,
    .value2 = value2,
    .returnValue = returnValue,
  };
  privShader.instructions.push_back(instr);
  return returnValue;
}

PuleGpuIr_Value puleGpuIr_opFloatAdd(
  PuleGpuIr_Shader shader,
  PuleGpuIr_Value value1,
  PuleGpuIr_Value value2
) {
  PRIV_SHADER_VALUE(priv::Opcode::floatAdd);
  instr.opFloatAdd = {
    .value1 = value1,
    .value2 = value2,
    .returnValue = returnValue,
  };
  privShader.instructions.push_back(instr);
  return returnValue;
}

PuleGpuIr_Value puleGpuIr_opFloatSubtract(
  PuleGpuIr_Shader shader,
  PuleGpuIr_Value value1,
  PuleGpuIr_Value value2
) {
  PRIV_SHADER_VALUE(priv::Opcode::floatSubtract);
  instr.opFloatSubtract = {
    .value1 = value1,
    .value2 = value2,
    .returnValue = returnValue,
  };
  privShader.instructions.push_back(instr);
  return returnValue;
}

PuleGpuIr_Value puleGpuIr_opFloatMultiply(
  PuleGpuIr_Shader shader,
  PuleGpuIr_Value value1,
  PuleGpuIr_Value value2
) {
  PRIV_SHADER_VALUE(priv::Opcode::floatMultiply);
  instr.opFloatMultiply = {
    .value1 = value1,
    .value2 = value2,
    .returnValue = returnValue,
  };
  privShader.instructions.push_back(instr);
  return returnValue;
}

PuleGpuIr_Value puleGpuIr_opFloatDivide(
  PuleGpuIr_Shader shader,
  PuleGpuIr_Value value1,
  PuleGpuIr_Value value2
) {
  PRIV_SHADER_VALUE(priv::Opcode::floatDivide);
  instr.opFloatDivide = {
    .value1 = value1,
    .value2 = value2,
    .returnValue = returnValue,
  };
  privShader.instructions.push_back(instr);
  return returnValue;
}

PuleGpuIr_Value puleGpuIr_opFloatRem(
  PuleGpuIr_Shader shader,
  PuleGpuIr_Value value1,
  PuleGpuIr_Value value2
) {
  PRIV_SHADER_VALUE(priv::Opcode::floatRem);
  instr.opFloatRem = {
    .value1 = value1,
    .value2 = value2,
    .returnValue = returnValue,
  };
  privShader.instructions.push_back(instr);
  return returnValue;
}

PuleGpuIr_Value puleGpuIr_opVectorMulScalar(
  PuleGpuIr_Shader shader,
  PuleGpuIr_Value vector,
  PuleGpuIr_Value scalar
) {
  PRIV_SHADER_VALUE(priv::Opcode::vectorMulScalar);
  instr.opVectorMulScalar = {
    .vector = vector,
    .scalar = scalar,
    .returnValue = returnValue,
  };
  privShader.instructions.push_back(instr);
  return returnValue;
}

PuleGpuIr_Value puleGpuIr_opMatrixMulScalar(
  PuleGpuIr_Shader shader,
  PuleGpuIr_Value matrix,
  PuleGpuIr_Value scalar
) {
  PRIV_SHADER_VALUE(priv::Opcode::matrixMulScalar);
  instr.opMatrixMulScalar = {
    .matrix = matrix,
    .scalar = scalar,
    .returnValue = returnValue,
  };
  privShader.instructions.push_back(instr);
  return returnValue;
}

PuleGpuIr_Value puleGpuIr_opVectorMulMatrix(
  PuleGpuIr_Shader shader,
  PuleGpuIr_Value vector,
  PuleGpuIr_Value matrix
) {
  PRIV_SHADER_VALUE(priv::Opcode::vectorMulMatrix);
  instr.opVectorMulMatrix = {
    .vector = vector,
    .matrix = matrix,
    .returnValue = returnValue,
  };
  privShader.instructions.push_back(instr);
  return returnValue;
}

PuleGpuIr_Value puleGpuIr_opMatrixMulVector(
  PuleGpuIr_Shader shader,
  PuleGpuIr_Value matrix,
  PuleGpuIr_Value vector
) {
  PRIV_SHADER_VALUE(priv::Opcode::matrixMulVector);
  instr.opMatrixMulVector = {
    .matrix = matrix,
    .vector = vector,
    .returnValue = returnValue,
  };
  privShader.instructions.push_back(instr);
  return returnValue;
}

PuleGpuIr_Value puleGpuIr_opMatrixMulMatrix(
  PuleGpuIr_Shader shader,
  PuleGpuIr_Value matrix1,
  PuleGpuIr_Value matrix2
) {
  PRIV_SHADER_VALUE(priv::Opcode::matrixMulMatrix);
  instr.opMatrixMulMatrix = {
    .matrix1 = matrix1,
    .matrix2 = matrix2,
    .returnValue = returnValue,
  };
  privShader.instructions.push_back(instr);
  return returnValue;
}

PuleGpuIr_Value puleGpuIr_opVectorOuterProduct(
  PuleGpuIr_Shader shader,
  PuleGpuIr_Value vector1,
  PuleGpuIr_Value vector2
) {
  PRIV_SHADER_VALUE(priv::Opcode::vectorOuterProduct);
  instr.opVectorOuterProduct = {
    .vector1 = vector1,
    .vector2 = vector2,
    .returnValue = returnValue,
  };
  privShader.instructions.push_back(instr);
  return returnValue;
}

PuleGpuIr_Value puleGpuIr_opVectorDotProduct(
  PuleGpuIr_Shader shader,
  PuleGpuIr_Value vector1,
  PuleGpuIr_Value vector2
) {
  PRIV_SHADER_VALUE(priv::Opcode::vectorDotProduct);
  instr.opVectorDotProduct = {
    .vector1 = vector1,
    .vector2 = vector2,
    .returnValue = returnValue,
  };
  privShader.instructions.push_back(instr);
  return returnValue;
}

PuleGpuIr_Value puleGpuIr_opConvertSignedToFloat(
  PuleGpuIr_Shader shader,
  PuleGpuIr_Type type,
  PuleGpuIr_Value value
) {
  PRIV_SHADER_VALUE(priv::Opcode::convertSignedToFloat);
  instr.opConvertSignedToFloat = {
    .type = type,
    .value = value,
    .returnValue = returnValue,
  };
  privShader.instructions.push_back(instr);
  return returnValue;
}

void puleGpuIr_opBranchJmp(
  PuleGpuIr_Shader shader,
  PuleGpuIr_Value target
) {
  PRIV_SHADER(priv::Opcode::branchJmp);
  instr.opBranchJmp = {
    .target = target,
  };
  privShader.instructions.push_back(instr);
}

void puleGpuIr_opBranchCond(
  PuleGpuIr_Shader shader,
  PuleGpuIr_Value condition,
  PuleGpuIr_Value targetTrue,
  PuleGpuIr_Value targetFalse
) {
  PRIV_SHADER(priv::Opcode::branchCond);
  instr.opBranchCond = {
    .condition = condition,
    .targetTrue = targetTrue,
    .targetFalse = targetFalse,
  };
  privShader.instructions.push_back(instr);
}

void puleGpuIr_opReturn(
  PuleGpuIr_Shader shader
) {
  PRIV_SHADER(priv::Opcode::returnVoid);
  instr.opReturnVoid = { };
  privShader.instructions.push_back(instr);
}

void puleGpuIr_opReturnValue(
  PuleGpuIr_Shader shader,
  PuleGpuIr_Value value
) {
  PRIV_SHADER(priv::Opcode::returnValue);
  instr.opReturnValue = {
    .value = value,
  };
  privShader.instructions.push_back(instr);
}

PuleGpuIr_Value puleGpuIr_opAccessChain(
  PuleGpuIr_Shader shader,
  PuleGpuIr_Value base,
  PuleGpuIr_Value * indices,
  size_t indexLen
) {
  PRIV_SHADER_VALUE(priv::Opcode::accessChain);
  instr.opAccessChain = {
    .base = base,
    .indices = (
      puleBufferCopyWithData(
        puleAllocateDefault(),
        (uint8_t const *)indices,
        indexLen * sizeof(PuleGpuIr_Value)
      )
    ),
    .returnValue = returnValue,
  };
  privShader.instructions.push_back(instr);
  return returnValue;
}

PuleGpuIr_Value puleGpuIr_opCompositeExtract(
  PuleGpuIr_Shader shader,
  PuleGpuIr_Type type,
  PuleGpuIr_Value value,
  size_t index
) {
  PRIV_SHADER_VALUE(priv::Opcode::compositeExtract);
  instr.opCompositeExtract = {
    .type = type,
    .value = value,
    .index = index,
    .returnValue = returnValue,
  };
  privShader.instructions.push_back(instr);
  return returnValue;
}

PuleGpuIr_Value puleGpuIr_opCompositeConstruct(
  PuleGpuIr_Shader shader,
  PuleGpuIr_Type type,
  PuleGpuIr_Value const * constituents,
  size_t constituentLen
) {
  PRIV_SHADER_VALUE(priv::Opcode::compositeConstruct);
  instr.opCompositeConstruct = {
    .type = type,
    .constituents = (
      puleBufferCopyWithData(
        puleAllocateDefault(),
        (uint8_t const *)constituents,
        constituentLen * sizeof(PuleGpuIr_Value)
      )
    ),
    .returnValue = returnValue,
  };
  privShader.instructions.push_back(instr);
  return returnValue;
}

PuleGpuIr_Value puleGpuIr_opVectorShuffle(
  PuleGpuIr_Shader shader,
  PuleGpuIr_Type type,
  PuleGpuIr_Value vec0,
  PuleGpuIr_Value vec1,
  uint32_t const * components,
  size_t componentLen
) {
  PRIV_SHADER_VALUE(priv::Opcode::vectorShuffle);
  instr.opVectorShuffle = {
    .type = type,
    .vec0 = vec0,
    .vec1 = vec1,
    .components = (
      puleBufferCopyWithData(
        puleAllocateDefault(),
        (uint8_t const *)components,
        componentLen * sizeof(uint32_t)
      )
    ),
    .returnValue = returnValue,
  };
  privShader.instructions.push_back(instr);
  return returnValue;
}

PuleGpuIr_Value puleGpuIr_opExtInst(
  PuleGpuIr_Shader shader,
  PuleGpuIr_Type type,
  PuleStringView instruction,
  PuleGpuIr_Value const * operands,
  size_t operandLen
) {
  PRIV_SHADER_VALUE(priv::Opcode::extInst);
  instr.opExtInst = {
    .type = type,
    .instruction = puleStringCopy(puleAllocateDefault(), instruction),
    .operands = (
      puleBufferCopyWithData(
        puleAllocateDefault(),
        (uint8_t const *)operands,
        operandLen * sizeof(PuleGpuIr_Value)
      )
    ),
    .returnValue = returnValue,
  };
  privShader.instructions.push_back(instr);
  return returnValue;
}

PuleGpuIr_Value puleGpuIr_opFunctionCall(
  PuleGpuIr_Shader shader,
  PuleGpuIr_Type fnType,
  PuleGpuIr_Value fn,
  PuleGpuIr_Value const * arguments,
  size_t argumentLen
) {
  PRIV_SHADER_VALUE(priv::Opcode::functionCall);
  instr.opFunctionCall = {
    .fnType = fnType,
    .fn = fn,
    .arguments = (
      puleBufferCopyWithData(
        puleAllocateDefault(),
        (uint8_t const *)arguments,
        argumentLen * sizeof(PuleGpuIr_Value)
      )
    ),
    .returnValue = returnValue,
  };
  privShader.instructions.push_back(instr);
  return returnValue;
}

PuleGpuIr_Value puleGpuIr_opLabel(
  PuleGpuIr_Shader shader
) {
  PRIV_SHADER_VALUE(priv::Opcode::label);
  instr.opLabel = {
    .target = returnValue
  };
  privShader.instructions.push_back(instr);
  return returnValue;
}

PuleGpuIr_Value puleGpuIr_opFunction(
  PuleGpuIr_Shader shader,
  PuleGpuIr_Type returnType,
  PuleGpuIr_FunctionControl functionControl,
  PuleGpuIr_Type fnType,
  PuleStringView functionLabel
) {
  PRIV_SHADER_VALUE(priv::Opcode::function);
  puleLogDev("fn type: %d", fnType.id);
  instr.opFunction = {
    .fnReturnType = returnType,
    .functionControl = functionControl,
    .fnType = fnType,
    .returnValue = returnValue,
    .label = puleStringCopy(puleAllocateDefault(), functionLabel),
  };
  privShader.instructions.push_back(instr);
  return returnValue;
}

void puleGpuIr_opFunctionEnd(PuleGpuIr_Shader shader) {
  PRIV_SHADER(priv::Opcode::functionEnd);
  instr.opFunction = { };
  privShader.instructions.push_back(instr);
}

void puleGpuIr_opEntryPoint(
  PuleGpuIr_Shader shader,
  PuleGpuIr_Value functionEntry,
  PuleGpuIr_Value const * globals,
  size_t globalLen
) {
  auto & shaderPriv = *priv::shaders.at(shader);
  shaderPriv.entryPoint = {
    .entryPointFn = functionEntry,
    .globals = std::vector<PuleGpuIr_Value>(globals, globals + globalLen),
  };
}

void puleGpuIr_OpDecorate(
  PuleGpuIr_Shader shader,
  PuleGpuIr_Type type,
  uint32_t const * values,
  size_t valueLen
) {
  auto & shaderPriv = *priv::shaders.at(shader);
  shaderPriv.decorations.push_back(priv::Shader::Decoration {
    .type = type,
    .memberIndex = -1,
    .values = std::vector<uint32_t>(values, values + valueLen),
  });
}

void puleGpuIr_opDecorateMember(
  PuleGpuIr_Shader shader,
  PuleGpuIr_Type type,
  int64_t memberIndex,
  uint32_t const * values,
  size_t valueLen
) {
  auto & shaderPriv = *priv::shaders.at(shader);
  shaderPriv.decorations.push_back(priv::Shader::Decoration {
    .type = type,
    .memberIndex = memberIndex,
    .values = std::vector<uint32_t>(values, values + valueLen),
  });
}

} // extern "C"

#undef PRIV_SHADER_VALUE
#undef PRIV_SHADER
