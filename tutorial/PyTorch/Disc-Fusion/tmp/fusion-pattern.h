// Basic information of fused operation set, including op-list, operands,
// results, et.al. It does not provide informations for codegen schedules, like
// dominant-op, subroot, fusion-type, et.al.
class FusionPatternBase {
 public:
  using FusionOpList = SmallVector<Operation*, 4>;
  using FusionValueList = SmallVector<Value, 4>;

  // Create a new fusion pattern from a single op.
  explicit FusionPatternBase(Operation* op);

  // Create a new fusion pattern from the ops inside the lmhlo fusion op.
  explicit FusionPatternBase(lmhlo::FusionOp op);

  // Create a new fusion pattern with the ops inside the list.
  explicit FusionPatternBase(SmallVectorImpl<Operation*>& op_list);

  // Returns the op list this fusion pattern represents.
  FusionOpList& getOpList() { return op_list_; }

  // Returns values that are consumed by the lmhlo ops inside the fusion
  // pattern.
  FusionValueList& getOperands() { return operands_; }

  // Returns values that are outputs of any lmhlo op in the fused pattern and
  // have consumers outside the fusion pattern.
  FusionValueList& getResults() { return results_; }

  // Returns values that are outputs of any lmhlo op in the fused pattern and
  // have consumers outside the fusion pattern.
  SmallVector<Operation*, 4>& getRootOps() { return root_ops_; }

  // Returns values that are outputs of any lmhlo op in the fused pattern and
  // are only consumed by the lmhlo ops inside the fused pattern.
  FusionValueList& getInternalResults() { return internal_results_; }

  // Returns values that are outputs of any lmhlo op in the fused pattern and
  // are only consumed by the lmhlo ops outside the fused pattern.
  FusionValueList& getExternalOnlyResults() { return external_only_results_; }

  // Return last writer map of ops in the fusion pattern.
  DenseMap<Value, Operation*>& getLastWriter() { return last_writer_; }

  // Returns the size of the ops this fusion pattern contains.
  int size() { return op_list_.size(); }

  // Returns the effective size (e.g. not counting const ops) of the ops this
  // fusion pattern contains.
  int effectiveSize();

  // Sorts the ops inside the fusion pattern according to the keys provided.
  void sortFusionOpListBy(DenseMap<Operation*, int>& op_to_idx);

  void sortFusionOpListWithTopologyOrder();

  // Here `value` is supposed to be a pointer to buffer.
  // Returns the defining op of `value `if no known op updates the buffer,
  // otherwise returns the last op that updates the buffer pointed by the
  // `value`.
  Operation* findLastWriter(Value value) {
    auto it = last_writer_.find(value);
    if (it != last_writer_.end()) {
      return it->second;
    }
    return value.getDefiningOp();
  }

  void updateLastWriter(Value value, Operation* op) {
    last_writer_[value] = op;
  }

  bool alreadyInRootOps(Operation* new_op) {
    for (Operation* op : root_ops_) {
      if (new_op == op) {
        return true;
      }
    }
    return false;
  }

 protected:
  // Calculates the inputs and outputs of the fusion pattern.
  void calculateOperandsAndResults();

  FusionOpList op_list_;
  FusionValueList operands_;
  FusionValueList results_;
  FusionValueList internal_results_;
  FusionValueList external_only_results_;
  SmallVector<Operation*, 4> root_ops_;
  DenseMap<Value, Operation*> last_writer_;
};

// Represents a list of lmhlo ops that are going to be fused.
// Concepts for a fusion pattern:
//   - Root op: the op whose output is the fusion-pattern's output.
//   - Sub-root op: the op whose output is to be maintained on shared-memory for
//     kStitch fusion. Currently, we only support row-reduction to be a sub-root
//     op.
//   - Regular xroot op: either a root op or a sub-root op, for whose operands
//     we successfully build tile information during kStitch fusion-pattern init
//     phase.
//   - Irregular xroot op: an root op for whose operands we fail to build tile
//     information durint kStitch fusion-pattern init phase.
//   - Skeleton op: the op who will be used to build the loop skeleton when
//     lowering a kStitch fusion to parallel loops. Currently, sub-root ops, and
//     regular xroot ops who generate external only results, are skeleton ops.
//     Other xroot ops are lowered with input-inline fusion phase.
//   Note: for an regular xroot op which is not an skeleton op, the output data
//     to be written should be coverred by its corresponding skeleton op.
//     Otherwise, this xroot are regared as irregular.
class FusionPattern : public FusionPatternBase {
 public:
  // Create a new fusion pattern from a single op.
  explicit FusionPattern(Operation* op);

  // Create a new fusion pattern from the ops inside the lmhlo fusion op.
  explicit FusionPattern(lmhlo::FusionOp op, ShapeAnalysis* shape_analysis);

  // Do not allow to build a fusion pattern with only FusionOp.
  explicit FusionPattern(lmhlo::FusionOp op) = delete;

  // Returns the dominant op of this fusion pattern.
  // For kLoop fusion, a dominant op may be any op that has external users.
  // For kInput fusion, a dominant op may be a row reduction (if exists), or
  // a column reduction op.
  Operation* getDominantOp() { return dominant_op_; }

  // Sets the dominant op to the op provided.
  void setDominantOp(Operation* op) { dominant_op_ = op; }

  // Returns the fusion kind of the fusion pattern.
  FusionType getFusionType() { return fusion_type_; }

  // Returns the fusion kind of the fusion pattern.
  StringRef getFusionTypeStr() { return fusionTypeToString(fusion_type_); }

  // Sets the fusion type to the the type provided.
  void setFusionType(FusionType type) { fusion_type_ = type; }

  // Returns true if this a fusible fusion pattern.
  bool isFusible() { return getFusionType() != FusionType::kNone; }

  // Returns true if this fusion pattern is a kLoop fusion.
  bool isKLoopFusion() { return getFusionType() == FusionType::kLoop; }

  // Returns true if this fusion pattern is a kInput fusion.
  bool isKInputFusion() {
    return (getFusionType() == FusionType::kRowReduction ||
            getFusionType() == FusionType::kColReduction);
  }

  // Returns true if the fusion type is stitch fusion.
  bool isStitchFusion() { return getFusionType() == FusionType::kStitch; }

  // Returns true if the fusion type is transform-based fusion.
  bool isTransformBasedFusion() {
    return getFusionType() == FusionType::kTransform;
  }

  // Merges two fusion patterns and returns the merged pattern. The original
  // pattern remains unmodified. The new merged pattern is uninitialized.
  FusionPattern mergeWithoutInit(FusionPattern& other);

  // Create a new fusion pattern with the given op list, without init.
  static FusionPattern createWithoutInit(SmallVectorImpl<Operation*>& op_list);

  DenseMap<Value, TileInfo>& getTilePlan() { return tile_plan_; }
  void setTilePlan(const DenseMap<Value, TileInfo>& tile_plan) {
    tile_plan_ = tile_plan;
  }

  SmallVector<Operation*, 4>& getSubRootOps() { return sub_root_ops_; }

  void setSubRootOps(const SmallVector<Operation*, 4>& sub_root_ops) {
    sub_root_ops_ = sub_root_ops;
  }

  struct SkeletonGroup {
    SmallVector<Operation*> skeletons;
    SmallVector<Operation*> root_member_list;
    // An irregular member means whose non-tiled dims are not exactly matched
    // with skeleton. This requires special designe for GPU block mapping when
    // generating the code.
    DenseSet<Operation*> irregular_root_member_set;
  };

  void findOpsOfSkeletonGroup(
      SkeletonGroup group, DenseSet<Operation*>& ops,
      DenseSet<Operation*>& shmem_cached_ops,
      const DenseMap<Operation*, SmallVector<Operation*>>& existing_group_ops,
      int row_per_block, int& shmem_usage_bits, const int shmem_limit_bits);

  int64_t getCollapsedTileDim(Value value);

  DenseSet<Operation*>& getRegularXroots() { return regular_xroots_; }

  DenseSet<Operation*>& getIrregularXroots() { return irregular_xroots_; }

 private:
  // Create a new fusion pattern with the ops inside the list.
  explicit FusionPattern(SmallVectorImpl<Operation*>& op_list);

  Operation* dominant_op_ = nullptr;
  FusionType fusion_type_ = FusionType::kNone;
  SmallVector<Operation*, 4> sub_root_ops_;
  DenseMap<Value, TileInfo> tile_plan_;
  // An xroot op is either a root or a sub-root op. Regular xroots are those
  // whose element-number of non-tileds dimes are the same with sub-root ops.
  // Otherwise an xroot is irregular.
  DenseSet<Operation*> regular_xroots_;
  DenseSet<Operation*> irregular_xroots_;
};