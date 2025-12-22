// TODO(leon): Key part of this lhlo fusion strategy 
// A fusion planner that can propose a fusion plan for a block of ops.
// The fusion plan is consisted of a group of fusion patterns.
//
// Currently all proposed patterns followed xla kLoop/kInput like fusion
// templates while are adapted to the fully dynamic shape world.
//
// kLoop fusion template satisfies:
//   - all ops in the fusion pattern are element-wise.
//   - all the shapes of outputs of fusion pattern are same or have same number
//   of elements, and thus can fit into a same parallel loop.
//
// kInput fusion template satisfies:
//   - any op in the fusion pattern is either element-wise or a reduction.
//   - if a op is a reduction, its output cannot be consumed by other
//     ops in the same fusion pattern.
//   - all the effective shapes of outputs of fusion pattern are same.
//     - For element-wise op, its effective shape is its output shape.
//     - For reduction op, its effective shape is its operand shape.
//   - currently our downstreaming codegen engine only support 2d -> 1d tensor
//   reduction. TODO(disc): lift this limitation.
//     - 2D row reduction: out[i] = sum({in[i][j] for all j})
//     - 2D column reduction: out[j] = sum({in[i][j] for all i}
class FusionPlanner {
 public:
  explicit FusionPlanner(FusionPipeline& pipeline, Block* block,
                         ShapeAnalysis* shapeAnalysis)
      : fusionPipeline_(pipeline),
        block_(block),
        shape_analysis_(shapeAnalysis) {
    assert(!fusionPipeline_.empty());
    assert(block_ != nullptr);
    assert(shape_analysis_ != nullptr);
    currentFusionStrategy_ = fusionPipeline_[0].get();
    // Move up metadata-only ops (e.g. dim, shape_of) as far as possible.
    MoveUpMetadataOnlyOpsForFusion();

    for (Operation& op : *block) {
      op_list_.push_back(&op);
    }
    cycle_detector_.reset(new GraphCycles(op_list_.size()));
    original_graph_with_explicit_edges_.reset(new GraphCycles(op_list_.size()));
    BuildNodeMap();
  }

  void dumpCluster() {
    llvm::dbgs() << "Fusion result:\n";
    DenseSet<Cluster*> seen_clusters;
    for (Operation* op : op_list_) {
      Cluster* cluster = GetClusterForNode(op);
      if (!seen_clusters.insert(cluster).second) continue;
      FusionPattern& fusion_pattern = cluster->fused_pattern();
      llvm::dbgs() << "  Cluster #" << seen_clusters.size() << "@"
                   << fusion_pattern.getFusionTypeStr() << "\n";
      for (Operation* subOp : fusion_pattern.getOpList()) {
        llvm::dbgs() << "    " << *subOp << "\n";
      }
    }
  }

  // Returns a fusion plan if success, otherwise none.
  std::optional<FusionPlan> Run() {
    // Greedily search connected fusible pattern, and ops belonging to
    // a same fusion pattern are grouped into a cluster.
    // 贪心地应用多种fusion plan策略
    for (auto& strategy : fusionPipeline_) {
      currentFusionStrategy_ = strategy.get();
      // Re-init non-fusible fusion pattern using the given fusion strategy
      // since different fusion strategy may support different set of ops.
      // 针对不同强度的fusioon策略，为落单的operator重新初始化fusion pattern
      initFusionPatterns();
      RunEdgeContractionLoop();
      if (!RunFusionPatternFinalization()) {
        return std::nullopt;
      }
      LLVM_DEBUG(dumpCluster());
    }

    // After doing edge contraction, each unique cluster having size
    // more than one represents a potential fusion pattern.
    // We collect all these clusters and construct a fusion plan.
    FusionPlan plan;
    DenseSet<Cluster*> seen_clusters;
    for (Operation* op : op_list_) {
      Cluster* cluster = GetClusterForNode(op);
      if (!seen_clusters.insert(cluster).second) continue;
      FusionPattern& fusion_pattern = cluster->fused_pattern();
      // Make sure the ops in a fusion pattern are in topological ordering.
      fusion_pattern.sortFusionOpListBy(op_to_node_id_);
      if (!fusion_pattern.isFusible() || fusion_pattern.effectiveSize() < 1 ||
          !fusion_pattern.isTransformBasedFusion() &&
              fusion_pattern.effectiveSize() == 1) {
        continue;
      }
      plan.emplace_back(fusion_pattern);
    }

    // Re-order ops inside the blocks to make sure all producers are placed
    // before its consumers after fusion.
    ReorderOperationsInsideBlock();
    return plan;
  }

  // Returns the op_list this planner operates on.
  const SmallVectorImpl<Operation*>& op_list() const { return op_list_; }

  FusionStrategy& getFusionStrategy() { return *currentFusionStrategy_; }

 private:
  // Represent a (partial) fused pattern
  class Cluster {
   public:
    Cluster(int node_id, FusionPlanner* planner)
        : node_id_(node_id), pattern_(planner->op_list()[node_id]) {}

    Cluster(int node_id, FusionPattern& fusion_pattern)
        : node_id_(node_id), pattern_(fusion_pattern) {}

    // The number of nodes in this cluster.
    int cluster_size() { return pattern_.size(); }

    // The ID of the cluster as represented in `cycle_detector_`.
    int cycles_graph_node_id() const { return node_id_; }

    // Sets the ID of the cluster as represented in `cycle_detector_`.
    void set_cycles_graph_node_id(int cycles_graph_node_id) {
      node_id_ = cycles_graph_node_id;
    }

    // Currently the fused pattern this cluster holds.
    FusionPattern& fused_pattern() { return pattern_; }

   private:
    // ID of the representative node of this cluster.
    int node_id_;

    // the fused pattern this cluster holds.
    FusionPattern pattern_;
  };

 private:
  // Returns a new cluster with specified `cycles_graph_node_id`
  Cluster* MakeCluster(int cycles_graph_node_id) {
    cluster_storage_.emplace_back(new Cluster(cycles_graph_node_id, this));
    getFusionStrategy().initFusionPattern(
        *shape_analysis_, cluster_storage_.back()->fused_pattern());
    return cluster_storage_.back().get();
  }

  // init non-fusible fusion pattern with only one op using the given fusion
  // strategy since different fusion strategy may support different set of ops.
  void initFusionPatterns() {
    for (int32_t node : cycle_detector_->AllNodesInPostOrder()) {
      Cluster* cluster = GetClusterForCyclesGraphNode(node);
      FusionPattern& pattern = cluster->fused_pattern();
      if (pattern.isFusible()) continue;
      getFusionStrategy().initFusionPattern(*shape_analysis_, pattern);
    }
  }

  // Metadata ops (e.g. shapeOf, dimOp) don't change data thus we move forward
  // them as far as possible inside the same block to enable more fusion
  // opportunities.
  void MoveUpMetadataOnlyOpsForFusion() {
    SmallVector<Operation*, 4> ops;
    for (Operation& op : *block_) {
      ops.push_back(&op);
    }

    auto inBlock = [&](Operation* op, Block* block) {
      return op && op->getBlock() == block;
    };

    for (Operation* op : ops) {
      Block* block = op->getBlock();
      if (isa<shape::ShapeOfOp>(op)) {
        Operation* definingOp = op->getOperand(0).getDefiningOp();
        // Move shape_of right after its defining op if possible.
        if (!inBlock(definingOp, block)) {
          op->moveBefore(block, block->begin());
        } else {
          op->moveAfter(definingOp);
        }
      } else if (isa<memref::DimOp>(op)) {
        Operation* firstOperandOp = op->getOperand(0).getDefiningOp();
        Operation* secondOperandOp = op->getOperand(1).getDefiningOp();
        if (!inBlock(firstOperandOp, block) &&
            !inBlock(secondOperandOp, block)) {
          op->moveBefore(block, block->begin());
        } else if (!inBlock(firstOperandOp, block)) {
          op->moveAfter(secondOperandOp);
        } else if (!inBlock(secondOperandOp, block)) {
          op->moveAfter(firstOperandOp);
        } else if (firstOperandOp->isBeforeInBlock(secondOperandOp)) {
          op->moveAfter(secondOperandOp);
        } else {
          op->moveAfter(firstOperandOp);
        }
      }
    }
  }

  // Returns all the values touched by this op or its nested ops.
  SmallVector<Value, 4> GetAllPossibleUsedValues(Operation* op) {
    SmallVector<Value, 4> values;
    op->walk([&](Operation* nest_op) {
      for (Value v : nest_op->getOperands()) {
        values.push_back(v);
      }
    });
    return values;
  }

  // Builds the initial dependency graph.
  void BuildNodeMap() {
    int num_nodes = op_list_.size();
    for (int node_id = 0; node_id < num_nodes; ++node_id) {
      // 初始化节点id，以及对应的cluster
      Operation* op = op_list_[node_id];
      MakeCluster(node_id);   // 这里会
      op_to_node_id_[op] = node_id;
      leader_for_node_.insert(node_id);

      // 依据数据依赖关系，建立有向边（主要是RAW依赖）
      for (Value operand : GetAllPossibleUsedValues(op)) {
        Operation* operand_op = FindLastWriter(operand);
        // Only consider the operand_op inside the target block.
        auto iter = op_to_node_id_.find(operand_op);
        if (iter == op_to_node_id_.end()) {
          continue;
        }
        // Add an edge to connect the last writer and the current consumer.
        cycle_detector_->InsertEdge(iter->second, node_id);
        original_graph_with_explicit_edges_->InsertEdge(iter->second, node_id);
      }

      // For some ops (e.g. lmhlo ops), some operands are the output memrefs
      // Thus these operands are supposed to be updated.
      // Suppose that an op (or its nested ops) can only write the buffers
      // explicit passed in as operands of this op.
      // 更新 last_writer_ 信息
      if (isInplaceOperator(op) ||
          (op->getDialect()->getTypeID() !=
               TypeID::get<lmhlo::LmhloDialect>() &&
           op->getDialect()->getTypeID() !=
               TypeID::get<lmhlo_disc::LmhloDiscDialect>())) {
        // If an op is not in lmhlo or lmhlo_disc dialect, it may be written
        // multiple times (e.g. multiple memref.store ops for the same
        // underlying buffer).
        for (Value v : op->getOperands()) {
          if (!IsOpWriteValue(op, v)) continue;
          last_writer_[v] = op;
        }
      } else {
        // If an op is in lmhlo or lmhlo_disc dialect, output operands can only
        // be in the end of operands sequence. Note that we also check that a
        // buffer can only be written once if it's the output operand of one
        // lmhlo op.
        // lmhlo的buffer依旧遵循SSA原则，单一写入原则
        int num_input_operand = op->getNumOperands() - getNumResultOperands(op);
        for (Value v : op->getOperands().drop_front(num_input_operand)) {
          auto it = last_writer_.try_emplace(v, op);
          (void)it;
          // Currently, a buffer is only supposed to be written once (as the
          // output operand of one lmhlo op).
          assert(it.second);
        }
      }
    }
  }

  // Returns the cluster contains this op.
  Cluster* GetClusterForNode(Operation* n) {
    int id = op_to_node_id_[n];
    id = leader_for_node_.getLeaderValue(id);
    return cluster_storage_[id].get();
  }

  // Returns the cluster contains the op having `node_id`.
  Cluster* GetClusterForCyclesGraphNode(int node_id) {
    return cluster_storage_[leader_for_node_.getLeaderValue(node_id)].get();
  }

  using FnTy = llvm::function_ref<bool(Cluster*, Cluster*)>;
  bool ForEachEdgeInPostOrder(FnTy fn, bool enable_cross_fusion = false) {
    bool changed = false;
    for (int32_t node : cycle_detector_->AllNodesInPostOrder()) {
      Cluster* cluster_from = GetClusterForCyclesGraphNode(node);
      // Make a copy of the set of successors because we may modify the graph in
      // TryToContractEdge.
      std::vector<int32_t> successors_copy =
          cycle_detector_->SuccessorsCopy(cluster_from->cycles_graph_node_id());

      // 尝试融合node所在的cluster和所有后继节点所在的cluster
      for (int to : successors_copy) {
        Cluster* cluster_to = GetClusterForCyclesGraphNode(to);
        bool contracted_edge = fn(cluster_from, cluster_to);
        changed |= contracted_edge;
      }
    }

    if (!enable_cross_fusion) return changed;

    bool mem_intensive_opt_experiment = isMemIntensiveOptExperimentalEnabled();

    // To enable even more fusion opportunities (e.g. horizontal fusion)
    for (int32_t lhs : cycle_detector_->AllNodesInPostOrder()) {
      Cluster* cluster_lhs = GetClusterForCyclesGraphNode(lhs);
      if (!cluster_lhs) {
        continue;
      }

      FusionPattern& pattern_lhs = cluster_lhs->fused_pattern();
      if (mem_intensive_opt_experiment) {
        auto& lhs_ops = pattern_lhs.getOpList();
        if (lhs_ops.size() == 1 &&
            isa<lmhlo::DynamicReshapeOp, lmhlo::ReshapeOp>(lhs_ops[0])) {
          continue;
        }
      }

      if (!pattern_lhs.isFusible()) continue;

      for (int32_t rhs : cycle_detector_->AllNodesInPostOrder()) {
        Cluster* cluster_rhs = GetClusterForCyclesGraphNode(rhs);
        if (!cluster_rhs || cluster_lhs == cluster_rhs) {
          continue;
        }

        FusionPattern& pattern_rhs = cluster_rhs->fused_pattern();
        if (mem_intensive_opt_experiment) {
          auto& rhs_ops = pattern_rhs.getOpList();
          if (rhs_ops.size() == 1 &&
              isa<lmhlo::DynamicReshapeOp, lmhlo::ReshapeOp>(rhs_ops[0])) {
            continue;
          }
        }

        if (!pattern_rhs.isFusible()) continue;

        int idx_lhs = cluster_lhs->cycles_graph_node_id();
        int idx_rhs = cluster_rhs->cycles_graph_node_id();
        // Early returns if already having data dependency.
        if (cycle_detector_->IsReachable(idx_lhs, idx_rhs) ||
            cycle_detector_->IsReachable(idx_rhs, idx_lhs))
          continue;

        // insert a "virtual" edge between `idx_lhs` and `idx_rhs` in order to
        // re-use data structure for normal fusion.
        cycle_detector_->InsertEdge(idx_lhs, idx_rhs);
        original_graph_with_explicit_edges_->InsertEdge(idx_lhs, idx_rhs);
        bool contracted_edge = fn(cluster_lhs, cluster_rhs);
        // Remove the "virtual" edge if failed to contract `idx_lhs` and
        // `idx_rhs`, otherwise the virtual edge will be removed by design after
        // merging `idx_lhs` and `idx_rhs`
        if (!contracted_edge) {
          cycle_detector_->RemoveEdge(idx_lhs, idx_rhs);
          original_graph_with_explicit_edges_->RemoveEdge(idx_lhs, idx_rhs);
        }
        changed |= contracted_edge;
      }
    }

    return changed;
  }

  // This function check if fusing `from` with `to` is valid and if so perform
  // the merge. The validity is based on the operations in the clusters and
  // the compatibility of the shapes of the outputs of the would-be fused
  // clusters.
  // Returns true is the merge was performed.
  bool TryToContractEdge(Cluster* cluster_from, Cluster* cluster_to) {
    int from = cluster_from->cycles_graph_node_id();
    int to = cluster_to->cycles_graph_node_id();

    if (!cycle_detector_->CanContractEdge(from, to)) {
      // cycle detected, recover the deleted edge.
      LLVM_DEBUG(llvm::dbgs()
                 << "Could not contract " << from << " -> " << to
                 << " because contracting the edge would create a cycle.");
      return false;
    }

    // 尝试融合两个cluster对应的fusion pattern
    if (!getFusionStrategy().tryFuseInplace(*shape_analysis_,
                                            cluster_from->fused_pattern(),
                                            cluster_to->fused_pattern())) {
      return false;
    }
    auto optional_merged_node = cycle_detector_->ContractEdge(from, to);
    assert(optional_merged_node.has_value());
    cluster_from->set_cycles_graph_node_id(*optional_merged_node);

    // Merge the UnionFind Set.
    leader_for_node_.unionSets(from, to);
    return true;
  }

  // Greedily fuse connected node.
  // 两轮贪心融合，第一轮做生产者消费者融合，第二轮做水平融合。
  bool RunEdgeContractionLoop() {
    using std::placeholders::_1;
    using std::placeholders::_2;
    bool changed = false;

    // Run fusion pass repeatedly until nothing to be fused
    while (ForEachEdgeInPostOrder(
        std::bind(&FusionPlanner::TryToContractEdge, this, _1, _2), false)) {
      // empty statement by design
    }

    // Run another round of horizontal fusion if enabled.
    // Note that we do not try to merge this loop with the above loop because we
    // want to make sure the normal fusion pattern (fusion cross def-use) take
    // first. Without this design, some horizontal fusion may break following
    // normal fusion pattern.
    bool enable_horizontal_fusion = useHorizontalFusion();
    PlacementAwareFusionStrategy* placement_aware_strategy =
        reinterpret_cast<PlacementAwareFusionStrategy*>(&getFusionStrategy());
    auto strategies = placement_aware_strategy->getStrategyMap();
    if (strategies.size() == 1 &&
        strategies.find(placement_utils::kGpu) != strategies.end()) {
      enable_horizontal_fusion &= typeid(*strategies[placement_utils::kGpu]) !=
                                  typeid(DotGpuFusionStrategy);
    }
    if (enable_horizontal_fusion) {
      while (ForEachEdgeInPostOrder(
          std::bind(&FusionPlanner::TryToContractEdge, this, _1, _2), true)) {
        // empty statement by design
      }
    }
    return changed;
  }

  int32_t reContractEdges(FusionPattern& fusion_pattern,
                          DenseMap<int32_t, DenseSet<int32_t>>& producers_map,
                          GraphCycles* cycle_detector,
                          EquivalenceClasses<int32_t>& leader_for_node) {
    auto op_list = fusion_pattern.getOpList();
    if (op_list.size() == 1) {
      return op_to_node_id_[op_list[0]];
    }
    DenseSet<int> node_set;
    for (auto op : op_list) {
      int32_t node_id = op_to_node_id_[op];
      node_set.insert(node_id);
    }
    auto roots = fusion_pattern.getRootOps();
    using NodeAndLead = std::pair<int32_t, int32_t>;
    SmallVector<NodeAndLead> worklist;
    for (auto root : roots) {
      auto node_id = op_to_node_id_[root];
      worklist.emplace_back(node_id, node_id);
    }

    while (!worklist.empty()) {
      auto& curr = worklist.back();
      int32_t curr_node_id = curr.first;
      int32_t curr_lead_id = curr.second;
      worklist.pop_back();
      SmallVector<int32_t> contracted;
      auto& producers = producers_map[curr_node_id];
      for (auto producer : producers) {
        if (!node_set.contains(producer)) {
          continue;
        }
        if (!cycle_detector->IsActivateNode(producer)) {
          continue;
        }
        if (!cycle_detector->CanContractEdge(producer, curr_lead_id)) {
          continue;
        }
        auto optional_lead_id =
            cycle_detector->ContractEdge(producer, curr_lead_id);
        assert(optional_lead_id.has_value());
        curr_lead_id = optional_lead_id.value();
        contracted.push_back(producer);
      }
      for (auto new_node : contracted) {
        worklist.emplace_back(new_node, curr_lead_id);
      }
    }

    SmallVector<int32_t> node_left;
    int32_t last_node = -1;
    for (auto op : fusion_pattern.getOpList()) {
      int32_t node_id = op_to_node_id_[op];
      if (cycle_detector->IsActivateNode(node_id)) {
        node_left.push_back(node_id);
      }
      if (last_node != -1) {
        leader_for_node.unionSets(last_node, node_id);
      }
      last_node = node_id;
    }

    // If contracted successfully, there should be only one activate node left.
    if (node_left.size() != 1) {
      return -1;
    } else {
      return node_left[0];
    }
  }

  bool RunFusionPatternFinalization() {
    auto original_nodes = cycle_detector_->AllNodesInPostOrder();
    std::vector<FusionPattern> fusion_patterns;
    std::vector<Operation*> excluded_ops;
    for (int32_t node : original_nodes) {
      Cluster* cluster = GetClusterForCyclesGraphNode(node);
      FusionPattern fusion_pattern = cluster->fused_pattern();
      SmallVector<Operation*> curr_excluded_ops;
      if (!getFusionStrategy().pruneFusionPattern(
              *shape_analysis_, fusion_pattern, curr_excluded_ops)) {
        return false;
      }
      fusion_patterns.emplace_back(std::move(fusion_pattern));
      excluded_ops.insert(excluded_ops.end(), curr_excluded_ops.begin(),
                          curr_excluded_ops.end());
    }

    if (excluded_ops.empty()) {
      return true;
    }

    // The ops inside `excluded_ops` are moved out from existing fusion pattern.
    // It requires to rebuild cycle_detector_, `cluster_storage_` and
    // `leader_for_node_`.

    for (auto op : excluded_ops) {
      fusion_patterns.emplace_back(op);
      getFusionStrategy().initFusionPattern(*shape_analysis_,
                                            fusion_patterns.back());
    }

    std::unique_ptr<GraphCycles> new_cycle_detector(
        new GraphCycles(op_list_.size()));
    EquivalenceClasses<int32_t> new_leader_for_node;
    std::vector<std::unique_ptr<Cluster>> new_cluster_storage(op_list_.size());

    DenseMap<int32_t, DenseSet<int32_t>> producers_map;
    for (auto op_a : op_list_) {
      int32_t node_id_a = op_to_node_id_[op_a];
      new_leader_for_node.insert(node_id_a);
      for (auto op_b : op_list_) {
        if (op_a == op_b) {
          continue;
        }
        int32_t node_id_b = op_to_node_id_[op_b];
        if (original_graph_with_explicit_edges_->HasEdge(node_id_a,
                                                         node_id_b)) {
          auto& producers = producers_map[node_id_b];
          producers.insert(node_id_a);
          new_cycle_detector->InsertEdge(node_id_a, node_id_b);
        }
      }
    }

    for (auto& fusion_pattern : fusion_patterns) {
      int32_t cycles_graph_node_id =
          reContractEdges(fusion_pattern, producers_map,
                          new_cycle_detector.get(), new_leader_for_node);
      if (cycles_graph_node_id == -1) {
        return false;
      }
      int32_t lead_node_id = new_leader_for_node.getLeaderValue(
          op_to_node_id_[fusion_pattern.getOpList()[0]]);
      new_cluster_storage[lead_node_id].reset(
          new Cluster(cycles_graph_node_id, fusion_pattern));
    }

    cycle_detector_ = std::move(new_cycle_detector);
    cluster_storage_ = std::move(new_cluster_storage);
    leader_for_node_ = std::move(new_leader_for_node);

    return true;
  }

  // Here `value` is supported to be a pointer to buffer.
  // Returns the defining op of `value `if no known op updates the buffer,
  // otherwise returns the last op that updates the buffer pointed by the
  // `value`.
  Operation* FindLastWriter(Value value) {
    auto it = last_writer_.find(value);
    if (it != last_writer_.end()) {
      return it->second;
    }
    return value.getDefiningOp();
  }

  // Re-order ops inside the block to make sure that producers are before
  // consumers after fusion.
  void ReorderOperationsInsideBlock() {
    auto reorder_func = [&](Cluster* from, Cluster* to) {
      FusionPattern& from_pattern = from->fused_pattern();
      FusionPattern& to_pattern = to->fused_pattern();

      bool changed = false;
      Operation* last_op_in_from = from_pattern.getOpList().back();
      for (Operation* op : llvm::reverse(to_pattern.getOpList())) {
        if (!last_op_in_from->isBeforeInBlock(op)) {
          changed = true;
          op->moveAfter(last_op_in_from);
        }
      }
      return changed;
    };

    while (ForEachEdgeInPostOrder(reorder_func)) {
      // empty statement by design
    }
  }

 private:
  // fusion pipeline that controls the behaviour of the fusion planner.
  FusionPipeline& fusionPipeline_;

  // Current fusion strategy in the pipeline to apply.
  FusionStrategy* currentFusionStrategy_;

  // The block that fusion planner works on.
  Block* block_;

  // Ops inside the block
  SmallVector<Operation*, 4> op_list_;

  // Shape equality checker
  ShapeAnalysis* shape_analysis_;

  // op -> node_id
  DenseMap<Operation*, int> op_to_node_id_;

  // make sure not introduce cycle after fusion
  std::unique_ptr<GraphCycles> cycle_detector_;
  std::vector<std::unique_ptr<Cluster>> cluster_storage_;

  // Backup the graph cycle information before contracting edges. It also
  // records the explicit inserted edges for horizontal fusions.
  std::unique_ptr<GraphCycles> original_graph_with_explicit_edges_;

  // a UnionFind set. Each set represents a (partial) fused pattern
  // and has a leader as representation.
  EquivalenceClasses<int32_t> leader_for_node_;

  // Here `value` is supported to be a pointer to buffer.
  // Returns the defining op of `value `if no known op updates the buffer,
  // otherwise returns the last op that updates the buffer pointed by the
  // `value`.
  DenseMap<Value, Operation*> last_writer_;
};