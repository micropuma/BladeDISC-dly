// Stich GPU fusion pattern initialization
bool StitchGpuFusionStrategy::initFusionPattern(ShapeAnalysis& shapeAnalysis,
                                                FusionPattern& fusion_pattern) {
  const auto& results = fusion_pattern.getResults();
  if (results.empty()) {
    return false;
  }

  if (!findFusionPatternTypeAndSubroot(shapeAnalysis, fusion_pattern)) {
    return false;
  } else if (fusion_pattern.getFusionType() != FusionType::kStitch) {
    return true;
  }
  FusionType fusion_type = fusion_pattern.getFusionType();
  Operation* dominant_op = fusion_pattern.getDominantOp();
  const auto& subroots = fusion_pattern.getSubRootOps();

  // Analyze tile information of sub-roots and roots, identify regular/irregular
  // xroots.
  if (!tileXroots(shapeAnalysis, fusion_pattern)) {
    return false;
  }
  DenseMap<Value, TileInfo>& tile_plan = fusion_pattern.getTilePlan();

  // Propagate tile information and data covering status back from skeleton ops.
  DenseSet<Operation*> skeleton_op_set(subroots.begin(), subroots.end());
  const auto& external_only_results = fusion_pattern.getExternalOnlyResults();
  for (Value res : external_only_results) {
    Operation* op = fusion_pattern.findLastWriter(res);
    skeleton_op_set.insert(op);
  }
  DenseSet<Operation*> subroots_set(subroots.begin(), subroots.end());
  for (auto op : skeleton_op_set) {
    Value value;
    if (subroots_set.contains(op)) {
      // Propagate from input for subroot (row-reduction).
      value = op->getOperand(0);
    } else {
      // Propagate from output, because we do not want to iterate on all inputs
      // one by one.
      value = cast<lmhlo::LmhloOp>(op).getResultBuffer();
    }
    if (!backtraceTileAndCover(shapeAnalysis, fusion_pattern, value)) {
      return false;
    }
  }

  // TODO: global memory buffer for intermeidate common operands.
  // TODO: add speculation hint here.

  return true;
}