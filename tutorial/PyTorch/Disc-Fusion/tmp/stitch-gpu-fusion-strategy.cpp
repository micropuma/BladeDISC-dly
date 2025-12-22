class StitchGpuFusionStrategy : public FusionStrategy {
 public:
  StitchGpuFusionStrategy(const FusionOptions& options)
      : FusionStrategy(options) {}
  virtual bool isFusible(Operation* op) override;
  virtual bool tryFuse(ShapeAnalysis& shapeAnalysis, FusionPattern& lhs,
                       FusionPattern& rhs, FusionPattern& target) override;
  virtual bool initFusionPattern(ShapeAnalysis& shapeAnalysis,
                                 FusionPattern& fusion_pattern) override;
  virtual StringRef getName() override { return "StitchGpuFusionStrategy"; }

 private:
  virtual Value getEffectiveShape(FusionPattern& target, Value value);

  bool tileCoverInfoPropagateO2I(
      ShapeAnalysis& shapeAnalysis, DenseMap<Value, TileInfo>& tile_plan,
      Operation* op, SmallVector<std::pair<Value, TileInfo>, 4>& in_info,
      bool& cover);
  bool findFusionPatternTypeAndSubroot(ShapeAnalysis& shapeAnalysis,
                                       FusionPattern& fusion_pattern);
  bool tileXroots(ShapeAnalysis& shapeAnalysis, FusionPattern& fusion_pattern);
  bool backtraceTileAndCover(ShapeAnalysis& shapeAnalysis,
                             FusionPattern& fusion_pattern, Value value);
};