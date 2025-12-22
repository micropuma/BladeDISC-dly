/// Describes the fusion strategy to be used in the Affine loop fusion
/// utilities. Currently, it is used to specialized the loop fusion utilities
/// with the assumptions made in the AffineLoopFusion pass for producer-consumer
/// and sibling fusion, while sharing a single implementation. The latter
/// strategies are also limited to scenarios where a single memref is involved
/// in the producer-consume or sibling relationship between the candidate
/// loops. We use 'memref' to keep track of such a memref.
// TODO: Remove 'memref' when we support more generic scenarios.
// TODO: Generalize utilities so that producer-consumer and sibling fusion
// strategies can be used without the assumptions made in the AffineLoopFusion
// pass.
class FusionStrategy {
public:
  enum StrategyEnum {
    // Generic loop fusion: Arbitrary loops are considered for fusion. No
    // assumptions about a specific fusion strategy from AffineLoopFusion pass
    // are made.
    // TODO: Generic fusion is not fully implemented by fusion utilities yet.
    // It should only be used for testing.
    Generic,
    // Producer-consumer fusion: Only loops with a producer-consumer
    // memref dependence are considered for fusion. Currently, assumptions from
    // the producer-consumer fusion implementation in AffineLoopFusion pass are
    // made. See pass for specific details.
    ProducerConsumer,
    // Sibling fusion: Only sibling loops with no producer-consumer memref
    // dependences are considered for fusion. Memref reuse is taken into account
    // for profitability. Currently, assumptions from the sibling fusion
    // implementation in AffineLoopFusion pass are made. See pass for specific
    // details.
    Sibling
  };

  /// Construct a generic or producer-consumer fusion strategy.
  FusionStrategy(StrategyEnum strategy) : strategy(strategy) {
    assert(strategy != Sibling &&
           "Sibling fusion strategy requires a specific memref");
  }

  /// Construct a sibling fusion strategy targeting 'memref'. This construct
  /// should only be used for sibling fusion.
  FusionStrategy(Value memref) : strategy(Sibling), memref(memref) {}

  /// Returns the fusion strategy.
  StrategyEnum getStrategy() const { return strategy; };

  /// Returns the memref attached to this sibling fusion strategy.
  Value getSiblingFusionMemRef() const {
    assert(strategy == Sibling && "Memref is only valid for sibling fusion");
    return memref;
  }

private:
  /// Fusion strategy.
  StrategyEnum strategy;

  /// Target memref for this fusion transformation. Only used for sibling
  /// fusion.
  Value memref;
};