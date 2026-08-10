package pipeline

import (
	"context"

	ctrl "sigs.k8s.io/controller-runtime"

	"github.com/llm-d/llm-d-workload-variant-autoscaler/internal/discovery"
	"github.com/llm-d/llm-d-workload-variant-autoscaler/internal/interfaces"
	"github.com/llm-d/llm-d-workload-variant-autoscaler/internal/logging"
)

// MultiDimensionalOptimizer wraps an inner ScalingOptimizer (typically
// GreedyByScoreOptimizer or CostAwareOptimizer) and adds a DRA-aware vertical
// scaling pass before/after the horizontal allocation loop.
//
// Scale-up path (vertical-first):
//  1. Call applyVerticalScaleUp — patches vc.PerReplicaCapacity in the working
//     slice only when DRA device headroom is sufficient; decrements draAvailable.
//  2. Run the inner optimizer's horizontal allocation with the (patched) PRC.
//
// Scale-down path (horizontal-first):
//  1. Run inner optimizer scale-down (scaleDownRoleIterated inside the inner optimizer).
//  2. Call applyVerticalScaleDown — records vertical target only when the variant
//     is already at minReplicas (no DRA headroom check needed for shrink).
//
// When draAvailable is nil (DRA CRD absent or Snapshot failed), applyVerticalScaleUp
// is a no-op and the optimizer falls through to a pure horizontal result.
type MultiDimensionalOptimizer struct {
	inner     ScalingOptimizer
	inventory *discovery.ResourceCapacityInventory
}

// NewMultiDimensionalOptimizer creates a MultiDimensionalOptimizer wrapping inner.
func NewMultiDimensionalOptimizer(inner ScalingOptimizer, inventory *discovery.ResourceCapacityInventory) *MultiDimensionalOptimizer {
	return &MultiDimensionalOptimizer{inner: inner, inventory: inventory}
}

// Name returns the optimizer identifier including the inner optimizer name.
func (o *MultiDimensionalOptimizer) Name() string {
	return "multi-dimensional(" + o.inner.Name() + ")"
}

// Optimize produces VariantDecisions with a vertical-first decision pass,
// falling back to pure horizontal when DRA headroom is insufficient or unavailable.
func (o *MultiDimensionalOptimizer) Optimize(
	ctx context.Context,
	requests []ModelScalingRequest,
	constraints []*ResourceConstraints,
) []interfaces.VariantDecision {
	logger := ctrl.LoggerFrom(ctx).WithName(o.Name())

	// Take a single DRA snapshot for the whole tick.
	var draAvailable map[string]int64
	var claimedByVariant map[string]map[string]int64
	if o.inventory != nil {
		draAvailable, claimedByVariant = o.inventory.Snapshot(ctx)
	}

	if draAvailable == nil {
		logger.V(logging.DEBUG).Info("DRA unavailable this tick — running horizontal-only path")
	}

	var allDecisions []interfaces.VariantDecision

	for _, req := range requests {
		satEntry := saturationEntry(req.AnalyzerResults)
		if satEntry == nil {
			continue
		}

		stateMap := buildStateMap(req.VariantStates)
		// vcMap is built BEFORE any vertical patching so CurrentPerReplicaCapacity
		// in VariantDecision always holds the original un-patched value.
		vcMap := buildCapacityMap(satEntry.VariantCapacities)
		targets := initTargets(req.VariantStates)

		s := req.AnalyzerResults
		roles, ps := initRoleState(s)

		verticalTargets := map[string]verticalTarget{}

		if anyRoleNeedsScaleUp(ps, roles) {
			// Vertical-first: patch PRC when DRA headroom allows, then run horizontal.
			applyVerticalScaleUpWithDRA(satEntry.VariantCapacities, stateMap, draAvailable, claimedByVariant, verticalTargets)
			allocateForModelPaired(ctx, s, satEntry.VariantCapacities, stateMap, nil, targets,
				costGreedyRolePick, ps, roles)
		} else {
			// Horizontal-first scale-down, then vertical if at minReplicas.
			scaleDownRoleIterated(ctx, s, satEntry.VariantCapacities, targets, stateMap)
			applyVerticalScaleDown(satEntry.VariantCapacities, stateMap, targets, verticalTargets)
		}

		decisions := buildDecisionsWithOptimizer(req, stateMap, vcMap, targets, o.Name(), verticalTargets)
		logger.V(logging.DEBUG).Info("Multi-dimensional optimizer decisions",
			"modelID", req.ModelID,
			"decisions", len(decisions),
			"verticalTargets", len(verticalTargets))
		allDecisions = append(allDecisions, decisions...)
	}

	return allDecisions
}

// applyVerticalScaleUpWithDRA is the DRA-aware variant of applyVerticalScaleUp.
// It patches vc.PerReplicaCapacity and records a verticalTarget only when:
//   - VerticalScalingEnabled is true
//   - ScaleUpPerReplicaCapacity > current PRC
//   - draAvailable is non-nil (DRA present this tick)
//   - draAvailable[capacityName] ≥ deltaPerReplica × readyReplicas
//
// On success: draAvailable is decremented (budget consumed).
// On insufficient headroom or draAvailable==nil: PRC left unchanged; no vertical target recorded.
//
// OOM safety: demand.MemoryBytes is floored at the current per-replica claimed
// allocation (currentClaimed/readyCount) before the delta is computed. This
// prevents recommending a resource slice smaller than what the pod currently
// holds — which would cause an OOM kill on the next pod restart if the
// empirical MemoryWeight derived from the KV-cache ratio underestimates the
// real model weight memory.
func applyVerticalScaleUpWithDRA(
	variants []interfaces.VariantCapacity,
	stateMap map[string]interfaces.VariantReplicaState,
	draAvailable map[string]int64,
	claimedByVariant map[string]map[string]int64,
	verticalTargets map[string]verticalTarget,
) {
	for i, vc := range variants {
		state, ok := stateMap[vc.VariantName]
		if !ok || !state.VerticalScalingEnabled {
			continue
		}
		hint := vc.VerticalHint
		if hint == nil || hint.ScaleUpPerReplicaCapacity <= vc.PerReplicaCapacity {
			continue
		}
		if draAvailable == nil {
			// DRA unavailable this tick — skip, horizontal proceeds with original PRC.
			continue
		}
		if hint.DemandPerReplicaResource == nil {
			// No resource sizing available yet (obs store not bootstrapped).
			// Still record vertical intent with nil DemandPerReplicaResource
			// but skip DRA headroom check since we have no capacity delta to measure.
			verticalTargets[vc.VariantName] = verticalTarget{
				TargetPRC:                hint.ScaleUpPerReplicaCapacity,
				DemandPerReplicaResource: nil,
				Action:                   interfaces.VerticalScaleUp,
			}
			variants[i].PerReplicaCapacity = hint.ScaleUpPerReplicaCapacity
			continue
		}

		// Check DRA headroom across all controlled capacity dimensions.
		demand := hint.DemandPerReplicaResource
		// Use MemoryBytes as the primary capacity dimension (in milli-units: 1 byte = 1000 milli-bytes).
		// This is the canonical DRA capacity dimension for GPU memory (e.g. "memory").
		readyCount := int64(max(state.CurrentReplicas-state.PendingReplicas, 0))
		if readyCount == 0 {
			readyCount = 1
		}

		// currentClaimed is what this variant already holds in DRA (per-replica × readyCount).
		currentClaimed := int64(0)
		if cm, found := claimedByVariant[vc.VariantName]; found {
			for _, v := range cm {
				currentClaimed += v
			}
		}

		// OOM safety floor: ensure demand.MemoryBytes is never less than the
		// current per-replica allocation. currentClaimed is in milli-units;
		// convert to bytes (÷1000) for comparison with demand.MemoryBytes.
		// Work on a local copy so the original hint is not mutated.
		currentPerReplicaBytes := currentClaimed / readyCount / 1000
		effectiveDemand := *demand
		if currentPerReplicaBytes > 0 && effectiveDemand.MemoryBytes < currentPerReplicaBytes {
			effectiveDemand.MemoryBytes = currentPerReplicaBytes
		}
		demand = &effectiveDemand

		// deltaTotal = (target memory − current claimed) × replicas.
		// demand.MemoryBytes is in raw bytes; DRA capacity is in milli-units.
		targetMemMillis := demand.MemoryBytes * 1000
		deltaPerReplica := targetMemMillis - (currentClaimed / readyCount)
		if deltaPerReplica <= 0 {
			// Already have enough claimed capacity — record target without consuming budget.
			verticalTargets[vc.VariantName] = verticalTarget{
				TargetPRC:                hint.ScaleUpPerReplicaCapacity,
				DemandPerReplicaResource: demand,
				Action:                   interfaces.VerticalScaleUp,
			}
			variants[i].PerReplicaCapacity = hint.ScaleUpPerReplicaCapacity
			continue
		}

		totalDelta := deltaPerReplica * readyCount

		// Check headroom across the first available capacity dimension.
		// In practice a GPU device exposes one "memory" capacity; we check the
		// total available headroom without tying to a specific capacity name.
		totalAvailable := int64(0)
		for _, v := range draAvailable {
			totalAvailable += v
		}

		if totalAvailable < totalDelta {
			// Insufficient DRA headroom — skip vertical, horizontal runs with original PRC.
			continue
		}

		// Sufficient headroom: consume budget, patch PRC, record target.
		for capName := range draAvailable {
			draAvailable[capName] -= totalDelta
			break // consume from the first dimension only
		}
		verticalTargets[vc.VariantName] = verticalTarget{
			TargetPRC:                hint.ScaleUpPerReplicaCapacity,
			DemandPerReplicaResource: demand,
			Action:                   interfaces.VerticalScaleUp,
		}
		variants[i].PerReplicaCapacity = hint.ScaleUpPerReplicaCapacity
	}
}

// max returns the larger of a and b.
func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
