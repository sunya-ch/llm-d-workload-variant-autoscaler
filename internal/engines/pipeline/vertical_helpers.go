package pipeline

import "github.com/llm-d/llm-d-workload-variant-autoscaler/internal/interfaces"

// verticalTarget records the vertical scaling intent for one variant.
type verticalTarget struct {
	// TargetPRC is the ScaleUp or ScaleDown per-replica capacity.
	TargetPRC float64
	// DemandPerReplicaResource is the Step-policy-rounded resource target.
	// Nil when the observation store was not yet bootstrapped.
	DemandPerReplicaResource *interfaces.ResourceRequirement
	// Action is the vertical scaling direction.
	Action interfaces.VerticalScalingAction
}

// applyVerticalScaleUp iterates variants, records ScaleUpPerReplicaCapacity in
// verticalTargets and patches vc.PerReplicaCapacity in the variants slice for
// the subsequent allocateForModelPaired call.
// Only acts when VerticalScalingEnabled and ScaleUpPerReplicaCapacity > current PRC.
func applyVerticalScaleUp(
	variants []interfaces.VariantCapacity,
	stateMap map[string]interfaces.VariantReplicaState,
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
		verticalTargets[vc.VariantName] = verticalTarget{
			TargetPRC:                hint.ScaleUpPerReplicaCapacity,
			DemandPerReplicaResource: hint.DemandPerReplicaResource,
			Action:                   interfaces.VerticalScaleUp,
		}
		// Patch working copy so allocateForModelPaired sees the higher PRC.
		variants[i].PerReplicaCapacity = hint.ScaleUpPerReplicaCapacity
	}
}

// applyVerticalScaleDown records ScaleDownPerReplicaCapacity in verticalTargets
// only when the variant is already at minReplicas (horizontal scale-down exhausted).
func applyVerticalScaleDown(
	variants []interfaces.VariantCapacity,
	stateMap map[string]interfaces.VariantReplicaState,
	targets map[string]int,
	verticalTargets map[string]verticalTarget,
) {
	for _, vc := range variants {
		state, ok := stateMap[vc.VariantName]
		if !ok || !state.VerticalScalingEnabled {
			continue
		}
		hint := vc.VerticalHint
		if hint == nil || hint.ScaleDownPerReplicaCapacity >= vc.PerReplicaCapacity {
			continue
		}
		// Only shrink resources when horizontal is already at its floor.
		currentTarget := targets[vc.VariantName]
		atMin := state.MinReplicas != nil && currentTarget <= *state.MinReplicas
		if !atMin {
			continue
		}
		verticalTargets[vc.VariantName] = verticalTarget{
			TargetPRC:                hint.ScaleDownPerReplicaCapacity,
			DemandPerReplicaResource: hint.DemandPerReplicaResource,
			Action:                   interfaces.VerticalScaleDown,
		}
	}
}
