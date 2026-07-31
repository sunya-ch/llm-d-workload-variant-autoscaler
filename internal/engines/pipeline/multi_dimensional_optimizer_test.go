package pipeline

import (
	"context"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	"github.com/llm-d/llm-d-workload-variant-autoscaler/internal/discovery"
	"github.com/llm-d/llm-d-workload-variant-autoscaler/internal/interfaces"
)

func mdiDecisionsByVariant(decisions []interfaces.VariantDecision) map[string]interfaces.VariantDecision {
	m := make(map[string]interfaces.VariantDecision, len(decisions))
	for _, d := range decisions {
		m[d.VariantName] = d
	}
	return m
}

var _ = Describe("MultiDimensionalOptimizer", func() {

	// draAbsent inventory — simulates DRA CRD not installed.
	absentInventory := discovery.NewResourceCapacityInventory(nil, true)

	makeReq := func(variants []interfaces.VariantCapacity, states []interfaces.VariantReplicaState, requiredCap float64) ModelScalingRequest {
		r := &interfaces.AnalyzerResult{
			ModelID:          "model-1",
			Namespace:        "default",
			RequiredCapacity: requiredCap,
			VariantCapacities: variants,
		}
		return ModelScalingRequest{
			ModelID:       "model-1",
			Namespace:     "default",
			VariantStates: states,
			AnalyzerResults: []NamedAnalyzerResult{{
				Name:      interfaces.SaturationAnalyzerName,
				Result:    r,
				Remaining: r.RequiredCapacity,
				Spare:     r.SpareCapacity,
			}},
		}
	}

	Describe("Name", func() {
		It("includes the inner optimizer name", func() {
			o := NewMultiDimensionalOptimizer(NewCostAwareOptimizer(), absentInventory)
			Expect(o.Name()).To(ContainSubstring("cost-aware"))
		})
	})

	Describe("horizontal-only path (DRA absent)", func() {
		It("produces normal horizontal scale-up when DRA is absent", func() {
			o := NewMultiDimensionalOptimizer(NewCostAwareOptimizer(), absentInventory)

			variants := []interfaces.VariantCapacity{
				makeVC("v1", 1000, nil),
			}
			states := []interfaces.VariantReplicaState{
				{VariantName: "v1", CurrentReplicas: 1},
			}
			req := makeReq(variants, states, 500)

			decisions := o.Optimize(context.Background(), []ModelScalingRequest{req}, nil)

			dm := mdiDecisionsByVariant(decisions)
			Expect(dm["v1"].TargetReplicas).To(BeNumerically(">=", 1))
			Expect(dm["v1"].VerticalAction).To(Equal(interfaces.VerticalNoChange))
		})

		It("no vertical action when VerticalScalingEnabled=false even with a hint", func() {
			o := NewMultiDimensionalOptimizer(NewCostAwareOptimizer(), absentInventory)

			variants := []interfaces.VariantCapacity{
				makeVC("v1", 1000, &interfaces.VerticalHint{ScaleUpPerReplicaCapacity: 1500}),
			}
			states := []interfaces.VariantReplicaState{
				{VariantName: "v1", CurrentReplicas: 1, VerticalScalingEnabled: false},
			}
			req := makeReq(variants, states, 500)

			decisions := o.Optimize(context.Background(), []ModelScalingRequest{req}, nil)

			dm := mdiDecisionsByVariant(decisions)
			Expect(dm["v1"].VerticalAction).To(Equal(interfaces.VerticalNoChange))
		})

		It("no vertical action when DRA is absent (draAvailable=nil), even with eligible variant", func() {
			o := NewMultiDimensionalOptimizer(NewCostAwareOptimizer(), absentInventory)

			variants := []interfaces.VariantCapacity{
				makeVC("v1", 1000, &interfaces.VerticalHint{ScaleUpPerReplicaCapacity: 1500}),
			}
			states := []interfaces.VariantReplicaState{
				{VariantName: "v1", CurrentReplicas: 2, VerticalScalingEnabled: true},
			}
			req := makeReq(variants, states, 500)

			decisions := o.Optimize(context.Background(), []ModelScalingRequest{req}, nil)

			dm := mdiDecisionsByVariant(decisions)
			// DRA absent → no vertical, PRC unchanged → horizontal decides normally
			Expect(dm["v1"].VerticalAction).To(Equal(interfaces.VerticalNoChange))
			// PRC must NOT have been patched (original 1000 used for horizontal math)
			Expect(dm["v1"].CurrentPerReplicaCapacity).To(Equal(1000.0))
		})
	})

	Describe("scale-down vertical at minReplicas (no DRA check needed)", func() {
		It("records VerticalScaleDown when at minReplicas and hint present", func() {
			o := NewMultiDimensionalOptimizer(NewCostAwareOptimizer(), absentInventory)

			variants := []interfaces.VariantCapacity{
				makeVC("v1", 2000, &interfaces.VerticalHint{ScaleDownPerReplicaCapacity: 1200}),
			}
			minR := 1
			states := []interfaces.VariantReplicaState{
				{VariantName: "v1", CurrentReplicas: 1, VerticalScalingEnabled: true, MinReplicas: &minR},
			}
			// spare capacity → scale-down path
			req := ModelScalingRequest{
				ModelID:   "model-1",
				Namespace: "default",
				VariantStates: states,
				AnalyzerResults: []NamedAnalyzerResult{{
					Name: interfaces.SaturationAnalyzerName,
					Result: &interfaces.AnalyzerResult{
						ModelID:           "model-1",
						Namespace:         "default",
						SpareCapacity:     5000,
						VariantCapacities: variants,
					},
					Spare: 5000,
				}},
			}

			decisions := o.Optimize(context.Background(), []ModelScalingRequest{req}, nil)

			dm := mdiDecisionsByVariant(decisions)
			Expect(dm["v1"].VerticalAction).To(Equal(interfaces.VerticalScaleDown))
			Expect(dm["v1"].TargetPerReplicaCapacity).To(Equal(1200.0))
		})
	})

	Describe("CurrentPerReplicaCapacity is always the original un-patched value", func() {
		It("carries the pre-patch PRC even when no vertical action is taken", func() {
			o := NewMultiDimensionalOptimizer(NewCostAwareOptimizer(), absentInventory)

			variants := []interfaces.VariantCapacity{makeVC("v1", 3000, nil)}
			states := []interfaces.VariantReplicaState{
				{VariantName: "v1", CurrentReplicas: 1},
			}
			req := makeReq(variants, states, 0)

			decisions := o.Optimize(context.Background(), []ModelScalingRequest{req}, nil)

			dm := mdiDecisionsByVariant(decisions)
			Expect(dm["v1"].CurrentPerReplicaCapacity).To(Equal(3000.0))
		})
	})
})
