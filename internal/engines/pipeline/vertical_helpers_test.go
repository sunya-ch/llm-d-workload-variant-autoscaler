package pipeline

import (
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	"github.com/llm-d/llm-d-workload-variant-autoscaler/internal/interfaces"
)

// helpers for building test data

func ptrInt(v int) *int { return &v }

func makeVC(name string, prc float64, hint *interfaces.VerticalHint) interfaces.VariantCapacity {
	return interfaces.VariantCapacity{
		VariantName:        name,
		PerReplicaCapacity: prc,
		VerticalHint:       hint,
	}
}

func makeState(name string, enabled bool, minReplicas *int) interfaces.VariantReplicaState {
	return interfaces.VariantReplicaState{
		VariantName:            name,
		VerticalScalingEnabled: enabled,
		MinReplicas:            minReplicas,
	}
}

var _ = Describe("applyVerticalScaleUp", func() {

	It("patches PRC and records target when ScaleUpPRC > current PRC", func() {
		variants := []interfaces.VariantCapacity{
			makeVC("v1", 1000, &interfaces.VerticalHint{ScaleUpPerReplicaCapacity: 1500}),
		}
		stateMap := map[string]interfaces.VariantReplicaState{
			"v1": makeState("v1", true, nil),
		}
		vt := map[string]verticalTarget{}

		applyVerticalScaleUp(variants, stateMap, vt)

		Expect(variants[0].PerReplicaCapacity).To(Equal(1500.0), "working copy PRC must be patched")
		Expect(vt).To(HaveKey("v1"))
		Expect(vt["v1"].Action).To(Equal(interfaces.VerticalScaleUp))
		Expect(vt["v1"].TargetPRC).To(Equal(1500.0))
	})

	It("no-ops when ScaleUpPRC <= current PRC (demand already satisfied)", func() {
		variants := []interfaces.VariantCapacity{
			makeVC("v1", 2000, &interfaces.VerticalHint{ScaleUpPerReplicaCapacity: 1500}),
		}
		stateMap := map[string]interfaces.VariantReplicaState{
			"v1": makeState("v1", true, nil),
		}
		vt := map[string]verticalTarget{}

		applyVerticalScaleUp(variants, stateMap, vt)

		Expect(variants[0].PerReplicaCapacity).To(Equal(2000.0), "PRC must not be patched")
		Expect(vt).To(BeEmpty())
	})

	It("no-ops when VerticalScalingEnabled is false", func() {
		variants := []interfaces.VariantCapacity{
			makeVC("v1", 1000, &interfaces.VerticalHint{ScaleUpPerReplicaCapacity: 1500}),
		}
		stateMap := map[string]interfaces.VariantReplicaState{
			"v1": makeState("v1", false, nil),
		}
		vt := map[string]verticalTarget{}

		applyVerticalScaleUp(variants, stateMap, vt)

		Expect(variants[0].PerReplicaCapacity).To(Equal(1000.0))
		Expect(vt).To(BeEmpty())
	})

	It("no-ops when hint is nil", func() {
		variants := []interfaces.VariantCapacity{makeVC("v1", 1000, nil)}
		stateMap := map[string]interfaces.VariantReplicaState{
			"v1": makeState("v1", true, nil),
		}
		vt := map[string]verticalTarget{}

		applyVerticalScaleUp(variants, stateMap, vt)

		Expect(vt).To(BeEmpty())
	})

	It("propagates DemandPerReplicaResource from hint", func() {
		demand := &interfaces.ResourceRequirement{ComputeFraction: 0.7, MemoryBytes: 8_000_000_000}
		variants := []interfaces.VariantCapacity{
			makeVC("v1", 1000, &interfaces.VerticalHint{
				ScaleUpPerReplicaCapacity: 1500,
				DemandPerReplicaResource:  demand,
			}),
		}
		stateMap := map[string]interfaces.VariantReplicaState{
			"v1": makeState("v1", true, nil),
		}
		vt := map[string]verticalTarget{}

		applyVerticalScaleUp(variants, stateMap, vt)

		Expect(vt["v1"].DemandPerReplicaResource).To(Equal(demand))
	})
})

var _ = Describe("applyVerticalScaleDown", func() {

	It("records scale-down target when at minReplicas", func() {
		variants := []interfaces.VariantCapacity{
			makeVC("v1", 2000, &interfaces.VerticalHint{ScaleDownPerReplicaCapacity: 1200}),
		}
		stateMap := map[string]interfaces.VariantReplicaState{
			"v1": makeState("v1", true, ptrInt(1)),
		}
		targets := map[string]int{"v1": 1} // at minReplicas
		vt := map[string]verticalTarget{}

		applyVerticalScaleDown(variants, stateMap, targets, vt)

		Expect(vt).To(HaveKey("v1"))
		Expect(vt["v1"].Action).To(Equal(interfaces.VerticalScaleDown))
		Expect(vt["v1"].TargetPRC).To(Equal(1200.0))
	})

	It("skips scale-down when above minReplicas (horizontal first)", func() {
		variants := []interfaces.VariantCapacity{
			makeVC("v1", 2000, &interfaces.VerticalHint{ScaleDownPerReplicaCapacity: 1200}),
		}
		stateMap := map[string]interfaces.VariantReplicaState{
			"v1": makeState("v1", true, ptrInt(1)),
		}
		targets := map[string]int{"v1": 3} // above minReplicas
		vt := map[string]verticalTarget{}

		applyVerticalScaleDown(variants, stateMap, targets, vt)

		Expect(vt).To(BeEmpty())
	})

	It("no-ops when VerticalScalingEnabled is false", func() {
		variants := []interfaces.VariantCapacity{
			makeVC("v1", 2000, &interfaces.VerticalHint{ScaleDownPerReplicaCapacity: 1200}),
		}
		stateMap := map[string]interfaces.VariantReplicaState{
			"v1": makeState("v1", false, ptrInt(1)),
		}
		targets := map[string]int{"v1": 1}
		vt := map[string]verticalTarget{}

		applyVerticalScaleDown(variants, stateMap, targets, vt)

		Expect(vt).To(BeEmpty())
	})

	It("no-ops when hint is nil", func() {
		variants := []interfaces.VariantCapacity{makeVC("v1", 2000, nil)}
		stateMap := map[string]interfaces.VariantReplicaState{
			"v1": makeState("v1", true, ptrInt(1)),
		}
		targets := map[string]int{"v1": 1}
		vt := map[string]verticalTarget{}

		applyVerticalScaleDown(variants, stateMap, targets, vt)

		Expect(vt).To(BeEmpty())
	})

	It("no-ops when ScaleDownPRC >= current PRC", func() {
		variants := []interfaces.VariantCapacity{
			makeVC("v1", 1000, &interfaces.VerticalHint{ScaleDownPerReplicaCapacity: 1000}),
		}
		stateMap := map[string]interfaces.VariantReplicaState{
			"v1": makeState("v1", true, ptrInt(1)),
		}
		targets := map[string]int{"v1": 1}
		vt := map[string]verticalTarget{}

		applyVerticalScaleDown(variants, stateMap, targets, vt)

		Expect(vt).To(BeEmpty())
	})

	It("skips when MinReplicas is nil (no floor set)", func() {
		variants := []interfaces.VariantCapacity{
			makeVC("v1", 2000, &interfaces.VerticalHint{ScaleDownPerReplicaCapacity: 1200}),
		}
		stateMap := map[string]interfaces.VariantReplicaState{
			"v1": makeState("v1", true, nil), // no minReplicas
		}
		targets := map[string]int{"v1": 1}
		vt := map[string]verticalTarget{}

		applyVerticalScaleDown(variants, stateMap, targets, vt)

		Expect(vt).To(BeEmpty())
	})
})
