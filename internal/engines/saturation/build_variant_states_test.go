package saturation

import (
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	appsv1 "k8s.io/api/apps/v1"
	autoscalingv2 "k8s.io/api/autoscaling/v2"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	vpav1 "k8s.io/autoscaler/vertical-pod-autoscaler/pkg/apis/autoscaling.k8s.io/v1"

	llmdv1alpha1 "github.com/llm-d/llm-d-workload-variant-autoscaler/api/v1alpha1"
	"github.com/llm-d/llm-d-workload-variant-autoscaler/internal/utils"
	"github.com/llm-d/llm-d-workload-variant-autoscaler/internal/utils/scaletarget"
)

// makeVA returns a minimal VariantAutoscaling for use in BuildVariantStates unit tests.
// scaleTargetName is the Deployment name; resourceClaimPolicy may be nil.
func makeVA(ns, name, scaleTargetName string, resourceClaimPolicy *vpav1.ResourceClaimPolicy) llmdv1alpha1.VariantAutoscaling {
	return llmdv1alpha1.VariantAutoscaling{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: ns,
		},
		Spec: llmdv1alpha1.VariantAutoscalingSpec{
			ScaleTargetRef: autoscalingv2.CrossVersionObjectReference{
				Kind: "Deployment",
				Name: scaleTargetName,
			},
			ResourceClaimPolicy: resourceClaimPolicy,
		},
	}
}

// makeDeploymentAccessor returns a minimal DeploymentAccessor for use in the
// scaleTargets map; all replica counters are zero (sufficient for the flag test).
func makeDeploymentAccessor() scaletarget.ScaleTargetAccessor {
	return scaletarget.NewDeploymentAccessor(&appsv1.Deployment{})
}

var _ = Describe("BuildVariantStates — VerticalScalingEnabled flag", func() {

	var e *Engine

	BeforeEach(func() {
		e = &Engine{}
	})

	It("sets VerticalScalingEnabled=false when ResourceClaimPolicy is nil", func() {
		va := makeVA("ns1", "va-no-vpa", "deploy-a", nil)

		scaleTargets := map[string]scaletarget.ScaleTargetAccessor{
			utils.GetNamespacedKey("ns1", "deploy-a"): makeDeploymentAccessor(),
		}

		states := e.BuildVariantStates(ctx, []llmdv1alpha1.VariantAutoscaling{va}, scaleTargets, nil)

		Expect(states).To(HaveLen(1))
		Expect(states[0].VerticalScalingEnabled).To(BeFalse())
	})

	It("sets VerticalScalingEnabled=true when ResourceClaimPolicy is set", func() {
		policy := &vpav1.ResourceClaimPolicy{
			ClaimTemplateName: "gpu-claim",
			DeviceClassName:   "gpu.nvidia.com",
		}
		va := makeVA("ns1", "va-with-vpa", "deploy-b", policy)

		scaleTargets := map[string]scaletarget.ScaleTargetAccessor{
			utils.GetNamespacedKey("ns1", "deploy-b"): makeDeploymentAccessor(),
		}

		states := e.BuildVariantStates(ctx, []llmdv1alpha1.VariantAutoscaling{va}, scaleTargets, nil)

		Expect(states).To(HaveLen(1))
		Expect(states[0].VerticalScalingEnabled).To(BeTrue())
	})

	It("handles mixed variants correctly", func() {
		policy := &vpav1.ResourceClaimPolicy{ClaimTemplateName: "gpu-claim"}
		vas := []llmdv1alpha1.VariantAutoscaling{
			makeVA("ns1", "va-plain", "deploy-plain", nil),
			makeVA("ns1", "va-vpa", "deploy-vpa", policy),
		}
		scaleTargets := map[string]scaletarget.ScaleTargetAccessor{
			utils.GetNamespacedKey("ns1", "deploy-plain"): makeDeploymentAccessor(),
			utils.GetNamespacedKey("ns1", "deploy-vpa"):   makeDeploymentAccessor(),
		}

		states := e.BuildVariantStates(ctx, vas, scaleTargets, nil)

		Expect(states).To(HaveLen(2))
		Expect(states[0].VerticalScalingEnabled).To(BeFalse(), "plain variant should not have vertical enabled")
		Expect(states[1].VerticalScalingEnabled).To(BeTrue(), "VPA variant should have vertical enabled")
	})
})
