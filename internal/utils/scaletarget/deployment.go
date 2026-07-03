package scaletarget

import (
	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	resourcev1 "k8s.io/api/resource/v1"
	v1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	"github.com/llm-d/llm-d-workload-variant-autoscaler/internal/resources"
)

type deploymentAccessor struct {
	deployment            *appsv1.Deployment
	resourceClaimTemplate *resourcev1.ResourceClaimTemplate
}

func NewDeploymentAccessor(deploy *appsv1.Deployment) ScaleTargetAccessor {
	if deploy == nil {
		return nil
	}
	return &deploymentAccessor{deployment: deploy}
}

// NewDeploymentAccessorWithClaim creates a deploymentAccessor that also
// exposes the first DRA ResourceClaimTemplate referenced by the pod template.
// For a Deployment leader == worker, so the same template is returned by both
// GetLeaderResourceClaimTemplate and GetWorkerResourceClaimTemplate.
func NewDeploymentAccessorWithClaim(deploy *appsv1.Deployment, rct *resourcev1.ResourceClaimTemplate) ScaleTargetAccessor {
	if deploy == nil {
		return nil
	}
	return &deploymentAccessor{deployment: deploy, resourceClaimTemplate: rct}
}

func (r *deploymentAccessor) GetReplicas() *int32 {
	// r.deployment is always not nil
	return r.deployment.Spec.Replicas
}

func (r *deploymentAccessor) GetStatusReplicas() int32 {
	// r.deployment is always not nil
	return r.deployment.Status.Replicas
}

func (r *deploymentAccessor) GetStatusReadyReplicas() int32 {
	// r.deployment is always not nil
	return r.deployment.Status.ReadyReplicas
}

func (r *deploymentAccessor) GetTotalGPUsPerReplica() int {
	// r.deployment is always not nil
	total := resources.GetContainersGPUs(r.deployment.Spec.Template.Spec.Containers)
	// Default to 1 GPU if no explicit requests found
	// (common for inference workloads that may not have resource requests)
	if total == 0 {
		return 1
	}
	return total
}

func (r *deploymentAccessor) GetDeletionTimestamp() *v1.Time {
	// r.deployment is always not nil
	return r.deployment.DeletionTimestamp
}

func (r *deploymentAccessor) GetLeaderPodTemplateSpec() *corev1.PodTemplateSpec {
	// r.deployment is always not nil
	return &r.deployment.Spec.Template
}

func (r *deploymentAccessor) GetWorkerPodTemplateSpec() *corev1.PodTemplateSpec {
	return r.GetLeaderPodTemplateSpec()
}

func (r *deploymentAccessor) GetGroupSize() int32 {
	return 1
}

func (r *deploymentAccessor) GetName() string {
	// r.deployment is always not nil
	return r.deployment.Name
}

func (r *deploymentAccessor) GetNamespace() string {
	// r.deployment is always not nil
	return r.deployment.Namespace
}

func (r *deploymentAccessor) GetLeaderResourceClaimTemplate() *resourcev1.ResourceClaimTemplate {
	return r.resourceClaimTemplate
}

func (r *deploymentAccessor) GetWorkerResourceClaimTemplate() *resourcev1.ResourceClaimTemplate {
	// A Deployment has a single pod template; leader == worker.
	return r.resourceClaimTemplate
}
