package scaletarget

import (
	"context"
	"fmt"

	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	resourcev1 "k8s.io/api/resource/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	lwsv1 "sigs.k8s.io/lws/api/leaderworkerset/v1"

	"github.com/llm-d/llm-d-workload-variant-autoscaler/internal/constants"
	"github.com/llm-d/llm-d-workload-variant-autoscaler/internal/logging"
	"github.com/llm-d/llm-d-workload-variant-autoscaler/internal/resources"
)

// FetchScaleTarget fetches the scale target resource and the first DRA ResourceClaimTemplate
// for each pod template. The returned accessor exposes the templates via
// GetLeaderResourceClaimTemplate() and GetWorkerResourceClaimTemplate(), enabling
// the VerticalActuator to locate and patch the GPU fraction on the associated DRA ResourceClaims.
// For Deployment both methods return the same template (single pod template).
// For LWS the leader and worker templates are fetched independently.
func FetchScaleTarget(ctx context.Context, c client.Client, vaName, kind, name, namespace string) (ScaleTargetAccessor, error) {
	switch kind {
	case constants.DeploymentKind, "": // matching "" for backward compatibility
		var deployment appsv1.Deployment
		if err := resources.GetResourceWithBackoff(ctx, c, client.ObjectKey{Name: name, Namespace: namespace}, &deployment, constants.StandardBackoff, kind); err != nil {
			if apierrors.IsNotFound(err) {
				// Deployment doesn't exist yet, this is expected for VAs without corresponding deployments
				ctrl.LoggerFrom(ctx).V(logging.DEBUG).Info("Deployment not found for VariantAutoscaling, skipping",
					"namespace", namespace,
					"deploymentName", name,
					"vaName", vaName)
			} else {
				// Unexpected error (permissions, network issues, etc.)
				ctrl.LoggerFrom(ctx).Error(err, "Failed to get deployment",
					"namespace", namespace,
					"deploymentName", name,
					"vaName", vaName)
			}
			return nil, err
		}
		rct := fetchResourceClaimTemplate(ctx, c, &deployment.Spec.Template, namespace)
		return NewDeploymentAccessorWithClaim(&deployment, rct), nil
	case constants.LeaderWorkerSetKind:
		var lws lwsv1.LeaderWorkerSet
		if err := resources.GetResourceWithBackoff(ctx, c, client.ObjectKey{Name: name, Namespace: namespace}, &lws, constants.StandardBackoff, kind); err != nil {
			if apierrors.IsNotFound(err) {
				// LWS doesn't exist yet, this is expected for VAs without corresponding LWSs
				ctrl.LoggerFrom(ctx).V(logging.DEBUG).Info("LWS not found for VariantAutoscaling, skipping",
					"namespace", namespace,
					"leaderWorkerSetName", name,
					"vaName", vaName)
			} else {
				// Unexpected error (permissions, network issues, etc.)
				ctrl.LoggerFrom(ctx).Error(err, "Failed to get leaderWorkerSet",
					"namespace", namespace,
					"leaderWorkerSetName", name,
					"vaName", vaName)
			}
			return nil, err
		}
		leaderTemplate := lws.Spec.LeaderWorkerTemplate.LeaderTemplate
		workerTemplate := &lws.Spec.LeaderWorkerTemplate.WorkerTemplate
		leaderRCT := fetchResourceClaimTemplate(ctx, c, leaderTemplate, namespace)
		workerRCT := fetchResourceClaimTemplate(ctx, c, workerTemplate, namespace)
		return NewLWSAccessorWithClaims(&lws, leaderRCT, workerRCT), nil
	}
	return nil, fmt.Errorf("invalid scale target kind %q", kind)
}

// fetchResourceClaimTemplate returns the first DRA ResourceClaimTemplate referenced by
// a pod template's spec.resourceClaims[*].resourceClaimTemplateName entries.
// Returns nil if none are referenced or the first matching template cannot be fetched.
func fetchResourceClaimTemplate(ctx context.Context, c client.Client, podTemplate *corev1.PodTemplateSpec, namespace string) *resourcev1.ResourceClaimTemplate {
	if podTemplate == nil {
		return nil
	}
	for _, prc := range podTemplate.Spec.ResourceClaims {
		if prc.ResourceClaimTemplateName == nil {
			continue
		}
		var rct resourcev1.ResourceClaimTemplate
		key := client.ObjectKey{Name: *prc.ResourceClaimTemplateName, Namespace: namespace}
		if err := c.Get(ctx, key, &rct); err != nil {
			ctrl.LoggerFrom(ctx).V(logging.DEBUG).Info("ResourceClaimTemplate not found, skipping",
				"name", *prc.ResourceClaimTemplateName,
				"namespace", namespace,
				"error", err)
			continue
		}
		return &rct
	}
	return nil
}
