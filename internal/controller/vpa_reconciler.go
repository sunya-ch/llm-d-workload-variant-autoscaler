package controller

import (
	"context"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	vpav1 "k8s.io/autoscaler/vertical-pod-autoscaler/pkg/apis/autoscaling.k8s.io/v1"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/builder"
	"sigs.k8s.io/controller-runtime/pkg/client"

	"github.com/llm-d/llm-d-workload-variant-autoscaler/internal/annotations"
	"github.com/llm-d/llm-d-workload-variant-autoscaler/internal/datastore"
	"github.com/llm-d/llm-d-workload-variant-autoscaler/internal/logging"
	"github.com/llm-d/llm-d-workload-variant-autoscaler/internal/utils"
)

// VPAReconciler tracks namespaces for annotation-based WVA discovery via VPA
// VerticalPodAutoscaler objects. Its sole job is to call NamespaceTrack /
// NamespaceUntrack so the engine's polling loop can scope its List calls to
// namespaces that contain managed scalers. Registered only when the VPA CRD
// is detected at startup.
type VPAReconciler struct {
	client.Client
	Datastore datastore.Datastore
}

// +kubebuilder:rbac:groups=autoscaling.k8s.io,resources=verticalpodautoscalers,verbs=get;list;watch
// Note: Kubernetes RBAC does not support annotation-based selectors, so this grant is
// necessarily cluster-wide. Access is read-only (get;list;watch). AnnotatedScalerPredicate
// limits event processing to managed objects (see TODO #1134 for scoped List calls).

// Reconcile tracks or untracks the namespace of a VPA bearing llm-d.ai/managed: "true".
// A VPA is only tracked when it is managed, not being deleted, and uses the
// "prometheus" external recommender. VPAs using the default built-in recommender
// or any other recommender are untracked.
func (r *VPAReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	vpa := &vpav1.VerticalPodAutoscaler{}
	if err := r.Get(ctx, req.NamespacedName, vpa); err != nil {
		if apierrors.IsNotFound(err) {
			r.Datastore.NamespaceUntrack("AnnotatedScaler", req.Name, req.Namespace)
			return ctrl.Result{}, nil
		}
		return ctrl.Result{}, err
	}

	if !vpa.DeletionTimestamp.IsZero() || !annotations.IsManaged(vpa) || !utils.HasPrometheusRecommender(vpa) {
		if annotations.IsManaged(vpa) && !utils.HasPrometheusRecommender(vpa) {
			ctrl.LoggerFrom(ctx).V(logging.DEBUG).Info(
				"VPA is managed but does not use the prometheus recommender, not tracking",
				"name", vpa.Name,
				"namespace", vpa.Namespace,
			)
		}
		r.Datastore.NamespaceUntrack("AnnotatedScaler", req.Name, req.Namespace)
	} else {
		r.Datastore.NamespaceTrack("AnnotatedScaler", req.Name, req.Namespace)
	}
	return ctrl.Result{}, nil
}

// SetupWithManager registers VPAReconciler with the controller manager.
func (r *VPAReconciler) SetupWithManager(mgr ctrl.Manager) error {
	return ctrl.NewControllerManagedBy(mgr).
		For(&vpav1.VerticalPodAutoscaler{},
			builder.WithPredicates(AnnotatedScalerPredicate()),
		).
		Named("vpa").
		Complete(r)
}
