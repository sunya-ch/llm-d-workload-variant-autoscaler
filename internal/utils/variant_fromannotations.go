/*
Copyright 2025 The llm-d Authors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package utils

import (
	"fmt"

	kedav1alpha1 "github.com/kedacore/keda/v2/apis/keda/v1alpha1"
	autoscalingv2 "k8s.io/api/autoscaling/v2"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	vpav1 "k8s.io/autoscaler/vertical-pod-autoscaler/pkg/apis/autoscaling.k8s.io/v1"
	"k8s.io/utils/ptr"

	wvav1alpha1 "github.com/llm-d/llm-d-workload-variant-autoscaler/api/v1alpha1"
	"github.com/llm-d/llm-d-workload-variant-autoscaler/internal/annotations"
)

const (
	KindDeployment = "Deployment"
	DeploymentGV   = "apps/v1"

	// VPAPrometheusRecommender is the recommender name WVA requires on a managed VPA.
	// VPAs using any other recommender (or the default built-in one) are not tracked.
	VPAPrometheusRecommender = "prometheus"
)

// HasPrometheusRecommender reports whether vpa specifies exactly the "prometheus"
// external recommender. VPAs with no recommenders (default built-in) or a
// different recommender name are not managed by WVA.
func HasPrometheusRecommender(vpa *vpav1.VerticalPodAutoscaler) bool {
	if len(vpa.Spec.Recommenders) != 1 {
		return false
	}
	return vpa.Spec.Recommenders[0] != nil && vpa.Spec.Recommenders[0].Name == VPAPrometheusRecommender
}

// IsSynthetic reports whether va was synthesized from annotations on a ScaledObject or HPA
// rather than read from a VariantAutoscaling CRD instance.
// Synthetic VAs exist only in-memory and must never be written to the Kubernetes API server.
func IsSynthetic(va *wvav1alpha1.VariantAutoscaling) bool {
	return va.GetAnnotations()[annotations.Synthetic] == "true"
}

// VariantAutoscalingFromScaledObject builds an in-memory VariantAutoscaling from a KEDA
// ScaledObject that bears the llm-d.ai/managed: "true" annotation.
// Returns an error if required annotations are absent or the scaleTargetRef is empty.
func VariantAutoscalingFromScaledObject(so *kedav1alpha1.ScaledObject) (*wvav1alpha1.VariantAutoscaling, error) {
	parsed, err := annotations.Parse(so)
	if err != nil {
		return nil, err
	}
	if so.Spec.ScaleTargetRef == nil || so.Spec.ScaleTargetRef.Name == "" {
		return nil, fmt.Errorf("ScaledObject %s/%s has no scaleTargetRef", so.Namespace, so.Name)
	}

	kind := so.Spec.ScaleTargetRef.Kind
	if kind == "" {
		kind = KindDeployment
	}
	apiVersion := so.Spec.ScaleTargetRef.APIVersion
	if apiVersion == "" {
		apiVersion = DeploymentGV
	}

	minReplicas := so.Spec.MinReplicaCount
	maxReplicas := so.GetHPAMaxReplicas()

	return &wvav1alpha1.VariantAutoscaling{
		ObjectMeta: metav1.ObjectMeta{
			Name:      so.Name,
			Namespace: so.Namespace,
			Labels:    so.Labels,
			Annotations: map[string]string{
				annotations.Synthetic: "true",
			},
		},
		Spec: wvav1alpha1.VariantAutoscalingSpec{
			ScaleTargetRef: autoscalingv2.CrossVersionObjectReference{
				APIVersion: apiVersion,
				Kind:       kind,
				Name:       so.Spec.ScaleTargetRef.Name,
			},
			ModelID:     parsed.ModelID,
			MinReplicas: minReplicas,
			MaxReplicas: maxReplicas,
			VariantAutoscalingConfigSpec: wvav1alpha1.VariantAutoscalingConfigSpec{
				VariantCost: parsed.VariantCost,
			},
		},
	}, nil
}

// VariantAutoscalingFromHPA builds an in-memory VariantAutoscaling from a Kubernetes HPA
// that bears the llm-d.ai/managed: "true" annotation.
// Returns an error if required annotations are absent or the scaleTargetRef is empty.
func VariantAutoscalingFromHPA(hpa *autoscalingv2.HorizontalPodAutoscaler) (*wvav1alpha1.VariantAutoscaling, error) {
	parsed, err := annotations.Parse(hpa)
	if err != nil {
		return nil, err
	}
	if hpa.Spec.ScaleTargetRef.Name == "" {
		return nil, fmt.Errorf("HPA %s/%s has no scaleTargetRef", hpa.Namespace, hpa.Name)
	}

	kind := hpa.Spec.ScaleTargetRef.Kind
	if kind == "" {
		kind = KindDeployment
	}
	apiVersion := hpa.Spec.ScaleTargetRef.APIVersion
	if apiVersion == "" {
		apiVersion = DeploymentGV
	}

	minReplicas := ptr.To(int32(1))
	if hpa.Spec.MinReplicas != nil {
		minReplicas = hpa.Spec.MinReplicas
	}

	return &wvav1alpha1.VariantAutoscaling{
		ObjectMeta: metav1.ObjectMeta{
			Name:      hpa.Name,
			Namespace: hpa.Namespace,
			Labels:    hpa.Labels,
			Annotations: map[string]string{
				annotations.Synthetic: "true",
			},
		},
		Spec: wvav1alpha1.VariantAutoscalingSpec{
			ScaleTargetRef: autoscalingv2.CrossVersionObjectReference{
				APIVersion: apiVersion,
				Kind:       kind,
				Name:       hpa.Spec.ScaleTargetRef.Name,
			},
			ModelID:     parsed.ModelID,
			MinReplicas: minReplicas,
			MaxReplicas: hpa.Spec.MaxReplicas,
			VariantAutoscalingConfigSpec: wvav1alpha1.VariantAutoscalingConfigSpec{
				VariantCost: parsed.VariantCost,
			},
		},
	}, nil
}

// VariantAutoscalingFromVPA builds an in-memory VariantAutoscaling from a Kubernetes VPA
// that bears the llm-d.ai/managed: "true" annotation.
// Returns an error if required annotations are absent or spec.targetRef is nil.
//
// VPA has no native min/max replica fields; those horizontal bounds default to 1.
// The first entry of spec.resourcePolicy.resourceClaimPolicies is carried on the
// synthetic VA's ResourceClaimPolicy field so the VerticalActuator can identify
// which ResourceClaimTemplate to patch and which device capacities it controls.
func VariantAutoscalingFromVPA(vpa *vpav1.VerticalPodAutoscaler) (*wvav1alpha1.VariantAutoscaling, error) {
	parsed, err := annotations.Parse(vpa)
	if err != nil {
		return nil, err
	}
	if vpa.Spec.TargetRef == nil || vpa.Spec.TargetRef.Name == "" {
		return nil, fmt.Errorf("VPA %s/%s has no targetRef", vpa.Namespace, vpa.Name)
	}

	kind := vpa.Spec.TargetRef.Kind
	if kind == "" {
		kind = KindDeployment
	}
	apiVersion := vpa.Spec.TargetRef.APIVersion
	if apiVersion == "" {
		apiVersion = DeploymentGV
	}

	var resourceClaimPolicy *vpav1.ResourceClaimPolicy
	if vpa.Spec.ResourcePolicy != nil && len(vpa.Spec.ResourcePolicy.ResourceClaimPolicies) > 0 {
		p := vpa.Spec.ResourcePolicy.ResourceClaimPolicies[0]
		resourceClaimPolicy = &p
	}

	return &wvav1alpha1.VariantAutoscaling{
		ObjectMeta: metav1.ObjectMeta{
			Name:      vpa.Name,
			Namespace: vpa.Namespace,
			Labels:    vpa.Labels,
			Annotations: map[string]string{
				annotations.Synthetic: "true",
			},
		},
		Spec: wvav1alpha1.VariantAutoscalingSpec{
			ScaleTargetRef: autoscalingv2.CrossVersionObjectReference{
				APIVersion: apiVersion,
				Kind:       kind,
				Name:       vpa.Spec.TargetRef.Name,
			},
			ModelID:     parsed.ModelID,
			MinReplicas: ptr.To(int32(1)),
			MaxReplicas: 1,
			VariantAutoscalingConfigSpec: wvav1alpha1.VariantAutoscalingConfigSpec{
				VariantCost: parsed.VariantCost,
			},
			ResourceClaimPolicy: resourceClaimPolicy,
		},
	}, nil
}
