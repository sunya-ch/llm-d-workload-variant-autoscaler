package discovery

import (
	"context"

	corev1 "k8s.io/api/core/v1"
	resourcev1beta1 "k8s.io/api/resource/v1beta1"
	"sigs.k8s.io/controller-runtime/pkg/client"

	"github.com/llm-d/llm-d-workload-variant-autoscaler/internal/constants"
)

// ResourceCapacityInventory tracks available and claimed DRA device capacity
// by reading ResourceSlices and ResourceClaims from the controller-runtime
// client cache (informer-backed, no per-tick network round-trips).
//
// Snapshot() returns (nil, nil) when the DRA CRD is absent — callers must
// treat a nil draAvailable map as "DRA unavailable this tick".
type ResourceCapacityInventory struct {
	client    client.Client
	draAbsent bool // set once at construction when DRA CRD is not installed
}

// NewResourceCapacityInventory creates an inventory backed by the given client.
// Set draAbsent=true when CheckDRACRD reports the CRD is not installed; all
// Snapshot() calls will return (nil, nil) without querying the API server.
func NewResourceCapacityInventory(c client.Client, draAbsent bool) *ResourceCapacityInventory {
	return &ResourceCapacityInventory{client: c, draAbsent: draAbsent}
}

// Snapshot returns a point-in-time view of available and claimed capacity.
//
//   - draAvailable:     capacityName → total unallocated headroom across all devices.
//   - claimedByVariant: variantName → capacityName → claimed quantity (in milli-units).
//
// Variant attribution uses the pod's llm-d.ai/variant label together with
// pod.status.resourceClaimStatuses to map each ResourceClaim back to its variant.
//
// Returns (nil, nil) when the DRA CRD is absent or when the list fails.
// Callers must treat nil draAvailable as "no vertical action this tick".
//
// The capacity values use milli-units (same as resource.Quantity.MilliValue())
// so comparisons against DemandPerReplicaResource must use the same unit.
func (r *ResourceCapacityInventory) Snapshot(ctx context.Context) (
	draAvailable map[string]int64,
	claimedByVariant map[string]map[string]int64,
) {
	if r.draAbsent {
		return nil, nil
	}

	// --- Build total device capacity from ResourceSlices ---
	var sliceList resourcev1beta1.ResourceSliceList
	if err := r.client.List(ctx, &sliceList); err != nil {
		// Cache not synced or DRA unavailable — degrade gracefully.
		return nil, nil
	}

	totalCapacity := map[string]int64{} // capacityName → sum over all devices
	for i := range sliceList.Items {
		for _, dev := range sliceList.Items[i].Spec.Devices {
			if dev.Basic == nil {
				continue
			}
			for capName, capVal := range dev.Basic.Capacity {
				totalCapacity[string(capName)] += capVal.Value.MilliValue()
			}
		}
	}

	// --- Build claim → variant mapping via pod.status.resourceClaimStatuses ---
	// Pods carry a llm-d.ai/variant label whose value is the variant name.
	// pod.status.resourceClaimStatuses[*].resourceClaimName gives the exact
	// ResourceClaim name for each ResourceClaimTemplate referenced by the pod.
	// Key: "namespace/claimName"
	claimVariant := map[string]string{}
	var podList corev1.PodList
	if err := r.client.List(ctx, &podList); err == nil {
		for i := range podList.Items {
			pod := &podList.Items[i]
			variantName := pod.Labels[constants.VariantLabelKey]
			if variantName == "" {
				continue
			}
			for _, cs := range pod.Status.ResourceClaimStatuses {
				if cs.ResourceClaimName == nil {
					continue
				}
				key := pod.Namespace + "/" + *cs.ResourceClaimName
				claimVariant[key] = variantName
			}
		}
	}

	// --- Build claimed capacity from ResourceClaims ---
	var claimList resourcev1beta1.ResourceClaimList
	if err := r.client.List(ctx, &claimList); err != nil {
		return nil, nil
	}

	claimedTotal := map[string]int64{} // capacityName → sum claimed across all claims
	claimedByVariant = map[string]map[string]int64{}

	for i := range claimList.Items {
		claim := &claimList.Items[i]
		if claim.Status.Allocation == nil {
			continue
		}
		variantName := claimVariant[claim.Namespace+"/"+claim.Name]

		for _, res := range claim.Status.Allocation.Devices.Results {
			for capName, qty := range res.ConsumedCapacity {
				millis := qty.MilliValue()
				claimedTotal[string(capName)] += millis
				if variantName == "" {
					continue
				}
				if claimedByVariant[variantName] == nil {
					claimedByVariant[variantName] = map[string]int64{}
				}
				claimedByVariant[variantName][string(capName)] += millis
			}
		}
	}

	// Available = total − claimed.
	draAvailable = make(map[string]int64, len(totalCapacity))
	for capName, total := range totalCapacity {
		draAvailable[capName] = total - claimedTotal[capName]
	}

	return draAvailable, claimedByVariant
}
