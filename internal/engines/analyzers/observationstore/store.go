package observationstore

import (
	"sync"
	"time"

	"github.com/llm-d/llm-d-workload-variant-autoscaler/internal/interfaces"
)

// VariantObservation holds empirically learned vertical scaling signals for one variant.
// All fields start at zero and converge toward stable values over successive ticks.
// Corresponds to "CapacityStore — Addition information" in vertical-scaling-logic.md.
type VariantObservation struct {
	// ComputeIntensity is the current-tick I_live = PromptTokenRate + α×GenerationTokenRate
	// (mean across ready replicas). Updated every tick regardless of queue state.
	// Used by computeVerticalHint for the ComputeDemand fraction (I_live / I_max).
	ComputeIntensity float64

	// MaxComputeIntensity is the high-water I_live observed while QueueLength > 0
	// (empirical compute wall, I_max). Updated only when queue is non-empty;
	// zero until first saturated tick.
	MaxComputeIntensity float64

	// MemoryWeight is model-weight memory in bytes, derived once from
	// TotalKvCapacityTokens and GpuMemoryUtilization:
	//   MemoryWeight = TotalKvCapacityTokens × (1−U)/U × BytesPerKVToken
	// Set on the first tick where TotalKvCapacityTokens > 0; never overwritten.
	MemoryWeight float64

	// BytePerToken is the empirically derived KV cache bytes per active token,
	// computed as DeltaCacheBytes / DeltaTokens. Refreshed each tick when deltas are available.
	BytePerToken float64

	// UpdatedAt is the last time this observation was written.
	// Used by EvictStale to remove entries for deleted variants.
	UpdatedAt time.Time
}

// VariantObservationStore is a thread-safe store of VariantObservation keyed by
// "namespace|modelID|variantName". Written by whichever analyzer is active each
// tick (sat V2 or QM) via UpdateFromReplicaMetrics; read by both when computing VerticalHint.
type VariantObservationStore struct {
	mu   sync.RWMutex
	data map[string]*VariantObservation
}

// NewVariantObservationStore creates an empty store.
func NewVariantObservationStore() *VariantObservationStore {
	return &VariantObservationStore{data: make(map[string]*VariantObservation)}
}

// Update writes (or replaces) the observation for the given key.
func (s *VariantObservationStore) Update(namespace, modelID, variantName string, obs VariantObservation) {
	obs.UpdatedAt = time.Now()
	key := namespace + "|" + modelID + "|" + variantName
	s.mu.Lock()
	s.data[key] = &obs
	s.mu.Unlock()
}

// Get returns a copy of the stored observation, or nil if absent.
// Callers must not modify the returned value's fields — it is a copy for safety.
func (s *VariantObservationStore) Get(namespace, modelID, variantName string) *VariantObservation {
	key := namespace + "|" + modelID + "|" + variantName
	s.mu.RLock()
	obs := s.data[key]
	s.mu.RUnlock()
	if obs == nil {
		return nil
	}
	cp := *obs
	return &cp
}

// EvictStale removes entries not updated within timeout. Returns the number evicted.
func (s *VariantObservationStore) EvictStale(timeout time.Duration) int {
	cutoff := time.Now().Add(-timeout)
	s.mu.Lock()
	defer s.mu.Unlock()
	n := 0
	for k, obs := range s.data {
		if obs.UpdatedAt.Before(cutoff) {
			delete(s.data, k)
			n++
		}
	}
	return n
}

// UpdateFromReplicaMetrics computes and persists vertical scaling signals
// from raw per-replica metrics for a single variant. Called by both sat V2
// and QM analyzers at the start of each Analyze() call.
//
// Signal update rules:
//   - ComputeIntensity: mean I_live = PromptTokenRate + α×GenerationTokenRate across all replicas.
//   - MaxComputeIntensity: high-water I_live only when QueueLength > 0 on that replica.
//   - MemoryWeight: derived once from TotalKvCapacityTokens × (1−U)/U × BytesPerKVToken;
//     never overwritten once set. U = rm.GpuMemoryUtilization (falls back to 0.9 when zero).
//   - BytePerToken: DeltaCacheBytes / DeltaTokens; refreshed each tick when both > 0.
func (s *VariantObservationStore) UpdateFromReplicaMetrics(
	namespace, modelID, variantName string,
	replicaMetrics []interfaces.ReplicaMetrics,
	alpha, bytesPerKVToken float64,
) {
	existing := s.Get(namespace, modelID, variantName)
	var maxI, memWeight, bpt float64
	if existing != nil {
		maxI, memWeight, bpt = existing.MaxComputeIntensity, existing.MemoryWeight, existing.BytePerToken
	}

	var iLiveSum float64
	var iLiveCount int
	for _, rm := range replicaMetrics {
		iLive := rm.PromptTokenRate + alpha*rm.GenerationTokenRate
		iLiveSum += iLive
		iLiveCount++
		// I_max: high-water under non-empty queue (spec: vllm:num_requests_waiting > 0)
		if rm.QueueLength > 0 && iLive > maxI {
			maxI = iLive
		}
		if memWeight == 0 && rm.TotalKvCapacityTokens > 0 {
			util := rm.GpuMemoryUtilization
			if util <= 0 {
				util = 0.9 // vLLM default for --gpu-memory-utilization
			}
			memWeight = float64(rm.TotalKvCapacityTokens) * (1 - util) / util * bytesPerKVToken
		}
		if rm.DeltaCacheBytes > 0 && rm.DeltaTokens > 0 {
			bpt = rm.DeltaCacheBytes / rm.DeltaTokens
		}
	}

	currentI := 0.0
	if iLiveCount > 0 {
		currentI = iLiveSum / float64(iLiveCount)
	}

	s.Update(namespace, modelID, variantName, VariantObservation{
		ComputeIntensity:    currentI,
		MaxComputeIntensity: maxI,
		MemoryWeight:        memWeight,
		BytePerToken:        bpt,
	})
}
