package interfaces

import (
	"context"
	"time"
)

// Analyzer is the common interface for all scaling analyzers.
// Each analyzer observes workload metrics and produces capacity signals
// (required_capacity, spare_capacity) that the engine combines into
// scaling decisions. Analyzers do NOT build scaling plans — the engine does.
//
// Saturation Analyzer V2 is the first implementation of this interface.
// Future analyzers (throughput, SLO) will implement the same interface.
type Analyzer interface {
	// Name returns the analyzer's identifier (e.g., "saturation", "throughput", "slo").
	Name() string

	// Analyze computes capacity signals for a model across all its variants.
	// Returns per-variant capacity breakdown and model-level scaling signals.
	Analyze(ctx context.Context, input AnalyzerInput) (*AnalyzerResult, error)
}

// AnalyzerConfig is the interface for analyzer-specific configuration.
// Each analyzer defines its own config type that implements this interface.
type AnalyzerConfig interface {
	// GetAnalyzerName returns the name of the analyzer this config is for.
	GetAnalyzerName() string
}

// AnalyzerInput is the common input provided to all analyzers.
type AnalyzerInput struct {
	ModelID        string
	Namespace      string
	ReplicaMetrics []ReplicaMetrics
	VariantStates  []VariantReplicaState
	Config         AnalyzerConfig

	// SchedulerQueue holds model-level queue metrics from the llm-d inference
	// scheduler flow control layer. These represent requests queued upstream
	// of any pod and are not yet attributed to a specific variant or role.
	// Any analyzer with a demand model may convert this into per-analyzer
	// demand using its own unit (e.g., kv-tokens for saturation_v2,
	// tokens/sec for a future throughput analyzer). Demand attribution to
	// roles or variants is each analyzer's choice.
	// Nil when flow control is disabled or metrics are unavailable.
	SchedulerQueue *SchedulerQueueMetrics
}

// SchedulerQueueMetrics holds model-level queue metrics from the llm-d
// inference scheduler flow control layer (inference_extension_flow_control_*).
// These are model-scoped, not per-pod, since the scheduler queues requests
// before routing them to a specific backend pod.
//
// TODO(#2309): The upstream metrics lack a namespace label. If the same model
// name exists in different namespaces, these values may include cross-namespace
// data. Once the upstream adds a namespace label, queries should filter by it.
type SchedulerQueueMetrics struct {
	// QueueSize is the number of requests currently queued in the
	// scheduler's flow control layer for this model.
	// Sourced from inference_extension_flow_control_queue_size.
	QueueSize int64

	// QueueBytes is the total bytes of request bodies currently queued
	// in the scheduler's flow control layer for this model.
	// Sourced from inference_extension_flow_control_queue_bytes.
	// Approximate token count: QueueBytes / BytesPerToken.
	QueueBytes int64
}

// RoleCapacity holds per-role capacity aggregation for P/D disaggregated models.
type RoleCapacity struct {
	Role        string
	TotalSupply float64
	TotalDemand float64
	// TotalAnticipatedSupply is the per-role anticipated supply used by the
	// engine's universal threshold post-step for the RC formula. Analyzer-
	// supplied; the engine reads it as-is (zero is a literal value, not a
	// sentinel). Use aggregation.AggregateByRole to populate this correctly.
	TotalAnticipatedSupply float64
	RequiredCapacity       float64
	SpareCapacity          float64
}

// AnalyzerResult is the common output produced by all analyzers.
// The engine consumes these results to build scaling plans.
type AnalyzerResult struct {
	// AnalyzerName identifies which analyzer produced this result.
	AnalyzerName string

	ModelID    string
	Namespace  string
	AnalyzedAt time.Time

	// Per-variant capacity breakdown (in analyzer-specific units).
	VariantCapacities []VariantCapacity

	// Model-level aggregates (in analyzer-specific units).
	TotalSupply float64 // Sum of all variant TotalCapacity
	TotalDemand float64 // Sum of all variant TotalDemand
	Utilization float64 // TotalDemand / TotalSupply (0.0-1.0)

	// TotalAnticipatedSupply is the anticipated supply the engine's universal
	// threshold post-step subtracts from TotalDemand/scaleUp to compute RC.
	// Analyzer-supplied; the engine reads it as-is (zero is a literal value,
	// not a sentinel). Saturation V2 sets this to
	// Σ_v (ReplicaCount + PendingReplicas) × PerReplicaCapacity so pending
	// replicas count against demand, preventing double-scaling. Use
	// aggregation.SumTotalAnticipatedSupply to populate this correctly.
	TotalAnticipatedSupply float64

	// Scaling signals — written by the engine post-step; read by the optimizer.
	// The engine overwrites both fields after each analyzer's Analyze() returns:
	//   RC = max(0, TotalDemand/scaleUp − TotalAnticipatedSupply)
	//   SC = max(0, TotalSupply    − TotalDemand/scaleDown)
	// Analyzer-written values are discarded; analyzers should leave these zero.
	RequiredCapacity float64 // >0 means scale-up needed
	SpareCapacity    float64 // >0 means scale-down possible

	// RoleCapacities holds per-role capacity aggregation for P/D disaggregated models.
	// nil when no disaggregation is active (all variants are role "both").
	RoleCapacities map[string]RoleCapacity
}

// VariantCapacity holds per-variant capacity data in analyzer-specific units.
// For saturation: units are tokens. For throughput: tokens/sec. For SLO: latency-constrained capacity.
type VariantCapacity struct {
	VariantName     string
	AcceleratorName string
	Cost            float64
	Role            string // "prefill", "decode", "both", "" (empty = non-disaggregated)

	ReplicaCount    int
	PendingReplicas int

	// PerReplicaCapacity is the representative capacity per replica.
	// For saturation V2: median(effectiveCapacity) in tokens across ready replicas.
	PerReplicaCapacity float64

	// Reason is a free-text string set by the analyzer to describe how the
	// variant's per-replica capacity was computed. Empty for analyzers that
	// do not set it. Saturation V2 uses "P0-store", "P1-obs", "P2-hist",
	// "P3-k2", "P4-k1", "no-data", "error". Throughput uses "T1-ols",
	// "T2-pinned", "T2-default", "T2-failed".
	Reason string

	// TotalCapacity is ReplicaCount × PerReplicaCapacity.
	TotalCapacity float64

	// TotalDemand is the aggregate demand on this variant.
	TotalDemand float64

	// Utilization is TotalDemand / TotalCapacity (0.0-1.0).
	Utilization float64

	// VerticalHint holds vertical scaling demand/supply signals for this variant.
	// Non-nil only when a demand/supply imbalance exists AND the analyzer has
	// sufficient empirical data. See VerticalHint for the exact nil conditions.
	VerticalHint *VerticalHint
}

// VerticalHint holds the vertical scaling demand/supply signals for a variant.
// Set only when a demand/supply imbalance exists:
//   - scale-up:   demand > supply  (RequiredCapacity > 0 at model level)
//   - scale-down: supply > demand + headroom (SpareCapacity > 0 at model level)
//
// Nil when supply and demand are balanced, the analyzer lacks empirical data
// (MaxComputeIntensity == 0 for sat V2; LearnedParameters == nil for QM),
// or no ready replicas are available.
type VerticalHint struct {
	// ScaleUpPerReplicaCapacity is the target per-replica capacity needed to
	// absorb all demand with the current replica count (no horizontal spill).
	// Sat V2: tokens (= totalDemand / readyCount).
	// QM:     req/s (= queueAnalyzer.Size at tighter SLO).
	ScaleUpPerReplicaCapacity float64

	// ScaleDownPerReplicaCapacity is the minimum per-replica capacity that
	// leaves zero spare capacity.
	// Sat V2: tokens (= totalCapacity / readyCount).
	// QM:     req/s (= queueAnalyzer.Size at looser SLO).
	ScaleDownPerReplicaCapacity float64

	// DemandPerReplicaResource is the Step-policy-rounded compute and memory
	// resource targets needed to serve ScaleUpPerReplicaCapacity (scale-up) or
	// ScaleDownPerReplicaCapacity (scale-down).
	// Nil when the observation store is not yet bootstrapped
	// (MaxComputeIntensity == 0 for sat V2; LearnedParameters == nil for QM).
	DemandPerReplicaResource *ResourceRequirement
}

// ResourceRequirement holds the compute and memory resource targets for a vertical scaling step.
type ResourceRequirement struct {
	// ComputeFraction is the %Threads target after Step policy rounding (0.0–1.0).
	// Sat V2: roundUp(ComputeIntensity / MaxComputeIntensity).
	// QM:     roundUp(scaleUpRPS × (avgInputTokens + α×avgOutputTokens) / MaxComputeIntensity).
	ComputeFraction float64
	// MemoryBytes is the memory target in bytes after Step policy rounding.
	// Sat V2: round(MemoryWeight + BytePerToken × ScaleUpPerReplicaCapacity).
	// QM:     round(MemoryWeight + BytePerToken × (scaleUpRPS × (avgInputTokens + avgOutputTokens))).
	MemoryBytes int64
}
