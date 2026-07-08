# Vertical Scaling Plan

## Top-Level Overview

Introduce vertical scaling support — adjusting per-replica GPU resource allocation via DRA — alongside the existing horizontal scaling. The design is a **"vertical-first, horizontal-spill"** model:

- **Scale-up path**: prefer growing per-replica GPU resources before adding replicas; spill to horizontal only when per-replica headroom is exhausted or unavailable.
- **Scale-down path**: try horizontal scale-down first (cheaper, no pod restart); only shrink per-replica resources when horizontal is blocked (e.g., already at `minReplicas`).

### Two analyzer paths — one optimizer/actuation

Both the **Saturation V2** and **Queueing Model (QM)** analyzers can drive vertical scaling. They produce the same shape of output (`VerticalHint` on `VariantCapacity`) using different internal mechanisms, then share identical optimizer and actuation sub-tasks downstream.

| Concern | Saturation V2 | QM Analyzer |
| --- | --- | --- |
| **Capacity unit** | tokens | req/s |
| **Scale-up target** | `totalDemand / readyCount` tokens/replica | `queueAnalyzer.Size` at tighter SLO multiplier |
| **Scale-down floor** | `totalCapacity / readyCount` tokens/replica | `queueAnalyzer.Size` at looser SLO multiplier |
| **ComputeDemand** | `medianI / medianMaxI` fraction (0–1) | `scaleUpRPS × (I + α×O) / I_max` |
| **MemoryDemand** | `MemoryWeight + BytePerToken × ScaleUpPRC` bytes | `MemoryWeight + BytePerToken × (scaleUpRPS × (I+O))` bytes |
| **Bootstrap requirement** | `MaxComputeIntensity > 0` (first saturated observation) | `LearnedParameters != nil` (Kalman tuner converged) |
| **New metric collection** | `PromptTokenRate`, `DeltaCacheBytes`, `DeltaTokens` | none — learned parameters already encode service-rate characteristics |

### Non-goals

- Helm chart changes (deprecated).
- V1 (percentage-based) saturation path.
- Throughput analyzer.

---

## Background: Logic from `vertical-scaling-logic.md`

### Compute Intensity (Saturation V2 path)

$$I_{live} = R_p + (\alpha \times R_g)$$

where $\alpha = 0.10$ (`ComputeIntensityAlpha`), and:

- $R_p$ = `PromptTokenRate` — rate(vllm:prompt\_tokens\_total[Δt])
- $R_g$ = `GenerationTokenRate` (already on `ReplicaMetrics`) — rate(vllm:generation\_tokens\_total[Δt])

The **Empirical Compute Wall** $I_{max}$ is the high-water $I_{live}$ observed while `QueueLength > 0`. It must be bootstrapped from a fully-resourced replica.

The **%Thread requirement**:

$$T_{req} = \max\!(\frac{I_{live}}{I_{max}} \times 100\%,\; minAllowed_{compute})$$

### QM Service Rate (QM path)

The QM analyzer learns `(alpha, beta, gamma)` via Kalman filter, encoding the GPU's service-time characteristics holistically. `queueAnalyzer.Size(targetPerf)` already computes `maxRequestRate` — the SLO-constrained throughput ceiling per replica. This single number already integrates both compute and memory effects.

Calling `Size` at a **tighter** SLO multiplier ($k_{up} < k_{nominal}$) gives the scale-up target throughput; at a **looser** multiplier ($k_{down} > k_{nominal}$) gives the scale-down floor — directly mapping to `ScaleUpPerReplicaCapacity` and `ScaleDownPerReplicaCapacity` without any extra metric collection.

### Memory Requirement (Saturation V2 path)

**Model weight floor** (derived once, set on first tick where `TotalKvCapacityTokens > 0`):

The spec formula ([`vertical-scaling-logic.md`](./vertical-scaling-logic.md)):
$$\text{Model Weight} = TotalKvCapacityTokens \times \frac{(1-U)}{U}$$

This gives the number of **tokens** occupied by model weights (the non-KV-cache portion of GPU memory).
To convert to **bytes** for comparison against DRA memory capacity dimensions, we multiply by `BytesPerKVToken`:
$$M_{weights} = TotalKvCapacityTokens \times \frac{1 - U}{U} \times BytesPerKVToken$$

where $U$ = `VLLMParams.GpuMemoryUtilization` (default `0.9` from `deployment_parser.go`), and `BytesPerKVToken = 128` bytes/token.

> **Note on `BytesPerKVToken`**: the spec's formula produces a token count; we apply `BytesPerKVToken` to convert to bytes. This constant (128 B/token) covers most 7B–70B BF16 models; it is replaced by the empirical `BytePerToken` once live KV-cache delta data is available.

**`QueueLength` maps to spec's `vllm:num_requests_waiting`**: `ReplicaMetrics.QueueLength` is collected from `vllm:num_requests_waiting`. The spec condition "I_max is I_live when vllm:num_requests_waiting > 0" maps directly to `rm.QueueLength > 0` in the implementation.

**Bytes-per-token from live KV cache deltas** (refreshed each tick):
$$Bpt = \frac{DeltaCacheBytes}{DeltaTokens}$$

**Required Memory at ScaleUpPerReplicaCapacity**:

The spec formula:
$$\text{Required Memory} = M_{\text{weights}} + (Bpt \times N_{\text{activeTokens}}) + M_{\text{overhead}}$$

where $N_{\text{activeTokens}}$ is capped by `ScaleUpPerReplicaCapacity` (the target token capacity when saturated).

Implementation as `MemoryDemand`:
$$\text{MemoryDemand} = M_{weights} + Bpt \times ScaleUpPRC$$

> **`M_overhead`**: treated as zero for the initial implementation. Reserved for future use when GPU driver/framework overhead becomes measurable.

## Sub-Tasks

---

### Sub-Task 1 — Shared: `VerticalHint` on `VariantCapacity`

#### Intent — Sub-Task 1

Define the common output type that both analyzers populate. The optimizer and actuator only consume `VerticalHint` — they are unaware of which analyzer produced it.

#### File: [`internal/interfaces/analyzer.go`](../../../internal/interfaces/analyzer.go)

Add after the existing `VariantCapacity` struct (line 155):

```go
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

    // ComputeDemand is the estimated compute requirement as a %Threads fraction (0.0–1.0).
    // Sat V2: medianComputeIntensity / medianMaxComputeIntensity.
    // QM:     scaleUpRPS × (avgInputTokens + α×avgOutputTokens) / MaxComputeIntensity.
    // Zero when MaxComputeIntensity is not yet available.
    ComputeDemand float64

    // MemoryDemand is the estimated memory requirement in bytes.
    // Sat V2: MemoryWeight + BytePerToken × ScaleUpPerReplicaCapacity.
    // QM:     MemoryWeight + BytePerToken × (scaleUpRPS × (avgInputTokens + avgOutputTokens)).
    // Zero when MemoryWeight / BytePerToken are not yet available.
    MemoryDemand float64

    // DemandPerReplicaResource is the estimated resource requirement to achieve
    // ScaleUpPerReplicaCapacity (when ScaleUp) or ScaleDownPerReplicaCapacity (when ScaleDown).
    // Derived by applying the Step range policy to ComputeDemand and MemoryDemand, then
    // computing the target capacity inversely from the rounded resource values.
    // Nil when ComputeDemand == 0 and MemoryDemand == 0 (observation store not ready).
    DemandPerReplicaResource *ResourceRequirement
}

// ResourceRequirement holds the compute and memory resource targets for a vertical scaling step.
type ResourceRequirement struct {
    // ComputeFraction is the %Threads target after Step policy rounding (0.0–1.0).
    ComputeFraction float64
    // MemoryBytes is the memory target in bytes after Step policy rounding.
    MemoryBytes int64
}
```

Add `VerticalHint *VerticalHint` field to `VariantCapacity`:

```go
type VariantCapacity struct {
    // ... existing fields unchanged ...

    // VerticalHint holds vertical scaling demand/supply signals for this variant.
    // Non-nil only when a demand/supply imbalance exists AND the analyzer has
    // sufficient empirical data. See VerticalHint for the exact conditions.
    VerticalHint *VerticalHint
}
```

#### Expected Outcomes — Sub-Task 1

- `VerticalHint` and `ResourceRequirement` structs added to [`internal/interfaces/analyzer.go`](../../../internal/interfaces/analyzer.go).
- `VerticalHint *VerticalHint` field appended to `VariantCapacity`.
- No other code changes in this sub-task.

#### Todo List — Sub-Task 1

1. Add `VerticalHint` struct (with `DemandPerReplicaResource *ResourceRequirement`) to [`internal/interfaces/analyzer.go`](../../../internal/interfaces/analyzer.go) after line 155.
2. Add `ResourceRequirement` struct to the same file.
3. Add `VerticalHint *VerticalHint` field to `VariantCapacity`.

**Status** — `[ ] pending`

---

### Sub-Task 2a — Shared `VariantObservationStore` and compute intensity tracking

#### Intent — Sub-Task 2a

`MaxComputeIntensity`, `MemoryWeight`, and `BytePerToken` are **learned signals** that accumulate across ticks from live observations. They must be accessible to both the saturation V2 and QM analyzers.

> **Design decision vs. spec**: [`vertical-scaling-logic.md`](./vertical-scaling-logic.md) says "CapacityStore — Addition information: ComputeIntensity, MaxComputeIntensity, MemoryWeight, BytePerToken", implying these signals be stored in the existing `CapacityKnowledgeStore`. We instead introduce a new **`VariantObservationStore`** for two reasons:
>
> 1. `CapacityKnowledgeStore` lives in `saturation_v2` — importing it from `queueingmodel` would create a circular dependency.
> 2. The signals are needed by *both* analyzers; the store must live in a **neutral package** that neither analyzer owns.
>
> The effect is identical to the spec: both analyzers read and write the four signals. The implementation location is different.

The clean solution is a **shared `VariantObservationStore`** — a new, minimal, thread-safe store in [`internal/engines/analyzers/observationstore`](../../../internal/engines/analyzers/observationstore) that is:

- **Written by both analyzers** — all three signals are computable purely from `ReplicaMetrics` fields (available to every analyzer), so both sat V2 and QM write to the store each tick using a shared helper
- **Read** by both analyzers when computing `VerticalHint`
- **Owned** by the engine and injected into both analyzers at construction time

This matters because sat V2 and QM are **mutually exclusive** per tick (engine `switch` selects one path). If only sat V2 wrote the store, a QM-only deployment would never populate it and `ComputeDemand`/`MemoryDemand` would always be zero.

This mirrors the existing `CapacityKnowledgeStore` pattern but is scoped to vertical scaling signals only and lives in a neutral package to avoid any circular dependency.

#### New file: [`internal/engines/analyzers/observationstore/store.go`](../../../internal/engines/analyzers/observationstore/store.go)

```go
package observationstore

import (
    "sync"
    "time"
)

// VariantObservation holds empirically learned vertical scaling signals for one variant.
// All fields start at zero and converge toward stable values over successive ticks.
// Corresponds to "CapacityStore — Addition information" in vertical-scaling-logic.md.
type VariantObservation struct {
    // ComputeIntensity is the current-tick I_live = PromptTokenRate + α×GenerationTokenRate
    // (median across ready replicas). Updated every tick regardless of queue state.
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

// Get returns a pointer to the stored observation, or nil if absent.
// The returned pointer is a copy — callers must not modify it.
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
```

#### New file: [`internal/engines/analyzers/constants.go`](../../../internal/engines/analyzers/constants.go)

Package name `analyzerconstants` (path `internal/engines/analyzers/constants.go`, but Go package name `analyzerconstants` to avoid clashing with `saturation_v2`'s local `constants.go`).

```go
package analyzerconstants

const (
    // BytesPerKVToken is the approximate GPU memory bytes per KV-cache token.
    // Used as a fallback when live BytePerToken has not yet been observed.
    // Covers most 7B–70B BF16 models.
    BytesPerKVToken = 128

    // ComputeIntensityAlpha is the generation-token weighting coefficient in
    // I_live = PromptTokenRate + alpha × GenerationTokenRate.
    // Captures the lower parallel processing density of token-by-token generation.
    ComputeIntensityAlpha = 0.1
)
```

**New `ReplicaMetrics` fields** ([`internal/interfaces/saturation_analyzer.go`](../../../internal/interfaces/saturation_analyzer.go))

Append after `VLLMRequestRate` (line 144), before the closing `}`:

```go
// --- Fields for Vertical Scaling ---

// PromptTokenRate is rate(vllm:prompt_tokens_total[Δt]) on this replica (tokens/s).
// Used to compute I_live = PromptTokenRate + α × GenerationTokenRate.
// Zero when metrics are unavailable.
PromptTokenRate float64

// DeltaCacheBytes is ΔCacheBytesUsed over the last collection interval.
// Computed as Δ(vllm:gpu_cache_usage_perc × vllm:available_kv_cache_memory_bytes).
// Used to derive BytePerToken = DeltaCacheBytes / DeltaTokens.
// Zero when metrics are unavailable.
DeltaCacheBytes float64

// DeltaTokens is Δ(prompt_tokens_total + generation_tokens_total) over the last interval.
// Used to derive BytePerToken = DeltaCacheBytes / DeltaTokens.
// Zero when metrics are unavailable.
DeltaTokens float64
```

**Shared write helper** (new function in [`internal/engines/analyzers/observationstore/store.go`](../../../internal/engines/analyzers/observationstore/store.go))

All three signals depend only on `ReplicaMetrics` fields — no analyzer-private state. The logic is extracted into a package-level helper called by both analyzers:

```go
// UpdateFromReplicaMetrics computes and persists vertical scaling signals
// from raw per-replica metrics for a single variant. Called by both sat V2
// and QM analyzers at the start of each Analyze() call.
//
// Signal update rules:
//   - MaxComputeIntensity: high-water I_live = PromptTokenRate + α×GenerationTokenRate
//     only when QueueLength > 0 on that replica.
//   - MemoryWeight: derived once from TotalKvCapacityTokens × (1−U)/U × BytesPerKVToken;
//     never overwritten once set. U = GpuMemoryUtilization (default 0.9).
//   - BytePerToken: DeltaCacheBytes / DeltaTokens; refreshed each tick when both > 0.
func (s *VariantObservationStore) UpdateFromReplicaMetrics(
    namespace, modelID, variantName string,
    replicaMetrics []interfaces.ReplicaMetrics,
    alpha float64,
    bytesPerKVToken float64,
)
```

#### Call sites

- **Sat V2** ([`saturation_v2/analyzer.go`](../../../internal/engines/analyzers/saturation_v2/analyzer.go)): call `a.observationStore.UpdateFromReplicaMetrics(...)` at the top of `Analyze()`, grouped by variant, **before** the `computeReplicaCapacity` loop. This replaces the previous per-replica inline write.
  - Note: sat V2 can also use `VLLMParams.GpuMemoryUtilization` from `capacityStore` for a more accurate `MemoryWeight`. If `vllmParams != nil && vllmParams.GpuMemoryUtilization > 0`, pass it as `U`; otherwise the helper uses the `0.9` default.
- **QM** ([`queueingmodel/analyzer.go`](../../../internal/engines/analyzers/queueingmodel/analyzer.go)): call `a.observationStore.UpdateFromReplicaMetrics(...)` at the top of `computeAllVariantCapacities()`, per variant, before the capacity loop. QM always uses the `0.9` default since it has no access to `VLLMParams`.

#### Implementation sketch of `UpdateFromReplicaMetrics`

```go
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

    // Compute current-tick I_live as the mean across ready replicas (those with
    // PromptTokenRate > 0 or GenerationTokenRate > 0). Used for ComputeIntensity field.
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
            util := 0.9 // default GpuMemoryUtilization
            memWeight = float64(rm.TotalKvCapacityTokens) * (1-util) / util * bytesPerKVToken
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
        ComputeIntensity:    currentI, // current tick I_live (mean across replicas)
        MaxComputeIntensity: maxI,
        MemoryWeight:        memWeight,
        BytePerToken:        bpt,
    })
}
```

Sat V2 may optionally override `util` with `VLLMParams.GpuMemoryUtilization` before calling — or call `Update` directly with the more precise `MemoryWeight` after the helper.

**Engine wiring** ([`internal/engines/saturation/engine.go`](../../../internal/engines/saturation/engine.go))

- Create one `*observationstore.VariantObservationStore` at engine construction time (alongside the existing `CapacityKnowledgeStore`).
- Inject it into `NewSaturationAnalyzer(store, obsStore)` and `NewQueueingModelAnalyzer(obsStore)`.
- Wire `EvictStale` alongside the existing capacity store eviction call in the engine's eviction loop.

**`SaturationAnalyzer` struct change** ([`saturation_v2/analyzer.go`](../../../internal/engines/analyzers/saturation_v2/analyzer.go))

```go
type SaturationAnalyzer struct {
    mu                     sync.Mutex
    computeCapacityHistory map[string]*rollingAverage
    capacityStore          *CapacityKnowledgeStore
    observationStore       *observationstore.VariantObservationStore // NEW
}

func NewSaturationAnalyzer(store *CapacityKnowledgeStore, obsStore *observationstore.VariantObservationStore) *SaturationAnalyzer {
    return &SaturationAnalyzer{
        computeCapacityHistory: make(map[string]*rollingAverage),
        capacityStore:          store,
        observationStore:       obsStore,
    }
}
```

**`QueueingModelAnalyzer` struct change** ([`queueingmodel/analyzer.go`](../../../internal/engines/analyzers/queueingmodel/analyzer.go))

```go
type QueueingModelAnalyzer struct {
    modelsParameterStore map[string]*ParameterStore
    observationStore     *observationstore.VariantObservationStore // NEW — read and write
}

func NewQueueingModelAnalyzer(obsStore *observationstore.VariantObservationStore) *QueueingModelAnalyzer {
    return &QueueingModelAnalyzer{
        modelsParameterStore: make(map[string]*ParameterStore),
        observationStore:     obsStore,
    }
}
```

#### Expected Outcomes — Sub-Task 2a

- New package `internal/engines/analyzers/observationstore` with `VariantObservation`, `VariantObservationStore`, and `UpdateFromReplicaMetrics` helper.
  - `VariantObservation` has four fields matching spec's "CapacityStore — Addition information": `ComputeIntensity`, `MaxComputeIntensity`, `MemoryWeight`, `BytePerToken`.
- New file `internal/engines/analyzers/constants.go` (package `analyzerconstants`) with `BytesPerKVToken` and `ComputeIntensityAlpha`.
- `ReplicaMetrics` gains `PromptTokenRate`, `DeltaCacheBytes`, `DeltaTokens`.
- `SaturationAnalyzer` and `QueueingModelAnalyzer` both gain `observationStore` field; both write via `UpdateFromReplicaMetrics` at the start of each `Analyze()` call.
- Engine creates one shared store instance injected into both analyzers.
- Store is always populated regardless of which analyzer path is active.
- Unit tests: `ComputeIntensity` updated every tick; `MaxComputeIntensity` only updates under non-empty queue; `MemoryWeight` set once and not overwritten; `BytePerToken` refreshed when deltas available; QM-only deployment populates the store correctly.

#### Todo List — Sub-Task 2a

1. Create [`internal/engines/analyzers/observationstore/store.go`](../../../internal/engines/analyzers/observationstore/store.go) with `VariantObservation`, `VariantObservationStore`, and `UpdateFromReplicaMetrics`.
2. Create [`internal/engines/analyzers/constants.go`](../../../internal/engines/analyzers/constants.go) with `BytesPerKVToken` and `ComputeIntensityAlpha`.
3. Add `PromptTokenRate`, `DeltaCacheBytes`, `DeltaTokens` to `ReplicaMetrics` in [`internal/interfaces/saturation_analyzer.go`](../../../internal/interfaces/saturation_analyzer.go).
4. Add `observationStore` field to `SaturationAnalyzer`; update `NewSaturationAnalyzer` signature.
5. Add `observationStore` field to `QueueingModelAnalyzer`; update `NewQueueingModelAnalyzer` signature.
6. Update engine to create one shared `VariantObservationStore` and inject it into both analyzers.
7. Call `observationStore.UpdateFromReplicaMetrics` at top of sat V2 `Analyze()` (grouped by variant, before `computeReplicaCapacity` loop).
8. Call `observationStore.UpdateFromReplicaMetrics` at top of QM `computeAllVariantCapacities()` per variant.
9. Update metric collector to populate `PromptTokenRate`, `DeltaCacheBytes`, `DeltaTokens`.
10. Unit tests for `VariantObservationStore` and both write paths.

**Status** — `[ ] pending (blocked on Sub-Task 1)`

---

### Sub-Task 2b — Saturation V2: populate `VerticalHint` in `aggregateByVariant`

#### Intent — Sub-Task 2b

Use the accumulated signals from Sub-Task 2a to compute `VerticalHint` for each variant in `aggregateByVariant`. The hint is only produced when `MaxComputeIntensity > 0` **and** a demand/supply imbalance exists (`totalDemand > totalCapacity` or `totalCapacity > totalDemand`).

The computation follows these steps from [`vertical-scaling-logic.md`](./vertical-scaling-logic.md):

1. **Determine `VerticalScaleOption`**: set when `Demand > Supply` (scale-up) or `Demand < Supply − headroom` (scale-down). If false, return nil.
2. **Compute `ComputeDemand`**: `ComputeIntensity / MaxComputeIntensity` where `ComputeIntensity` targets the expected token rate at `ScaleTargetPerReplica`. Clamped to `[0, 1]`.
3. **Compute `MemoryDemand`**: `MemoryWeight + BytePerToken × ScaleTargetPerReplica` (bytes).
4. **Apply Step range policy**: round up `ComputeDemand` and round `MemoryDemand` to the nearest step boundary defined by the VPA resource policy. Check that both values are within `[minAllowed, maxAllowed]`; if out of range, return nil.
5. **Compute `ScaleTargetPerReplica` inversely** from the rounded resource values — ensuring the emitted PRC is consistent with the resource amount the optimizer will actually request.
6. **Set `ScaleUpPerReplicaCapacity`** (when `ScaleUp == true`) or **`ScaleDownPerReplicaCapacity`** (when the target is still below current PRC).

**New helper `computeVerticalHint`** (add to [`saturation_v2/analyzer.go`](../../../internal/engines/analyzers/saturation_v2/analyzer.go))

```go
// computeVerticalHint produces a VerticalHint from per-replica saturation signals.
// Returns nil when:
//   - obs is nil (observation store has no entry yet — not bootstrapped), or
//   - obs.MaxComputeIntensity == 0 (no saturated tick observed yet), or
//   - readyCount == 0, or
//   - supply and demand are balanced (totalDemand == totalCapacity — nothing to adjust), or
//   - ComputeDemand/MemoryDemand fall outside the VPA allowed range after Step policy.
func computeVerticalHint(
    obs *observationstore.VariantObservation,
    replicas []ReplicaCapacity,
    totalDemand float64,
    totalCapacity float64,
    readyCount int,
    policy *vpaPolicy, // VPA min/max/step bounds; nil = no clamping
) *interfaces.VerticalHint {
    if obs == nil || obs.MaxComputeIntensity == 0 || readyCount == 0 {
        return nil
    }
    // VerticalScaleOption: only produce a hint when there is actual imbalance.
    scaleUp := totalDemand > totalCapacity
    scaleDown := totalCapacity > totalDemand
    if !scaleUp && !scaleDown {
        return nil
    }

    // Target tokens per replica for scale-up (eliminate required capacity)
    // or scale-down (floor at current capacity; optimizer shrinks toward demand).
    scaleUpPRC := totalDemand / float64(readyCount)
    scaleDownPRC := totalCapacity / float64(readyCount)
    targetPRC := scaleUpPRC
    if !scaleUp {
        targetPRC = scaleDownPRC
    }

    // ComputeDemand: I_live / I_max at the target token rate.
    computeDemand := 0.0
    if obs.MaxComputeIntensity > 0 {
        computeDemand = obs.ComputeIntensity / obs.MaxComputeIntensity
        if computeDemand > 1.0 {
            computeDemand = 1.0
        }
    }

    bpt := obs.BytePerToken
    if bpt == 0 {
        bpt = analyzerconstants.BytesPerKVToken // fallback constant
    }
    memoryDemand := obs.MemoryWeight + bpt*targetPRC

    // Apply Step range policy: round up ComputeDemand and MemoryDemand to the
    // nearest step boundary, then check they are within [minAllowed, maxAllowed].
    // Return nil when either value is outside the allowed range.
    roundedCompute, roundedMemory, ok := applyStepPolicy(computeDemand, memoryDemand, policy)
    if !ok {
        return nil
    }

    // Compute ScaleTargetPerReplica inversely from rounded resource values so that
    // the emitted PRC is consistent with what the optimizer will actually request.
    finalPRC := inversePRC(roundedCompute, roundedMemory, obs, bpt)

    // Only set ScaleDownPerReplicaCapacity when the final target is still below current PRC.
    if !scaleUp && finalPRC >= scaleDownPRC {
        return nil
    }

    hint := &interfaces.VerticalHint{
        ComputeDemand: roundedCompute,
        MemoryDemand:  float64(roundedMemory),
        DemandPerReplicaResource: &interfaces.ResourceRequirement{
            ComputeFraction: roundedCompute,
            MemoryBytes:     roundedMemory,
        },
    }
    if scaleUp {
        hint.ScaleUpPerReplicaCapacity = finalPRC
    } else {
        hint.ScaleDownPerReplicaCapacity = finalPRC
    }
    return hint
}
```

> **Note**: `applyStepPolicy` and `inversePRC` are small helpers added alongside `computeVerticalHint`. `applyStepPolicy` rounds compute/memory to VPA step boundaries and validates range. `inversePRC` derives the PRC that corresponds to the rounded resource values (inverse of the `ComputeDemand = I_live/I_max` and `MemoryDemand = MemoryWeight + bpt×PRC` formulae).

**Call site in `aggregateByVariant`** ([`saturation_v2/analyzer.go`](../../../internal/engines/analyzers/saturation_v2/analyzer.go))

Inside the `len(replicas) > 0` branch, after `totalCapacity` and `totalDemand` are computed, before the `result = append(result, ...)`:

```go
// Vertical hint: only when live replicas are present and observation store is ready.
obs := a.observationStore.Get(namespace, modelID, vs.VariantName)
verticalHint := computeVerticalHint(obs, replicas, totalDemand, totalCapacity, readyCount)

result = append(result, interfaces.VariantCapacity{
    // ... existing fields ...
    VerticalHint: verticalHint,
})
```

#### Expected Outcomes — Sub-Task 2b

- `computeVerticalHint` helper (+ `applyStepPolicy`, `inversePRC`) added to [`saturation_v2/analyzer.go`](../../../internal/engines/analyzers/saturation_v2/analyzer.go).
- Called in `aggregateByVariant` live-replicas branch only.
- Nil when `obs == nil`, `MaxComputeIntensity == 0`, `readyCount == 0`, balanced demand/supply, or resource values outside VPA allowed range after Step policy.
- `ComputeDemand` and `MemoryDemand` are rounded to Step policy boundaries before being stored in the hint.
- `DemandPerReplicaResource` populated with rounded compute fraction and memory bytes whenever the hint is non-nil.
- `ScaleTargetPerReplica` derived inversely from rounded resource values; set on `ScaleUpPerReplicaCapacity` (scale-up) or `ScaleDownPerReplicaCapacity` (scale-down, only when target < current PRC).
- Unit tests: hint set when demand > capacity with Step rounding; hint set when capacity > demand (scale-down room); nil when balanced; nil when obs nil; nil when not bootstrapped; nil when resource values out of VPA range; `ComputeDemand` zero when `MaxComputeIntensity == 0`.

#### Todo List — Sub-Task 2b

1. Add `computeVerticalHint`, `applyStepPolicy`, `inversePRC` helpers to [`saturation_v2/analyzer.go`](../../../internal/engines/analyzers/saturation_v2/analyzer.go).
2. Call it in `aggregateByVariant` inside the `len(replicas) > 0` branch and assign to `vc.VerticalHint`.
3. Add `ResourceRequirement` struct to [`internal/interfaces/analyzer.go`](../../../internal/interfaces/analyzer.go) alongside `VerticalHint` (Sub-Task 1 can include this).
4. Unit tests in [`saturation_v2/analyzer_test.go`](../../../internal/engines/analyzers/saturation_v2/analyzer_test.go).

**Status** — `[ ] pending (blocked on Sub-Tasks 1 and 2a)`

---

### Sub-Task 3 — QM: populate `VerticalHint` in `computeAllVariantCapacities`

#### Intent — Sub-Task 3

Extend the QM analyzer to emit `VerticalHint` by calling `queueAnalyzer.Size` at two additional SLO operating points. `ComputeDemand` and `MemoryDemand` are read from the shared `VariantObservationStore` (written by the sat V2 analyzer in Sub-Task 2a).

**Where to insert** ([`queueingmodel/analyzer.go`](../../../internal/engines/analyzers/queueingmodel/analyzer.go))

After the existing `variantCapacity := interfaces.VariantCapacity{...}` struct literal (currently line 385–396), before `variantCapacities = append(...)`:

```go
// Vertical hint: only when there is actual scale-up pressure or scale-down room.
rc := math.Max(0, totalArrivalRate-variantCapacity.TotalCapacity)
sc := math.Max(0, variantCapacity.TotalCapacity-totalArrivalRate)
if rc > 0 || sc > 0 {
    // Scale-up target: tighter SLO (divide TargetTTFT/ITL by ScaleUpSLOFactor)
    scaleUpTarget := &analyzer.TargetPerf{
        TargetTTFT: sloTarget.TargetTTFT / float32(config.ScaleUpSLOFactor),
        TargetITL:  sloTarget.TargetITL  / float32(config.ScaleUpSLOFactor),
    }
    var scaleUpRPS float64
    if _, m, _, err := queueAnalyzer.Size(scaleUpTarget); err == nil {
        scaleUpRPS = float64(m.Throughput)
    }

    // Scale-down floor: looser SLO (multiply TargetTTFT/ITL by ScaleDownSLOFactor)
    scaleDownTarget := &analyzer.TargetPerf{
        TargetTTFT: sloTarget.TargetTTFT * float32(config.ScaleDownSLOFactor),
        TargetITL:  sloTarget.TargetITL  * float32(config.ScaleDownSLOFactor),
    }
    var scaleDownRPS float64
    if _, m, _, err := queueAnalyzer.Size(scaleDownTarget); err == nil {
        scaleDownRPS = float64(m.Throughput)
    }

    if scaleUpRPS > 0 || scaleDownRPS > 0 {
        obs := a.observationStore.Get(namespace, modelID, variantName)
        I := wm.avgInputTokens
        O := wm.avgOutputTokens

        var memoryDemand, computeDemand float64
        if obs != nil {
            bpt := obs.BytePerToken
            if bpt == 0 {
                bpt = analyzerconstants.BytesPerKVToken
            }
            memoryDemand = obs.MemoryWeight + bpt*(scaleUpRPS*(I+O))
            if obs.MaxComputeIntensity > 0 {
                iTarget := scaleUpRPS * (I + analyzerconstants.ComputeIntensityAlpha*O)
                computeDemand = iTarget / obs.MaxComputeIntensity
            }
        }
        // obs == nil: demands remain zero; VerticalHint still set so optimizer
        // can act on the RPS targets; demand fields are for the recommender only.

        variantCapacity.VerticalHint = &interfaces.VerticalHint{
            ScaleUpPerReplicaCapacity:   scaleUpRPS,
            ScaleDownPerReplicaCapacity: scaleDownRPS,
            ComputeDemand:               computeDemand,
            MemoryDemand:                memoryDemand,
        }
    }
}
```

Note: `config` is not currently in scope of `computeAllVariantCapacities` — it must be passed as a parameter. The current signature is:

```go
func (a *QueueingModelAnalyzer) computeAllVariantCapacities(
    ctx context.Context,
    namespace string,
    modelID string,
    variantMetrics map[string][]interfaces.ReplicaMetrics,
    variantStates []interfaces.VariantReplicaState,
    sloTarget *SLOTarget,
) []interfaces.VariantCapacity
```

Add `config *QMConfig` as a parameter. The caller in `Analyze` already has a `QMConfig` from `input.Config.(*QMConfig)` (line ~100) and passes `sloTarget` from `a.getSLOTarget`; add `config` alongside.

**New `QMConfig` fields** ([`queueingmodel/config.go`](../../../internal/engines/analyzers/queueingmodel/config.go))

```go
// ScaleUpSLOFactor tightens the SLO for scale-up throughput target computation.
// TargetTTFT/ITL are divided by this factor (e.g. 0.75 → 25% tighter budget → lower maxRPS → more resource).
// Zero value uses DefaultScaleUpSLOFactor.
ScaleUpSLOFactor float64

// ScaleDownSLOFactor relaxes the SLO for scale-down floor computation.
// TargetTTFT/ITL are multiplied by this factor (e.g. 2.0 → 2× looser budget → higher maxRPS → less resource).
// Zero value uses DefaultScaleDownSLOFactor.
ScaleDownSLOFactor float64
```

**New constants** ([`queueingmodel/defaults.go`](../../../internal/engines/analyzers/queueingmodel/defaults.go))

```go
// DefaultScaleUpSLOFactor tightens the SLO for scale-up target computation.
const DefaultScaleUpSLOFactor = 0.75

// DefaultScaleDownSLOFactor relaxes the SLO for scale-down floor computation.
const DefaultScaleDownSLOFactor = 2.0
```

Apply defaults in `QMConfig` (same pattern as existing `SLOMultiplier` zero-means-default):

```go
func (c *QMConfig) scaleUpFactor() float64 {
    if c.ScaleUpSLOFactor == 0 {
        return DefaultScaleUpSLOFactor
    }
    return c.ScaleUpSLOFactor
}
func (c *QMConfig) scaleDownFactor() float64 {
    if c.ScaleDownSLOFactor == 0 {
        return DefaultScaleDownSLOFactor
    }
    return c.ScaleDownSLOFactor
}
```

**Propagation** ([`internal/interfaces/queueing_model_scaling.go`](../../../internal/interfaces/queueing_model_scaling.go) → [`buildQMConfig`](../../../internal/engines/saturation/engine_queueing_model.go))

Add to `QueueingModelScalingConfig`:

```go
ScaleUpSLOFactor   float64 `yaml:"scaleUpSLOFactor"`
ScaleDownSLOFactor float64 `yaml:"scaleDownSLOFactor"`
```

Propagate in `buildQMConfig`:

```go
ScaleUpSLOFactor:   qmConfig.ScaleUpSLOFactor,
ScaleDownSLOFactor: qmConfig.ScaleDownSLOFactor,
```

#### Expected Outcomes — Sub-Task 3

- `QMConfig` gains `ScaleUpSLOFactor` and `ScaleDownSLOFactor` with zero-means-default semantics.
- `DefaultScaleUpSLOFactor = 0.75` and `DefaultScaleDownSLOFactor = 2.0` added to `defaults.go`.
- `computeAllVariantCapacities` signature gains `config *QMConfig`.
- `VerticalHint` populated when `rc > 0 || sc > 0` and at least one `Size` call succeeds.
- `ComputeDemand` and `MemoryDemand` are zero when `obs == nil` (sat V2 hasn't written yet).
- Nil `VerticalHint` when supply equals demand exactly (`rc == 0 && sc == 0`).
- Unit tests: hint set when RC > 0 with obs; hint set with zero demands when obs nil; nil when RC == SC == 0; nil when params missing.

#### Todo List — Sub-Task 3

1. Add `ScaleUpSLOFactor`, `ScaleDownSLOFactor` to `QMConfig` in [`config.go`](../../../internal/engines/analyzers/queueingmodel/config.go).
2. Add `scaleUpFactor()` / `scaleDownFactor()` helper methods on `QMConfig`.
3. Add `DefaultScaleUpSLOFactor` and `DefaultScaleDownSLOFactor` to [`defaults.go`](../../../internal/engines/analyzers/queueingmodel/defaults.go).
4. Add both fields to [`QueueingModelScalingConfig`](../../../internal/interfaces/queueing_model_scaling.go) with YAML tags.
5. Propagate through [`buildQMConfig`](../../../internal/engines/saturation/engine_queueing_model.go).
6. Add `config *QMConfig` parameter to `computeAllVariantCapacities`; update its call site in `Analyze`.
7. Add the two extra `Size` calls, observation store lookup, demand derivation, and `VerticalHint` population.
8. Unit tests in [`queueingmodel/analyzer_test.go`](../../../internal/engines/analyzers/queueingmodel/analyzer_test.go) (file does not exist yet — create it).

**Status** — `[ ] pending (blocked on Sub-Task 1)`

---

### Sub-Task 4 — Config: `VerticalScalingEnabled` per variant

#### Intent — Sub-Task 4

Gate vertical scaling per variant via `ResourceClaimPolicy != nil` in the WVA spec. The optimizer checks this flag — no vertical hint is acted upon unless the variant is opted in.

**Completed (partially)** — two commits landed the foundational infrastructure that populates `ResourceClaimPolicy` on synthetic VAs. The `VerticalScalingEnabled` flag itself and `BuildVariantStates` wiring remain pending.

#### Completed in `875d631` — Add VPA reconciler

- New [`internal/controller/vpa_reconciler.go`](../../../internal/controller/vpa_reconciler.go): `VPAReconciler` watches `VerticalPodAutoscaler` objects bearing `llm-d.ai/managed: "true"` and the `prometheus` recommender; calls `Datastore.NamespaceTrack` / `NamespaceUntrack`.
- [`internal/utils/crd/crd.go`](../../../internal/utils/crd/crd.go): `CheckVPACRD` — detects whether the `VerticalPodAutoscaler` CRD is installed at startup.
- [`cmd/main.go`](../../../cmd/main.go): registers `vpav1` scheme unconditionally; conditionally registers `VPAReconciler` when VPA CRD is present.

#### Completed in `e6c7b8a` — Add VPA to VA with ResourceClaimTemplate in ScaleTarget

- [`internal/utils/variant_fromannotations.go`](../../../internal/utils/variant_fromannotations.go): `VariantAutoscalingFromVPA` — builds an in-memory `VariantAutoscaling` from a managed VPA, carrying `spec.resourcePolicy.resourceClaimPolicies[0]` as `ResourceClaimPolicy`.
- [`internal/utils/variant.go`](../../../internal/utils/variant.go): `annotationSourcedVariants` now lists VPAs; merges `ResourceClaimPolicy` onto an existing HPA/ScaledObject entry.
- [`internal/utils/scaletarget/accessor.go`](../../../internal/utils/scaletarget/accessor.go): `ScaleTargetAccessor` gains `GetLeaderResourceClaimTemplate()` and `GetWorkerResourceClaimTemplate()`.
- [`internal/utils/scaletarget/fetch.go`](../../../internal/utils/scaletarget/fetch.go): `FetchScaleTarget` fetches the first `ResourceClaimTemplate` referenced by each pod template.

#### Remaining Todo List

1. Add `VerticalScalingEnabled bool` to `VariantReplicaState` in [`internal/interfaces/saturation_analyzer.go`](../../../internal/interfaces/saturation_analyzer.go).
2. Update [`BuildVariantStates`](../../../internal/engines/saturation/engine.go) to set `VerticalScalingEnabled = va.Spec.ResourceClaimPolicy != nil`.
3. Unit tests for the flag.

**Status** — `[~] in progress — infrastructure complete; VerticalScalingEnabled flag pending`

---

### Sub-Task 5 — Optimizer: vertical-first decision logic

#### Intent — Sub-Task 5

Extend the optimizer with a vertical-first decision pass that consumes `VerticalHint` regardless of which analyzer produced it. The unit difference (tokens vs. req/s) is irrelevant — the hint fields are directly comparable to `PerReplicaCapacity` which is always in the same unit as the analyzer that set it.

#### Decision rules

```text
ScaleUpWork:
  for each variant where VerticalScalingEnabled && VerticalHint != nil:
    if VerticalHint.ScaleUpPerReplicaCapacity > vc.PerReplicaCapacity:
      // growing per-replica capacity reduces required horizontal replicas
      → verticalTargets[name] = {ScaleUpPerReplicaCapacity, ComputeDemand, MemoryDemand, VerticalScaleUp}
      → patch vc.PerReplicaCapacity = ScaleUpPerReplicaCapacity (working copy only)
      // higher prc → allocateForModelPaired requests fewer horizontal replicas

  → run allocateForModelPaired as today (spills to horizontal if headroom exhausted)

ScaleDownIterate:
  → run scaleDownRoleIterated as today (horizontal first — cheaper, no pod restart)
  for each variant where VerticalScalingEnabled && VerticalHint != nil:
    if VerticalHint.ScaleDownPerReplicaCapacity < vc.PerReplicaCapacity AND
       targets[name] already == *state.MinReplicas (horizontal exhausted):
      → verticalTargets[name] = {ScaleDownPerReplicaCapacity, ComputeDemand, MemoryDemand, VerticalScaleDown}

nil VerticalHint or !VerticalScalingEnabled:
  → skip vertical step entirely; horizontal-only path unchanged
```

**New `VariantDecision` fields** ([`internal/interfaces/saturation_analyzer.go`](../../../internal/interfaces/saturation_analyzer.go))

Add to `VariantDecision` struct (after `WasLimited`/`LimitedBy`):

```go
// VerticalAction is the vertical scaling direction for this variant.
// VerticalNoChange when vertical scaling is not applicable or not enabled.
VerticalAction VerticalScalingAction

// TargetPerReplicaCapacity is the desired per-replica capacity after vertical scaling.
// Sat V2: tokens. QM: req/s. Zero when VerticalAction == VerticalNoChange.
TargetPerReplicaCapacity float64

// CurrentPerReplicaCapacity is PerReplicaCapacity at analysis time (before any
// vertical patching). Used by the actuator: scaleFactor = Target / Current.
// Zero when VerticalAction == VerticalNoChange.
CurrentPerReplicaCapacity float64

// ComputeDemand is the compute resource requirement for the vertical action
// as a fraction (0.0–1.0) of the GPU's observed compute wall.
// Zero when VerticalAction == VerticalNoChange or observation store not ready.
ComputeDemand float64

// MemoryDemand is the memory resource requirement in bytes for the vertical action.
// Zero when VerticalAction == VerticalNoChange or observation store not ready.
MemoryDemand float64
```

**New type** (add before `VariantDecision` in the same file):

```go
// VerticalScalingAction describes the vertical scaling direction for a variant.
type VerticalScalingAction string

const (
    VerticalNoChange  VerticalScalingAction = "no-change"
    VerticalScaleUp   VerticalScalingAction = "scale-up"
    VerticalScaleDown VerticalScalingAction = "scale-down"
)
```

#### New file: [`internal/engines/pipeline/vertical_helpers.go`](../../../internal/engines/pipeline/vertical_helpers.go)

```go
package pipeline

import "github.com/llm-d/llm-d-workload-variant-autoscaler/internal/interfaces"

// verticalTarget records the vertical scaling intent for one variant.
type verticalTarget struct {
    TargetPRC     float64                        // ScaleUp or ScaleDown per-replica capacity
    ComputeDemand float64                        // from VerticalHint
    MemoryDemand  float64                        // from VerticalHint
    Action        interfaces.VerticalScalingAction
}

// applyVerticalScaleUp iterates variants, records ScaleUpPerReplicaCapacity in
// verticalTargets and patches vc.PerReplicaCapacity in the variants slice for
// the subsequent allocateForModelPaired call.
// Only acts when VerticalScalingEnabled and ScaleUpPerReplicaCapacity > current PRC.
func applyVerticalScaleUp(
    variants []interfaces.VariantCapacity,
    stateMap map[string]interfaces.VariantReplicaState,
    verticalTargets map[string]verticalTarget,
) {
    for i, vc := range variants {
        state, ok := stateMap[vc.VariantName]
        if !ok || !state.VerticalScalingEnabled {
            continue
        }
        hint := vc.VerticalHint
        if hint == nil || hint.ScaleUpPerReplicaCapacity <= vc.PerReplicaCapacity {
            continue
        }
        verticalTargets[vc.VariantName] = verticalTarget{
            TargetPRC:     hint.ScaleUpPerReplicaCapacity,
            ComputeDemand: hint.ComputeDemand,
            MemoryDemand:  hint.MemoryDemand,
            Action:        interfaces.VerticalScaleUp,
        }
        // Patch working copy so allocateForModelPaired sees the higher PRC.
        variants[i].PerReplicaCapacity = hint.ScaleUpPerReplicaCapacity
    }
}

// applyVerticalScaleDown records ScaleDownPerReplicaCapacity in verticalTargets
// only when the variant is already at minReplicas (horizontal scale-down exhausted).
func applyVerticalScaleDown(
    variants []interfaces.VariantCapacity,
    stateMap map[string]interfaces.VariantReplicaState,
    targets map[string]int,
    verticalTargets map[string]verticalTarget,
) {
    for _, vc := range variants {
        state, ok := stateMap[vc.VariantName]
        if !ok || !state.VerticalScalingEnabled {
            continue
        }
        hint := vc.VerticalHint
        if hint == nil || hint.ScaleDownPerReplicaCapacity >= vc.PerReplicaCapacity {
            continue
        }
        // Only shrink resources when horizontal is already at its floor.
        currentTarget := targets[vc.VariantName]
        atMin := state.MinReplicas != nil && currentTarget <= *state.MinReplicas
        if !atMin {
            continue
        }
        verticalTargets[vc.VariantName] = verticalTarget{
            TargetPRC:     hint.ScaleDownPerReplicaCapacity,
            ComputeDemand: hint.ComputeDemand,
            MemoryDemand:  hint.MemoryDemand,
            Action:        interfaces.VerticalScaleDown,
        }
    }
}
```

**Integration: `CostAwareOptimizer.Optimize`** ([`cost_aware_optimizer.go`](../../../internal/engines/pipeline/cost_aware_optimizer.go))

Inside the per-model loop, after `initRoleState`:

```go
verticalTargets := make(map[string]verticalTarget)

if anyRoleNeedsScaleUp(ps, roles) {
    applyVerticalScaleUp(satEntry.VariantCapacities, stateMap, verticalTargets)
    allocateForModelPaired(ctx, s, satEntry.VariantCapacities, stateMap, nil, targets,
        costGreedyRolePick, ps, roles)
} else {
    scaleDownRoleIterated(ctx, s, satEntry.VariantCapacities, targets, stateMap)
    applyVerticalScaleDown(satEntry.VariantCapacities, stateMap, targets, verticalTargets)
}

decisions := buildDecisionsWithOptimizer(req, stateMap, vcMap, targets, verticalTargets, "cost-aware")
```

Same pattern in `GreedyByScoreOptimizer`.

#### `buildDecisionsWithOptimizer` update

Add `verticalTargets map[string]verticalTarget` parameter. Before building each `VariantDecision`, save `CurrentPerReplicaCapacity` from `vcMap[name].PerReplicaCapacity` (the **original** un-patched value, from `vcMap` which is built before any vertical patching). Then populate vertical fields:

```go
currentPRC := vcMap[name].PerReplicaCapacity  // un-patched original

decision := interfaces.VariantDecision{
    // ... existing fields ...
}

if vt, ok := verticalTargets[name]; ok {
    decision.VerticalAction            = vt.Action
    decision.TargetPerReplicaCapacity  = vt.TargetPRC
    decision.CurrentPerReplicaCapacity = currentPRC
    decision.ComputeDemand             = vt.ComputeDemand
    decision.MemoryDemand              = vt.MemoryDemand
}
```

Key: `vcMap` is built by `buildCapacityMap(satEntry.VariantCapacities)` **before** `applyVerticalScaleUp` patches the slice, so `vcMap[name].PerReplicaCapacity` always holds the original value.

#### Expected Outcomes — Sub-Task 5

- `VerticalScalingAction` type + constants added to [`saturation_analyzer.go`](../../../internal/interfaces/saturation_analyzer.go).
- `VariantDecision` gains five new fields.
- [`vertical_helpers.go`](../../../internal/engines/pipeline/vertical_helpers.go) with `verticalTarget` type, `applyVerticalScaleUp`, `applyVerticalScaleDown`.
- Both optimizers initialise `verticalTargets` and call helpers; pass to `buildDecisionsWithOptimizer`.
- `buildDecisionsWithOptimizer` signature gains `verticalTargets`; populates vertical fields from it.
- `vcMap` is always built before any patching so `CurrentPerReplicaCapacity` is always the original value.
- Unit tests in [`vertical_helpers_test.go`](../../../internal/engines/pipeline/vertical_helpers_test.go): scale-up patches PRC and records target; demand already satisfied (no vertical); scale-down blocked by minReplicas; scale-down not blocked (skipped — horizontal runs first); nil hint no-op; `!VerticalScalingEnabled` no-op.

#### Todo List — Sub-Task 5

1. Add `VerticalScalingAction` type + constants to [`interfaces/saturation_analyzer.go`](../../../internal/interfaces/saturation_analyzer.go).
2. Add `VerticalAction`, `TargetPerReplicaCapacity`, `CurrentPerReplicaCapacity`, `ComputeDemand`, `MemoryDemand` to `VariantDecision`.
3. Create [`internal/engines/pipeline/vertical_helpers.go`](../../../internal/engines/pipeline/vertical_helpers.go) with `verticalTarget` type, `applyVerticalScaleUp`, `applyVerticalScaleDown`.
4. Update `CostAwareOptimizer.Optimize` in [`cost_aware_optimizer.go`](../../../internal/engines/pipeline/cost_aware_optimizer.go).
5. Update `GreedyByScoreOptimizer` similarly.
6. Update `buildDecisionsWithOptimizer` signature and body to save `CurrentPerReplicaCapacity` and populate vertical fields.
7. Add [`internal/engines/pipeline/vertical_helpers_test.go`](../../../internal/engines/pipeline/vertical_helpers_test.go).
8. Add vertical scenarios to `cost_aware_optimizer_test.go`.

**Status** — `[ ] pending (blocked on Sub-Tasks 1, 2b, 3, 4)`

---

### Sub-Task 5b — DRA-aware vertical optimizer

#### Why the wrapper pattern is wrong

`applyVerticalScaleUp` (Sub-Task 5) patches `vc.PerReplicaCapacity` in the working slice **before** `allocateForModelPaired` runs. That patched PRC directly determines how many horizontal replicas are requested — `roleBottleneckReplicas` computes `ceil(demand / PRC)`. A post-hoc veto after `buildDecisionsWithOptimizer` leaves horizontal counts that were computed assuming vertical will succeed, producing an incoherent decision.

#### Correct design: DRA capacity is a resource budget inside the allocation loop

This mirrors exactly how `GreedyByScoreOptimizer` handles GPU count: `available map[string]int` is decremented inside `allocateForModelPaired` → `fairShareRolePick` on every joint commit. DRA device capacity headroom must be tracked and consumed in the **same single pass**, so if vertical headroom is insufficient the optimizer never patches PRC in the first place and falls back to horizontal.

**New `MultiDimensionalOptimizer`** — new file [`internal/engines/pipeline/multi_dimensional_optimizer.go`](../../../internal/engines/pipeline/multi_dimensional_optimizer.go)

`MultiDimensionalOptimizer` implements `ScalingOptimizer` with the same outer structure as `GreedyByScoreOptimizer`, but adds DRA capacity tracking via `ResourceCapacityInventory` alongside the GPU budget.

```text
MultiDimensionalOptimizer.Optimize(requests, constraints)
  │
  ├─ ResourceCapacityInventory.Snapshot()
  │    Read from ResourceSlices informer cache → draAvailable map[capacityName]int64  (total headroom per dim)
  │    Read from ResourceClaims informer cache → claimedByVariant map[variantName]map[capacityName]int64
  │    If cache not synced → draAvailable = nil (vertical disabled for this tick)
  │
  ├─ for each model (same outer loop as GreedyByScoreOptimizer):
  │    stateMap, vcMap, targets = existing helpers
  │    roles, ps = initRoleState(s)
  │
  │    if anyRoleNeedsScaleUp:
  │      ① applyVerticalScaleUp(variants, stateMap, draAvailable, claimedByVariant, verticalTargets)
  │           — only patches vc.PerReplicaCapacity AND records verticalTarget
  │             when draAvailable[capacityName] ≥ (targetPRC - claimed) × replicaCount
  │           — decrements draAvailable by the capacity delta (same as available[accType] -= GPUs)
  │           — if headroom insufficient: skip vertical; PRC left unchanged; horizontal spill
  │      ② allocateForModelPaired(... draAwareRolePick ...)  ← patched or original PRC
  │
  │    else (scale-down):
  │      scaleDownRoleIterated(...)  ← horizontal first
  │      applyVerticalScaleDown(variants, stateMap, targets, verticalTargets)
  │           — only records verticalTarget when targets[name] == *minReplicas
  │           — no DRA check needed for scale-down (shrinking always valid if claim exists)
  │
  └─ buildDecisionsWithOptimizer(..., verticalTargets)
```

#### `applyVerticalScaleUp` revised signature

```go
// applyVerticalScaleUp patches vc.PerReplicaCapacity and records verticalTargets
// only when DRA device headroom is sufficient. draAvailable and claimedByVariant
// are updated in place (same pattern as available map[string]int for GPUs).
//
// When draAvailable is nil (DRA CRD absent or Refresh failed), no vertical
// action is taken and vc.PerReplicaCapacity is left unchanged — the optimizer
// falls through to the normal horizontal path.
func applyVerticalScaleUp(
    variants       []interfaces.VariantCapacity,
    stateMap       map[string]interfaces.VariantReplicaState,
    draAvailable   map[string]int64,          // capacityName → remaining headroom; nil = DRA unavailable
    claimedByVariant map[string]map[string]int64, // variantName → capacityName → currently claimed
    verticalTargets map[string]verticalTarget,
)
```

#### Inside `applyVerticalScaleUp` — headroom check

```go
for i, vc := range variants {
    state := stateMap[vc.VariantName]
    if !state.VerticalScalingEnabled || vc.VerticalHint == nil {
        continue
    }
    hint := vc.VerticalHint
    if hint.ScaleUpPerReplicaCapacity <= vc.PerReplicaCapacity {
        continue
    }
    if draAvailable == nil {
        // DRA unavailable this tick — no vertical, fall through to horizontal
        continue
    }

    // Capacity needed per replica = target − current allocation
    currentClaimed := claimedByVariant[vc.VariantName][capacityName] // e.g. "gpu.nvidia.com/vram"
    deltaPerReplica := int64(hint.ScaleUpPerReplicaCapacity) - currentClaimed
    readyCount := max(state.CurrentReplicas-state.PendingReplicas, 0)
    totalDelta := deltaPerReplica * int64(readyCount)

    if draAvailable[capacityName] < totalDelta {
        // Not enough headroom — skip vertical, leave PRC unchanged
        continue
    }

    // Sufficient headroom: patch PRC, record target, consume budget
    draAvailable[capacityName] -= totalDelta
    verticalTargets[vc.VariantName] = verticalTarget{...}
    variants[i].PerReplicaCapacity = hint.ScaleUpPerReplicaCapacity
}
```

**Resource capacity inventory** — new file [`internal/discovery/dra_inventory.go`](../../../internal/discovery/dra_inventory.go)

`ResourceCapacityInventory` is informed by **ResourceSlices and ResourceClaims informers (watchers)** — not per-tick API calls. It maintains an in-memory view of the available resource capacity of each device in each pool, updated by the controller-runtime cache. `Snapshot()` reads directly from the cache without any network round-trip.

```go
// ResourceCapacityInventory keeps the available resource capacity of each
// device in each pool, informed by ResourceSlices and ResourceClaims watchers.
type ResourceCapacityInventory struct { ... }

func NewResourceCapacityInventory(cache cache.Cache) *ResourceCapacityInventory

// Snapshot returns a point-in-time view of available and claimed capacity.
// Returns (nil, nil) when the informer cache has not yet synced.
func (r *ResourceCapacityInventory) Snapshot() (
    draAvailable map[string]int64,             // capacityName → total unallocated headroom
    claimedByVariant map[string]map[string]int64, // variantName → capacityName → claimed
)
```

**Engine wiring** ([`internal/engines/saturation/engine.go`](../../../internal/engines/saturation/engine.go))

Replace the optimizer selection in `optimize()`:

```go
if enableLimiter {
    e.optimizer = pipeline.NewMultiDimensionalOptimizer(
        pipeline.NewGreedyByScoreOptimizer(), e.rcInventory)
} else {
    e.optimizer = pipeline.NewMultiDimensionalOptimizer(
        pipeline.NewCostAwareOptimizer(), e.rcInventory)
}
```

- `e.rcInventory *discovery.ResourceCapacityInventory` added to `Engine` struct; set from the controller-runtime cache at construction time.
- `MultiDimensionalOptimizer` handles nil `draAvailable` (from an unsynced cache snapshot) internally — vertical disabled that tick, horizontal proceeds normally.
- Created in `NewEngine`; when the DRA CRD is absent, `Snapshot()` returns `(nil, nil)`, keeping `draAvailable = nil` for the tick.

#### Sub-Task 5 revised: `vertical_helpers.go` changes

Sub-Task 5's `applyVerticalScaleUp` and `applyVerticalScaleDown` are now **only called by `MultiDimensionalOptimizer`**, not by `CostAwareOptimizer` or `GreedyByScoreOptimizer` directly. Both existing optimizers remain completely unchanged. The helpers gain the `draAvailable`/`claimedByVariant` parameters as described above.

#### Expected Outcomes — Sub-Task 5b

- New [`internal/discovery/dra_inventory.go`](../../../internal/discovery/dra_inventory.go) with `ResourceCapacityInventory` backed by ResourceSlices and ResourceClaims informers; `Snapshot()` returns available/claimed maps from cache.
- New [`internal/engines/pipeline/multi_dimensional_optimizer.go`](../../../internal/engines/pipeline/multi_dimensional_optimizer.go) with `MultiDimensionalOptimizer` — same outer loop as `GreedyByScoreOptimizer`, DRA budget consumed inside `applyVerticalScaleUp`.
- `CostAwareOptimizer` and `GreedyByScoreOptimizer` are **completely unchanged**.
- `applyVerticalScaleUp` (Sub-Task 5) gains `draAvailable` + `claimedByVariant` parameters; skips vertical when `draAvailable == nil` or headroom insufficient; decrements budget on success.
- `applyVerticalScaleDown` unchanged (no DRA check needed for shrink).
- When cache not synced: `draAvailable = nil`; `applyVerticalScaleUp` skips all vertical; `allocateForModelPaired` sees original unpatched PRCs; horizontal-only result is coherent.
- Unit tests: vertical taken when headroom sufficient, PRC patched, budget decremented; vertical skipped when headroom insufficient, PRC unpatched, horizontal runs normally; vertical skipped when `draAvailable == nil`; scale-down at minReplicas triggers vertical regardless of DRA.

#### Todo List — Sub-Task 5b

1. Add `CheckDRACRD(ctx, client)` to [`internal/utils/crd/crd.go`](../../../internal/utils/crd/crd.go) (mirrors existing `CheckVPACRD`).
2. Create [`internal/discovery/dra_inventory.go`](../../../internal/discovery/dra_inventory.go) with `ResourceCapacityInventory` backed by ResourceSlices and ResourceClaims informers; `Snapshot()` method reading from cache.
3. Revise `applyVerticalScaleUp` in [`vertical_helpers.go`](../../../internal/engines/pipeline/vertical_helpers.go) to accept `draAvailable map[string]int64` and `claimedByVariant map[string]map[string]int64`; add headroom check and budget decrement.
4. Create [`internal/engines/pipeline/multi_dimensional_optimizer.go`](../../../internal/engines/pipeline/multi_dimensional_optimizer.go) with `MultiDimensionalOptimizer` — same outer structure as `GreedyByScoreOptimizer`, calls `applyVerticalScaleUp` with DRA budgets.
5. Add `rcInventory *discovery.ResourceCapacityInventory` to `Engine` struct; always pass to `MultiDimensionalOptimizer`; remove vertical helper calls from `CostAwareOptimizer` and `GreedyByScoreOptimizer`.
6. Unit tests in [`multi_dimensional_optimizer_test.go`](../../../internal/engines/pipeline/multi_dimensional_optimizer_test.go).
7. Unit tests in [`dra_inventory_test.go`](../../../internal/discovery/dra_inventory_test.go).

**Status** — `[ ] pending (blocked on Sub-Task 5)`

---

### Sub-Task 6 — Observability: metrics

#### Intent — Sub-Task 6

Expose vertical scaling signals through Prometheus, identically for both analyzer paths.

| Metric | Type | Labels | Description |
| --- | --- | --- | --- |
| `wva_variant_target_capacity_per_gpu` | Gauge | `variant_name`, `namespace`, `model_id`, `accelerator_type`, `capacity` | Last recommended value for each DRA capacity dimension. `capacity` label = `ControlledCapacities` name (e.g. `gpu.nvidia.com/vram`). |
| `wva_vertical_scaling_total` | Counter | `variant_name`, `namespace`, `model_id`, `direction` (`up`/`down`) | Cumulative count of vertical scaling decisions produced. |

#### Todo List — Sub-Task 6

1. Add `WVAVariantTargetCapacityPerGPU` and `WVAVerticalScalingTotal` constants to [`internal/constants/metrics.go`](../../../internal/constants/metrics.go).
2. Declare `variantTargetCapacityPerGPU *prometheus.GaugeVec` and `verticalScalingTotal *prometheus.CounterVec` in [`internal/metrics/metrics.go`](../../../internal/metrics/metrics.go); register in `InitMetrics`.
3. Add `RecordVerticalScalingMetrics(variantName, namespace, modelID, acceleratorType string, capacityValues map[string]int64, direction string)` to `MetricsEmitter` — iterates `capacityValues` map to set the gauge, then increments the counter.
4. Call `RecordVerticalScalingMetrics` in `applySaturationDecisions` when `VerticalAction != VerticalNoChange`.
5. Unit tests.

**Status** — `[ ] pending (blocked on Sub-Task 5)`

---

### Sub-Task 7 — Documentation

#### Todo List — Sub-Task 7

1. Write [`docs/developer-guide/vertical-scaling.md`](../../developer-guide/vertical-scaling.md) covering:
   - The two analyzer paths and what `VerticalHint` means in each.
   - Compute intensity formula and empirical compute wall (sat V2).
   - QM SLO-factor parameters (`ScaleUpSLOFactor`, `ScaleDownSLOFactor`) and their effect.
   - Memory model (`MemoryWeight`, `BytePerToken`, `MemoryDemand`) (sat V2).
   - Vertical-first decision rule and scale-down iterate order.
   - How `ResourceClaimPolicy` in the WVA spec enables vertical scaling.
   - New Prometheus metrics and the actuation boundary.
2. Update `docs/developer-guide/configuration.md` with `scaleUpSLOFactor` and `scaleDownSLOFactor`.

**Status** — `[ ] pending`

---

## External Systems

### Actuation — custom VPA recommender (out of scope for this project)

WVA's responsibility ends at producing the Prometheus metrics (`wva_variant_target_capacity_per_gpu`, `wva_desired_replicas`). The actual `ResourceClaimTemplate` patching and pod rolling restart are handled by a **separate custom recommender** project that:

1. Scrapes `wva_variant_target_capacity_per_gpu` from WVA's Prometheus endpoint.
2. Translates the per-capacity target values into a `vpa.Status.Recommendation.ContainerRecommendations` status update, acting as the `"prometheus"` external recommender.
3. The VPA applier reads the recommendation and patches the `ResourceClaimTemplate` + triggers a rolling restart.

No WVA code changes are required for this path.

---

## Completed Work

| Commit | Message | Sub-Task |
| --- | --- | --- |
| [`875d631`](https://github.com/llm-d/llm-d-workload-variant-autoscaler/commit/875d631dc3fb4c04e8e29e0e570c9481314edba5) | Add VPA reconciler | Sub-Task 4 (infrastructure) |
| [`e6c7b8a`](https://github.com/llm-d/llm-d-workload-variant-autoscaler/commit/e6c7b8a87593a0a9bd93658910ae7f57f9c8f437) | Add VPA to VA with ResourceClaimTemplate in ScaleTarget | Sub-Task 4 (infrastructure) |

---

## Implementation Order

```text
Sub-Task 1  (VerticalHint type)
  ├─ Sub-Task 2a (observationstore package + ReplicaMetrics fields + both write paths)
  │    └─ Sub-Task 2b (sat V2: computeVerticalHint in aggregateByVariant)
  ├─ Sub-Task 3  (QM: two Size calls + VerticalHint)   ← parallel with 2a/2b
  └─ Sub-Task 4  (VerticalScalingEnabled flag)          ← in progress; infrastructure done
       │
       └─ Sub-Task 5  (Optimizer: vertical_helpers.go + buildDecisionsWithOptimizer)
            │           existing CostAware/GreedyByScore unchanged
            │
            └─ Sub-Task 5b (MultiDimensionalOptimizer wrapper)
                 │  new: ResourceCapacityInventory (ResourceSlices + Claims informers)
                 │  wraps inner optimizer; gates vertical decisions against DRA headroom
                 │  gracefully degrades when DRA CRD absent
                 │
                 └─ Sub-Task 6  (Observability / metrics)  ─┐
                      Sub-Task 7  (Docs)                    ─┘ parallel

[External] Custom VPA recommender reads wva_variant_target_capacity_per_gpu
           → writes vpa.Status.Recommendation → VPA applier patches ResourceClaimTemplate
```

Sub-Tasks 2a/2b, 3, and 4 are independent and can proceed in parallel after Sub-Task 1. Sub-Task 5b is parallel-safe with Sub-Task 5 for the `ResourceCapacityInventory` work (no shared files), but `MultiDimensionalOptimizer` and engine wiring require Sub-Task 5 to be merged first. Sub-Tasks 6 and 7 can proceed in parallel once Sub-Task 5b is merged.
