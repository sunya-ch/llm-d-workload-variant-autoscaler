# Logic of Vertical Scaling

## Compute requirement

### Compute Intensity

Compute intensity is the ratio of computational operations (FLOPs) to data movement (Bytes), determining whether an execution phase is bottlenecked by the GPU's processing speed or its memory bandwidth.

$$I_{live} = R_p + (\alpha \times R_g)$$

Where

$\alpha \approx 0.1$ to $0.15$, capturing the lower parallel processing density of token-by-token generation. The scaling factor $\alpha$ (alpha) acts as an equivalence coefficient.Because generation is memory-bound and sequential, generating one token uses significantly less parallel processing density and raw computational power than processing one prompt token.If $\alpha \approx 0.1$, the formula asserts that generating 10 tokens takes roughly the same computational/intensity toll on the system as processing 1 prompt token.

$$
\text{Prompt Token Rate } (R_p) = \frac{\Delta \text{vllm:promptTokensTotal}}{\Delta t}
$$

$$
\text{Generation Token Rate } (R_g) = \frac{\Delta \text{vllm:generationTokensTotal}}{\Delta t}
$$

### Compute wall and %Thread Requirement

- Empirical Compute Wall ($I_{max}$) is $I_{live}$ when vllm:num_requests_waiting > 0 (or a spike in vllm:request_queue_time_seconds).

- To have the $I_{max}$ set from fully empherical, must start from max resource.

- %Threads requirement is

  $$T_{req} = Max(\frac{I_{live}}{I_{max}} \times 100\%, minAllowed_{compute})$$

## Memory requirement

### Minimum requirement

- In VPA, there is allowed minimum compute, memory set
- Minimum memory needed for model weight can be calculated by

    $$
    \text{Model Weight} = TotalKvCapacityTokens * \frac{(1-U)}{U}
    $$

- Actual Minimum memory requirement is

  $$Max(\text{Model Weight}, minAllowed_{memory})$$

### Token-aware requirement

$$\Delta \text{Tokens} = \Delta \text{vllm:promptTokensTotal} + \Delta \text{vllm:generationTokensTotal}$$

$$\Delta \text{Cache Bytes Used} = \Delta \text{GPUCacheUsagePerc} \times \text{availableKvCacheMemoryBytes}$$

$$\text{Bytes Per Token } (\text{Bpt}) = \frac{\Delta \text{Cache Bytes Used}}{\Delta \text{Tokens}}$$

Then,

$$\text{Required Memory} = M_{\text{weights}} + (\text{Bpt} \times N_{\text{activeTokens}}) + M_{\text{overhead}}$$

$N_{\text{activeTokens}}$ is capped by effective capacity when saturate.

## KnowledgeStore

- ComputeIntensity
- MaxComputeIntensity
- MemoryWeight
- BytePerToken

## VariantCapacity (Demand/Supply Analysis)

If MaxComputeIntensity is set, we can calculate demandPerReplicaResource in case that required capacity or spare capacity exists.

- **ScaleUpPerReplicaCapacity:** expected capacity if execute vertical scaling up, targeting no required capacity from horizontal scaling
- **ScaleDownPerReplicaCapacity:** expected capacity if execute vertical scaling down, targeting no resource waste (no spare capacity)
- **DemandPerReplicaResource:** estimated resource need to for achiving ScaleUpPerReplicaCapacity or ScaleDownPerReplicaCapacity

## Multi-dimensional optimization

### Resource Constraints

- GPU available with consumable capacity (ResourceSlices)
- Recreate method need rollout, minAvailable into consideration

### ScaleUpWork

- If ScaleUpPerReplicaCapacity available, and applicable, do vertical scale with DemandPerReplicaResource.
- If not, horizontal scale with current resource (as-is)

### ScaleDownItereate

- Try horizontal scale down first (cheaper and quicker as no resizing).
- If cannot do horizontal scale down, check if DemandPerReplicaResource available and applicable.

## Analyze VerticalHint

- For each Analyze call, Analyzer (either SaturationAnalyzer or QueueingModelAnalyzer) gets and updates VariantObservation (obs) from ReplicaMetrics.
  - MemoryWeight is computed once from KVCache capacity and GPU Memory Utilization at maximum resource
  - MaxComputeIntensity is set when Queue Length > 0 at maximum resource.
  - ComputeIntensity is recomputed from input and output token metrics for every tick
  - BytePerToken is recomputed when delta values are available
- VerticalHint can be calculated only when MaxComputeIntensity has been set.
  - If Demand > Supply or Demand < Supply - scale down headroom
    - Target Tokens = demand (Tokens or RPS * Tokens per request)
    - VerticalScaleOption = true
    - ScaleUp = Demand > Supply
  - If VerticalScaleOption = true:
    - ComputeDemand = ComputeIntensity / MaxComputeIntensity when ComputeIntensity = Target PromptTokens + alpha x GeneratedTokens
    - MemoryDemand uses obs.MemoryWeight and obs.BytePerToken
      MemoryWeight + (BytePerToken * Target Tokens) + overhead
    - If ComputeDemand and MemoryDemand after applied “Step” range policy is within allowed range
  - ScaleTargetPerReplica is calculated inversely from Rounded-up ComputeDemand and Rounded MemoryDemand
  - Set ScaleTargetPerReplica to ScaleUpPerReplicaCapacity if ScaleUp = true
  - Otherwise, Set ScaleTargetPerReplica to ScaleDownPerReplicaCapacity when ScaleTargetPerReplica is still lower than PerReplicaCapacity

## Optimize with Multi-dimensional Optimizer

MultiDimensionalOptimizer implements ScalingOptimizer with the same outer structure as GreedyByScoreOptimizer, but adds DRA capacity tracking inventory (ResourceCapacityInventory) alongside the GPU budget.

### ResourceCapacityInventory

Inventory informed by ResourceSlices and ResourceClaims informers (watcher).
This inventory keeps the available resource capacity of each device in each pool.

### Optimize Steps

Fetch status from ResourceCapacityInventory

For each model,

- initRoleState
- if anyRoleNeedsScaleUp:
  - applyVerticalScaleUp
    - Only patches vc.ScaleUpPerReplicaCapacity as a verticalTarget when enough remaining capacity with headroom
  - If success, update remaining capacity
  - If not, proceed HorizontalScaling
- Otherwise, try scale down:
  - scaleDownRoleIterated (Horizontal scaling down)
  - applyVerticalScaleDown
    - Only set a verticalTarget when not scaling down on horizontal
- buildDecisionWithOptimizer
