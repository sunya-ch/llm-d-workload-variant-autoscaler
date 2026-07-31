package queueingmodel

import (
	"github.com/llm-d/llm-d-workload-variant-autoscaler/internal/engines/analyzers/queueingmodel/tuner"
	"github.com/llm-d/llm-d-workload-variant-autoscaler/internal/interfaces"
)

// QMConfig implements interfaces.AnalyzerConfig
type QMConfig struct {
	// SLOTargets maps (modelID, namespace) to SLO targets
	// Key format: "namespace/modelID"
	SLOTargets map[string]*SLOTarget

	// SLOMultiplier is the queueing delay multiplier (k>1) used when inferring SLO
	// targets from the queueing model. It controls how much the iteration time is
	// allowed to inflate relative to the idle baseline alpha:
	//   TargetTTFT = k×alpha + (beta+gamma)×input_len
	//   TargetITL  = k×alpha + beta + gamma×(input_len + (output_len+1)/2)
	// The utilization correspondence is rho = 1 - 1/k.
	// Zero value means use DefaultSLOMultiplier (3.0, rho=0.67).
	SLOMultiplier float64

	// ScaleUpSLOFactor tightens the SLO for the vertical scale-up throughput target.
	// TargetTTFT/ITL are divided by this factor before calling queueAnalyzer.Size,
	// yielding a lower maxRPS (more resource needed). e.g. 0.75 → 25% tighter budget.
	// Zero value uses DefaultScaleUpSLOFactor.
	ScaleUpSLOFactor float64

	// ScaleDownSLOFactor relaxes the SLO for the vertical scale-down floor computation.
	// TargetTTFT/ITL are multiplied by this factor, yielding a higher maxRPS (less
	// resource needed). e.g. 2.0 → 2× looser budget.
	// Zero value uses DefaultScaleDownSLOFactor.
	ScaleDownSLOFactor float64

	// Tuning configuration
	TuningEnabled bool

	// FilterConfig provides user customization for the Kalman filter behavior.
	// These parameters (GammaFactor, ErrorLevel, TPercentile) control the filter's
	// noise model and are hardware-independent, so they are shared across all variants.
	// If nil, default filter configuration will be used.
	FilterConfig *tuner.FilterData
}

// SLOTarget defines TTFT/ITL targets for a model
type SLOTarget struct {
	TargetTTFT float32 // Target time-to-first-token (ms)
	TargetITL  float32 // Target inter-token latency (ms)
}

// GetAnalyzerName implements interfaces.AnalyzerConfig
func (c *QMConfig) GetAnalyzerName() string {
	return interfaces.QueueingModelAnalyzerName
}

// scaleUpFactor returns the effective ScaleUpSLOFactor, applying the default when zero.
func (c *QMConfig) scaleUpFactor() float64 {
	if c.ScaleUpSLOFactor == 0 {
		return DefaultScaleUpSLOFactor
	}
	return c.ScaleUpSLOFactor
}

// scaleDownFactor returns the effective ScaleDownSLOFactor, applying the default when zero.
func (c *QMConfig) scaleDownFactor() float64 {
	if c.ScaleDownSLOFactor == 0 {
		return DefaultScaleDownSLOFactor
	}
	return c.ScaleDownSLOFactor
}

// GetSLOForModel retrieves SLO targets for a model in a namespace
func (c *QMConfig) GetSLOForModel(namespace, modelID string) *SLOTarget {
	if c.SLOTargets == nil {
		return nil
	}
	key := MakeModelKey(namespace, modelID)
	return c.SLOTargets[key]
}

// make an SLOTarget the maximum of itself and other target
func (t *SLOTarget) Max(other *SLOTarget) {
	if other != nil {
		t.TargetITL = max(t.TargetITL, other.TargetITL)
		t.TargetTTFT = max(t.TargetTTFT, other.TargetTTFT)
	}
}
