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
