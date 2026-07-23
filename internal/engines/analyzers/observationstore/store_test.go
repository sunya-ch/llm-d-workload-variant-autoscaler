package observationstore

import (
	"testing"
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	"github.com/llm-d/llm-d-workload-variant-autoscaler/internal/interfaces"
)

func TestObservationStore(t *testing.T) {
	RegisterFailHandler(Fail)
	RunSpecs(t, "ObservationStore Suite")
}

const (
	testAlpha           = 0.1
	testBytesPerKVToken = 128.0
)

var _ = Describe("VariantObservationStore", func() {
	var s *VariantObservationStore

	BeforeEach(func() {
		s = NewVariantObservationStore()
	})

	Describe("Update and Get", func() {
		It("returns a stored observation by key", func() {
			s.Update("ns", "model", "v1", VariantObservation{
				ComputeIntensity:    1.0,
				MaxComputeIntensity: 2.0,
				MemoryWeight:        3.0,
				BytePerToken:        4.0,
			})
			got := s.Get("ns", "model", "v1")
			Expect(got).NotTo(BeNil())
			Expect(got.ComputeIntensity).To(Equal(1.0))
			Expect(got.MaxComputeIntensity).To(Equal(2.0))
			Expect(got.MemoryWeight).To(Equal(3.0))
			Expect(got.BytePerToken).To(Equal(4.0))
		})

		It("returns nil for a missing key", func() {
			Expect(s.Get("ns", "model", "missing")).To(BeNil())
		})

		It("returns a copy — mutations do not affect the store", func() {
			s.Update("ns", "model", "v1", VariantObservation{ComputeIntensity: 5.0})
			got := s.Get("ns", "model", "v1")
			got.ComputeIntensity = 99.0
			Expect(s.Get("ns", "model", "v1").ComputeIntensity).To(Equal(5.0))
		})
	})

	Describe("EvictStale", func() {
		It("removes entries older than the timeout and preserves fresh ones", func() {
			s.Update("ns", "model", "old", VariantObservation{})
			// Backdate the old entry.
			s.mu.Lock()
			s.data["ns|model|old"].UpdatedAt = time.Now().Add(-48 * time.Hour)
			s.mu.Unlock()

			s.Update("ns", "model", "new", VariantObservation{ComputeIntensity: 1.0})

			evicted := s.EvictStale(24 * time.Hour)
			Expect(evicted).To(Equal(1))
			Expect(s.Get("ns", "model", "old")).To(BeNil())
			Expect(s.Get("ns", "model", "new")).NotTo(BeNil())
		})
	})

	Describe("UpdateFromReplicaMetrics", func() {
		Context("ComputeIntensity", func() {
			It("is updated every tick as the mean I_live across replicas", func() {
				rms := []interfaces.ReplicaMetrics{
					{PromptTokenRate: 10.0, GenerationTokenRate: 20.0}, // I_live = 10 + 0.1*20 = 12
					{PromptTokenRate: 20.0, GenerationTokenRate: 40.0}, // I_live = 20 + 0.1*40 = 24
				}
				s.UpdateFromReplicaMetrics("ns", "model", "v1", rms, testAlpha, testBytesPerKVToken)

				got := s.Get("ns", "model", "v1")
				Expect(got).NotTo(BeNil())
				Expect(got.ComputeIntensity).To(Equal((12.0 + 24.0) / 2)) // mean = 18
			})

			It("is zero when replica list is empty", func() {
				s.UpdateFromReplicaMetrics("ns", "model", "v1", nil, testAlpha, testBytesPerKVToken)
				Expect(s.Get("ns", "model", "v1").ComputeIntensity).To(Equal(0.0))
			})
		})

		Context("MaxComputeIntensity", func() {
			It("stays zero when queue is empty", func() {
				rms := []interfaces.ReplicaMetrics{
					{PromptTokenRate: 50.0, QueueLength: 0},
				}
				s.UpdateFromReplicaMetrics("ns", "model", "v1", rms, testAlpha, testBytesPerKVToken)
				Expect(s.Get("ns", "model", "v1").MaxComputeIntensity).To(Equal(0.0))
			})

			It("updates to the current I_live when queue is non-empty", func() {
				rms := []interfaces.ReplicaMetrics{
					{PromptTokenRate: 100.0, QueueLength: 5},
				}
				s.UpdateFromReplicaMetrics("ns", "model", "v1", rms, testAlpha, testBytesPerKVToken)
				Expect(s.Get("ns", "model", "v1").MaxComputeIntensity).To(Equal(100.0))
			})

			It("holds the high-water mark and does not decrease", func() {
				rms := []interfaces.ReplicaMetrics{{PromptTokenRate: 100.0, QueueLength: 5}}
				s.UpdateFromReplicaMetrics("ns", "model", "v1", rms, testAlpha, testBytesPerKVToken)

				rms2 := []interfaces.ReplicaMetrics{{PromptTokenRate: 30.0, QueueLength: 2}}
				s.UpdateFromReplicaMetrics("ns", "model", "v1", rms2, testAlpha, testBytesPerKVToken)

				Expect(s.Get("ns", "model", "v1").MaxComputeIntensity).To(Equal(100.0))
			})
		})

		Context("MemoryWeight", func() {
			It("uses the default util (0.9) when GpuMemoryUtilization is zero", func() {
				rms := []interfaces.ReplicaMetrics{
					{TotalKvCapacityTokens: 1000},
				}
				s.UpdateFromReplicaMetrics("ns", "model", "v1", rms, testAlpha, testBytesPerKVToken)

				// util=0.9 (default): 1000 × (1-0.9)/0.9 × 128 ≈ 14222.2
				want := float64(1000) * (1 - 0.9) / 0.9 * testBytesPerKVToken
				Expect(s.Get("ns", "model", "v1").MemoryWeight).To(BeNumerically("~", want, 1e-6))
			})

			It("uses GpuMemoryUtilization from ReplicaMetrics when set", func() {
				rms := []interfaces.ReplicaMetrics{
					{TotalKvCapacityTokens: 1000, GpuMemoryUtilization: 0.85},
				}
				s.UpdateFromReplicaMetrics("ns", "model", "v1", rms, testAlpha, testBytesPerKVToken)

				// util=0.85: 1000 × (1-0.85)/0.85 × 128 ≈ 22588.2
				want := float64(1000) * (1 - 0.85) / 0.85 * testBytesPerKVToken
				Expect(s.Get("ns", "model", "v1").MemoryWeight).To(BeNumerically("~", want, 1e-6))
			})

			It("is never overwritten once set", func() {
				rms := []interfaces.ReplicaMetrics{{TotalKvCapacityTokens: 1000, GpuMemoryUtilization: 0.9}}
				s.UpdateFromReplicaMetrics("ns", "model", "v1", rms, testAlpha, testBytesPerKVToken)
				firstMW := s.Get("ns", "model", "v1").MemoryWeight

				rms2 := []interfaces.ReplicaMetrics{{TotalKvCapacityTokens: 9999, GpuMemoryUtilization: 0.9}}
				s.UpdateFromReplicaMetrics("ns", "model", "v1", rms2, testAlpha, testBytesPerKVToken)
				Expect(s.Get("ns", "model", "v1").MemoryWeight).To(Equal(firstMW))
			})
		})

		Context("BytePerToken", func() {
			It("stays zero when no deltas are available", func() {
				rms := []interfaces.ReplicaMetrics{{DeltaCacheBytes: 0, DeltaTokens: 0}}
				s.UpdateFromReplicaMetrics("ns", "model", "v1", rms, testAlpha, testBytesPerKVToken)
				Expect(s.Get("ns", "model", "v1").BytePerToken).To(Equal(0.0))
			})

			It("is computed as DeltaCacheBytes / DeltaTokens when deltas are available", func() {
				rms := []interfaces.ReplicaMetrics{{DeltaCacheBytes: 1024.0, DeltaTokens: 8.0}}
				s.UpdateFromReplicaMetrics("ns", "model", "v1", rms, testAlpha, testBytesPerKVToken)
				Expect(s.Get("ns", "model", "v1").BytePerToken).To(Equal(128.0))
			})

			It("is refreshed each tick when new deltas are available", func() {
				rms := []interfaces.ReplicaMetrics{{DeltaCacheBytes: 1024.0, DeltaTokens: 8.0}}
				s.UpdateFromReplicaMetrics("ns", "model", "v1", rms, testAlpha, testBytesPerKVToken)

				rms2 := []interfaces.ReplicaMetrics{{DeltaCacheBytes: 512.0, DeltaTokens: 8.0}}
				s.UpdateFromReplicaMetrics("ns", "model", "v1", rms2, testAlpha, testBytesPerKVToken)
				Expect(s.Get("ns", "model", "v1").BytePerToken).To(Equal(64.0))
			})
		})

		Context("QM-only deployment", func() {
			It("populates all signals correctly without sat V2 ever writing", func() {
				rms := []interfaces.ReplicaMetrics{
					{
						PromptTokenRate:       5.0,
						GenerationTokenRate:   10.0,
						QueueLength:           3,
						TotalKvCapacityTokens: 500,
						DeltaCacheBytes:       256.0,
						DeltaTokens:           4.0,
					},
				}
				s.UpdateFromReplicaMetrics("ns", "model", "qm-variant", rms, testAlpha, testBytesPerKVToken)

				got := s.Get("ns", "model", "qm-variant")
				Expect(got).NotTo(BeNil())
				// I_live = 5 + 0.1*10 = 6
				Expect(got.ComputeIntensity).To(Equal(6.0))
				// queue > 0 → MaxComputeIntensity = 6
				Expect(got.MaxComputeIntensity).To(Equal(6.0))
				// MemoryWeight = 500 × (1-0.9)/0.9 × 128
				wantMW := float64(500) * (1 - 0.9) / 0.9 * testBytesPerKVToken
				Expect(got.MemoryWeight).To(BeNumerically("~", wantMW, 1e-6))
				// BytePerToken = 256/4 = 64
				Expect(got.BytePerToken).To(Equal(64.0))
			})
		})
	})
})
