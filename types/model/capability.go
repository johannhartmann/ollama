package model

type Capability string

const (
	CapabilityCompletion  = Capability("completion")
	CapabilityTools       = Capability("tools")
	CapabilityInsert      = Capability("insert")
	CapabilityVision      = Capability("vision")
	CapabilityEmbedding   = Capability("embedding")
	CapabilityMultivector = Capability("multivector")
	CapabilityThinking    = Capability("thinking")
	CapabilityImage       = Capability("image")
	CapabilityAudio       = Capability("audio")
)

func (c Capability) String() string {
	return string(c)
}
