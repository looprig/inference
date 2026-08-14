package conformance

// The types in this file are the on-disk contract between cmd/schemagen and
// the gate. They are exported so the generator can build exactly what the gate
// reads, rather than the two agreeing by convention.

// Index maps an api-format to its message kinds. It is serialised as
// schema/index.json.
type Index map[string]map[string]*Entry

// Entry locates the schema for one (api-format, message-kind) pair.
type Entry struct {
	// Document is the slash-separated path of the schema document inside the
	// schema/ tree.
	Document string `json:"document"`
	// Direction is DirectionRequest for what Looprig sends and
	// DirectionResponse for what the provider returns.
	Direction string `json:"direction"`
	// Root is the JSON pointer of the entry subschema within that document,
	// always of the form "#/$defs/Name".
	Root string `json:"root"`
	// Union is set when the kind is a union of message shapes, such as a
	// stream event. It lets the gate report a violation against the one member
	// the payload claims to be instead of against every branch at once.
	Union *Union `json:"union,omitempty"`
}

// Union describes how a payload selects its member of a union kind.
type Union struct {
	// Style is UnionStyleProperty or UnionStyleMemberKey.
	Style string `json:"style"`
	// Property is the discriminating property name, for UnionStyleProperty.
	Property string `json:"property,omitempty"`
	// Members maps a discriminator value (or, for UnionStyleMemberKey, a
	// member property name) to that member's pointer within the document.
	Members map[string]string `json:"members"`
	// Ambiguous lists discriminator values that more than one member claims.
	// They are deliberately absent from Members: focusing on one of them would
	// be a guess, so such payloads are validated against the whole union.
	Ambiguous []string `json:"ambiguous,omitempty"`
}

const (
	// DirectionRequest marks a kind that describes an outbound request body.
	// Validating one catches an encoder bug before the bytes reach a live API,
	// which is why request kinds exist at all.
	DirectionRequest = "request"
	// DirectionResponse marks a kind that describes an inbound provider
	// message.
	DirectionResponse = "response"
)

const (
	// UnionStyleProperty selects a member by the value of a shared property,
	// as OpenAI and Anthropic stream events do with "type".
	UnionStyleProperty = "property"
	// UnionStyleMemberKey selects a member by which single property is
	// present, as the Smithy union encoding does.
	UnionStyleMemberKey = "member-key"
)

// Provenance records where every input document came from. It is serialised as
// schema/provenance.json and asserted by provenance_test.go.
type Provenance struct {
	Comment string             `json:"comment"`
	Sources map[string]*Source `json:"sources"`
}

// Source is the provenance of one upstream API description.
type Source struct {
	// URL is the exact document the schemas were derived from.
	URL string `json:"url"`
	// File is the name it is stored under in the generator's -specs directory.
	File string `json:"file"`
	// Dialect names the conversion applied to it.
	Dialect string `json:"dialect"`
	// Publisher records who publishes the bytes.
	Publisher string `json:"publisher"`
	// Hosting is empty for first-party-hosted sources, and otherwise states
	// the chain of custody and why no first-party option exists.
	Hosting string `json:"hosting,omitempty"`
	// Retrieved is the UTC date the recorded bytes were last observed. It is
	// carried forward unchanged when a refresh produces identical bytes.
	Retrieved string `json:"retrieved"`
	// Bytes and SHA256 identify the exact document consumed.
	Bytes  int    `json:"bytes"`
	SHA256 string `json:"sha256"`
	// CanonicalSHA256 is the hash of a key-sorted re-encoding, present only for
	// sources whose serialisation is not byte-stable across fetches.
	CanonicalSHA256 string `json:"canonical_sha256,omitempty"`
	CanonicalNote   string `json:"canonical_note,omitempty"`
	// Fields holds a pointer document's parsed key/value pairs.
	Fields map[string]string `json:"fields,omitempty"`
	// PointerSource, PointerHash and PointerHashNote record the first-party
	// document that names this source, for sources that are not first-party
	// hosted.
	PointerSource   string `json:"pointer_source,omitempty"`
	PointerHash     string `json:"pointer_hash,omitempty"`
	PointerHashNote string `json:"pointer_hash_note,omitempty"`
}
