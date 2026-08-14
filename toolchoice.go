package inference

// ToolChoiceMode enumerates the tool-choice variants. It is the discriminant a
// codec switches on; it is not itself a choice, because two of the three
// variants carry no data and the third is meaningless without its tool name.
type ToolChoiceMode uint8

const (
	// ToolChoiceModeAuto lets the model decide between text and tools.
	ToolChoiceModeAuto ToolChoiceMode = iota
	// ToolChoiceModeRequired forces some tool call, leaving the model the choice
	// of which.
	ToolChoiceModeRequired
	// ToolChoiceModeNamed forces the one tool the choice names.
	ToolChoiceModeNamed
)

// ToolChoice controls whether the model may choose between text and tools,
// must call some tool, or must call one named tool.
//
// It is an opaque comparable value built only by ToolAuto, ToolRequired and
// ToolNamed, so the forced tool name cannot be separated from the variant that
// gives it meaning: "named choice, no name" and "name, but not a named choice"
// are unspellable rather than merely validated. Carrying the name in a sibling
// Request field would make both reachable and would need a validation error
// code apiece to catch after the fact; keeping it inside the variant removes
// the states instead of policing them.
//
// One cross-field invariant a type cannot encode remains, and is checked in
// ValidateRequestFeatures: a named choice whose name matches no declared tool.
// No provider schema enforces that either — measured across all five.
//
// The zero value is ToolAuto(), so a Request struct literal that never mentions
// ToolChoice keeps the automatic behavior. That requirement is why this is a
// struct and not a sealed interface in the shape of content.Block: an interface
// has nil for a zero value, so "auto" would have two spellings and every
// consumer would need a nil arm — reintroducing exactly the kind of invalid
// state this type exists to remove.
//
// A variant is added here, not by callers: the discriminant is unexported and
// the constructors are the only way in. A future multi-name allowlist (Gemini's
// allowedFunctionNames, OpenAI's allowed_tools) is therefore an additive change
// to this file — one mode, one field, one constructor — with the single
// cross-cutting cost that a slice-valued field would end ToolChoice's
// comparability with ==.
type ToolChoice struct {
	mode ToolChoiceMode
	// name is meaningful only when mode is ToolChoiceModeNamed; no constructor
	// sets it for any other mode.
	name string
}

// ToolAuto lets the model choose between text and tools. It is the zero value.
func ToolAuto() ToolChoice { return ToolChoice{} }

// ToolRequired forces the model to call some tool of its own choosing. The
// request must declare at least one tool.
func ToolRequired() ToolChoice { return ToolChoice{mode: ToolChoiceModeRequired} }

// ToolNamed forces the model to call the single tool called name, which the
// request must declare. Every dialect this module encodes has a wire form for
// it.
func ToolNamed(name string) ToolChoice {
	return ToolChoice{mode: ToolChoiceModeNamed, name: name}
}

// Mode reports which variant this choice is.
func (c ToolChoice) Mode() ToolChoiceMode { return c.mode }

// Named reports the single tool the model must call. ok is false for every
// variant other than ToolChoiceModeNamed, so a caller cannot read a forced name out
// of a choice that does not force one.
func (c ToolChoice) Named() (name string, ok bool) {
	if c.mode != ToolChoiceModeNamed {
		return "", false
	}
	return c.name, true
}

// String renders the choice for diagnostics.
func (c ToolChoice) String() string {
	switch c.mode {
	case ToolChoiceModeAuto:
		return "auto"
	case ToolChoiceModeRequired:
		return "required"
	case ToolChoiceModeNamed:
		return "tool:" + c.name
	default:
		return "unknown"
	}
}

// ToolChoiceAuto and ToolChoiceRequired preserve the values exposed by the
// original two-state ToolChoice API. New code may use ToolAuto and
// ToolRequired; both spellings are equal and may be compared directly.
var (
	ToolChoiceAuto     = ToolAuto()
	ToolChoiceRequired = ToolRequired()
)
