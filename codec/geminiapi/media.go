package geminiapi

import (
	"net/url"
	"strings"

	"github.com/looprig/core/content"
)

// The media types Gemini's Blob accepts, transcribed from the v1beta discovery
// document's own description of Blob.mimeType. They are the contract, so the
// encoder holds a request to them locally rather than sending bytes it can
// prove will be refused: a local error names the block and the media type, a
// provider 400 names neither.
//
// The list is split the way Google's description splits it. Audio is published
// as the wildcard `audio/*` plus two container-qualified forms, so it is matched
// by prefix (isBlobAudioMIME) rather than enumerated; the Images, Text and
// Applications groups are closed lists and are enumerated here. Video is not
// listed, because no neutral block type maps to it.
//
// Note what is absent. The Office open XML types core/content names —
// application/vnd.openxmlformats-officedocument.* (.docx/.xlsx) — are not in
// Google's list, so a document holding those bytes fails closed. Its extracted
// text, if it has any, still travels as Part.text, which carries no mime.
var blobDocumentMIMETypes = map[content.MediaType]struct{}{
	// Text
	"text/plain":           {},
	"text/html":            {},
	"text/css":             {},
	"text/javascript":      {},
	"text/x-typescript":    {},
	"text/csv":             {},
	"text/markdown":        {},
	"text/x-python":        {},
	"text/xml":             {},
	"text/rtf":             {},
	"video/text/timestamp": {},
	// Applications
	"application/x-javascript":  {},
	"application/x-typescript":  {},
	"application/x-python-code": {},
	"application/json":          {},
	"application/x-ipynb+json":  {},
	"application/rtf":           {},
	"application/pdf":           {},
}

// blobImageMIMETypes is the Images half of the same description: "Images:
// image/png, image/jpeg, image/jpg, image/webp, image/heic, image/heif,
// image/gif, image/avif". A closed list, so it is an allowlist — a type Google
// adds later fails closed here rather than reaching the model as a Blob whose
// mimeType it does not admit.
//
// Note what this excludes. content.MediaTypeImageSVG (image/svg+xml) is a
// first-class member of the neutral vocabulary and is NOT on Google's list, so
// an SVG now fails closed instead of travelling as bytes Gemini refuses. So do
// image/bmp and image/tiff, which callers reach for freely.
//
// image/jpg is not an IANA type, but Google lists it, and an MCP server that
// declares it is naming the same container as image/jpeg — it is transcribed
// rather than normalised so the wire value stays exactly what Google published.
var blobImageMIMETypes = map[content.MediaType]struct{}{
	"image/png":  {},
	"image/jpeg": {},
	"image/jpg":  {},
	"image/webp": {},
	"image/heic": {},
	"image/heif": {},
	"image/gif":  {},
	"image/avif": {},
}

// blobAudioMIMETypes holds the two audio forms Google lists explicitly outside
// the `audio/*` wildcard.
var blobAudioMIMETypes = map[content.MediaType]struct{}{
	"video/audio/s16le": {},
	"video/audio/wav":   {},
}

const audioMIMEPrefix = "audio/"

// isBlobAudioMIME reports whether mediaType is an audio type Blob accepts.
// `audio/*` is Google's own wildcard, so the prefix match is a transcription of
// the published list and not a relaxation of it; a bare "audio/" with no subtype
// is not a media type and is refused.
func isBlobAudioMIME(mediaType content.MediaType) bool {
	if _, ok := blobAudioMIMETypes[mediaType]; ok {
		return true
	}
	return strings.HasPrefix(string(mediaType), audioMIMEPrefix) && len(mediaType) > len(audioMIMEPrefix)
}

// isBlobDocumentMIME reports whether mediaType is one of the document types
// Blob accepts.
func isBlobDocumentMIME(mediaType content.MediaType) bool {
	_, ok := blobDocumentMIMETypes[mediaType]
	return ok
}

// isBlobImageMIME reports whether mediaType is one of the image types Blob
// accepts.
func isBlobImageMIME(mediaType content.MediaType) bool {
	_, ok := blobImageMIMETypes[mediaType]
	return ok
}

// The URI forms FileData.fileUri accepts. The discovery document types fileUri
// as "Required. URI." and says nothing more, so the constraint is transcribed
// from Google's file-input documentation instead — which is why it is a small,
// explicitly-listed set rather than a parse of the URI grammar:
//
//   - the Files API resource URI, which is what files.upload returns, on
//     generativelanguage.googleapis.com;
//   - a Cloud Storage object, gs://…, which the Vertex flavour of this dialect
//     accepts;
//   - a YouTube watch URL, the one public web URL Gemini will fetch itself.
//
// Everything else — an arbitrary https:// image on someone's CDN, above all —
// is refused. Gemini does not fetch such a URL: the part is ignored or the
// request is rejected, and either way the caller learns nothing. That silent
// drop is the defect this list exists to close.
const (
	filesAPIHost       = "generativelanguage.googleapis.com"
	cloudStorageScheme = "gs://"
)

// youTubeHosts are the hosts a Gemini-fetchable YouTube URL may carry. Matched
// on the host alone, because the path shapes differ per host and none of them
// is the thing being validated.
var youTubeHosts = map[string]struct{}{
	"www.youtube.com": {},
	"youtube.com":     {},
	"m.youtube.com":   {},
	"youtu.be":        {},
}

// fileURIReason reports why uri is not a URI FileData.fileUri accepts, or ""
// when it is one. It returns a bare reason so the call site can attach the
// typed error its context already uses, matching toolNameReason and the
// sibling codecs.
func fileURIReason(uri string) string {
	if uri == "" {
		return "fileData.fileUri is required and the block carries no URI"
	}
	if strings.HasPrefix(uri, cloudStorageScheme) && len(uri) > len(cloudStorageScheme) {
		return ""
	}
	parsed, err := url.Parse(uri)
	if err != nil {
		return "URI is unparseable"
	}
	if parsed.Scheme == "https" {
		if parsed.Host == filesAPIHost {
			return ""
		}
		if _, ok := youTubeHosts[parsed.Host]; ok {
			return ""
		}
	}
	return "fileData.fileUri accepts only a Files API URI (https://" + filesAPIHost +
		"/…), a Cloud Storage gs:// URI, or a YouTube URL; Gemini does not fetch an arbitrary web URL, " +
		"so materialize the bytes into the block instead"
}
