package jsonbody_test

import (
	"errors"
	"io"
	"testing"

	"github.com/looprig/inference/wire/jsonbody"
)

type sample struct {
	Name  string `json:"name"`
	Count int    `json:"count"`
}

func TestEncode(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name     string
		value    any
		wantBody string
		wantErr  bool
	}{
		{name: "struct", value: sample{Name: "a", Count: 2}, wantBody: `{"name":"a","count":2}`},
		{name: "empty struct", value: sample{}, wantBody: `{"name":"","count":0}`},
		{name: "map", value: map[string]int{"x": 1}, wantBody: `{"x":1}`},
		{name: "nil", value: nil, wantBody: `null`},
		{name: "unmarshalable channel", value: make(chan int), wantErr: true},
	}

	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			r, ct, err := jsonbody.Encode(tc.value)
			if (err != nil) != tc.wantErr {
				t.Fatalf("Encode() err = %v, wantErr %v", err, tc.wantErr)
			}
			if tc.wantErr {
				var ee *jsonbody.EncodeError
				if !errors.As(err, &ee) {
					t.Fatalf("Encode() err = %T, want *jsonbody.EncodeError", err)
				}
				return
			}
			if ct != jsonbody.ContentType {
				t.Errorf("content type = %q, want %q", ct, jsonbody.ContentType)
			}
			got, err := io.ReadAll(r)
			if err != nil {
				t.Fatalf("ReadAll: %v", err)
			}
			if string(got) != tc.wantBody {
				t.Errorf("body = %s, want %s", got, tc.wantBody)
			}
		})
	}
}

func TestDecode(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name    string
		body    string
		want    sample
		wantErr bool
	}{
		{name: "valid", body: `{"name":"z","count":9}`, want: sample{Name: "z", Count: 9}},
		{name: "empty object", body: `{}`, want: sample{}},
		{name: "malformed", body: `{not json`, wantErr: true},
		{name: "truncated", body: `{"name":`, wantErr: true},
	}

	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			var got sample
			err := jsonbody.Decode([]byte(tc.body), &got)
			if (err != nil) != tc.wantErr {
				t.Fatalf("Decode() err = %v, wantErr %v", err, tc.wantErr)
			}
			if tc.wantErr {
				var de *jsonbody.DecodeError
				if !errors.As(err, &de) {
					t.Fatalf("Decode() err = %T, want *jsonbody.DecodeError", err)
				}
				return
			}
			if got != tc.want {
				t.Errorf("Decode() = %+v, want %+v", got, tc.want)
			}
		})
	}
}

// TestRoundTrip proves Encode then Decode preserves the value.
func TestRoundTrip(t *testing.T) {
	t.Parallel()
	in := sample{Name: "round", Count: 42}
	r, _, err := jsonbody.Encode(in)
	if err != nil {
		t.Fatalf("Encode: %v", err)
	}
	raw, err := io.ReadAll(r)
	if err != nil {
		t.Fatalf("ReadAll: %v", err)
	}
	var out sample
	if err := jsonbody.Decode(raw, &out); err != nil {
		t.Fatalf("Decode: %v", err)
	}
	if out != in {
		t.Errorf("round trip = %+v, want %+v", out, in)
	}
}
