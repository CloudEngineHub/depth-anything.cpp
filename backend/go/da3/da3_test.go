package da3

import (
	"os"
	"testing"
)

func TestDepthPose(t *testing.T) {
	so := os.Getenv("DA3_SO")
	gguf := os.Getenv("DA3_TEST_GGUF")
	img := os.Getenv("DA3_TEST_IMAGE")
	if so == "" || gguf == "" || img == "" {
		t.Skip("DA3_SO/DA3_TEST_GGUF/DA3_TEST_IMAGE unset")
	}
	if err := Open(so); err != nil {
		t.Fatal(err)
	}
	if ABIVersion() != 1 {
		t.Fatalf("abi=%d", ABIVersion())
	}
	m, err := Load(gguf, 4)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	info, err := m.Info()
	if err != nil || info == "" {
		t.Fatalf("info: %v %q", err, info)
	}
	t.Logf("info=%s", info)
	depth, h, w, ext, intr, err := m.DepthPose(img)
	if err != nil {
		t.Fatal(err)
	}
	if len(depth) != h*w || h == 0 {
		t.Fatalf("depth len=%d h=%d w=%d", len(depth), h, w)
	}
	for _, v := range depth {
		if v != v {
			t.Fatal("NaN in depth")
		}
	}
	t.Logf("depth %dx%d depth[0]=%f ext[0]=%f intr[0]=%f", w, h, depth[0], ext[0], intr[0])
}
