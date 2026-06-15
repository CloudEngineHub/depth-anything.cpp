// Package da3 provides a purego (cgo-less) binding to libdepthanything.so,
// the static-linked Depth Anything 3 C-API shared library.
//
// Usage:
//
//	if err := da3.Open("/path/to/libdepthanything.so"); err != nil { ... }
//	m, err := da3.Load("/path/to/model.gguf", 4)
//	defer m.Close()
//	depth, h, w, ext, intr, err := m.DepthPose("/path/to/image.png")
//
// The library is loaded once per process via Open. All C-API calls are routed
// through function pointers registered with purego.RegisterLibFunc.
package da3

import (
	"errors"
	"fmt"
	"sync"
	"unsafe"

	"github.com/ebitengine/purego"
)

// C-API function pointers, populated by Open.
var (
	cABIVersion func() int32
	cLoad       func(ggufPath string, nThreads int32) uintptr
	cFree       func(ctx uintptr)
	cInfoJSON   func(ctx uintptr) *byte
	cFreeString func(s *byte)
	cLastError  func(ctx uintptr) *byte
	cDepthPath  func(ctx uintptr, imagePath string, outH *int32, outW *int32) *float32
	cFreeFloats func(p *float32)
	cPosePath   func(ctx uintptr, imagePath string, outExt *float32, outIntr *float32) int32
)

var (
	openOnce sync.Once
	openErr  error
	opened   bool
	openedMu sync.Mutex
)

// Open loads libdepthanything.so and registers the C-API functions. It is safe
// to call Open multiple times; the library is loaded only once.
func Open(soPath string) error {
	openedMu.Lock()
	defer openedMu.Unlock()
	if opened {
		return nil
	}
	openOnce.Do(func() {
		handle, err := purego.Dlopen(soPath, purego.RTLD_NOW|purego.RTLD_GLOBAL)
		if err != nil {
			openErr = fmt.Errorf("da3: dlopen %q: %w", soPath, err)
			return
		}
		purego.RegisterLibFunc(&cABIVersion, handle, "da_capi_abi_version")
		purego.RegisterLibFunc(&cLoad, handle, "da_capi_load")
		purego.RegisterLibFunc(&cFree, handle, "da_capi_free")
		purego.RegisterLibFunc(&cInfoJSON, handle, "da_capi_info_json")
		purego.RegisterLibFunc(&cFreeString, handle, "da_capi_free_string")
		purego.RegisterLibFunc(&cLastError, handle, "da_capi_last_error")
		purego.RegisterLibFunc(&cDepthPath, handle, "da_capi_depth_path")
		purego.RegisterLibFunc(&cFreeFloats, handle, "da_capi_free_floats")
		purego.RegisterLibFunc(&cPosePath, handle, "da_capi_pose_path")
	})
	if openErr != nil {
		return openErr
	}
	opened = true
	return nil
}

// ABIVersion returns the C-API ABI version reported by the library.
func ABIVersion() int {
	if cABIVersion == nil {
		return 0
	}
	return int(cABIVersion())
}

// Model is a loaded Depth Anything 3 model context. It is NOT reentrant: the C
// side shares a ggml graph allocator, so callers must serialize inference.
type Model struct {
	ctx uintptr
	mu  sync.Mutex
}

// Load loads a GGUF model and returns a Model. nThreads controls CPU threads.
func Load(ggufPath string, nThreads int) (*Model, error) {
	if cLoad == nil {
		return nil, errors.New("da3: library not opened (call Open first)")
	}
	ctx := cLoad(ggufPath, int32(nThreads))
	if ctx == 0 {
		return nil, fmt.Errorf("da3: failed to load model %q", ggufPath)
	}
	return &Model{ctx: ctx}, nil
}

// Close frees the model context. Safe to call on a nil/zero Model.
func (m *Model) Close() {
	if m == nil || m.ctx == 0 || cFree == nil {
		return
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	cFree(m.ctx)
	m.ctx = 0
}

// lastError returns the context's last error string, or "" if none.
func (m *Model) lastError() string {
	if cLastError == nil || m.ctx == 0 {
		return ""
	}
	return goString(cLastError(m.ctx))
}

// Info returns the model's config as a JSON string.
func (m *Model) Info() (string, error) {
	if m == nil || m.ctx == 0 {
		return "", errors.New("da3: nil/closed model")
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	ptr := cInfoJSON(m.ctx)
	if ptr == nil {
		return "", fmt.Errorf("da3: info_json failed: %s", m.lastError())
	}
	s := goString(ptr)
	cFreeString(ptr)
	return s, nil
}

// DepthPose runs depth estimation and pose recovery on an image file. It
// returns the row-major depth map (length h*w), its dimensions, the 3x4
// extrinsics (row-major, 12 floats) and 3x3 intrinsics (9 floats).
func (m *Model) DepthPose(imagePath string) (depth []float32, h, w int, ext [12]float32, intr [9]float32, err error) {
	if m == nil || m.ctx == 0 {
		err = errors.New("da3: nil/closed model")
		return
	}
	m.mu.Lock()
	defer m.mu.Unlock()

	var ch, cw int32
	ptr := cDepthPath(m.ctx, imagePath, &ch, &cw)
	if ptr == nil {
		err = fmt.Errorf("da3: depth_path failed: %s", m.lastError())
		return
	}
	h, w = int(ch), int(cw)
	n := h * w
	if n > 0 {
		src := unsafe.Slice(ptr, n)
		depth = make([]float32, n)
		copy(depth, src)
	}
	cFreeFloats(ptr)

	if rc := cPosePath(m.ctx, imagePath, &ext[0], &intr[0]); rc != 0 {
		err = fmt.Errorf("da3: pose_path failed (rc=%d): %s", rc, m.lastError())
		return
	}
	return
}

// goString copies a NUL-terminated C string at p into a Go string. Returns ""
// for a nil pointer.
func goString(p *byte) string {
	if p == nil {
		return ""
	}
	var n int
	for *(*byte)(unsafe.Add(unsafe.Pointer(p), n)) != 0 {
		n++
	}
	if n == 0 {
		return ""
	}
	return string(unsafe.Slice(p, n))
}
