// engine.go — purego bindings to libdepthanything (the flat C API in
// include/da_capi.h, ABI 7). No cgo: the shared library is dlopen'd once and each
// model is loaded as its own da_ctx. A context is NOT thread-safe (one ggml
// backend + compute graph), so all inference is serialized by the caller.
package main

import (
	"fmt"
	"runtime"
	"unsafe"

	"github.com/ebitengine/purego"
)

// capi is the dlopen'd library with the DA3 C API bound by name.
type capi struct {
	handle uintptr

	abiVersion  func() int32
	load        func(string, int32) uintptr
	loadNested  func(string, string, int32) uintptr
	freeCtx     func(uintptr)
	lastErrP    func(uintptr) uintptr
	freeFloats  func(uintptr)
	freeBytes   func(uintptr)
	pointsMulti func(ctx uintptr, paths uintptr, n int32, confPct float64, ptSize float32,
		outN, outCounts, outXyz, outRgb, outRad unsafe.Pointer) int32
	pointsStream func(ctx uintptr, paths uintptr, n, chunk, overlap int32, confPct float64,
		ptSize float32, budget int32,
		outN, outCounts, outXyz, outRgb, outRad unsafe.Pointer) int32
	gaussians func(ctx uintptr, path string,
		outN, outXyz, outScale, outQuat, outRgb, outOpacity,
		outIntr, outW, outH unsafe.Pointer) int32
}

func openCAPI(libPath string) (*capi, error) {
	h, err := purego.Dlopen(libPath, purego.RTLD_NOW|purego.RTLD_GLOBAL)
	if err != nil {
		return nil, fmt.Errorf("dlopen %s: %w", libPath, err)
	}
	a := &capi{handle: h}
	purego.RegisterLibFunc(&a.abiVersion, h, "da_capi_abi_version")
	purego.RegisterLibFunc(&a.load, h, "da_capi_load")
	purego.RegisterLibFunc(&a.loadNested, h, "da_capi_load_nested")
	purego.RegisterLibFunc(&a.freeCtx, h, "da_capi_free")
	purego.RegisterLibFunc(&a.lastErrP, h, "da_capi_last_error")
	purego.RegisterLibFunc(&a.freeFloats, h, "da_capi_free_floats")
	purego.RegisterLibFunc(&a.freeBytes, h, "da_capi_free_bytes")
	purego.RegisterLibFunc(&a.pointsMulti, h, "da_capi_points_multi")
	purego.RegisterLibFunc(&a.pointsStream, h, "da_capi_points_stream")
	purego.RegisterLibFunc(&a.gaussians, h, "da_capi_gaussians")
	return a, nil
}

func (a *capi) lastErr(ctx uintptr) string { return cstr(a.lastErrP(ctx)) }

// cstr reads a NUL-terminated C string at p (0 -> "").
func cstr(p uintptr) string {
	if p == 0 {
		return ""
	}
	var b []byte
	for i := uintptr(0); ; i++ {
		c := *(*byte)(unsafe.Pointer(p + i))
		if c == 0 {
			break
		}
		b = append(b, c)
	}
	return string(b)
}

func cFloats(p uintptr, n int) []float32 {
	if p == 0 || n <= 0 {
		return nil
	}
	out := make([]float32, n)
	copy(out, unsafe.Slice((*float32)(unsafe.Pointer(p)), n))
	return out
}

func cBytes(p uintptr, n int) []byte {
	if p == 0 || n <= 0 {
		return nil
	}
	out := make([]byte, n)
	copy(out, unsafe.Slice((*byte)(unsafe.Pointer(p)), n))
	return out
}

// Cloud is a fused multi-view colored point cloud (world frame, OpenCV axes).
type Cloud struct {
	N      int
	XYZ    []float32 // 3N
	RGB    []byte    // 3N
	Rad    []float32 // N (per-point world radius)
	Counts []int32   // per source view (frames outer; for build-up prefix sums)
}

// PointsMulti runs ONE cross-view pass over the given image files and returns the
// fused coherent cloud. confPct in [0,100], ptSize multiplies the per-point radius.
func (a *capi) PointsMulti(ctx uintptr, paths []string, confPct float64, ptSize float32) (*Cloud, error) {
	if len(paths) == 0 {
		return nil, fmt.Errorf("no images")
	}
	// Build a C char*[] in Go memory. Go's heap is non-moving, and the C side only
	// reads the strings during the (synchronous) call, so this is safe with KeepAlive.
	cs := make([][]byte, len(paths))
	pp := make([]*byte, len(paths))
	for i, s := range paths {
		b := append([]byte(s), 0)
		cs[i] = b
		pp[i] = &b[0]
	}
	var n int32
	counts := make([]int32, len(paths))
	var pXyz, pRgb, pRad uintptr
	rc := a.pointsMulti(ctx, uintptr(unsafe.Pointer(&pp[0])), int32(len(paths)), confPct, ptSize,
		unsafe.Pointer(&n), unsafe.Pointer(&counts[0]),
		unsafe.Pointer(&pXyz), unsafe.Pointer(&pRgb), unsafe.Pointer(&pRad))
	runtime.KeepAlive(cs)
	runtime.KeepAlive(pp)
	if rc != 0 {
		return nil, fmt.Errorf("points_multi: %s", a.lastErr(ctx))
	}
	np := int(n)
	cl := &Cloud{N: np, Counts: counts,
		XYZ: cFloats(pXyz, np*3), RGB: cBytes(pRgb, np*3), Rad: cFloats(pRad, np)}
	a.freeFloats(pXyz)
	a.freeBytes(pRgb)
	a.freeFloats(pRad)
	return cl, nil
}

// PointsStream runs the SLIDING-WINDOW streaming pipeline over an ordered frame
// list (time-lapse video): overlapping fused windows of `chunk` frames (sharing
// `overlap`) stitched into one global cloud. budget caps total points (0 = unlimited).
// Counts is per-input-frame (frame-major) for progressive build-up.
func (a *capi) PointsStream(ctx uintptr, paths []string, chunk, overlap int, confPct float64, ptSize float32, budget int) (*Cloud, error) {
	if len(paths) == 0 {
		return nil, fmt.Errorf("no images")
	}
	cs := make([][]byte, len(paths))
	pp := make([]*byte, len(paths))
	for i, s := range paths {
		b := append([]byte(s), 0)
		cs[i] = b
		pp[i] = &b[0]
	}
	var n int32
	counts := make([]int32, len(paths))
	var pXyz, pRgb, pRad uintptr
	rc := a.pointsStream(ctx, uintptr(unsafe.Pointer(&pp[0])), int32(len(paths)),
		int32(chunk), int32(overlap), confPct, ptSize, int32(budget),
		unsafe.Pointer(&n), unsafe.Pointer(&counts[0]),
		unsafe.Pointer(&pXyz), unsafe.Pointer(&pRgb), unsafe.Pointer(&pRad))
	runtime.KeepAlive(cs)
	runtime.KeepAlive(pp)
	if rc != 0 {
		return nil, fmt.Errorf("points_stream: %s", a.lastErr(ctx))
	}
	np := int(n)
	cl := &Cloud{N: np, Counts: counts,
		XYZ: cFloats(pXyz, np*3), RGB: cBytes(pRgb, np*3), Rad: cFloats(pRad, np)}
	a.freeFloats(pXyz)
	a.freeBytes(pRgb)
	a.freeFloats(pRad)
	return cl, nil
}

// GS is a single-image 3D-gaussian set (world frame, OpenCV axes).
type GS struct {
	N       int
	XYZ     []float32 // 3N means
	Scale   []float32 // 3N
	Quat    []float32 // 4N wxyz
	RGB     []float32 // 3N linear [0,1]
	Opacity []float32 // N
	// Input camera (pose is canonical identity): K at the processed resolution.
	Fx, Fy, Cx, Cy float32
	W, H           int
}

// Gaussians runs the GS reconstruction on one image (DA3-GIANT only).
func (a *capi) Gaussians(ctx uintptr, path string) (*GS, error) {
	var n, w, h int32
	var pXyz, pScale, pQuat, pRgb, pOp uintptr
	var intr [9]float32
	rc := a.gaussians(ctx, path, unsafe.Pointer(&n),
		unsafe.Pointer(&pXyz), unsafe.Pointer(&pScale), unsafe.Pointer(&pQuat),
		unsafe.Pointer(&pRgb), unsafe.Pointer(&pOp),
		unsafe.Pointer(&intr[0]), unsafe.Pointer(&w), unsafe.Pointer(&h))
	if rc != 0 {
		return nil, fmt.Errorf("gaussians: %s", a.lastErr(ctx))
	}
	np := int(n)
	g := &GS{N: np,
		XYZ: cFloats(pXyz, np*3), Scale: cFloats(pScale, np*3), Quat: cFloats(pQuat, np*4),
		RGB: cFloats(pRgb, np*3), Opacity: cFloats(pOp, np),
		Fx: intr[0], Fy: intr[4], Cx: intr[2], Cy: intr[5], W: int(w), H: int(h)}
	a.freeFloats(pXyz)
	a.freeFloats(pScale)
	a.freeFloats(pQuat)
	a.freeFloats(pRgb)
	a.freeFloats(pOp)
	return g, nil
}
