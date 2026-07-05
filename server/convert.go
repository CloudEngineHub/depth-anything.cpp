// convert.go — encode engine output into antimatter15 .splat bytes (32 B/record),
// byte-identical to depth-anything.cpp's viewer contract and free-splatter's
// src/splat.h: the OpenCV(y-down,z-fwd)->OpenGL(y-up) flip is applied here.
package main

import (
	"encoding/binary"
	"math"
	"sort"
)

const splatRow = 32

func clamp01(v float32) float32 {
	if v < 0 {
		return 0
	}
	if v > 1 {
		return 1
	}
	return v
}

func u8(v float32) byte {
	if v < 0 {
		v = 0
	}
	if v > 255 {
		v = 255
	}
	return byte(v)
}

// encodeRecord writes one 32-byte splat. pos/scale in world (OpenCV) units;
// quat as (w,x,y,z); rgb and opacity in [0,1].
func encodeRecord(out []byte, pos [3]float32, scale [3]float32, quat [4]float32, rgb [3]float32, opacity float32) {
	p := [3]float32{pos[0], -pos[1], -pos[2]}
	q := [4]float32{-quat[1], quat[0], -quat[3], quat[2]}
	nrm := float32(math.Sqrt(float64(q[0]*q[0]+q[1]*q[1]+q[2]*q[2]+q[3]*q[3]))) + 1e-12
	binary.LittleEndian.PutUint32(out[0:], math.Float32bits(p[0]))
	binary.LittleEndian.PutUint32(out[4:], math.Float32bits(p[1]))
	binary.LittleEndian.PutUint32(out[8:], math.Float32bits(p[2]))
	binary.LittleEndian.PutUint32(out[12:], math.Float32bits(scale[0]))
	binary.LittleEndian.PutUint32(out[16:], math.Float32bits(scale[1]))
	binary.LittleEndian.PutUint32(out[20:], math.Float32bits(scale[2]))
	out[24] = u8(clamp01(rgb[0]) * 255)
	out[25] = u8(clamp01(rgb[1]) * 255)
	out[26] = u8(clamp01(rgb[2]) * 255)
	out[27] = u8(clamp01(opacity) * 255)
	for c := 0; c < 4; c++ {
		out[28+c] = u8(q[c]/nrm*128 + 128)
	}
}

// pointsToSplat encodes the first `n` points of a cloud. If the cloud carries normals
// (surfels requested), each point becomes an oriented flat disk (rotation from the
// normal, scale flattened along it); otherwise it's an isotropic dot. Points with a
// zero normal (sparse neighbourhood) stay isotropic. n<=0 means all points.
func pointsToSplat(c *Cloud, n int, opacity float32) []byte {
	if n <= 0 || n > c.N {
		n = c.N
	}
	surfel := len(c.Normals) == 3*c.N
	buf := make([]byte, n*splatRow)
	ident := [4]float32{1, 0, 0, 0}
	for i := 0; i < n; i++ {
		r := c.Rad[i]
		scale := [3]float32{r, r, r}
		quat := ident
		if surfel {
			nx, ny, nz := c.Normals[3*i], c.Normals[3*i+1], c.Normals[3*i+2]
			if nx != 0 || ny != 0 || nz != 0 {
				quat = quatFromNormal(nx, ny, nz)
				scale = [3]float32{r * surfelTangent, r * surfelTangent, r * surfelThin}
			}
		}
		encodeRecord(buf[i*splatRow:],
			[3]float32{c.XYZ[3*i], c.XYZ[3*i+1], c.XYZ[3*i+2]},
			scale, quat,
			[3]float32{float32(c.RGB[3*i]) / 255, float32(c.RGB[3*i+1]) / 255, float32(c.RGB[3*i+2]) / 255},
			opacity)
	}
	return buf
}

// dedupFirstFrame makes the build-up reveal purely additive. Overlapping frames re-observe
// the same surface, so the frame-major cloud holds many coincident points per patch; as the
// reveal crosses into a new frame it piles redundant points onto already-shown surface,
// re-blending it. This walks points in frame order and keeps a point only if its voxel cell
// (size `cell`, world units) is still empty — the FIRST frame to cover a cell wins, later
// frames' re-observations are dropped. Frame order is preserved (kept points stay a
// frame-major subsequence, so the flythrough + incremental reveal still work) and Counts are
// rebuilt. No-op if cell<=0. Returns the number of points dropped.
func dedupFirstFrame(c *Cloud, cell float64) int {
	if c.N == 0 || cell <= 0 || len(c.Counts) == 0 || len(c.Rad) != c.N {
		return 0
	}
	hasNrm := len(c.Normals) == 3*c.N
	type key struct{ x, y, z int32 }
	seen := make(map[key]struct{}, c.N)
	kf := func(i int) key {
		return key{
			int32(math.Floor(float64(c.XYZ[3*i]) / cell)),
			int32(math.Floor(float64(c.XYZ[3*i+1]) / cell)),
			int32(math.Floor(float64(c.XYZ[3*i+2]) / cell)),
		}
	}
	nx := make([]float32, 0, len(c.XYZ))
	nc := make([]byte, 0, len(c.RGB))
	nr := make([]float32, 0, len(c.Rad))
	var nn []float32
	if hasNrm {
		nn = make([]float32, 0, len(c.Normals))
	}
	newCounts := make([]int32, len(c.Counts))
	off := 0
	for f, cnt := range c.Counts {
		var kept int32
		for j := 0; j < int(cnt) && off < c.N; j++ {
			k := kf(off)
			if _, ok := seen[k]; !ok {
				seen[k] = struct{}{}
				nx = append(nx, c.XYZ[3*off], c.XYZ[3*off+1], c.XYZ[3*off+2])
				nc = append(nc, c.RGB[3*off], c.RGB[3*off+1], c.RGB[3*off+2])
				nr = append(nr, c.Rad[off])
				if hasNrm {
					nn = append(nn, c.Normals[3*off], c.Normals[3*off+1], c.Normals[3*off+2])
				}
				kept++
			}
			off++
		}
		newCounts[f] = kept
	}
	dropped := c.N - len(nr)
	c.XYZ, c.RGB, c.Rad, c.Counts = nx, nc, nr, newCounts
	if hasNrm {
		c.Normals = nn
	}
	c.N = len(nr)
	return dropped
}

// orderWithinFramesByRadius reorders points INSIDE each frame block so the smallest
// radius comes first. Radius is proportional to depth, and confidence anti-correlates
// with depth, so smallest-radius ≈ nearest ≈ highest-confidence: the viewer's growing
// reveal then fades in solid near surface before the noisy far field. Frame boundaries
// (Counts) are preserved; XYZ/RGB/Rad/Normals are permuted together. No-op if the cloud
// has no radii.
func orderWithinFramesByRadius(c *Cloud) {
	if c.N == 0 || len(c.Rad) != c.N || len(c.Counts) == 0 {
		return
	}
	hasNrm := len(c.Normals) == 3*c.N
	perm := make([]int, 0, c.N)
	off := 0
	for _, cnt := range c.Counts {
		n := int(cnt)
		if n < 0 || off+n > c.N {
			n = c.N - off
		}
		blk := make([]int, n)
		for i := 0; i < n; i++ {
			blk[i] = off + i
		}
		sort.SliceStable(blk, func(a, b int) bool { return c.Rad[blk[a]] < c.Rad[blk[b]] })
		perm = append(perm, blk...)
		off += n
	}
	if len(perm) != c.N {
		return // ragged Counts; leave as-is rather than corrupt the cloud
	}
	nx := make([]float32, 3*c.N)
	nc := make([]byte, 3*c.N)
	nr := make([]float32, c.N)
	var nn []float32
	if hasNrm {
		nn = make([]float32, 3*c.N)
	}
	for i, p := range perm {
		nx[3*i], nx[3*i+1], nx[3*i+2] = c.XYZ[3*p], c.XYZ[3*p+1], c.XYZ[3*p+2]
		nc[3*i], nc[3*i+1], nc[3*i+2] = c.RGB[3*p], c.RGB[3*p+1], c.RGB[3*p+2]
		nr[i] = c.Rad[p]
		if hasNrm {
			nn[3*i], nn[3*i+1], nn[3*i+2] = c.Normals[3*p], c.Normals[3*p+1], c.Normals[3*p+2]
		}
	}
	c.XYZ, c.RGB, c.Rad = nx, nc, nr
	if hasNrm {
		c.Normals = nn
	}
}

// thinByDepth returns a copy of c that keeps proportionally MORE near-camera points
// and FEWER far ones — "more detail close, less far away". Each point's radius is
// proportional to its depth (rr_base = 0.5*(d/fx+d/fy)*point_size in stream.cpp), so
// radius is a scale-invariant depth proxy: keep-probability = clamp((rRef/r)^alpha, pmin, 1)
// with rRef a robust near radius (p10 of the cloud). A deterministic error-diffusion
// accumulator over the frame-major point list spreads survivors smoothly and — because
// each point's fate depends only on points at or before it — keeps the build-up prefixes
// monotone (acc_k stays a clean subset). Per-view Counts are rebuilt so the acc_k
// build-up still lines up with frames. alpha<=0 (or empty cloud) returns c unchanged.
func thinByDepth(c *Cloud, alpha float64) *Cloud {
	if alpha <= 0 || c == nil || c.N == 0 || len(c.Rad) != c.N {
		return c
	}
	rRef := float64(pctile(c.Rad, 10))
	if rRef <= 0 { // degenerate (mostly zero radii): nothing sensible to bias on
		return c
	}
	const pmin = 0.02 // never fully erase the far field
	out := &Cloud{
		XYZ:    make([]float32, 0, len(c.XYZ)),
		RGB:    make([]byte, 0, len(c.RGB)),
		Rad:    make([]float32, 0, len(c.Rad)),
		Counts: make([]int32, len(c.Counts)),
		// per-frame poses are independent of point thinning — carry them through
		FramePos: c.FramePos,
		FrameFwd: c.FrameFwd,
	}
	acc, idx := 0.0, 0
	for f := 0; f < len(c.Counts); f++ {
		var kept int32
		for j := 0; j < int(c.Counts[f]) && idx < c.N; j++ {
			r := float64(c.Rad[idx])
			p := 1.0
			if r > rRef {
				p = math.Pow(rRef/r, alpha)
			}
			if p < pmin {
				p = pmin
			}
			acc += p
			if acc >= 1.0 {
				acc -= 1.0
				out.XYZ = append(out.XYZ, c.XYZ[3*idx], c.XYZ[3*idx+1], c.XYZ[3*idx+2])
				out.RGB = append(out.RGB, c.RGB[3*idx], c.RGB[3*idx+1], c.RGB[3*idx+2])
				out.Rad = append(out.Rad, c.Rad[idx])
				kept++
			}
			idx++
		}
		out.Counts[f] = kept
	}
	out.N = len(out.Rad)
	return out
}

func pctile(v []float32, p float64) float32 {
	if len(v) == 0 {
		return 0
	}
	s := append([]float32(nil), v...)
	sort.Slice(s, func(a, b int) bool { return s[a] < s[b] })
	i := int(p/100*float64(len(s)-1) + 0.5)
	if i < 0 {
		i = 0
	}
	if i >= len(s) {
		i = len(s) - 1
	}
	return s[i]
}

// gaussiansToSplat encodes a GS set, importance-sorted (opacity*volume) and capped
// at maxSplats. Single-image gaussians include sky / low-confidence pixels that
// back-project to huge depths; we drop that long spatial tail (per-axis 1..99
// percentile box over reasonably-opaque gaussians) and clamp giant blobs so the
// bulk of the surface frames and renders cleanly.
func gaussiansToSplat(g *GS, maxSplats int) []byte {
	N := g.N
	var xs, ys, zs, ss []float32
	for i := 0; i < N; i++ {
		if g.Opacity[i] < 0.03 {
			continue
		}
		xs = append(xs, g.XYZ[3*i])
		ys = append(ys, g.XYZ[3*i+1])
		zs = append(zs, g.XYZ[3*i+2])
		ss = append(ss, (g.Scale[3*i]+g.Scale[3*i+1]+g.Scale[3*i+2])/3)
	}
	keepAll := len(xs) < 32
	// Robust RADIAL trim: keep gaussians within the p92 radius of the median
	// centroid. This catches scattered floaters (huge back-projected depths) that
	// per-axis percentiles miss, so the bulk surface frames cleanly.
	var cx, cy, cz, rCut float32
	scaleCap := float32(1e30)
	if !keepAll {
		cx, cy, cz = pctile(xs, 50), pctile(ys, 50), pctile(zs, 50)
		rs := make([]float32, len(xs))
		for i := range xs {
			dx, dy, dz := xs[i]-cx, ys[i]-cy, zs[i]-cz
			rs[i] = dx*dx + dy*dy + dz*dz
		}
		rCut = pctile(rs, 92)
		scaleCap = pctile(ss, 98) * 3
	}

	idx := make([]int, 0, N)
	for i := 0; i < N; i++ {
		if g.Opacity[i] < 0.02 {
			continue
		}
		if !keepAll {
			dx, dy, dz := g.XYZ[3*i]-cx, g.XYZ[3*i+1]-cy, g.XYZ[3*i+2]-cz
			if dx*dx+dy*dy+dz*dz > rCut {
				continue
			}
		}
		idx = append(idx, i)
	}
	imp := func(i int) float32 {
		return g.Opacity[i] * g.Scale[3*i] * g.Scale[3*i+1] * g.Scale[3*i+2]
	}
	sort.Slice(idx, func(a, b int) bool { return imp(idx[a]) > imp(idx[b]) })
	n := len(idx)
	if maxSplats > 0 && n > maxSplats {
		n = maxSplats
	}
	clampS := func(v float32) float32 {
		if v > scaleCap {
			return scaleCap
		}
		return v
	}
	buf := make([]byte, n*splatRow)
	for k := 0; k < n; k++ {
		i := idx[k]
		encodeRecord(buf[k*splatRow:],
			[3]float32{g.XYZ[3*i], g.XYZ[3*i+1], g.XYZ[3*i+2]},
			[3]float32{clampS(g.Scale[3*i]), clampS(g.Scale[3*i+1]), clampS(g.Scale[3*i+2])},
			[4]float32{g.Quat[4*i], g.Quat[4*i+1], g.Quat[4*i+2], g.Quat[4*i+3]},
			[3]float32{g.RGB[3*i], g.RGB[3*i+1], g.RGB[3*i+2]},
			g.Opacity[i])
	}
	return buf
}
