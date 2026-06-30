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

// pointsToSplat encodes the first `n` points of a cloud as isotropic dots.
// quat is identity, opacity 1; n<=0 means all points.
func pointsToSplat(c *Cloud, n int, opacity float32) []byte {
	if n <= 0 || n > c.N {
		n = c.N
	}
	buf := make([]byte, n*splatRow)
	quat := [4]float32{1, 0, 0, 0}
	for i := 0; i < n; i++ {
		r := c.Rad[i]
		encodeRecord(buf[i*splatRow:],
			[3]float32{c.XYZ[3*i], c.XYZ[3*i+1], c.XYZ[3*i+2]},
			[3]float32{r, r, r}, quat,
			[3]float32{float32(c.RGB[3*i]) / 255, float32(c.RGB[3*i+1]) / 255, float32(c.RGB[3*i+2]) / 255},
			opacity)
	}
	return buf
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
