// voxels.go — snap a point cloud onto a regular grid and emit one cube per occupied
// cell (a "Minecraft" voxelisation). Cubes reuse the .splat record layout: position =
// grid-aligned cell centre, the three scale fields = cube half-extent, colour = the
// cell's average colour. The viewer's cube program reads exactly these fields and draws
// an opaque, face-lit box, so no new on-disk format is needed.
package main

import (
	"math"
	"sort"
)

const voxelFill = 0.97 // cube half-extent = fill * (cell/2); <1 leaves a thin seam between blocks

// cloudBBoxDiag returns the world-space bounding-box diagonal of the first n points.
func cloudBBoxDiag(c *Cloud, n int) float64 {
	if n <= 0 || n > c.N {
		n = c.N
	}
	if n == 0 {
		return 0
	}
	lo := [3]float64{math.Inf(1), math.Inf(1), math.Inf(1)}
	hi := [3]float64{math.Inf(-1), math.Inf(-1), math.Inf(-1)}
	for i := 0; i < n; i++ {
		for k := 0; k < 3; k++ {
			v := float64(c.XYZ[3*i+k])
			if v < lo[k] {
				lo[k] = v
			}
			if v > hi[k] {
				hi[k] = v
			}
		}
	}
	return math.Sqrt((hi[0]-lo[0])*(hi[0]-lo[0]) + (hi[1]-lo[1])*(hi[1]-lo[1]) + (hi[2]-lo[2])*(hi[2]-lo[2]))
}

// cloudToCubesOrdered voxelises the WHOLE cloud at the given cell size and returns the
// cube records ordered by the earliest input frame that touched each cell, plus that
// per-cube first-frame index. Ordering by first frame lets the viewer reveal cubes as
// the build-up (and flythrough) advances through frames, instead of reloading. cell<=0
// yields nil.
func cloudToCubesOrdered(c *Cloud, cell float64) (blob []byte, firstFrame []int) {
	if cell <= 0 || c.N == 0 {
		return nil, nil
	}
	// point -> input-frame index, from the per-frame Counts
	pf := make([]int32, c.N)
	off := 0
	for f, cnt := range c.Counts {
		for i := 0; i < int(cnt) && off < c.N; i++ {
			pf[off] = int32(f)
			off++
		}
	}
	for ; off < c.N; off++ {
		pf[off] = int32(len(c.Counts))
	}
	type key struct{ x, y, z int32 }
	type acc struct {
		r, g, b float64
		cnt     int
		first   int32
	}
	cells := make(map[key]*acc, c.N/4+1)
	kf := func(i int) key {
		return key{
			int32(math.Floor(float64(c.XYZ[3*i]) / cell)),
			int32(math.Floor(float64(c.XYZ[3*i+1]) / cell)),
			int32(math.Floor(float64(c.XYZ[3*i+2]) / cell)),
		}
	}
	for i := 0; i < c.N; i++ {
		k := kf(i)
		a := cells[k]
		if a == nil {
			a = &acc{first: pf[i]}
			cells[k] = a
		} else if pf[i] < a.first {
			a.first = pf[i]
		}
		a.r += float64(c.RGB[3*i])
		a.g += float64(c.RGB[3*i+1])
		a.b += float64(c.RGB[3*i+2])
		a.cnt++
	}
	keys := make([]key, 0, len(cells))
	for k := range cells {
		keys = append(keys, k)
	}
	sort.Slice(keys, func(a, b int) bool {
		ca, cb := cells[keys[a]], cells[keys[b]]
		if ca.first != cb.first {
			return ca.first < cb.first
		}
		// deterministic tie-break by grid coordinate
		if keys[a].x != keys[b].x {
			return keys[a].x < keys[b].x
		}
		if keys[a].y != keys[b].y {
			return keys[a].y < keys[b].y
		}
		return keys[a].z < keys[b].z
	})
	half := float32(0.5 * cell * voxelFill)
	scale := [3]float32{half, half, half}
	ident := [4]float32{1, 0, 0, 0}
	blob = make([]byte, len(keys)*splatRow)
	firstFrame = make([]int, len(keys))
	for i, k := range keys {
		a := cells[k]
		inv := 1 / float64(a.cnt)
		center := [3]float32{
			float32((float64(k.x) + 0.5) * cell),
			float32((float64(k.y) + 0.5) * cell),
			float32((float64(k.z) + 0.5) * cell),
		}
		rgb := [3]float32{float32(a.r * inv / 255), float32(a.g * inv / 255), float32(a.b * inv / 255)}
		encodeRecord(blob[i*splatRow:], center, scale, ident, rgb, 1)
		firstFrame[i] = int(a.first)
	}
	return blob, firstFrame
}

// countLess returns how many of the (ascending-sorted) values are < k — i.e. the number
// of cubes revealed by build-up step k.
func countLess(sorted []int, k int) int {
	lo, hi := 0, len(sorted)
	for lo < hi {
		mid := (lo + hi) / 2
		if sorted[mid] < k {
			lo = mid + 1
		} else {
			hi = mid
		}
	}
	return lo
}

// cloudToCubes voxelises the first n points at the given cell size and returns the
// cube records as .splat bytes. cell<=0 yields no cubes.
func cloudToCubes(c *Cloud, n int, cell float64) []byte {
	if n <= 0 || n > c.N {
		n = c.N
	}
	if cell <= 0 || n == 0 {
		return nil
	}
	type key struct{ x, y, z int32 }
	type acc struct {
		r, g, b float64
		cnt     int
	}
	cells := make(map[key]*acc, n/4+1)
	kf := func(i int) key {
		return key{
			int32(math.Floor(float64(c.XYZ[3*i]) / cell)),
			int32(math.Floor(float64(c.XYZ[3*i+1]) / cell)),
			int32(math.Floor(float64(c.XYZ[3*i+2]) / cell)),
		}
	}
	for i := 0; i < n; i++ {
		k := kf(i)
		a := cells[k]
		if a == nil {
			a = &acc{}
			cells[k] = a
		}
		a.r += float64(c.RGB[3*i])
		a.g += float64(c.RGB[3*i+1])
		a.b += float64(c.RGB[3*i+2])
		a.cnt++
	}
	half := float32(0.5 * cell * voxelFill)
	scale := [3]float32{half, half, half}
	ident := [4]float32{1, 0, 0, 0}
	buf := make([]byte, len(cells)*splatRow)
	i := 0
	for k, a := range cells {
		inv := 1 / float64(a.cnt)
		center := [3]float32{
			float32((float64(k.x) + 0.5) * cell),
			float32((float64(k.y) + 0.5) * cell),
			float32((float64(k.z) + 0.5) * cell),
		}
		rgb := [3]float32{float32(a.r * inv / 255), float32(a.g * inv / 255), float32(a.b * inv / 255)}
		encodeRecord(buf[i*splatRow:], center, scale, ident, rgb, 1)
		i++
	}
	return buf
}
