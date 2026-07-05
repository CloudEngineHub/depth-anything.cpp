// surfels.go — turn a bare point cloud into oriented surface elements ("surfels").
// Each point becomes a thin disk tangent to the local surface instead of a round
// isotropic blob: we estimate a per-point normal by local PCA (smallest principal
// axis of the neighbourhood covariance) and encode it as the splat's rotation, with
// the scale flattened along the normal. Everything is scene-relative (neighbourhood
// radius derives from the cloud's own median point radius) so it works at any scale.
package main

import "math"

const (
	surfelTangent = 1.35 // in-plane disk radius = tangent * point radius (overlap to close gaps)
	surfelThin    = 0.15 // along-normal thickness = thin * point radius (flat disk)
)

// smallestEigenvector3 returns the unit eigenvector of the smallest eigenvalue of the
// symmetric 3x3 matrix packed as [a00,a01,a02,a11,a12,a22], via cyclic Jacobi rotations.
func smallestEigenvector3(c [6]float64) [3]float64 {
	a := [3][3]float64{{c[0], c[1], c[2]}, {c[1], c[3], c[4]}, {c[2], c[4], c[5]}}
	v := [3][3]float64{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}
	for sweep := 0; sweep < 16; sweep++ {
		// target the largest off-diagonal element
		p, q, off := 0, 1, math.Abs(a[0][1])
		if math.Abs(a[0][2]) > off {
			p, q, off = 0, 2, math.Abs(a[0][2])
		}
		if math.Abs(a[1][2]) > off {
			p, q, off = 1, 2, math.Abs(a[1][2])
		}
		if off < 1e-20 {
			break
		}
		apq := a[p][q]
		tau := (a[q][q] - a[p][p]) / (2 * apq)
		var t float64
		if tau >= 0 {
			t = 1 / (tau + math.Sqrt(1+tau*tau))
		} else {
			t = -1 / (-tau + math.Sqrt(1+tau*tau))
		}
		cs := 1 / math.Sqrt(1+t*t)
		sn := t * cs
		for k := 0; k < 3; k++ { // A = Rᵀ A R  (rows then cols of the (p,q) plane)
			akp, akq := a[k][p], a[k][q]
			a[k][p] = cs*akp - sn*akq
			a[k][q] = sn*akp + cs*akq
		}
		for k := 0; k < 3; k++ {
			apk, aqk := a[p][k], a[q][k]
			a[p][k] = cs*apk - sn*aqk
			a[q][k] = sn*apk + cs*aqk
		}
		for k := 0; k < 3; k++ { // accumulate eigenvectors
			vkp, vkq := v[k][p], v[k][q]
			v[k][p] = cs*vkp - sn*vkq
			v[k][q] = sn*vkp + cs*vkq
		}
	}
	m := 0
	if a[1][1] < a[0][0] {
		m = 1
	}
	if a[2][2] < a[m][m] {
		m = 2
	}
	return [3]float64{v[0][m], v[1][m], v[2][m]}
}

// estimateNormals fills c.Normals (3N) with per-point unit surface normals. Points
// whose neighbourhood is too sparse for a stable plane keep a zero normal (rendered
// isotropically). Sign is left arbitrary — a surfel disk is symmetric. A hash grid at
// cell == neighbourhood radius means each point only scans its 27 adjacent cells.
func estimateNormals(c *Cloud) {
	n := c.N
	if n < 8 || len(c.XYZ) < 3*n || len(c.Rad) != n {
		return
	}
	rmed := float64(pctile(c.Rad, 50))
	if rmed <= 0 {
		return
	}
	radius := 3.0 * rmed
	r2 := radius * radius
	cell := radius
	type key struct{ x, y, z int32 }
	kf := func(i int) key {
		return key{
			int32(math.Floor(float64(c.XYZ[3*i]) / cell)),
			int32(math.Floor(float64(c.XYZ[3*i+1]) / cell)),
			int32(math.Floor(float64(c.XYZ[3*i+2]) / cell)),
		}
	}
	grid := make(map[key][]int32, n)
	for i := 0; i < n; i++ {
		k := kf(i)
		grid[k] = append(grid[k], int32(i))
	}
	c.Normals = make([]float32, 3*n)
	const maxNb = 48
	for i := 0; i < n; i++ {
		px, py, pz := float64(c.XYZ[3*i]), float64(c.XYZ[3*i+1]), float64(c.XYZ[3*i+2])
		b := kf(i)
		var sx, sy, sz, mxx, mxy, mxz, myy, myz, mzz float64
		cnt := 0
		for dx := int32(-1); dx <= 1 && cnt < maxNb; dx++ {
			for dy := int32(-1); dy <= 1 && cnt < maxNb; dy++ {
				for dz := int32(-1); dz <= 1 && cnt < maxNb; dz++ {
					for _, j := range grid[key{b.x + dx, b.y + dy, b.z + dz}] {
						qx, qy, qz := float64(c.XYZ[3*j]), float64(c.XYZ[3*j+1]), float64(c.XYZ[3*j+2])
						ex, ey, ez := qx-px, qy-py, qz-pz
						if ex*ex+ey*ey+ez*ez > r2 {
							continue
						}
						sx += qx
						sy += qy
						sz += qz
						mxx += qx * qx
						mxy += qx * qy
						mxz += qx * qz
						myy += qy * qy
						myz += qy * qz
						mzz += qz * qz
						cnt++
						if cnt >= maxNb {
							break
						}
					}
				}
			}
		}
		if cnt < 6 {
			continue // leave normal zero => isotropic point
		}
		inv := 1 / float64(cnt)
		mx, my, mz := sx*inv, sy*inv, sz*inv
		cov := [6]float64{
			mxx*inv - mx*mx, mxy*inv - mx*my, mxz*inv - mx*mz,
			myy*inv - my*my, myz*inv - my*mz, mzz*inv - mz*mz,
		}
		nrm := smallestEigenvector3(cov)
		l := math.Sqrt(nrm[0]*nrm[0] + nrm[1]*nrm[1] + nrm[2]*nrm[2])
		if l < 1e-9 {
			continue
		}
		c.Normals[3*i] = float32(nrm[0] / l)
		c.Normals[3*i+1] = float32(nrm[1] / l)
		c.Normals[3*i+2] = float32(nrm[2] / l)
	}
}

// quatFromNormal builds a rotation whose 3rd (local z) axis is the given unit normal,
// returned as a (w,x,y,z) quaternion in the same OpenCV world space as the cloud. The
// splat renderer flattens along local z (scale.z), so this makes a disk in the tangent
// plane. The in-plane axes are arbitrary (the disk is rotationally symmetric).
func quatFromNormal(nx, ny, nz float32) [4]float32 {
	// helper axis least aligned with the normal, for a stable tangent
	ax, ay, az := float32(1), float32(0), float32(0)
	if math.Abs(float64(nx)) > 0.9 {
		ax, ay, az = 0, 1, 0
	}
	// t1 = normalize(ax × n)
	t1x, t1y, t1z := ay*nz-az*ny, az*nx-ax*nz, ax*ny-ay*nx
	l := float32(math.Sqrt(float64(t1x*t1x + t1y*t1y + t1z*t1z)))
	if l < 1e-9 {
		return [4]float32{1, 0, 0, 0}
	}
	t1x, t1y, t1z = t1x/l, t1y/l, t1z/l
	// t2 = n × t1
	t2x, t2y, t2z := ny*t1z-nz*t1y, nz*t1x-nx*t1z, nx*t1y-ny*t1x
	// rotation matrix columns = [t1, t2, n]; m[i][j] = column j, component i
	m00, m01, m02 := t1x, t2x, nx
	m10, m11, m12 := t1y, t2y, ny
	m20, m21, m22 := t1z, t2z, nz
	tr := float64(m00 + m11 + m22)
	var w, x, y, z float64
	switch {
	case tr > 0:
		s := math.Sqrt(tr+1) * 2
		w = 0.25 * s
		x = float64(m21-m12) / s
		y = float64(m02-m20) / s
		z = float64(m10-m01) / s
	case m00 > m11 && m00 > m22:
		s := math.Sqrt(float64(1+m00-m11-m22)) * 2
		w = float64(m21-m12) / s
		x = 0.25 * s
		y = float64(m01+m10) / s
		z = float64(m02+m20) / s
	case m11 > m22:
		s := math.Sqrt(float64(1+m11-m00-m22)) * 2
		w = float64(m02-m20) / s
		x = float64(m01+m10) / s
		y = 0.25 * s
		z = float64(m12+m21) / s
	default:
		s := math.Sqrt(float64(1+m22-m00-m11)) * 2
		w = float64(m10-m01) / s
		x = float64(m02+m20) / s
		y = float64(m12+m21) / s
		z = 0.25 * s
	}
	return [4]float32{float32(w), float32(x), float32(y), float32(z)}
}
