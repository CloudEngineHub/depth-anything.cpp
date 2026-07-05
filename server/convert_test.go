package main

import "testing"

// Build a 2-view cloud: view 0 all NEAR (radius 1), view 1 all FAR (radius 8).
func mkCloud(nNear, nFar int) *Cloud {
	c := &Cloud{Counts: []int32{int32(nNear), int32(nFar)}}
	for i := 0; i < nNear+nFar; i++ {
		r := float32(1)
		if i >= nNear {
			r = 8
		}
		c.XYZ = append(c.XYZ, float32(i), 0, 0)
		c.RGB = append(c.RGB, 10, 20, 30)
		c.Rad = append(c.Rad, r)
	}
	c.N = nNear + nFar
	return c
}

func TestThinByDepthNoop(t *testing.T) {
	c := mkCloud(100, 100)
	if got := thinByDepth(c, 0); got != c {
		t.Fatalf("alpha=0 must return the same cloud unchanged")
	}
}

func TestThinByDepthNearHeavier(t *testing.T) {
	c := mkCloud(1000, 1000)
	out := thinByDepth(c, 1.0)
	if out == c {
		t.Fatalf("alpha>0 must return a new (thinned) cloud")
	}
	// invariants: N matches slices and rebuilt Counts sum to N
	if out.N != len(out.Rad) || 3*out.N != len(out.XYZ) || 3*out.N != len(out.RGB) {
		t.Fatalf("slice lengths inconsistent: N=%d rad=%d xyz=%d rgb=%d", out.N, len(out.Rad), len(out.XYZ), len(out.RGB))
	}
	var sum int32
	for _, k := range out.Counts {
		sum += k
	}
	if int(sum) != out.N {
		t.Fatalf("Counts sum %d != N %d", sum, out.N)
	}
	near, far := int(out.Counts[0]), int(out.Counts[1])
	// rRef = p10 radius = 1 (near), so near points keep at p=1, far at (1/8)^1 = 0.125.
	if near <= far {
		t.Fatalf("expected more near kept than far: near=%d far=%d", near, far)
	}
	if near < 950 { // near should be kept almost entirely
		t.Fatalf("near kept too few: %d/1000", near)
	}
	// far keep ratio ~0.125 -> ~125; allow generous band around error-diffusion result
	if far < 90 || far > 170 {
		t.Fatalf("far keep ratio off: far=%d (want ~125)", far)
	}
	t.Logf("alpha=1.0: near %d/1000  far %d/1000  total %d", near, far, out.N)
}

func TestThinByDepthStronger(t *testing.T) {
	c := mkCloud(1000, 1000)
	a1 := thinByDepth(c, 0.5)
	a2 := thinByDepth(c, 1.5)
	// Stronger bias must drop MORE far points (monotone in alpha) while near stays full.
	if int(a2.Counts[1]) >= int(a1.Counts[1]) {
		t.Fatalf("stronger alpha should keep fewer far: a1.far=%d a2.far=%d", a1.Counts[1], a2.Counts[1])
	}
	t.Logf("far kept: alpha0.5=%d  alpha1.5=%d", a1.Counts[1], a2.Counts[1])
}
