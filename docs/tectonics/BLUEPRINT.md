# Tectonica: Physics-Based Procedural Planet Generation

**Technical Blueprint — From Physical Foundations to Numerical Implementation**

> This document describes the complete generation pipeline as implemented in
> `rust/planet_engine/src/lib.rs` (compiled to WASM). Every formula, parameter,
> and approximation is stated explicitly with its source. Where physics is
> simplified or a heuristic is used, it is flagged.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Plate Tectonics](#2-plate-tectonics)
3. [Crustal Structure and Isostasy](#3-crustal-structure-and-isostasy)
4. [Surface Evolution: Erosion](#4-surface-evolution-erosion)
5. [Climate](#5-climate)
6. [Hydrology](#6-hydrology)
7. [Biome Classification](#7-biome-classification)
8. [Settlement Suitability](#8-settlement-suitability)
9. [Island-as-Crop](#9-island-as-crop)
10. [Complete Parameter Table](#10-complete-parameter-table)
11. [Pipeline Execution Order](#11-pipeline-execution-order)
12. [Known Limitations and Future Work](#12-known-limitations-and-future-work)
13. [References](#13-references)

---

## 1. System Overview

### 1.1 What This System Does

Tectonica generates procedural Earth-like planets. Given a seed number and
physical parameters (radius, gravity, atmosphere, ocean fraction), it produces:

- A tectonic plate field with classified boundaries
- A physically-motivated relief (elevation) map
- Temperature and precipitation fields
- River networks and lakes
- Biome classification
- Settlement suitability

All output is deterministic: the same seed and inputs always produce the same
planet.

### 1.2 Design Philosophy

**One pipeline, two scales.** There is a single generation pipeline. The
"planet" scope runs it on a 4096×2048 spherical grid (~10 km/cell). The
"island" scope first generates the full planet, then crops an interesting region
and refines it on a 1024×512 flat grid (~0.4 km/cell).

**Physics-first, not art-directed.** Relief emerges from isostasy and stream
power erosion, not from hand-tuned noise blending. Climate follows from
latitude, altitude, and atmospheric circulation, not from painting. Biomes
follow from temperature and precipitation via Whittaker classification.

### 1.3 Grid Abstraction

All functions operate on a generic `GridConfig`:

| Field | Planet | Island |
|-------|--------|--------|
| `width` | 4096 | 1024 |
| `height` | 2048 | 512 |
| `is_spherical` | true | false |
| `km_per_cell_x` | ~9.8 | ~0.39 |
| `km_per_cell_y` | ~9.8 | ~0.39 |

**Planet cell size derivation:**
```
km_per_cell_y = π × R_earth / height = π × 6371 / 2048 ≈ 9.77 km
km_per_cell_x = 2π × R_earth / width  = 2π × 6371 / 4096 ≈ 9.77 km (at equator)
```

**Neighbor lookup.** On the spherical grid, x wraps (left edge → right edge);
y clamps (no polar wraparound). On the flat grid, both axes clamp. 8
neighbors (N, NE, E, SE, S, SW, W, NW) are supported.

### 1.4 Coordinate Systems

Each cell `i` has pre-computed:
- **Latitude** `lat_deg[i]` in [−90, +90]
- **Longitude** `lon_deg[i]` in [−180, +180]
- **Unit sphere** `(sx, sy, sz)` for 3D noise sampling
- For island crops: lat/lon are real planetary coordinates of the source region

---

## 2. Plate Tectonics

### 2.1 Physical Background

Earth's lithosphere is divided into ~15 major tectonic plates that move relative
to each other. Plate motion is described by **Euler's theorem**: any rigid-body
rotation on a sphere can be expressed as rotation about a fixed pole (the Euler
pole) with angular velocity ω.

> **Cox & Hart (1986)** *Plate Tectonics: How It Works*

The velocity of a point **r** on a plate with rotation vector **Ω** is:

```
v(r) = Ω × r                                                           (2.1)
```

where **Ω** = (ω_x, ω_y, ω_z) is in radians/year and **r** is on the unit
sphere scaled by radius R (in cm for cm/yr velocities).

### 2.2 Plate Field Generation

**Input:** `plate_count` ∈ [2, 20], `plate_speed_cm_per_year`, `mantle_heat`.

**Step 1: Plate specification.** For each plate k:
- Random position (lat_k, lon_k) on the sphere
- Random pole direction (uniform on sphere via acos(2u−1) for latitude)
- Speed: `speed_k = plate_speed × uniform(0.5, 1.5)` cm/yr
- Angular velocity magnitude: `|Ω_k| = speed_k / R_cm × uniform(0.82, 1.24)`
- Buoyancy: `uniform(−1, +1)` (>0 → continental, <0 → oceanic)
- Heat: `0.5 + mantle_heat × uniform(0, 1) × 2.5`

**Step 2: Irregular plate field growth** (`build_irregular_plate_field`).

This is a priority-queue (Dijkstra-like) flood fill from plate nuclei. Each
plate has:
- 1 primary nucleus at its center
- 2–7 secondary nuclei at random offsets
- 6–18 historical path nuclei along a curved trajectory

The growth cost from cell `a` to neighbor `b` is:

```
cost(a→b) = w × spread × f_rough × f_drift × f_structure × f_polar      (2.2)

where:
  w           = 1.0 (orthogonal) or 1.414 (diagonal)
  spread      = random(0.82, 1.22) / size_bias^0.36
  f_rough     = 1 + roughness × (0.22 sin(ξ_a) + 0.18 cos(ξ_b))
  f_drift     = 1.03 − 0.12 × dot(direction_to_b, plate_drift)
  f_structure = clamp(0.62 + 0.98 × noise + 0.36 − roughness × 0.17, 0.45, 1.9)
  f_polar     = 1 + |lat|/90 × 0.1
```

**⚠ Approximation:** This is a geometric heuristic, not geophysics. Real plate
shapes emerge from rheology, driving forces, and ~200 Myr of history. The
priority-queue growth produces irregular, non-Voronoi shapes that *look*
plausible, but the cost function parameters (0.22, 0.18, 0.12, 0.62, etc.) are
tuned by eye, not derived from physics.

**Step 3: Fragment cleanup.** Plate regions smaller than
`clamp(WORLD_SIZE / (plate_count × 420), 240, 1800)` cells are merged into
their largest neighbor.

### 2.3 Plate Evolution

The plate field evolves over geological time via `evolve_plate_field`. Each
step:

1. Time span: `random(0.9, 3.4) Myr × (0.92 + 0.32 × age_norm)`
2. For each cell: compute plate velocity → displacement in cells
3. Bilinear vote interpolation (which plate "arrives" at each cell)
4. Relaxation: 3×3 majority filter
5. Fragment cleanup

Number of steps = `detail.plate_evolution_steps` (2–10 depending on preset).

**Velocity at a cell** (`plate_velocity_xy_from_omega`):

Given unit-sphere position (sx, sy, sz) and plate rotation vector Ω:

```
v_3D = Ω × r̂ × R_cm                                                    (2.3)

Transform to east-north:
  cos_lat = sqrt(sx² + sy²)
  v_east  = v_3D · ê_east   where ê_east = (−sy, sx, 0) / cos_lat
  v_north = v_3D · ê_north  where ê_north = (−sz·sx, −sz·sy, cos_lat) / cos_lat
```

### 2.4 Boundary Classification

At each cell, the relative velocity between adjacent plates is decomposed into
normal (convergence/divergence) and tangential (shear) components.

**Classification rules** (implemented at lines 1638–1649):

| Type | Condition |
|------|-----------|
| Convergent (1) | conv > 0.14 × scale AND conv ≥ 0.95 × div AND conv ≥ 0.75 × shear |
| Divergent (2) | div > 0.14 × scale AND div ≥ 0.95 × conv AND div ≥ 0.75 × shear |
| Transform (3) | shear > 0.18 × scale |
| Interior (0) | otherwise |

where `scale = max(1.2, plate_speed × 1.25)`.

**Boundary strength** `bstr ∈ [0, 1]`: the magnitude of the dominant velocity
component, normalized against the maximum observed.

---

## 3. Crustal Structure and Isostasy

### 3.1 Physical Background

The elevation of a point on Earth's surface is primarily determined by the
thickness and density of the crust beneath it. Thick, light continental crust
"floats" high on the denser mantle (mountains). Thin, dense oceanic crust sits
low (ocean floor). This is **Airy isostasy**.

> **Turcotte & Schubert (2002)** *Geodynamics*, Chapter 3

### 3.2 Deformation Propagation (England & McKenzie 1982)

In real orogens, deformation is **distributed** over hundreds of kilometres,
not confined to the plate boundary itself. The boundary detection yields
`boundary_strength` only on cells touching a different plate (~4 cells wide =
80 km). To create geologically realistic wide mountain belts, we propagate
the boundary signal outward with **exponential distance decay**.

> **England & McKenzie (1982)** *A thin viscous sheet model for continental
> deformation*, EPSL. Characteristic deformation length L_d ≈ 200–500 km.

**Method:** Max-propagation (iterative morphological dilation with exponential
decay). Each cell takes `max(self, max_neighbor × exp(-d/L_d))` where d is the
cardinal/diagonal distance and L_d is the characteristic deformation length.
Peak amplitude is preserved (unlike diffusive smoothing).

Three separate fields are propagated (one per boundary type):

| Boundary type | L_d (km) | Physical reference |
|---|---|---|
| Convergent | 200 | England & McKenzie 1982: 200–500 km; 200 km prevents small plates from being entirely covered by deformation |
| Divergent | 150 | Mid-ocean ridge thinning ~200-300 km |
| Transform | 100 | Narrow shear zones ~100-150 km |

After propagation, **8 passes** of diffusive smoothing eliminate the 120°/60°
angular wedges that max-dilation inherits from Voronoi plate boundaries
(σ ≈ 90 km at ~10 km/cell).

**Interior suppression (thermal age proxy — Artemieva & Mooney 2001):**

Continental lithosphere far from active boundaries is old, cold, and rigid —
it resists deformation.  We compute a "lithospheric weakness" field via BFS
from all plate boundaries:

```
weakness(d) = exp(−d / L_rheol)                                       (3.2a)
L_rheol = 300 km   (rheological decay length, Artemieva & Mooney 2001)
```

| Distance | Weakness | Tectonic interpretation |
|----------|----------|------------------------|
| 0 km (boundary) | 1.00 | Active orogen |
| 300 km | 0.37 | Mobile belt |
| 600 km | 0.14 | Shield margin |
| 900 km | 0.05 | Deep craton |

After computing weakness, convergent deformation is multiplied by the weakness
field for continental cells (cf > 0.1): `conv_def *= weakness`.  This creates
flat continental lowlands (Canadian Shield, Russian Platform) instead of
uniform mild plateaus from residual deformation tails.

### 3.3 Crustal Thickness from Deformation Fields

For each cell, crustal thickness `C` is computed from the **propagated**
deformation fields:

```
C_base = 43 km + noise_var  (continental)                               (3.1)
       =  7 km              (oceanic; White et al. 1992)

noise_var = N(3f, seed) × 10 + N(6f, seed+1) × 5   [km]               (3.1a)
            range: ±15 km  (wavelengths ~1300–2600 km on unit sphere)

ΔC = + conv_def × 22 km    (continental convergent, CC collision)
     + conv_def × 13 km    (oceanic convergent, subduction arc)
     − div_def  × 20 km    (divergent, rifting)
     + trans_def × 2 km    (transform, transpression)

C = clamp(C_base + ΔC, 5, 75) km
```

**Physical basis:**
- Continental base 43 km: cratonic/shield average (Rudnick & Gao 2003, Table 2).
  With ±15 km noise → range 28–58 km, matching the observed global distribution
  (25–55 km; thin mobile belts to thick shields).
- Peak thickness: 43 + 22 = 65 km ≈ Tibet (Dewey & Burke, 1973)
- Mid-ocean ridges thin to ~5 km
- Subduction arcs: ~25 km (Christensen & Mooney, 1995)

**Continental assignment (area-aware):**

Plates are sorted by buoyancy (descending) and greedily assigned to the
continental category until total area reaches a target fraction:

```
target_continental = min(land_frac + 0.20, 0.85) × total_cells          (3.1b)
```

where `land_frac = 1 − ocean_percent / 100`. The +0.20 buffer creates
submerged continental shelf: on Earth, ~40% of the surface is continental crust
but only ~29% is exposed land (Cogley 1984; Taylor & McLennan 1995).

**Coastline perturbation:** After smoothing the continental fraction field
(20 passes, σ ≈ 80 km), multi-octave noise breaks Voronoi-straight coastlines:

```
n = N(4f) × 0.10 + N(8f) × 0.04 + N(16f) × 0.01                      (3.1c)
cf_perturbed = clamp(cf + n × margin_factor, 0, 1)
margin_factor = 1 − |2 × cf − 1|    [tent: peak at cf = 0.5, zero at 0/1]
```

Followed by 3 smoothing passes. Only applies to spherical (planet) grids.

### 3.4 Flexural Smoothing

After computing crustal thickness from the already-wide deformation fields,
flexural smoothing approximates lithospheric rigidity.

> **Watts (2001)** *Isostasy and Flexure of the Lithosphere*

**Implementation:** **12 passes** of 4-neighbor diffusion averaging.

```
C_new = 0.5 × C_old + 0.5 × C_neighbors_mean                          (3.2)
```

12 passes at ~10 km/cell → σ ≈ 70 km. Combined with the 8-pass deformation
smoothing (§3.2) and the deformation propagation, the total smoothing creates
geologically realistic wide mountain belts with gradual transitions.

**cos(φ) cap in iterative smoothing:** On the equirectangular grid, the E-W
physical distance per cell is `Δx × cos(φ)`. At high latitudes, smoothing
weights the E-W direction by `1/cos(φ)`, which compounds across multiple
passes. To prevent horizontal strip artifacts at poles, iterative smoothing
(but NOT single-pass propagation) caps `cos(φ) ≥ 0.30` (≈ 72.5°), limiting
the E-W/N-S ratio to 3.3:1 instead of 11.5:1 at 85°.

**⚠ Approximation:** True flexure solves a 4th-order PDE (D∇⁴w = q). The
diffusion averaging is a low-pass filter that produces similar spatial
smoothing but does not capture foreland basins or moat-and-bulge geometry.

### 3.5 Rock Type Classification

Each cell is assigned a `RockType` based on its tectonic setting:

| Setting | Rock Type | K_eff (m^0.5/yr) |
|---------|-----------|-------------------|
| Oceanic, no subduction | Basalt | 1.0 × 10⁻⁶ |
| Oceanic, conv_def > 0.3 | Schist | 1.2 × 10⁻⁶ |
| Continental, conv_def > 0.5 | Granite | 0.5 × 10⁻⁶ |
| Continental, conv_def > 0.15 | Quartzite | 0.8 × 10⁻⁶ |
| Continental, div_def > 0.3 | Sandstone | 2.0 × 10⁻⁶ |
| Continental, trans_def > 0.2 | Schist | 1.2 × 10⁻⁶ |
| Continental, interior | Noise → Limestone/Sandstone/Granite | 0.5–3.0 × 10⁻⁶ |

> **Harel et al. (2016)** reported erodibility varies by ~2 orders of magnitude
> across lithologies. Our 6× range (0.5–3.0 × 10⁻⁶) is conservative.

**K_eff smoothing:** 3 passes of neighbor averaging. Fault zones (boundary
cells) get K_eff × 1.5 (Hovius & Stark 2006: fracturing increases
erodibility).

### 3.6 Airy Isostatic Elevation

Given crustal thickness C, the elevation relative to sea level is:

**Airy column buoyancy (Turcotte & Schubert 2002, eq 2.4):**
```
h = C × 1000 × (ρ_m − ρ_c_eff) / ρ_m     [meters]                    (3.3)
```

This gives absolute freeboard above the compensation depth. Sea level is
determined separately (§4.7) as the `ocean_percent` percentile of all
elevations, which acts as the C_ref subtraction.

**Densities:**

| Symbol | Value | Source |
|--------|-------|--------|
| ρ_c (continental) | 2800 kg/m³ | Turcotte & Schubert (2002) |
| ρ_c (oceanic) | 2900 kg/m³ | Turcotte & Schubert (2002) |
| ρ_m (mantle) | 3300 kg/m³ | Turcotte & Schubert (2002) |
| ρ_w (water) | 1030 kg/m³ | Standard |

**Note on water loading:** Earth's ocean basins are deepened by the weight
of the water column (yielding eq 3.4 in earlier versions of this document).
The implementation uses the simpler eq 3.3 for all cells. The missing water
loading, thermal subsidence (Parsons & Sclater 1977), and dynamic topography
are compensated by the hypsometric correction (§4.5).

**Thermal correction:**
```
ρ_c_eff = ρ_c × (1 − heat_anomaly × 0.02)                             (3.5)
```
Hot mantle reduces crustal density by ~2% per unit heat anomaly, causing
thermal uplift. This is a linearized approximation of thermal expansion
(α ≈ 3 × 10⁻⁵ K⁻¹, ΔT ≈ 200 K → Δρ/ρ ≈ 0.6%). The 2% coefficient is
~3× larger than thermal expansion alone, implicitly including dynamic
topography from mantle convection.

**Noise overlay** (topographic roughness not captured by isostasy):
```
h += 60 × noise(3f) + 35 × noise(6f) + 18 × noise(12f) + 7 × noise(24f)  (3.6)
```
where `f` is the base frequency and noise is spherical FBM. This adds ~120 m
of structure at scales from ~700 km down to ~80 km.

**⚠ Approximation:** Equations (3.3–3.4) are Airy (local) isostasy, not
flexural. This means each cell's elevation depends only on the crust directly
beneath it, ignoring the rigidity of the lithosphere. The flexural smoothing in
§3.3 partially compensates but cannot reproduce foreland basins or flexural
bulges.

---

## 4. Surface Evolution: Erosion

### 4.1 Physical Background

Once the initial isostatic relief is established, it is modified by fluvial
erosion and hillslope diffusion over geological time. The governing equation is:

```
∂h/∂t = U(x) − K_eff × A^m × S^n − κ∇²h                              (4.1)
```

| Term | Physical meaning |
|------|-----------------|
| U(x) | Tectonic uplift rate (m/yr) |
| K_eff × A^m × S^n | Fluvial erosion (stream power law) |
| κ∇²h | Hillslope diffusion (soil creep) |

> **Howard (1994)**, **Whipple & Tucker (1999)**: The stream power law.
> **Braun & Willett (2013)**: O(n) implicit solver.

### 4.2 Stream Power Law

The fluvial erosion rate at a point is proportional to the power dissipated by
flowing water at the bed:

```
E = K_eff × A^m × S^n                                                   (4.2)
```

| Parameter | Value | Source |
|-----------|-------|--------|
| m (area exponent) | 0.5 | Whipple & Tucker (1999) |
| n (slope exponent) | 1.0 | Whipple & Tucker (1999) |
| K_eff (erodibility) | 0.5–3.0 × 10⁻⁶ m^0.5/yr | Harel et al. (2016), §3.4 |

A is drainage area (km²), S is local slope (dimensionless).

These m, n values correspond to the "detachment-limited" end-member where
erosion is controlled by the river's ability to detach bedrock, not by its
capacity to transport sediment.

### 4.3 Braun-Willett O(n) Implicit Solver

The implicit Euler discretization of Eq. (4.1) for a single time step Δt:

```
h_i^{new} = h_i^{old} + U_i × Δt + factor_i × h_{recv(i)}^{new}       (4.3)
            ─────────────────────────────────────────────────────
                           1 + factor_i

where factor_i = K_eff_i × Δt × A_i^m / Δx^n
```

`recv(i)` is the downstream receiver of cell i (D8 steepest descent).

**Algorithm:**

1. **D8 flow routing.** Each cell drains to its steepest-descent neighbor
   among 8 directions. Cells with no lower neighbor are local sinks.

2. **Topological sort.** Cells ordered from highest to lowest elevation.

3. **Drainage area** — scale-dependent routing:

   - **Planet scale (Δx ≈ 10 km): MFD (Multiple Flow Direction).**
     Each cell distributes its area to all downslope neighbours, weighted
     by slope^p (Freeman 1991):
     ```
     f_i = max(0, S_i)^p / Σ_j max(0, S_j)^p                         (4.3a)
     S_i = (h_center − h_neighbor) / distance_i
     p = 1.5
     ```
     This produces a smooth area field without single-pixel channel
     artefacts. At 10 km/cell, individual river channels are sub-grid
     (typical river width ~100 m = 1% of one cell). D8 routing at this
     scale concentrates all flow into 1-pixel-wide channels with vertical
     walls — a numerical artefact, not physics.

     > **Freeman (1991):** MFD algorithm with p = 1.5 (higher than standard
     > 1.1 to increase flow diffusion and reduce grid-aligned channel artifacts).
     > **Salles et al. (2023), *Science*:** goSPL uses MFD area + D8
     > implicit solver at 10 km global resolution — the same approach.

   - **Island scale (Δx ≈ 0.4–2 km): D8 (Single Flow Direction).**
     At this resolution, individual valleys are resolvable (the
     characteristic length l_c = κ/K ≈ 10 km spans many cells).
     Standard D8 accumulation: `A[recv(i)] += A[i]`, each cell starting
     with `A = Δx²`.

4. **Implicit update.** Traverse low→high (downstream first). For each land
   cell (h > 0):
   ```
   factor = K × Δt × A^m / Δx^n
   h_new = (h_old + U×Δt + factor × h_recv_new) / (1 + factor)
   ```
   This is unconditionally stable for any Δt (implicit Euler).
   The D8 receiver tree is used for the implicit solver at both scales
   (the solver requires a tree structure). Only the area values differ
   between MFD and D8.

5. **Hillslope diffusion** (explicit):
   ```
   Δh = κ × Δt × ∇²h                                                  (4.4)
   ```
   CFL stability requires `κ × Δt / Δx² < 0.25`.

**Boundary conditions:** Ocean cells (h ≤ 0) are held fixed. Coastal cells
drain to the ocean (receiver elevation = 0).

### 4.4 Light Erosion with Noise Injection

Full stream power erosion (multiple epochs × many steps) creates radial channel
artifacts at 10 km/cell resolution because D8/MFD flow routing aligns channels
to the cardinal/diagonal grid directions.  The solution is **light erosion**:
just enough to create terrain asymmetry and regional lowering, not enough for
channel artifacts to dominate.

**Single epoch, 3 steps:**

```
Uplift rates:                                                            (4.5)
  U_convergent = bstr × 0.005 × speed_factor × epoch_scale   m/yr (up to 5 mm/yr)
  U_divergent  = −bstr × 0.002 × speed_factor                m/yr (up to −2 mm/yr)
  U_transform  = bstr × 0.001 × speed_factor                 m/yr

  speed_factor = plate_speed / 5.0
  epoch_scale  = 1.0   (single epoch, no decay)
```

Stream power parameters:
- Δt = 500,000 years
- κ = 0.02 m²/yr (doubled hillslope diffusivity to smooth incipient channels)
- MFD p = 1.5 (more diffuse flow distribution than standard 1.1)
- 3 steps only
- Total simulated time: 3 × 0.5 Myr = **1.5 Myr**

**Noise injection** (after erosion, before isostatic relaxation):

```
perturbation[i] = value_noise3(48f) × 5.0    [meters]                  (4.5a)
h[i] = max(h[i] + perturbation[i], 0.5)
```

The ±5 m random perturbation breaks residual radial channel coherence from
MFD routing without altering macro-scale terrain.  The high frequency (48×
base) ensures the perturbation is spatially uncorrelated at the cell scale.

**Isostatic relaxation** (after noise injection):
```
h = 0.85 × h_eroded + 0.15 × h_isostatic_target                       (4.6)
```
This represents the mantle's viscous response: as erosion removes mass,
isostatic rebound raises the surface. The 85/15 blend corresponds to partial
relaxation over ~2.5 Myr, consistent with a Maxwell time of ~1 Myr for the
upper mantle.

**Post-erosion land smoothing:** 5 passes of 4-neighbor averaging (land cells
only) blend residual channel artifacts into smooth terrain.

**Note:** Convergent uplift of up to 5 mm/yr is consistent with GPS
observations of 0.5–10 mm/yr for active orogens (Bevis et al.).

### 4.5 Hypsometric Curve Correction

The Airy model (eq 3.3) produces ~3 km median continental freeboard because
it ignores water loading, thermal subsidence (Parsons & Sclater 1977),
sediment loading, and dynamic topography. A power-law remapping compresses
land hypsometry to match Earth's observed distribution (Cogley 1984):

1. Compute preliminary sea level (ocean_percent percentile)
2. Find **median** of land elevations (P50) and max land above sea level

```
α = ln(target_median / max_land) / ln(median_fb / max_land)            (4.7)
α = clamp(α, 1.0, 5.0)

For each cell above sea level:
  t = (h − sea_level) / max_land         [normalized: 0..1]
  h_new = sea_level + t^α × max_land                                   (4.8)
```

**Target:** median land elevation = 400 m (Cogley 1984, Harrison et al. 1983).

**Delta smoothing:** To avoid gradient amplification at mountain flanks
(the derivative α·t^(α−1) reaches 2.5× at peaks), the correction is applied
via a smoothed delta field rather than directly:

```
delta[i] = h_original[i] − h_remapped[i]          [large in lowlands, ~0 at peaks]
smooth(delta, 10 passes)                           [blends correction boundaries]
h_final[i] = h_original[i] − delta[i]                                  (4.9)
```

This preserves peak heights (delta ≈ 0) while softening transitions.

### 4.6 ETOPO1-Calibrated Terrain Detail Noise

> **Sayles, R.S. & Thomas, T.R. (1978)** "Surface topography as a non-stationary
> random process". *Nature*, 271, 431-434.
>
> **Huang, J. & Turcotte, D.L. (1989)** "Fractal mapping of digitized images".
> *Computers in the Geosciences*, 15(3), 325-333.

Earth's topographic power spectrum follows P(k) ∝ k^(−β) with β ≈ 2.0 for
continental surfaces (Huang & Turcotte 1989).  This corresponds to fractional
Brownian motion with Hurst exponent H = (β−1)/2 = 0.5.

Four octaves of value noise with amplitude ratio 1/2 per octave reproduce the
spectral structure measured from ETOPO1 land cells:

```
octave 1: noise(16f) × 80   →  λ ≈ 400 km                            (4.9a)
octave 2: noise(32f) × 40   →  λ ≈ 200 km
octave 3: noise(64f) × 20   →  λ ≈ 100 km
octave 4: noise(128f) × 10  →  λ ≈  50 km

elev_factor = sqrt(clamp(h / 5000, 0, 1))
h += Σ octaves × elev_factor
```

The elevation-dependent scaling captures the roughness–relief relationship
(Montgomery & Brandon 2002): mountain flanks (h ≈ 5000 m) get up to ±150 m
of texture; lowlands near sea level get ±20 m (gentle rolling hills).
Absolute amplitudes calibrated against ETOPO1 land RMS roughness at λ = 400 km
(~200 m orogenic, ~30 m lowland).

### 4.7 Sea Level Determination

After the hypsometric correction, sea level is set by `ocean_percent`:

1. Sort all elevation values
2. Sea level = elevation at the `ocean_percent` percentile
3. Subtract sea level from all cells

This guarantees that exactly `ocean_percent` of the surface is below zero
(underwater).

---

## 5. Climate

### 5.1 Physical Background

Earth's climate is governed by:
1. **Solar insolation** — decreases from equator to poles
2. **Atmospheric circulation** — Hadley, Ferrel, and Polar cells
3. **Orographic effects** — mountains force air upward, causing rain
4. **Lapse rate** — temperature decreases with altitude

> **Held & Hou (1980)** Hadley circulation theory.
> **Peixoto & Oort (1992)** Physics of Climate.
> **Holton & Hakim (2013)** Dynamic Meteorology.

### 5.2 Temperature Model

**Sea-level temperature** (quadratic fit to zonal-mean Earth data):
```
T_sea(lat) = 28.0 − 0.007 × lat²     [°C]                             (5.1)
```
This gives 28°C at the equator, ~0°C at 63°, −31°C at 90°.

**Fit quality:** This is a good approximation of the annual zonal mean.
Observations: ~27°C equator, ~−20°C at 80°N (annual mean). The quadratic
slightly overestimates polar cold.

**Environmental lapse rate:**
```
T(lat, h) = T_sea(lat) − Γ × h                                        (5.2)

Γ = 6.0 K/km = 0.006 K/m
```

> Source: ICAO Standard Atmosphere, Holton & Hakim (2013). The environmental
> lapse rate (6.0 K/km) is less than the dry adiabatic rate (9.8 K/km) due to
> latent heat release.

**Modifiers:**
```
T_final = T_sea − Γ×h + T_ocean + T_atm + T_noise − T_aerosol         (5.3)
        − T_continentality

T_ocean          = +2°C if cell is underwater (ocean thermal inertia)
T_atm            = 5 × ln(1 + P_atm)  [°C, greenhouse effect]
T_noise          = ±2°C (spatial noise for local variability)
T_aerosol        = event-driven cooling (meteorite impacts)
T_continentality = 0.008 × d_coast_km × sin(|lat|)   [°C]             (5.3a)
                   (Terjung & Louie 1972; Conrad continentality index)
```

**Final:** clamp to [−70, +55] °C.

**⚠ Approximation:** The greenhouse term `5 × ln(1 + P_atm)` is a rough fit
to Earth (1 bar → +3.5°C). Real greenhouse forcing depends on atmospheric
composition (CO₂, H₂O, CH₄), not just total pressure. For Earth-like planets
this is adequate; for Venus-like or Mars-like it would be wrong.

### 5.3 Atmospheric Circulation and Wind

Wind direction depends on latitude via the three-cell circulation model:

| Latitude band | Wind direction | Name |
|---------------|---------------|------|
| 0°–30° | From east (→ +x) | Trade winds |
| 30°–60° | From west (→ −x) | Westerlies |
| 60°–90° | From east (→ +x) | Polar easterlies |

Transitions are smooth (linear blend over 25°–35° and 55°–65°).

A meridional component `0.35 × sin(lat)` represents Coriolis deflection: zero
at the equator (no deflection), maximum at the poles.

> **Peixoto & Oort (1992)** — zonal wind pattern.

### 5.4 Precipitation Model

Precipitation is built from five components:

**Component 1: Hadley cell zonal base** (Held & Hou 1980)

```
P_hadley(lat):                                                           (5.4)
  |lat| < 10°:   2000 mm/yr   (ITCZ, deep convection)
  10°–20°:        2000 → 600   (subsidence begins)
  20°–35°:        600 → 250    (subtropical high, desert belt)
  35°–50°:        250 → 900    (westerly storm track)
  50°–70°:        900 → 400    (diminishing moisture)
  > 70°:          400 → 150    (polar desert)
```

**Component 2: Windward moisture** (exponential decay from coast)

```
P_windward = 600 × exp(−d_land × λ)    [mm/yr]                          (5.5)

λ = 0.02 × (km_per_cell / 20)    [per cell]
```

`d_land` is the distance (in cells) from the upwind coast, traced in the wind
direction. Moisture decays exponentially as air moves inland.

The trace continues over ocean cells (straits, bays) without breaking,
counting only land cells for distance. Maximum cumulative land distance: 50
cells (~1000 km).

**Component 3: Coastal exposure**

```
P_coastal = f_coast × 600    [mm/yr]                                     (5.6)
```

`f_coast` is the fraction of cells within 3 cells (~60 km) that are ocean.
Measures how "coastal" a location is.

Coastal exposure uses a multi-cell radius kernel counting ocean fraction
(omnidirectional).

**Component 4: Orographic lift** (Roe 2005, Smith 1979)

```
P_orographic = max(0, h_local − h_upwind) × 0.8    [mm/yr]             (5.7)
```

When terrain rises in the wind direction, air is forced upward, cools
adiabatically, and precipitates. The 0.8 factor converts elevation gain
(meters) to additional precipitation (mm/yr).

**Component 5: Rain shadow with distance decay** (Smith 1979)

```
P_shadow = max over upwind cells of:                                    (5.8)
    max(0, h_upwind − h_local) × 0.40 × exp(−d × Δx / 250 km)
```

Behind mountains, descending air warms and dries. A 2000 m range at the foot
creates ~800 mm/yr shadow. Shadow decays exponentially with distance from the
barrier (e-folding distance 250 km). At 500 km: 13% remaining. Maximum trace
distance: 500 km.

**Altitude factor** (Clausius-Clapeyron, cold air holds less moisture):

```
f_altitude = max(0.1, exp(−h × 0.0004))                                 (5.9)
```

At sea level: 1.0. At 1000 m: 0.67. At 3000 m: 0.30. At 5000 m: 0.14.
Floor at 0.1.

The coefficient 0.0004 /m derives from: lapse rate 6 K/km × Clausius-Clapeyron
~7%/K = 42%/km = 0.00042/m, rounded to 0.0004.

**Final precipitation (land cells):**

```
P = (P_hadley × f_cont + P_windward + P_coastal + P_orographic          (5.10)
    − P_shadow) × f_altitude + noise × 60

P_final = clamp(P × f_atmosphere − P_aerosol, 20, 4500) mm/yr

f_cont        = exp(−d_coast / 800 km)    [continentality drying]      (5.10a)
f_atmosphere  = √(P_atm)
P_aerosol     = aerosol × 200 mm/yr       [impact winter]
```

**Continentality drying (eq 5.10a):** inland areas receive less precipitation
as moisture decays exponentially with distance from coast. e-folding distance
800 km: at 800 km inland, 37% of coastal precipitation remains. Central Asia
(~1500 km inland) receives only 200–300 mm/yr despite 40°N latitude.

Ocean cells receive `P_hadley × 1.2 × f_atmosphere` (open-water evaporation
exceeds zonal mean by ~20%, Trenberth et al. 2007).

---

## 6. Hydrology

### 6.1 Physical Background

Water flows downhill. The **D8 algorithm** (O'Callaghan & Mark 1984) routes
each cell's water to its steepest-descent neighbor among 8 directions.
Accumulating flow gives drainage area, which determines river size.

### 6.2 Slope Calculation

For each cell, slope is the maximum elevation drop to any of the 8 neighbors,
normalized by distance:

```
slope = max over 8 neighbors of: (h_center − h_neighbor) / d            (6.1)
```

where d = 1 for orthogonal neighbors, √2 for diagonal.

### 6.3 Flow Routing (D8)

Each cell receives a flow direction pointing to its lowest neighbor:
```
receiver[i] = argmin_{j ∈ neighbors(i)} h[j]                           (6.2)
```

If all neighbors are higher (local pit), `receiver[i] = −1` (sink/lake).

### 6.4 Flow Accumulation

Cells sorted topologically (high → low). Each cell passes its area to its
receiver:

```
A[receiver[i]] += A[i]                                                   (6.3)
```

Cells with large `A` are rivers. The threshold for "visible river":

```
threshold = max(22, size × (0.00018 + fluvial_rounds × 0.00007))       (6.4)
```

### 6.5 River Intensity

```
river[i] = ((A − threshold) / (A_max − threshold))^0.45                (6.5)
           × (0.34 + 0.86 × slope / (slope + 35))
```

Rivers are stronger where drainage area is large and slope is moderate.

### 6.6 Fluvial Valley Carving (Leopold & Maddock 1953)

> **Leopold, L.B. & Maddock, T. (1953)** "The hydraulic geometry of stream
> channels and some physiographic implications". *USGS Professional Paper 252*.
>
> **Schumm, S.A. (1977)** *The Fluvial System*. Wiley.

After initial hydrology, river valleys are carved using hydraulic geometry:

```
Q = flow_accumulation × cell_area [m²] × runoff [m/s]                 (6.6a)
  runoff = 400 mm/yr = 1.27 × 10⁻⁸ m/s  (Fekete et al. 2002)

D_bankfull = 0.2 × Q^0.36    [m]  (Leopold & Maddock 1953)            (6.6b)
D_valley   = 80 × D_bankfull      (geological incision ratio,         (6.6c)
                                    Schumm 1977, Bull 1991)
           = 16 × Q^0.36

cut = min(D_valley / rounds, h × 0.23)

For bedrock channels (h > 1600 m):
  h_new = lerp(h − cut, h, 0.42)  (Whipple 2004: bedrock resistance)
```

Followed by 3×3 smoothing (20% blend toward neighborhood mean).

The 0.36 exponent is empirical from USGS stream-gauge data across hundreds
of US rivers.  The 80× incision ratio converts bankfull channel depth to
geological valley depth accumulated over ~10⁷ yr.

---

## 7. Biome Classification

### 7.1 Method: Whittaker Diagram

> **Whittaker (1975)** *Communities and Ecosystems*

Biomes are classified by nearest-centroid matching in normalized
(temperature, precipitation) space:

```
biome[i] = argmin_k  ((T_i/15 − T_k/15)² + (P_i/500 − P_k/500)²)     (7.1)
```

### 7.2 Biome Centroids

| ID | Biome | T (°C) | P (mm/yr) |
|----|-------|--------|-----------|
| 0 | Ocean | — | — |
| 1 | Tundra / Ice Sheet | −8 | 250 |
| 2 | Boreal Forest / Taiga | 2 | 600 |
| 3 | Temperate Forest | 12 | 1200 |
| 4 | Temperate Grassland | 10 | 500 |
| 5 | Mediterranean Scrub | 16 | 550 |
| 6 | Tropical Rainforest | 25 | 2500 |
| 7 | Tropical Savanna | 25 | 1000 |
| 8 | Hot Desert | 25 | 150 |
| 9 | Subtropical Forest | 20 | 1400 |
| 10 | Alpine | (derived) | (derived) |
| 11 | Steppe | 8 | 350 |

The normalization (T/15, P/500) makes temperature and precipitation contribute
roughly equally to the distance metric. Without normalization, precipitation
(range ~4000 mm) would dominate over temperature (range ~100°C).

### 7.3 Alpine Override

```
treeline = 2000 + noise × 300    [meters]                               (7.2)

Soft transition from 1700 m to treeline.
If h > treeline and biome is forest/grassland → Alpine
```

### 7.4 Riparian Corridors

```
If river_map[i] > 0.12 and biome ∈ {Desert, Steppe, Mediterranean,      (7.3)
   Temperate Grassland} → upgrade to Savanna or Forest
```

River corridors support vegetation even in dry climates (gallery forests,
riparian woodlands).

### 7.5 Biome Smoothing

2-pass mode filter applied after classification.  For each land cell, if fewer
than 2 of its 4 cardinal neighbors share its biome, the cell is replaced by the
most common non-ocean neighbor biome.  This removes single-cell biome anomalies
— primarily the coastal "eyelash" fringe where thin river channels create
isolated forest pixels in desert zones.

---

## 8. Settlement Suitability (Miami Model NPP — Lieth 1975)

> **Lieth, H. (1975)** "Modeling the primary productivity of the world".
> In *Primary Productivity of the Biosphere*, Springer, 237-263.
>
> **Diamond, J. (1997)** *Guns, Germs, and Steel*. W.W. Norton.

Settlement suitability is based on Net Primary Productivity (NPP), which
determines the carrying capacity of the land for agriculture.

**Miami model (Lieth 1975):**

```
NPP_T = 3000 / (1 + exp(1.315 − 0.119 × T))    [g/m²/yr]            (8.1)
NPP_P = 3000 × (1 − exp(−0.000664 × P))         [g/m²/yr]            (8.2)
NPP   = min(NPP_T, NPP_P)  (Liebig's law of the minimum)
```

**Planet scope** (`compute_settlement`):

```
elev_factor = max(1 − max(h − 500, 0) / 4000, 0)   (Körner 2003)     (8.3)
settlement = clamp(NPP / 2500 × elev_factor, 0, 1)
```

**Island scope** (`compute_settlement_grid`):

```
base = NPP / 2500 × elev_factor                                       (8.4)
river_bonus = river_map × 0.25    (Diamond 1997: navigable water)
coast_bonus = coastal_exposure × 0.15  (maritime access, fishing)

settlement = clamp(base + river_bonus + coast_bonus, 0, 1)            (8.5)
```

**Interpretation:** NPP peaks at ~2500 g/m²/yr in tropical rainforest
(T ≈ 27°C, P > 2000 mm).  The min() operator (Liebig's law) ensures that
either insufficient warmth or insufficient moisture limits productivity.
Elevation penalty models agricultural difficulty above 500 m (Körner 2003:
hypoxia, frost, short growing season).

---

## 9. Island-as-Crop

### 9.1 Concept

The island scope does not run a separate pipeline. Instead:

1. Generate the full planet (§2–§8)
2. Select an interesting rectangular region
3. Crop and upsample to fine resolution
4. Refine with additional erosion, hydrology, climate, and biomes

### 9.2 Region Selection

**Scoring function** (`find_interesting_region`):

The planet grid is scanned with a window of size
`(ISLAND_WIDTH × planet_km / island_km) × (ISLAND_HEIGHT × planet_km / island_km)`.

For each window position:

```
land_fraction = count(h > 0) / window_size                              (9.1)
mix_score     = 1 − |land_fraction − 0.4| × 2.5            ← peak at 40%
coast_norm    = count(coast cells) / window_size            ∈ [0, 1]
elev_norm     = min(elev_range / 5000, 1)                   ∈ [0, 1]
boundary_norm = 1 + min(boundary_count / window_size, 0.5) × 2   ∈ [1, 2]

score = mix_score × coast_norm × elev_norm × boundary_norm + noise × 0.3
```

All components normalized to [0, 1] or [1, 2]. Noise amplitude 0.3 provides
seed-dependent variety without dominating the signal. Coast detection checks
4 cardinal neighbors. Polar 10% excluded.

### 9.3 Bicubic Upsampling

The coarse planet relief (e.g., 20 × 10 cells in the window) is upsampled to
1024 × 512 using **Catmull-Rom bicubic interpolation**:

```
For each fine cell (fx, fy):
  (cx, cy) = map to coarse coordinates
  h = bicubic_sample(coarse_relief, cx, cy)                             (9.2)
```

Catmull-Rom is a C¹-continuous interpolant that passes through all data points
(no overshooting like cubic B-spline, no ringing like Lanczos).

### 9.4 Fractal Detail

Sub-grid topographic detail is added via Fractal Brownian Motion (FBM):

```
detail = FBM(x, y, octaves=6, freq₀=8, decay=0.5)                     (9.3)
amplitude = |h| × 0.15      (15% of local elevation)

h_fine = h_upsampled + detail × amplitude
```

This adds realistic roughness at scales below the planet grid resolution. The
Hurst exponent (implicit in the 0.5 decay) is ~0.7, consistent with observed
terrain power spectra.

### 9.5 Edge Fade

To avoid sharp edges at the island boundary:

```
margin = 0.06 × grid_dimension ± noise                                  (9.4)
fade   = smoothstep(distance_from_edge / margin)
h      = h × fade + (−500) × (1 − fade)
```

Terrain smoothly transitions to −500 m ocean at edges.

### 9.6 Fine-Scale Erosion

10 stream power steps at island resolution:

```
Δt = 100,000 years                                                      (9.5)
Δx = km_per_cell × 1000   (≈ 400 m)
U  = 0                     (no uplift — planet already has large-scale relief)

K_eff from boundary type:
  convergent → 2.0 × 10⁻⁶
  divergent  → 5.0 × 10⁻⁶
  interior   → 8.0 × 10⁻⁶
```

These 10 steps simulate 1 Myr of erosion at fine scale, creating realistic
valleys, ridges, and drainage networks from the upsampled + fractalized terrain.

### 9.7 Island Climate

`compute_climate_unified` is called with a `CellCache` that maps island cells
to their real planetary coordinates. This means:

- Temperature uses the correct latitude
- Precipitation uses the correct Hadley cell zone
- Wind direction matches the latitude band

The climate is self-consistent with the planet — not independently generated.

### 9.8 Island Biomes and Settlement

Same functions as planet scope (§7, §8), applied to the fine-resolution
temperature and precipitation fields. Temperature and precipitation are
smoothed (5-pass neighbor averaging) before biome classification to avoid
noisy boundaries.

---

## 10. Complete Parameter Table

### 10.1 Physical Constants

| Symbol | Value | Units | Source |
|--------|-------|-------|--------|
| R_earth | 6371 | km | Standard |
| ρ_c (continental) | 2800 | kg/m³ | Turcotte & Schubert (2002) |
| ρ_c (oceanic) | 2900 | kg/m³ | Turcotte & Schubert (2002) |
| ρ_m | 3300 | kg/m³ | Turcotte & Schubert (2002) |
| ρ_w | 1030 | kg/m³ | Standard |
| Γ (lapse rate) | 6.0 | K/km | ICAO, Holton & Hakim (2013) |
| g | 9.81 | m/s² | (user configurable) |

### 10.2 Tectonic Parameters

| Parameter | Value | Units | Source |
|-----------|-------|-------|--------|
| C_base continental | 43 | km | Rudnick & Gao (2003), cratonic base |
| C_base noise | ±15 | km | Multi-octave 3D noise (28–58 km range) |
| C_base oceanic | 7 | km | White et al. (1992) |
| CC collision thickening | bstr × 22 | km | Peak: 43+22=65 km ≈ Tibet |
| OC subduction thickening | bstr × 13 | km | Andes ~55 km total |
| Rift thinning | bstr × 20 | km | General |
| Transform thickening | bstr × 2 | km | Minor transpression (Alpine Fault) |
| Crust clamp | [5, 75] | km | Physical bounds |
| Deformation smoothing | 8 passes | — | σ ≈ 90 km, removes Voronoi wedges |
| Interior suppression L_rheol | 300 | km | Artemieva & Mooney 2001 (§3.2) |
| Flexural smoothing | 12 passes | — | σ ≈ 70 km (Watts 2001) |
| Fault K_eff boost | ×1.5 | — | Hovius & Stark (2006) |

### 10.3 Erosion Parameters

| Parameter | Value | Units | Source |
|-----------|-------|-------|--------|
| m (area exponent) | 0.5 | — | Whipple & Tucker (1999) |
| n (slope exponent) | 1.0 | — | Whipple & Tucker (1999) |
| κ (hillslope diffusivity) | 0.02 | m²/yr | CFL-stable for Δx=10km |
| MFD exponent p (planet) | 1.5 | — | Freeman (1991), diffuse routing |
| MFD exponent p (island) | 0.0 (D8) | — | D8 correct at ≤1 km |
| K_eff Granite | 0.5 × 10⁻⁶ | m^0.5/yr | Harel et al. (2016) |
| K_eff Quartzite | 0.8 × 10⁻⁶ | m^0.5/yr | Harel et al. (2016) |
| K_eff Basalt | 1.0 × 10⁻⁶ | m^0.5/yr | Harel et al. (2016) |
| K_eff Schist | 1.2 × 10⁻⁶ | m^0.5/yr | Harel et al. (2016) |
| K_eff Sandstone | 2.0 × 10⁻⁶ | m^0.5/yr | Harel et al. (2016) |
| K_eff Limestone | 3.0 × 10⁻⁶ | m^0.5/yr | Harel et al. (2016) |
| Convergent uplift | bstr × 0.005 × speed_factor | m/yr | GPS: 0.5–10 mm/yr (Bevis et al.) |
| Divergent subsidence | bstr × 0.002 × speed_factor | m/yr | — |
| Epochs × steps | 1 × 3 = 3 | — | Light erosion (§4.4) |
| Δt per step | 500,000 | yr | — |
| Total sim time | 1.5 | Myr | — |
| Noise injection | ±5 | m | Breaks radial channel coherence |
| Isostatic relaxation | 85/15 | — | ~1 Myr Maxwell time |

### 10.4 Climate Parameters

| Parameter | Value | Units | Source |
|-----------|-------|-------|--------|
| T_equator | 28 | °C | Zonal mean obs. |
| T_lat coefficient | 0.007 | °C/deg² | Quadratic fit |
| Greenhouse factor | 5 × ln(1+P) | °C | **⚠ Rough** |
| Ocean T modifier | +2 | °C | Thermal inertia |
| ITCZ precipitation | 2000 | mm/yr | Held & Hou (1980) |
| Subtropical minimum | 250 | mm/yr | Held & Hou (1980) |
| Westerly belt max | 900 | mm/yr | — |
| Polar minimum | 150 | mm/yr | — |
| Windward decay rate | 0.02 | per cell | Normalized to ~10 km cells |
| Windward amplitude | 600 | mm/yr | — |
| Coast detection range | 3 cells | ~60 km | — |
| Coastal P amplitude | 600 | mm/yr | — |
| Orographic factor | 0.8 | mm/m | Roe (2005) |
| Rain shadow factor | 0.40 × exp(−d/250km) | mm/m | Smith (1979), distance decay |
| CC altitude factor | exp(−h × 4e-4), floor 0.1 | — | Clausius-Clapeyron exponential |
| Coriolis deflection | 0.35 × sin(lat) | — | f = 2Ω sin(φ) |
| Noise amplitude | 60 | mm/yr | — |

### 10.5 Biome Parameters

| Parameter | Value | Source |
|-----------|-------|--------|
| T normalization | /15 °C | Whittaker (1975) |
| P normalization | /500 mm | Whittaker (1975) |
| Treeline base | 2000 m | General alpine ecology |
| Treeline noise | ±300 m | — |
| Riparian threshold | river > 0.12 | — |

### 10.6 Island Crop Parameters

| Parameter | Value | Units |
|-----------|-------|-------|
| FBM octaves | 6 | — |
| FBM base frequency | 8.0 | — |
| FBM amplitude decay | 0.5 | /octave |
| Elevation amplitude | 15% | of local |h| |
| Edge margin | 6% | of grid dimension |
| Fine erosion steps | 10 | — |
| Fine Δt | 100,000 | yr |
| Fine total time | 1 | Myr |
| K_eff convergent | 2.0 × 10⁻⁶ | m^0.5/yr |
| K_eff divergent | 5.0 × 10⁻⁶ | m^0.5/yr |
| K_eff interior | 8.0 × 10⁻⁶ | m^0.5/yr |
| Smoothing passes (T/P) | 5 | before biome classification |

---

## 11. Pipeline Execution Order

```
┌──────────────────────────────────────────────────────────────────┐
│  INPUT: seed, PlanetInputs, TectonicInputs, detail, scope      │
└──────────────────┬───────────────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────────────────┐
│  1. PLATE TECTONICS  (§2)                          [0% → 24%]   │
│     build_irregular_plate_field → evolve_plate_field             │
│     → classify boundaries → plate_field, boundary_types, bstr   │
└──────────────────┬───────────────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────────────────┐
│  2. RELIEF PHYSICS  (§3, §4)                      [24% → 74%]   │
│     crustal_thickness → flexural_smooth → rock_type → K_eff     │
│     → isostatic_elevation → noise overlay                        │
│     → 1 epoch × 3 stream_power_step + noise injection           │
│     → isostatic relaxation → land smoothing                     │
│     → hypsometric correction → detail noise                     │
│     → sea_level from ocean_percent                               │
└──────────────────┬───────────────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────────────────┐
│  3. EVENTS  (§4.5*)                                              │
│     apply meteorite impacts, rifts, ocean shifts                 │
└──────────────────┬───────────────────────────────────────────────┘
                   │
            ┌──────┴──────┐
            │             │
     [Planet scope]  [Island scope]
            │             │
            ▼             ▼
┌──────────────┐  ┌──────────────────────────────────────────────┐
│ 4. SLOPE     │  │ 4'. ISLAND CROP  (§9)           [70% → 100%] │
│ 5. CLIMATE   │  │     find_interesting_region                   │
│ 6. HYDROLOGY │  │     → bicubic upsample + FBM fractal         │
│ 7. BIOMES    │  │     → edge fade                               │
│ 8. SETTLEMENT│  │     → 10× stream_power (fine)                 │
│              │  │     → slope → hydrology → fluvial carving     │
│ [74% → 100%]│  │     → climate (real lat/lon)                   │
│              │  │     → smooth T/P → biomes → settlement        │
└──────────────┘  └──────────────────────────────────────────────┘
            │             │
            ▼             ▼
┌──────────────────────────────────────────────────────────────────┐
│  OUTPUT: WasmSimulationResult                                    │
│    plates, boundary_types, height_map, slope_map, river_map,     │
│    lake_map, flow_direction, flow_accumulation, temperature_map, │
│    precipitation_map, biome_map, settlement_map                  │
└──────────────────────────────────────────────────────────────────┘
```

---

## 12. Known Limitations and Future Work

### 12.1 Resolved Issues (fixed 2026-02-25)

All critical issues from the initial audit have been resolved:

| # | Issue | Was | Now |
|---|-------|-----|-----|
| 1 | Uplift rates | 0.003 m/yr (3 mm/yr max) | 0.005 m/yr (5 mm/yr max) |
| 2 | Rain shadow | No distance decay, 1200 km trace | Exponential decay (250 km e-fold), 500 km max |
| 3 | Coriolis | Constant 0.2 | 0.35 × sin(lat) |
| 4 | Clausius-Clapeyron | Linear: 1 − h×2.5e-4 | Exponential: exp(−h×4e-4) |
| 5 | Coast detection (scoring) | East neighbor only | 4 cardinal neighbors |
| 6 | Region scoring | Unnormalized (cells × meters) | All components in [0, 1] |
| 7 | Transform thickening | 5 km (heuristic) | 2 km (transpression only) |
| 8 | Windward moisture | Breaks on first ocean | Continues over water, counts land only |
| 9 | Divergent subsidence | 1 mm/yr | 2 mm/yr |

### 12.2 Missing Physics

| Feature | Description | Why it matters |
|---------|-------------|---------------|
| Ocean age–depth | GDH1: d(t) = 2600 + 365√t | Ocean bathymetry is uniform; should deepen away from ridges |
| Flexural isostasy | D∇⁴w = q(x) − (ρ_m−ρ_c)gw | No foreland basins or flexural bulges |
| Slab pull / ridge push | Driving forces → plate speeds | Plate speeds are random, not physics-driven |
| No-Net-Rotation | ΣΩ_k × A_k = 0 | Plates can have net angular momentum (unphysical) |
| Sediment transport | Rivers deposit, not just erode | No deltas, alluvial plains, or sedimentary basins |
| Ocean currents | Sverdrup balance, thermohaline | No Gulf Stream warming or Humboldt cooling |
| Glaciation | Ice sheets at high lat / altitude | Tundra is classified but no ice dynamics |
| Seasonality | Axial tilt → seasons | Single annual mean, no monsoons |
| Subduction dip | Slab angle from age + speed | Uniform subduction geometry |

### 12.3 Remaining Heuristic Components

These parts of the code are not derived from physics:

1. **Plate field growth** (§2.2) — Dijkstra flood fill with tuned cost function
2. **Noise overlay** (§3.5, Eq. 3.6) — 120 m of arbitrary topographic noise
3. ~~**Interior suppression**~~ → replaced with thermal age proxy (Artemieva & Mooney 2001, §3.2)
4. ~~**Terrain detail noise**~~ → replaced with ETOPO1-calibrated fBm (Sayles & Thomas 1978, §4.6)
5. **Noise injection** (§4.4, Eq. 4.5a) — ±5 m perturbation to break grid artifacts
6. ~~**Fluvial valley carving**~~ → replaced with hydraulic geometry (Leopold & Maddock 1953, §6.6)
7. **Biome smoothing** (§7.5) — 2-pass mode filter, not ecology
8. ~~**Settlement model**~~ → replaced with Miami model NPP (Lieth 1975, §8)
9. **FBM fractal detail** (§9.4) — sub-grid roughness is not geology

### 12.4 Remaining Improvement Priority

**Tier 1 (high impact, moderate effort):**
- Implement ocean age field + GDH1 bathymetry
- Add sediment deposition in lowlands

**Tier 2 (significant effort):**
- Flexural isostasy (4th-order PDE solver)
- Ocean currents (energy balance model)
- Seasonal climate cycle

---

## 13. References

1. **Braun, J. & Willett, S.D. (2013).** A very efficient O(n), implicit and parallel method to solve the stream power equation governing fluvial incision and landscape evolution. *Geomorphology*, 180-181, 170-179.

2. **Christensen, N.I. & Mooney, W.D. (1995).** Seismic velocity structure and composition of the continental crust: A global view. *J. Geophys. Res.*, 100(B6), 9761-9788.

3. **Cox, A. & Hart, R.B. (1986).** *Plate Tectonics: How It Works*. Blackwell Scientific.

4. **England, P. & McKenzie, D. (1982).** A thin viscous sheet model for continental deformation. *Geophysical Journal International*, 70(2), 295-321.

5. **Freeman, T.G. (1991).** Calculating catchment area with divergent flow based on a regular grid. *Computers & Geosciences*, 17(3), 413-422.

6. **Harel, M.-A., Mudd, S.M. & Attal, M. (2016).** Global analysis of the stream power law parameters based on worldwide 10Be denudation rates. *Geomorphology*, 268, 184-196.

7. **Held, I.M. & Hou, A.Y. (1980).** Nonlinear axially symmetric circulations in a nearly inviscid atmosphere. *J. Atmos. Sci.*, 37, 515-533.

8. **Holton, J.R. & Hakim, G.J. (2013).** *An Introduction to Dynamic Meteorology*, 5th ed. Academic Press.

9. **Hovius, N. & Stark, C.P. (2006).** Landslide-driven erosion and topographic evolution of active mountain belts. In *Landslides from Massive Rock Slope Failure*, S.G. Evans et al. (eds), NATO Science Series, 573-590.

10. **Howard, A.D. (1994).** A detachment-limited model of drainage basin evolution. *Water Resources Research*, 30(7), 2261-2285.

11. **O'Callaghan, J.F. & Mark, D.M. (1984).** The extraction of drainage networks from digital elevation data. *Computer Vision, Graphics, and Image Processing*, 28(3), 323-344.

12. **Peixoto, J.P. & Oort, A.H. (1992).** *Physics of Climate*. American Institute of Physics.

13. **Roe, G.H. (2005).** Orographic precipitation. *Annual Review of Earth and Planetary Sciences*, 33, 645-671.

14. **Salles, T. et al. (2023).** Hundred million years of landscape dynamics from catchment to global scale. *Science*, 379(6635), 918-923.

15. **Smith, R.B. (1979).** The influence of mountains on the atmosphere. *Advances in Geophysics*, 21, 87-230.

16. **Turcotte, D.L. & Schubert, G. (2002).** *Geodynamics*, 2nd ed. Cambridge University Press.

17. **Watts, A.B. (2001).** *Isostasy and Flexure of the Lithosphere*. Cambridge University Press.

18. **Whipple, K.X. & Tucker, G.E. (1999).** Dynamics of the stream-power river incision model. *J. Geophys. Res.*, 104(B8), 17661-17674.

19. **Whittaker, R.H. (1975).** *Communities and Ecosystems*, 2nd ed. Macmillan.

20. **Lieth, H. (1975).** Modeling the primary productivity of the world. In *Primary Productivity of the Biosphere*, Springer, 237-263.

21. **Artemieva, I.M. & Mooney, W.D. (2001).** Thermal thickness and evolution of Precambrian lithosphere: A global study. *J. Geophys. Res.*, 106(B8), 16387-16414.

22. **Leopold, L.B. & Maddock, T. (1953).** The hydraulic geometry of stream channels and some physiographic implications. *USGS Professional Paper 252*.

23. **Sayles, R.S. & Thomas, T.R. (1978).** Surface topography as a non-stationary random process. *Nature*, 271, 431-434.

24. **Huang, J. & Turcotte, D.L. (1989).** Fractal mapping of digitized images: Application to the topography of Arizona and comparisons with synthetic images. *J. Geophys. Res.*, 94(B6), 7491-7495.

25. **Fekete, B.M., Vörösmarty, C.J. & Grabs, W. (2002).** High-resolution fields of global runoff combining observed river discharge and simulated water balances. *Global Biogeochemical Cycles*, 16(3), 1042.

26. **Montgomery, D.R. & Brandon, M.T. (2002).** Topographic controls on erosion rates in tectonically active mountain ranges. *Earth and Planetary Science Letters*, 201(3-4), 481-489.

27. **Schumm, S.A. (1977).** *The Fluvial System*. Wiley.

28. **Körner, C. (2003).** *Alpine Plant Life*, 2nd ed. Springer.

29. **Diamond, J. (1997).** *Guns, Germs, and Steel: The Fates of Human Societies*. W.W. Norton.

---

*Document updated 2026-02-26. Reflects `rust/planet_engine/src/lib.rs` after
Phase A–C unification, physics fixes, deformation propagation (England &
McKenzie 1982), MFD area routing (Freeman 1991 / goSPL), and science
replacement of 4 heuristic components: settlement (Lieth 1975), interior
suppression (Artemieva & Mooney 2001), valley carving (Leopold & Maddock 1953),
and terrain noise (Sayles & Thomas 1978 / Huang & Turcotte 1989).*
