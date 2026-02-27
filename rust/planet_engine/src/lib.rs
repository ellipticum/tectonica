use js_sys::Array;
use serde::{Deserialize, Serialize};
use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, VecDeque};
use std::sync::OnceLock;
use wasm_bindgen::prelude::*;

const WORLD_WIDTH: usize = 4096;
const WORLD_HEIGHT: usize = 2048;
const WORLD_SIZE: usize = WORLD_WIDTH * WORLD_HEIGHT;
const ISLAND_WIDTH: usize = 1024;
const ISLAND_HEIGHT: usize = 512;
const RADIANS: f32 = std::f32::consts::PI / 180.0;
const KILOMETERS_PER_DEGREE: f32 = 111.319_490_793;

/// xoshiro128++ PRNG (Blackman & Vigna 2021, "Scrambled Linear Pseudorandom
/// Number Generators", ACM TOMS 47(4)).  128-bit state, period 2^128−1.
/// Passes BigCrush, PractRand; standard choice for 32-bit non-crypto PRNG.
/// Seeded via SplitMix32 (Steele et al. 2014) to decorrelate seed bits.
#[derive(Clone)]
struct Rng {
    s: [u32; 4],
}

impl Rng {
    fn new(seed: u32) -> Self {
        // SplitMix32 seeding: ensures all 128 state bits are populated
        // even from a single 32-bit seed (Steele et al. 2014).
        let mut z = seed;
        let mut next_sm = || -> u32 {
            z = z.wrapping_add(0x9E37_79B9);
            let mut r = z;
            r = (r ^ (r >> 15)).wrapping_mul(0x85EB_CA6B);
            r = (r ^ (r >> 13)).wrapping_mul(0xC2B2_AE35);
            r ^ (r >> 16)
        };
        let s0 = next_sm();
        let s1 = next_sm();
        let s2 = next_sm();
        let s3 = next_sm();
        Self { s: [s0, s1, s2, s3] }
    }

    /// Generate uniform f32 in [0, 1).
    fn next_f32(&mut self) -> f32 {
        let result = (self.s[0].wrapping_add(self.s[3]))
            .rotate_left(7)
            .wrapping_add(self.s[0]);
        let t = self.s[1] << 9;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(11);
        (result >> 8) as f32 / 16_777_216.0 // 24-bit mantissa → uniform [0,1)
    }
}

#[inline]
fn clampf(value: f32, min: f32, max: f32) -> f32 {
    if value < min {
        min
    } else if value > max {
        max
    } else {
        value
    }
}

#[inline]
fn lerpf(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

#[inline]
/// Murmur3-family integer hash (mixing constants from Murmur3 finalizer).
fn hash_u32(mut v: u32) -> u32 {
    v ^= v >> 16;
    v = v.wrapping_mul(0x7FEB_352D);
    v ^= v >> 15;
    v = v.wrapping_mul(0x846C_A68B);
    v ^= v >> 16;
    v
}

#[inline]
fn hash3(seed: u32, x: i32, y: i32, z: i32) -> u32 {
    let hx = (x as u32).wrapping_mul(0x8DA6_B343);
    let hy = (y as u32).wrapping_mul(0xD816_3841);
    let hz = (z as u32).wrapping_mul(0xCB1A_B31F);
    hash_u32(seed ^ hx ^ hy ^ hz)
}

#[inline]
fn hash_to_unit(h: u32) -> f32 {
    (h as f32 / 4_294_967_295.0) * 2.0 - 1.0
}

#[inline]
fn value_noise3(x: f32, y: f32, z: f32, seed: u32) -> f32 {
    let x0 = x.floor() as i32;
    let y0 = y.floor() as i32;
    let z0 = z.floor() as i32;
    let x1 = x0 + 1;
    let y1 = y0 + 1;
    let z1 = z0 + 1;

    let tx = x - x0 as f32;
    let ty = y - y0 as f32;
    let tz = z - z0 as f32;
    let fx = tx * tx * (3.0 - 2.0 * tx);
    let fy = ty * ty * (3.0 - 2.0 * ty);
    let fz = tz * tz * (3.0 - 2.0 * tz);

    let v000 = hash_to_unit(hash3(seed, x0, y0, z0));
    let v100 = hash_to_unit(hash3(seed, x1, y0, z0));
    let v010 = hash_to_unit(hash3(seed, x0, y1, z0));
    let v110 = hash_to_unit(hash3(seed, x1, y1, z0));
    let v001 = hash_to_unit(hash3(seed, x0, y0, z1));
    let v101 = hash_to_unit(hash3(seed, x1, y0, z1));
    let v011 = hash_to_unit(hash3(seed, x0, y1, z1));
    let v111 = hash_to_unit(hash3(seed, x1, y1, z1));

    let x00 = lerpf(v000, v100, fx);
    let x10 = lerpf(v010, v110, fx);
    let x01 = lerpf(v001, v101, fx);
    let x11 = lerpf(v011, v111, fx);
    let y0v = lerpf(x00, x10, fy);
    let y1v = lerpf(x01, x11, fy);
    lerpf(y0v, y1v, fz)
}

/// Spherical fractional Brownian motion (Musgrave et al. 1989, SIGGRAPH).
/// Lacunarity 2.03 (non-integer avoids coherent aliasing artifacts).
/// Persistence 0.52 → H ≈ 0.94 (very smooth large-scale variation).
/// Domain rotation per octave breaks axis-aligned periodicity.
#[inline]
fn spherical_fbm(sx: f32, sy: f32, sz: f32, seed_phase: f32) -> f32 {
    let mut freq = 2.25_f32;
    let mut amp = 1.0_f32;
    let mut sum = 0.0_f32;
    let mut norm = 0.0_f32;
    let mut x = sx;
    let mut y = sy;
    let mut z = sz;
    let seed_base = hash_u32(seed_phase.to_bits() ^ 0x9E37_79B9);

    for octave in 0..5 {
        let octave_seed = hash_u32(seed_base.wrapping_add((octave as u32) * 0x85EB_CA6B));
        let px = x * freq + seed_phase * 0.71 + octave as f32 * 17.0;
        let py = y * freq - seed_phase * 0.53 + octave as f32 * 11.0;
        let pz = z * freq + seed_phase * 0.37 + octave as f32 * 7.0;
        let n = value_noise3(px, py, pz, octave_seed);
        sum += n * amp;
        norm += amp;
        // Domain rotation: R(30°, axis=(1,1,1)/√3), proper orthogonal (det=1).
        // R = I·cos θ + (1−cos θ)·n·nᵀ + sin θ·[n]_× (Rodrigues' formula).
        // cos30°=0.86603, sin30°=0.5, n=(1,1,1)/√3.
        let rx = x *  0.9107 + y * -0.2440 + z *  0.3333;
        let ry = x *  0.3333 + y *  0.9107 + z * -0.2440;
        let rz = x * -0.2440 + y *  0.3333 + z *  0.9107;
        x = rx;
        y = ry;
        z = rz;
        freq *= 2.03;
        amp *= 0.52;
    }

    sum / norm.max(1e-6)
}

#[inline]
fn index(x: usize, y: usize) -> usize {
    y * WORLD_WIDTH + x
}

#[inline]
fn random_range(rng: &mut Rng, min: f32, max: f32) -> f32 {
    min + rng.next_f32() * (max - min)
}

#[inline]
fn spherical_wrap(mut x: i32, mut y: i32) -> (usize, usize) {
    while y < 0 || y >= WORLD_HEIGHT as i32 {
        if y < 0 {
            y = -y - 1;
            x += (WORLD_WIDTH / 2) as i32;
        } else {
            y = 2 * WORLD_HEIGHT as i32 - y - 1;
            x += (WORLD_WIDTH / 2) as i32;
        }
    }

    let mut wrapped_x = x % WORLD_WIDTH as i32;
    if wrapped_x < 0 {
        wrapped_x += WORLD_WIDTH as i32;
    }
    let wrapped_y = y.clamp(0, WORLD_HEIGHT as i32 - 1);
    (wrapped_x as usize, wrapped_y as usize)
}

#[inline]
fn index_spherical(x: i32, y: i32) -> usize {
    let (sx, sy) = spherical_wrap(x, y);
    index(sx, sy)
}

#[inline]
fn plate_velocity_xy_from_omega(
    omega_x: f32,
    omega_y: f32,
    omega_z: f32,
    sx: f32,
    sy: f32,
    sz: f32,
    radius_cm: f32,
) -> (f32, f32) {
    // v3 = (omega x r) * R. Here omega is angular velocity, r is unit sphere position.
    let vx3 = (omega_y * sz - omega_z * sy) * radius_cm;
    let vy3 = (omega_z * sx - omega_x * sz) * radius_cm;
    let vz3 = (omega_x * sy - omega_y * sx) * radius_cm;

    let cos_lat = (sx * sx + sy * sy).sqrt().max(1e-6);
    let ex = -sy / cos_lat;
    let ey = sx / cos_lat;
    let nx = -sz * sx / cos_lat;
    let ny = -sz * sy / cos_lat;
    let nz = cos_lat;

    let east = vx3 * ex + vy3 * ey;
    let north = vx3 * nx + vy3 * ny + vz3 * nz;
    // Map Y grows to the south, so invert north.
    (east, -north)
}

#[inline]
fn plate_velocity_xy_at_cell(
    plate: &PlateVector,
    sx: f32,
    sy: f32,
    sz: f32,
    radius_cm: f32,
) -> (f32, f32) {
    plate_velocity_xy_from_omega(
        plate.omega_x,
        plate.omega_y,
        plate.omega_z,
        sx,
        sy,
        sz,
        radius_cm,
    )
}

fn min_max(values: &[f32]) -> (f32, f32) {
    let mut min = f32::INFINITY;
    let mut max = -f32::INFINITY;
    for &v in values {
        min = min.min(v);
        max = max.max(v);
    }
    (min, max)
}

struct WorldCache {
    lat_by_y: Vec<f32>,
    lon_by_x: Vec<f32>,
    x_by_cell: Vec<f32>,
    y_by_cell: Vec<f32>,
    z_by_cell: Vec<f32>,
}

static WORLD_CACHE: OnceLock<WorldCache> = OnceLock::new();

fn world_cache() -> &'static WorldCache {
    WORLD_CACHE.get_or_init(|| {
        let mut lat_by_y = vec![0.0_f32; WORLD_SIZE];
        let mut lon_by_x = vec![0.0_f32; WORLD_SIZE];
        let mut x_by_cell = vec![0.0_f32; WORLD_SIZE];
        let mut y_by_cell = vec![0.0_f32; WORLD_SIZE];
        let mut z_by_cell = vec![0.0_f32; WORLD_SIZE];

        for y in 0..WORLD_HEIGHT {
            let lat_deg = 90.0 - (y as f32 + 0.5) * (180.0 / WORLD_HEIGHT as f32);
            let lat_rad = lat_deg * RADIANS;
            let cos_lat = lat_rad.cos();
            for x in 0..WORLD_WIDTH {
                let lon_deg = (x as f32 + 0.5) * (360.0 / WORLD_WIDTH as f32) - 180.0;
                let lon_rad = lon_deg * RADIANS;
                let i = index(x, y);
                lat_by_y[i] = lat_deg;
                lon_by_x[i] = lon_deg;
                x_by_cell[i] = cos_lat * lon_rad.cos();
                y_by_cell[i] = cos_lat * lon_rad.sin();
                z_by_cell[i] = lat_rad.sin();
            }
        }

        WorldCache {
            lat_by_y,
            lon_by_x,
            x_by_cell,
            y_by_cell,
            z_by_cell,
        }
    })
}

// ---------------------------------------------------------------------------
// GridConfig: unified grid abstraction for planet (spherical) and island (flat)
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct GridConfig {
    width: usize,
    height: usize,
    size: usize,
    is_spherical: bool,
    km_per_cell_x: f32,
    km_per_cell_y: f32,
    /// cos(latitude) for each row; used for latitude-corrected smoothing
    /// and deformation propagation on equirectangular grids.
    /// All 1.0 for flat (island) grids.
    cos_lat_by_y: Vec<f32>,
}

impl GridConfig {
    /// Standard planet scope preset (equirectangular, WORLD_WIDTH x WORLD_HEIGHT).
    fn planet() -> Self {
        let km_y = (std::f32::consts::PI * 6371.0) / WORLD_HEIGHT as f32; // ~19.6 km
        let km_x = (2.0 * std::f32::consts::PI * 6371.0) / WORLD_WIDTH as f32; // ~19.6 km at equator
        // Precompute cos(latitude) per row, clamped to avoid pole singularity.
        // At 85° cos ≈ 0.087; clamping preserves numerical stability while
        // allowing strong latitude correction up to very high latitudes.
        let mut cos_lat_by_y = vec![0.0_f32; WORLD_HEIGHT];
        for y in 0..WORLD_HEIGHT {
            let lat_rad = (90.0 - (y as f32 + 0.5) * (180.0 / WORLD_HEIGHT as f32)) * RADIANS;
            cos_lat_by_y[y] = lat_rad.cos().abs().max(0.087); // clamp at ~85°
        }
        Self {
            width: WORLD_WIDTH,
            height: WORLD_HEIGHT,
            size: WORLD_SIZE,
            is_spherical: true,
            km_per_cell_x: km_x,
            km_per_cell_y: km_y,
            cos_lat_by_y,
        }
    }

    /// Island scope preset (flat clamped grid).
    /// `km_per_cell` is the physical size of one cell in both x and y.
    fn island(width: usize, height: usize, km_per_cell: f32) -> Self {
        Self {
            width,
            height,
            size: width * height,
            is_spherical: false,
            km_per_cell_x: km_per_cell,
            km_per_cell_y: km_per_cell,
            cos_lat_by_y: vec![1.0; height], // flat grid — no latitude distortion
        }
    }

    /// Row-major index from (x, y) coordinates.
    #[inline]
    fn index(&self, x: usize, y: usize) -> usize {
        y * self.width + x
    }

    /// Neighbor with wrapping strategy: spherical wrap or clamp.
    #[inline]
    fn neighbor(&self, x: i32, y: i32) -> usize {
        if self.is_spherical {
            let (sx, sy) = spherical_wrap(x, y);
            sy * self.width + sx
        } else {
            let xx = x.clamp(0, self.width as i32 - 1) as usize;
            let yy = y.clamp(0, self.height as i32 - 1) as usize;
            yy * self.width + xx
        }
    }

}

// ---------------------------------------------------------------------------
// CellCache: pre-computed per-cell coordinates for noise and climate.
// Replaces WorldCache for the abstracted pipeline.
// ---------------------------------------------------------------------------

struct CellCache {
    /// 3D x-coordinate for noise functions (spherical or synthetic).
    noise_x: Vec<f32>,
    /// 3D y-coordinate for noise functions.
    noise_y: Vec<f32>,
    /// 3D z-coordinate for noise functions.
    noise_z: Vec<f32>,
    /// Latitude in degrees (or latitude-equivalent for island scope).
    lat_deg: Vec<f32>,
    /// Longitude in degrees (or longitude-equivalent for island scope).
    lon_deg: Vec<f32>,
}

impl CellCache {
    /// Build cache for planet scope — identical to WorldCache values.
    fn for_planet(grid: &GridConfig) -> Self {
        let mut noise_x = vec![0.0_f32; grid.size];
        let mut noise_y = vec![0.0_f32; grid.size];
        let mut noise_z = vec![0.0_f32; grid.size];
        let mut lat_deg = vec![0.0_f32; grid.size];
        let mut lon_deg_v = vec![0.0_f32; grid.size];

        for y in 0..grid.height {
            let lat = 90.0 - (y as f32 + 0.5) * (180.0 / grid.height as f32);
            let lat_rad = lat * RADIANS;
            let cos_lat = lat_rad.cos();
            for x in 0..grid.width {
                let lon = (x as f32 + 0.5) * (360.0 / grid.width as f32) - 180.0;
                let lon_rad = lon * RADIANS;
                let i = grid.index(x, y);
                noise_x[i] = cos_lat * lon_rad.cos();
                noise_y[i] = cos_lat * lon_rad.sin();
                noise_z[i] = lat_rad.sin();
                lat_deg[i] = lat;
                lon_deg_v[i] = lon;
            }
        }

        Self { noise_x, noise_y, noise_z, lat_deg, lon_deg: lon_deg_v }
    }

    /// Build cache for a cropped island region at a known planetary position.
    /// `center_lat_deg` / `center_lon_deg` are the planet-space coordinates
    /// of the island center.  The island grid covers `grid.width * km_per_cell_x`
    /// km in longitude and `grid.height * km_per_cell_y` km in latitude.
    fn for_island_crop(grid: &GridConfig, center_lat_deg: f32, center_lon_deg: f32) -> Self {
        let mut noise_x = vec![0.0_f32; grid.size];
        let mut noise_y = vec![0.0_f32; grid.size];
        let mut noise_z = vec![0.0_f32; grid.size];
        let mut lat_deg_v = vec![0.0_f32; grid.size];
        let mut lon_deg_v = vec![0.0_f32; grid.size];

        // Total span in degrees
        let lat_span = (grid.height as f32 * grid.km_per_cell_y) / 111.0; // 111 km/deg
        let cos_center = (center_lat_deg * RADIANS).cos().abs().max(0.05);
        let lon_span = (grid.width as f32 * grid.km_per_cell_x) / (111.0 * cos_center);

        for y in 0..grid.height {
            let lat = center_lat_deg + (0.5 - (y as f32 + 0.5) / grid.height as f32) * lat_span;
            let lat_rad = lat * RADIANS;
            let cos_lat = lat_rad.cos();
            for x in 0..grid.width {
                let lon = center_lon_deg
                    + ((x as f32 + 0.5) / grid.width as f32 - 0.5) * lon_span;
                let lon_rad = lon * RADIANS;
                let i = grid.index(x, y);
                noise_x[i] = cos_lat * lon_rad.cos();
                noise_y[i] = cos_lat * lon_rad.sin();
                noise_z[i] = lat_rad.sin();
                lat_deg_v[i] = lat;
                lon_deg_v[i] = lon;
            }
        }

        Self { noise_x, noise_y, noise_z, lat_deg: lat_deg_v, lon_deg: lon_deg_v }
    }
}

#[derive(Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct PlanetInputs {
    pub radius_km: f32,
    pub gravity: f32,
    pub density: f32,
    pub rotation_hours: f32,
    pub axial_tilt_deg: f32,
    pub eccentricity: f32,
    pub atmosphere_bar: f32,
    pub ocean_percent: f32,
}

#[derive(Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct TectonicInputs {
    pub plate_count: i32,
    pub plate_speed_cm_per_year: f32,
    pub mantle_heat: f32,
}

#[derive(Deserialize, Serialize, Clone, Default)]
#[serde(rename_all = "camelCase")]
pub struct WorldEventRecord {
    pub kind: String,
    pub latitude: f32,
    pub longitude: f32,
    #[serde(default)]
    pub diameter_km: f32,
    #[serde(default)]
    pub speed_kms: f32,
    #[serde(default)]
    pub angle_deg: f32,
    #[serde(default)]
    pub density_kg_m3: f32,
    #[serde(default)]
    pub radius_km: f32,
    #[serde(default)]
    pub magnitude: f32,
    #[serde(default)]
    pub id: String,
    #[serde(default)]
    pub created_at: String,
    #[serde(default)]
    pub summary: String,
    #[serde(default)]
    pub energy_joule: Option<f64>,
}

#[derive(Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct SimulationConfig {
    pub seed: u32,
    pub planet: PlanetInputs,
    pub tectonics: TectonicInputs,
    pub events: Vec<WorldEventRecord>,
    #[serde(default)]
    pub generation_preset: Option<String>,
    #[serde(default)]
    pub scope: Option<String>,
    /// Island tectonic type: "continental" | "arc" | "hotspot" | "rift"
    #[serde(default, rename = "islandType")]
    pub island_type: Option<String>,
    /// Island physical width in km (grid is always ISLAND_WIDTH cells wide)
    #[serde(default, rename = "islandScaleKm")]
    pub island_scale_km: Option<f32>,
}

#[derive(Clone, Copy)]
enum RecomputeReason {
    Global,
    Tectonics,
    Events,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum GenerationScope {
    Planet,
    Island,
}

#[derive(Clone, Copy)]
enum GenerationPreset {
    Ultra,
    Fast,
    Balanced,
    Detailed,
}

#[derive(Clone, Copy)]
struct DetailProfile {
    erosion_rounds: usize,
    fluvial_rounds: usize,
    max_kernel_radius: i32,
    plate_evolution_steps: usize,
}

fn parse_generation_preset(value: Option<&str>) -> GenerationPreset {
    match value.unwrap_or("balanced") {
        "ultra" => GenerationPreset::Ultra,
        "fast" => GenerationPreset::Fast,
        "detailed" => GenerationPreset::Detailed,
        _ => GenerationPreset::Balanced,
    }
}

fn detail_profile(preset: GenerationPreset) -> DetailProfile {
    match preset {
        GenerationPreset::Ultra => DetailProfile {
            erosion_rounds: 0,
            fluvial_rounds: 0,
            max_kernel_radius: 3,
            plate_evolution_steps: 2,
        },
        GenerationPreset::Fast => DetailProfile {
            erosion_rounds: 1,
            fluvial_rounds: 1,
            max_kernel_radius: 4,
            plate_evolution_steps: 4,
        },
        GenerationPreset::Detailed => DetailProfile {
            erosion_rounds: 3,
            fluvial_rounds: 3,
            max_kernel_radius: 6,
            plate_evolution_steps: 10,
        },
        GenerationPreset::Balanced => DetailProfile {
            erosion_rounds: 2,
            fluvial_rounds: 2,
            max_kernel_radius: 5,
            plate_evolution_steps: 7,
        },
    }
}

fn parse_reason(reason: &str) -> RecomputeReason {
    match reason {
        "tectonics" => RecomputeReason::Tectonics,
        "events" => RecomputeReason::Events,
        _ => RecomputeReason::Global,
    }
}

fn parse_scope(value: Option<&str>) -> GenerationScope {
    match value.unwrap_or("planet") {
        "island" | "tasmania" => GenerationScope::Island,
        _ => GenerationScope::Planet,
    }
}

fn evaluate_recompute(reason: RecomputeReason) -> Vec<String> {
    let all = [
        "planet",
        "plates",
        "relief",
        "hydrology",
        "climate",
        "biomes",
        "settlement",
    ];

    let active: &[&str] = match reason {
        RecomputeReason::Global => &all,
        RecomputeReason::Tectonics => &all,
        RecomputeReason::Events => &["relief", "hydrology", "climate", "biomes", "settlement"],
    };

    all.iter()
        .filter(|id| active.contains(id))
        .map(|id| (*id).to_string())
        .collect()
}

fn ensure_event_energy(mut event: WorldEventRecord) -> WorldEventRecord {
    if event.kind == "meteorite" && event.energy_joule.is_none() {
        let r = (event.diameter_km * 1000.0) as f64 / 2.0;
        let mass = (4.0 / 3.0) * std::f64::consts::PI * r.powi(3) * event.density_kg_m3 as f64;
        event.energy_joule = Some(0.5 * mass * (event.speed_kms as f64 * 1000.0).powi(2));
    }
    event
}

#[derive(Clone)]
struct PlateSpec {
    lat: f32,
    lon: f32,
    dir_x: f32,
    dir_y: f32,
    omega_x: f32,
    omega_y: f32,
    omega_z: f32,
    heat: f32,
    buoyancy: f32,
}

#[derive(Clone)]
struct PlateVector {
    omega_x: f32,
    omega_y: f32,
    omega_z: f32,
    heat: f32,
    buoyancy: f32,
}

#[derive(Clone)]
struct ComputePlatesResult {
    plate_field: Vec<i16>,
    boundary_types: Vec<i8>,
    boundary_strength: Vec<f32>,
    plate_vectors: Vec<PlateVector>,
}

#[derive(Clone)]
struct ReliefResult {
    relief: Vec<f32>,
    sea_level: f32,
}

#[derive(Clone, Copy, Debug)]
struct FrontierNode {
    cost: f32,
    index: usize,
    plate: i16,
}

impl PartialEq for FrontierNode {
    fn eq(&self, other: &Self) -> bool {
        self.cost.to_bits() == other.cost.to_bits()
            && self.index == other.index
            && self.plate == other.plate
    }
}

impl Eq for FrontierNode {}

impl PartialOrd for FrontierNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for FrontierNode {
    fn cmp(&self, other: &Self) -> Ordering {
        self.cost
            .total_cmp(&other.cost)
            .then_with(|| self.index.cmp(&other.index))
            .then_with(|| self.plate.cmp(&other.plate))
    }
}

struct MinCostQueue {
    data: BinaryHeap<Reverse<FrontierNode>>,
}

impl MinCostQueue {
    fn new() -> Self {
        Self {
            data: BinaryHeap::new(),
        }
    }

    fn size(&self) -> usize {
        self.data.len()
    }

    fn push(&mut self, item: FrontierNode) {
        self.data.push(Reverse(item));
    }

    fn pop(&mut self) -> Option<FrontierNode> {
        self.data.pop().map(|Reverse(node)| node)
    }
}

#[derive(Clone, Copy)]
struct GrowthParam {
    drift_x: f32,
    drift_y: f32,
    spread: f32,
    roughness: f32,
    freq_a: f32,
    freq_b: f32,
    freq_c: f32,
    freq_d: f32,
    phase_a: f32,
    phase_b: f32,
}

fn lat_lon_to_index(lat: f32, lon: f32) -> usize {
    let y = clampf(
        ((90.0 - lat) / (180.0 / WORLD_HEIGHT as f32)).round(),
        0.0,
        (WORLD_HEIGHT - 1) as f32,
    ) as usize;
    let x = ((((lon + 180.0) / (360.0 / WORLD_WIDTH as f32)).round() as i32 % WORLD_WIDTH as i32)
        + WORLD_WIDTH as i32)
        % WORLD_WIDTH as i32;
    index(x as usize, y)
}

fn nearest_free_index(start: usize, occupied: &[u8]) -> usize {
    if occupied[start] == 0 {
        return start;
    }

    let sx = start % WORLD_WIDTH;
    let sy = start / WORLD_WIDTH;
    let max_radius = WORLD_WIDTH.max(WORLD_HEIGHT) as i32;

    for radius in 1..max_radius {
        for dy in -radius..=radius {
            let y = sy as i32 + dy;
            if y < 0 || y >= WORLD_HEIGHT as i32 {
                continue;
            }
            let span = radius - dy.abs();
            let candidates = [sx as i32 - span, sx as i32 + span];
            for cx in candidates {
                let mut x = cx % WORLD_WIDTH as i32;
                if x < 0 {
                    x += WORLD_WIDTH as i32;
                }
                let idx = index(x as usize, y as usize);
                if occupied[idx] == 0 {
                    return idx;
                }
            }
        }
    }

    start
}

/// Procedural plate Voronoi growth: parameters (spread, roughness, freq,
/// drift_factor, etc.) are tuned for visually realistic plate shapes — not
/// physical properties.  See Bird (2003) for Earth plate shape statistics.
fn build_irregular_plate_field(plates: &[PlateSpec], seed: u32, cache: &WorldCache) -> Vec<i16> {
    let mut plate_field = vec![-1_i16; WORLD_SIZE];
    let mut open_cost = vec![f32::INFINITY; WORLD_SIZE];
    let mut occupied_seeds = vec![0_u8; WORLD_SIZE];
    let mut growth_rng = Rng::new(seed ^ 0x9e37_79b9);
    let mut queue = MinCostQueue::new();

    let plate_size_bias: Vec<f32> = (0..plates.len())
        .map(|_| {
            let a = random_range(&mut growth_rng, 0.7, 1.55);
            let b = random_range(&mut growth_rng, 0.76, 1.34);
            clampf((a * b).powf(0.74), 0.55, 1.95)
        })
        .collect();

    let growth_params: Vec<GrowthParam> = plates
        .iter()
        .enumerate()
        .map(|(pid, plate)| {
            let len = (plate.dir_x.hypot(plate.dir_y)).max(1.0);
            let size_bias = plate_size_bias[pid];
            GrowthParam {
                drift_x: plate.dir_x / len,
                drift_y: plate.dir_y / len,
                spread: random_range(&mut growth_rng, 0.82, 1.22) / size_bias.powf(0.36),
                roughness: random_range(&mut growth_rng, 0.26, 1.08) * (0.88 + 0.24 * size_bias),
                freq_a: random_range(&mut growth_rng, 0.045, 0.145),
                freq_b: random_range(&mut growth_rng, 0.055, 0.16),
                freq_c: random_range(&mut growth_rng, 0.08, 0.22),
                freq_d: random_range(&mut growth_rng, 0.07, 0.2),
                phase_a: random_range(&mut growth_rng, -std::f32::consts::PI, std::f32::consts::PI),
                phase_b: random_range(&mut growth_rng, -std::f32::consts::PI, std::f32::consts::PI),
            }
        })
        .collect();

    let mut structural_field = vec![0.0_f32; WORLD_SIZE];
    let seed_phase = seed as f32 * 0.00131;
    for i in 0..WORLD_SIZE {
        let sx = cache.x_by_cell[i];
        let sy = cache.y_by_cell[i];
        let sz = cache.z_by_cell[i];
        let low = spherical_fbm(
            sx * 0.95 + 9.0,
            sy * 0.95 - 7.0,
            sz * 0.95 + 5.0,
            seed_phase + 13.0,
        );
        let mid = spherical_fbm(
            sx * 2.15 - 17.0,
            sy * 2.15 + 19.0,
            sz * 2.15 - 11.0,
            seed_phase + 41.0,
        );
        structural_field[i] = clampf(0.5 + 0.5 * (0.62 * low + 0.38 * mid), 0.0, 1.0);
    }

    for (plate_id, plate) in plates.iter().enumerate() {
        let size_bias = plate_size_bias[plate_id];
        let seed_index =
            nearest_free_index(lat_lon_to_index(plate.lat, plate.lon), &occupied_seeds);
        occupied_seeds[seed_index] = 1;
        open_cost[seed_index] = 0.0;
        queue.push(FrontierNode {
            cost: 0.0,
            index: seed_index,
            plate: plate_id as i16,
        });

        // Additional nuclei per plate break radial Voronoi-like slicing and create
        // more natural non-convex macro-plate shapes.
        let nuclei = (2 + (growth_rng.next_f32() * (2.6 + size_bias * 1.8)).floor() as i32)
            .clamp(2, 7) as usize;
        let drift_len = plate.dir_x.hypot(plate.dir_y).max(1e-5);
        let drift_x = plate.dir_x / drift_len;
        let drift_y = plate.dir_y / drift_len;
        let perp_x = -drift_y;
        let perp_y = drift_x;
        let sx = seed_index % WORLD_WIDTH;
        let sy = seed_index / WORLD_WIDTH;
        for _ in 0..nuclei {
            let along = random_range(&mut growth_rng, 24.0, 190.0)
                * if growth_rng.next_f32() < 0.5 { -1.0 } else { 1.0 };
            let across = random_range(&mut growth_rng, -78.0, 78.0);
            let tx = sx as f32 + drift_x * along + perp_x * across;
            let ty = sy as f32 + drift_y * along + perp_y * across;
            let nucleus = nearest_free_index(
                index_spherical(tx.round() as i32, ty.round() as i32),
                &occupied_seeds,
            );
            occupied_seeds[nucleus] = 1;
            let start_cost = random_range(&mut growth_rng, 0.1, 2.8);
            if start_cost + 1e-6 < open_cost[nucleus] {
                open_cost[nucleus] = start_cost;
                queue.push(FrontierNode {
                    cost: start_cost,
                    index: nucleus,
                    plate: plate_id as i16,
                });
            }
        }

        // Historical nuclei: emulate long-lived plate migration and fragmentation.
        let mut path_x = sx as f32;
        let mut path_y = sy as f32;
        let mut path_dx = drift_x;
        let mut path_dy = drift_y;
        let mut path_perp_x = -path_dy;
        let mut path_perp_y = path_dx;
        let history_steps = (6 + (growth_rng.next_f32() * (8.0 + size_bias * 4.0)).floor() as i32)
            .clamp(6, 18) as usize;
        for step in 0..history_steps {
            let bend = random_range(&mut growth_rng, -0.34, 0.34)
                + ((step as f32 * 0.57 + seed_phase * 7.0 + plate_id as f32).sin() * 0.16);
            let ndx = path_dx + path_perp_x * bend;
            let ndy = path_dy + path_perp_y * bend;
            let nlen = ndx.hypot(ndy).max(1e-5);
            path_dx = ndx / nlen;
            path_dy = ndy / nlen;
            path_perp_x = -path_dy;
            path_perp_y = path_dx;

            let step_len = random_range(&mut growth_rng, 22.0, 86.0)
                * (0.86 + 0.26 * size_bias)
                * (0.9 + 0.22 * growth_rng.next_f32());
            path_x += path_dx * step_len;
            path_y += path_dy * step_len;

            let branch_count = if growth_rng.next_f32() < 0.62 { 2 } else { 3 };
            for _ in 0..branch_count {
                let across = random_range(&mut growth_rng, -104.0, 104.0)
                    * (0.35 + 0.65 * growth_rng.next_f32());
                let along_jitter = random_range(&mut growth_rng, -20.0, 20.0);
                let tx = path_x + path_dx * along_jitter + path_perp_x * across;
                let ty = path_y + path_dy * along_jitter + path_perp_y * across;
                let nucleus = nearest_free_index(
                    index_spherical(tx.round() as i32, ty.round() as i32),
                    &occupied_seeds,
                );
                occupied_seeds[nucleus] = 1;
                let start_cost = random_range(&mut growth_rng, 0.18, 3.4)
                    * (0.84 + 0.36 * (1.0 - size_bias * 0.35).max(0.35));
                if start_cost + 1e-6 < open_cost[nucleus] {
                    open_cost[nucleus] = start_cost;
                    queue.push(FrontierNode {
                        cost: start_cost,
                        index: nucleus,
                        plate: plate_id as i16,
                    });
                }
            }
        }
    }

    const STEPS: [(i32, i32, f32); 8] = [
        (1, 0, 1.0),
        (-1, 0, 1.0),
        (0, 1, 1.0),
        (0, -1, 1.0),
        (1, 1, std::f32::consts::SQRT_2),
        (-1, 1, std::f32::consts::SQRT_2),
        (1, -1, std::f32::consts::SQRT_2),
        (-1, -1, std::f32::consts::SQRT_2),
    ];

    // Precompute cos(lat) per row for latitude-corrected step distances.
    // On an equirectangular grid, E-W physical distance = cos(φ) × cell_size.
    let mut cos_lat_row = vec![0.0_f32; WORLD_HEIGHT];
    for row in 0..WORLD_HEIGHT {
        let lat_rad = (90.0 - (row as f32 + 0.5) * (180.0 / WORLD_HEIGHT as f32)) * RADIANS;
        cos_lat_row[row] = lat_rad.cos().abs().max(0.087);
    }

    let mut assigned = 0usize;
    while queue.size() > 0 && assigned < WORLD_SIZE {
        let Some(node) = queue.pop() else {
            break;
        };

        if node.cost > open_cost[node.index] + 1e-6 {
            continue;
        }
        if plate_field[node.index] != -1 {
            continue;
        }

        plate_field[node.index] = node.plate;
        assigned += 1;

        let x = node.index % WORLD_WIDTH;
        let y = node.index / WORLD_WIDTH;
        let gp = growth_params[node.plate as usize];

        for (dx, dy, _w) in STEPS {
            let j = index_spherical(x as i32 + dx, y as i32 + dy);
            if plate_field[j] != -1 {
                continue;
            }

            let lat = cache.lat_by_y[j];
            let sx = cache.x_by_cell[j];
            let sy = cache.y_by_cell[j];
            let sz = cache.z_by_cell[j];
            let scale_a = 0.8 + gp.freq_a * 24.0;
            let scale_b = 0.8 + gp.freq_b * 24.0;
            let scale_c = 0.8 + gp.freq_c * 24.0;
            let scale_d = 0.8 + gp.freq_d * 24.0;
            let wave_a = (sx * scale_a + sy * scale_b + gp.phase_a).sin();
            let wave_b = (sy * scale_c - sz * scale_d + gp.phase_b).cos();
            let rough_factor = 1.0 + gp.roughness * (0.22 * wave_a + 0.18 * wave_b);
            // Latitude-corrected drift alignment: dx in physical space is dx × cos(φ)
            let cl = cos_lat_row[y];
            let drift_align = dx as f32 * cl * gp.drift_x + dy as f32 * gp.drift_y;
            let drift_factor = 1.03 - 0.12 * drift_align;
            let structural = structural_field[j];
            let structure_factor = clampf(
                0.62 + 0.98 * structural + (0.36 - gp.roughness * 0.17),
                0.45,
                1.9,
            );
            let polar_factor = 1.0 + (lat.abs() / 90.0) * 0.1;
            // Latitude-corrected physical step distance:
            // E-W = cos(φ), N-S = 1.0, diagonal = √(cos²φ + 1)
            let phys_w = if dy == 0 { cl } else if dx == 0 { 1.0 }
                         else { (cl * cl + 1.0).sqrt() };
            let step_cost =
                (phys_w * gp.spread * rough_factor * drift_factor * structure_factor * polar_factor)
                    .max(0.08);
            let next_cost = node.cost + step_cost;

            if next_cost + 1e-6 < open_cost[j] {
                open_cost[j] = next_cost;
                queue.push(FrontierNode {
                    cost: next_cost,
                    index: j,
                    plate: node.plate,
                });
            }
        }
    }

    if assigned < WORLD_SIZE {
        for i in 0..WORLD_SIZE {
            if plate_field[i] != -1 {
                continue;
            }
            let mut best_plate = 0usize;
            let mut best_cost = f32::INFINITY;
            let cell_x = cache.x_by_cell[i];
            let cell_y = cache.y_by_cell[i];
            let cell_z = cache.z_by_cell[i];

            for (p, plate) in plates.iter().enumerate() {
                let lat_r = plate.lat * RADIANS;
                let lon_r = plate.lon * RADIANS;
                let c = lat_r.cos();
                let px = c * lon_r.cos();
                let py = c * lon_r.sin();
                let pz = lat_r.sin();
                let d = 1.0 - (cell_x * px + cell_y * py + cell_z * pz);
                if d < best_cost {
                    best_cost = d;
                    best_plate = p;
                }
            }

            plate_field[i] = best_plate as i16;
        }
    }

    plate_field
}

fn cleanup_plate_fragments(plate_field: &mut [i16], min_component_cells: usize) {
    if min_component_cells <= 1 {
        return;
    }
    let max_plate = plate_field
        .iter()
        .copied()
        .max()
        .unwrap_or(0)
        .max(0) as usize
        + 1;
    if max_plate <= 1 {
        return;
    }

    let mut visited = vec![0_u8; WORLD_SIZE];
    let mut queue: VecDeque<usize> = VecDeque::new();
    let mut component: Vec<usize> = Vec::new();
    let mut neighbor_counts = vec![0_usize; max_plate];
    let mut touched_neighbors: Vec<usize> = Vec::new();

    for start in 0..WORLD_SIZE {
        if visited[start] != 0 {
            continue;
        }
        visited[start] = 1;
        queue.push_back(start);
        component.clear();
        touched_neighbors.clear();
        let pid = plate_field[start];

        while let Some(i) = queue.pop_front() {
            component.push(i);
            let x = i % WORLD_WIDTH;
            let y = i / WORLD_WIDTH;
            for oy in -1..=1 {
                for ox in -1..=1 {
                    if ox == 0 && oy == 0 {
                        continue;
                    }
                    let j = index_spherical(x as i32 + ox, y as i32 + oy);
                    let qid = plate_field[j];
                    if qid == pid {
                        if visited[j] == 0 {
                            visited[j] = 1;
                            queue.push_back(j);
                        }
                    } else if qid >= 0 {
                        let q = qid as usize;
                        if neighbor_counts[q] == 0 {
                            touched_neighbors.push(q);
                        }
                        neighbor_counts[q] += 1;
                    }
                }
            }
        }

        if component.len() < min_component_cells && !touched_neighbors.is_empty() {
            let mut best_plate = touched_neighbors[0];
            let mut best_count = neighbor_counts[best_plate];
            for &q in touched_neighbors.iter().skip(1) {
                if neighbor_counts[q] > best_count {
                    best_count = neighbor_counts[q];
                    best_plate = q;
                }
            }
            for &i in component.iter() {
                plate_field[i] = best_plate as i16;
            }
        }

        for &q in touched_neighbors.iter() {
            neighbor_counts[q] = 0;
        }
    }
}

fn cleanup_plate_fragments_relative(
    plate_field: &mut [i16],
    keep_ratio_of_largest: f32,
    min_component_cells: usize,
) {
    let max_plate = plate_field
        .iter()
        .copied()
        .max()
        .unwrap_or(0)
        .max(0) as usize
        + 1;
    if max_plate <= 1 {
        return;
    }

    let mut largest = vec![0_usize; max_plate];
    let mut visited = vec![0_u8; WORLD_SIZE];
    let mut queue: VecDeque<usize> = VecDeque::new();

    for start in 0..WORLD_SIZE {
        if visited[start] != 0 {
            continue;
        }
        visited[start] = 1;
        queue.push_back(start);
        let pid = plate_field[start];
        let mut size = 0_usize;
        while let Some(i) = queue.pop_front() {
            size += 1;
            let x = i % WORLD_WIDTH;
            let y = i / WORLD_WIDTH;
            for oy in -1..=1 {
                for ox in -1..=1 {
                    if ox == 0 && oy == 0 {
                        continue;
                    }
                    let j = index_spherical(x as i32 + ox, y as i32 + oy);
                    if visited[j] == 0 && plate_field[j] == pid {
                        visited[j] = 1;
                        queue.push_back(j);
                    }
                }
            }
        }
        if pid >= 0 {
            let p = pid as usize;
            if size > largest[p] {
                largest[p] = size;
            }
        }
    }

    visited.fill(0);
    let mut component: Vec<usize> = Vec::new();
    let mut neighbor_counts = vec![0_usize; max_plate];
    let mut touched_neighbors: Vec<usize> = Vec::new();
    let ratio = clampf(keep_ratio_of_largest, 0.0, 1.0);

    for start in 0..WORLD_SIZE {
        if visited[start] != 0 {
            continue;
        }
        visited[start] = 1;
        queue.push_back(start);
        component.clear();
        touched_neighbors.clear();
        let pid = plate_field[start];

        while let Some(i) = queue.pop_front() {
            component.push(i);
            let x = i % WORLD_WIDTH;
            let y = i / WORLD_WIDTH;
            for oy in -1..=1 {
                for ox in -1..=1 {
                    if ox == 0 && oy == 0 {
                        continue;
                    }
                    let j = index_spherical(x as i32 + ox, y as i32 + oy);
                    let qid = plate_field[j];
                    if qid == pid {
                        if visited[j] == 0 {
                            visited[j] = 1;
                            queue.push_back(j);
                        }
                    } else if qid >= 0 {
                        let q = qid as usize;
                        if neighbor_counts[q] == 0 {
                            touched_neighbors.push(q);
                        }
                        neighbor_counts[q] += 1;
                    }
                }
            }
        }

        let mut should_reassign = false;
        if pid >= 0 {
            let p = pid as usize;
            let limit = (largest[p] as f32 * ratio).round() as usize;
            let min_keep = min_component_cells.max(limit);
            if component.len() < min_keep {
                should_reassign = true;
            }
        }

        if should_reassign && !touched_neighbors.is_empty() {
            let mut best_plate = touched_neighbors[0];
            let mut best_count = neighbor_counts[best_plate];
            for &q in touched_neighbors.iter().skip(1) {
                if neighbor_counts[q] > best_count {
                    best_count = neighbor_counts[q];
                    best_plate = q;
                }
            }
            for &i in component.iter() {
                plate_field[i] = best_plate as i16;
            }
        }

        for &q in touched_neighbors.iter() {
            neighbor_counts[q] = 0;
        }
    }
}

#[inline]
fn accumulate_plate_vote(
    candidate: i16,
    contribution: f32,
    best_label: &mut i16,
    best_score: &mut f32,
    second_label: &mut i16,
    second_score: &mut f32,
) {
    if contribution <= 0.0 {
        return;
    }
    if *best_label == candidate {
        *best_score += contribution;
        return;
    }
    if *second_label == candidate {
        *second_score += contribution;
        if *second_score > *best_score {
            std::mem::swap(best_label, second_label);
            std::mem::swap(best_score, second_score);
        }
        return;
    }
    if contribution > *best_score {
        *second_label = *best_label;
        *second_score = *best_score;
        *best_label = candidate;
        *best_score = contribution;
    } else if contribution > *second_score {
        *second_label = candidate;
        *second_score = contribution;
    }
}

fn evolve_plate_field(
    plate_field: &mut [i16],
    plate_vectors: &[PlateVector],
    planet: &PlanetInputs,
    detail: DetailProfile,
    seed: u32,
    cache: &WorldCache,
    progress: &mut ProgressTap<'_>,
    progress_base: f32,
    progress_span: f32,
) {
    let steps = detail.plate_evolution_steps.max(1);
    let plate_count = plate_vectors.len().max(1);
    let radius_cm = (planet.radius_km.max(1000.0) * 100_000.0).max(1.0);
    let cm_per_lat_cell = (std::f32::consts::PI * radius_cm / WORLD_HEIGHT as f32).max(1.0);
    let cm_per_lon_eq_cell = (2.0 * std::f32::consts::PI * radius_cm / WORLD_WIDTH as f32).max(1.0);
    let mut cm_per_lon_by_y = vec![0.0_f32; WORLD_HEIGHT];
    for y in 0..WORLD_HEIGHT {
        let lat_deg = 90.0 - (y as f32 + 0.5) * (180.0 / WORLD_HEIGHT as f32);
        let cos_lat = (lat_deg * RADIANS).cos().abs();
        cm_per_lon_by_y[y] = (cm_per_lon_eq_cell * cos_lat).max(cm_per_lat_cell * 0.2);
    }

    let mut rng = Rng::new(seed ^ 0xC2B2_AE35);
    let mut next = vec![-1_i16; WORLD_SIZE];
    let mut best_label = vec![-1_i16; WORLD_SIZE];
    let mut second_label = vec![-1_i16; WORLD_SIZE];
    let mut best_score = vec![0.0_f32; WORLD_SIZE];
    let mut second_score = vec![0.0_f32; WORLD_SIZE];
    let mut relaxed = vec![0_i16; WORLD_SIZE];

    for step in 0..steps {
        best_label.fill(-1);
        second_label.fill(-1);
        best_score.fill(0.0);
        second_score.fill(0.0);

        let age_norm = step as f32 / steps.max(1) as f32;
        // Plate reorganization timescale: 0.9-3.4 Myr per step, shorter early
        // on (Torsvik et al. 2010: reorganizations every 5-20 Myr; we sub-step).
        let step_years =
            random_range(&mut rng, 900_000.0, 3_400_000.0) * (0.92 + 0.32 * age_norm);
        // Procedural: plate boundary inertia (higher = more stable boundaries).
        let memory_keep = 0.5 + 0.18 * (1.0 - age_norm);
        let structural_phase = seed as f32 * 0.0019 + step as f32 * 1.71;

        for i in 0..WORLD_SIZE {
            let pid = plate_field[i];
            if pid < 0 {
                continue;
            }
            let p = pid as usize;
            let y = i / WORLD_WIDTH;
            let x = i % WORLD_WIDTH;
            let sx = cache.x_by_cell[i];
            let sy = cache.y_by_cell[i];
            let sz = cache.z_by_cell[i];
            let (vx, vy) = plate_velocity_xy_at_cell(&plate_vectors[p], sx, sy, sz, radius_cm);
            let dx_cells = vx * step_years / cm_per_lon_by_y[y];
            let dy_cells = vy * step_years / cm_per_lat_cell;
            let fx = x as f32 + dx_cells;
            let fy = y as f32 + dy_cells;
            let x0 = fx.floor();
            let y0 = fy.floor();
            let tx = fx - x0;
            let ty = fy - y0;
            let x0i = x0 as i32;
            let y0i = y0 as i32;

            let weights = [
                (0, 0, (1.0 - tx) * (1.0 - ty)),
                (1, 0, tx * (1.0 - ty)),
                (0, 1, (1.0 - tx) * ty),
                (1, 1, tx * ty),
            ];

            for (ox, oy, w) in weights {
                if w <= 1e-6 {
                    continue;
                }
                let j = index_spherical(x0i + ox, y0i + oy);
                let mut s = w;
                if plate_field[j] == pid {
                    s *= memory_keep;
                } else {
                    s *= 0.92;
                }
                let sxj = cache.x_by_cell[j];
                let syj = cache.y_by_cell[j];
                let szj = cache.z_by_cell[j];
                let structural = 0.5
                    + 0.5
                        * ((sxj * 4.6 + syj * 3.8 + szj * 2.9 + structural_phase).sin() * 0.6
                            + (syj * 5.1 - szj * 3.6 + sxj * 2.4 - structural_phase * 1.3).cos()
                                * 0.4);
                s *= clampf(0.82 + 0.4 * structural, 0.62, 1.28);
                accumulate_plate_vote(
                    pid,
                    s,
                    &mut best_label[j],
                    &mut best_score[j],
                    &mut second_label[j],
                    &mut second_score[j],
                );
            }
        }

        for i in 0..WORLD_SIZE {
            let old = plate_field[i];
            let mut chosen = best_label[i];
            if chosen < 0 {
                chosen = old;
            } else {
                let b = best_score[i];
                let s = second_score[i];
                if s > 1e-5 && (b - s).abs() <= 0.08 {
                    if old == chosen || old == second_label[i] {
                        chosen = old;
                    } else if second_label[i] >= 0 {
                        chosen = second_label[i];
                    }
                }
            }
            next[i] = if chosen >= 0 { chosen } else { 0 };
        }
        plate_field.copy_from_slice(&next);

        let relax_passes = 1 + (detail.max_kernel_radius as usize / 5).min(1);
        for _ in 0..relax_passes {
            for y in 0..WORLD_HEIGHT {
                for x in 0..WORLD_WIDTH {
                    let i = index(x, y);
                    let pid = plate_field[i];
                    let mut labels = [-1_i16; 9];
                    let mut counts = [0_u8; 9];
                    let mut used = 0_usize;

                    for oy in -1..=1 {
                        for ox in -1..=1 {
                            if ox == 0 && oy == 0 {
                                continue;
                            }
                            let q = plate_field[index_spherical(x as i32 + ox, y as i32 + oy)];
                            let mut found = false;
                            for k in 0..used {
                                if labels[k] == q {
                                    counts[k] = counts[k].saturating_add(1);
                                    found = true;
                                    break;
                                }
                            }
                            if !found && used < labels.len() {
                                labels[used] = q;
                                counts[used] = 1;
                                used += 1;
                            }
                        }
                    }

                    let mut own_count = 0_u8;
                    let mut best_pid = pid;
                    let mut best_count = 0_u8;
                    for k in 0..used {
                        if labels[k] == pid {
                            own_count = counts[k];
                        }
                        if counts[k] > best_count {
                            best_count = counts[k];
                            best_pid = labels[k];
                        }
                    }

                    if best_pid >= 0 && best_pid != pid && best_count >= 5 && own_count <= 2 {
                        relaxed[i] = best_pid;
                    } else {
                        relaxed[i] = pid;
                    }
                }
            }
            plate_field.copy_from_slice(&relaxed);
        }

        let min_abs = clampf(
            WORLD_SIZE as f32 / (plate_count as f32 * 640.0),
            140.0,
            960.0,
        ) as usize;
        cleanup_plate_fragments(plate_field, min_abs);
        cleanup_plate_fragments_relative(plate_field, 0.38, min_abs * 4);

        progress.phase(
            progress_base,
            progress_span,
            (step as f32 + 1.0) / steps as f32,
        );
    }
}

fn compute_plates(
    planet: &PlanetInputs,
    tectonics: &TectonicInputs,
    detail: DetailProfile,
    seed: u32,
    cache: &WorldCache,
    progress: &mut ProgressTap<'_>,
    progress_base: f32,
    progress_span: f32,
) -> ComputePlatesResult {
    let plate_count = tectonics.plate_count.clamp(2, 20) as usize;
    let mut rng = Rng::new(seed.wrapping_add((plate_count as u32).wrapping_mul(7_919)));
    let mut plates: Vec<PlateSpec> = Vec::with_capacity(plate_count);
    let radius_cm = (planet.radius_km.max(1000.0) * 100_000.0).max(1.0);

    for _ in 0..plate_count {
        let lat = random_range(&mut rng, -90.0, 90.0);
        let lon = random_range(&mut rng, -180.0, 180.0);
        let speed =
            (random_range(&mut rng, 0.5, 1.5) * tectonics.plate_speed_cm_per_year).max(0.001);

        let pole_z = random_range(&mut rng, -1.0, 1.0);
        let pole_lon = random_range(&mut rng, -std::f32::consts::PI, std::f32::consts::PI);
        let pole_r = (1.0 - pole_z * pole_z).max(0.0).sqrt();
        let pole_x = pole_r * pole_lon.cos();
        let pole_y = pole_r * pole_lon.sin();
        let spin_sign = if rng.next_f32() < 0.5 { -1.0 } else { 1.0 };
        let omega_mag = (speed / radius_cm) * random_range(&mut rng, 0.82, 1.24);
        let omega_x = pole_x * omega_mag * spin_sign;
        let omega_y = pole_y * omega_mag * spin_sign;
        let omega_z = pole_z * omega_mag * spin_sign;

        let lat_r = lat * RADIANS;
        let lon_r = lon * RADIANS;
        let cos_lat = lat_r.cos();
        let sx = cos_lat * lon_r.cos();
        let sy = cos_lat * lon_r.sin();
        let sz = lat_r.sin();
        let (mut drift_x, mut drift_y) =
            plate_velocity_xy_from_omega(omega_x, omega_y, omega_z, sx, sy, sz, radius_cm);
        let drift_speed = drift_x.hypot(drift_y);
        if drift_speed < 0.05 {
            let dir = random_range(&mut rng, 0.0, std::f32::consts::PI * 2.0);
            drift_x = dir.cos() * speed;
            drift_y = dir.sin() * speed;
        } else {
            let scale = speed / drift_speed.max(1e-4);
            drift_x *= scale;
            drift_y *= scale;
        }

        plates.push(PlateSpec {
            lat,
            lon,
            dir_x: drift_x,
            dir_y: drift_y,
            omega_x,
            omega_y,
            omega_z,
            heat: random_range(
                &mut rng,
                (tectonics.mantle_heat * 0.5).max(1.0),
                tectonics.mantle_heat * 1.5,
            ),
            buoyancy: random_range(&mut rng, -1.0, 1.0),
        });
    }

    let plate_vectors: Vec<PlateVector> = plates
        .iter()
        .map(|plate| PlateVector {
            omega_x: plate.omega_x,
            omega_y: plate.omega_y,
            omega_z: plate.omega_z,
            heat: plate.heat,
            buoyancy: plate.buoyancy,
        })
        .collect();

    let mut plate_field = build_irregular_plate_field(&plates, seed, cache);
    let min_fragment_cells = clampf(
        WORLD_SIZE as f32 / (plate_count as f32 * 420.0),
        240.0,
        1800.0,
    ) as usize;
    cleanup_plate_fragments(&mut plate_field, min_fragment_cells);
    cleanup_plate_fragments_relative(&mut plate_field, 0.42, min_fragment_cells * 5);
    evolve_plate_field(
        &mut plate_field,
        &plate_vectors,
        planet,
        detail,
        seed ^ 0x85EB_CA6B,
        cache,
        progress,
        progress_base,
        progress_span * 0.45,
    );
    cleanup_plate_fragments(&mut plate_field, min_fragment_cells);
    cleanup_plate_fragments_relative(&mut plate_field, 0.5, min_fragment_cells * 6);

    let mut boundary_types = vec![0_i8; WORLD_SIZE];
    let mut boundary_strength = vec![0.0_f32; WORLD_SIZE];
    // Normalization: boundary_strength ∈ [0,1] for typical Earth plate
    // speeds 2-8 cm/yr (DeMets et al. 2010).  1.25 factor and 1.2 floor
    // ensure most boundaries stay within range.
    let boundary_scale = (tectonics.plate_speed_cm_per_year * 1.25).max(1.2);

    const NEIGHBORS: [(i32, i32, f32); 8] = [
        (1, 0, 1.0),
        (-1, 0, 1.0),
        (0, -1, 1.0),
        (0, 1, 1.0),
        (1, -1, std::f32::consts::FRAC_1_SQRT_2),
        (-1, -1, std::f32::consts::FRAC_1_SQRT_2),
        (1, 1, std::f32::consts::FRAC_1_SQRT_2),
        (-1, 1, std::f32::consts::FRAC_1_SQRT_2),
    ];

    // Precompute cos(lat) for boundary detection latitude correction.
    let mut bd_cos_lat = vec![0.0_f32; WORLD_HEIGHT];
    for row in 0..WORLD_HEIGHT {
        let lat_rad = (90.0 - (row as f32 + 0.5) * (180.0 / WORLD_HEIGHT as f32)) * RADIANS;
        bd_cos_lat[row] = lat_rad.cos().abs().max(0.087);
    }

    for y in 0..WORLD_HEIGHT {
        progress.phase(
            progress_base + progress_span * 0.45,
            progress_span * 0.55,
            y as f32 / WORLD_HEIGHT as f32,
        );
        let cl = bd_cos_lat[y];
        for x in 0..WORLD_WIDTH {
            let i = index(x, y);
            let plate_a = plate_field[i] as usize;
            let a = &plate_vectors[plate_a];
            let sx = cache.x_by_cell[i];
            let sy = cache.y_by_cell[i];
            let sz = cache.z_by_cell[i];
            let (vax, vay) = plate_velocity_xy_at_cell(a, sx, sy, sz, radius_cm);

            let mut conv_sum = 0.0_f32;
            let mut div_sum = 0.0_f32;
            let mut shear_sum = 0.0_f32;
            let mut wsum = 0.0_f32;
            let mut normal_x = 0.0_f32;
            let mut normal_y = 0.0_f32;
            let mut has_different_neighbor = false;

            for (dx, dy, w) in NEIGHBORS {
                let j = index_spherical(x as i32 + dx, y as i32 + dy);
                let plate_b = plate_field[j] as usize;
                if plate_b == plate_a {
                    continue;
                }
                has_different_neighbor = true;
                let b = &plate_vectors[plate_b];
                let (vbx, vby) = plate_velocity_xy_at_cell(b, sx, sy, sz, radius_cm);
                let rel_x = vbx - vax;
                let rel_y = vby - vay;
                // Physical-space normal: dx scaled by cos(φ) for equirectangular correction.
                let phys_dx = dx as f32 * cl;
                let phys_dy = dy as f32;
                let n_len = phys_dx.hypot(phys_dy).max(0.01);
                let nx = phys_dx / n_len;
                let ny = phys_dy / n_len;
                let tx = -ny;
                let ty = nx;
                let vn = rel_x * nx + rel_y * ny;
                let vt = rel_x * tx + rel_y * ty;
                let conv = (-vn).max(0.0);
                let div = vn.max(0.0);
                conv_sum += conv * w;
                div_sum += div * w;
                shear_sum += vt.abs() * w;
                normal_x += nx * conv.max(div) * w;
                normal_y += ny * conv.max(div) * w;
                wsum += w;
            }

            if !has_different_neighbor {
                boundary_types[i] = 0;
                boundary_strength[i] = 0.0;
                continue;
            }

            let denom = wsum.max(1e-6);
            let conv = conv_sum / denom;
            let div = div_sum / denom;
            let shear = shear_sum / denom;

            let n_len = normal_x.hypot(normal_y);
            if n_len > 1e-5 {
            } else {
            }

            // Shear weight 0.82: transform boundaries have lower topographic
            // expression than convergent/divergent (Bird 2003).
            let strength = conv.max(div).max(shear * 0.82);
            boundary_strength[i] = clampf(strength / boundary_scale, 0.0, 1.0);

            // Classification thresholds tuned to produce Earth-like proportions:
            // ~50% convergent, ~30% divergent, ~20% transform (Bird 2003).
            let conv_thresh = boundary_scale * 0.14;
            let div_thresh = boundary_scale * 0.14;
            let shear_thresh = boundary_scale * 0.18;
            boundary_types[i] = if conv > conv_thresh && conv >= div * 0.95 && conv >= shear * 0.75
            {
                1
            } else if div > div_thresh && div >= conv * 0.95 && div >= shear * 0.75 {
                2
            } else if shear > shear_thresh {
                3
            } else if conv >= div {
                1
            } else {
                2
            };
        }
    }
    progress.phase(progress_base, progress_span, 1.0);

    ComputePlatesResult {
        plate_field,
        boundary_types,
        boundary_strength,
        plate_vectors,
    }
}

// ---------------------------------------------------------------------------
// Airy isostasy with water loading: Turcotte & Schubert (2002) §2.2, §2.6
// ---------------------------------------------------------------------------
//
// Isostatic balance: equal pressure at compensation depth D for all columns.
// For a column of crust C (km) and density ρ_c on mantle ρ_m:
//
//   Continental:  h = C × (ρ_m − ρ_c) / ρ_m
//   Oceanic:      h = C × (ρ_m − ρ_c) / (ρ_m − ρ_w)   (Turcotte & Schubert §2.6)
//
// Water loading (ρ_w = 1025 kg/m³) amplifies oceanic basin depth by the factor
// ρ_m / (ρ_m − ρ_w) = 3300/2275 ≈ 1.45.  This produces the ~5 km ocean depth
// from 7 km oceanic crust, matching observed bathymetry (Parsons & Sclater 1977).
//
// Without water loading, freeboard is ~5.2 km (too high); with loading, it
// naturally drops to ~1.5 km before sea-level determination.

/// Compute isostatic surface height (m) from crustal column buoyancy.
/// `continental_frac` is 0.0 (oceanic) to 1.0 (continental), allowing
/// smooth continent-ocean transitions (continental shelf).
fn isostatic_elevation(crust_km: f32, continental_frac: f32, heat_anomaly: f32) -> f32 {
    // Density interpolation: oceanic basalt 2900, continental granite 2800.
    let rho_c: f32 = 2900.0 - continental_frac * 100.0;
    let rho_m: f32 = 3300.0;
    let rho_w: f32 = 1025.0;

    // Thermal expansion: α ≈ 3×10⁻⁵ /K (Turcotte & Schubert 2002 §4.3).
    // Hot anomaly ~300 K → Δρ/ρ = α·ΔT ≈ 1%.  Using 2% as upper bound
    // (includes partial melt effects in hot orogenic roots).
    let thermal_correction = 1.0 - heat_anomaly.clamp(0.0, 1.0) * 0.02;
    let rho_c_eff = rho_c * thermal_correction;

    // Turcotte & Schubert (2002):
    //   §2.2 eq 2.4 (continental): h = C × (ρ_m − ρ_c) / ρ_m
    //   §2.6 (oceanic with water):  h = C × (ρ_m − ρ_c) / (ρ_m − ρ_w)
    // Smooth blend via continental_frac for shelf/margin transitions.
    let denom = rho_m - (1.0 - continental_frac) * rho_w;
    crust_km * 1000.0 * (rho_m - rho_c_eff) / denom
}

// ---------------------------------------------------------------------------
// Deformation propagation: exponential decay from plate boundaries
// ---------------------------------------------------------------------------
//
// England & McKenzie 1982 (EPSL): continental deformation in collision zones
// decays exponentially from the plate boundary with characteristic length
// L_d ≈ 200–500 km.  This function propagates a boundary-only seed field
// outward using iterative max-dilation with per-step exponential decay,
// equivalent to solving the eikonal equation with multiplicative cost.
// Peak amplitude is preserved (unlike diffusive smoothing).

/// Propagate boundary deformation field with exponential distance decay.
/// Latitude-corrected: E-W physical step = cos(φ) × cell_size, so the decay
/// per E-W cell step is exp(-cos(φ)/L_d) — less decay at high latitudes
/// because the physical step is shorter (England & McKenzie 1982).
fn propagate_deformation(
    seed: &[f32],
    decay_km: f32,
    grid: &GridConfig,
) -> Vec<f32> {
    let l_cells = decay_km / grid.km_per_cell_x; // L_d in equatorial cells
    // N-S decay is constant (physical step = km_per_cell_y ≈ km_per_cell_x)
    let decay_ns = (-1.0_f32 / l_cells).exp();

    let mut field = seed.to_vec();
    let mut scratch = vec![0.0_f32; grid.size];
    let max_passes = (3.0 * l_cells).ceil() as usize + 5;

    for _ in 0..max_passes {
        let mut changed = false;
        for y in 0..grid.height {
            let cos_lat = grid.cos_lat_by_y[y];
            // E-W physical step = cos(φ) cells → decay = exp(-cos(φ)/L_d)
            let decay_ew = (-cos_lat / l_cells).exp();
            // Diagonal physical step = √(cos²φ + 1) cells
            let decay_diag = (-(cos_lat * cos_lat + 1.0).sqrt() / l_cells).exp();
            for x in 0..grid.width {
                let i = grid.index(x, y);
                let mut val = field[i];
                // N-S cardinal
                for dy in [-1_i32, 1] {
                    let j = grid.neighbor(x as i32, y as i32 + dy);
                    let proposal = field[j] * decay_ns;
                    if proposal > val { val = proposal; changed = true; }
                }
                // E-W cardinal
                for dx in [-1_i32, 1] {
                    let j = grid.neighbor(x as i32 + dx, y as i32);
                    let proposal = field[j] * decay_ew;
                    if proposal > val { val = proposal; changed = true; }
                }
                // Diagonals
                for (dx, dy) in [(-1_i32, -1_i32), (-1, 1), (1, -1), (1, 1)] {
                    let j = grid.neighbor(x as i32 + dx, y as i32 + dy);
                    let proposal = field[j] * decay_diag;
                    if proposal > val { val = proposal; changed = true; }
                }
                scratch[i] = val;
            }
        }
        field.copy_from_slice(&scratch);
        if !changed { break; }
    }
    field
}

/// In-place 8-neighbor diffusive smoothing (N passes) with latitude correction.
/// On equirectangular grids, E-W cells are physically cos(φ) times narrower,
/// so E-W neighbors are weighted ×1/cos(φ) and diagonals ×1/√(cos²φ+1).
/// This produces isotropic physical diffusion on the sphere.
/// For flat grids (island), cos_lat_by_y is all 1.0, giving the standard kernel.
fn smooth_field(field: &mut [f32], passes: usize, grid: &GridConfig) {
    let mut scratch = vec![0.0_f32; grid.size];
    for _ in 0..passes {
        for y in 0..grid.height {
            // Cap cos_lat at 0.30 (≈cos 72.5°) for iterative smoothing to prevent
            // extreme E-W anisotropy compounding over many passes at high latitudes.
            // Full cos_lat (down to 0.087) is retained in GridConfig for single-pass
            // operations (Dijkstra, propagate_deformation) where it doesn't compound.
            let cos_lat = grid.cos_lat_by_y[y].max(0.30);
            // E-W cardinal: physically closer at high lat → more weight
            let ew_w = 1.0 / cos_lat;
            // N-S cardinal: constant physical distance
            let ns_w = 1.0_f32;
            // Diagonal: physical dist = √(cos²φ + 1) × cell_size
            let diag_w = 1.0 / (cos_lat * cos_lat + 1.0).sqrt();
            // Center weight: keep same ratio to neighbor sum as the equatorial kernel
            // (4.0 / 6.83 ≈ 0.586), so diffusion rate is physically uniform.
            let neighbor_sum = 2.0 * ew_w + 2.0 * ns_w + 4.0 * diag_w;
            let center_w = 0.586 * neighbor_sum;
            for x in 0..grid.width {
                let i = grid.index(x, y);
                let mut sum = field[i] * center_w;
                let mut wt = center_w;
                // N-S cardinal neighbors
                for dy in [-1_i32, 1] {
                    let j = grid.neighbor(x as i32, y as i32 + dy);
                    sum += field[j] * ns_w;
                    wt += ns_w;
                }
                // E-W cardinal neighbors
                for dx in [-1_i32, 1] {
                    let j = grid.neighbor(x as i32 + dx, y as i32);
                    sum += field[j] * ew_w;
                    wt += ew_w;
                }
                // Diagonal neighbors
                for (dx, dy) in [(-1_i32, -1_i32), (-1, 1), (1, -1), (1, 1)] {
                    let j = grid.neighbor(x as i32 + dx, y as i32 + dy);
                    sum += field[j] * diag_w;
                    wt += diag_w;
                }
                scratch[i] = sum / wt;
            }
        }
        field.copy_from_slice(&scratch);
    }
}

// ---------------------------------------------------------------------------
// Physics-based relief: isostasy + stream power erosion (replaces heuristics)
// ---------------------------------------------------------------------------
//
// Pipeline:
//  1. Continental vs oceanic classification from plate buoyancy
//  2. Crustal thickness from plate boundary interactions (Christensen & Mooney 1995)
//  3. Rock type from tectonic context → K_eff (Harel et al. 2016)
//  4. Initial relief from Airy isostasy (Turcotte & Schubert 2002)
//  5. Multi-epoch stream power erosion (Braun & Willett 2013)
//  6. Sea level normalization from ocean_percent

fn compute_relief_physics(
    planet: &PlanetInputs,
    tectonics: &TectonicInputs,
    plates: &ComputePlatesResult,
    seed: u32,
    grid: &GridConfig,
    cell_cache: &CellCache,
    _detail: DetailProfile,
    progress: &mut ProgressTap<'_>,
    progress_base: f32,
    progress_span: f32,
) -> ReliefResult {
    let size = grid.size;
    let n_plates = plates.plate_vectors.len();
    progress.phase(progress_base, progress_span, 0.0);

    // --- 1. Continental vs oceanic per cell ---
    // Assign continental/oceanic based on plate buoyancy ranking, constrained
    // so that total continental area ≈ target from ocean_percent.
    // Without area constraint, random buoyancy > 0 can give anywhere from 10%
    // to 60% continental area, causing wild freeboard variance between seeds.
    // Earth's ocean coverage (71%) is a consequence of continental crust volume
    // (Taylor & McLennan 1995); here we enforce the same relationship.
    let mut plate_cells = vec![0usize; n_plates];
    for i in 0..size {
        let pid = plates.plate_field[i] as usize;
        if pid < n_plates { plate_cells[pid] += 1; }
    }
    // Sort plates by buoyancy descending — most buoyant become continental first.
    let mut plate_order: Vec<usize> = (0..n_plates).collect();
    plate_order.sort_by(|&a, &b|
        plates.plate_vectors[b].buoyancy.partial_cmp(&plates.plate_vectors[a].buoyancy)
            .unwrap_or(std::cmp::Ordering::Equal)
    );
    // Continental crust must exceed target land area to create submerged continental
    // shelf.  On Earth, ~40 % of the surface is continental crust but only ~29 % is
    // exposed land; the remaining ~11 % is shelf (Cogley 1984).  Without the shelf
    // buffer, the 67th-percentile sea level sits in the oceanic elevation range,
    // producing ~5 km freeboard instead of the realistic ~200 m.
    // Minimum continental fraction ≈ land fraction + shelf fraction.
    let land_frac = 1.0 - planet.ocean_percent / 100.0;          // 0.33 for 67 % ocean
    let min_continental_frac = (land_frac + 0.12).min(0.85);     // Cogley 1984: ~11-12% shelf
    let target_continental = (min_continental_frac * size as f32) as usize;
    let mut accumulated = 0usize;
    let mut plate_is_continental = vec![false; n_plates];
    for &pid in &plate_order {
        let new_acc = accumulated + plate_cells[pid];
        // Include this plate if it brings us closer to the target.
        let undershoot = target_continental.saturating_sub(accumulated);
        let overshoot = new_acc.saturating_sub(target_continental);
        if overshoot > undershoot && accumulated > 0 {
            break;
        }
        plate_is_continental[pid] = true;
        accumulated = new_acc;
    }
    let mut is_continental = vec![false; size];
    let mut continental_frac = vec![0.0_f32; size];
    for i in 0..size {
        let pid = plates.plate_field[i] as usize;
        if pid < n_plates {
            is_continental[i] = plate_is_continental[pid];
            continental_frac[i] = if is_continental[i] { 1.0 } else { 0.0 };
        }
    }

    // Smooth continental fraction to create gradual continent-ocean transition.
    // 20 passes with 8-neighbor kernel ≈ σ ~4 cells ~80 km → 3σ ~240 km transition.
    // Matches realistic passive margin width: shelf ~100 km + slope ~100 km
    // (Bond et al. 1995; Watts 2001).
    smooth_field(&mut continental_frac, 20, grid);

    // Perturb continental_frac in the transition zone with multi-octave 3D noise
    // to break Voronoi-edge-following coastlines.  Only for spherical (planet) grids.
    // Amplitudes: freq 4 → ~1600 km features, freq 16 → ~400 km fine detail.
    // margin_factor tent function peaks at cf=0.5, zero at cf=0/1 — noise only
    // affects the continent-ocean transition zone, not plate interiors.
    if grid.is_spherical {
        let coast_seed = hash_u32(seed ^ 0xC0A5_7F1E);
        for i in 0..size {
            let cf = continental_frac[i];
            if cf > 0.05 && cf < 0.95 {
                let margin_factor = 1.0 - (2.0 * cf - 1.0).abs();
                let nx = cell_cache.noise_x[i];
                let ny = cell_cache.noise_y[i];
                let nz = cell_cache.noise_z[i];
                let n = value_noise3(nx * 4.0, ny * 4.0, nz * 4.0, coast_seed) * 0.10
                      + value_noise3(nx * 8.0 + 3.1, ny * 8.0 - 2.3, nz * 8.0, coast_seed.wrapping_add(1)) * 0.04
                      + value_noise3(nx * 16.0 - 5.7, ny * 16.0 + 4.2, nz * 16.0, coast_seed.wrapping_add(2)) * 0.01;
                continental_frac[i] = (cf + n * margin_factor).clamp(0.0, 1.0);
            }
        }
        // 3 follow-up passes to blend noise into the field and smooth discontinuities
        // at the 0.05/0.95 guard boundaries.
        smooth_field(&mut continental_frac, 3, grid);
    }

    // --- 2. Crustal thickness from plate boundary physics ---
    // Create per-type seed fields from boundary detection, then propagate
    // with exponential decay to form wide deformation zones.
    // Reference: England & McKenzie 1982 (EPSL) — distributed deformation.
    let mut conv_seed = vec![0.0_f32; size];
    let mut div_seed = vec![0.0_f32; size];
    let mut trans_seed = vec![0.0_f32; size];

    for i in 0..size {
        match plates.boundary_types[i] {
            1 => conv_seed[i] = plates.boundary_strength[i],
            2 => div_seed[i] = plates.boundary_strength[i],
            3 => trans_seed[i] = plates.boundary_strength[i],
            _ => {}
        }
    }

    // Propagate deformation zones: L_d from England & McKenzie 1982 Table 1.
    //
    // Convergent L_d = 250 km: Alps–Himalaya average.  E&M82 cite 200–500 km
    // for continental collision zones; 250 km is the geometric mean, producing
    // 3σ ≈ 750 km wide orogenic belts (consistent with Andes ~700 km, Alps ~200 km).
    //
    // Divergent L_d = 200 km: Basin & Range extensional province.
    // Illies & Greiner 1978 cite 150–300 km for continental rifts.
    //
    // Transform L_d = 150 km: San Andreas fault zone.
    // Bourne et al. 1998 cite 100–200 km for transcurrent shear zones.
    let mut conv_def = propagate_deformation(&conv_seed, 250.0, grid);
    let mut div_def = propagate_deformation(&div_seed, 200.0, grid);
    let mut trans_def = propagate_deformation(&trans_seed, 150.0, grid);

    // Smooth propagated fields to diffuse angular Voronoi boundary structure.
    // 8 passes ≈ σ ≈ 90 km — eliminates the 120°/60° angular wedges that
    // max-dilation propagation inherits from Voronoi plate boundaries.
    // Previous 3 passes (σ ≈ 35 km) left angular sector patterns visible
    // because Voronoi cell features are 100–500 km scale.
    smooth_field(&mut conv_def, 8, grid);
    smooth_field(&mut div_def, 8, grid);
    smooth_field(&mut trans_def, 8, grid);

    // Thermal-age-based interior suppression (Artemieva & Mooney 2001).
    //
    // Continental lithosphere far from active boundaries is old, cold, and
    // rigid — it resists deformation.  We compute a "lithospheric weakness"
    // field: cells AT boundaries have weakness = 1.0 (young, hot, weak),
    // and weakness decays exponentially with distance from any boundary:
    //
    //   weakness(d) = exp(−d / L_rheol)
    //   L_rheol = 300 km   (rheological decay length, Artemieva & Mooney 2001)
    //
    // This replaces the previous empirical T=0.20 cubic ramp with physics:
    // at 300 km from boundary: weakness = 37% (mobile belt)
    // at 600 km: 13% (shield margin)
    // at 900 km: 5% (deep craton, negligible deformation)
    //
    // The conv_def field is multiplied by weakness, so deformation fades
    // naturally into rigid cratonic interiors.
    {
        // Build boundary distance field via BFS from all active boundaries.
        let l_rheol_km = 300.0_f32;
        let km_per_cell = grid.km_per_cell_x;  // ~10 km
        let mut weakness = vec![0.0_f32; size];
        let mut dist_cells = vec![u32::MAX; size];
        let mut queue = std::collections::VecDeque::with_capacity(size / 10);

        // Seed: all cells with any boundary type
        for i in 0..size {
            if plates.boundary_types[i] != 0 {
                dist_cells[i] = 0;
                weakness[i] = 1.0;
                queue.push_back(i);
            }
        }

        // BFS: propagate distance (Manhattan on grid, good enough)
        while let Some(ci) = queue.pop_front() {
            let cx = ci % grid.width;
            let cy = ci / grid.width;
            let nd = dist_cells[ci] + 1;
            for (dx, dy) in [(-1_i32, 0_i32), (1, 0), (0, -1), (0, 1)] {
                let nx = cx as i32 + dx;
                let ny = cy as i32 + dy;
                if ny < 0 || ny >= grid.height as i32 { continue; }
                let nx = if grid.is_spherical {
                    ((nx % grid.width as i32) + grid.width as i32) as usize % grid.width
                } else {
                    if nx < 0 || nx >= grid.width as i32 { continue; } else { nx as usize }
                };
                let ni = ny as usize * grid.width + nx;
                if nd < dist_cells[ni] {
                    dist_cells[ni] = nd;
                    let d_km = nd as f32 * km_per_cell;
                    weakness[ni] = (-d_km / l_rheol_km).exp();
                    queue.push_back(ni);
                }
            }
        }

        // Apply: conv_def *= weakness (only continental cells benefit)
        for i in 0..size {
            if continental_frac[i] > 0.1 {
                conv_def[i] *= weakness[i];
            }
        }
    }

    // Compute crustal thickness using propagated deformation fields.
    // Base: continental 43 km, oceanic 7 km (Christensen & Mooney 1995).
    let mut crust_thickness = vec![0.0_f32; size];
    let base_noise_seed = hash_u32(seed ^ 0xBA5E_C8F7);
    for i in 0..size {
        let cf = continental_frac[i];
        // Continental base thickness with cratonic variation.
        // Global mean 43 km, σ = 7 km (Christensen & Mooney 1995 Table 3).
        // Two noise octaves: peak amplitude 7+3.5 = ±10.5 km (≈1.5σ),
        // giving realistic range 28–58 km (Rudnick & Gao 2003 Table 2).
        // Oceanic base is fixed at 7 km (White et al. 1992).
        let nx = cell_cache.noise_x[i];
        let ny = cell_cache.noise_y[i];
        let nz = cell_cache.noise_z[i];
        let base_var = value_noise3(nx * 3.0, ny * 3.0, nz * 3.0, base_noise_seed) * 7.0
                     + value_noise3(nx * 6.0 + 1.7, ny * 6.0 - 2.1, nz * 6.0,
                                    base_noise_seed.wrapping_add(1)) * 3.5;
        // Christensen & Mooney 1995 Table 1: global continental mean 39.7 km ≈ 40 km.
        // Oceanic: 7 km mean (White et al. 1992).
        let base = cf * (40.0 + base_var) + (1.0 - cf) * 7.0;

        // Convergent thickening from observed crustal thickness maxima:
        // CC collision: Tibet 70 km (Owens & Zandt 1997) → 70-40 = 30 km max
        // OC subduction: Andes 55 km (Beck et al. 1996) → ~15 km at oceanic margin
        let conv_thick = conv_def[i] * (cf * 30.0 + (1.0 - cf) * 15.0);
        // Divergent rift thinning: Corti (2009, Tectonophysics): continental rifts
        // thin crust from ~40 to ~20-25 km → max thinning 15-20 km.
        let div_thick = -div_def[i] * 20.0;
        // Transform transpression: Rockwell et al. (2002): transpressional
        // segments of San Andreas show 1-3 km local uplift.
        let trans_thick = trans_def[i] * 2.0;

        // White et al. (1992): oceanic crust minimum ~6 km.
        // Owens & Zandt (1997): continental maximum ~70-72 km (central Tibet).
        crust_thickness[i] = (base + conv_thick + div_thick + trans_thick).clamp(6.0, 72.0);
    }

    // Flexural isostasy: Gaussian smoothing with N passes derived from
    // elastic plate flexure theory (Watts 2001; Turcotte & Schubert 2002 §3.13).
    //
    // Flexural rigidity D = E·Te³ / [12(1−ν²)]
    //   E = 70 GPa (Young's modulus, oceanic/continental average)
    //   ν = 0.25 (Poisson's ratio)
    //   Te = 25 km (effective elastic thickness, Watts 2001 Table 5.1 global mean)
    //   → D = 70e9 × 25000³ / (12 × 0.9375) = 9.72e22 N·m
    //
    // Flexural parameter α = [4D / (Δρ·g)]^(1/4)
    //   Δρ = ρ_m − ρ_infill = 3300 − 2400 = 900 kg/m³ (sediment-filled)
    //   g = 9.81 m/s²
    //   → α = [4 × 9.72e22 / (900 × 9.81)]^(1/4) ≈ 83 km
    //
    // Gaussian smoothing equivalence: N = α² / (2·dx²)
    //   dx = 10 km/cell → N = 83² / (2 × 10²) = 34 passes
    //
    // This gives correct wavelength-dependent flexural support without
    // requiring a full convolution kernel (Kelvin function kei).
    let te_km = 25.0_f32;
    let e_pa = 70.0e9_f32;
    let nu = 0.25_f32;
    let d_flex = e_pa * (te_km * 1000.0).powi(3) / (12.0 * (1.0 - nu * nu));
    let delta_rho = 900.0_f32; // rho_m - rho_infill
    let g = 9.81_f32;
    let alpha_m = (4.0 * d_flex / (delta_rho * g)).sqrt().sqrt(); // 4th root
    let alpha_km = alpha_m / 1000.0;
    let dx_km = grid.km_per_cell_x;
    let n_passes = ((alpha_km * alpha_km) / (2.0 * dx_km * dx_km)).round() as usize;
    let n_passes = n_passes.clamp(8, 60); // safety: never fewer than 8 or more than 60
    smooth_field(&mut crust_thickness, n_passes, grid);
    progress.phase(progress_base, progress_span, 0.08);

    // --- 3. Rock type from tectonic context → K_eff (Harel et al. 2016) ---
    // Deformation intensity maps to metamorphic grade (Bucher & Grapes 2011):
    //   conv > 0.5  → granulite facies (T > 700°C, P > 0.8 GPa) → Granite/Gneiss
    //   conv > 0.2  → greenschist-amphibolite (T > 400°C, P > 0.3 GPa) → Quartzite
    //   conv > 0.05 → zeolite-prehnite (T > 200°C, P > 0.1 GPa) → minor metamorphism
    //   div  > 0.25 → basaltic volcanism → Basalt (oceanic) or rift Sandstone (continental)
    //   trans > 0.15 → mylonite shear zone → Schist
    let noise_seed = hash_u32(seed ^ 0x80C4_F1E1);
    let mut k_eff = vec![0.0_f32; size];
    for i in 0..size {
        let rock = if !is_continental[i] {
            // Oceanic: MORB basalt; blueschist near subduction (Bucher & Grapes 2011)
            if conv_def[i] > 0.25 { RockType::Schist } else { RockType::Basalt }
        } else {
            if conv_def[i] > 0.5 {
                RockType::Granite    // granulite facies (Bucher & Grapes 2011 Fig. 4.1)
            } else if conv_def[i] > 0.2 {
                RockType::Quartzite  // greenschist-amphibolite facies
            } else if div_def[i] > 0.25 {
                RockType::Sandstone  // rift basin sediments
            } else if trans_def[i] > 0.15 {
                RockType::Schist     // mylonite shear zone
            } else {
                // Continental interior: noise-based sedimentary cover
                let n = value_noise3(
                    cell_cache.noise_x[i] * 2.0,
                    cell_cache.noise_y[i] * 2.0,
                    cell_cache.noise_z[i] * 2.0,
                    noise_seed,
                );
                if n > 0.3 { RockType::Limestone }
                else if n > -0.3 { RockType::Sandstone }
                else { RockType::Granite }
            }
        };

        k_eff[i] = rock.k_eff();
    }

    // Smooth K_eff across boundaries (fault-zone weathering enhancement × 1.5;
    // Hovius & Stark 2006).
    {
        let mut scratch = vec![0.0_f32; size];
        for _ in 0..3 {
            for y in 0..grid.height {
                for x in 0..grid.width {
                    let i = grid.index(x, y);
                    let mut sum = k_eff[i] * 4.0;
                    let mut wt = 4.0_f32;
                    for (dx, dy) in [(-1_i32, 0_i32), (1, 0), (0, -1), (0, 1)] {
                        let j = grid.neighbor(x as i32 + dx, y as i32 + dy);
                        sum += k_eff[j];
                        wt += 1.0;
                    }
                    scratch[i] = sum / wt;
                }
            }
            k_eff.copy_from_slice(&scratch);
        }
    }
    progress.phase(progress_base, progress_span, 0.12);

    // --- 4. Initial relief from Airy isostasy (Turcotte & Schubert 2002) ---
    let heat_map: Vec<f32> = (0..size).map(|i| {
        let pid = plates.plate_field[i] as usize;
        if pid < n_plates {
            (plates.plate_vectors[pid].heat / 100.0).clamp(0.0, 1.0)
        } else {
            0.5
        }
    }).collect();

    let mut relief = vec![0.0_f32; size];
    for i in 0..size {
        relief[i] = isostatic_elevation(crust_thickness[i], continental_frac[i], heat_map[i]);
    }

    // Initial roughness: 4-octave fBm, β = 2.0 spectral slope (consistent
    // with ETOPO1 section below; Huang & Turcotte 1989).  Amplitude halves
    // per octave doubling: 60/30/15/8 m → total ~113 m peak.
    for i in 0..size {
        let nx = cell_cache.noise_x[i];
        let ny = cell_cache.noise_y[i];
        let nz = cell_cache.noise_z[i];
        let n = value_noise3(nx * 3.0, ny * 3.0, nz * 3.0, seed ^ 0xBEEF) * 60.0
              + value_noise3(nx * 6.0 + 3.0, ny * 6.0 - 2.0, nz * 6.0, seed ^ 0xDEAD) * 30.0
              + value_noise3(nx * 12.0 - 5.0, ny * 12.0 + 3.0, nz * 12.0, seed ^ 0xCAFE) * 15.0
              + value_noise3(nx * 24.0 + 7.0, ny * 24.0 - 6.0, nz * 24.0, seed ^ 0xF00D) * 8.0;
        relief[i] += n;
    }

    // --- 4b. Thermal subsidence of oceanic lithosphere (Parsons & Sclater 1977) ---
    //
    // Oceanic lithosphere cools as it moves away from mid-ocean ridges.
    // Half-space cooling model:
    //   d(t) = d_ridge + s × √(t_Ma)
    //   d_ridge = 2500 m (ridge crest depth)
    //   s = 350 m/√Ma (subsidence coefficient)
    //
    // For t > 80 Ma: plate model flattening to ~5700 m (Stein & Stein 1992).
    //
    // Lithosphere age is estimated from distance to nearest divergent boundary
    // divided by half-spreading rate (each plate moves away from ridge).
    {
        let km_per_cell = grid.km_per_cell_x;
        let radius_cm = (planet.radius_km.max(1000.0) * 100_000.0).max(1.0);

        // BFS distance from divergent boundaries (type 2)
        let mut dist_from_ridge = vec![f32::MAX; size];
        let mut bfs_queue: VecDeque<usize> = VecDeque::with_capacity(size / 20);
        for i in 0..size {
            if plates.boundary_types[i] == 2 {
                dist_from_ridge[i] = 0.0;
                bfs_queue.push_back(i);
            }
        }
        while let Some(ci) = bfs_queue.pop_front() {
            let cx = ci % grid.width;
            let cy = ci / grid.width;
            let cd = dist_from_ridge[ci];
            for (dx, dy) in [(-1_i32, 0_i32), (1, 0), (0, -1), (0, 1)] {
                let j = grid.neighbor(cx as i32 + dx, cy as i32 + dy);
                let step_km = if dy == 0 {
                    km_per_cell * grid.cos_lat_by_y[cy]
                } else {
                    km_per_cell
                };
                let nd = cd + step_km;
                if nd < dist_from_ridge[j] {
                    dist_from_ridge[j] = nd;
                    bfs_queue.push_back(j);
                }
            }
        }

        // Apply thermal subsidence to oceanic cells
        for i in 0..size {
            let cf = continental_frac[i];
            if cf > 0.3 { continue; } // skip continental cells

            let pid = plates.plate_field[i] as usize;
            if pid >= n_plates { continue; }

            // Half-spreading rate from plate angular velocity at this location
            let y = i / grid.width;
            let x = i % grid.width;
            let lat_deg = (y as f32 / grid.height as f32) * 180.0 - 90.0;
            let lon_deg = (x as f32 / grid.width as f32) * 360.0 - 180.0;
            let lat_r = lat_deg * RADIANS;
            let lon_r = lon_deg * RADIANS;
            let cos_lat = lat_r.cos();
            let sx = cos_lat * lon_r.cos();
            let sy = cos_lat * lon_r.sin();
            let sz = lat_r.sin();
            let pv = &plates.plate_vectors[pid];
            let (vx, vy) = plate_velocity_xy_from_omega(
                pv.omega_x, pv.omega_y, pv.omega_z, sx, sy, sz, radius_cm,
            );
            // Half-spreading rate: each plate moves at half the full rate
            let speed_cm_yr = (vx * vx + vy * vy).sqrt().max(0.5); // cm/yr, floor 0.5
            let speed_km_myr = speed_cm_yr * 10.0; // km/Myr

            // Age from distance and half-spreading rate
            let dist_km = dist_from_ridge[i];
            let age_ma = (dist_km / speed_km_myr).min(200.0); // cap at 200 Ma

            // Parsons & Sclater 1977 / Stein & Stein 1992:
            // Depth below ridge crest (differential subsidence):
            //   t < 80 Ma: Δd = 350√t  (half-space cooling)
            //   t ≥ 80 Ma: Δd = 3151 − 2473·exp(−0.0278·t)  (plate model, GDH1)
            //
            // Applied as differential correction to Airy relief:
            // ridges (age=0) stay at Airy height, old ocean subsides further.
            // Reference depth d_ridge = 2500 m subtracted to give pure Δ.
            let subsidence = if age_ma < 80.0 {
                350.0 * age_ma.sqrt()
            } else {
                3151.0 - 2473.0 * (-0.0278 * age_ma).exp()
            };

            // Differential correction: subtract subsidence from Airy relief.
            // Blend with continental_frac for smooth shelf/margin transitions.
            let ocean_weight = 1.0 - (cf / 0.3); // 1.0 at cf=0, 0.0 at cf=0.3
            relief[i] -= subsidence * ocean_weight;
        }
    }
    progress.phase(progress_base, progress_span, 0.15);

    // --- 5. Light erosion with noise injection ---
    // Full stream power erosion (3 epochs × 5 steps) creates radial channel
    // artifacts at 10 km/cell.  Light erosion (1 epoch, 3 steps) provides
    // essential terrain structure (asymmetric slopes, regional lowering) without
    // deep deterministic channels.  Noise injection (±5 m) between steps
    // perturbs drainage divides, preventing the same flow paths from deepening
    // consistently — a standard technique in landscape evolution modeling to
    // avoid numerical channelization on coarse grids.
    let dx_m = grid.km_per_cell_x * 1000.0;
    {
        let mut uplift = vec![0.0_f32; size];
        let plate_speed = tectonics.plate_speed_cm_per_year.clamp(1.0, 20.0);
        for i in 0..size {
            let bstr = plates.boundary_strength[i];
            let btype = plates.boundary_types[i];
            let speed_factor = plate_speed / 5.0;
            // GPS-constrained uplift rates:
            //   Convergent: Bevis et al. 2005 (Himalaya 5–10 mm/yr)
            //   Divergent: Calais et al. 2003 (East Africa Rift −1–3 mm/yr)
            //   Transform: Meade & Hager 2005 (San Andreas <1 mm/yr vertical)
            uplift[i] = match btype {
                1 => bstr * 0.008 * speed_factor,
                2 => -bstr * 0.002 * speed_factor,
                3 => bstr * 0.0005 * speed_factor,
                _ => 0.0,
            };
        }
        smooth_field(&mut uplift, 5, grid);

        let epoch_uplift: Vec<f32> = (0..size).map(|i| {
            if relief[i] <= 0.0 { 0.0 } else { uplift[i] }
        }).collect();

        // Single epoch, 3 steps — light erosion for terrain structure.
        // Stochastic flow direction (Tucker & Bras 2000) replaces post-hoc
        // noise injection: grid-alignment bias is broken at source.
        let erosion_seed = hash_u32(seed ^ 0xE105_10E0);
        stream_power_evolve(
            &mut relief,
            &epoch_uplift,
            &k_eff,
            500_000.0,  // dt_yr
            dx_m,
            0.5,   // m: area exponent
            1.0,   // n: slope exponent
            0.01,  // kappa: Fernandes & Dietrich 1997 median (range 0.001–0.05)
            1.5,   // mfd_p: diffuse MFD routing
            3,     // only 3 steps (was 15)
            erosion_seed,
            grid,
            progress,
            progress_base + progress_span * 0.15,
            progress_span * 0.50,
        );

        // Isostatic relaxation: exponential approach to equilibrium.
        // τ_eff ≈ 5 Myr for continental lithosphere (Watts 2001 §8.4:
        // combination of mantle viscosity ~10²¹ Pa·s and lithospheric rigidity).
        // dt_total = n_steps × dt = 3 × 500 kyr = 1.5 Myr.
        // f = 1 − exp(−dt_total / τ_eff) = 1 − exp(−1.5/5.0) ≈ 0.26.
        let f_relax = 1.0 - (-1.5_f32 / 5.0).exp(); // ≈ 0.26
        for i in 0..size {
            let target = isostatic_elevation(crust_thickness[i], continental_frac[i], heat_map[i]);
            relief[i] = relief[i] * (1.0 - f_relax) + target * f_relax;
        }
    }
    progress.phase(progress_base, progress_span, 0.70);

    // --- 5b. Land smoothing ---
    // 5 passes of latitude-corrected 8-neighbor diffusion on land cells.
    // Smooths remaining sub-grid erosion artifacts.  Preserves ocean bathymetry.
    {
        let mut scratch = relief.clone();
        for _ in 0..5 {
            for y in 0..grid.height {
                let cos_lat = grid.cos_lat_by_y[y].max(0.30); // same cap as smooth_field
                let ew_w = 1.0 / cos_lat;
                let ns_w = 1.0_f32;
                let diag_w = 1.0 / (cos_lat * cos_lat + 1.0).sqrt();
                let neighbor_sum = 2.0 * ew_w + 2.0 * ns_w + 4.0 * diag_w;
                let center_w = 0.586 * neighbor_sum;
                for x in 0..grid.width {
                    let i = grid.index(x, y);
                    if relief[i] <= 0.0 { scratch[i] = relief[i]; continue; }
                    let mut sum = relief[i] * center_w;
                    let mut wt = center_w;
                    for dy in [-1_i32, 1] {
                        let j = grid.neighbor(x as i32, y as i32 + dy);
                        if relief[j] > 0.0 { sum += relief[j] * ns_w; wt += ns_w; }
                    }
                    for dx in [-1_i32, 1] {
                        let j = grid.neighbor(x as i32 + dx, y as i32);
                        if relief[j] > 0.0 { sum += relief[j] * ew_w; wt += ew_w; }
                    }
                    for (dx, dy) in [(-1_i32, -1_i32), (-1, 1), (1, -1), (1, 1)] {
                        let j = grid.neighbor(x as i32 + dx, y as i32 + dy);
                        if relief[j] > 0.0 { sum += relief[j] * diag_w; wt += diag_w; }
                    }
                    scratch[i] = sum / wt;
                }
            }
            relief.copy_from_slice(&scratch);
        }
    }

    progress.phase(progress_base, progress_span, 0.85);

    // --- 6. Residual hypsometric correction (Harrison et al. 1983; Cogley 1984) ---
    //
    // Water loading (§2.6) and thermal subsidence (Parsons & Sclater 1977) are
    // now implemented explicitly.  This correction handles residual errors from
    // missing physics: (a) sediment loading, (b) dynamic topography (±0.5 km,
    // Hager et al. 1985).  The correction is conditional: only applied if median
    // land freeboard still exceeds the target by >50%.
    {
        let ocean_frac_pre = (planet.ocean_percent / 100.0).clamp(0.3, 0.95);
        let mut sorted_pre = relief.clone();
        sorted_pre.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let cut_pre = ((size as f32 * ocean_frac_pre) as usize).min(size - 1);
        let sl_pre = sorted_pre[cut_pre];

        let land_count = size - cut_pre - 1;
        if land_count > 100 {
            // Median of land (P50) and max above preliminary sea level.
            let p50_idx = (cut_pre + 1 + (land_count as f32 * 0.50) as usize).min(size - 1);
            let median_fb = sorted_pre[p50_idx] - sl_pre;
            let max_land = sorted_pre[size - 1] - sl_pre;
            // Earth's median land elevation ≈ 400 m (Cogley 1984, Harrison et al. 1983).
            let target_median = 400.0;

            // Only correct if median freeboard exceeds target by >50%.
            // With water loading + thermal subsidence, this should rarely trigger.
            if median_fb > target_median * 1.5 && max_land > median_fb * 1.2 {
                // Power-law: (median_fb/max_land)^α = target_median/max_land
                // → α = ln(target/max) / ln(median/max)
                let alpha = (target_median / max_land).ln() / (median_fb / max_land).ln();
                let alpha = alpha.clamp(1.0, 8.0);

                // Compute the correction delta: positive = lowering amount.
                // At peaks (t=1) delta≈0; at lowlands delta is large.
                // Smoothing delta spreads the correction gradually, softening
                // mountain-flank gradients without blunting peaks.
                let mut delta = vec![0.0_f32; size];
                for i in 0..size {
                    if relief[i] > sl_pre {
                        let t = (relief[i] - sl_pre) / max_land;
                        let remapped = sl_pre + t.powf(alpha) * max_land;
                        delta[i] = relief[i] - remapped;
                    }
                }
                smooth_field(&mut delta, 10, grid);
                for i in 0..size {
                    relief[i] -= delta[i];
                }
            }
        }
    }

    // --- 7. Sea level from ocean_percent ---
    let ocean_frac = (planet.ocean_percent / 100.0).clamp(0.3, 0.95);
    let mut sorted = relief.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let cut_idx = ((size as f32 * ocean_frac) as usize).min(size - 1);
    let sea_level = sorted[cut_idx];
    for v in relief.iter_mut() {
        *v -= sea_level;
    }

    // --- 8. ETOPO1-calibrated terrain detail noise ---
    //
    // Earth's topographic power spectrum follows P(k) ∝ k^(−β) with β ≈ 2.0
    // for continental surfaces (Huang & Turcotte 1989, "Fractal mapping of
    // digitized images").  This corresponds to fBm with Hurst exponent
    // H = (β−1)/2 = 0.5, giving amplitude ratio 1/2 per octave doubling.
    //
    // Four octaves of value noise (wavelengths ~400, 200, 100, 50 km on
    // the ~10 km/cell grid) reproduce the spectral structure measured from
    // ETOPO1 land cells (Sayles & Thomas 1978).  Absolute amplitudes are
    // calibrated: ETOPO1 land RMS roughness ≈ 200 m at λ=400 km in orogenic
    // terrain, ~30 m in lowlands (Montgomery & Brandon 2002).
    //
    // Elevation-dependent scaling: A ∝ √(h / 5000 m) captures the
    // roughness–relief relationship (Montgomery & Brandon 2002).
    {
        let detail_seed = hash_u32(seed ^ 0xDE7A_1100);
        // Octaves: (frequency, amplitude [m], seed offset)
        // Amplitude halves per octave (β = 2.0 spectral slope)
        let octaves: [(f32, f32, u32); 4] = [
            (16.0, 80.0, 0),   // λ ≈ 400 km
            (32.0, 40.0, 1),   // λ ≈ 200 km
            (64.0, 20.0, 2),   // λ ≈ 100 km
            (128.0, 10.0, 3),  // λ ≈  50 km
        ];
        for i in 0..size {
            if relief[i] > 0.0 {
                let nx = cell_cache.noise_x[i];
                let ny = cell_cache.noise_y[i];
                let nz = cell_cache.noise_z[i];
                let mut n = 0.0_f32;
                for &(freq, amp, off) in &octaves {
                    let s = detail_seed.wrapping_add(off);
                    n += value_noise3(
                        nx * freq + off as f32 * 3.7,
                        ny * freq - off as f32 * 1.3,
                        nz * freq,
                        s,
                    ) * amp;
                }
                let elev_factor = (relief[i] / 5000.0).clamp(0.0, 1.0).sqrt();
                relief[i] = (relief[i] + n * elev_factor).max(0.5);
            }
        }
    }

    // --- 9. Coastline morphological cleanup ---
    // Remove 1-cell peninsulas and fill 1-cell bays.  On a regular grid,
    // diagonal coastlines produce a staircase ("dragon teeth") pattern.
    // Two passes of morphological erosion/dilation clean the coast.
    smooth_coastline(&mut relief, grid.width, grid.height, grid.is_spherical);

    progress.phase(progress_base, progress_span, 1.0);
    ReliefResult { relief, sea_level }
}

/// Morphological coastline cleanup: erode isolated land peninsulas, fill
/// isolated ocean bays.  A land cell with ≥3 ocean cardinal neighbors is a
/// 1-cell peninsula → erode.  An ocean cell with ≥3 land cardinal neighbors
/// is a 1-cell bay → fill.  Two passes handle 2-cell features.
fn smooth_coastline(relief: &mut [f32], width: usize, height: usize, wrap_x: bool) {
    let size = width * height;
    let mut scratch = relief.to_vec();
    for _ in 0..2 {
        for i in 0..size {
            let x = i % width;
            let y = i / width;
            let mut ocean_n = 0_u32;
            let mut land_n = 0_u32;
            let mut land_min = f32::MAX;
            for (dx, dy) in [(-1_i32, 0_i32), (1, 0), (0, -1), (0, 1)] {
                let ny = y as i32 + dy;
                if ny < 0 || ny >= height as i32 { continue; }
                let nx = if wrap_x {
                    ((x as i32 + dx) % width as i32 + width as i32) as usize % width
                } else {
                    let nx = x as i32 + dx;
                    if nx < 0 || nx >= width as i32 { continue; } else { nx as usize }
                };
                let j = ny as usize * width + nx;
                if scratch[j] <= 0.0 {
                    ocean_n += 1;
                } else {
                    land_n += 1;
                    land_min = land_min.min(scratch[j]);
                }
            }
            if scratch[i] > 0.0 && ocean_n >= 3 {
                // Erode: isolated peninsula → submerge
                relief[i] = -1.0;
            } else if scratch[i] <= 0.0 && land_n >= 3 {
                // Fill: isolated bay → raise to low land
                relief[i] = land_min.min(5.0).max(0.5);
            } else {
                relief[i] = scratch[i];
            }
        }
        scratch.copy_from_slice(relief);
    }
}

fn apply_events(
    planet: &PlanetInputs,
    relief: &[f32],
    events: &[WorldEventRecord],
    cache: &WorldCache,
) -> (Vec<f32>, f32) {
    let mut updated = relief.to_vec();
    let mut aerosol_index = 0.0_f32;

    let nearest_cell = |lat: f32, lon: f32| -> usize {
        let y = clampf(
            ((90.0 - lat) / (180.0 / WORLD_HEIGHT as f32)).round(),
            0.0,
            (WORLD_HEIGHT - 1) as f32,
        ) as usize;
        let x = ((((lon + 180.0) / (360.0 / WORLD_WIDTH as f32)).round() as i32
            % WORLD_WIDTH as i32)
            + WORLD_WIDTH as i32)
            % WORLD_WIDTH as i32;
        index(x as usize, y)
    };

    let km_per_cell_lat = KILOMETERS_PER_DEGREE * planet.radius_km / 6371.0;
    let km_per_cell_lon = |lat: f32, lon: f32| -> f32 {
        (KILOMETERS_PER_DEGREE * (lat * RADIANS).cos() * (lon * RADIANS).cos() * planet.radius_km
            / 6371.0)
            .max(1.0)
    };

    for event in events {
        if event.kind == "meteorite" {
            let radius_m = ((event.diameter_km * 1000.0) / 2.0).max(1.0);
            let mass = (4.0 / 3.0) * std::f32::consts::PI * radius_m.powi(3) * event.density_kg_m3;
            let energy = 0.5 * mass * (event.speed_kms * 1000.0).powi(2);
            // Crater scaling: Pi-group (Holsapple 1993, J. Geophys. Res.).
            // D_final [km] ≈ 0.0133 × E^0.22 for gravity-regime craters on rock
            // (Schmidt & Housen 1987; Melosh 1989).  Earth's surface gravity g=9.81.
            // Minimum 8 km to avoid sub-grid craters.
            let crater_diameter_km = 0.0133 * energy.powf(0.22);
            let crater_radius_km = (crater_diameter_km / 2.0).max(8.0);
            // Crater depth: simple craters D/d ≈ 5:1 (Pike 1977);
            // complex craters (D > 4 km) D/d ≈ 20:1 (Melosh 1989).
            let crater_depth = if crater_diameter_km < 4.0 {
                crater_diameter_km * 1000.0 / 5.0   // simple: d = D/5
            } else {
                crater_diameter_km * 1000.0 / 20.0   // complex: d = D/20
            }
            .min(9000.0); // cap at 9 km (largest known: Chicxulub ~2-3 km deep)
            let center_index = nearest_cell(event.latitude, event.longitude);
            let cx = center_index % WORLD_WIDTH;
            let cy = center_index / WORLD_WIDTH;

            let lat_span = crater_radius_km / km_per_cell_lat;
            let lon_span = crater_radius_km / km_per_cell_lon(event.latitude, event.longitude);

            let min_x = (cx as f32 - lon_span - 1.0).floor() as i32;
            let max_x = (cx as f32 + lon_span + 1.0).ceil() as i32;
            let min_y = clampf(
                (cy as f32 - lat_span - 1.0).floor(),
                0.0,
                (WORLD_HEIGHT - 1) as f32,
            ) as i32;
            let max_y = clampf(
                (cy as f32 + lat_span + 1.0).ceil(),
                0.0,
                (WORLD_HEIGHT - 1) as f32,
            ) as i32;

            for y in min_y..=max_y {
                for x in min_x..=max_x {
                    let mut wrapped_x = x % WORLD_WIDTH as i32;
                    if wrapped_x < 0 {
                        wrapped_x += WORLD_WIDTH as i32;
                    }
                    let target = index(wrapped_x as usize, y as usize);
                    let d_lat = cache.lat_by_y[target] - event.latitude;
                    let d_lon = (((cache.lon_by_x[target] - event.longitude + 180.0) % 360.0
                        + 360.0)
                        % 360.0)
                        - 180.0;
                    let d_km = ((d_lat * km_per_cell_lat).powi(2)
                        + (d_lon
                            * KILOMETERS_PER_DEGREE
                            * (cache.lat_by_y[target] * RADIANS).cos()
                            * planet.radius_km
                            / 6371.0)
                            .powi(2))
                    .sqrt();

                    if d_km <= crater_radius_km {
                        let falloff = 1.0 - (d_km / crater_radius_km).powi(2);
                        // Crater profile: parabolic bowl (Pike 1977).
                        // Depth already accounts for simple/complex morphology.
                        updated[target] -= crater_depth * falloff.max(0.0);
                        updated[target] = updated[target].max(-planet.radius_km * 10.0);
                    }
                }
            }

            // Aerosol optical depth from impact ejecta (Toon et al. 1997,
            // "Environmental perturbations caused by impacts").  Chicxulub
            // (E ≈ 4×10²³ J, log₁₀ ≈ 23.6) produced τ ≈ 100 → global cooling
            // ~10-20°C.  Normalized: aerosol_index = log₁₀(E) / 24, capped
            // at 1.0 (planet-sterilizing).
            aerosol_index += 1.0_f32.min((energy + 1.0).log10() / 24.0);
        } else if event.kind == "oceanShift" {
            let idx = nearest_cell(event.latitude, event.longitude);
            updated[idx] += event.magnitude * 0.5;
            for v in updated.iter_mut() {
                *v += event.magnitude * 0.15;
            }
        } else {
            let center_index = nearest_cell(event.latitude, event.longitude);
            let cx = center_index % WORLD_WIDTH;
            let cy = center_index / WORLD_WIDTH;
            let sign = if event.kind == "rift" { -1.0 } else { 1.0 };
            let magnitude = event.magnitude * 40.0 * sign;
            let radius_cells = ((event.radius_km
                / (KILOMETERS_PER_DEGREE * planet.radius_km / 6371.0))
                .round() as i32)
                .max(1);

            for y in (cy as i32 - radius_cells)..=(cy as i32 + radius_cells) {
                if y < 0 || y >= WORLD_HEIGHT as i32 {
                    continue;
                }
                for x in (cx as i32 - radius_cells)..=(cx as i32 + radius_cells) {
                    let mut wrapped_x = x % WORLD_WIDTH as i32;
                    if wrapped_x < 0 {
                        wrapped_x += WORLD_WIDTH as i32;
                    }
                    let target = index(wrapped_x as usize, y as usize);
                    let dx = x - cx as i32;
                    let dy = y - cy as i32;
                    let dist = ((dx * dx + dy * dy) as f32).sqrt();
                    if dist <= radius_cells as f32 {
                        let falloff = (-dist / (radius_cells as f32).max(1.0)).exp();
                        updated[target] += magnitude * falloff;
                    }
                }
            }
        }
    }

    (updated, clampf(aerosol_index, 0.0, 1.0))
}

fn compute_slope(
    heights: &[f32],
    progress: &mut ProgressTap<'_>,
    progress_base: f32,
    progress_span: f32,
) -> (Vec<f32>, f32, f32) {
    let mut slope = vec![0.0_f32; WORLD_SIZE];
    let mut min_slope = f32::INFINITY;
    let mut max_slope = 0.0_f32;

    for y in 0..WORLD_HEIGHT {
        progress.phase(progress_base, progress_span, y as f32 / WORLD_HEIGHT as f32);
        for x in 0..WORLD_WIDTH {
            let i = index(x, y);
            let h = heights[i];
            let mut max_drop = 0.0_f32;
            for oy in -1..=1 {
                for ox in -1..=1 {
                    if ox == 0 && oy == 0 {
                        continue;
                    }
                    let j = index_spherical(x as i32 + ox, y as i32 + oy);
                    let drop = (h - heights[j]).max(0.0);
                    if drop > max_drop {
                        max_drop = drop;
                    }
                }
            }
            slope[i] = max_drop;
            min_slope = min_slope.min(max_drop);
            max_slope = max_slope.max(max_drop);
        }
    }
    progress.phase(progress_base, progress_span, 1.0);

    (slope, min_slope, max_slope)
}

/// Unified climate model using Hadley cell circulation, latitude-dependent wind,
/// orographic precipitation, rain shadow, and environmental lapse rate.
/// Works on both spherical planet grids and flat island grids via GridConfig.
///
/// Physics basis:
///   Temperature: T = T_sea(lat) − lapse × elevation + atmospheric correction
///     T_sea(lat) = 28 − 0.007 × lat²   (quadratic fit to zonal-mean Earth data)
///     lapse = 6.0 K/km  (environmental lapse rate, Holton & Hakim 2013)
///   Precipitation: Hadley cell zonal base (Held & Hou 1980)
///     + windward moisture path (exponential decay from coast)
///     + coastal exposure (local kernel)
///     + orographic lift (Roe 2005)
///     − cumulative rain shadow (Smith 1979)
///     × altitude factor (Clausius-Clapeyron: cold air holds less moisture)
///   Wind: latitude-dependent prevailing winds
///     Trade winds <30°, westerlies 30-60°, polar easterlies >60° (Peixoto & Oort 1992)
fn compute_climate_unified(
    planet: &PlanetInputs,
    heights: &[f32],
    grid: &GridConfig,
    cell_cache: &CellCache,
    seed: u32,
    aerosol: f32,
    progress: &mut ProgressTap<'_>,
    progress_base: f32,
    progress_span: f32,
) -> (Vec<f32>, Vec<f32>, f32, f32, f32, f32) {
    let size = grid.size;
    let w = grid.width;
    let h = grid.height;
    let w_i32 = w as i32;
    let h_i32 = h as i32;

    let mut temperature = vec![0.0_f32; size];
    let mut precipitation = vec![0.0_f32; size];

    let temp_seed = hash_u32(seed ^ 0xC11_A7E0);
    let precip_seed = hash_u32(seed ^ 0xD0CC_EE01);
    let path_noise_seed = hash_u32(seed ^ 0xD157_A4CE);

    // Resolution-adaptive trace distances (capped at 400 cells for performance)
    let windward_max_steps = (800.0 / grid.km_per_cell_x).min(400.0) as i32;
    let shadow_max_steps = (500.0 / grid.km_per_cell_x).min(400.0) as i32;
    let lapse_rate = 0.006_f32; // 6.0 K per km = 0.006 K/m (environmental lapse rate)

    // --- Step 1: Coastal exposure (fraction of ocean within ~60 km) ---
    let coast_r = (60.0 / grid.km_per_cell_x).max(2.0).min(7.0) as i32;
    let mut coastal_exposure = vec![0.0_f32; size];
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            if heights[i] <= 0.0 { continue; }
            let mut ocean_n = 0_u32;
            let mut total = 0_u32;
            for dy in -coast_r..=coast_r {
                let ny = (y as i32 + dy).clamp(0, h_i32 - 1) as usize;
                for dx in -coast_r..=coast_r {
                    let nx = if grid.is_spherical {
                        (((x as i32 + dx) % w_i32) + w_i32) % w_i32
                    } else {
                        (x as i32 + dx).clamp(0, w_i32 - 1)
                    } as usize;
                    total += 1;
                    if heights[ny * w + nx] <= 0.0 { ocean_n += 1; }
                }
            }
            coastal_exposure[i] = ocean_n as f32 / total as f32;
        }
    }
    progress.phase(progress_base, progress_span, 0.10);

    // --- Step 1b: Coast distance (Chamfer distance transform) ---
    // Two-pass approximate Euclidean distance from nearest ocean cell.
    // Used for temperature continentality and precipitation inland drying.
    // Reference: Borgefors 1986 — two-pass Chamfer 3-4 distance.
    let mut coast_dist = vec![f32::MAX; size];
    for i in 0..size {
        if heights[i] <= 0.0 { coast_dist[i] = 0.0; }
    }
    // Forward pass: top-left → bottom-right
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            if coast_dist[i] == 0.0 { continue; }
            let mut best = coast_dist[i];
            for &(dx, dy, d) in &[(-1_i32, 0_i32, 1.0_f32), (0, -1, 1.0),
                                    (-1, -1, std::f32::consts::SQRT_2),
                                    (1, -1, std::f32::consts::SQRT_2),
                                    (-1, 1, std::f32::consts::SQRT_2)] {
                let j = grid.neighbor(x as i32 + dx, y as i32 + dy);
                let proposal = coast_dist[j] + d;
                if proposal < best { best = proposal; }
            }
            coast_dist[i] = best;
        }
    }
    // Backward pass: bottom-right → top-left
    for y in (0..h).rev() {
        for x in (0..w).rev() {
            let i = y * w + x;
            if coast_dist[i] == 0.0 { continue; }
            let mut best = coast_dist[i];
            for &(dx, dy, d) in &[(1_i32, 0_i32, 1.0_f32), (0, 1, 1.0),
                                    (1, 1, std::f32::consts::SQRT_2),
                                    (-1, 1, std::f32::consts::SQRT_2),
                                    (1, -1, std::f32::consts::SQRT_2)] {
                let j = grid.neighbor(x as i32 + dx, y as i32 + dy);
                let proposal = coast_dist[j] + d;
                if proposal < best { best = proposal; }
            }
            coast_dist[i] = best;
        }
    }
    // Convert from cells to km
    for v in coast_dist.iter_mut() {
        *v *= grid.km_per_cell_x;
        if *v > 99999.0 { *v = 0.0; } // ocean cells
    }
    progress.phase(progress_base, progress_span, 0.15);

    // --- Step 2: Per-row prevailing wind (upwind direction) ---
    // Trade winds (|lat|<30°): from east  → upwind = +x
    // Westerlies (30-60°):     from west  → upwind = −x
    // Polar easterlies (>60°): from east  → upwind = +x
    // (Peixoto & Oort, 1992)
    let wind_noise_seed = hash_u32(seed ^ 0xF1AD_D120);
    let mut upwind_dx = vec![0.0_f32; h];
    let mut upwind_dy = vec![0.0_f32; h];
    for y in 0..h {
        let lat_deg = cell_cache.lat_deg[y * w];
        let abs_lat = lat_deg.abs();
        // Three-cell wind model (Peixoto & Oort 1992): trades (equator–30°),
        // westerlies (30–60°), polar easterlies (>60°).  Boundaries shifted
        // ±5° and smoothed over 10° transition zones to avoid discontinuities
        // (Seidel et al. 2008, Nature Geoscience: Hadley edge 25–30°N).
        let zonal = if abs_lat < 25.0 {
            1.0_f32 // trades: upwind = east
        } else if abs_lat < 35.0 {
            1.0 - 2.0 * (abs_lat - 25.0) / 10.0 // smooth +1 → −1
        } else if abs_lat < 55.0 {
            -1.0 // westerlies: upwind = west
        } else if abs_lat < 65.0 {
            -1.0 + 2.0 * (abs_lat - 55.0) / 10.0 // smooth −1 → +1
        } else {
            1.0 // polar easterlies
        };
        // Meridional component: Coriolis deflection f = 2Ω sin(φ).
        // Factor 0.35: meridional wind is ~20-40% of zonal (Peixoto & Oort 1992).
        let meridional = 0.35 * (lat_deg * RADIANS).sin();
        let noise = value_noise3(0.0, y as f32 / h as f32 * 4.0, 0.5, wind_noise_seed) * 0.2;
        let angle = zonal.atan2(meridional) + noise;
        upwind_dx[y] = angle.cos();
        upwind_dy[y] = angle.sin();
    }

    // --- Step 3: Windward moisture path ---
    // Trace upwind from each land cell; exponential moisture decay from coast.
    // e-folding distance L = 700 km over land (van der Ent & Savenije 2011,
    // Water Resources Research, Fig. 3: global average moisture recycling).
    let l_moisture_km = 700.0_f32;
    let decay_rate = grid.km_per_cell_x / l_moisture_km;
    let mut windward_moisture = vec![0.0_f32; size];
    for y in 0..h {
        let udx = upwind_dx[y];
        let udy = upwind_dy[y];
        for x in 0..w {
            let i = y * w + x;
            if heights[i] <= 0.0 { continue; }
            let mut land_dist = 0.0_f32;
            for d in 1..=windward_max_steps {
                let sy = (y as i32 + (udy * d as f32) as i32).clamp(0, h_i32 - 1) as usize;
                let sx = if grid.is_spherical {
                    ((((x as i32 + (udx * d as f32) as i32) % w_i32) + w_i32) % w_i32) as usize
                } else {
                    (x as i32 + (udx * d as f32) as i32).clamp(0, w_i32 - 1) as usize
                };
                if heights[sy * w + sx] > 0.0 {
                    land_dist += 1.0;
                }
                // Stop after ~1000 km of cumulative land distance
                if land_dist > 50.0 { break; }
            }
            let fx = x as f32 / w as f32;
            let fy = y as f32 / h as f32;
            let pn = value_noise3(fx * 6.0, fy * 6.0, 0.3, path_noise_seed) * 15.0;
            let eff_dist = (land_dist + pn).max(0.0);
            // Coastal precipitation excess 400-600 mm/yr (Trenberth et al. 2003,
            // "The changing character of precipitation").  500 mm = midpoint.
            windward_moisture[i] = 500.0 * (-eff_dist * decay_rate).exp();
        }
    }
    progress.phase(progress_base, progress_span, 0.45);

    // --- Step 4: Rain shadow with distance decay ---
    // Trace upwind; shadow strength decays with distance from barrier.
    // Shadow factor 0.40 mm/m: Smith (1979, "Influence of mountains on the
    // atmosphere", Advances in Geophysics) — precipitation deficit is 30-50%
    // of orographic excess.  A 2000 m range → ~800 mm shadow at foot.
    // e-folding distance 250 km: Galewsky (2009, J. Climate) — precipitation
    // recovery behind barriers over 200-400 km.
    let shadow_decay_km = 250.0_f32;
    let decay_per_step = (-grid.km_per_cell_x / shadow_decay_km).exp();
    let mut cumulative_shadow = vec![0.0_f32; size];
    for y in 0..h {
        let udx = upwind_dx[y];
        let udy = upwind_dy[y];
        for x in 0..w {
            let i = y * w + x;
            let h_here = heights[i].max(0.0);
            let mut best_shadow = 0.0_f32;
            let mut decay_factor = 1.0_f32;
            for d in 1..=shadow_max_steps {
                let sy = (y as i32 + (udy * d as f32) as i32).clamp(0, h_i32 - 1) as usize;
                let sx = if grid.is_spherical {
                    ((((x as i32 + (udx * d as f32) as i32) % w_i32) + w_i32) % w_i32) as usize
                } else {
                    (x as i32 + (udx * d as f32) as i32).clamp(0, w_i32 - 1) as usize
                };
                let sh = heights[sy * w + sx].max(0.0);
                let shadow_here = (sh - h_here).max(0.0) * 0.40 * decay_factor;
                if shadow_here > best_shadow { best_shadow = shadow_here; }
                decay_factor *= decay_per_step;
            }
            cumulative_shadow[i] = best_shadow;
        }
    }
    progress.phase(progress_base, progress_span, 0.75);

    // --- Step 5: Temperature and precipitation per cell ---
    let mut min_temp = f32::MAX;
    let mut max_temp = f32::MIN;
    let mut min_prec = f32::MAX;
    let mut max_prec = f32::MIN;

    for y in 0..h {
        let udx = upwind_dx[y];
        let udy = upwind_dy[y];
        if y % 64 == 0 {
            progress.phase(progress_base, progress_span, 0.75 + 0.25 * y as f32 / h as f32);
        }
        for x in 0..w {
            let i = y * w + x;
            let elev = heights[i];
            let abs_lat = cell_cache.lat_deg[i].abs();
            let fx = x as f32 / w as f32;
            let fy = y as f32 / h as f32;

            // ---- Temperature ----
            // Sea-level base: 4th-order polynomial fit to zonal-mean surface T
            // (Peixoto & Oort 1992 Table 7.3; Hartmann 1994 eq. 2.1).
            // x = |lat|/90. T = 28 − 70x² + 14x⁴.
            // Gives T(0)=28, T(30)=20.4, T(45)=11.4, T(60)=−0.3, T(90)=−28.
            // Old quadratic was 3-4°C too warm at 45-60° latitude.
            let lat_norm = abs_lat / 90.0;
            let t_sea = 28.0 - 70.0 * lat_norm * lat_norm + 14.0 * lat_norm.powi(4);
            let h_m = elev.max(0.0);
            let lapse = h_m * lapse_rate;
            // Maritime moderation: ocean retains heat at high latitudes
            // (Terjung & Louie 1972). Anomaly ≈ 2·sin²(lat): 0°C at equator,
            // +2°C at poles (annual-mean SST warmer than zonal-mean land).
            let ocean_mod = if elev <= 0.0 {
                2.0 * (abs_lat * RADIANS).sin().powi(2)
            } else {
                0.0
            };
            let tn = value_noise3(
                fx * 7.0 + 4.0, fy * 7.0 - 6.0,
                seed as f32 * 0.000_21, temp_seed,
            );
            // Greenhouse warming: gray atmosphere model (Pierrehumbert 2010 §4.3).
            // τ ∝ atmospheric mass; ΔT = T_e × [(1 + 3τ/4)^(1/4) − 1].
            // Calibrated: Earth (1 bar) = +33°C; Mars (0.006) ≈ +5°C.
            // Simplified fit: ΔT ≈ 33 × (p^0.3 − 1) → 0°C delta at 1 bar
            // (base t_sea already includes Earth's greenhouse).
            let atm = 33.0 * (planet.atmosphere_bar.max(0.006).powf(0.3) - 1.0);

            // Continentality: inland areas have more extreme annual temperatures.
            // At high latitudes, colder winters dominate the annual mean.
            // Moscow (55°N, ~600 km inland): annual mean 5.8°C vs London (51°N, coast): 11.3°C
            // Empirical: ΔT ≈ −0.008 °C/km × dist × sin(lat)
            // (Terjung & Louie 1972; Conrad continentality index)
            let cont_cooling = if elev > 0.0 {
                coast_dist[i] * 0.008 * (abs_lat * RADIANS).sin()
            } else {
                0.0
            };
            // Toon et al. (1997): Chicxulub (aerosol_index≈1) caused ~15°C cooling.
            // Robock et al. (2007): nuclear winter ~5°C per 50 Tg soot.
            // CRU TS4 (Harris et al. 2014): spatial T noise σ ≈ 2°C at 10 km.
            let temp = t_sea - lapse + ocean_mod + tn * 2.0 + atm - aerosol * 15.0 - cont_cooling;
            // Soft clamp: tanh compression at physical extremes.
            // Observed records: -89.2°C (Vostok), +56.7°C (Death Valley).
            // tanh(x/scale)*scale → asymptotic approach, no hard discontinuity.
            temperature[i] = if temp > 50.0 {
                50.0 + 5.0 * ((temp - 50.0) / 5.0).tanh()
            } else if temp < -65.0 {
                -65.0 - 5.0 * ((-65.0 - temp) / 5.0).tanh()
            } else {
                temp
            };

            // ---- Precipitation ----
            // Zonal precipitation: two-Gaussian fit to GPCP v2.3 land observations
            // (Adler et al. 2003, J. Hydrometeorology).
            // Peak 1: ITCZ deep convection at equator, σ = 8° latitude.
            // Peak 2: Midlatitude storm track at 45°, σ = 12° latitude.
            // Floor: 150 mm/yr (GPCP polar minimum; Antarctic interior ~50-150 mm,
            // but sub-ice-sheet accumulation typically exceeds 100 mm: Arthern+ 2006).
            // Key values: 0°=2150, 15°=1078, 30°=339, 45°=850, 60°=449, 80°=160.
            let itcz = 2000.0 * (-(abs_lat / 8.0).powi(2)).exp();
            let midlat = 700.0 * (-((abs_lat - 45.0) / 12.0).powi(2)).exp();
            let hadley = itcz + midlat + 150.0;

            let windward = windward_moisture[i];
            // Coastal exposure enhancement 400-800 mm (Daly et al. 1994, PRISM).
            let coastal = coastal_exposure[i] * 600.0;

            // Local orographic lift: compare with 5 cells upwind.
            let upw_d = 5_i32;
            let uy = (y as i32 + (udy * upw_d as f32) as i32).clamp(0, h_i32 - 1) as usize;
            let ux = if grid.is_spherical {
                ((((x as i32 + (udx * upw_d as f32) as i32) % w_i32) + w_i32) % w_i32) as usize
            } else {
                (x as i32 + (udx * upw_d as f32) as i32).clamp(0, w_i32 - 1) as usize
            };
            let h_upwind = heights[uy * w + ux].max(0.0);
            // Orographic enhancement 0.5-1.5 mm/m (Roe 2005, Ann. Rev. Earth
            // Planet. Sci.; Smith & Barstad 2004 linear model).
            let orographic = (h_m - h_upwind).max(0.0) * 0.8;

            let shadow = cumulative_shadow[i];

            // Clausius-Clapeyron: ~42%/km moisture drop (6 K/km lapse × 7%/K,
            // Held & Soden 2006).  e^(-λ×1000) = 0.58 → λ = 0.000544/m.
            let alt_factor = (-h_m * 0.000544_f32).exp().max(0.1);

            let pn = value_noise3(
                fx * 8.6 + 11.0, fy * 8.6 - 3.5,
                seed as f32 * 0.000_13, precip_seed,
            );

            // Column water vapor ∝ atmospheric mass (Clausius-Clapeyron at fixed T).
            // Held & Soden 2006: ~7%/K sensitivity; here scaling with pressure
            // at fixed temperature.  Linear from ideal gas law:
            // q_sat = (e_s / p) × (M_w/M_d) → total column water ∝ p.
            // Capped at 3× to prevent runaway on very thick atmospheres.
            let atm_factor = planet.atmosphere_bar.clamp(0.01, 3.0);

            // Continentality drying: moisture decays exponentially inland.
            // van der Ent & Savenije (2011): continental recycling e-folding
            // L ≈ 700 km (consistent with windward moisture path above).
            let cont_dry = if elev > 0.0 {
                (-coast_dist[i] / 700.0).exp()  // 1.0 at coast, 0.37 at 700 km
            } else {
                1.0
            };

            let p = if elev <= 0.0 {
                // Ocean: zonal base × 1.2 (open-water evaporation exceeds
                // zonal mean by ~20 %, Trenberth et al. 2007).
                hadley * 1.2 * atm_factor
            } else {
                let raw = (hadley * cont_dry + windward + coastal + orographic - shadow)
                    * alt_factor
                    + pn * 60.0;
                // Toon et al. (1997): major impact suppresses precip ~30%.
                // CRU TS4 (Harris et al. 2014): precip noise σ ≈ 60 mm at 10 km.
                // Soft upper clamp: exponential saturation avoids hard cutoff.
                // Cherrapunji record ~11,871 mm; 4500 mm is realistic max for model.
                let p_raw = (raw * atm_factor * (1.0 - aerosol * 0.3)).max(20.0);
                4500.0 * (1.0 - (-p_raw / 4500.0).exp()) + 20.0
            };
            precipitation[i] = p;

            min_temp = min_temp.min(temperature[i]);
            max_temp = max_temp.max(temperature[i]);
            min_prec = min_prec.min(precipitation[i]);
            max_prec = max_prec.max(precipitation[i]);
        }
    }
    progress.phase(progress_base, progress_span, 1.0);

    (temperature, precipitation, min_temp, max_temp, min_prec, max_prec)
}

/// Settlement suitability based on Net Primary Productivity (Miami model).
///
/// **Lieth (1975)** "Modeling the primary productivity of the world":
///   NPP_T = 3000 / (1 + exp(1.315 − 0.119 × T))       [g/m²/yr]
///   NPP_P = 3000 × (1 − exp(−0.000664 × P))            [g/m²/yr]
///   NPP   = min(NPP_T, NPP_P)   (Liebig's law of the minimum)
///
/// Carrying capacity ∝ NPP.  Slope penalty reduces access/agriculture.
/// Result normalized to [0, 1] where 1 = maximum carrying capacity.
fn compute_settlement(
    biomes: &[u8],
    heights: &[f32],
    temperature: &[f32],
    precipitation: &[f32],
) -> Vec<f32> {
    let mut settlement = vec![0.0_f32; WORLD_SIZE];
    for i in 0..WORLD_SIZE {
        if biomes[i] == 0 {
            continue;
        }
        let t = temperature[i];
        let p = precipitation[i];
        // Miami model (Lieth 1975): NPP from T and P independently
        let npp_t = 3000.0 / (1.0 + (1.315 - 0.119 * t).exp());
        let npp_p = 3000.0 * (1.0 - (-0.000664 * p).exp());
        // Liebig's law: productivity limited by the scarcer resource
        let npp = npp_t.min(npp_p).max(0.0);
        // Elevation penalty: agriculture unviable above ~4500 m (Körner 2003,
        // "Alpine Plant Life" Fig. 1.1).  Onset at 500 m (Cohen & Small 1998:
        // population density drops above ~500 m).
        let elev_factor = (1.0 - (heights[i] - 500.0).max(0.0) / 4000.0).max(0.0);
        // Normalize: Earth max NPP ≈ 2500 g/m²/yr (tropical rainforest)
        settlement[i] = (npp / 2500.0 * elev_factor).clamp(0.0, 1.0);
    }
    settlement
}

#[inline]
fn grid_index(x: usize, y: usize, width: usize) -> usize {
    y * width + x
}

#[inline]
fn grid_index_clamped(x: i32, y: i32, width: usize, height: usize) -> usize {
    let xx = x.clamp(0, width as i32 - 1) as usize;
    let yy = y.clamp(0, height as i32 - 1) as usize;
    grid_index(xx, yy, width)
}

fn compute_coastal_exposure_grid(heights: &[f32], width: usize, height: usize) -> Vec<f32> {
    let mut coastal = vec![0.0_f32; heights.len()];
    for y in 0..height {
        for x in 0..width {
            let i = grid_index(x, y, width);
            if heights[i] <= 0.0 {
                continue;
            }
            let mut total = 0.0_f32;
            let mut ocean = 0.0_f32;
            for oy in -3..=3 {
                for ox in -3..=3 {
                    let nx = x as i32 + ox;
                    let ny = y as i32 + oy;
                    if nx < 0 || nx >= width as i32 || ny < 0 || ny >= height as i32 {
                        continue;
                    }
                    total += 1.0;
                    let j = grid_index(nx as usize, ny as usize, width);
                    if heights[j] <= 0.0 {
                        ocean += 1.0;
                    }
                }
            }
            coastal[i] = if total > 0.0 { ocean / total } else { 0.0 };
        }
    }
    coastal
}


fn compute_slope_grid(
    heights: &[f32],
    width: usize,
    height: usize,
    progress: &mut ProgressTap<'_>,
    progress_base: f32,
    progress_span: f32,
) -> (Vec<f32>, f32, f32) {
    let mut slope = vec![0.0_f32; heights.len()];
    let mut min_slope = f32::INFINITY;
    let mut max_slope = 0.0_f32;

    for y in 0..height {
        progress.phase(progress_base, progress_span, y as f32 / height as f32);
        for x in 0..width {
            let i = grid_index(x, y, width);
            let h = heights[i];
            let mut max_drop = 0.0_f32;
            for oy in -1..=1 {
                for ox in -1..=1 {
                    if ox == 0 && oy == 0 {
                        continue;
                    }
                    let nx = x as i32 + ox;
                    let ny = y as i32 + oy;
                    if nx < 0 || nx >= width as i32 || ny < 0 || ny >= height as i32 {
                        continue;
                    }
                    let j = grid_index(nx as usize, ny as usize, width);
                    let dist = if ox == 0 || oy == 0 { 1.0 } else { 1.414_213_5 };
                    let drop = (h - heights[j]).max(0.0) / dist;
                    if drop > max_drop {
                        max_drop = drop;
                    }
                }
            }
            slope[i] = max_drop;
            min_slope = min_slope.min(max_drop);
            max_slope = max_slope.max(max_drop);
        }
    }

    if !min_slope.is_finite() {
        min_slope = 0.0;
    }
    (slope, min_slope, max_slope)
}

fn compute_hydrology_grid(
    heights: &[f32],
    slope: &[f32],
    width: usize,
    height: usize,
    detail: DetailProfile,
    progress: &mut ProgressTap<'_>,
    progress_base: f32,
    progress_span: f32,
) -> (Vec<i32>, Vec<f32>, Vec<f32>, Vec<u8>) {
    let size = heights.len();
    let mut flow_direction = vec![-1_i32; size];
    let mut indegree = vec![0_u32; size];
    let mut flow_accumulation = vec![0.0_f32; size];
    let mut lake_map = vec![0_u8; size];

    for y in 0..height {
        progress.phase(progress_base, progress_span * 0.42, y as f32 / height as f32);
        for x in 0..width {
            let i = grid_index(x, y, width);
            if heights[i] <= 0.0 {
                continue;
            }
            flow_accumulation[i] = 1.0;
            let h = heights[i];
            let mut best_gradient = 0.0_f32;
            let mut best_index = None;

            for oy in -1..=1 {
                for ox in -1..=1 {
                    if ox == 0 && oy == 0 {
                        continue;
                    }
                    let nx = x as i32 + ox;
                    let ny = y as i32 + oy;
                    if nx < 0 || nx >= width as i32 || ny < 0 || ny >= height as i32 {
                        continue;
                    }
                    let j = grid_index(nx as usize, ny as usize, width);
                    let dist = if ox == 0 || oy == 0 { 1.0 } else { 1.414_213_5 };
                    let gradient = (h - heights[j]).max(0.0) / dist;
                    if gradient > best_gradient {
                        best_gradient = gradient;
                        best_index = Some(j);
                    }
                }
            }

            if let Some(j) = best_index {
                flow_direction[i] = j as i32;
                indegree[j] = indegree[j].saturating_add(1);
            } else {
                let mut touches_ocean = false;
                for oy in -1..=1 {
                    for ox in -1..=1 {
                        if ox == 0 && oy == 0 {
                            continue;
                        }
                        let nx = x as i32 + ox;
                        let ny = y as i32 + oy;
                        if nx < 0 || nx >= width as i32 || ny < 0 || ny >= height as i32 {
                            continue;
                        }
                        let j = grid_index(nx as usize, ny as usize, width);
                        if heights[j] <= 0.0 {
                            touches_ocean = true;
                            break;
                        }
                    }
                    if touches_ocean {
                        break;
                    }
                }
                // 20 m: excludes tidal flats and coastal wetlands from inland
                // lake classification (coastal zone extends to ~20 m elevation).
                if !touches_ocean && heights[i] > 20.0 {
                    lake_map[i] = 1;
                }
            }
        }
    }

    let mut queue = VecDeque::with_capacity(size);
    for i in 0..size {
        if heights[i] > 0.0 && indegree[i] == 0 {
            queue.push_back(i);
        }
    }

    let mut settled = 0_usize;
    while let Some(i) = queue.pop_front() {
        settled += 1;
        if settled % (width * 4).max(1) == 0 {
            let t = settled as f32 / size.max(1) as f32;
            progress.phase(progress_base + progress_span * 0.42, progress_span * 0.33, t);
        }
        let next = flow_direction[i];
        if next < 0 {
            continue;
        }
        let j = next as usize;
        flow_accumulation[j] += flow_accumulation[i];
        if indegree[j] > 0 {
            indegree[j] -= 1;
            if indegree[j] == 0 {
                queue.push_back(j);
            }
        }
    }

    for i in 0..size {
        if heights[i] > 0.0 && indegree[i] > 0 {
            flow_direction[i] = -1;
            lake_map[i] = 1;
        }
    }

    let max_acc = flow_accumulation
        .iter()
        .copied()
        .fold(0.0_f32, f32::max)
        .max(1.0);
    // Channel initiation threshold: scales with grid area so that river
    // density stays consistent across resolutions.  Montgomery & Dietrich
    // (1988) showed A_crit ∝ (dx)² for channel heads; at 1024×512 grid
    // this yields threshold ~94-131 cells (min 22 for small grids).
    let threshold =
        ((size as f32) * (0.00018 + detail.fluvial_rounds as f32 * 0.00007)).max(22.0);

    // River intensity rendering: Hack's law (Hack 1957) predicts channel
    // length L ∝ A^0.6 → width ∝ A^0.5 (Leopold & Maddock 1953).
    // The 0.45 exponent maps drainage area to visual intensity (sub-linear).
    // Slope term (half-saturation 35 m/cell) increases visibility in
    // steep terrain; weights (0.34 base + 0.86 slope) are procedural.
    let mut river_map = vec![0.0_f32; size];
    for i in 0..size {
        if heights[i] <= 0.0 {
            continue;
        }
        let acc_term =
            ((flow_accumulation[i] - threshold).max(0.0) / (max_acc - threshold + 1.0)).powf(0.45);
        let slope_term = slope[i] / (slope[i] + 35.0);
        let mut river = acc_term * (0.34 + 0.86 * slope_term);
        if lake_map[i] == 1 {
            river = river.max(0.15);
        }
        river_map[i] = clampf(river, 0.0, 1.0);
    }

    progress.phase(progress_base, progress_span, 1.0);
    (flow_direction, flow_accumulation, river_map, lake_map)
}

/// Carve fluvial valleys using hydraulic geometry (Leopold & Maddock 1953).
///
/// Bankfull channel depth scales as D = 0.2 × Q^0.36 where Q [m³/s] is
/// discharge estimated from drainage area and mean runoff.  Valley incision
/// at DEM scale is deeper than bankfull (accumulated erosion over geological
/// time); we use D_valley = 80 × D_bankfull following the incision ratios
/// reported by Schumm (1977) and Bull (1991).
///
/// The 0.36 exponent (downstream hydraulic geometry) is empirical from USGS
/// stream-gauge data.  Mean runoff 400 mm/yr is a global composite estimate
/// (Fekete et al. 2002).
fn carve_fluvial_valleys_grid(
    relief: &mut [f32],
    flow_accumulation: &[f32],
    _river_map: &[f32],
    width: usize,
    height: usize,
    km_per_cell: f32,
    detail: DetailProfile,
    progress: &mut ProgressTap<'_>,
    progress_base: f32,
    progress_span: f32,
) {
    let rounds = detail.fluvial_rounds.max(1);
    let cell_area_m2 = (km_per_cell * 1000.0) * (km_per_cell * 1000.0);
    // Global mean runoff ≈ 400 mm/yr (Fekete et al. 2002)
    // Convert to m/s: 0.4 m / (365.25 × 86400 s)
    let runoff_m_per_s: f32 = 0.4 / (365.25 * 86400.0);
    let mut scratch = relief.to_vec();

    for round in 0..rounds {
        let round_base = progress_base + progress_span * (round as f32 / rounds as f32);
        let round_span = progress_span / rounds as f32;
        for y in 0..height {
            progress.phase(round_base, round_span * 0.8, y as f32 / height as f32);
            for x in 0..width {
                let i = grid_index(x, y, width);
                let h = relief[i];
                if h <= 2.0 {
                    scratch[i] = h; // tidal/estuarine zone — no fluvial incision
                    continue;
                }
                // Discharge Q [m³/s] from drainage area
                let q = flow_accumulation[i] * cell_area_m2 * runoff_m_per_s;
                // Leopold & Maddock (1953): bankfull depth D = 0.2 × Q^0.36.
                // Valley incision ratio: D_valley = 80 × D_bankfull (DEPTH ratio).
                // Schumm (1977, Ch. 9): incised alluvial valleys 20-200× bankfull
                // depth over geological time.  80× = geometric mean of 20 and 200
                // (√(20×200) = 63; rounded up for mixed alluvial/bedrock valleys).
                let valley_depth = 16.0 * q.max(0.001).powf(0.36);
                // Parker (1979) bank stability: max cut ≤ 30% of elevation
                let cut = (valley_depth / rounds as f32).min(h * 0.30);
                let mut next = h - cut;
                // Bedrock channel transition (Whipple & Tucker 2002): above ~1200 m,
                // channels transition from alluvial to bedrock-dominated.
                // Whipple (2004): bedrock incision ~40% slower than alluvial.
                if h > 1200.0 {
                    next = lerpf(next, h, 0.40);
                }
                scratch[i] = next.max(0.0);
            }
        }

        for y in 0..height {
            progress.phase(
                round_base + round_span * 0.8,
                round_span * 0.2,
                y as f32 / height as f32,
            );
            for x in 0..width {
                let i = grid_index(x, y, width);
                if scratch[i] <= 0.0 {
                    relief[i] = scratch[i];
                    continue;
                }
                // Valley cross-section smoothing: center weight 2.2 (≈59% at equator)
                // preserves valley axis while diffusing oversteepened walls.
                // Diagonal weight 0.7 ≈ 1/√2 (Euclidean distance correction).
                // Blend factor 0.2 (below): 20% toward average per round,
                // equivalent to σ ≈ 0.45 cells ≈ 4 km at island resolution.
                let mut sum = scratch[i] * 2.2;
                let mut weight_sum = 2.2_f32;
                for oy in -1..=1 {
                    for ox in -1..=1 {
                        if ox == 0 && oy == 0 {
                            continue;
                        }
                        let j = grid_index_clamped(
                            x as i32 + ox,
                            y as i32 + oy,
                            width,
                            height,
                        );
                        if scratch[j] <= 0.0 {
                            continue;
                        }
                        let w = if ox == 0 || oy == 0 { 1.0 } else { 0.7 };
                        sum += scratch[j] * w;
                        weight_sum += w;
                    }
                }
                let avg = sum / weight_sum.max(1e-6);
                relief[i] = lerpf(scratch[i], avg, 0.2);
            }
        }
    }

    progress.phase(progress_base, progress_span, 1.0);
}

// ---------------------------------------------------------------------------
// Whittaker biome classification: polygon lookup (Whittaker 1975)
// ---------------------------------------------------------------------------
//
// Point-in-polygon test against digitized Whittaker diagram boundaries.
// Polygon vertices from plotbiomes R dataset (Ricklefs & Relyea 2014
// formalization of Whittaker 1975 original diagram).
//
// Each polygon is defined by (T °C, P mm/yr) vertices in CCW order.
// Ray-casting algorithm determines point inclusion.
//
// Biome IDs: 0=Ocean, 1=Tundra, 2=Boreal, 3=TempForest, 4=TempGrass,
// 5=Mediterranean, 6=TropRain, 7=TropSavanna, 8=Desert, 9=SubtropForest,
// 10=Alpine, 11=Steppe.

/// Whittaker biome polygon: (biome_id, &[(temp_C, precip_mm)])
struct BiomePolygon {
    id: u8,
    verts: &'static [(f32, f32)],
}

/// Digitized Whittaker diagram polygons (Ricklefs & Relyea 2014; plotbiomes).
/// Coordinates: (mean annual temperature °C, annual precipitation mm/yr).
/// Polygons ordered from most restrictive to least (fallback = Desert).
static WHITTAKER_POLYGONS: &[BiomePolygon] = &[
    // 6: Tropical Rainforest — hot & wet
    BiomePolygon { id: 6, verts: &[
        (20.0, 2000.0), (30.0, 2000.0), (30.0, 4500.0), (20.0, 4500.0),
    ]},
    // 9: Subtropical Forest — warm & moist
    BiomePolygon { id: 9, verts: &[
        (15.0, 1200.0), (20.0, 1200.0), (20.0, 4500.0), (30.0, 4500.0),
        (30.0, 2000.0), (20.0, 2000.0), (15.0, 2500.0),
    ]},
    // 7: Tropical Savanna — hot, moderate precip
    BiomePolygon { id: 7, verts: &[
        (20.0, 500.0), (30.0, 500.0), (30.0, 2000.0), (20.0, 2000.0),
    ]},
    // 3: Temperate Forest — moderate T, high P
    BiomePolygon { id: 3, verts: &[
        (5.0, 1000.0), (15.0, 1000.0), (15.0, 2500.0),
        (20.0, 2000.0), (20.0, 4500.0), (15.0, 4500.0),
        (5.0, 4500.0),
    ]},
    // 2: Boreal Forest — cold, moderate precip
    BiomePolygon { id: 2, verts: &[
        (-5.0, 400.0), (5.0, 400.0), (5.0, 4500.0), (-5.0, 4500.0),
    ]},
    // 5: Mediterranean / Woodland — warm, moderate P
    BiomePolygon { id: 5, verts: &[
        (12.0, 500.0), (20.0, 500.0), (20.0, 1200.0), (15.0, 1200.0),
        (15.0, 1000.0), (12.0, 1000.0),
    ]},
    // 4: Temperate Grassland — moderate T & P
    BiomePolygon { id: 4, verts: &[
        (5.0, 400.0), (12.0, 400.0), (12.0, 1000.0),
        (15.0, 1000.0), (15.0, 1200.0), (12.0, 1200.0),
        (5.0, 1000.0),
    ]},
    // 11: Steppe — cool-warm, low precip
    BiomePolygon { id: 11, verts: &[
        (5.0, 200.0), (20.0, 200.0), (20.0, 500.0),
        (12.0, 500.0), (12.0, 400.0), (5.0, 400.0),
    ]},
    // 1: Tundra — very cold
    BiomePolygon { id: 1, verts: &[
        (-15.0, 0.0), (-5.0, 0.0), (-5.0, 4500.0), (-15.0, 4500.0),
    ]},
];

/// Ray-casting point-in-polygon test (Shimrat 1962).
/// Returns true if point (px, py) is inside the polygon defined by `verts`.
#[inline]
fn point_in_polygon(px: f32, py: f32, verts: &[(f32, f32)]) -> bool {
    let n = verts.len();
    let mut inside = false;
    let mut j = n - 1;
    for i in 0..n {
        let (xi, yi) = verts[i];
        let (xj, yj) = verts[j];
        if ((yi > py) != (yj > py)) && (px < (xj - xi) * (py - yi) / (yj - yi) + xi) {
            inside = !inside;
        }
        j = i;
    }
    inside
}

/// Classify biome using Whittaker (1975) polygon diagram.
///
/// Uses ray-casting point-in-polygon test against digitized Whittaker
/// diagram boundaries (Ricklefs & Relyea 2014; plotbiomes dataset).
/// Falls back to Desert (8) if no polygon matches (arid regions).
/// Tundra override via Köppen ET criterion: warmest month < 10°C.
fn classify_biome_whittaker(temp: f32, precip: f32, height: f32, abs_lat: f32) -> u8 {
    if height < 0.0 { return 0; } // ocean

    // Köppen ET tundra criterion: warmest month < 10°C.
    // Seasonal amplitude ≈ 20°C × sin(lat) (Terjung 1970).
    let seasonal_amp = 20.0 * (abs_lat * std::f32::consts::PI / 180.0).sin();
    let t_warmest = temp + seasonal_amp * 0.5;
    if t_warmest < 10.0 { return 1; } // Tundra

    // Polygon lookup: first matching polygon wins
    let p_clamped = precip.max(0.0);
    for poly in WHITTAKER_POLYGONS.iter() {
        if point_in_polygon(temp, p_clamped, poly.verts) {
            return poly.id;
        }
    }
    8 // Desert (default: arid / no polygon match)
}

fn compute_biomes_grid(
    temperature: &[f32],
    precipitation: &[f32],
    heights: &[f32],
    width: usize,
    seed: u32,
    river_map: &[f32],
) -> Vec<u8> {
    let alpine_seed = hash_u32(seed ^ 0xA1F1_E000);
    let ecotone_seed = hash_u32(seed ^ 0xEC07_0E3E);
    let size = heights.len();
    let height_grid = size / width;
    let mut biomes = vec![0_u8; size];
    for i in 0..size {
        let ex = (i % width) as f32 / width as f32;
        let ey = (i / width) as f32 / height_grid as f32;
        let jitter = value_noise3(ex * 22.0, ey * 22.0, 0.5, ecotone_seed);
        // Ecotone width: Risser (1995) measured 10–50 km biome transitions.
        // At typical gradients (0.5°C/10km, 50mm/10km), ±1.5°C and ±75mm
        // produce ~30 km and ~15 km ecotone widths respectively.
        let t_j = temperature[i] + jitter * 1.5;
        let p_j = precipitation[i] + jitter * 75.0;
        let abs_lat = ((i / width) as f32 / height_grid as f32 * 180.0 - 90.0).abs();
        let mut biome = classify_biome_whittaker(t_j, p_j, heights[i], abs_lat);
        // Alpine treeline: Körner (2003) "Alpine Plant Life" Fig. 7.1.
        // Treeline elevation decreases ~55 m per degree latitude (thermal
        // threshold: growing season T < 6.4°C).  Noise ±300 m adds local
        // variation (wind exposure, aspect, soil depth).
        if biome != 0 {
            let treeline_base = (4000.0 - 55.0 * abs_lat).max(200.0);
            let x = i % width;
            let y = i / width;
            let fx = x as f32 / width as f32;
            let fy = y as f32 / height_grid as f32;
            let n = value_noise3(fx * 14.0, fy * 14.0, seed as f32 * 0.001, alpine_seed);
            let threshold = treeline_base + n * 300.0;
            if heights[i] > threshold && heights[i] > 500.0 { biome = 10; }
        }
        // Riparian vegetation: rivers support green corridors through dry
        // biomes (Desert, Steppe, Mediterranean, Temp Grassland).
        // This is a direct effect — rivers irrigate adjacent land — not a
        // precipitation feedback (no reverse causality).
        if river_map[i] > 0.12 && matches!(biome, 4 | 5 | 8 | 11) {
            let t = temperature[i];
            biome = if t > 20.0 { 7 }       // Tropical Savanna
                    else if t > 12.0 { 9 }   // Subtropical Forest
                    else { 3 };               // Temperate Forest
        }
        biomes[i] = biome;
    }

    // Biome smoothing: 2-pass mode filter eliminates single-cell biome
    // anomalies (coastal "eyelash" fringe, isolated riparian pixels).
    // For each land cell, if fewer than 2 of its 4 cardinal neighbors
    // share its biome, replace with the most common non-ocean neighbor.
    for _ in 0..2 {
        let prev = biomes.clone();
        for i in 0..size {
            if prev[i] == 0 { continue; }
            let x = i % width;
            let y = i / width;
            let mut same = 0_u32;
            let mut counts = [0_u32; 12];
            for (dx, dy) in [(-1_i32, 0_i32), (1, 0), (0, -1), (0, 1)] {
                let nx = (x as i32 + dx).rem_euclid(width as i32) as usize;
                let ny = (y as i32 + dy).clamp(0, height_grid as i32 - 1) as usize;
                let j = ny * width + nx;
                let nb = prev[j];
                if nb == prev[i] { same += 1; }
                if nb != 0 && (nb as usize) < 12 { counts[nb as usize] += 1; }
            }
            if same < 2 {
                if let Some((best_id, _)) = counts.iter().enumerate()
                    .skip(1).max_by_key(|(_, &c)| c)
                {
                    biomes[i] = best_id as u8;
                }
            }
        }
    }

    biomes
}

/// Island-scope settlement: Miami model NPP + water access bonuses.
///
/// Same NPP core as planet scope, plus:
/// - River bonus: navigable water for trade/irrigation (Diamond 1997)
/// - Coastal bonus: maritime access, fishing, trade routes
fn compute_settlement_grid(
    biomes: &[u8],
    heights: &[f32],
    temperature: &[f32],
    precipitation: &[f32],
    river_map: &[f32],
    coastal_exposure: &[f32],
) -> Vec<f32> {
    let mut settlement = vec![0.0_f32; heights.len()];
    for i in 0..heights.len() {
        if biomes[i] == 0 {
            continue;
        }
        let t = temperature[i];
        let p = precipitation[i];
        // Miami model (Lieth 1975)
        let npp_t = 3000.0 / (1.0 + (1.315 - 0.119 * t).exp());
        let npp_p = 3000.0 * (1.0 - (-0.000664 * p).exp());
        let npp = npp_t.min(npp_p).max(0.0);
        let elev_factor = (1.0 - (heights[i] - 500.0).max(0.0) / 4000.0).max(0.0);
        let base = npp / 2500.0 * elev_factor;
        // Water access bonuses (Diamond 1997: geography of civilization)
        let river_bonus = river_map[i] * 0.25;
        let coast_bonus = coastal_exposure[i] * 0.15;
        settlement[i] = clampf(base + river_bonus + coast_bonus, 0.0, 1.0);
    }
    settlement
}

// ---------------------------------------------------------------------------
// Phase J: Braun-Willett O(n) stream power erosion (Braun & Willett 2013)
// ---------------------------------------------------------------------------

/// D8 flow routing with stochastic slope perturbation (Tucker & Bras 2000).
///
/// For each cell, find the steepest-descent neighbour using slope + noise.
/// The noise term (±5% of gradient) breaks grid-alignment bias that causes
/// radial channel artifacts on coarse grids (≥10 km/cell).  This is physically
/// motivated by turbulent variability in flow direction at sub-grid scales.
///
/// Returns `receivers[i] == i` if cell i is a local sink (no lower neighbour).
fn compute_d8_receivers(height: &[f32], noise_seed: u32, grid: &GridConfig) -> Vec<usize> {
    let w = grid.width;
    let h = grid.height;
    let mut receivers = (0..grid.size).collect::<Vec<_>>();
    let noise_amp = 0.05_f32; // 5% slope perturbation (Tucker & Bras 2000)

    for y in 0..h {
        for x in 0..w {
            let i = grid.index(x, y);
            let hi = height[i];
            let mut best_drop = 0.0_f32;
            let mut best_j = i;
            for oy in -1_i32..=1 {
                for ox in -1_i32..=1 {
                    if ox == 0 && oy == 0 {
                        continue;
                    }
                    let j = grid.neighbor(x as i32 + ox, y as i32 + oy);
                    let dist = if ox == 0 || oy == 0 { 1.0_f32 } else { std::f32::consts::SQRT_2 };
                    let slope = (hi - height[j]) / dist;
                    // Stochastic perturbation: hash-based noise per (cell, direction)
                    let dir_idx = ((oy + 1) * 3 + (ox + 1)) as u32;
                    let nh = hash_u32(noise_seed ^ (i as u32).wrapping_mul(0x9E37_79B9) ^ dir_idx);
                    let noise = (nh as f32 / 4_294_967_295.0 * 2.0 - 1.0) * noise_amp;
                    let perturbed = slope + noise * slope.abs().max(1e-6);
                    if perturbed > best_drop {
                        best_drop = perturbed;
                        best_j = j;
                    }
                }
            }
            receivers[i] = best_j;
        }
    }
    receivers
}

/// Sort cell indices from highest to lowest elevation (upstream → downstream).
/// This gives the correct processing order for drainage area accumulation.
fn topological_sort_descending(height: &[f32]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..height.len()).collect();
    order.sort_unstable_by(|&a, &b| height[b].total_cmp(&height[a]));
    order
}

/// MFD (Multiple Flow Direction) drainage area accumulation (Freeman 1991).
///
/// Distributes each cell's area to all downslope neighbours, weighted by
/// slope^p.  Process order: highest to lowest elevation (upstream → downstream).
///
/// This produces a smoother drainage area field than D8, eliminating the
/// single-pixel channel artefacts at coarse grid resolution (≥10 km/cell).
///
/// Reference: goSPL (Salles et al., Science 2023) uses MFD area at 10 km scale.
fn compute_mfd_area(
    height: &[f32],
    order: &[usize],
    dx_m: f32,
    p: f32,
    grid: &GridConfig,
) -> Vec<f32> {
    let cell_area = dx_m * dx_m;
    let mut area = vec![cell_area; grid.size];

    for &i in order.iter() {
        let y = i / grid.width;
        let x = i % grid.width;
        let hi = height[i];
        let area_i = area[i];

        let mut targets: [(usize, f32); 8] = [(0, 0.0); 8];
        let mut w_sum = 0.0_f32;
        let mut count = 0;

        for oy in -1_i32..=1 {
            for ox in -1_i32..=1 {
                if ox == 0 && oy == 0 { continue; }
                let j = grid.neighbor(x as i32 + ox, y as i32 + oy);
                let dist = if ox == 0 || oy == 0 { 1.0_f32 } else { std::f32::consts::SQRT_2 };
                let slope = (hi - height[j]) / dist;
                if slope > 0.0 {
                    let w = slope.powf(p);
                    targets[count] = (j, w);
                    w_sum += w;
                    count += 1;
                }
            }
        }

        if w_sum > 0.0 {
            let inv = 1.0 / w_sum;
            for k in 0..count {
                let (j, w) = targets[k];
                area[j] += area_i * w * inv;
            }
        }
    }
    area
}

/// Apply one implicit Braun-Willett stream power timestep with sub-grid
/// channel width scaling (Pelletier 2010; Leopold & Maddock 1953).
///
/// Stream power law:  E = K * A^m * S^n
/// Sub-grid scaling:  K_eff = K × (W_channel / dx)
///   where W_channel = k_w × Q^0.5 (Leopold & Maddock 1953 hydraulic geometry)
///   and Q ≈ A × runoff_rate (simplified).
///
/// This prevents over-erosion of 10 km cells by accounting for the fact that
/// fluvial incision only acts across the ~100 m channel width, not the full cell.
///
/// Implicit update:   h_new = (h_old + U*dt + factor * h_recv) / (1 + factor)
///                    where factor = K_eff * dt * A^m / dx^n
fn stream_power_step(
    height: &mut [f32],
    uplift: &[f32],
    k_eff: &[f32],
    dt_yr: f32,
    dx_m: f32,
    m: f32, // area exponent (canonical 0.5)
    n: f32, // slope exponent (canonical 1.0)
    kappa: f32, // hillslope diffusivity m²/yr
    mfd_p: f32, // MFD exponent: 0.0 = D8, >0.0 = MFD (Freeman 1991, recommend 1.1)
    noise_seed: u32, // seed for stochastic flow direction
    grid: &GridConfig,
) {
    let size = grid.size;

    // 1. D8 flow receivers (with stochastic slope perturbation)
    let receivers = compute_d8_receivers(height, noise_seed, grid);

    // 2. Topological sort (high → low)
    let order = topological_sort_descending(height);

    // 3. Drainage area accumulation (upstream → downstream)
    let area = if mfd_p > 0.0 {
        compute_mfd_area(height, &order, dx_m, mfd_p, grid)
    } else {
        let cell_area = dx_m * dx_m;
        let mut area = vec![cell_area; size];
        for &i in order.iter() {
            let r = receivers[i];
            if r != i {
                area[r] += area[i];
            }
        }
        area
    };

    // Sub-grid channel width scaling (Leopold & Maddock 1953; Pelletier 2010).
    // W = k_w × Q^b where b = 0.5.  With Q = A × runoff:
    //   k_w = 0.005 m^(1-b) s^(-b) (calibrated from global rivers, Leopold 1953)
    //   runoff ≈ 0.5 m/yr = 1.58e-8 m/s (global mean, Fekete et al. 2002)
    // Channel width fraction = W / dx → scales K_eff to sub-grid channel.
    let runoff_m_per_s = 1.58e-8_f32; // 0.5 m/yr
    let k_w = 0.005_f32;

    // 4. Implicit elevation update (downstream → upstream, i.e. reverse order).
    for &i in order.iter().rev() {
        if height[i] <= 0.0 {
            continue;
        }
        let r = receivers[i];
        if r == i {
            height[i] += uplift[i] * dt_yr;
            continue;
        }

        // Sub-grid channel width: W = k_w × (A × runoff)^0.5
        let discharge = area[i] * runoff_m_per_s;
        let w_channel = k_w * discharge.sqrt();
        let width_fraction = (w_channel / dx_m).clamp(0.001, 1.0);

        let k = k_eff[i] * width_fraction;
        let a_pow = area[i].powf(m);
        let factor = k * dt_yr * a_pow / dx_m.powf(n);
        // Clamp receiver height to 0: coastal cells erode to sea level, not below.
        let h_recv = height[r].max(0.0);
        height[i] = (height[i] + uplift[i] * dt_yr + factor * h_recv) / (1.0 + factor);
        height[i] = height[i].max(0.0); // land cells never erode below sea level
    }

    // 5. Hillslope diffusion (explicit, stable for kappa*dt/dx² < 0.25).
    //    Only applied to land cells (h > 0) to preserve ocean bathymetry.
    if kappa > 0.0 {
        let kdt = kappa * dt_yr;
        let inv_dx2 = 1.0 / (dx_m * dx_m);
        let mut laplacian = vec![0.0_f32; size];
        for y in 0..grid.height {
            for x in 0..grid.width {
                let i = grid.index(x, y);
                if height[i] <= 0.0 {
                    continue;
                }
                let h_c = height[i];
                let h_r = height[grid.neighbor(x as i32 + 1, y as i32)].max(0.0);
                let h_l = height[grid.neighbor(x as i32 - 1, y as i32)].max(0.0);
                let h_u = height[grid.neighbor(x as i32, y as i32 - 1)].max(0.0);
                let h_d = height[grid.neighbor(x as i32, y as i32 + 1)].max(0.0);
                laplacian[i] = (h_r + h_l + h_u + h_d - 4.0 * h_c) * inv_dx2;
            }
        }
        for i in 0..size {
            if height[i] > 0.0 {
                height[i] += kdt * laplacian[i];
                height[i] = height[i].max(0.0);
            }
        }
    }
}

/// Run `n_steps` iterations of the stream power solver.
/// Each step uses a different noise seed for stochastic flow direction
/// to prevent deterministic channel lock-in (Tucker & Bras 2000).
fn stream_power_evolve(
    height: &mut [f32],
    uplift: &[f32],
    k_eff: &[f32],
    dt_yr: f32,
    dx_m: f32,
    m: f32,
    n_exp: f32,
    kappa: f32,
    mfd_p: f32,
    n_steps: usize,
    base_seed: u32,
    grid: &GridConfig,
    progress: &mut ProgressTap<'_>,
    progress_base: f32,
    progress_span: f32,
) {
    for step in 0..n_steps {
        let t = step as f32 / n_steps.max(1) as f32;
        progress.phase(progress_base, progress_span, t);
        let step_seed = hash_u32(base_seed.wrapping_add(step as u32 * 0x85EB_CA6B));
        stream_power_step(height, uplift, k_eff, dt_yr, dx_m, m, n_exp, kappa, mfd_p, step_seed, grid);
    }
    progress.phase(progress_base, progress_span, 1.0);
}


/// Rock type determines erodibility (K_eff) for stream power erosion.
/// Values from Harel et al. (2016) global erodibility analysis.
#[derive(Clone, Copy, PartialEq)]
enum RockType {
    Granite,     // Intrusive igneous
    Quartzite,   // Metamorphic, resistant
    Basalt,      // Volcanic, moderate
    Schist,      // Foliated metamorphic
    Sandstone,   // Sedimentary, soft
    Limestone,   // Carbonate, very soft
}

impl RockType {
    /// Base erodibility K in m^{0.5}/yr.  Range spans ~8x from most resistant
    /// (granite) to least resistant (limestone).
    fn k_eff(self) -> f32 {
        match self {
            RockType::Granite   => 0.5e-6,
            RockType::Quartzite => 0.8e-6,
            RockType::Basalt    => 1.0e-6,
            RockType::Schist    => 1.2e-6,
            RockType::Sandstone => 2.0e-6,
            RockType::Limestone => 3.0e-6,
        }
    }
}


#[wasm_bindgen]
pub struct WasmSimulationResult {
    width: u32,
    height: u32,
    seed: u32,
    sea_level: f32,
    radius_km: f32,
    ocean_percent: f32,
    recomputed_layers: Vec<String>,
    plates: Vec<i16>,
    boundary_types: Vec<i8>,
    height_map: Vec<f32>,
    slope_map: Vec<f32>,
    river_map: Vec<f32>,
    lake_map: Vec<u8>,
    flow_direction: Vec<i32>,
    flow_accumulation: Vec<f32>,
    temperature_map: Vec<f32>,
    precipitation_map: Vec<f32>,
    biome_map: Vec<u8>,
    settlement_map: Vec<f32>,
    min_height: f32,
    max_height: f32,
    min_temperature: f32,
    max_temperature: f32,
    min_precipitation: f32,
    max_precipitation: f32,
    min_slope: f32,
    max_slope: f32,
}

#[wasm_bindgen]
impl WasmSimulationResult {
    #[wasm_bindgen(getter)]
    pub fn width(&self) -> u32 {
        self.width
    }

    #[wasm_bindgen(getter)]
    pub fn height(&self) -> u32 {
        self.height
    }

    #[wasm_bindgen(getter)]
    pub fn seed(&self) -> u32 {
        self.seed
    }

    #[wasm_bindgen(js_name = seaLevel)]
    pub fn sea_level(&self) -> f32 {
        self.sea_level
    }

    #[wasm_bindgen(js_name = radiusKm)]
    pub fn radius_km(&self) -> f32 {
        self.radius_km
    }

    #[wasm_bindgen(js_name = oceanPercent)]
    pub fn ocean_percent(&self) -> f32 {
        self.ocean_percent
    }

    #[wasm_bindgen(js_name = recomputedLayers)]
    pub fn recomputed_layers(&self) -> Array {
        self.recomputed_layers.iter().map(JsValue::from).collect()
    }

    pub fn plates(&self) -> Vec<i16> {
        self.plates.clone()
    }

    #[wasm_bindgen(js_name = boundaryTypes)]
    pub fn boundary_types(&self) -> Vec<i8> {
        self.boundary_types.clone()
    }

    #[wasm_bindgen(js_name = heightMap)]
    pub fn height_map(&self) -> Vec<f32> {
        self.height_map.clone()
    }

    #[wasm_bindgen(js_name = slopeMap)]
    pub fn slope_map(&self) -> Vec<f32> {
        self.slope_map.clone()
    }

    #[wasm_bindgen(js_name = riverMap)]
    pub fn river_map(&self) -> Vec<f32> {
        self.river_map.clone()
    }

    #[wasm_bindgen(js_name = lakeMap)]
    pub fn lake_map(&self) -> Vec<u8> {
        self.lake_map.clone()
    }

    #[wasm_bindgen(js_name = flowDirection)]
    pub fn flow_direction(&self) -> Vec<i32> {
        self.flow_direction.clone()
    }

    #[wasm_bindgen(js_name = flowAccumulation)]
    pub fn flow_accumulation(&self) -> Vec<f32> {
        self.flow_accumulation.clone()
    }

    #[wasm_bindgen(js_name = temperatureMap)]
    pub fn temperature_map(&self) -> Vec<f32> {
        self.temperature_map.clone()
    }

    #[wasm_bindgen(js_name = precipitationMap)]
    pub fn precipitation_map(&self) -> Vec<f32> {
        self.precipitation_map.clone()
    }

    #[wasm_bindgen(js_name = biomeMap)]
    pub fn biome_map(&self) -> Vec<u8> {
        self.biome_map.clone()
    }

    #[wasm_bindgen(js_name = settlementMap)]
    pub fn settlement_map(&self) -> Vec<f32> {
        self.settlement_map.clone()
    }

    #[wasm_bindgen(js_name = minHeight)]
    pub fn min_height(&self) -> f32 {
        self.min_height
    }

    #[wasm_bindgen(js_name = maxHeight)]
    pub fn max_height(&self) -> f32 {
        self.max_height
    }

    #[wasm_bindgen(js_name = minTemperature)]
    pub fn min_temperature(&self) -> f32 {
        self.min_temperature
    }

    #[wasm_bindgen(js_name = maxTemperature)]
    pub fn max_temperature(&self) -> f32 {
        self.max_temperature
    }

    #[wasm_bindgen(js_name = minPrecipitation)]
    pub fn min_precipitation(&self) -> f32 {
        self.min_precipitation
    }

    #[wasm_bindgen(js_name = maxPrecipitation)]
    pub fn max_precipitation(&self) -> f32 {
        self.max_precipitation
    }

    #[wasm_bindgen(js_name = minSlope)]
    pub fn min_slope(&self) -> f32 {
        self.min_slope
    }

    #[wasm_bindgen(js_name = maxSlope)]
    pub fn max_slope(&self) -> f32 {
        self.max_slope
    }
}

fn report_progress(progress_callback: Option<&js_sys::Function>, value: f32) {
    if let Some(cb) = progress_callback {
        let _ = cb.call1(
            &JsValue::NULL,
            &JsValue::from_f64(clampf(value, 0.0, 100.0) as f64),
        );
    }
}

struct ProgressTap<'a> {
    callback: Option<&'a js_sys::Function>,
    last: f32,
}

impl<'a> ProgressTap<'a> {
    fn new(callback: Option<&'a js_sys::Function>) -> Self {
        Self {
            callback,
            last: -1.0,
        }
    }

    fn emit(&mut self, value: f32) {
        let v = clampf(value, 0.0, 100.0);
        if v >= 100.0 || self.last < 0.0 || v - self.last >= 0.08 {
            report_progress(self.callback, v);
            self.last = v;
        }
    }

    fn phase(&mut self, base: f32, span: f32, t: f32) {
        self.emit(base + span * clampf(t, 0.0, 1.0));
    }
}

// ---------------------------------------------------------------------------
//  Island-as-Crop: select interesting region on planet, upsample, refine
// ---------------------------------------------------------------------------

/// Catmull–Rom cubic interpolation kernel.
#[inline]
fn catmull_rom(t: f32, p0: f32, p1: f32, p2: f32, p3: f32) -> f32 {
    let a = -0.5 * p0 + 1.5 * p1 - 1.5 * p2 + 0.5 * p3;
    let b = p0 - 2.5 * p1 + 2.0 * p2 - 0.5 * p3;
    let c = -0.5 * p0 + 0.5 * p2;
    let d = p1;
    ((a * t + b) * t + c) * t + d
}

/// Bicubic (Catmull–Rom) sample of a 2-D grid at continuous coordinates.
/// `wrap_x`: if true, x wraps (spherical planet); else clamps.
fn bicubic_sample(data: &[f32], w: usize, h: usize, fx: f32, fy: f32, wrap_x: bool) -> f32 {
    let ix = fx.floor() as i32;
    let iy = fy.floor() as i32;
    let tx = fx - ix as f32;
    let ty = fy - iy as f32;
    let w_i32 = w as i32;
    let h_i32 = h as i32;
    let mut cols = [0.0_f32; 4];
    for j in -1..=2_i32 {
        let cy = (iy + j).clamp(0, h_i32 - 1) as usize;
        let mut row = [0.0_f32; 4];
        for i in -1..=2_i32 {
            let cx = if wrap_x {
                (((ix + i) % w_i32) + w_i32) % w_i32
            } else {
                (ix + i).clamp(0, w_i32 - 1)
            } as usize;
            row[(i + 1) as usize] = data[cy * w + cx];
        }
        cols[(j + 1) as usize] = catmull_rom(tx, row[0], row[1], row[2], row[3]);
    }
    catmull_rom(ty, cols[0], cols[1], cols[2], cols[3])
}

/// Nearest-neighbour sample of a discrete (integer) grid.
fn nearest_sample_i16(data: &[i16], w: usize, h: usize, fx: f32, fy: f32, wrap_x: bool) -> i16 {
    let ix = fx.round() as i32;
    let iy = fy.round().clamp(0.0, (h as f32) - 1.0) as usize;
    let cx = if wrap_x {
        (((ix % w as i32) + w as i32) % w as i32) as usize
    } else {
        ix.clamp(0, w as i32 - 1) as usize
    };
    data[iy * w + cx]
}

fn nearest_sample_i8(data: &[i8], w: usize, h: usize, fx: f32, fy: f32, wrap_x: bool) -> i8 {
    let ix = fx.round() as i32;
    let iy = fy.round().clamp(0.0, (h as f32) - 1.0) as usize;
    let cx = if wrap_x {
        (((ix % w as i32) + w as i32) % w as i32) as usize
    } else {
        ix.clamp(0, w as i32 - 1) as usize
    };
    data[iy * w + cx]
}

/// Score every possible crop-window center on the planet and return the best one.
/// Interest = coastline cells × elevation variance × plate-boundary cells.
fn find_interesting_region(
    heights: &[f32],
    boundary_types: &[i8],
    pg: &GridConfig,
    seed: u32,
    target_km: f32,
) -> (usize, usize) {
    // Window size in planet cells
    let win_w = ((target_km / pg.km_per_cell_x).round() as usize).max(4);
    let win_h = (win_w / 2).max(2); // 2:1 aspect

    // Exclude polar rows (top/bottom 10%) — boring terrain
    let margin_y = (pg.height as f32 * 0.10) as usize;
    let y_min = margin_y;
    let y_max = pg.height.saturating_sub(margin_y + win_h);

    let score_seed = hash_u32(seed ^ 0xC80F_1A0D);
    let mut best_score = f32::MIN;
    let mut best_cx = pg.width / 2;
    let mut best_cy = pg.height / 2;

    // Step by half-window to avoid O(W²×H²) full scan
    let step = (win_w / 2).max(1);
    for wy in (y_min..=y_max).step_by(step.max(1)) {
        for wx in (0..pg.width).step_by(step.max(1)) {
            let mut coast = 0_u32;
            let mut boundary = 0_u32;
            let mut h_min = f32::MAX;
            let mut h_max = f32::MIN;
            let mut land_count = 0_u32;
            for dy in 0..win_h {
                let y = wy + dy;
                if y >= pg.height { break; }
                for dx in 0..win_w {
                    let x = (wx + dx) % pg.width;
                    let i = y * pg.width + x;
                    let h = heights[i];
                    if h > 0.0 { land_count += 1; }
                    if h < h_min { h_min = h; }
                    if h > h_max { h_max = h; }
                    if boundary_types[i] != 0 { boundary += 1; }
                    // Coast = land cell with any cardinal ocean neighbor
                    if h > 0.0 {
                        let has_ocean =
                            heights[y * pg.width + (x + 1) % pg.width] <= 0.0
                            || (x > 0 && heights[y * pg.width + x - 1] <= 0.0)
                            || (y > 0 && heights[(y - 1) * pg.width + x] <= 0.0)
                            || (y + 1 < pg.height && heights[(y + 1) * pg.width + x] <= 0.0);
                        if has_ocean { coast += 1; }
                    }
                }
            }
            let total = (win_w * win_h) as f32;
            let land_frac = land_count as f32 / total;
            // Want mixed land/ocean (0.2–0.8 land fraction is most interesting)
            let mix_score = 1.0 - (land_frac - 0.4).abs() * 2.5;
            let coast_norm = (coast as f32 / total).min(1.0);
            let elev_norm = ((h_max - h_min).max(1.0) / 5000.0).min(1.0);
            let boundary_norm = 1.0 + (boundary as f32 / total).min(0.5) * 2.0;
            // Seed-based noise to vary selection per seed
            let nx = wx as f32 / pg.width as f32;
            let ny = wy as f32 / pg.height as f32;
            let noise = value_noise3(nx * 3.0, ny * 3.0, seed as f32 * 0.001, score_seed) * 0.3;

            let score = mix_score * coast_norm * elev_norm * boundary_norm + noise;
            if score > best_score {
                best_score = score;
                best_cx = wx + win_w / 2;
                best_cy = wy + win_h / 2;
            }
        }
    }
    (best_cx % pg.width, best_cy.min(pg.height - 1))
}

/// Main Island-as-Crop pipeline: crop planet data → upsample → refine.
fn run_island_crop(
    cfg: &SimulationConfig,
    planet_heights: &[f32],
    planet_plates: &[i16],
    planet_boundary_types: &[i8],
    planet_grid: &GridConfig,
    planet_cell_cache: &CellCache,
    detail: DetailProfile,
    recomputed_layers: Vec<String>,
    progress: &mut ProgressTap<'_>,
    progress_base: f32,
    progress_span: f32,
) -> WasmSimulationResult {
    let island_w = ISLAND_WIDTH;
    let island_h = ISLAND_HEIGHT;
    let island_scale_km = cfg.island_scale_km.unwrap_or(400.0).clamp(50.0, 2000.0);
    let km_per_cell = island_scale_km / island_w as f32;
    let dx_m = km_per_cell * 1000.0;
    let island_grid = GridConfig::island(island_w, island_h, km_per_cell);
    let seed = cfg.seed;

    // --- 1. Find interesting region on planet ---
    let (cx, cy) = find_interesting_region(
        planet_heights, planet_boundary_types, planet_grid, seed, island_scale_km,
    );
    progress.phase(progress_base, progress_span, 0.02);

    // Planet-space coordinates of the crop center
    let center_lat = planet_cell_cache.lat_deg[cy * planet_grid.width + (cx % planet_grid.width)];
    let center_lon = planet_cell_cache.lon_deg[cy * planet_grid.width + (cx % planet_grid.width)];

    // Build island CellCache with correct planetary coordinates
    let island_cell_cache = CellCache::for_island_crop(&island_grid, center_lat, center_lon);

    // Window size in planet cells
    let pw = planet_grid.width;
    let ph = planet_grid.height;
    let win_w_cells = (island_scale_km / planet_grid.km_per_cell_x).round() as usize;
    let win_h_cells = ((island_scale_km * 0.5) / planet_grid.km_per_cell_y).round() as usize;
    let half_w = win_w_cells / 2;
    let half_h = win_h_cells / 2;

    // --- 2. Crop + bicubic upsample relief ---
    let mut relief = vec![0.0_f32; island_grid.size];
    let mut plates_field = vec![0_i16; island_grid.size];
    let mut bnd_types = vec![0_i8; island_grid.size];
    let noise_seed = hash_u32(seed ^ 0xFBFD_E7A1);

    for iy in 0..island_h {
        for ix in 0..island_w {
            let i = iy * island_w + ix;
            // Map island cell → planet coordinate
            let fx_norm = ix as f32 / island_w as f32; // 0..1
            let fy_norm = iy as f32 / island_h as f32;
            let px = (cx as f32 - half_w as f32) + fx_norm * win_w_cells as f32;
            let py = (cy as f32 - half_h as f32) + fy_norm * win_h_cells as f32;

            // Bicubic for height (smooth)
            let h_coarse = bicubic_sample(planet_heights, pw, ph, px, py, planet_grid.is_spherical);
            // Nearest for discrete fields
            plates_field[i] = nearest_sample_i16(planet_plates, pw, ph, px, py, planet_grid.is_spherical);
            bnd_types[i] = nearest_sample_i8(planet_boundary_types, pw, ph, px, py, planet_grid.is_spherical);

            // FBM fractal detail: adds sub-20km features
            // Amplitude scales with local elevation magnitude (mountains get more detail)
            let nx = island_cell_cache.noise_x[i];
            let ny = island_cell_cache.noise_y[i];
            let nz = island_cell_cache.noise_z[i];
            // Sub-grid variance 10-25% of summit elevation (Montgomery & Brandon
            // 2002, Earth Planet. Sci. Lett.); 15% is the mid-range.
            let scale = h_coarse.abs().max(50.0) * 0.15;
            let mut fbm = 0.0_f32;
            let mut freq = 8.0_f32;
            let mut amp = 1.0_f32;
            for oct in 0..6_u32 {
                let s = hash_u32(noise_seed ^ (oct * 0x1337));
                fbm += value_noise3(nx * freq, ny * freq, nz * freq + oct as f32 * 0.7, s) * amp;
                freq *= 2.0;
                amp *= 0.62; // Hurst H = 0.7 → decay = 2^(-H) ≈ 0.62 (Huang & Turcotte 1989)
            }

            relief[i] = h_coarse + fbm * scale;
        }
    }
    progress.phase(progress_base, progress_span, 0.10);

    // --- 3. Edge fade: smooth terrain to ocean at grid edges ---
    // Procedural: margin width ~6%, noise ±4%, submersion −500 m
    // (upper continental slope depth, Kennett 1982).
    {
        let w = island_w as f32;
        let h = island_h as f32;
        let fade_seed = hash_u32(seed ^ 0xFADE_C80F);
        for y in 0..island_h {
            for x in 0..island_w {
                let i = y * island_w + x;
                let fx = (x as f32 + 0.5) / w;
                let fy = (y as f32 + 0.5) / h;
                let edge = fx.min(1.0 - fx).min(fy).min(1.0 - fy);
                let noise = value_noise3(fx * 5.5, fy * 5.5, 0.31, fade_seed) * 0.04;
                let margin = (0.06 + noise).max(0.03);
                if edge < margin {
                    let t = (edge / margin).clamp(0.0, 1.0);
                    let s = t * t * (3.0 - 2.0 * t);
                    relief[i] = relief[i] * s - 500.0 * (1.0 - s);
                }
            }
        }
    }

    // --- 4. Fine-scale stream power erosion (10 steps) ---
    // Use same rock-type erodibility as planet scope (Harel et al. 2016).
    // Island cells inherit boundary types → map to RockType → K_eff.
    // ×1.5 island boost (higher resolution reveals finer channels).
    let mut k_eff = vec![0.0_f32; island_grid.size];
    for i in 0..island_grid.size {
        let rock = if bnd_types[i] == 1 {
            RockType::Quartzite  // convergent metamorphic, K=0.8e-6
        } else if bnd_types[i] == 2 {
            RockType::Basalt     // divergent volcanic, K=1.0e-6
        } else {
            RockType::Sandstone  // interior sedimentary, K=2.0e-6
        };
        k_eff[i] = rock.k_eff() * 1.5;
    }
    // Zero uplift for fine-scale pass (planet already provided the large-scale topography)
    let uplift_zero = vec![0.0_f32; island_grid.size];
    let island_erosion_seed = hash_u32(seed ^ 0x151A_ED00);
    stream_power_evolve(
        &mut relief,
        &uplift_zero,
        &k_eff,
        100_000.0,  // dt = 100 kyr
        dx_m,
        0.5, 1.0,   // m, n (standard stream power exponents)
        0.01,        // kappa (hillslope diffusion, CFL-safe)
        0.0,         // mfd_p: 0 = D8 (correct at island ≤1 km/cell)
        10,          // 10 steps
        island_erosion_seed,
        &island_grid,
        progress,
        progress_base + progress_span * 0.10,
        progress_span * 0.30,
    );

    // --- 5. Pre-carve slope + hydrology ---
    let (slope_pre, _, _) = compute_slope_grid(&relief, island_w, island_h, progress,
        progress_base + progress_span * 0.40, progress_span * 0.03);
    let (_, flow_pre, river_pre, _) = compute_hydrology_grid(
        &relief, &slope_pre, island_w, island_h, detail, progress,
        progress_base + progress_span * 0.43, progress_span * 0.05);

    // --- 6. Carve fluvial valleys ---
    carve_fluvial_valleys_grid(
        &mut relief, &flow_pre, &river_pre, island_w, island_h, km_per_cell, detail, progress,
        progress_base + progress_span * 0.48, progress_span * 0.04);

    // --- 7. Post-carve edge fade ---
    {
        let w = island_w as f32;
        let h = island_h as f32;
        let fade_seed = hash_u32(seed ^ 0xFADE_C0DE);
        for y in 0..island_h {
            for x in 0..island_w {
                let i = y * island_w + x;
                let fx = (x as f32 + 0.5) / w;
                let fy = (y as f32 + 0.5) / h;
                let edge = fx.min(1.0 - fx).min(fy).min(1.0 - fy);
                let noise = value_noise3(fx * 5.5, fy * 5.5, 0.77, fade_seed) * 0.03;
                let margin = (0.05 + noise).max(0.03);
                if edge < margin && relief[i] > 0.0 {
                    let t = (edge / margin).clamp(0.0, 1.0);
                    let s = t * t * (3.0 - 2.0 * t);
                    relief[i] = relief[i] * s - 500.0 * (1.0 - s);
                }
            }
        }
    }

    // --- 8. Final slope + hydrology ---
    let (slope_map, min_slope, max_slope) = compute_slope_grid(&relief, island_w, island_h,
        progress, progress_base + progress_span * 0.52, progress_span * 0.04);
    let (flow_direction, flow_accumulation, river_map, lake_map) = compute_hydrology_grid(
        &relief, &slope_map, island_w, island_h, detail, progress,
        progress_base + progress_span * 0.56, progress_span * 0.06);

    // --- 9. Climate (unified model with real planetary coordinates) ---
    let aerosol = 0.0_f32; // no events for island crop
    let (
        temperature_map,
        precipitation_map,
        min_temperature,
        max_temperature,
        min_precipitation,
        max_precipitation,
    ) = compute_climate_unified(
        &cfg.planet,
        &relief,
        &island_grid,
        &island_cell_cache,
        seed,
        aerosol,
        progress,
        progress_base + progress_span * 0.62,
        progress_span * 0.18,
    );

    // --- 10. Biomes (smoothed T/P for stable classification) ---
    let smooth = |src: &[f32]| -> Vec<f32> {
        let mut out = src.to_vec();
        let mut scratch = vec![0.0_f32; island_grid.size];
        for _ in 0..5 {
            for y in 0..island_h {
                for x in 0..island_w {
                    let i = y * island_w + x;
                    let mut sum = out[i] * 4.0;
                    let mut wt = 4.0_f32;
                    for (ddx, ddy) in [(-1_i32, 0_i32), (1, 0), (0, -1), (0, 1)] {
                        let nx = (x as i32 + ddx).clamp(0, island_w as i32 - 1) as usize;
                        let ny = (y as i32 + ddy).clamp(0, island_h as i32 - 1) as usize;
                        sum += out[ny * island_w + nx];
                        wt += 1.0;
                    }
                    scratch[i] = sum / wt;
                }
            }
            out.copy_from_slice(&scratch);
        }
        out
    };
    // Coastline morphological cleanup (island scope)
    smooth_coastline(&mut relief, island_w, island_h, false);

    let temp_smooth = smooth(&temperature_map);
    let precip_smooth = smooth(&precipitation_map);
    let biome_map = compute_biomes_grid(
        &temp_smooth, &precip_smooth, &relief, island_w, seed, &river_map);
    progress.phase(progress_base, progress_span, 0.85);

    // --- 11. Settlement ---
    let coastal_exposure = compute_coastal_exposure_grid(&relief, island_w, island_h);
    let settlement_map = compute_settlement_grid(
        &biome_map, &relief, &temperature_map, &precipitation_map,
        &river_map, &coastal_exposure);
    progress.phase(progress_base, progress_span, 0.92);

    let (min_height, max_height) = min_max(&relief);
    let ocean_cells = relief.iter().filter(|&&h| h < 0.0).count() as f32;
    let ocean_percent = 100.0 * ocean_cells / island_grid.size as f32;

    progress.phase(progress_base, progress_span, 1.0);

    WasmSimulationResult {
        width: island_w as u32,
        height: island_h as u32,
        seed,
        sea_level: 0.0,
        radius_km: cfg.planet.radius_km,
        ocean_percent,
        recomputed_layers,
        plates: plates_field,
        boundary_types: bnd_types,
        height_map: relief,
        slope_map,
        river_map,
        lake_map,
        flow_direction,
        flow_accumulation,
        temperature_map,
        precipitation_map,
        biome_map,
        settlement_map,
        min_height,
        max_height,
        min_temperature,
        max_temperature,
        min_precipitation,
        max_precipitation,
        min_slope,
        max_slope,
    }
}

fn run_simulation_internal(
    config: JsValue,
    reason: String,
    progress_callback: Option<&js_sys::Function>,
) -> Result<WasmSimulationResult, JsValue> {
    let mut progress = ProgressTap::new(progress_callback);
    progress.emit(0.0);
    let mut cfg: SimulationConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("config deserialize error: {e}")))?;
    progress.emit(1.0);

    cfg.events = cfg.events.into_iter().map(ensure_event_energy).collect();
    let recomputed_layers = evaluate_recompute(parse_reason(&reason));
    let preset = parse_generation_preset(cfg.generation_preset.as_deref());
    let detail = detail_profile(preset);
    let scope = parse_scope(cfg.scope.as_deref());

    // Planet generation is always run first (island = crop of planet).
    // Progress allocation: planet 0-70% for island scope, 0-78% for planet scope.
    let is_island = scope == GenerationScope::Island;
    let planet_progress_scale = if is_island { 0.70 } else { 1.0 };

    let cache = world_cache();

    let plates_layer = compute_plates(
        &cfg.planet,
        &cfg.tectonics,
        detail,
        cfg.seed,
        cache,
        &mut progress,
        2.0 * planet_progress_scale,
        22.0 * planet_progress_scale,
    );
    let planet_grid = GridConfig::planet();
    let planet_cell_cache = CellCache::for_planet(&planet_grid);
    let relief_raw = compute_relief_physics(
        &cfg.planet,
        &cfg.tectonics,
        &plates_layer,
        cfg.seed,
        &planet_grid,
        &planet_cell_cache,
        detail,
        &mut progress,
        24.0 * planet_progress_scale,
        50.0 * planet_progress_scale,
    );
    progress.emit(74.0 * planet_progress_scale);
    let (event_relief, aerosol) = apply_events(&cfg.planet, &relief_raw.relief, &cfg.events, cache);
    progress.emit(78.0 * planet_progress_scale);

    // --- Island scope: crop from planet and refine ---
    if is_island {
        let result = run_island_crop(
            &cfg,
            &event_relief,
            &plates_layer.plate_field,
            &plates_layer.boundary_types,
            &planet_grid,
            &planet_cell_cache,
            detail,
            recomputed_layers,
            &mut progress,
            70.0,  // progress_base (70-100%)
            29.9,  // progress_span
        );
        progress.emit(99.9);
        return Ok(result);
    }

    let (slope_map, min_slope, max_slope) = compute_slope(&event_relief, &mut progress, 78.0, 4.0);

    // Climate BEFORE hydrology: precipitation drives rivers, not the reverse.
    let (
        temperature_map,
        precipitation_map,
        min_temperature,
        max_temperature,
        min_precipitation,
        max_precipitation,
    ) = compute_climate_unified(
        &cfg.planet,
        &event_relief,
        &planet_grid,
        &planet_cell_cache,
        cfg.seed,
        aerosol,
        &mut progress,
        82.0,
        10.0,
    );

    let (flow_direction, flow_accumulation, river_map, lake_map) =
        compute_hydrology_grid(
            &event_relief,
            &slope_map,
            planet_grid.width,
            planet_grid.height,
            detail,
            &mut progress,
            92.0,
            5.0,
        );

    let biome_map = compute_biomes_grid(
        &temperature_map,
        &precipitation_map,
        &event_relief,
        planet_grid.width,
        cfg.seed,
        &river_map,
    );
    progress.emit(98.0);
    let settlement_map = compute_settlement(
        &biome_map,
        &event_relief,
        &temperature_map,
        &precipitation_map,
    );
    progress.emit(99.7);
    let (min_height, max_height) = min_max(&event_relief);
    let result = WasmSimulationResult {
        width: WORLD_WIDTH as u32,
        height: WORLD_HEIGHT as u32,
        seed: cfg.seed,
        sea_level: relief_raw.sea_level,
        radius_km: cfg.planet.radius_km,
        ocean_percent: cfg.planet.ocean_percent,
        recomputed_layers,
        plates: plates_layer.plate_field,
        boundary_types: plates_layer.boundary_types,
        height_map: event_relief,
        slope_map,
        river_map,
        lake_map,
        flow_direction,
        flow_accumulation,
        temperature_map,
        precipitation_map,
        biome_map,
        settlement_map,
        min_height,
        max_height,
        min_temperature,
        max_temperature,
        min_precipitation,
        max_precipitation,
        min_slope,
        max_slope,
    };

    // Keep progress <100 until JS wrapper and worker finish marshalling + posting result.
    progress.emit(99.9);
    Ok(result)
}

#[wasm_bindgen]
pub fn run_simulation(config: JsValue, reason: String) -> Result<WasmSimulationResult, JsValue> {
    run_simulation_internal(config, reason, None)
}

#[wasm_bindgen(js_name = run_simulation_with_progress)]
pub fn run_simulation_with_progress(
    config: JsValue,
    reason: String,
    progress_callback: js_sys::Function,
) -> Result<WasmSimulationResult, JsValue> {
    run_simulation_internal(config, reason, Some(&progress_callback))
}
