use js_sys::Array;
use serde::{Deserialize, Serialize};
use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, VecDeque};
use std::sync::OnceLock;
use wasm_bindgen::prelude::*;

const WORLD_WIDTH: usize = 2048;
const WORLD_HEIGHT: usize = 1024;
const WORLD_SIZE: usize = WORLD_WIDTH * WORLD_HEIGHT;
const ISLAND_WIDTH: usize = 1024;
const ISLAND_HEIGHT: usize = 512;
const ISLAND_SIZE: usize = ISLAND_WIDTH * ISLAND_HEIGHT;
const RADIANS: f32 = std::f32::consts::PI / 180.0;
const KILOMETERS_PER_DEGREE: f32 = 111.319_490_793;

#[derive(Clone)]
struct Rng {
    state: u32,
}

impl Rng {
    fn new(seed: u32) -> Self {
        Self { state: seed }
    }

    fn next_f32(&mut self) -> f32 {
        self.state = self
            .state
            .wrapping_mul(1_664_525)
            .wrapping_add(1_013_904_223);
        self.state as f32 / 4_294_967_296.0
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
fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    if (edge1 - edge0).abs() < 1e-6 {
        return if x < edge0 { 0.0 } else { 1.0 };
    }
    let t = clampf((x - edge0) / (edge1 - edge0), 0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[inline]
fn ridge(value: f32) -> f32 {
    (1.0 - value.abs()).max(0.0)
}

#[inline]
fn lerpf(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

#[inline]
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
        let rx = x * 0.82 - y * 0.46 + z * 0.33;
        let ry = x * 0.51 + y * 0.79 - z * 0.28;
        let rz = -x * 0.24 + y * 0.41 + z * 0.88;
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

fn quantile_sorted(sorted_values: &[f32], q: f32) -> f32 {
    if sorted_values.is_empty() {
        return 0.0;
    }

    let t = clampf(q, 0.0, 1.0) * (sorted_values.len() as f32 - 1.0);
    let lo = t.floor() as usize;
    let hi = (lo + 1).min(sorted_values.len() - 1);
    let k = t - lo as f32;
    sorted_values[lo] * (1.0 - k) + sorted_values[hi] * k
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
}

impl GridConfig {
    /// Standard planet scope preset (equirectangular, WORLD_WIDTH x WORLD_HEIGHT).
    fn planet() -> Self {
        let km_y = (std::f32::consts::PI * 6371.0) / WORLD_HEIGHT as f32; // ~19.6 km
        let km_x = (2.0 * std::f32::consts::PI * 6371.0) / WORLD_WIDTH as f32; // ~19.6 km at equator
        Self {
            width: WORLD_WIDTH,
            height: WORLD_HEIGHT,
            size: WORLD_SIZE,
            is_spherical: true,
            km_per_cell_x: km_x,
            km_per_cell_y: km_y,
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
        }
    }

    /// Row-major index from (x, y) coordinates.
    #[inline]
    fn index(&self, x: usize, y: usize) -> usize {
        y * self.width + x
    }

    /// Reverse: index → (x, y).
    #[inline]
    fn index_to_xy(&self, i: usize) -> (usize, usize) {
        (i % self.width, i / self.width)
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

    /// Area of a single cell in km², accounting for latitude on a sphere.
    #[inline]
    fn cell_area_km2(&self, y: usize) -> f32 {
        if self.is_spherical {
            let lat_rad = (0.5 - (y as f32 + 0.5) / self.height as f32) * std::f32::consts::PI;
            self.km_per_cell_x * self.km_per_cell_y * lat_rad.cos().abs()
        } else {
            self.km_per_cell_x * self.km_per_cell_y
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

    /// Build cache for island scope — flat grid with synthetic 3D coords for noise.
    /// `center_lat_deg` is the latitude of the island center (affects climate).
    /// `seed_phase` provides a unique z-offset so different islands get different noise.
    fn for_island(grid: &GridConfig, center_lat_deg: f32, seed_phase: f32) -> Self {
        let w = grid.width as f32;
        let h = grid.height as f32;
        let total_height_km = grid.height as f32 * grid.km_per_cell_y;

        let mut noise_x = vec![0.0_f32; grid.size];
        let mut noise_y = vec![0.0_f32; grid.size];
        let mut noise_z = vec![seed_phase; grid.size];
        let mut lat_deg = vec![0.0_f32; grid.size];
        let mut lon_deg_v = vec![0.0_f32; grid.size];

        for y in 0..grid.height {
            let y_km = (y as f32 + 0.5) * grid.km_per_cell_y - total_height_km * 0.5;
            let lat_offset_deg = y_km / 111.0;
            let lat = center_lat_deg - lat_offset_deg;

            for x in 0..grid.width {
                let i = grid.index(x, y);

                // Normalize coordinates to [-1, 1] to match planet-scope spherical
                // coordinate range. This ensures the noise functions (spherical_fbm,
                // value_noise3) produce continent-scale features rather than
                // microscale texture (which would happen if we used x_km/R_earth ≈ 0.06).
                noise_x[i] = 2.0 * (x as f32 + 0.5) / w - 1.0;
                noise_y[i] = 2.0 * (y as f32 + 0.5) / h - 1.0;
                noise_z[i] = seed_phase;

                lat_deg[i] = lat;
                lon_deg_v[i] = (x as f32 / w - 0.5) * 10.0;
            }
        }

        Self { noise_x, noise_y, noise_z, lat_deg, lon_deg: lon_deg_v }
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
    buoyancy_smooth_passes: usize,
    erosion_rounds: usize,
    fluvial_rounds: usize,
    ocean_smooth_passes: usize,
    max_kernel_radius: i32,
    fault_iterations: usize,
    fault_smooth_passes: usize,
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
            buoyancy_smooth_passes: 1,
            erosion_rounds: 0,
            fluvial_rounds: 0,
            ocean_smooth_passes: 0,
            max_kernel_radius: 3,
            fault_iterations: 720,
            fault_smooth_passes: 1,
            plate_evolution_steps: 2,
        },
        GenerationPreset::Fast => DetailProfile {
            buoyancy_smooth_passes: 6,
            erosion_rounds: 1,
            fluvial_rounds: 1,
            ocean_smooth_passes: 1,
            max_kernel_radius: 4,
            fault_iterations: 1300,
            fault_smooth_passes: 1,
            plate_evolution_steps: 4,
        },
        GenerationPreset::Detailed => DetailProfile {
            buoyancy_smooth_passes: 18,
            erosion_rounds: 3,
            fluvial_rounds: 3,
            ocean_smooth_passes: 3,
            max_kernel_radius: 6,
            fault_iterations: 3400,
            fault_smooth_passes: 2,
            plate_evolution_steps: 10,
        },
        GenerationPreset::Balanced => DetailProfile {
            buoyancy_smooth_passes: 14,
            erosion_rounds: 2,
            fluvial_rounds: 2,
            ocean_smooth_passes: 2,
            max_kernel_radius: 5,
            fault_iterations: 2600,
            fault_smooth_passes: 2,
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

fn parse_island_type(value: Option<&str>) -> IslandType {
    match value.unwrap_or("continental") {
        "arc" => IslandType::Arc,
        "hotspot" => IslandType::Hotspot,
        "rift" => IslandType::Rift,
        _ => IslandType::Continental,
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
    speed: f32,
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
    speed: f32,
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
    boundary_normal_x: Vec<f32>,
    boundary_normal_y: Vec<f32>,
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

#[derive(Clone, Copy, Debug)]
struct DistanceNode {
    cost: f32,
    index: usize,
}

impl PartialEq for DistanceNode {
    fn eq(&self, other: &Self) -> bool {
        self.cost.to_bits() == other.cost.to_bits() && self.index == other.index
    }
}

impl Eq for DistanceNode {}

impl PartialOrd for DistanceNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for DistanceNode {
    fn cmp(&self, other: &Self) -> Ordering {
        self.cost
            .total_cmp(&other.cost)
            .then_with(|| self.index.cmp(&other.index))
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

        for (dx, dy, w) in STEPS {
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
            let drift_align = dx as f32 * gp.drift_x + dy as f32 * gp.drift_y;
            let drift_factor = 1.03 - 0.12 * drift_align;
            let structural = structural_field[j];
            let structure_factor = clampf(
                0.62 + 0.98 * structural + (0.36 - gp.roughness * 0.17),
                0.45,
                1.9,
            );
            let polar_factor = 1.0 + (lat.abs() / 90.0) * 0.1;
            let step_cost =
                (w * gp.spread * rough_factor * drift_factor * structure_factor * polar_factor)
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
        let step_years =
            random_range(&mut rng, 900_000.0, 3_400_000.0) * (0.92 + 0.32 * age_norm);
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
            speed,
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
            speed: plate.speed,
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
    let mut boundary_normal_x = vec![0.0_f32; WORLD_SIZE];
    let mut boundary_normal_y = vec![0.0_f32; WORLD_SIZE];
    let mut boundary_strength = vec![0.0_f32; WORLD_SIZE];
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

    for y in 0..WORLD_HEIGHT {
        progress.phase(
            progress_base + progress_span * 0.45,
            progress_span * 0.55,
            y as f32 / WORLD_HEIGHT as f32,
        );
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
                let n_len = (dx as f32).hypot(dy as f32).max(1.0);
                let nx = dx as f32 / n_len;
                let ny = dy as f32 / n_len;
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
                boundary_normal_x[i] = 0.0;
                boundary_normal_y[i] = 0.0;
                boundary_strength[i] = 0.0;
                continue;
            }

            let denom = wsum.max(1e-6);
            let conv = conv_sum / denom;
            let div = div_sum / denom;
            let shear = shear_sum / denom;

            let n_len = normal_x.hypot(normal_y);
            if n_len > 1e-5 {
                boundary_normal_x[i] = normal_x / n_len;
                boundary_normal_y[i] = normal_y / n_len;
            } else {
                boundary_normal_x[i] = 0.0;
                boundary_normal_y[i] = 1.0;
            }

            let strength = conv.max(div).max(shear * 0.82);
            boundary_strength[i] = clampf(strength / boundary_scale, 0.0, 1.0);

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
        boundary_normal_x,
        boundary_normal_y,
        boundary_strength,
        plate_vectors,
    }
}

fn build_fault_backbone(
    seed: u32,
    iterations: usize,
    smooth_passes: usize,
    grid: &GridConfig,
    progress: &mut ProgressTap<'_>,
    progress_base: f32,
    progress_span: f32,
) -> Vec<f32> {
    let width = grid.width;
    let height = grid.height;
    let width_i32 = width as i32;
    let height_i32 = height as i32;
    let x_half = width / 2;

    // Column-major storage (x-major) to match classic worldgen fault accumulation.
    let mut fault_col = vec![f32::NAN; grid.size];
    for x in 0..width {
        fault_col[x * height] = 0.0;
    }

    let mut sin_iter_phi = vec![0.0_f32; width * 2];
    let step = 2.0 * std::f32::consts::PI / width as f32;
    for i in 0..width {
        let s = (i as f32 * step).sin();
        sin_iter_phi[i] = s;
        sin_iter_phi[i + width] = s;
    }

    let mut rng = Rng::new(seed ^ 0xA341_316C);
    let y_div2 = height as f32 * 0.5;
    let y_div_pi = height as f32 / std::f32::consts::PI;

    let fault_iterations = iterations.max(1);
    for iter in 0..fault_iterations {
        if iter % 24 == 0 {
            let t = iter as f32 / fault_iterations as f32;
            progress.phase(progress_base, progress_span, 0.8 * t);
        }

        let flag_negative = rng.next_f32() < 0.5;
        let alpha = (rng.next_f32() - 0.5) * std::f32::consts::PI;
        let beta = (rng.next_f32() - 0.5) * std::f32::consts::PI;
        let cos_term = clampf(alpha.cos() * beta.cos(), -1.0, 1.0);
        let tan_b = cos_term.acos().tan();
        let xsi = (width as f32 * 0.5 - (width as f32 / std::f32::consts::PI) * beta).round() as i32;

        for phi in 0..x_half {
            let sin_idx = (xsi - phi as i32 + width_i32).clamp(0, (2 * width - 1) as i32) as usize;
            let theta = (y_div_pi * (sin_iter_phi[sin_idx] * tan_b).atan() + y_div2).round() as i32;
            let y = theta.clamp(0, height_i32 - 1) as usize;
            let idx = phi * height + y;
            let delta = if flag_negative { -1.0 } else { 1.0 };
            if fault_col[idx].is_nan() {
                fault_col[idx] = delta;
            } else {
                fault_col[idx] += delta;
            }
        }
    }
    progress.phase(progress_base, progress_span, 0.82);

    // Mirror the second hemisphere from the first (same trick as classic algorithm).
    let offset = x_half * height;
    for x in 0..x_half {
        let row = x * height;
        for y in 1..height {
            fault_col[row + offset + (height - y)] = fault_col[row + y];
        }
    }

    // Integrate step-lines into signed relief profile for each longitude.
    for x in 0..width {
        let row = x * height;
        let mut acc = if fault_col[row].is_nan() {
            0.0
        } else {
            fault_col[row]
        };
        fault_col[row] = acc;
        for y in 1..height {
            let cur = fault_col[row + y];
            if !cur.is_nan() {
                acc += cur;
            }
            fault_col[row + y] = acc;
        }
    }

    let mut min_v = f32::INFINITY;
    let mut max_v = -f32::INFINITY;
    for &v in fault_col.iter() {
        if v.is_nan() {
            continue;
        }
        min_v = min_v.min(v);
        max_v = max_v.max(v);
    }
    let mid = (min_v + max_v) * 0.5;
    let half_span = ((max_v - min_v) * 0.5).max(1e-4);

    let mut field = vec![0.0_f32; grid.size];
    for y in 0..height {
        for x in 0..width {
            let v = fault_col[x * height + y];
            let n = clampf((v - mid) / half_span, -1.0, 1.0);
            field[grid.index(x, y)] = n;
        }
    }
    progress.phase(progress_base, progress_span, 0.9);

    if smooth_passes == 0 {
        progress.phase(progress_base, progress_span, 1.0);
        return field;
    }

    let mut scratch = field.clone();
    for pass in 0..smooth_passes {
        for y in 0..grid.height {
            for x in 0..grid.width {
                let i = grid.index(x, y);
                let mut sum = field[i] * 0.46;
                let mut wsum = 0.46;
                for oy in -1..=1 {
                    for ox in -1..=1 {
                        if ox == 0 && oy == 0 {
                            continue;
                        }
                        let j = grid.neighbor(x as i32 + ox, y as i32 + oy);
                        let w = if ox == 0 || oy == 0 { 0.1 } else { 0.065 };
                        sum += field[j] * w;
                        wsum += w;
                    }
                }
                scratch[i] = sum / wsum.max(1e-6);
            }
        }
        field.copy_from_slice(&scratch);
        let t = (pass as f32 + 1.0) / smooth_passes as f32;
        progress.phase(progress_base, progress_span, 0.9 + 0.1 * t);
    }

    field
}

fn build_continent_blob_field(seed: u32, grid: &GridConfig, cell_cache: &CellCache) -> Vec<f32> {
    #[derive(Clone, Copy)]
    struct Blob {
        x: f32,
        y: f32,
        z: f32,
        amp: f32,
        sharpness: f32,
    }

    let mut rng = Rng::new(seed ^ 0x85EB_CA6B);
    let pos_count = 13usize;
    let neg_count = 9usize;
    let mut blobs: Vec<Blob> = Vec::with_capacity(pos_count + neg_count);

    let mut make_blob = |amp_min: f32, amp_max: f32, sh_min: f32, sh_max: f32| -> Blob {
        let z = random_range(&mut rng, -1.0, 1.0);
        let lon = random_range(
            &mut rng,
            -std::f32::consts::PI,
            std::f32::consts::PI,
        );
        let r = (1.0 - z * z).max(0.0).sqrt();
        Blob {
            x: r * lon.cos(),
            y: r * lon.sin(),
            z,
            amp: random_range(&mut rng, amp_min, amp_max),
            sharpness: random_range(&mut rng, sh_min, sh_max),
        }
    };

    for _ in 0..pos_count {
        blobs.push(make_blob(0.65, 1.3, 3.8, 9.4));
    }
    for _ in 0..neg_count {
        blobs.push(make_blob(-1.15, -0.42, 5.5, 13.0));
    }

    let mut out = vec![0.0_f32; grid.size];
    let mut min_v = f32::INFINITY;
    let mut max_v = -f32::INFINITY;

    for i in 0..grid.size {
        let sx = cell_cache.noise_x[i];
        let sy = cell_cache.noise_y[i];
        let sz = cell_cache.noise_z[i];
        let mut sum = 0.0_f32;

        for blob in blobs.iter() {
            let dot = clampf(sx * blob.x + sy * blob.y + sz * blob.z, -1.0, 1.0);
            let kernel = ((dot - 1.0) * blob.sharpness).exp();
            sum += blob.amp * kernel;
        }

        // Mild extra isotropic perturbation to avoid perfect radial blobs.
        let jitter = (sx * 2.9 + sy * 3.4 + sz * 2.1 + seed as f32 * 0.0011).sin() * 0.12
            + (sy * 3.1 - sz * 2.6 + sx * 2.4 - seed as f32 * 0.0013).cos() * 0.08;
        let v = sum + jitter;
        min_v = min_v.min(v);
        max_v = max_v.max(v);
        out[i] = v;
    }

    let mid = (min_v + max_v) * 0.5;
    let half = ((max_v - min_v) * 0.5).max(1e-5);
    for v in out.iter_mut() {
        *v = clampf((*v - mid) / half, -1.0, 1.0);
    }

    out
}

fn build_continentality_field(
    seed: u32,
    grid: &GridConfig,
    cell_cache: &CellCache,
    buoyancy: &[f32],
    blob_backbone: &[f32],
    smooth_passes: usize,
) -> Vec<f32> {
    let mut field = vec![0.0_f32; grid.size];
    let seed_phase = seed as f32 * 0.00131;

    for i in 0..grid.size {
        let sx = cell_cache.noise_x[i];
        let sy = cell_cache.noise_y[i];
        let sz = cell_cache.noise_z[i];
        let warp_a = spherical_fbm(
            sx * 1.15 + 3.0,
            sy * 1.15 - 5.0,
            sz * 1.15 + 7.0,
            seed_phase + 11.0,
        );
        let warp_b = spherical_fbm(
            sx * 1.7 - 13.0,
            sy * 1.7 + 17.0,
            sz * 1.7 - 19.0,
            seed_phase + 29.0,
        );
        let warp_c = spherical_fbm(
            sx * 2.2 + 23.0,
            sy * 2.2 - 7.0,
            sz * 2.2 + 31.0,
            seed_phase + 47.0,
        );
        let wx = sx + 0.36 * warp_a + 0.2 * warp_b + 0.12 * warp_c;
        let wy = sy + 0.36 * warp_b + 0.2 * warp_c + 0.12 * warp_a;
        let wz = sz + 0.3 * warp_c + 0.16 * (warp_a - warp_b);

        let low = spherical_fbm(wx * 0.7, wy * 0.7, wz * 0.7, seed_phase + 37.0);
        let mid = spherical_fbm(wx * 1.45, wy * 1.45, wz * 1.45, seed_phase + 53.0);
        let high = spherical_fbm(wx * 2.95, wy * 2.95, wz * 2.95, seed_phase + 71.0);
        let ridgeish = ridge(spherical_fbm(
            wx * 2.1 + 17.0,
            wy * 2.1 - 19.0,
            wz * 2.1 + 11.0,
            seed_phase + 83.0,
        )) * 2.0
            - 1.0;
        let fine = spherical_fbm(
            wx * 5.1 - 41.0,
            wy * 5.1 + 37.0,
            wz * 5.1 - 29.0,
            seed_phase + 97.0,
        );
        field[i] = low * 0.48
            + mid * 0.32
            + high * 0.16
            + ridgeish * 0.04
            + fine * 0.03
            + buoyancy[i] * 0.03
            + blob_backbone[i] * 0.01;
    }

    if smooth_passes > 0 {
        let mut next = field.clone();
        for _ in 0..smooth_passes {
            for y in 0..grid.height {
                for x in 0..grid.width {
                    let i = grid.index(x, y);
                    let mut sum = field[i] * 0.36;
                    let mut wsum = 0.36;
                    for oy in -1..=1 {
                        for ox in -1..=1 {
                            if ox == 0 && oy == 0 {
                                continue;
                            }
                            let j = grid.neighbor(x as i32 + ox, y as i32 + oy);
                            let w = if ox == 0 || oy == 0 { 0.11 } else { 0.07 };
                            sum += field[j] * w;
                            wsum += w;
                        }
                    }
                    next[i] = sum / wsum.max(1e-6);
                }
            }
            field.copy_from_slice(&next);
        }
    }

    let (min_v, max_v) = min_max(&field);
    let mid = (min_v + max_v) * 0.5;
    let half = ((max_v - min_v) * 0.5).max(1e-6);
    for v in field.iter_mut() {
        *v = clampf((*v - mid) / half, -1.0, 1.0);
    }

    field
}

fn normalize_height_range(relief: &mut [f32], planet: &PlanetInputs, tectonics: &TectonicInputs) {
    let mut land_heights: Vec<f32> = Vec::new();
    let mut ocean_depths: Vec<f32> = Vec::new();

    for &h in relief.iter() {
        if h > 0.0 {
            land_heights.push(h);
        } else if h < 0.0 {
            ocean_depths.push(-h);
        }
    }

    if land_heights.len() < 8 || ocean_depths.len() < 8 {
        return;
    }

    land_heights.sort_by(|a, b| a.total_cmp(b));
    ocean_depths.sort_by(|a, b| a.total_cmp(b));

    let land_ref = quantile_sorted(&land_heights, 0.995).max(1.0);
    let ocean_ref = quantile_sorted(&ocean_depths, 0.995).max(1.0);

    let speed_factor = clampf(tectonics.plate_speed_cm_per_year / 5.0, 0.35, 2.8);
    let heat_factor = clampf(tectonics.mantle_heat / 55.0, 0.45, 2.2);
    let gravity_factor = clampf(9.81 / planet.gravity.max(1.0), 0.45, 2.4);
    let ocean_factor = clampf(planet.ocean_percent / 67.0, 0.45, 1.9);

    let target_land_max = clampf(
        4300.0 * speed_factor.powf(0.45) * heat_factor.powf(0.32) * gravity_factor.powf(0.62),
        1200.0,
        6600.0,
    );
    let target_ocean_depth = clampf(
        7500.0 * speed_factor.powf(0.5) * heat_factor.powf(0.35) * ocean_factor.powf(0.55),
        1400.0,
        12000.0,
    );
    let land_gamma = clampf(
        1.9 + 0.07 * speed_factor + 0.03 * heat_factor - 0.1 * gravity_factor,
        1.68,
        2.4,
    );
    let ocean_gamma = clampf(0.9 + 0.06 * ocean_factor - 0.05 * speed_factor, 0.74, 1.02);

    for h in relief.iter_mut() {
        if *h > 0.0 {
            let n = (*h / land_ref).max(0.0);
            let core = clampf(n, 0.0, 1.0).powf(land_gamma);
            let tail = (n - 1.0).max(0.0);
            *h = clampf(
                core * target_land_max + tail * target_land_max * 0.14,
                -12000.0,
                9000.0,
            );
        } else if *h < 0.0 {
            let d = (-*h / ocean_ref).max(0.0);
            let core = clampf(d, 0.0, 1.0).powf(ocean_gamma);
            let tail = (d - 1.0).max(0.0);
            *h = clampf(
                -(core * target_ocean_depth + tail * target_ocean_depth * 0.34),
                -12000.0,
                9000.0,
            );
        }
    }
}

fn defuse_orogenic_ribbons(relief: &mut [f32], seed: u32, cache: &WorldCache, detail: DetailProfile) {
    let passes = if detail.erosion_rounds >= 2 { 2 } else { 1 };
    let mut next = relief.to_vec();
    let seed_phase = seed as f32 * 0.00153;

    for pass in 0..passes {
        for y in 0..WORLD_HEIGHT {
            for x in 0..WORLD_WIDTH {
                let i = index(x, y);
                let h = relief[i];
                if h < 700.0 {
                    next[i] = h;
                    continue;
                }

                let sx = cache.x_by_cell[i];
                let sy = cache.y_by_cell[i];
                let sz = cache.z_by_cell[i];
                let cluster_noise = 0.5
                    + 0.5
                        * ((sx * 4.6 + sy * 3.9 + sz * 3.3 + seed_phase + pass as f32 * 1.9).sin()
                            * 0.58
                            + (sy * 5.1 - sz * 3.8 + sx * 2.7 - seed_phase * 1.2).cos() * 0.42);
                let cluster = smoothstep(0.28, 0.88, cluster_noise);

                let mut sum = h * 0.38;
                let mut wsum = 0.38;
                for oy in -1..=1 {
                    for ox in -1..=1 {
                        if ox == 0 && oy == 0 {
                            continue;
                        }
                        let j = index_spherical(x as i32 + ox, y as i32 + oy);
                        if relief[j] < 0.0 {
                            continue;
                        }
                        let w = if ox == 0 || oy == 0 { 0.1 } else { 0.066 };
                        sum += relief[j] * w;
                        wsum += w;
                    }
                }
                let avg = sum / wsum.max(1e-6);
                let ridge_excess = (h - avg).max(0.0);

                let east = relief[index_spherical(x as i32 + 1, y as i32)].max(0.0);
                let west = relief[index_spherical(x as i32 - 1, y as i32)].max(0.0);
                let north = relief[index_spherical(x as i32, y as i32 - 1)].max(0.0);
                let south = relief[index_spherical(x as i32, y as i32 + 1)].max(0.0);
                let ne = relief[index_spherical(x as i32 + 1, y as i32 - 1)].max(0.0);
                let nw = relief[index_spherical(x as i32 - 1, y as i32 - 1)].max(0.0);
                let se = relief[index_spherical(x as i32 + 1, y as i32 + 1)].max(0.0);
                let sw = relief[index_spherical(x as i32 - 1, y as i32 + 1)].max(0.0);

                let d0 = (east - west).abs();
                let d1 = (north - south).abs();
                let d2 = (ne - sw).abs();
                let d3 = (nw - se).abs();
                let dominant = d0.max(d1).max(d2).max(d3);
                let dir_sum = d0 + d1 + d2 + d3;
                let linearness = if dominant > 1e-4 {
                    clampf((dominant * 4.0 - dir_sum) / dominant, 0.0, 1.0)
                } else {
                    0.0
                };

                let mix = clampf(
                    0.1
                        + 0.34 * (1.0 - cluster)
                        + 0.22 * linearness
                        + 0.18 * clampf(ridge_excess / 900.0, 0.0, 1.0),
                    0.0,
                    0.76,
                );
                let perturb = ((sx * 10.3 + sy * 8.2 + sz * 6.1 + seed_phase).sin() * 0.55
                    + (sy * 9.4 - sz * 7.6 + sx * 5.7 - seed_phase * 1.37).cos() * 0.45)
                    * (18.0 + 58.0 * cluster);
                let target = avg + perturb - linearness * 70.0 * (1.0 - cluster);
                let mut reshaped = h * (1.0 - mix) + target * mix;
                if linearness > 0.55 {
                    reshaped = reshaped.min(avg + 1200.0 + 420.0 * cluster);
                }
                reshaped = reshaped.max(h * 0.4);
                next[i] = reshaped;
            }
        }
        relief.copy_from_slice(&next);
    }
}

fn reshape_ocean_boundaries(relief: &mut [f32], boundary_types: &[i8], boundary_strength: &[f32]) {
    let mut deepest = 0.0_f32;
    for &h in relief.iter() {
        if h < 0.0 {
            deepest = deepest.max(-h);
        }
    }
    let depth_scale = deepest.max(1.0);

    for i in 0..WORLD_SIZE {
        if relief[i] >= 0.0 || relief[i] > -2200.0 {
            continue;
        }

        let x = i % WORLD_WIDTH;
        let y = i / WORLD_WIDTH;
        let mut conv = 0.0_f32;
        let mut div = 0.0_f32;
        let mut transform = 0.0_f32;
        let mut wsum = 0.0_f32;
        for oy in -1..=1 {
            for ox in -1..=1 {
                let j = index_spherical(x as i32 + ox, y as i32 + oy);
                let t = boundary_types[j];
                if t == 0 {
                    continue;
                }
                let s = clampf(boundary_strength[j], 0.0, 1.0);
                let w = if ox == 0 && oy == 0 {
                    1.0
                } else if ox == 0 || oy == 0 {
                    0.62
                } else {
                    0.44
                };
                let value = s * w;
                if t == 1 {
                    conv += value;
                } else if t == 2 {
                    div += value;
                } else if t == 3 {
                    transform += value;
                }
                wsum += w;
            }
        }
        if wsum < 1e-4 {
            continue;
        }
        conv /= wsum;
        div /= wsum;
        transform /= wsum;

        let depth_t = clampf(-relief[i] / depth_scale, 0.0, 1.0);

        if div > 0.08 {
            let uplift = (60.0 + 260.0 * div) * (0.62 + 0.38 * (1.0 - depth_t));
            relief[i] = (relief[i] + uplift).min(-15.0);
        } else if transform > 0.08 {
            let uplift = (14.0 + 65.0 * transform) * (0.6 + 0.4 * (1.0 - depth_t));
            relief[i] += uplift;
        }
        if conv > 0.1 {
            let trench = (45.0 + 220.0 * conv) * (0.54 + 0.46 * depth_t);
            relief[i] -= trench;
        }
    }

    let mut ocean_smoothed = relief.to_vec();
    for _ in 0..2 {
        for y in 0..WORLD_HEIGHT {
            for x in 0..WORLD_WIDTH {
                let i = index(x, y);
                if relief[i] >= 0.0 {
                    continue;
                }
                let mut sum = relief[i] * 3.9;
                let mut wsum = 3.9;
                for oy in -1..=1 {
                    for ox in -1..=1 {
                        if ox == 0 && oy == 0 {
                            continue;
                        }
                        let j = index_spherical(x as i32 + ox, y as i32 + oy);
                        if relief[j] >= 0.0 {
                            continue;
                        }
                        let w = if ox == 0 || oy == 0 { 0.72 } else { 0.54 };
                        sum += relief[j] * w;
                        wsum += w;
                    }
                }
                ocean_smoothed[i] = sum / wsum.max(1e-6);
            }
        }
        for i in 0..WORLD_SIZE {
            if relief[i] < 0.0 {
                relief[i] = ocean_smoothed[i];
            }
        }
    }
}

fn apply_coastal_detail(relief: &mut [f32], seed: u32, cache: &WorldCache) {
    for i in 0..WORLD_SIZE {
        let h = relief[i];
        let near_sea = (-h.abs() / 900.0).exp();
        if near_sea < 0.035 {
            continue;
        }

        let sx = cache.x_by_cell[i];
        let sy = cache.y_by_cell[i];
        let sz = cache.z_by_cell[i];
        let seed_phase = seed as f32 * 0.00091;
        let warp_x =
            sx + 0.22 * (sy * 6.1 + seed_phase).sin() + 0.14 * (sz * 7.4 - seed_phase * 1.7).cos();
        let warp_y = sy
            + 0.2 * (sz * 5.8 - seed_phase * 1.2).sin()
            + 0.12 * (sx * 6.9 + seed_phase * 0.8).cos();
        let warp_z = sz
            + 0.18 * (sx * 5.2 + seed_phase * 1.5).sin()
            + 0.1 * (sy * 7.2 - seed_phase * 0.6).cos();
        let micro = (warp_x * 7.6 + warp_y * 6.4 + warp_z * 5.1 + seed_phase * 1.9).sin() * 0.62
            + (warp_y * 8.1 - warp_z * 5.7 + warp_x * 4.6 - seed_phase * 1.3).cos() * 0.38;
        let macro_n = spherical_fbm(
            warp_x * 0.95 + 2.7,
            warp_y * 0.95 - 3.1,
            warp_z * 0.95 + 1.9,
            seed_phase * 9.2,
        );
        let coast_weight = near_sea.powf(1.08);
        let delta = coast_weight * (macro_n * 26.0 + micro * 14.0);
        if h >= 0.0 {
            relief[i] = (h + delta).max(8.0);
        } else {
            relief[i] = (h + delta).min(-8.0);
        }
    }
}

fn cleanup_coastal_speckles(relief: &mut [f32]) {
    let mut next = relief.to_vec();

    for y in 0..WORLD_HEIGHT {
        for x in 0..WORLD_WIDTH {
            let i = index(x, y);
            let h = relief[i];
            if h.abs() > 260.0 {
                continue;
            }

            let mut land = 0;
            let mut water = 0;
            for oy in -1..=1 {
                for ox in -1..=1 {
                    if ox == 0 && oy == 0 {
                        continue;
                    }
                    let j = index_spherical(x as i32 + ox, y as i32 + oy);
                    if relief[j] >= 0.0 {
                        land += 1;
                    } else {
                        water += 1;
                    }
                }
            }

            if h >= 0.0 && water >= 7 {
                next[i] = next[i].min(-18.0 - h.abs() * 0.15);
            } else if h < 0.0 && land >= 7 {
                next[i] = next[i].max(18.0 + h.abs() * 0.15);
            }
        }
    }

    relief.copy_from_slice(&next);
}

fn clean_landmask_components(mask: &mut [u8], min_land_cells: usize, min_inland_water_cells: usize) {
    let mut visited = vec![0_u8; WORLD_SIZE];
    let mut queue: VecDeque<usize> = VecDeque::new();
    let mut component: Vec<usize> = Vec::new();

    // Remove tiny land islands.
    for start in 0..WORLD_SIZE {
        if mask[start] != 1 || visited[start] != 0 {
            continue;
        }
        visited[start] = 1;
        queue.push_back(start);
        component.clear();
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
                    if visited[j] == 0 && mask[j] == 1 {
                        visited[j] = 1;
                        queue.push_back(j);
                    }
                }
            }
        }
        if component.len() < min_land_cells {
            for &i in component.iter() {
                mask[i] = 0;
            }
        }
    }

    // Find largest water body (global ocean candidate).
    visited.fill(0);
    let mut largest_water = 0_usize;
    for start in 0..WORLD_SIZE {
        if mask[start] != 0 || visited[start] != 0 {
            continue;
        }
        visited[start] = 1;
        queue.push_back(start);
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
                    if visited[j] == 0 && mask[j] == 0 {
                        visited[j] = 1;
                        queue.push_back(j);
                    }
                }
            }
        }
        largest_water = largest_water.max(size);
    }

    // Fill tiny inland seas/lakes but keep major water bodies.
    visited.fill(0);
    for start in 0..WORLD_SIZE {
        if mask[start] != 0 || visited[start] != 0 {
            continue;
        }
        visited[start] = 1;
        queue.push_back(start);
        component.clear();
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
                    if visited[j] == 0 && mask[j] == 0 {
                        visited[j] = 1;
                        queue.push_back(j);
                    }
                }
            }
        }
        let size = component.len();
        if size < min_inland_water_cells && size < largest_water / 18 {
            for &i in component.iter() {
                mask[i] = 1;
            }
        }
    }
}

fn compute_component_sizes_for_mask(mask: &[u8], target: u8) -> Vec<usize> {
    let mut sizes = vec![0_usize; WORLD_SIZE];
    let mut visited = vec![0_u8; WORLD_SIZE];
    let mut queue: VecDeque<usize> = VecDeque::new();
    let mut component: Vec<usize> = Vec::new();

    for start in 0..WORLD_SIZE {
        if mask[start] != target || visited[start] != 0 {
            continue;
        }
        visited[start] = 1;
        queue.push_back(start);
        component.clear();

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
                    if visited[j] == 0 && mask[j] == target {
                        visited[j] = 1;
                        queue.push_back(j);
                    }
                }
            }
        }

        let size = component.len();
        for &i in component.iter() {
            sizes[i] = size;
        }
    }

    sizes
}

fn smooth_landmask(mask: &mut [u8], passes: usize) {
    if passes == 0 {
        return;
    }
    let mut next = mask.to_vec();
    for _ in 0..passes {
        for y in 0..WORLD_HEIGHT {
            for x in 0..WORLD_WIDTH {
                let i = index(x, y);
                let mut land_neighbors = 0_i32;
                for oy in -1..=1 {
                    for ox in -1..=1 {
                        if ox == 0 && oy == 0 {
                            continue;
                        }
                        let j = index_spherical(x as i32 + ox, y as i32 + oy);
                        if mask[j] == 1 {
                            land_neighbors += 1;
                        }
                    }
                }

                let cur = mask[i];
                next[i] = if cur == 1 {
                    if land_neighbors <= 2 {
                        0
                    } else {
                        1
                    }
                } else if land_neighbors >= 6 {
                    1
                } else {
                    0
                };
            }
        }
        mask.copy_from_slice(&next);
    }
}

fn blend_topology_edges(relief: &mut [f32]) {
    let mut next = relief.to_vec();
    for _ in 0..2 {
        for y in 0..WORLD_HEIGHT {
            for x in 0..WORLD_WIDTH {
                let i = index(x, y);
                let sign_i = relief[i] >= 0.0;
                let mut opposite = 0_u8;
                let mut sum = relief[i] * 0.45;
                let mut wsum = 0.45;

                for oy in -1..=1 {
                    for ox in -1..=1 {
                        if ox == 0 && oy == 0 {
                            continue;
                        }
                        let j = index_spherical(x as i32 + ox, y as i32 + oy);
                        let sign_j = relief[j] >= 0.0;
                        if sign_i != sign_j {
                            opposite += 1;
                        }
                        let w = if ox == 0 || oy == 0 { 0.14 } else { 0.09 };
                        sum += relief[j] * w;
                        wsum += w;
                    }
                }

                if opposite >= 3 {
                    next[i] = sum / wsum.max(1e-6);
                } else {
                    next[i] = relief[i];
                }
            }
        }
        relief.copy_from_slice(&next);
    }
}

fn stabilize_landmass_topology(relief: &mut [f32], planet: &PlanetInputs, detail: DetailProfile) {
    let mut landmask = vec![0_u8; WORLD_SIZE];
    for i in 0..WORLD_SIZE {
        landmask[i] = if relief[i] >= 0.0 { 1 } else { 0 };
    }

    let smooth_passes = if detail.erosion_rounds >= 2 { 3 } else { 2 };
    smooth_landmask(&mut landmask, smooth_passes);

    let min_land_cells = clampf(
        WORLD_SIZE as f32
            * (0.0008 + (planet.ocean_percent / 100.0) * 0.0010),
        650.0,
        9_500.0,
    ) as usize;
    let min_inland_water_cells = clampf(
        WORLD_SIZE as f32 * (0.0009 + (1.0 - planet.ocean_percent / 100.0) * 0.0010),
        700.0,
        10_000.0,
    ) as usize;
    clean_landmask_components(&mut landmask, min_land_cells, min_inland_water_cells);

    for i in 0..WORLD_SIZE {
        if landmask[i] == 1 {
            if relief[i] < 0.0 {
                relief[i] = relief[i] * 0.3 + 70.0;
            }
        } else if relief[i] >= 0.0 {
            relief[i] = relief[i] * 0.25 - 75.0;
        }
    }

    blend_topology_edges(relief);
}

fn suppress_subgrid_islands(relief: &mut [f32], planet: &PlanetInputs) {
    let mut landmask = vec![0_u8; WORLD_SIZE];
    for i in 0..WORLD_SIZE {
        if relief[i] >= 0.0 {
            landmask[i] = 1;
        }
    }

    let min_land_cells = clampf(
        WORLD_SIZE as f32 * (0.00014 + (planet.ocean_percent / 100.0) * 0.00006),
        180.0,
        2400.0,
    ) as usize;
    let min_inland_water_cells = clampf(
        WORLD_SIZE as f32 * (0.00014 + (1.0 - planet.ocean_percent / 100.0) * 0.00007),
        200.0,
        2600.0,
    ) as usize;
    clean_landmask_components(&mut landmask, min_land_cells, min_inland_water_cells);

    for i in 0..WORLD_SIZE {
        if landmask[i] == 1 {
            if relief[i] < 0.0 {
                relief[i] = relief[i] * 0.24 + 24.0;
            }
        } else if relief[i] >= 0.0 {
            relief[i] = relief[i] * 0.24 - 24.0;
        }
    }

    blend_topology_edges(relief);
}

fn dampen_isolated_shallow_ocean(relief: &mut [f32]) {
    let mut next = relief.to_vec();
    for _ in 0..2 {
        for y in 0..WORLD_HEIGHT {
            for x in 0..WORLD_WIDTH {
                let i = index(x, y);
                let h = relief[i];
                if h >= 0.0 || h <= -260.0 {
                    continue;
                }

                let mut land_neighbors = 0_i32;
                let mut ocean_neighbors = 0_i32;
                let mut ocean_sum = 0.0_f32;
                for oy in -1..=1 {
                    for ox in -1..=1 {
                        if ox == 0 && oy == 0 {
                            continue;
                        }
                        let j = index_spherical(x as i32 + ox, y as i32 + oy);
                        if relief[j] >= 0.0 {
                            land_neighbors += 1;
                        } else {
                            ocean_neighbors += 1;
                            ocean_sum += relief[j];
                        }
                    }
                }

                if land_neighbors > 0 || ocean_neighbors < 6 {
                    continue;
                }

                let avg = ocean_sum / ocean_neighbors.max(1) as f32;
                let target = avg.min(-220.0);
                let shallow_t = smoothstep(-260.0, -30.0, h);
                let mix = 0.22 + 0.46 * shallow_t;
                next[i] = h * (1.0 - mix) + target * mix;
            }
        }
        relief.copy_from_slice(&next);
    }
}

fn defuse_plate_linearity(
    relief: &mut [f32],
    plates: &ComputePlatesResult,
    seed: u32,
    cache: &WorldCache,
    detail: DetailProfile,
) {
    let strength_scale = if detail.buoyancy_smooth_passes >= 20 {
        1.0
    } else if detail.buoyancy_smooth_passes >= 12 {
        0.85
    } else {
        0.65
    };

    let mut imprint = vec![0.0_f32; WORLD_SIZE];
    for i in 0..WORLD_SIZE {
        if relief[i] < -1600.0 {
            continue;
        }
        let base = clampf(plates.boundary_strength[i], 0.0, 1.0);
        let k = match plates.boundary_types[i] {
            1 => 1.0,
            2 => 0.78,
            3 => 0.58,
            _ => 0.0,
        };
        imprint[i] = base * k;
    }

    let mut imprint_next = imprint.clone();
    for _ in 0..3 {
        for y in 0..WORLD_HEIGHT {
            for x in 0..WORLD_WIDTH {
                let i = index(x, y);
                let mut sum = imprint[i] * 0.5;
                let mut wsum = 0.5;
                for oy in -1..=1 {
                    for ox in -1..=1 {
                        if ox == 0 && oy == 0 {
                            continue;
                        }
                        let j = index_spherical(x as i32 + ox, y as i32 + oy);
                        let w = if ox == 0 || oy == 0 { 0.11 } else { 0.07 };
                        sum += imprint[j] * w;
                        wsum += w;
                    }
                }
                imprint_next[i] = sum / wsum.max(1e-6);
            }
        }
        imprint.copy_from_slice(&imprint_next);
    }

    let mut softened = relief.to_vec();
    let seed_phase = seed as f32 * 0.0017;
    for y in 0..WORLD_HEIGHT {
        for x in 0..WORLD_WIDTH {
            let i = index(x, y);
            let mask = clampf(imprint[i] * strength_scale, 0.0, 1.0);
            if mask < 0.08 || relief[i] < -2400.0 {
                continue;
            }

            let mut sum = relief[i] * 0.42;
            let mut wsum = 0.42;
            for oy in -1..=1 {
                for ox in -1..=1 {
                    if ox == 0 && oy == 0 {
                        continue;
                    }
                    let j = index_spherical(x as i32 + ox, y as i32 + oy);
                    let w = if ox == 0 || oy == 0 { 0.1 } else { 0.065 };
                    sum += relief[j] * w;
                    wsum += w;
                }
            }
            let avg = sum / wsum.max(1e-6);
            let sx = cache.x_by_cell[i];
            let sy = cache.y_by_cell[i];
            let sz = cache.z_by_cell[i];
            let depth_softener = smoothstep(-2400.0, 220.0, relief[i]);
            let perturb = ((sx * 12.1 + sy * 9.4 + sz * 8.3 + seed_phase).sin() * 0.58
                + (sy * 13.4 - sz * 7.8 + sx * 6.1 - seed_phase * 1.4).cos() * 0.42)
                * (14.0 + 74.0 * mask)
                * depth_softener;
            let mix = clampf(0.16 + 0.6 * mask, 0.0, 0.84) * depth_softener;
            let blended = relief[i] * (1.0 - mix) + (avg + perturb) * mix;
            softened[i] = if relief[i] >= 0.0 {
                blended.max(10.0)
            } else {
                blended.min(-10.0)
            };
        }
    }

    relief.copy_from_slice(&softened);
}

fn defuse_coastal_linearity(
    relief: &mut [f32],
    plates: &ComputePlatesResult,
    seed: u32,
    cache: &WorldCache,
) {
    let mut imprint = vec![0.0_f32; WORLD_SIZE];
    for i in 0..WORLD_SIZE {
        let k = match plates.boundary_types[i] {
            1 => 1.0,
            2 => 0.85,
            3 => 0.7,
            _ => 0.0,
        };
        imprint[i] = clampf(plates.boundary_strength[i], 0.0, 1.0) * k;
    }

    let mut blur = imprint.clone();
    for _ in 0..2 {
        for y in 0..WORLD_HEIGHT {
            for x in 0..WORLD_WIDTH {
                let i = index(x, y);
                let mut sum = imprint[i] * 0.42;
                let mut wsum = 0.42;
                for oy in -1..=1 {
                    for ox in -1..=1 {
                        if ox == 0 && oy == 0 {
                            continue;
                        }
                        let j = index_spherical(x as i32 + ox, y as i32 + oy);
                        let w = if ox == 0 || oy == 0 { 0.1 } else { 0.062 };
                        sum += imprint[j] * w;
                        wsum += w;
                    }
                }
                blur[i] = sum / wsum.max(1e-6);
            }
        }
        imprint.copy_from_slice(&blur);
    }

    let mut next = relief.to_vec();
    let seed_phase = seed as f32 * 0.00123;
    for y in 0..WORLD_HEIGHT {
        for x in 0..WORLD_WIDTH {
            let i = index(x, y);
            let h = relief[i];
            if h.abs() > 920.0 {
                continue;
            }

            let mask = imprint[i];
            if mask < 0.1 {
                continue;
            }

            let coast = 1.0 - smoothstep(120.0, 920.0, h.abs());
            if coast <= 0.0 {
                continue;
            }

            let sx = cache.x_by_cell[i];
            let sy = cache.y_by_cell[i];
            let sz = cache.z_by_cell[i];
            let macro_n = spherical_fbm(
                sx * 1.05 + 5.7,
                sy * 1.05 - 4.1,
                sz * 1.05 + 3.3,
                seed_phase + 17.0,
            );
            let micro_n = spherical_fbm(
                sx * 3.4 + 11.0,
                sy * 3.4 - 7.0,
                sz * 3.4 + 9.0,
                seed_phase + 41.0,
            );

            let mut sum = h * 0.34;
            let mut wsum = 0.34;
            for oy in -1..=1 {
                for ox in -1..=1 {
                    if ox == 0 && oy == 0 {
                        continue;
                    }
                    let j = index_spherical(x as i32 + ox, y as i32 + oy);
                    if h >= 0.0 && relief[j] < -350.0 {
                        continue;
                    }
                    if h < 0.0 && relief[j] > 350.0 {
                        continue;
                    }
                    let w = if ox == 0 || oy == 0 { 0.11 } else { 0.07 };
                    sum += relief[j] * w;
                    wsum += w;
                }
            }
            let avg = sum / wsum.max(1e-6);

            let jitter = (macro_n * 290.0 + micro_n * 120.0) * coast * (0.5 + 0.62 * mask);
            let mix = clampf(0.14 + 0.56 * mask * coast, 0.0, 0.78);
            let mut v = h * (1.0 - mix) + (avg + jitter) * mix;
            if h >= 0.0 {
                v = v.max(-120.0);
            } else {
                v = v.min(120.0);
            }
            next[i] = v;
        }
    }

    relief.copy_from_slice(&next);
}

fn break_straight_coasts(relief: &mut [f32], seed: u32, cache: &WorldCache) {
    let mut next = relief.to_vec();
    let seed_phase = seed as f32 * 0.00097;

    for y in 0..WORLD_HEIGHT {
        for x in 0..WORLD_WIDTH {
            let i = index(x, y);
            let h = relief[i];
            if h.abs() > 620.0 {
                continue;
            }

            let east = relief[index_spherical(x as i32 + 1, y as i32)];
            let west = relief[index_spherical(x as i32 - 1, y as i32)];
            let north = relief[index_spherical(x as i32, y as i32 - 1)];
            let south = relief[index_spherical(x as i32, y as i32 + 1)];
            let gx = east - west;
            let gy = south - north;
            let gl = gx.hypot(gy);
            if gl < 14.0 {
                continue;
            }
            let nx = gx / gl;
            let ny = gy / gl;

            let mut coh_sum = 0.0_f32;
            let mut coh_count = 0.0_f32;
            for oy in -1..=1 {
                for ox in -1..=1 {
                    if ox == 0 && oy == 0 {
                        continue;
                    }
                    let j = index_spherical(x as i32 + ox, y as i32 + oy);
                    if relief[j].abs() > 420.0 {
                        continue;
                    }
                    let je = relief[index_spherical(x as i32 + ox + 1, y as i32 + oy)];
                    let jw = relief[index_spherical(x as i32 + ox - 1, y as i32 + oy)];
                    let jn = relief[index_spherical(x as i32 + ox, y as i32 + oy - 1)];
                    let js = relief[index_spherical(x as i32 + ox, y as i32 + oy + 1)];
                    let jgx = je - jw;
                    let jgy = js - jn;
                    let jgl = jgx.hypot(jgy);
                    if jgl < 12.0 {
                        continue;
                    }
                    let jnx = jgx / jgl;
                    let jny = jgy / jgl;
                    coh_sum += (nx * jnx + ny * jny).abs();
                    coh_count += 1.0;
                }
            }

            if coh_count < 3.0 {
                continue;
            }
            let coherence = coh_sum / coh_count;
            if coherence < 0.58 {
                continue;
            }

            let coast = 1.0 - smoothstep(40.0, 620.0, h.abs());
            let sx = cache.x_by_cell[i];
            let sy = cache.y_by_cell[i];
            let sz = cache.z_by_cell[i];
            let jitter = spherical_fbm(
                sx * 2.4 + 7.0,
                sy * 2.4 - 9.0,
                sz * 2.4 + 11.0,
                seed_phase + 23.0,
            );
            let gain = ((coherence - 0.58) / 0.42).clamp(0.0, 1.0);
            next[i] = h + jitter * 460.0 * coast * gain;
        }
    }

    relief.copy_from_slice(&next);
}

fn fracture_coastline_band(relief: &mut [f32], seed: u32, cache: &WorldCache) {
    let seed_phase = seed as f32 * 0.00191;
    for i in 0..WORLD_SIZE {
        let h = relief[i];
        if h.abs() > 1200.0 {
            continue;
        }

        let coast = 1.0 - smoothstep(90.0, 1200.0, h.abs());
        if coast <= 0.0 {
            continue;
        }

        let sx = cache.x_by_cell[i];
        let sy = cache.y_by_cell[i];
        let sz = cache.z_by_cell[i];
        let n1 = spherical_fbm(
            sx * 2.9 + 9.0,
            sy * 2.9 - 5.0,
            sz * 2.9 + 2.0,
            seed_phase + 13.0,
        );
        let n2 = spherical_fbm(
            sx * 6.2 - 17.0,
            sy * 6.2 + 11.0,
            sz * 6.2 - 7.0,
            seed_phase + 31.0,
        );
        let n3 = ridge(spherical_fbm(
            sx * 4.3 + 21.0,
            sy * 4.3 - 19.0,
            sz * 4.3 + 15.0,
            seed_phase + 57.0,
        )) * 2.0
            - 1.0;
        let signal = n1 * 0.52 + n2 * 0.33 + n3 * 0.15;
        relief[i] += signal * 560.0 * coast;
    }
}

fn warp_coastal_band(relief: &mut [f32], seed: u32, cache: &WorldCache) {
    let mut next = relief.to_vec();
    let seed_phase = seed as f32 * 0.00137;

    for y in 0..WORLD_HEIGHT {
        for x in 0..WORLD_WIDTH {
            let i = index(x, y);
            let h = relief[i];
            if h.abs() > 950.0 {
                continue;
            }

            let coast = 1.0 - smoothstep(80.0, 950.0, h.abs());
            if coast <= 0.0 {
                continue;
            }

            let sx = cache.x_by_cell[i];
            let sy = cache.y_by_cell[i];
            let sz = cache.z_by_cell[i];
            let noise_u = spherical_fbm(
                sx * 2.2 + 5.0,
                sy * 2.2 - 7.0,
                sz * 2.2 + 9.0,
                seed_phase + 13.0,
            );
            let noise_v = spherical_fbm(
                sx * 2.5 - 11.0,
                sy * 2.5 + 3.0,
                sz * 2.5 - 17.0,
                seed_phase + 31.0,
            );

            let shift = 0.45 + 2.9 * coast;
            let ox = (noise_u * shift).round() as i32;
            let oy = (noise_v * shift).round() as i32;
            let j = index_spherical(x as i32 + ox, y as i32 + oy);

            let mix = clampf(0.16 + 0.62 * coast, 0.0, 0.78);
            let warped = relief[i] * (1.0 - mix) + relief[j] * mix;
            let fine = spherical_fbm(
                sx * 4.4 + 23.0,
                sy * 4.4 - 19.0,
                sz * 4.4 + 29.0,
                seed_phase + 57.0,
            ) * 130.0 * coast;
            next[i] = warped + fine;
        }
    }

    relief.copy_from_slice(&next);
}

fn carve_fluvial_valleys(
    relief: &mut [f32],
    grid: &GridConfig,
    detail: DetailProfile,
    progress: &mut ProgressTap<'_>,
    progress_base: f32,
    progress_span: f32,
) {
    if detail.fluvial_rounds == 0 {
        return;
    }

    let rounds = detail.fluvial_rounds.min(4);
    let mut flow_direction = vec![-1_i32; grid.size];
    let mut flow_accumulation = vec![1.0_f32; grid.size];
    let mut order: Vec<usize> = (0..grid.size).collect();
    let mut updated = relief.to_vec();

    for round in 0..rounds {
        for y in 0..grid.height {
            let t = (round as f32 + y as f32 / grid.height as f32) / rounds as f32;
            progress.phase(progress_base, progress_span, 0.35 * t);
            for x in 0..grid.width {
                let i = grid.index(x, y);
                let h = relief[i];
                let mut best_drop = 0.0_f32;
                let mut best = -1_i32;
                for oy in -1..=1 {
                    for ox in -1..=1 {
                        if ox == 0 && oy == 0 {
                            continue;
                        }
                        let j = grid.neighbor(x as i32 + ox, y as i32 + oy);
                        let drop = h - relief[j];
                        if drop > best_drop {
                            best_drop = drop;
                            best = j as i32;
                        }
                    }
                }
                flow_direction[i] = best;
            }
        }

        flow_accumulation.fill(1.0);
        order.sort_by(|a, b| relief[*b].total_cmp(&relief[*a]));
        for (rank, i) in order.iter().copied().enumerate() {
            if rank % (grid.width * 4) == 0 {
                progress.phase(
                    progress_base,
                    progress_span,
                    0.35 + 0.25 * (rank as f32 / grid.size as f32),
                );
            }
            if relief[i] <= -25.0 {
                continue;
            }
            let to = flow_direction[i];
            if to >= 0 {
                let j = to as usize;
                let drop = (relief[i] - relief[j]).max(0.0);
                flow_accumulation[j] += flow_accumulation[i] * (1.0 + drop / 1600.0);
            }
        }

        let mut sorted_acc = flow_accumulation.clone();
        sorted_acc.sort_by(|a, b| a.total_cmp(b));
        let channel_ref = quantile_sorted(&sorted_acc, 0.992).max(1.0);

        updated.copy_from_slice(relief);
        for y in 0..grid.height {
            let t = (round as f32 + y as f32 / grid.height as f32) / rounds as f32;
            progress.phase(progress_base, progress_span, 0.6 + 0.4 * t);
            for x in 0..grid.width {
                let i = grid.index(x, y);
                let h = relief[i];
                if h <= 8.0 {
                    continue;
                }

                let mut max_drop = 0.0_f32;
                for oy in -1..=1 {
                    for ox in -1..=1 {
                        if ox == 0 && oy == 0 {
                            continue;
                        }
                        let j = grid.neighbor(x as i32 + ox, y as i32 + oy);
                        max_drop = max_drop.max((h - relief[j]).max(0.0));
                    }
                }
                if max_drop < 2.0 {
                    continue;
                }

                let channel = ((flow_accumulation[i] / channel_ref) - 1.0).max(0.0).min(8.0);
                if channel <= 0.0 {
                    continue;
                }

                let slope_t = clampf(max_drop / 340.0, 0.0, 1.8);
                let coast_t = 1.0 - smoothstep(60.0, 1900.0, h);
                let incision = (7.5 + 26.0 * channel.powf(0.65) + 20.0 * slope_t.powf(0.7))
                    * (0.4 + 0.6 * coast_t);
                let cut = incision.min((h - 4.0).max(0.0));
                if cut <= 0.0 {
                    continue;
                }

                updated[i] = h - cut;
                let to = flow_direction[i];
                if to >= 0 {
                    let j = to as usize;
                    if relief[j] > 0.0 {
                        updated[j] += cut * 0.04;
                    }
                }
            }
        }

        relief.copy_from_slice(&updated);
    }

    progress.phase(progress_base, progress_span, 1.0);
}

fn apply_ocean_profile(
    relief: &mut [f32],
    boundary_types: &[i8],
    boundary_strength: &[f32],
    planet: &PlanetInputs,
    seed: u32,
    cache: &WorldCache,
    detail: DetailProfile,
    progress: &mut ProgressTap<'_>,
    progress_base: f32,
    progress_span: f32,
) {
    let mut distance_to_coast = vec![f32::INFINITY; WORLD_SIZE];
    let mut is_ocean = vec![false; WORLD_SIZE];
    let mut queue: BinaryHeap<Reverse<DistanceNode>> = BinaryHeap::new();
    let km_lat_step = (KILOMETERS_PER_DEGREE * (180.0 / WORLD_HEIGHT as f32) * planet.radius_km
        / 6371.0)
        .max(0.01);
    let km_lon_equator = (KILOMETERS_PER_DEGREE * (360.0 / WORLD_WIDTH as f32) * planet.radius_km
        / 6371.0)
        .max(0.01);
    let km_equator_cell =
        ((2.0 * std::f32::consts::PI * planet.radius_km) / WORLD_WIDTH as f32).max(0.01);
    let mut lon_step_by_y = vec![0.0_f32; WORLD_HEIGHT];
    for y in 0..WORLD_HEIGHT {
        let lat_deg = 90.0 - (y as f32 + 0.5) * (180.0 / WORLD_HEIGHT as f32);
        let lon_step = km_lon_equator * (lat_deg * RADIANS).cos().abs();
        lon_step_by_y[y] = lon_step.max(km_lat_step * 0.22);
    }

    let mut land_mask = vec![0_u8; WORLD_SIZE];
    for i in 0..WORLD_SIZE {
        if relief[i] >= 0.0 {
            land_mask[i] = 1;
        }
    }
    let land_component_sizes = compute_component_sizes_for_mask(&land_mask, 1);
    let major_land_component = clampf(
        WORLD_SIZE as f32 * (0.00034 + (planet.ocean_percent / 100.0) * 0.00025),
        420.0,
        5200.0,
    ) as usize;
    let minor_land_component = (major_land_component / 5).max(65);

    let mut ocean_cells = 0_usize;
    let mut major_seed_count = 0_usize;
    let mut minor_coast_seeds: Vec<(usize, f32)> = Vec::new();

    for y in 0..WORLD_HEIGHT {
        progress.phase(
            progress_base,
            progress_span,
            0.1 * (y as f32 / WORLD_HEIGHT as f32),
        );
        for x in 0..WORLD_WIDTH {
            let i = index(x, y);
            if relief[i] >= 0.0 {
                continue;
            }
            is_ocean[i] = true;
            ocean_cells += 1;

            let mut is_coast_adjacent = false;
            let mut adjacent_land_component = 0_usize;
            for oy in -1..=1 {
                for ox in -1..=1 {
                    if ox == 0 && oy == 0 {
                        continue;
                    }
                    let j = index_spherical(x as i32 + ox, y as i32 + oy);
                    if relief[j] >= 0.0 {
                        is_coast_adjacent = true;
                        adjacent_land_component =
                            adjacent_land_component.max(land_component_sizes[j]);
                        break;
                    }
                }
                if is_coast_adjacent {
                    break;
                }
            }

            if is_coast_adjacent {
                if adjacent_land_component >= major_land_component {
                    distance_to_coast[i] = 0.0;
                    queue.push(Reverse(DistanceNode { cost: 0.0, index: i }));
                    major_seed_count += 1;
                } else if adjacent_land_component >= minor_land_component {
                    // Small islands should not generate unrealistically wide continental shelves.
                    // Start them with a positive distance penalty so shelf influence decays quickly.
                    let size_t = clampf(
                        adjacent_land_component as f32 / major_land_component.max(1) as f32,
                        0.0,
                        1.0,
                    );
                    let start_penalty = km_equator_cell * (0.45 + (1.0 - size_t) * 2.4);
                    minor_coast_seeds.push((i, start_penalty));
                }
            }
        }
    }

    const FLOW_STEPS: [(i32, i32, f32, f32); 24] = [
        (1, 0, 1.0, 0.0),
        (-1, 0, 1.0, 0.0),
        (0, 1, 0.0, 1.0),
        (0, -1, 0.0, 1.0),
        (1, 1, 1.0, 1.0),
        (-1, 1, 1.0, 1.0),
        (1, -1, 1.0, 1.0),
        (-1, -1, 1.0, 1.0),
        (2, 0, 2.0, 0.0),
        (-2, 0, 2.0, 0.0),
        (0, 2, 0.0, 2.0),
        (0, -2, 0.0, 2.0),
        (2, 1, 2.0, 1.0),
        (-2, 1, 2.0, 1.0),
        (2, -1, 2.0, 1.0),
        (-2, -1, 2.0, 1.0),
        (1, 2, 1.0, 2.0),
        (-1, 2, 1.0, 2.0),
        (1, -2, 1.0, 2.0),
        (-1, -2, 1.0, 2.0),
        (2, 2, 2.0, 2.0),
        (-2, 2, 2.0, 2.0),
        (2, -2, 2.0, 2.0),
        (-2, -2, 2.0, 2.0),
    ];

    if major_seed_count == 0 && minor_coast_seeds.is_empty() {
        for i in 0..WORLD_SIZE {
            if is_ocean[i] {
                distance_to_coast[i] = 0.0;
            }
        }
    } else {
        if major_seed_count == 0 {
            for (idx, penalty) in minor_coast_seeds.iter().copied() {
                let cost = penalty * 0.35;
                if cost < distance_to_coast[idx] {
                    distance_to_coast[idx] = cost;
                    queue.push(Reverse(DistanceNode { cost, index: idx }));
                }
            }
        } else {
            for (idx, penalty) in minor_coast_seeds.iter().copied() {
                let cost = penalty * 1.25;
                if cost < distance_to_coast[idx] {
                    distance_to_coast[idx] = cost;
                    queue.push(Reverse(DistanceNode { cost, index: idx }));
                }
            }
        }

        let mut settled = 0_usize;
        while let Some(Reverse(node)) = queue.pop() {
            if node.cost > distance_to_coast[node.index] + 1e-5 {
                continue;
            }
            settled += 1;
            if settled % (WORLD_WIDTH * 6) == 0 {
                let t = settled as f32 / ocean_cells.max(1) as f32;
                progress.phase(progress_base, progress_span, 0.1 + 0.1 * t.min(1.0));
            }
            let x = node.index % WORLD_WIDTH;
            let y = node.index / WORLD_WIDTH;

            for (dx, dy, mx, my) in FLOW_STEPS {
                let j = index_spherical(x as i32 + dx, y as i32 + dy);
                if !is_ocean[j] {
                    continue;
                }
                let jy = j / WORLD_WIDTH;
                let step_lon = lon_step_by_y[jy];
                let step = ((mx * step_lon).powi(2) + (my * km_lat_step).powi(2)).sqrt();
                let next_dist = node.cost + step;
                if next_dist + 1e-5 < distance_to_coast[j] {
                    distance_to_coast[j] = next_dist;
                    queue.push(Reverse(DistanceNode {
                        cost: next_dist,
                        index: j,
                    }));
                }
            }
        }
    }
    progress.phase(progress_base, progress_span, 0.2);

    let mut max_distance_km = km_lat_step.max(km_lon_equator);
    for i in 0..WORLD_SIZE {
        if is_ocean[i] {
            let d = distance_to_coast[i];
            if d.is_finite() {
                max_distance_km = max_distance_km.max(d);
            }
        }
    }
    let max_distance = (max_distance_km / km_equator_cell).max(1.0);

    let shelf_cells = clampf(4.8 + (planet.radius_km / 6371.0 - 1.0) * 1.2, 3.8, 8.4);
    let slope_cells = clampf(15.0 + (planet.ocean_percent - 67.0) * 0.08, 11.0, 23.0);
    let abyssal_base = clampf(
        4200.0 + (planet.ocean_percent - 67.0) * 26.0,
        3600.0,
        5600.0,
    );

    for i in 0..WORLD_SIZE {
        if i % (WORLD_WIDTH * 4) == 0 {
            progress.phase(
                progress_base,
                progress_span,
                0.2 + 0.55 * (i as f32 / WORLD_SIZE as f32),
            );
        }
        if relief[i] >= 0.0 {
            continue;
        }

        let d_km = if distance_to_coast[i].is_finite() {
            distance_to_coast[i]
        } else {
            max_distance_km
        };
        let d = (d_km / km_equator_cell).max(0.0);
        let sx = cache.x_by_cell[i];
        let sy = cache.y_by_cell[i];
        let sz = cache.z_by_cell[i];
        let seed_phase = seed as f32 * 0.0012;
        let margin_signal = 0.5
            + 0.5
                * ((sx * 6.2 + sy * 5.1 + sz * 4.3 + seed_phase).sin() * 0.62
                    + (sy * 7.4 - sz * 4.9 + sx * 3.8 - seed_phase * 1.1).cos() * 0.38);
        let x = i % WORLD_WIDTH;
        let y = i / WORLD_WIDTH;
        let mut conv = 0.0_f32;
        let mut div = 0.0_f32;
        let mut transform = 0.0_f32;
        let mut wsum = 0.0_f32;
        for oy in -1..=1 {
            for ox in -1..=1 {
                let j = index_spherical(x as i32 + ox, y as i32 + oy);
                let t = boundary_types[j];
                if t == 0 {
                    continue;
                }
                let s = clampf(boundary_strength[j], 0.0, 1.0);
                let w = if ox == 0 && oy == 0 {
                    1.0
                } else if ox == 0 || oy == 0 {
                    0.58
                } else {
                    0.4
                };
                let value = s * w;
                if t == 1 {
                    conv += value;
                } else if t == 2 {
                    div += value;
                } else if t == 3 {
                    transform += value;
                }
                wsum += w;
            }
        }

        if wsum > 1e-4 {
            conv /= wsum;
            div /= wsum;
            transform /= wsum;
        }

        let regime_bias = match boundary_types[i] {
            1 => -0.28, // active convergent margins -> narrower shelf
            2 => 0.22,  // passive/divergent margins -> wider shelf
            3 => -0.08,
            _ => 0.0,
        };
        let shelf_texture = 0.5
            + 0.5
                * ((sx * 4.7 + sy * 3.8 + sz * 3.1 + seed_phase * 0.9).sin() * 0.58
                    + (sy * 5.1 - sz * 3.6 + sx * 2.9 - seed_phase * 1.1).cos() * 0.42);
        let tectonic_shelf_factor = clampf(1.0 + 1.35 * div - 1.45 * conv - 0.4 * transform, 0.4, 2.15);
        let tectonic_slope_factor = clampf(1.0 + 1.15 * conv - 0.32 * div, 0.62, 1.95);
        let local_shelf = shelf_cells
            * clampf(
                0.34 + 1.18 * margin_signal + regime_bias + 0.55 * (shelf_texture - 0.5),
                0.22,
                2.7,
            )
            * tectonic_shelf_factor;
        let local_slope = slope_cells
            * clampf(
                0.56 + 1.02 * (1.0 - margin_signal) - regime_bias * 0.35 + 0.22 * conv,
                0.36,
                2.2,
            )
            * tectonic_slope_factor;
        let shore_depth = -12.0
            - 98.0 * clampf(margin_signal + regime_bias * 0.35, 0.0, 1.0)
            - 140.0 * conv
            + 62.0 * div;
        let mut target_depth = if d <= local_shelf {
            let t = clampf(d / local_shelf.max(0.5), 0.0, 1.0);
            shore_depth - (150.0 + 240.0 * margin_signal + 110.0 * conv) * t.powf(1.42)
        } else if d <= local_shelf + local_slope {
            let t = clampf((d - local_shelf) / local_slope.max(0.8), 0.0, 1.0);
            -245.0 - (abyssal_base - 245.0) * t.powf(0.74)
        } else {
            let t = clampf(
                (d - local_shelf - local_slope) / (max_distance + 1.0),
                0.0,
                1.0,
            );
            -abyssal_base - 1650.0 * t.powf(0.58)
        };
        let basin_weight = clampf((d - local_shelf) / (local_shelf + local_slope + 1.0), 0.0, 1.0);
        let undulation = (sx * 7.2 + sy * 5.3 + sz * 4.1 + seed_phase).sin() * 0.6
            + (sy * 8.6 - sz * 4.9 + sx * 3.7 - seed_phase * 1.35).cos() * 0.4;
        target_depth += undulation * (120.0 + 390.0 * basin_weight);
        target_depth += -760.0 * conv + 320.0 * div - 130.0 * transform;

        let blend = 0.72 + 0.24 * basin_weight;
        relief[i] = clampf(
            relief[i] * (1.0 - blend) + target_depth * blend,
            -12000.0,
            -5.0,
        );
    }

    let mut smoothed = relief.to_vec();
    for pass in 0..detail.ocean_smooth_passes {
        for y in 0..WORLD_HEIGHT {
            let pass_t = (pass as f32 + y as f32 / WORLD_HEIGHT as f32)
                / detail.ocean_smooth_passes.max(1) as f32;
            progress.phase(progress_base, progress_span, 0.75 + 0.25 * pass_t);
            for x in 0..WORLD_WIDTH {
                let i = index(x, y);
                if relief[i] >= 0.0 {
                    continue;
                }
                let mut sum = relief[i] * 2.8;
                let mut wsum = 2.8;
                for oy in -1..=1 {
                    for ox in -1..=1 {
                        if ox == 0 && oy == 0 {
                            continue;
                        }
                        let j = index_spherical(x as i32 + ox, y as i32 + oy);
                        if relief[j] >= 0.0 {
                            continue;
                        }
                        let w = if ox == 0 || oy == 0 { 0.5 } else { 0.35 };
                        sum += relief[j] * w;
                        wsum += w;
                    }
                }
                smoothed[i] = sum / wsum.max(1e-6);
            }
        }
        for i in 0..WORLD_SIZE {
            if relief[i] < 0.0 {
                relief[i] = smoothed[i];
            }
        }
    }
    progress.phase(progress_base, progress_span, 1.0);
}

// ---------------------------------------------------------------------------
// Airy isostasy: crust thickness → elevation (m)
// Based on: Turcotte & Schubert (2002), Geodynamics, Ch. 3
// ---------------------------------------------------------------------------

/// Compute isostatic elevation in metres from crust thickness.
///
/// * `crust_km`      — crustal thickness in km
/// * `is_continental` — true = continental (ρ_c = 2800), false = oceanic (ρ_c = 2900)
/// * `heat_anomaly`  — normalised thermal anomaly [0..1]; hot crust expands → lower ρ
fn isostatic_elevation(crust_km: f32, is_continental: bool, heat_anomaly: f32) -> f32 {
    let rho_c: f32 = if is_continental { 2800.0 } else { 2900.0 };
    let rho_m: f32 = 3300.0;
    let rho_w: f32 = 1030.0;
    // Reference crustal thickness: 35 km continental, 7 km oceanic
    let c_ref: f32 = if is_continental { 35.0 } else { 7.0 };

    // Thermal correction: hot crust is up to 2 % less dense
    let thermal_correction = 1.0 - heat_anomaly.clamp(0.0, 1.0) * 0.02;
    let rho_c_eff = rho_c * thermal_correction;

    let delta_c = crust_km - c_ref; // km

    if delta_c >= 0.0 {
        // Continental root → positive elevation
        delta_c * 1000.0 * (rho_m - rho_c_eff) / rho_c_eff
    } else {
        // Thinned crust → negative elevation (depth)
        delta_c * 1000.0 * (rho_m - rho_c_eff) / (rho_m - rho_w)
    }
}

fn compute_relief(
    planet: &PlanetInputs,
    tectonics: &TectonicInputs,
    plates: &ComputePlatesResult,
    seed: u32,
    grid: &GridConfig,
    cell_cache: &CellCache,
    detail: DetailProfile,
    progress: &mut ProgressTap<'_>,
    progress_base: f32,
    progress_span: f32,
) -> ReliefResult {
    let mut relief = vec![0.0_f32; grid.size];
    let radius_cm = (planet.radius_km.max(1000.0) * 100_000.0).max(1.0);

    let seed_mix = (((planet.radius_km * 1000.0) as i32)
        ^ ((tectonics.plate_speed_cm_per_year * 1000.0) as i32)
        ^ ((tectonics.mantle_heat * 1000.0) as i32)
        ^ (seed.wrapping_mul(2_654_435_761) as i32)) as u32;
    let mut random_seed = Rng::new(seed_mix);
    let k_relief = 560.0_f32;
    let fault_backbone = build_fault_backbone(
        seed_mix,
        detail.fault_iterations,
        detail.fault_smooth_passes,
        grid,
        progress,
        progress_base,
        progress_span * 0.12,
    );
    let blob_backbone = build_continent_blob_field(seed_mix ^ 0xD1B5_4A35, grid, cell_cache);

    // Smooth per-plate buoyancy so continent/ocean macro-shape is not cut by hard plate polygons.
    let mut buoyancy_field = vec![0.0_f32; grid.size];
    for (i, b) in buoyancy_field.iter_mut().enumerate() {
        let pid = plates.plate_field[i] as usize;
        *b = plates.plate_vectors[pid].buoyancy;
    }
    let mut buoyancy_scratch = vec![0.0_f32; grid.size];
    for pass in 0..detail.buoyancy_smooth_passes {
        for y in 0..grid.height {
            let pass_t = (pass as f32 + y as f32 / grid.height as f32)
                / detail.buoyancy_smooth_passes.max(1) as f32;
            progress.phase(progress_base, progress_span, 0.12 + 0.12 * pass_t);
            for x in 0..grid.width {
                let i = grid.index(x, y);
                let mut sum = buoyancy_field[i] * 0.34;
                let mut wsum = 0.34;
                for oy in -1..=1 {
                    for ox in -1..=1 {
                        if ox == 0 && oy == 0 {
                            continue;
                        }
                        let j = grid.neighbor(x as i32 + ox, y as i32 + oy);
                        let w = if ox == 0 || oy == 0 { 0.125 } else { 0.07 };
                        sum += buoyancy_field[j] * w;
                        wsum += w;
                    }
                }
                buoyancy_scratch[i] = sum / wsum.max(1e-6);
            }
        }
        buoyancy_field.copy_from_slice(&buoyancy_scratch);
    }

    let continentality_field = build_continentality_field(
        seed_mix ^ 0x7FEB_352D,
        grid,
        cell_cache,
        &buoyancy_field,
        &blob_backbone,
        (detail.buoyancy_smooth_passes / 4).clamp(1, 4),
    );

    let plate_count = plates.plate_vectors.len();
    let mut plate_land_sum = vec![0.0_f32; plate_count];
    let mut plate_land_cells = vec![0_u32; plate_count];
    for i in 0..grid.size {
        let pid = plates.plate_field[i] as usize;
        let landness = smoothstep(-0.14, 0.84, continentality_field[i]);
        plate_land_sum[pid] += landness;
        plate_land_cells[pid] += 1;
    }
    let mut plate_landness = vec![0.5_f32; plate_count];
    for pid in 0..plate_count {
        let cnt = plate_land_cells[pid].max(1) as f32;
        plate_landness[pid] = clampf(plate_land_sum[pid] / cnt, 0.0, 1.0);
    }

    let mut heat_norm_field = vec![0.0_f32; grid.size];
    let mut weakness_field = vec![0.0_f32; grid.size];
    let mut strength_field = vec![0.0_f32; grid.size];
    let mut comp_source = vec![0.0_f32; grid.size];
    let mut ext_source = vec![0.0_f32; grid.size];
    let mut shear_source = vec![0.0_f32; grid.size];
    let mut cc_collision = vec![0.0_f32; grid.size];
    let mut oc_collision_cont = vec![0.0_f32; grid.size];
    let mut oc_collision_ocean = vec![0.0_f32; grid.size];
    let mut oo_collision = vec![0.0_f32; grid.size];
    let mut subduction_source = vec![0.0_f32; grid.size];
    let mut rift_source = vec![0.0_f32; grid.size];
    let mut velocity_x = vec![0.0_f32; grid.size];
    let mut velocity_y = vec![0.0_f32; grid.size];
    const LOCAL_NEIGHBORS: [(i32, i32, f32); 8] = [
        (1, 0, 1.0),
        (-1, 0, 1.0),
        (0, -1, 1.0),
        (0, 1, 1.0),
        (1, -1, std::f32::consts::FRAC_1_SQRT_2),
        (-1, -1, std::f32::consts::FRAC_1_SQRT_2),
        (1, 1, std::f32::consts::FRAC_1_SQRT_2),
        (-1, 1, std::f32::consts::FRAC_1_SQRT_2),
    ];

    for i in 0..grid.size {
        if i % (grid.width * 4) == 0 {
            progress.phase(
                progress_base,
                progress_span,
                0.24 + 0.08 * (i as f32 / grid.size as f32),
            );
        }
        let plate_id = plates.plate_field[i] as usize;
        let heat = plates.plate_vectors[plate_id].heat;
        let heat_norm = clampf(heat / tectonics.mantle_heat.max(1.0), 0.2, 2.3);
        heat_norm_field[i] = heat_norm;

        let sx = cell_cache.noise_x[i];
        let sy = cell_cache.noise_y[i];
        let sz = cell_cache.noise_z[i];
        let boundary = clampf(plates.boundary_strength[i], 0.0, 1.0);
        let fault = 0.5 + 0.5 * fault_backbone[i];
        let inherited_noise = 0.5
            + 0.5
                * ((sx * 5.7 + sy * 4.8 + sz * 3.9 + seed_mix as f32 * 0.0011).sin() * 0.61
                    + (sy * 6.3 - sz * 4.4 + sx * 3.2 - seed_mix as f32 * 0.0009).cos() * 0.39);
        let segment_noise = 0.5
            + 0.5
                * ((sx * 12.1 + sy * 8.7 + sz * 6.5 + seed_mix as f32 * 0.0018).sin() * 0.54
                    + (sy * 11.4 - sz * 7.9 + sx * 5.8 - seed_mix as f32 * 0.0014).cos() * 0.46);
        let segment_gate = smoothstep(0.24, 0.88, segment_noise);
        let inherited_gate = smoothstep(0.22, 0.9, inherited_noise * 0.76 + fault * 0.24);
        let weakness = clampf(
            0.14 + 0.58 * inherited_gate + 0.34 * boundary * segment_gate,
            0.0,
            1.0,
        );
        weakness_field[i] = weakness;

        let strength = clampf(
            0.72 + 0.23 * continentality_field[i] + 0.18 * (1.0 - weakness) - 0.24 * heat_norm,
            0.05,
            1.5,
        );
        strength_field[i] = strength;

        let mut conv_cc = 0.0_f32;
        let mut conv_oc_cont = 0.0_f32;
        let mut conv_oc_ocean = 0.0_f32;
        let mut conv_oo = 0.0_f32;
        let mut div_mix = 0.0_f32;
        let mut shear_mix = 0.0_f32;
        let mut local_wsum = 0.0_f32;
        let land_a = plate_landness[plate_id];
        let a = &plates.plate_vectors[plate_id];
        let (a_vx, a_vy) = plate_velocity_xy_at_cell(a, sx, sy, sz, radius_cm);
        for (dx, dy, w) in LOCAL_NEIGHBORS {
            let j = grid.neighbor((i % grid.width) as i32 + dx, (i / grid.width) as i32 + dy);
            let plate_b = plates.plate_field[j] as usize;
            if plate_b == plate_id {
                continue;
            }

            let b = &plates.plate_vectors[plate_b];
            let (b_vx, b_vy) = plate_velocity_xy_at_cell(b, sx, sy, sz, radius_cm);
            let rel_x = b_vx - a_vx;
            let rel_y = b_vy - a_vy;
            let n_len = (dx as f32).hypot(dy as f32).max(1.0);
            let nx = dx as f32 / n_len;
            let ny = dy as f32 / n_len;
            let tx = -ny;
            let ty = nx;
            let conv = (rel_x * nx + rel_y * ny).max(0.0);
            let div = (-(rel_x * nx + rel_y * ny)).max(0.0);
            let shear = (rel_x * tx + rel_y * ty).abs();
            let land_b = plate_landness[plate_b];
            let w_eff = w * (0.72 + 0.28 * boundary);

            if land_a > 0.57 && land_b > 0.57 {
                conv_cc += conv * w_eff;
                div_mix += div * 0.28 * w_eff;
            } else if land_a < 0.43 && land_b < 0.43 {
                conv_oo += conv * w_eff;
                div_mix += div * 0.94 * w_eff;
            } else {
                if land_a >= land_b {
                    conv_oc_cont += conv * w_eff;
                } else {
                    conv_oc_ocean += conv * w_eff;
                }
                div_mix += div * 0.62 * w_eff;
            }

            shear_mix += shear * w_eff;
            local_wsum += w_eff;
        }

        if local_wsum > 1e-5 {
            conv_cc /= local_wsum;
            conv_oc_cont /= local_wsum;
            conv_oc_ocean /= local_wsum;
            conv_oo /= local_wsum;
            div_mix /= local_wsum;
            shear_mix /= local_wsum;
        }

        let speed_norm = tectonics.plate_speed_cm_per_year.max(0.2);
        let conv_cc_n = clampf(conv_cc / speed_norm, 0.0, 1.0);
        let conv_oc_cont_n = clampf(conv_oc_cont / speed_norm, 0.0, 1.0);
        let conv_oc_ocean_n = clampf(conv_oc_ocean / speed_norm, 0.0, 1.0);
        let conv_oo_n = clampf(conv_oo / speed_norm, 0.0, 1.0);
        let div_n = clampf(div_mix / speed_norm, 0.0, 1.0);
        let shear_n = clampf(shear_mix / (speed_norm * 1.8), 0.0, 1.0);

        cc_collision[i] = conv_cc_n;
        oc_collision_cont[i] = conv_oc_cont_n;
        oc_collision_ocean[i] = conv_oc_ocean_n;
        oo_collision[i] = conv_oo_n;
        subduction_source[i] = clampf(conv_oc_ocean_n * 0.95 + conv_oo_n * 0.55, 0.0, 1.4);
        rift_source[i] = clampf(div_n * (0.58 + 0.42 * (1.0 - conv_cc_n)), 0.0, 1.4);

        let source = boundary * (0.38 + 0.62 * segment_gate) * (0.86 + 0.24 * fault);
        let collisional_mix = 0.6 * conv_cc_n + 0.42 * conv_oc_cont_n + 0.28 * conv_oo_n;
        match plates.boundary_types[i] {
            1 => {
                comp_source[i] = source
                    * (0.34
                        + 1.04 * collisional_mix
                        + 0.44 * inherited_gate
                        + 0.16 * shear_n);
                shear_source[i] =
                    source * (0.16 + 0.54 * shear_n + 0.18 * conv_oc_cont_n + 0.12 * conv_cc_n);
                ext_source[i] = source * 0.08 * div_n;
            }
            2 => {
                ext_source[i] = source * (0.4 + 0.82 * rift_source[i] + 0.2 * (1.0 - collisional_mix));
                shear_source[i] = source * (0.18 + 0.4 * shear_n + 0.18 * (1.0 - segment_gate));
                comp_source[i] = source * 0.05 * conv_oc_cont_n;
            }
            3 => {
                let transpress = shear_n * (0.46 + 0.54 * collisional_mix);
                shear_source[i] = source * (0.34 + 0.72 * transpress);
                comp_source[i] = source * 0.2 * transpress;
                ext_source[i] = source * 0.14 * (div_n * (1.0 - transpress));
            }
            _ => {}
        }

        velocity_x[i] = a_vx;
        velocity_y[i] = a_vy;
    }

    let collision_smooth_passes = (detail.max_kernel_radius as usize / 2 + 2).clamp(2, 6);
    let mut cc_next = cc_collision.clone();
    let mut oc_cont_next = oc_collision_cont.clone();
    let mut oc_ocean_next = oc_collision_ocean.clone();
    let mut oo_next = oo_collision.clone();
    let mut subduction_next = subduction_source.clone();
    let mut rift_next = rift_source.clone();
    for pass in 0..collision_smooth_passes {
        for y in 0..grid.height {
            let t =
                (pass as f32 + y as f32 / grid.height as f32) / collision_smooth_passes as f32;
            progress.phase(progress_base, progress_span, 0.28 + 0.02 * t);
            for x in 0..grid.width {
                let i = grid.index(x, y);
                let mut cc_sum = cc_collision[i] * 0.5;
                let mut oc_cont_sum = oc_collision_cont[i] * 0.5;
                let mut oc_ocean_sum = oc_collision_ocean[i] * 0.5;
                let mut oo_sum = oo_collision[i] * 0.5;
                let mut sub_sum = subduction_source[i] * 0.5;
                let mut rift_sum = rift_source[i] * 0.5;
                let mut wsum = 0.5;
                for oy in -1..=1 {
                    for ox in -1..=1 {
                        if ox == 0 && oy == 0 {
                            continue;
                        }
                        let j = grid.neighbor(x as i32 + ox, y as i32 + oy);
                        let w = if ox == 0 || oy == 0 { 0.11 } else { 0.075 };
                        cc_sum += cc_collision[j] * w;
                        oc_cont_sum += oc_collision_cont[j] * w;
                        oc_ocean_sum += oc_collision_ocean[j] * w;
                        oo_sum += oo_collision[j] * w;
                        sub_sum += subduction_source[j] * w;
                        rift_sum += rift_source[j] * w;
                        wsum += w;
                    }
                }
                cc_next[i] = cc_sum / wsum.max(1e-6);
                oc_cont_next[i] = oc_cont_sum / wsum.max(1e-6);
                oc_ocean_next[i] = oc_ocean_sum / wsum.max(1e-6);
                oo_next[i] = oo_sum / wsum.max(1e-6);
                subduction_next[i] = sub_sum / wsum.max(1e-6);
                rift_next[i] = rift_sum / wsum.max(1e-6);
            }
        }
        cc_collision.copy_from_slice(&cc_next);
        oc_collision_cont.copy_from_slice(&oc_cont_next);
        oc_collision_ocean.copy_from_slice(&oc_ocean_next);
        oo_collision.copy_from_slice(&oo_next);
        subduction_source.copy_from_slice(&subduction_next);
        rift_source.copy_from_slice(&rift_next);
    }

    let vel_smooth_passes = (detail.max_kernel_radius as usize + detail.buoyancy_smooth_passes / 6)
        .clamp(3, 8);
    let mut vel_x_next = velocity_x.clone();
    let mut vel_y_next = velocity_y.clone();
    for pass in 0..vel_smooth_passes {
        for y in 0..grid.height {
            let t = (pass as f32 + y as f32 / grid.height as f32) / vel_smooth_passes as f32;
            progress.phase(progress_base, progress_span, 0.28 + 0.04 * t);
            for x in 0..grid.width {
                let i = grid.index(x, y);
                let mut sx = velocity_x[i] * 0.34;
                let mut sy = velocity_y[i] * 0.34;
                let mut wsum = 0.34;
                for oy in -1..=1 {
                    for ox in -1..=1 {
                        if ox == 0 && oy == 0 {
                            continue;
                        }
                        let j = grid.neighbor(x as i32 + ox, y as i32 + oy);
                        let w = if ox == 0 || oy == 0 { 0.11 } else { 0.07 };
                        sx += velocity_x[j] * w;
                        sy += velocity_y[j] * w;
                        wsum += w;
                    }
                }
                vel_x_next[i] = sx / wsum.max(1e-6);
                vel_y_next[i] = sy / wsum.max(1e-6);
            }
        }
        velocity_x.copy_from_slice(&vel_x_next);
        velocity_y.copy_from_slice(&vel_y_next);
    }

    let mut comp_grad = vec![0.0_f32; grid.size];
    let mut ext_grad = vec![0.0_f32; grid.size];
    let mut shear_grad = vec![0.0_f32; grid.size];
    let mut comp_peak = 1e-6_f32;
    let mut ext_peak = 1e-6_f32;
    let mut shear_peak = 1e-6_f32;
    for y in 0..grid.height {
        let t = y as f32 / grid.height as f32;
        progress.phase(progress_base, progress_span, 0.32 + 0.02 * t);
        for x in 0..grid.width {
            let i = grid.index(x, y);
            let vx_r = velocity_x[grid.neighbor(x as i32 + 1, y as i32)];
            let vx_l = velocity_x[grid.neighbor(x as i32 - 1, y as i32)];
            let vx_u = velocity_x[grid.neighbor(x as i32, y as i32 - 1)];
            let vx_d = velocity_x[grid.neighbor(x as i32, y as i32 + 1)];
            let vy_r = velocity_y[grid.neighbor(x as i32 + 1, y as i32)];
            let vy_l = velocity_y[grid.neighbor(x as i32 - 1, y as i32)];
            let vy_u = velocity_y[grid.neighbor(x as i32, y as i32 - 1)];
            let vy_d = velocity_y[grid.neighbor(x as i32, y as i32 + 1)];
            let div = ((vx_r - vx_l) + (vy_d - vy_u)) * 0.5;
            let shear = ((vx_d - vx_u).abs() + (vy_r - vy_l).abs()) * 0.5;
            let cg = (-div).max(0.0);
            let eg = div.max(0.0);
            comp_grad[i] = cg;
            ext_grad[i] = eg;
            shear_grad[i] = shear;
            comp_peak = comp_peak.max(cg);
            ext_peak = ext_peak.max(eg);
            shear_peak = shear_peak.max(shear);
        }
    }

    for i in 0..grid.size {
        let weak = weakness_field[i];
        let cg = clampf(comp_grad[i] / comp_peak, 0.0, 1.0).powf(0.9);
        let eg = clampf(ext_grad[i] / ext_peak, 0.0, 1.0).powf(0.9);
        let sg = clampf(shear_grad[i] / shear_peak, 0.0, 1.0).powf(0.92);
        let collisional_boost =
            cc_collision[i] * 0.7 + oc_collision_cont[i] * 0.52 + oo_collision[i] * 0.34;
        let subduction_boost = subduction_source[i];
        let rift_boost = rift_source[i];
        comp_source[i] = clampf(
            comp_source[i] * 0.68
                + cg * (0.2 + 0.18 * weak)
                + collisional_boost * (0.24 + 0.26 * weak),
            0.0,
            1.9,
        );
        ext_source[i] = clampf(
            ext_source[i] * 0.68
                + eg * (0.2 + 0.14 * (1.0 - weak))
                + rift_boost * (0.22 + 0.18 * (1.0 - weak))
                + subduction_boost * 0.08,
            0.0,
            1.9,
        );
        shear_source[i] = clampf(
            shear_source[i] * 0.66
                + sg * (0.19 + 0.2 * weak)
                + collisional_boost * 0.12
                + subduction_boost * 0.1,
            0.0,
            1.7,
        );
    }

    let mut comp_field = comp_source.clone();
    let mut ext_field = ext_source.clone();
    let mut shear_field = shear_source.clone();
    let mut comp_next = vec![0.0_f32; grid.size];
    let mut ext_next = vec![0.0_f32; grid.size];
    let mut shear_next = vec![0.0_f32; grid.size];

    const STRAIN_NEIGHBORS: [(i32, i32, f32); 8] = [
        (1, 0, 0.17),
        (-1, 0, 0.17),
        (0, 1, 0.17),
        (0, -1, 0.17),
        (1, 1, 0.12),
        (-1, 1, 0.12),
        (1, -1, 0.12),
        (-1, -1, 0.12),
    ];

    let strain_passes = (detail.buoyancy_smooth_passes / 2 + detail.max_kernel_radius as usize + 5)
        .clamp(9, 26);
    for pass in 0..strain_passes {
        let pass_t = pass as f32 / strain_passes.max(1) as f32;
        for y in 0..grid.height {
            let row_t = (pass as f32 + y as f32 / grid.height as f32) / strain_passes as f32;
            progress.phase(progress_base, progress_span, 0.32 + 0.18 * row_t);
            for x in 0..grid.width {
                let i = grid.index(x, y);
                let mut nx = plates.boundary_normal_x[i];
                let mut ny = plates.boundary_normal_y[i];
                let mut nlen = nx.hypot(ny);
                if nlen < 0.15 {
                    let left = continentality_field[grid.neighbor(x as i32 - 1, y as i32)];
                    let right = continentality_field[grid.neighbor(x as i32 + 1, y as i32)];
                    let up = continentality_field[grid.neighbor(x as i32, y as i32 - 1)];
                    let down = continentality_field[grid.neighbor(x as i32, y as i32 + 1)];
                    nx = right - left;
                    ny = down - up;
                    nlen = nx.hypot(ny);
                }
                if nlen < 1e-5 {
                    nx = 1.0;
                    ny = 0.0;
                    nlen = 1.0;
                }
                let nux = nx / nlen;
                let nuy = ny / nlen;
                let tx = -nuy;
                let ty = nux;
                let weak = weakness_field[i];
                let heat_norm = heat_norm_field[i];
                let strength = strength_field[i];
                let conductivity = clampf(
                    0.12 + 0.54 * weak + 0.2 * heat_norm - 0.2 * strength,
                    0.05,
                    0.96,
                );

                let mut comp_sum = comp_field[i] * 0.42;
                let mut ext_sum = ext_field[i] * 0.42;
                let mut shear_sum = shear_field[i] * 0.42;
                let mut wsum = 0.42;

                for (dx, dy, base_w) in STRAIN_NEIGHBORS {
                    let j = grid.neighbor(x as i32 + dx, y as i32 + dy);
                    let along = ((dx as f32) * tx + (dy as f32) * ty).abs();
                    let across = ((dx as f32) * nux + (dy as f32) * nuy).abs();
                    let orient = clampf(0.78 + 0.34 * along - 0.12 * across, 0.5, 1.35);
                    let bridge = 0.55 + 0.45 * conductivity * (0.42 + 0.58 * weakness_field[j]);
                    let w = base_w * orient * bridge;
                    comp_sum += comp_field[j] * w;
                    ext_sum += ext_field[j] * w;
                    shear_sum += shear_field[j] * w;
                    wsum += w;
                }

                let comp_avg = comp_sum / wsum.max(1e-6);
                let ext_avg = ext_sum / wsum.max(1e-6);
                let shear_avg = shear_sum / wsum.max(1e-6);

                let comp_source_drive = comp_source[i] * (0.46 + 0.42 * weak);
                let ext_source_drive = ext_source[i] * (0.48 + 0.38 * (1.0 - weak));
                let shear_source_drive = shear_source[i] * (0.4 + 0.45 * weak);

                comp_next[i] = clampf(
                    comp_field[i] * (0.52 + 0.2 * (1.0 - conductivity))
                        + comp_avg * (0.3 + 0.44 * conductivity)
                        + comp_source_drive * 0.3
                        - (0.008 + 0.006 * pass_t),
                    0.0,
                    2.8,
                );
                ext_next[i] = clampf(
                    ext_field[i] * (0.54 + 0.2 * (1.0 - conductivity))
                        + ext_avg * (0.3 + 0.4 * conductivity)
                        + ext_source_drive * 0.31
                        - (0.007 + 0.005 * pass_t),
                    0.0,
                    2.8,
                );
                shear_next[i] = clampf(
                    shear_field[i] * (0.58 + 0.16 * (1.0 - conductivity))
                        + shear_avg * (0.27 + 0.36 * conductivity)
                        + shear_source_drive * 0.36
                        - (0.006 + 0.004 * pass_t),
                    0.0,
                    2.6,
                );
            }
        }

        comp_field.copy_from_slice(&comp_next);
        ext_field.copy_from_slice(&ext_next);
        shear_field.copy_from_slice(&shear_next);
    }

    let mut comp_peak = 1e-6_f32;
    let mut ext_peak = 1e-6_f32;
    let mut shear_peak = 1e-6_f32;
    for i in 0..grid.size {
        comp_peak = comp_peak.max(comp_field[i]);
        ext_peak = ext_peak.max(ext_field[i]);
        shear_peak = shear_peak.max(shear_field[i]);
    }
    for i in 0..grid.size {
        comp_field[i] = clampf(comp_field[i] / comp_peak, 0.0, 1.0).powf(0.86);
        ext_field[i] = clampf(ext_field[i] / ext_peak, 0.0, 1.0).powf(0.9);
        shear_field[i] = clampf(shear_field[i] / shear_peak, 0.0, 1.0).powf(0.94);
    }

    let corridor_passes = (detail.max_kernel_radius as usize / 2 + 2).clamp(2, 5);
    for pass in 0..corridor_passes {
        for y in 0..grid.height {
            let t = (pass as f32 + y as f32 / grid.height as f32) / corridor_passes as f32;
            progress.phase(progress_base, progress_span, 0.5 + 0.02 * t);
            for x in 0..grid.width {
                let i = grid.index(x, y);
                let weak = weakness_field[i];
                let mut comp_sum = comp_field[i] * 0.44;
                let mut ext_sum = ext_field[i] * 0.44;
                let mut shear_sum = shear_field[i] * 0.44;
                let mut wsum = 0.44;
                for oy in -1..=1 {
                    for ox in -1..=1 {
                        if ox == 0 && oy == 0 {
                            continue;
                        }
                        let j = grid.neighbor(x as i32 + ox, y as i32 + oy);
                        let w = if ox == 0 || oy == 0 { 0.1 } else { 0.065 };
                        comp_sum += comp_field[j] * w;
                        ext_sum += ext_field[j] * w;
                        shear_sum += shear_field[j] * w;
                        wsum += w;
                    }
                }
                let comp_avg = comp_sum / wsum.max(1e-6);
                let ext_avg = ext_sum / wsum.max(1e-6);
                let shear_avg = shear_sum / wsum.max(1e-6);
                comp_next[i] = clampf(
                    comp_field[i] * (0.6 - 0.12 * weak) + comp_avg * (0.3 + 0.32 * weak),
                    0.0,
                    1.0,
                );
                ext_next[i] = clampf(
                    ext_field[i] * (0.62 - 0.08 * weak) + ext_avg * (0.28 + 0.26 * (1.0 - weak)),
                    0.0,
                    1.0,
                );
                shear_next[i] = clampf(
                    shear_field[i] * (0.64 - 0.06 * weak)
                        + shear_avg * (0.24 + 0.22 * weak),
                    0.0,
                    1.0,
                );
            }
        }
        comp_field.copy_from_slice(&comp_next);
        ext_field.copy_from_slice(&ext_next);
        shear_field.copy_from_slice(&shear_next);
    }

    let mut orogen_drive = vec![0.0_f32; grid.size];
    let mut crust_eq = vec![0.0_f32; grid.size];
    let mut crust = vec![0.0_f32; grid.size];
    let mut crust_next = vec![0.0_f32; grid.size];
    for i in 0..grid.size {
        let continental = continentality_field[i];
        let landness = smoothstep(-0.08, 0.84, continental);
        let eq = lerpf(7.0, 34.8, landness) * (1.02 - 0.14 * heat_norm_field[i]);
        crust_eq[i] = eq;
        let inherited = smoothstep(0.2, 0.9, 0.5 + 0.5 * fault_backbone[i]);
        let collisionality =
            cc_collision[i] * 0.88 + oc_collision_cont[i] * 0.62 + oo_collision[i] * 0.42;
        let drive = smoothstep(
            0.1,
            0.9,
            comp_field[i] * (0.56 + 0.46 * weakness_field[i])
                + shear_field[i] * 0.18
                + inherited * 0.12 * weakness_field[i]
                + collisionality * (0.24 + 0.18 * landness),
        ) * (0.42 + 0.58 * landness);
        orogen_drive[i] = drive;
        crust[i] = clampf(
            eq + drive * (2.1 + 1.2 * collisionality)
                - ext_field[i] * (0.95 + 0.45 * rift_source[i])
                - subduction_source[i] * (0.6 + 0.62 * (1.0 - landness)),
            4.0,
            64.0,
        );
    }

    let time_steps =
        (detail.erosion_rounds * 9 + detail.max_kernel_radius as usize * 3 + 8).clamp(12, 40);
    for step in 0..time_steps {
        for y in 0..grid.height {
            let t = (step as f32 + y as f32 / grid.height as f32) / time_steps as f32;
            progress.phase(progress_base, progress_span, 0.5 + 0.12 * t);
            for x in 0..grid.width {
                let i = grid.index(x, y);
                let landness = smoothstep(-0.08, 0.84, continentality_field[i]);
                let weak = weakness_field[i];
                let strength = strength_field[i];
                let collisionality =
                    cc_collision[i] * 0.92 + oc_collision_cont[i] * 0.66 + oo_collision[i] * 0.48;
                let subduction = subduction_source[i];
                let rift = rift_source[i];
                let comp_eff = comp_field[i] * (0.54 + 0.62 * weak)
                    + shear_field[i] * 0.24 * (0.32 + 0.68 * landness)
                    + collisionality * (0.22 + 0.28 * landness);
                let ext_eff = ext_field[i] * (0.64 + 0.36 * (1.0 - landness))
                    + rift * 0.42
                    + subduction * 0.22;
                let thickening = (0.08 + 0.68 * comp_eff)
                    * (0.42 + 0.58 * landness)
                    * (1.14 - 0.3 * strength)
                    * (0.86 + 0.48 * collisionality);
                let thinning = (0.04 + 0.46 * ext_eff)
                    * (0.56 + 0.52 * heat_norm_field[i])
                    * (0.42 + 0.58 * (1.0 - landness));
                let relaxation = (crust[i] - crust_eq[i]).max(0.0)
                    * (0.017 + 0.03 * (1.0 - landness) + 0.01 * (step as f32 / time_steps as f32));
                let evolved = crust[i] + thickening - thinning - relaxation;

                let mut sum = evolved * 0.38;
                let mut wsum = 0.38;
                for oy in -1..=1 {
                    for ox in -1..=1 {
                        if ox == 0 && oy == 0 {
                            continue;
                        }
                        let j = grid.neighbor(x as i32 + ox, y as i32 + oy);
                        let w = if ox == 0 || oy == 0 { 0.1 } else { 0.065 };
                        sum += crust[j] * w;
                        wsum += w;
                    }
                }
                let avg = sum / wsum.max(1e-6);
                let corridor = comp_field[i]
                    .max(shear_field[i] * 0.85)
                    .max(ext_field[i] * 0.55)
                    .max(collisionality * 0.7);
                let lateral = clampf(
                    0.034 + 0.064 * (1.0 - corridor) + 0.03 * (1.0 - weak) - 0.012 * collisionality,
                    0.022,
                    0.16,
                );
                crust_next[i] = clampf(evolved * (1.0 - lateral) + avg * lateral, 4.0, 78.0);
            }
        }
        crust.copy_from_slice(&crust_next);
    }

    for i in 0..grid.size {
        if i % (grid.width * 4) == 0 {
            progress.phase(
                progress_base,
                progress_span,
                0.62 + 0.2 * (i as f32 / grid.size as f32),
            );
        }
        let plate_id = plates.plate_field[i] as usize;
        let plate_speed = plates.plate_vectors[plate_id].speed;
        let heat = plates.plate_vectors[plate_id].heat;
        let plate_buoyancy = buoyancy_field[i];
        let sx = cell_cache.noise_x[i];
        let sy = cell_cache.noise_y[i];
        let sz = cell_cache.noise_z[i];
        let continental_raw = continentality_field[i];

        let warp_u = spherical_fbm(
            sx * 1.15 + 3.0,
            sy * 1.15 - 5.0,
            sz * 1.15 + 7.0,
            seed as f32 * 0.0019 + 11.0,
        );
        let warp_v = spherical_fbm(
            sx * 1.67 - 13.0,
            sy * 1.67 + 17.0,
            sz * 1.67 - 19.0,
            seed as f32 * 0.0023 + 29.0,
        );
        let wx = sx + 0.34 * warp_u + 0.2 * warp_v;
        let wy = sy + 0.34 * warp_v + 0.2 * warp_u;
        let wz = sz + 0.27 * (warp_u - warp_v);

        let macro_noise = spherical_fbm(
            wx * 0.82 + 5.0,
            wy * 0.82 - 7.0,
            wz * 0.82 + 9.0,
            seed as f32 * 0.0027 + 17.0,
        );
        let regional_signal = spherical_fbm(
            wx * 1.72 - 11.0,
            wy * 1.72 + 13.0,
            wz * 1.72 - 3.0,
            seed as f32 * 0.0035 + 37.0,
        );
        let micro_signal = spherical_fbm(
            wx * 3.85 + 19.0,
            wy * 3.85 - 17.0,
            wz * 3.85 + 23.0,
            seed as f32 * 0.0049 + 53.0,
        );
        let shelf_noise = spherical_fbm(
            wx * 1.3 + 29.0,
            wy * 1.3 - 31.0,
            wz * 1.3 + 23.0,
            seed as f32 * 0.0039 + 61.0,
        );

        let land_core = continental_raw.max(0.0);
        let ocean_core = (-continental_raw).max(0.0);
        let shelf_span = 0.18 + 0.16 * (0.5 + 0.5 * shelf_noise);
        let coast_weight = 1.0 - smoothstep(0.05, shelf_span.max(0.08), continental_raw.abs());
        let interior_weight = 1.0 - coast_weight;
        let mountain_patch = smoothstep(0.2, 0.88, 0.5 + 0.5 * regional_signal);

        let tectonic_scale = ((k_relief * plate_speed) / planet.gravity.max(1.0)) * 0.0135;
        let land_gate = smoothstep(0.1, 0.95, continental_raw);
        let ocean_gate = smoothstep(0.2, 0.98, -continental_raw);
        let comp = comp_field[i];
        let ext = ext_field[i];
        let shear = shear_field[i];
        let collision_cc = smoothstep(0.08, 0.86, cc_collision[i]);
        let collision_oc_cont = smoothstep(0.08, 0.86, oc_collision_cont[i]);
        let collision_oc_ocean = smoothstep(0.08, 0.86, oc_collision_ocean[i]);
        let collision_oo = smoothstep(0.08, 0.86, oo_collision[i]);
        let subduction = clampf(subduction_source[i], 0.0, 1.6);
        let rift = clampf(rift_source[i], 0.0, 1.6);
        let crust_anom = crust[i] - crust_eq[i];
        let orogen = smoothstep(0.08, 0.9, orogen_drive[i]);
        let plateau = smoothstep(3.0, 18.0, crust_anom) * smoothstep(0.24, 0.92, land_gate);
        let segmentation = smoothstep(
            0.16,
            0.88,
            weakness_field[i] * 0.78 + 0.22 * (0.5 + 0.5 * fault_backbone[i]),
        );
        let collisional_mix = clampf(
            collision_cc * 0.74 + collision_oc_cont * 0.52 + collision_oo * 0.36,
            0.0,
            1.0,
        );
        let orogen_cc =
            smoothstep(0.16, 0.9, orogen * 0.62 + collision_cc * 0.92 + plateau * 0.24);
        let arc_orogen = smoothstep(
            0.14,
            0.9,
            collision_oc_cont * 0.84 + collision_oo * 0.66 + comp * 0.26,
        );

        let mountain_uplift = tectonic_scale
            * land_gate
            * (70.0
                + 1280.0 * orogen_cc * (0.46 + 0.54 * segmentation)
                + 690.0 * arc_orogen * (0.42 + 0.58 * (1.0 - segmentation * 0.55))
                + 280.0 * plateau
                + 210.0 * clampf(crust_anom / 20.0, 0.0, 1.0))
            * (0.56 + 0.44 * mountain_patch)
            * (0.38 + 0.62 * interior_weight);
        let transpress_uplift = tectonic_scale
            * land_gate
            * (42.0 + 260.0 * shear.powf(1.08))
            * smoothstep(0.14, 0.8, shear + comp * 0.32 + collisional_mix * 0.2);
        let ridge_ocean_uplift = tectonic_scale
            * ocean_gate
            * (30.0 + 540.0 * clampf(ext * 0.66 + rift * 0.58, 0.0, 2.0).powf(0.84))
            * (0.58 + 0.42 * (1.0 - collision_cc));
        let trench_driver = clampf(
            subduction * 0.94 + collision_oc_ocean * 0.7 + collision_oo * 0.56 + comp * 0.46,
            0.0,
            1.8,
        );
        let trench_cut = tectonic_scale
            * ocean_gate
            * (70.0 + 1650.0 * trench_driver.powf(0.9))
            * (0.66 + 0.34 * ocean_core.powf(0.42));
        let foreland_sag = tectonic_scale
            * land_gate
            * (22.0 + 150.0 * collision_cc)
            * (0.52 + 0.48 * (1.0 - orogen_cc))
            * (0.7 + 0.3 * interior_weight);
        let backarc_sag = tectonic_scale
            * land_gate
            * (18.0 + 138.0 * collision_oc_cont)
            * (0.45 + 0.55 * (1.0 - arc_orogen))
            * (0.68 + 0.32 * interior_weight);
        let transform_term = tectonic_scale
            * 140.0
            * shear.powf(0.9)
            * (regional_signal * 0.6 + micro_signal * 0.4)
            * (land_gate + ocean_gate).min(1.0)
            * (0.62 + 0.38 * (1.0 - collision_cc));
        let base = mountain_uplift + transpress_uplift + ridge_ocean_uplift
            - trench_cut
            - foreland_sag
            - backarc_sag
            + transform_term;

        // Empirical continental/oceanic base (legacy formula kept for blend)
        let empirical_base = if continental_raw >= 0.0 {
            85.0 + land_core.powf(1.16) * (1480.0 + 2350.0 * smoothstep(0.18, 0.96, land_core))
        } else {
            -(170.0 + ocean_core.powf(1.18) * (1860.0 + 3180.0 * smoothstep(0.22, 0.98, ocean_core)))
        };

        // Airy isostasy: crust thickness → elevation (Turcotte & Schubert 2002)
        let is_continental = continental_raw >= 0.0;
        let heat_anomaly = (heat_norm_field[i] - 0.2) / 2.1; // normalise [0.2,2.3] → [0,1]
        let isostatic_base = isostatic_elevation(crust[i], is_continental, heat_anomaly);

        // Blend: 60 % physics + 40 % empirical ensures smooth transition to full isostasy
        let continental_base = isostatic_base * 0.6 + empirical_base * 0.4;

        let litho_variation = (regional_signal * 112.0
            + micro_signal * 58.0
            + macro_noise * 44.0
            + plate_buoyancy * 14.0
            + crust_anom * (14.0 + 26.0 * orogen))
            * (0.2 + 0.8 * interior_weight);
        let macro_base = continental_base + litho_variation;

        let noise = (random_seed.next_f32() - 0.5) * 10.0
            + regional_signal * 21.0
            + micro_signal * (6.5 + 9.0 * interior_weight);
        let heat_term = (heat - tectonics.mantle_heat * 0.5) * 0.95;

        relief[i] = base + macro_base + noise + heat_term;
    }

    let mut macro_blend = vec![0.0_f32; grid.size];
    let macro_radius = 1_i32;

    for y in 0..grid.height {
        progress.phase(
            progress_base,
            progress_span,
            0.6 + 0.1 * (y as f32 / grid.height as f32),
        );
        for x in 0..grid.width {
            let i = grid.index(x, y);
            let mut sum = 0.0_f32;
            let mut wsum = 0.0_f32;
            for oy in -macro_radius..=macro_radius {
                for ox in -macro_radius..=macro_radius {
                    let j = grid.neighbor(x as i32 + ox, y as i32 + oy);
                    let dist2 = (ox * ox + oy * oy) as f32;
                    let w = (-dist2 / 1.9).exp();
                    sum += relief[j] * w;
                    wsum += w;
                }
            }
            macro_blend[i] = sum / wsum.max(1e-6);
        }
    }

    for i in 0..grid.size {
        relief[i] = relief[i] * 0.9 + macro_blend[i] * 0.1;
    }

    let erosion_rounds = detail.erosion_rounds;
    let mut smoothed = relief.clone();
    let mut scratch = vec![0.0_f32; grid.size];

    for round in 0..erosion_rounds {
        for y in 0..grid.height {
            let round_t =
                (round as f32 + y as f32 / grid.height as f32) / erosion_rounds.max(1) as f32;
            progress.phase(progress_base, progress_span, 0.72 + 0.14 * round_t);
            for x in 0..grid.width {
                let i = grid.index(x, y);
                let mut sum = 0.0_f32;
                let mut count = 0.0_f32;
                for oy in -1..=1 {
                    for ox in -1..=1 {
                        let j = grid.neighbor(x as i32 + ox, y as i32 + oy);
                        sum += smoothed[j];
                        count += 1.0;
                    }
                }
                let avg = sum / count;
                scratch[i] = smoothed[i] * 0.95 + avg * 0.05;
            }
        }

        for i in 0..grid.size {
            let drop = (smoothed[i] - scratch[i]).max(0.0);
            let height_loss = (drop * 0.14).min(16.0);
            smoothed[i] = scratch[i] - height_loss;
        }
    }

    relief.copy_from_slice(&smoothed);

    let ocean_cut = (((planet.ocean_percent / 100.0) * grid.size as f32).floor() as isize)
        .clamp(0, (grid.size - 1) as isize) as usize;

    let mut sorted = relief.clone();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let sea_level = *sorted.get(ocean_cut).unwrap_or(&0.0);
    for h in relief.iter_mut() {
        *h -= sea_level;
    }

    // Coastal/ocean processing — planet scope only.
    // For island scope, coastline emerges from stream power erosion (Phase J).
    if grid.is_spherical {
        let cache = world_cache();
        stabilize_landmass_topology(&mut relief, planet, detail);
        apply_coastal_detail(&mut relief, seed, cache);
        defuse_plate_linearity(&mut relief, plates, seed, cache, detail);
        defuse_coastal_linearity(&mut relief, plates, seed_mix ^ 0x1F23_BA47, cache);
        fracture_coastline_band(&mut relief, seed_mix ^ 0x7B91_5C31, cache);
        warp_coastal_band(&mut relief, seed_mix ^ 0x6A31_91D7, cache);
        break_straight_coasts(&mut relief, seed_mix ^ 0x2CD8_7A43, cache);
        cleanup_coastal_speckles(&mut relief);

        let mut sorted_after_coast = relief.clone();
        sorted_after_coast.sort_by(|a, b| a.total_cmp(b));
        let coast_recenter = *sorted_after_coast.get(ocean_cut).unwrap_or(&0.0);
        for h in relief.iter_mut() {
            *h -= coast_recenter;
        }
    }

    normalize_height_range(&mut relief, planet, tectonics);
    if grid.is_spherical && detail.erosion_rounds == 0 {
        let cache = world_cache();
        defuse_orogenic_ribbons(&mut relief, seed_mix, cache, detail);
    }
    // Fluvial valley carving — planet scope only.
    // Island scope uses stream_power_evolve + carve_fluvial_valleys_grid instead.
    if grid.is_spherical {
        carve_fluvial_valleys(
            &mut relief,
            grid,
            detail,
            progress,
            progress_base + progress_span * 0.82,
            progress_span * 0.08,
        );
    }
    progress.phase(progress_base, progress_span, 0.9);

    // Ocean profile and final coastal cleanup — planet scope only.
    if grid.is_spherical {
        let cache = world_cache();
        apply_ocean_profile(
            &mut relief,
            &plates.boundary_types,
            &plates.boundary_strength,
            planet,
            seed,
            cache,
            detail,
            progress,
            progress_base + progress_span * 0.9,
            progress_span * 0.08,
        );
        reshape_ocean_boundaries(
            &mut relief,
            &plates.boundary_types,
            &plates.boundary_strength,
        );
        cleanup_coastal_speckles(&mut relief);
        suppress_subgrid_islands(&mut relief, planet);
        dampen_isolated_shallow_ocean(&mut relief);
        progress.phase(progress_base, progress_span, 0.985);

        let mut sorted_final = relief.clone();
        sorted_final.sort_by(|a, b| a.total_cmp(b));
        let final_recenter = *sorted_final.get(ocean_cut).unwrap_or(&0.0);
        for h in relief.iter_mut() {
            *h -= final_recenter;
        }
        suppress_subgrid_islands(&mut relief, planet);
        dampen_isolated_shallow_ocean(&mut relief);
    }
    progress.phase(progress_base, progress_span, 1.0);

    ReliefResult { relief, sea_level }
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
            let crater_radius_km = 8.0_f32.max(energy.powf(1.0 / 5.0) / 2500.0);
            let crater_depth = 9000.0_f32.min(800.0 + ((energy + 1.0).log10() - 10.0) * 250.0);
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
                        updated[target] -= crater_depth * falloff.max(0.0) * 0.5;
                        updated[target] = updated[target].max(-planet.radius_km * 10.0);
                    }
                }
            }

            aerosol_index += 0.45_f32.min((energy + 1.0).log10() / 18.0);
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

fn compute_hydrology(
    heights: &[f32],
    slope: &[f32],
    progress: &mut ProgressTap<'_>,
    progress_base: f32,
    progress_span: f32,
) -> (Vec<i32>, Vec<f32>, Vec<f32>, Vec<u8>) {
    let mut flow_direction = vec![-1_i32; WORLD_SIZE];
    let mut secondary_direction = vec![-1_i32; WORLD_SIZE];
    let mut primary_share = vec![1.0_f32; WORLD_SIZE];
    let mut flow_accumulation = vec![1.0_f32; WORLD_SIZE];

    for y in 0..WORLD_HEIGHT {
        progress.phase(
            progress_base,
            progress_span,
            0.45 * (y as f32 / WORLD_HEIGHT as f32),
        );
        for x in 0..WORLD_WIDTH {
            let i = index(x, y);
            let h = heights[i];
            let mut best_drop = 0.0_f32;
            let mut second_drop = 0.0_f32;
            let mut best = -1_i32;
            let mut second = -1_i32;

            for oy in -1..=1 {
                for ox in -1..=1 {
                    if ox == 0 && oy == 0 {
                        continue;
                    }
                    let j = index_spherical(x as i32 + ox, y as i32 + oy);
                    let drop = h - heights[j];
                    if drop > best_drop {
                        second_drop = best_drop;
                        second = best;
                        best_drop = drop;
                        best = j as i32;
                    } else if drop > second_drop {
                        second_drop = drop;
                        second = j as i32;
                    }
                }
            }

            flow_direction[i] = best;
            secondary_direction[i] = second;
            if best >= 0 && second >= 0 {
                let total = (best_drop + second_drop).max(1e-6);
                primary_share[i] = clampf(best_drop / total, 0.55, 0.92);
            } else {
                primary_share[i] = 1.0;
            }
        }
    }

    let mut order: Vec<usize> = (0..WORLD_SIZE).collect();
    order.sort_by(|a, b| heights[*b].total_cmp(&heights[*a]));
    progress.phase(progress_base, progress_span, 0.48);

    for (rank, i) in order.iter().copied().enumerate() {
        if rank % (WORLD_WIDTH * 4) == 0 {
            progress.phase(
                progress_base,
                progress_span,
                0.48 + 0.35 * (rank as f32 / WORLD_SIZE as f32),
            );
        }
        let to = flow_direction[i];
        let scale = 1.0 + slope[i] / 1000.0;
        if to >= 0 {
            flow_accumulation[to as usize] += flow_accumulation[i] * primary_share[i] * scale;
            let alt = secondary_direction[i];
            if alt >= 0 {
                let spill = (1.0 - primary_share[i]) * 0.9;
                flow_accumulation[alt as usize] += flow_accumulation[i] * spill * scale;
            }
        }
    }
    progress.phase(progress_base, progress_span, 0.83);

    let mut sorted = flow_accumulation.clone();
    sorted.sort_by(|a, b| b.total_cmp(a));
    let threshold_index = ((sorted.len() as f32 * 0.992).floor() as usize).min(sorted.len() - 1);
    let threshold = sorted[threshold_index].max(1.0);

    let mut rivers = vec![0.0_f32; WORLD_SIZE];
    let mut lakes = vec![0_u8; WORLD_SIZE];

    for i in 0..WORLD_SIZE {
        if i % (WORLD_WIDTH * 4) == 0 {
            progress.phase(
                progress_base,
                progress_span,
                0.83 + 0.17 * (i as f32 / WORLD_SIZE as f32),
            );
        }
        rivers[i] = (flow_accumulation[i] / threshold).powf(0.9);
        if flow_direction[i] < 0 && heights[i] > 0.0 && slope[i] < 25.0 {
            lakes[i] = 1;
        }
    }
    progress.phase(progress_base, progress_span, 1.0);

    (flow_direction, flow_accumulation, rivers, lakes)
}

fn compute_climate(
    planet: &PlanetInputs,
    heights: &[f32],
    slope: &[f32],
    rivers: &[f32],
    flow: &[f32],
    aerosol: f32,
    cache: &WorldCache,
    progress: &mut ProgressTap<'_>,
    progress_base: f32,
    progress_span: f32,
) -> (Vec<f32>, Vec<f32>, f32, f32, f32, f32) {
    let mut temperature = vec![0.0_f32; WORLD_SIZE];
    let mut precipitation = vec![0.0_f32; WORLD_SIZE];
    let tilt_rad = planet.axial_tilt_deg * RADIANS;
    let rotation_factor = 20.0 / planet.rotation_hours.max(1.0);

    let mut min_temp = f32::INFINITY;
    let mut max_temp = -f32::INFINITY;
    let mut min_prec = f32::INFINITY;
    let mut max_prec = -f32::INFINITY;

    for i in 0..WORLD_SIZE {
        if i % (WORLD_WIDTH * 4) == 0 {
            progress.phase(progress_base, progress_span, i as f32 / WORLD_SIZE as f32);
        }
        let lat = cache.lat_by_y[i] * RADIANS;
        let base_insolation = (lat - tilt_rad).cos().max(0.0) * (1.0 + planet.eccentricity * 0.35);
        let elevation_for_temperature = heights[i].max(0.0);
        let base_temp =
            55.0 * base_insolation - 0.0065 * elevation_for_temperature - lat.abs() * 0.4;
        let pressure_term = 12.0 * planet.atmosphere_bar.max(0.0).ln_1p();
        let ocean_term = if heights[i] < 0.0 { 6.0 } else { 0.0 };
        let seasonal = 1.0 + 0.08 * ((lat + 0.2 * std::f32::consts::PI * tilt_rad) * 2.0).sin();
        let temp =
            base_temp + pressure_term + ocean_term + 0.12 * rotation_factor * 100.0 * seasonal
                - aerosol * 8.0;
        let slope_orographic = 1.0 + (slope[i].abs() / 400.0).min(1.4);
        let precip_raw = 600.0
            + 1800.0 * base_insolation * seasonal
            + 1200.0 * rivers[i].min(1.0) * slope_orographic
            + flow[i] * 0.1;
        let precip = (precip_raw - aerosol * 250.0).max(0.0);

        temperature[i] = clampf(temp, -90.0, 70.0);
        precipitation[i] = precip;

        min_temp = min_temp.min(temperature[i]);
        max_temp = max_temp.max(temperature[i]);
        min_prec = min_prec.min(precip);
        max_prec = max_prec.max(precip);
    }
    progress.phase(progress_base, progress_span, 1.0);

    (
        temperature,
        precipitation,
        min_temp,
        max_temp,
        min_prec,
        max_prec,
    )
}

#[inline]
fn classify_biome(temp: f32, precip: f32, height: f32) -> u8 {
    if height < 0.0 {
        return 0;
    }
    if temp < -5.0 {
        return 1;
    }
    if precip > 1600.0 && temp >= 5.0 && temp < 22.0 {
        return 2;
    }
    if temp > 12.0 && precip > 900.0 {
        return 3;
    }
    if temp > 24.0 && precip > 500.0 && precip <= 1400.0 {
        return 4;
    }
    if precip < 400.0 && temp > 18.0 {
        return 5;
    }
    if precip > 700.0 && temp > 5.0 && temp < 22.0 {
        return 6;
    }
    if height > 1800.0 {
        return 7;
    }
    8
}

fn compute_biomes(temperature: &[f32], precipitation: &[f32], heights: &[f32]) -> Vec<u8> {
    let mut biomes = vec![0_u8; WORLD_SIZE];
    for i in 0..WORLD_SIZE {
        biomes[i] = classify_biome(temperature[i], precipitation[i], heights[i]);
    }
    biomes
}

fn compute_settlement(
    biomes: &[u8],
    heights: &[f32],
    temperature: &[f32],
    precipitation: &[f32],
) -> Vec<f32> {
    let mut settlement = vec![0.0_f32; WORLD_SIZE];
    for i in 0..WORLD_SIZE {
        if biomes[i] == 0 {
            settlement[i] = 0.0;
            continue;
        }
        let comfort_t = 1.0 - (temperature[i] - 18.0).abs() / 45.0;
        let comfort_p = 1.0 - (precipitation[i] - 1400.0).abs() / 2200.0;
        let elevation_penalty = (heights[i] - 1200.0).max(0.0) / 2600.0;
        settlement[i] = ((comfort_t + comfort_p) / 2.0 - elevation_penalty).clamp(0.0, 1.0);
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
    let threshold =
        ((size as f32) * (0.00018 + detail.fluvial_rounds as f32 * 0.00007)).max(22.0);

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

fn carve_fluvial_valleys_grid(
    relief: &mut [f32],
    flow_accumulation: &[f32],
    river_map: &[f32],
    width: usize,
    height: usize,
    detail: DetailProfile,
    progress: &mut ProgressTap<'_>,
    progress_base: f32,
    progress_span: f32,
) {
    let rounds = detail.fluvial_rounds.max(1);
    let max_acc = flow_accumulation
        .iter()
        .copied()
        .fold(0.0_f32, f32::max)
        .max(1.0);
    let mut scratch = relief.to_vec();

    for round in 0..rounds {
        let round_base = progress_base + progress_span * (round as f32 / rounds as f32);
        let round_span = progress_span / rounds as f32;
        for y in 0..height {
            progress.phase(round_base, round_span * 0.8, y as f32 / height as f32);
            for x in 0..width {
                let i = grid_index(x, y, width);
                let h = relief[i];
                if h <= 8.0 {
                    scratch[i] = h;
                    continue;
                }
                let acc_n = (flow_accumulation[i] / max_acc).powf(0.52);
                let river = river_map[i];
                let raw_cut =
                    acc_n * (0.28 + river * 1.55) * (24.0 + round as f32 * 11.0 + detail.erosion_rounds as f32 * 2.0);
                let cut = raw_cut.min(h * 0.23);
                let mut next = h - cut;
                if h > 1600.0 {
                    next = lerpf(next, h, 0.42);
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

fn compute_climate_grid(
    planet: &PlanetInputs,
    heights: &[f32],
    slope: &[f32],
    river_map: &[f32],
    flow_accumulation: &[f32],
    coastal_exposure: &[f32],
    width: usize,
    height: usize,
    seed: u32,
    progress: &mut ProgressTap<'_>,
    progress_base: f32,
    progress_span: f32,
) -> (Vec<f32>, Vec<f32>, f32, f32, f32, f32) {
    let mut temperature = vec![0.0_f32; heights.len()];
    let mut precipitation = vec![0.0_f32; heights.len()];
    let mut min_temp = f32::INFINITY;
    let mut max_temp = -f32::INFINITY;
    let mut min_prec = f32::INFINITY;
    let mut max_prec = -f32::INFINITY;

    let temp_seed = hash_u32(seed ^ 0x71F1_2A8B);
    let precip_seed = hash_u32(seed ^ 0x92E5_14C3);

    for y in 0..height {
        progress.phase(progress_base, progress_span, y as f32 / height as f32);
        let lat = ((y as f32 + 0.5) / height as f32 - 0.5).abs() * 2.0;
        for x in 0..width {
            let i = grid_index(x, y, width);
            let h = heights[i].max(0.0);
            let uvx = x as f32 / width as f32;
            let uvy = y as f32 / height as f32;

            let temp_noise = value_noise3(
                uvx * 7.2 + 4.0,
                uvy * 7.2 - 6.0,
                seed as f32 * 0.000_21,
                temp_seed,
            );
            let precip_noise = value_noise3(
                uvx * 8.6 - 9.0,
                uvy * 8.6 + 7.0,
                seed as f32 * 0.000_31,
                precip_seed,
            );

            let base_temp = 25.0
                - lat * (9.0 + planet.axial_tilt_deg * 0.09)
                - h * (0.0054 + 0.00008 * planet.gravity);
            let ocean_inertia = if heights[i] <= 0.0 { 1.6 } else { 0.0 };
            let t = base_temp + temp_noise * 2.2 + ocean_inertia;
            temperature[i] = t;
            min_temp = min_temp.min(t);
            max_temp = max_temp.max(t);

            let west = heights[grid_index_clamped(
                x as i32 - 1,
                y as i32,
                width,
                height,
            )]
            .max(0.0);
            let east = heights[grid_index_clamped(
                x as i32 + 1,
                y as i32,
                width,
                height,
            )]
            .max(0.0);
            let uplift = (h - west).max(0.0);
            let rain_shadow = (west - h).max(0.0) + 0.35 * (east - h).max(0.0);
            let acc_term = (flow_accumulation[i].ln_1p() / 8.0).clamp(0.0, 1.0);
            let slope_term = slope[i] / (slope[i] + 80.0);

            let p = 340.0
                + coastal_exposure[i] * 980.0
                + river_map[i] * 540.0
                + acc_term * 240.0
                + uplift * 0.24
                - rain_shadow * 0.2
                + slope_term * 60.0
                + (precip_noise * 0.5 + 0.5) * 190.0
                - h * 0.047
                + (planet.atmosphere_bar - 1.0) * 130.0;

            let precip = clampf(p, 65.0, 3600.0);
            precipitation[i] = precip;
            min_prec = min_prec.min(precip);
            max_prec = max_prec.max(precip);
        }
    }

    (
        temperature,
        precipitation,
        min_temp,
        max_temp,
        min_prec,
        max_prec,
    )
}

fn compute_biomes_grid(temperature: &[f32], precipitation: &[f32], heights: &[f32]) -> Vec<u8> {
    let mut biomes = vec![0_u8; heights.len()];
    for i in 0..heights.len() {
        biomes[i] = classify_biome(temperature[i], precipitation[i], heights[i]);
    }
    biomes
}

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
        let comfort_t = 1.0 - (temperature[i] - 17.0).abs() / 32.0;
        let comfort_p = 1.0 - (precipitation[i] - 1200.0).abs() / 1800.0;
        let elevation_penalty = heights[i].max(0.0) / 3000.0;
        let river_bonus = river_map[i] * 0.42;
        let coast_bonus = coastal_exposure[i] * 0.3;
        settlement[i] = clampf(
            comfort_t * 0.45 + comfort_p * 0.45 + river_bonus + coast_bonus - elevation_penalty,
            0.0,
            1.0,
        );
    }
    settlement
}

// ---------------------------------------------------------------------------
// Phase J: Braun-Willett O(n) stream power erosion (Braun & Willett 2013)
// ---------------------------------------------------------------------------

/// D8 flow routing: for each cell, find the steepest-descent neighbour index.
/// Returns `receivers[i] == i` if cell i is a local sink (no lower neighbour).
fn compute_d8_receivers(height: &[f32], grid: &GridConfig) -> Vec<usize> {
    let w = grid.width;
    let h = grid.height;
    let mut receivers = (0..grid.size).collect::<Vec<_>>();

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
                    // Diagonal neighbours have slightly larger distance.
                    let dist = if ox == 0 || oy == 0 { 1.0_f32 } else { std::f32::consts::SQRT_2 };
                    let drop = (hi - height[j]) / dist;
                    if drop > best_drop {
                        best_drop = drop;
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

/// Apply one implicit Braun-Willett stream power timestep.
///
/// Stream power law:  E = K * A^m * S^n
/// Implicit update:   h_new = (h_old + U*dt + factor * h_recv) / (1 + factor)
///                    where factor = K * dt * A^m / dx^n
///
/// Parameters use SI units throughout (metres, years).
fn stream_power_step(
    height: &mut [f32],
    uplift: &[f32],
    k_eff: &[f32],
    dt_yr: f32,
    dx_m: f32,
    m: f32, // area exponent (canonical 0.5)
    n: f32, // slope exponent (canonical 1.0)
    kappa: f32, // hillslope diffusivity m²/yr
    grid: &GridConfig,
) {
    let size = grid.size;

    // 1. D8 flow receivers
    let receivers = compute_d8_receivers(height, grid);

    // 2. Topological sort (high → low)
    let order = topological_sort_descending(height);

    // 3. Drainage area accumulation (upstream → downstream, i.e. forward in order)
    let cell_area = dx_m * dx_m;
    let mut area = vec![cell_area; size];
    for &i in order.iter() {
        let r = receivers[i];
        if r != i {
            let area_i = area[i];
            area[r] += area_i;
        }
    }

    // 4. Implicit elevation update (downstream → upstream, i.e. reverse order).
    //    Ocean/below-sea-level cells (h <= 0) are base-level boundaries: skip them.
    //    For coastal land cells whose receiver is ocean, use sea level (0) as the
    //    base level so rivers erode to sea level, not to the ocean floor depth.
    for &i in order.iter().rev() {
        if height[i] <= 0.0 {
            // Ocean floor: preserve as-is — it is the erosion base level.
            continue;
        }
        let r = receivers[i];
        if r == i {
            // Land depression (local sink): apply uplift only.
            height[i] += uplift[i] * dt_yr;
            continue;
        }
        let k = k_eff[i];
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
/// Also handles progress reporting.
fn stream_power_evolve(
    height: &mut [f32],
    uplift: &[f32],
    k_eff: &[f32],
    dt_yr: f32,
    dx_m: f32,
    m: f32,
    n_exp: f32,
    kappa: f32,
    n_steps: usize,
    grid: &GridConfig,
    progress: &mut ProgressTap<'_>,
    progress_base: f32,
    progress_span: f32,
) {
    for step in 0..n_steps {
        let t = step as f32 / n_steps.max(1) as f32;
        progress.phase(progress_base, progress_span, t);
        stream_power_step(height, uplift, k_eff, dt_yr, dx_m, m, n_exp, kappa, grid);
    }
    progress.phase(progress_base, progress_span, 1.0);
}

/// Generate the per-cell uplift field (m/yr) for island scope.
/// Based on tectonic type from the plate context and island geometry.
fn generate_island_uplift_field(
    island_type: IslandType,
    plates: &ComputePlatesResult,
    grid: &GridConfig,
    params: &TectonicInputs,
    seed: u32,
) -> Vec<f32> {
    let w = grid.width;
    let h = grid.height;
    let mut uplift = vec![0.0_f32; grid.size];
    let mut rng = Rng::new(seed ^ 0x3A7F_991C);
    let base_speed_m_yr = params.plate_speed_cm_per_year * 0.01; // cm/yr → m/yr

    match island_type {
        IslandType::Continental => {
            // Horst-graben uplift field along NNW-trending faults.
            let u_base = base_speed_m_yr * 0.004; // ~0.2 mm/yr
            let fault_angle = -20.0_f32.to_radians(); // NNW trend
            let fault_spacing = (w.min(h) as f32 * grid.km_per_cell_x) * 0.25; // ~25% grid width
            let fault_spacing_px = fault_spacing / grid.km_per_cell_x;
            let asym = 0.3 + rng.next_f32() * 0.4;

            for y in 0..h {
                for x in 0..w {
                    let i = grid.index(x, y);
                    let xf = (x as f32 + 0.5 - w as f32 * 0.5) * grid.km_per_cell_x;
                    let yf = (y as f32 + 0.5 - h as f32 * 0.5) * grid.km_per_cell_y;
                    // Projection onto fault-perpendicular axis
                    let perp = xf * fault_angle.cos() + yf * fault_angle.sin();
                    let phase = perp / fault_spacing_px;
                    let block = (phase * std::f32::consts::PI).sin();
                    // Boundary strength as proxy for existing plate boundary proximity
                    let bs = plates.boundary_strength[i];
                    // Asymmetric east-west tilt
                    let asym_factor = asym * (xf / (w as f32 * grid.km_per_cell_x * 0.5));
                    uplift[i] = u_base * (0.5 + 0.5 * block + bs * 0.3 + asym_factor * 0.2)
                        .max(0.0);
                }
            }
        }

        IslandType::Arc => {
            // Gaussian arc uplift centred on subduction boundary.
            let arc_cx = w as f32 * 0.5;
            let sigma_px = w as f32 * 0.18;
            let u_max = base_speed_m_yr * 0.04; // ~2 mm/yr at arc centre

            for y in 0..h {
                for x in 0..w {
                    let i = grid.index(x, y);
                    let dx_px = x as f32 + 0.5 - arc_cx;
                    let dy_px = y as f32 + 0.5 - h as f32 * 0.5;
                    let r2 = dx_px * dx_px + dy_px * dy_px;
                    uplift[i] = u_max * (-(r2) / (2.0 * sigma_px * sigma_px)).exp();
                }
            }
        }

        IslandType::Hotspot => {
            // Central Gaussian uplift (shield volcano).
            let cx = w as f32 * 0.5;
            let cy = h as f32 * 0.5;
            let sigma_px = w.min(h) as f32 * 0.25;
            let u_max = base_speed_m_yr * 0.15; // ~7.5 mm/yr at shield centre (Hawaii-like)

            for y in 0..h {
                for x in 0..w {
                    let i = grid.index(x, y);
                    let dx_px = x as f32 + 0.5 - cx;
                    let dy_px = y as f32 + 0.5 - cy;
                    let r2 = dx_px * dx_px + dy_px * dy_px;
                    uplift[i] = u_max * (-(r2) / (2.0 * sigma_px * sigma_px)).exp();
                }
            }
        }

        IslandType::Rift => {
            // Exponential decay from rift shoulder (left edge) toward right.
            let lambda_px = w as f32 * 0.30;
            let u_shoulder = base_speed_m_yr * 0.01; // ~0.5 mm/yr at rift shoulder

            for y in 0..h {
                for x in 0..w {
                    let i = grid.index(x, y);
                    let d_px = x as f32 + 0.5; // distance from left (rift) edge
                    // Shoulder on left, subsidence on right
                    let shoulder = u_shoulder * (-d_px / lambda_px).exp();
                    let subsidence = u_shoulder * 0.3 * (1.0 - (-d_px / (lambda_px * 2.0)).exp());
                    uplift[i] = (shoulder - subsidence).max(0.0);
                }
            }
        }
    }

    uplift
}

// ---------------------------------------------------------------------------
// Phase I: Synthetic island plate context
// ---------------------------------------------------------------------------

/// Island tectonic type — controls the generated plate configuration.
#[derive(Clone, Copy, PartialEq)]
enum IslandType {
    /// Continental fragment (Tasmania-like): mixed transform + divergent boundaries.
    Continental,
    /// Convergent arc (Japan-like): subduction with back-arc basin.
    Arc,
    /// Hot-spot volcanic shield (Hawaii-like): single plate, central uplift.
    Hotspot,
    /// Rift shoulder (Corsica-like): asymmetric divergent boundary.
    Rift,
}

/// Generate a synthetic `ComputePlatesResult` for island scope without running
/// the full Voronoi plate simulation.
///
/// For **Continental** type the algorithm:
/// 1. Creates 3 plates separated by 1–2 linear fault lines through the grid.
/// 2. Assigns continental vs oceanic buoyancy/heat per plate.
/// 3. Sets boundary_types (transform + divergent mix), boundary normals,
///    and boundary_strength from distance to the fault line.
/// 4. Gives each plate a plausible velocity via small Euler poles.
fn generate_island_plate_context(
    seed: u32,
    island_type: IslandType,
    grid: &GridConfig,
    params: &TectonicInputs,
) -> ComputePlatesResult {
    let size = grid.size;
    let w = grid.width;
    let h = grid.height;
    let mut rng = Rng::new(seed ^ 0xC4E2_91AB);

    match island_type {
        IslandType::Continental => {
            // ------------------------------------------------------------------
            // Continental fragment: 2–3 plates separated by diagonal fault lines.
            // ------------------------------------------------------------------
            let _num_plates = 3usize;

            // --- Plate vectors ------------------------------------------------
            let base_speed = params.plate_speed_cm_per_year.max(0.5);
            let base_heat = params.mantle_heat;
            let plate_vectors = vec![
                // Plate 0 — continental block (west / central)
                PlateVector {
                    speed: base_speed * random_range(&mut rng, 0.6, 1.2),
                    omega_x: (rng.next_f32() - 0.5) * 0.04,
                    omega_y: (rng.next_f32() - 0.5) * 0.04,
                    omega_z: 0.01 + rng.next_f32() * 0.04,
                    heat: base_heat * random_range(&mut rng, 0.6, 0.9),
                    buoyancy: random_range(&mut rng, 0.55, 0.80),
                },
                // Plate 1 — second continental / transitional block
                PlateVector {
                    speed: base_speed * random_range(&mut rng, 0.5, 1.1),
                    omega_x: (rng.next_f32() - 0.5) * 0.04,
                    omega_y: (rng.next_f32() - 0.5) * 0.04,
                    omega_z: -(0.01 + rng.next_f32() * 0.04),
                    heat: base_heat * random_range(&mut rng, 0.7, 1.0),
                    buoyancy: random_range(&mut rng, 0.20, 0.55),
                },
                // Plate 2 — oceanic plate surrounding the island
                PlateVector {
                    speed: base_speed * random_range(&mut rng, 1.0, 1.8),
                    omega_x: (rng.next_f32() - 0.5) * 0.06,
                    omega_y: (rng.next_f32() - 0.5) * 0.06,
                    omega_z: (rng.next_f32() - 0.5) * 0.02,
                    heat: base_heat * random_range(&mut rng, 1.1, 1.6),
                    buoyancy: random_range(&mut rng, -0.70, -0.30),
                },
            ];

            // --- Fault lines (1 or 2) in grid space --------------------------
            // Primary fault: angled NNW-SSE + seed randomisation.
            // Each fault is parameterised as a line ax + by = c in pixel coords.
            let angle1 = std::f32::consts::FRAC_PI_4
                + (rng.next_f32() - 0.5) * std::f32::consts::FRAC_PI_4;
            let cx1 = w as f32 * random_range(&mut rng, 0.35, 0.65);
            let cy1 = h as f32 * 0.5;
            let (fa1, fb1) = (angle1.sin(), -angle1.cos()); // normal
            let fc1 = fa1 * cx1 + fb1 * cy1;

            // Secondary fault (for plate 2 assignment), further towards the edge.
            let angle2 = angle1 + std::f32::consts::FRAC_PI_6 * (1.0 + rng.next_f32());
            let cx2 = w as f32 * random_range(&mut rng, 0.15, 0.38);
            let cy2 = h as f32 * 0.5;
            let (fa2, fb2) = (angle2.sin(), -angle2.cos());
            let fc2 = fa2 * cx2 + fb2 * cy2;

            let boundary_width_px = (w.min(h) as f32 * 0.04).max(3.0);

            // --- Assign plate_field and boundary data -------------------------
            let mut plate_field = vec![0_i16; size];
            let mut boundary_types = vec![0_i8; size];
            let mut boundary_normal_x = vec![0.0_f32; size];
            let mut boundary_normal_y = vec![0.0_f32; size];
            let mut boundary_strength = vec![0.0_f32; size];

            for y in 0..h {
                for x in 0..w {
                    let i = grid.index(x, y);
                    let xf = x as f32 + 0.5;
                    let yf = y as f32 + 0.5;

                    // Signed distance from each fault line (positive = one side, negative = other)
                    let d1 = fa1 * xf + fb1 * yf - fc1;
                    let d2 = fa2 * xf + fb2 * yf - fc2;

                    // Plate assignment: partition by fault signs
                    let pid: i16 = if d1 >= 0.0 && d2 >= 0.0 {
                        0 // main continental block
                    } else if d1 < 0.0 && d2 >= 0.0 {
                        1 // secondary block
                    } else {
                        2 // oceanic surroundings
                    };
                    plate_field[i] = pid;

                    // Distance to nearest fault
                    let dist1 = d1.abs();
                    let dist2 = d2.abs();
                    let min_dist = dist1.min(dist2);

                    if min_dist < boundary_width_px * 3.0 {
                        // Determine which fault is closer and set normal
                        let (nx, ny, raw_type) = if dist1 <= dist2 {
                            // Fault 1: transform + slight divergent (NNW-SSE rift shoulder)
                            (fa1, fb1, if d1 >= 0.0 { 3_i8 } else { 2_i8 })
                        } else {
                            // Fault 2: more transform character
                            (fa2, fb2, 3_i8)
                        };

                        let strength = (-min_dist / boundary_width_px).exp().clamp(0.0, 1.0);
                        if strength > 0.05 {
                            boundary_types[i] = raw_type;
                            boundary_normal_x[i] = nx;
                            boundary_normal_y[i] = ny;
                            boundary_strength[i] = strength;
                        }
                    }
                }
            }

            ComputePlatesResult {
                plate_field,
                boundary_types,
                boundary_normal_x,
                boundary_normal_y,
                boundary_strength,
                plate_vectors,
            }
        }

        IslandType::Arc => {
            // ------------------------------------------------------------------
            // Island arc: 2 plates, convergent boundary curved through center.
            // Plate 0 = overriding (continental/mixed), Plate 1 = subducting (oceanic).
            // ------------------------------------------------------------------
            let plate_vectors = vec![
                PlateVector {
                    speed: params.plate_speed_cm_per_year * random_range(&mut rng, 0.8, 1.2),
                    omega_x: (rng.next_f32() - 0.5) * 0.03,
                    omega_y: (rng.next_f32() - 0.5) * 0.03,
                    omega_z: 0.015,
                    heat: params.mantle_heat * 0.9,
                    buoyancy: random_range(&mut rng, 0.1, 0.5),
                },
                PlateVector {
                    speed: params.plate_speed_cm_per_year * random_range(&mut rng, 1.2, 2.0),
                    omega_x: (rng.next_f32() - 0.5) * 0.05,
                    omega_y: (rng.next_f32() - 0.5) * 0.05,
                    omega_z: -0.02,
                    heat: params.mantle_heat * 1.4,
                    buoyancy: random_range(&mut rng, -0.7, -0.3),
                },
            ];

            // Arc axis: sinusoidal line through the grid (curved trench)
            let arc_freq = std::f32::consts::TAU / h as f32 * random_range(&mut rng, 0.5, 1.5);
            let arc_amp = w as f32 * random_range(&mut rng, 0.05, 0.15);
            let arc_cx = w as f32 * random_range(&mut rng, 0.38, 0.56);
            let boundary_width_px = (w.min(h) as f32 * 0.05).max(3.0);

            let mut plate_field = vec![0_i16; size];
            let mut boundary_types = vec![0_i8; size];
            let mut boundary_normal_x = vec![0.0_f32; size];
            let mut boundary_normal_y = vec![0.0_f32; size];
            let mut boundary_strength = vec![0.0_f32; size];

            for y in 0..h {
                for x in 0..w {
                    let i = grid.index(x, y);
                    let xf = x as f32 + 0.5;
                    let yf = y as f32 + 0.5;
                    let arc_x = arc_cx + arc_amp * (yf * arc_freq).sin();
                    let signed_dist = xf - arc_x;

                    plate_field[i] = if signed_dist >= 0.0 { 0 } else { 1 };

                    let dist = signed_dist.abs();
                    if dist < boundary_width_px * 3.0 {
                        let strength = (-dist / boundary_width_px).exp().clamp(0.0, 1.0);
                        if strength > 0.05 {
                            // Normal points towards overriding plate (positive x)
                            boundary_types[i] = 1; // convergent / subduction
                            boundary_normal_x[i] = if signed_dist >= 0.0 { -1.0 } else { 1.0 };
                            boundary_normal_y[i] = 0.0;
                            boundary_strength[i] = strength;
                        }
                    }
                }
            }

            ComputePlatesResult {
                plate_field,
                boundary_types,
                boundary_normal_x,
                boundary_normal_y,
                boundary_strength,
                plate_vectors,
            }
        }

        IslandType::Hotspot => {
            // ------------------------------------------------------------------
            // Hotspot / oceanic shield: single plate, Gaussian heat anomaly.
            // ------------------------------------------------------------------
            let plate_vectors = vec![PlateVector {
                speed: params.plate_speed_cm_per_year * random_range(&mut rng, 0.8, 1.2),
                omega_x: (rng.next_f32() - 0.5) * 0.02,
                omega_y: (rng.next_f32() - 0.5) * 0.02,
                omega_z: (rng.next_f32() - 0.5) * 0.01,
                heat: params.mantle_heat * random_range(&mut rng, 1.8, 2.8),
                buoyancy: random_range(&mut rng, -0.4, 0.0),
            }];

            // All cells belong to plate 0; no real boundaries.
            let plate_field = vec![0_i16; size];
            let boundary_types = vec![0_i8; size];
            let boundary_normal_x = vec![0.0_f32; size];
            let boundary_normal_y = vec![0.0_f32; size];
            // Gaussian boundary_strength peaks at center — used as uplift source by compute_relief.
            let cx = w as f32 * 0.5;
            let cy = h as f32 * 0.5;
            let sigma2 = (w.min(h) as f32 * 0.25).powi(2);
            let mut boundary_strength = vec![0.0_f32; size];
            for y in 0..h {
                for x in 0..w {
                    let i = grid.index(x, y);
                    let dx = x as f32 + 0.5 - cx;
                    let dy = y as f32 + 0.5 - cy;
                    boundary_strength[i] = (-(dx * dx + dy * dy) / sigma2).exp();
                }
            }

            ComputePlatesResult {
                plate_field,
                boundary_types,
                boundary_normal_x,
                boundary_normal_y,
                boundary_strength,
                plate_vectors,
            }
        }

        IslandType::Rift => {
            // ------------------------------------------------------------------
            // Rift shoulder: 2 plates, divergent boundary along one side.
            // Asymmetric uplift: one side is the rift shoulder, other subsides.
            // ------------------------------------------------------------------
            let plate_vectors = vec![
                PlateVector {
                    speed: params.plate_speed_cm_per_year * random_range(&mut rng, 0.6, 1.0),
                    omega_x: 0.0,
                    omega_y: 0.0,
                    omega_z: 0.015,
                    heat: params.mantle_heat * random_range(&mut rng, 1.0, 1.3),
                    buoyancy: random_range(&mut rng, 0.3, 0.65),
                },
                PlateVector {
                    speed: params.plate_speed_cm_per_year * random_range(&mut rng, 0.6, 1.0),
                    omega_x: 0.0,
                    omega_y: 0.0,
                    omega_z: -0.015,
                    heat: params.mantle_heat * random_range(&mut rng, 0.9, 1.2),
                    buoyancy: random_range(&mut rng, -0.1, 0.4),
                },
            ];

            // Rift line: roughly vertical with slight tilt, near one side of grid
            let rift_x = w as f32 * random_range(&mut rng, 0.25, 0.42);
            let rift_tilt = (rng.next_f32() - 0.5) * 0.3; // slight angle
            let boundary_width_px = (w as f32 * 0.06).max(3.0);

            let mut plate_field = vec![0_i16; size];
            let mut boundary_types = vec![0_i8; size];
            let mut boundary_normal_x = vec![0.0_f32; size];
            let mut boundary_normal_y = vec![0.0_f32; size];
            let mut boundary_strength = vec![0.0_f32; size];

            for y in 0..h {
                for x in 0..w {
                    let i = grid.index(x, y);
                    let xf = x as f32 + 0.5;
                    let yf = y as f32 + 0.5;
                    let local_rift_x = rift_x + rift_tilt * (yf - h as f32 * 0.5);
                    let signed_dist = xf - local_rift_x;

                    plate_field[i] = if signed_dist >= 0.0 { 0 } else { 1 };

                    let dist = signed_dist.abs();
                    if dist < boundary_width_px * 3.0 {
                        let strength = (-dist / boundary_width_px).exp().clamp(0.0, 1.0);
                        if strength > 0.05 {
                            boundary_types[i] = 2; // divergent
                            boundary_normal_x[i] = if signed_dist >= 0.0 { -1.0 } else { 1.0 };
                            boundary_normal_y[i] = rift_tilt;
                            boundary_strength[i] = strength;
                        }
                    }
                }
            }

            ComputePlatesResult {
                plate_field,
                boundary_types,
                boundary_normal_x,
                boundary_normal_y,
                boundary_strength,
                plate_vectors,
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Island envelope mask: forces ocean outside deformed ellipse boundary
// ---------------------------------------------------------------------------

/// Enforce a deformed-ellipse island boundary after compute_relief.
/// Shape the island by **sea-level cutoff**, not by geometric masking.
///
/// 1. Edge fade: smoothly push cells near grid boundaries to deep ocean
///    so the island never touches the frame.  Noise on the fade margin
///    prevents a rectangular look.
/// 2. Find the height percentile that gives `target_land_frac` of cells
///    above sea level, shift the whole field so that height == 0 at shore.
///
/// The coastline emerges from the tectonic relief itself: faults create
/// bays, ridges create peninsulas.  No geometric shape is imposed.
fn apply_island_sea_level(
    relief: &mut [f32],
    grid: &GridConfig,
    target_land_frac: f32,
    seed: u32,
) {
    let w = grid.width as f32;
    let h = grid.height as f32;

    // --- Step 1: find sea level for target land fraction --------------------
    let size = grid.size;
    let mut sorted: Vec<f32> = relief.to_vec();
    sorted.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let ocean_frac = (1.0 - target_land_frac).clamp(0.3, 0.95);
    let cut_idx = ((size as f32 * ocean_frac) as usize).min(size - 1);
    let sea_level = sorted[cut_idx];

    for val in relief.iter_mut() {
        *val -= sea_level;
    }

    // --- Step 2: edge fade AFTER cutoff — guarantees ocean at grid borders --
    let fade_base = 0.12_f32; // 12 % of each edge
    let fade_seed = hash_u32(seed ^ 0xFADE_B0CE);

    for y in 0..grid.height {
        for x in 0..grid.width {
            let i = grid.index(x, y);
            let fx = (x as f32 + 0.5) / w;
            let fy = (y as f32 + 0.5) / h;

            let edge = fx.min(1.0 - fx).min(fy).min(1.0 - fy);
            let noise = value_noise3(fx * 5.5, fy * 5.5, 0.31, fade_seed) * 0.04;
            let margin = (fade_base + noise).max(0.03);

            if edge < margin {
                let t = (edge / margin).clamp(0.0, 1.0);
                let s = t * t * (3.0 - 2.0 * t); // smoothstep
                // Force ocean at grid borders: positive → lerp toward -500
                relief[i] = relief[i] * s - 500.0 * (1.0 - s);
            }
        }
    }
}

/// After sea-level cutoff the terrain may contain many disconnected land
/// fragments.  Keep only the single largest connected component (4-connected
/// flood fill on land cells, h > 0) and sink everything else to shallow
/// ocean.  Tiny offshore islets (< 0.5 % of the main island) are also sunk.
fn keep_largest_island(relief: &mut [f32], grid: &GridConfig) {
    let size = grid.size;
    let mut comp_id = vec![-1_i32; size];
    let mut comp_sizes: Vec<usize> = Vec::new();
    let mut queue: Vec<usize> = Vec::with_capacity(size / 4);

    for start in 0..size {
        if relief[start] <= 0.0 || comp_id[start] >= 0 {
            continue;
        }
        let cid = comp_sizes.len() as i32;
        let mut count = 0_usize;
        queue.clear();
        queue.push(start);
        comp_id[start] = cid;

        while let Some(idx) = queue.pop() {
            count += 1;
            let x = (idx % grid.width) as i32;
            let y = (idx / grid.width) as i32;
            for (dx, dy) in [(-1_i32, 0_i32), (1, 0), (0, -1), (0, 1)] {
                let nx = x + dx;
                let ny = y + dy;
                if nx < 0 || nx >= grid.width as i32 || ny < 0 || ny >= grid.height as i32 {
                    continue;
                }
                let j = ny as usize * grid.width + nx as usize;
                if relief[j] > 0.0 && comp_id[j] < 0 {
                    comp_id[j] = cid;
                    queue.push(j);
                }
            }
        }
        comp_sizes.push(count);
    }

    if comp_sizes.is_empty() {
        return;
    }

    let largest = comp_sizes
        .iter()
        .enumerate()
        .max_by_key(|(_, &s)| s)
        .map(|(i, _)| i)
        .unwrap_or(0);
    // Sink ALL components except the largest one.  This guarantees a single
    // contiguous island.  Small offshore rocks are noise, not geology.
    for i in 0..size {
        if relief[i] > 0.0 {
            let c = comp_id[i] as usize;
            if c != largest {
                relief[i] = -20.0; // shallow ocean
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Phase WIRE: run_island_scope — unified physical pipeline for island scope
// ---------------------------------------------------------------------------

fn run_island_scope(
    cfg: &SimulationConfig,
    recomputed_layers: Vec<String>,
    detail: DetailProfile,
    progress: &mut ProgressTap<'_>,
) -> WasmSimulationResult {
    let island_type = parse_island_type(cfg.island_type.as_deref());
    let island_scale_km = cfg.island_scale_km.unwrap_or(400.0).clamp(50.0, 2000.0);

    let width = ISLAND_WIDTH;
    let height = ISLAND_HEIGHT;
    let km_per_cell = island_scale_km / width as f32;
    let dx_m = km_per_cell * 1000.0;

    // 1. Grid abstraction (flat, clamped edges)
    let grid = GridConfig::island(width, height, km_per_cell);
    // 2. Cell cache for noise functions
    let cell_cache = CellCache::for_island(&grid, 0.0, 0.5);

    // 3. Synthetic tectonic plate context
    progress.emit(2.0);
    let plates = generate_island_plate_context(cfg.seed, island_type, &grid, &cfg.tectonics);

    // 4. Tectonic relief via the unified compute_relief pipeline
    //    (coastal/ocean processing is skipped for non-spherical grids)
    let relief_result = compute_relief(
        &cfg.planet,
        &cfg.tectonics,
        &plates,
        cfg.seed,
        &grid,
        &cell_cache,
        detail,
        progress,
        5.0,
        45.0,
    );
    let mut relief = relief_result.relief;

    // 4b. Sea-level cutoff: the coastline emerges from the terrain itself.
    //     Edge fade prevents land touching the grid frame.
    // Target ~25 % land on the grid.  keep_largest_island (run twice: before
    // and after stream power) enforces a single contiguous island, so a higher
    // fraction just makes the island larger, not more fragmented.
    let target_land = 0.25_f32;
    apply_island_sea_level(&mut relief, &grid, target_land, cfg.seed);

    // 4c. Remove tiny fragments — keep one main island + large satellites.
    keep_largest_island(&mut relief, &grid);

    // 5. Per-cell uplift field (m/yr) based on tectonic type.
    //    Zero uplift on ocean cells so stream power doesn't resurrect sunken fragments.
    let uplift = {
        let mut u = generate_island_uplift_field(island_type, &plates, &grid, &cfg.tectonics, cfg.seed);
        for i in 0..grid.size {
            if relief[i] <= 0.0 { u[i] = 0.0; }
        }
        u
    };

    // 6. Braun-Willett O(n) stream power erosion
    //    K_eff: erodibility [yr^-1] — calibrated so mountains retain 50-70% height
    //    after n_steps (avoids over-planation from aggressive initial K values).
    //    Rationale: factor = K * dt * A^0.5 / dx ≈ K * dt at headwater.
    //    Targeting factor ≈ 0.12 per step → ~10% loss → 0.9^n over n steps.
    let k_base: f32 = match island_type {
        IslandType::Continental => 1.2e-6,
        IslandType::Arc         => 2.0e-6,
        IslandType::Hotspot     => 1.6e-6,
        IslandType::Rift        => 1.4e-6,
    };
    let k_eff = vec![k_base; grid.size];
    let dt_yr = 100_000.0_f32; // 100 kyr per timestep
    let n_steps = 2 + detail.erosion_rounds * 2; // 2..8 steps
    stream_power_evolve(
        &mut relief,
        &uplift,
        &k_eff,
        dt_yr,
        dx_m,
        0.5,  // m: area exponent
        1.0,  // n: slope exponent
        0.08, // kappa: hillslope diffusivity m²/yr — suppresses D8 channel streaking
        n_steps,
        &grid,
        progress,
        50.0,
        15.0,
    );

    // 7. Slope (pre-carve pass)
    let (slope_pre, _, _) = compute_slope_grid(&relief, width, height, progress, 65.0, 3.0);

    // 8. Hydrology (pre-carve pass)
    let (_, flow_accum_pre, river_map_pre, _) =
        compute_hydrology_grid(&relief, &slope_pre, width, height, detail, progress, 68.0, 5.0);

    // 9. Carve fluvial valleys
    carve_fluvial_valleys_grid(
        &mut relief,
        &flow_accum_pre,
        &river_map_pre,
        width,
        height,
        detail,
        progress,
        73.0,
        5.0,
    );

    // 9b. Post-erosion cleanup: re-sink any ocean cells that got pushed above
    //     sea level by stream power uplift or fluvial carving artefacts, and
    //     re-enforce the edge fade so land never touches the grid frame.
    keep_largest_island(&mut relief, &grid);
    {
        let w = grid.width as f32;
        let h = grid.height as f32;
        let fade_base = 0.12_f32;
        let fade_seed = hash_u32(cfg.seed ^ 0xFADE_B0CE);
        for y in 0..grid.height {
            for x in 0..grid.width {
                let i = grid.index(x, y);
                let fx = (x as f32 + 0.5) / w;
                let fy = (y as f32 + 0.5) / h;
                let edge = fx.min(1.0 - fx).min(fy).min(1.0 - fy);
                let noise = value_noise3(fx * 5.5, fy * 5.5, 0.31, fade_seed) * 0.04;
                let margin = (fade_base + noise).max(0.03);
                if edge < margin && relief[i] > 0.0 {
                    let t = (edge / margin).clamp(0.0, 1.0);
                    let s = t * t * (3.0 - 2.0 * t);
                    relief[i] = relief[i] * s - 500.0 * (1.0 - s);
                }
            }
        }
    }

    // 10. Final slope
    let (slope_map, min_slope, max_slope) =
        compute_slope_grid(&relief, width, height, progress, 78.0, 4.0);

    // 11. Final hydrology
    let (flow_direction, flow_accumulation, river_map, lake_map) =
        compute_hydrology_grid(&relief, &slope_map, width, height, detail, progress, 82.0, 5.0);

    // 12. Climate — use a smoothed height copy to suppress narrow D8-channel
    //     streaks from bleeding into temperature/precipitation/biome bands.
    let relief_for_climate = {
        let mut smoothed = relief.clone();
        let mut scratch = vec![0.0_f32; grid.size];
        for _ in 0..3 {
            for y in 0..height {
                for x in 0..width {
                    let i = grid.index(x, y);
                    let mut sum = smoothed[i] * 4.0;
                    let mut wt = 4.0_f32;
                    for (dx, dy) in [(-1_i32, 0_i32), (1, 0), (0, -1), (0, 1)] {
                        let nx = (x as i32 + dx).clamp(0, width as i32 - 1) as usize;
                        let ny = (y as i32 + dy).clamp(0, height as i32 - 1) as usize;
                        sum += smoothed[grid.index(nx, ny)];
                        wt += 1.0;
                    }
                    scratch[i] = sum / wt;
                }
            }
            smoothed.copy_from_slice(&scratch);
        }
        smoothed
    };

    let coastal_exposure = compute_coastal_exposure_grid(&relief, width, height);
    let (
        temperature_map,
        precipitation_map,
        min_temperature,
        max_temperature,
        min_precipitation,
        max_precipitation,
    ) = compute_climate_grid(
        &cfg.planet,
        &relief_for_climate,
        &slope_map,
        &river_map,
        &flow_accumulation,
        &coastal_exposure,
        width,
        height,
        cfg.seed,
        progress,
        87.0,
        8.0,
    );

    // 13. Biomes — blur temperature and precipitation before classification
    //     to suppress D8/carve channel streaks bleeding into biome bands.
    let smooth_map = |src: &[f32]| -> Vec<f32> {
        let mut out = src.to_vec();
        let mut scratch = vec![0.0_f32; grid.size];
        for _ in 0..5 {
            for y in 0..height {
                for x in 0..width {
                    let i = grid.index(x, y);
                    let mut sum = out[i] * 4.0;
                    let mut wt = 4.0_f32;
                    for (dx, dy) in [(-1_i32, 0_i32), (1, 0), (0, -1), (0, 1)] {
                        let nx = (x as i32 + dx).clamp(0, width as i32 - 1) as usize;
                        let ny = (y as i32 + dy).clamp(0, height as i32 - 1) as usize;
                        sum += out[grid.index(nx, ny)];
                        wt += 1.0;
                    }
                    scratch[i] = sum / wt;
                }
            }
            out.copy_from_slice(&scratch);
        }
        out
    };
    let temp_smooth = smooth_map(&temperature_map);
    let precip_smooth = smooth_map(&precipitation_map);
    let biome_map = compute_biomes_grid(&temp_smooth, &precip_smooth, &relief_for_climate);
    progress.emit(96.0);

    // 14. Settlement
    let settlement_map = compute_settlement_grid(
        &biome_map,
        &relief,
        &temperature_map,
        &precipitation_map,
        &river_map,
        &coastal_exposure,
    );
    progress.emit(99.0);

    let (min_height, max_height) = min_max(&relief);
    let ocean_cells = relief.iter().filter(|&&h| h < 0.0).count() as f32;
    let ocean_percent = 100.0 * ocean_cells / grid.size as f32;

    WasmSimulationResult {
        width: width as u32,
        height: height as u32,
        seed: cfg.seed,
        sea_level: 0.0,
        radius_km: cfg.planet.radius_km,
        ocean_percent,
        recomputed_layers,
        plates: plates.plate_field,
        boundary_types: plates.boundary_types,
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

    if scope == GenerationScope::Island {
        let result = run_island_scope(&cfg, recomputed_layers, detail, &mut progress);
        // Keep progress <100 until JS wrapper and worker finish marshalling + posting result.
        progress.emit(99.9);
        return Ok(result);
    }

    let cache = world_cache();

    let plates_layer = compute_plates(
        &cfg.planet,
        &cfg.tectonics,
        detail,
        cfg.seed,
        cache,
        &mut progress,
        2.0,
        22.0,
    );
    let planet_grid = GridConfig::planet();
    let planet_cell_cache = CellCache::for_planet(&planet_grid);
    let relief_raw = compute_relief(
        &cfg.planet,
        &cfg.tectonics,
        &plates_layer,
        cfg.seed,
        &planet_grid,
        &planet_cell_cache,
        detail,
        &mut progress,
        24.0,
        50.0,
    );
    progress.emit(74.0);
    let (event_relief, aerosol) = apply_events(&cfg.planet, &relief_raw.relief, &cfg.events, cache);
    progress.emit(78.0);

    let (slope_map, min_slope, max_slope) = compute_slope(&event_relief, &mut progress, 78.0, 8.0);
    let (flow_direction, flow_accumulation, river_map, lake_map) =
        compute_hydrology(&event_relief, &slope_map, &mut progress, 86.0, 7.0);

    let (
        temperature_map,
        precipitation_map,
        min_temperature,
        max_temperature,
        min_precipitation,
        max_precipitation,
    ) = compute_climate(
        &cfg.planet,
        &event_relief,
        &slope_map,
        &river_map,
        &flow_accumulation,
        aerosol,
        cache,
        &mut progress,
        93.0,
        5.0,
    );
    progress.emit(98.0);

    let biome_map = compute_biomes(&temperature_map, &precipitation_map, &event_relief);
    progress.emit(99.0);
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
