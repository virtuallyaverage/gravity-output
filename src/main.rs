use bytemuck;
use flate2::Compression;
use flate2::write::GzEncoder;
use glam::Vec3;
use rand::prelude::*;
use rayon::prelude::*;
use std::io::Write;
use std::sync::{LazyLock, RwLock};
use std::time::Instant;

mod util;
use util::{load_settings, Settings, };

static SETTINGS: LazyLock<Settings> = LazyLock::new(|| load_settings());
static PARTICLES: LazyLock<RwLock<Vec<Particle>>> = LazyLock::new(|| {
    let particles = init_particles();
    println!("Done with particle init");
    RwLock::new(particles)
});

fn main() {
    let mut frame_list: Vec<Vec<Vec3>> =
        vec![vec![Vec3::ZERO; SETTINGS.num_particles]; SETTINGS.frames_per_file];

    let num_batches = SETTINGS.frames_total / SETTINGS.frames_per_file;
    for batch in 0..num_batches {
        let time_start = Instant::now();
        process_frame_group(&mut frame_list, batch.clone());
        println!(
            "Done with batch: {}, frames: {}-{}, Seconds: {} per frame: {}",
            batch,
            batch * SETTINGS.frames_per_file,
            (batch + 1) * SETTINGS.frames_per_file - 1,
            time_start.elapsed().as_secs_f32(),
            time_start.elapsed().as_secs_f32() / SETTINGS.frames_per_file as f32
        );
    }

    println!("Finished!");
}

/// process a single files worth of frames.
fn process_frame_group(frame_list: &mut Vec<Vec<Vec3>>, batch_num: usize) {
    let mut forces = vec![Vec3::ZERO; SETTINGS.num_particles];

    for frame in frame_list.iter_mut() {
        // Parallel force accumulation
        forces
            .par_iter_mut()
            .enumerate()
            .for_each(|(idx, force_ref): (usize, &mut Vec3)| {
                // if we already computed this, take the inverse (cause gravity always pulls)
                let particles = PARTICLES.read().unwrap();
                let particle = &particles[idx]; // Direct indexing instead of get()
                let particle_pos = particle.pos;
                let particle_mass = particle.mass;

                *force_ref = Vec3::ZERO;
                for (idx2, other) in particles.iter().enumerate() {
                    if idx == idx2 { // particle doesn't affect itself
                        continue;
                    } else if idx2 < idx { // Since we itterate over 


                        *force_ref += r_vec * force_over_r3;
                    }

                    // caluclate force
                    let r_vec = other.pos - particle_pos;
                    let r_sq = r_vec.dot(r_vec).max(1e-8);
                    let force_over_r3 = SETTINGS.g_const * particle_mass * other.mass / (r_sq * r_sq.sqrt());

                    *force_ref += r_vec * force_over_r3;
                }
            });
        // propagate force and store result
        {
            let mut particles = PARTICLES.write().unwrap();
            for (idx, out_pos) in frame.iter_mut().enumerate() {
                let part = particles.get_mut(idx).unwrap();
                let force = &forces[idx];
                part.tick(force);
                *out_pos = part.pos;
            }
        }
    }

    let start = Instant::now();
    write_frame_group(frame_list, &batch_num);
    println!("Took to save: {}", start.elapsed().as_secs_f32());
}

// Write batch of frames
fn write_frame_group(frame_list: &mut Vec<Vec<Vec3>>, batch_num: &usize) {
    let filename = SETTINGS.out_path.join(format!("batch_{:04}.bin.gz", batch_num));
    let file = std::fs::File::create(filename).unwrap();
    let mut encoder = GzEncoder::new(file, Compression::fast());

    // header - convert to u32 for consistent 4-byte format
    encoder
        .write_all(&(SETTINGS.frames_per_file as u32).to_le_bytes())
        .unwrap();
    encoder
        .write_all(&(SETTINGS.num_particles as u32).to_le_bytes())
        .unwrap();

    for frame in frame_list.iter() {
        for pos in frame.iter() {
            encoder.write_all(bytemuck::bytes_of(pos)).unwrap();
        }
    }
    encoder.finish().unwrap();
}

/// Finds how all other particles interact with this one
/// Returns the resulting force vector
fn one_particle(idx: usize) -> Vec3 {
    let particles = PARTICLES.read().unwrap();
    let particle = &particles[idx]; // Direct indexing instead of get()
    let particle_pos = particle.pos;
    let particle_mass = particle.mass;

    let mut force = Vec3::ZERO;
    for (idx2, other) in particles.iter().enumerate() {
        if idx == idx2 {
            continue;
        }

        // Inline the force calculation to avoid function call overhead
        let r_vec = other.pos - particle_pos;
        let r_sq = r_vec.dot(r_vec).max(1e-8);
        let force_over_r3 = SETTINGS.g_const * particle_mass * other.mass / (r_sq * r_sq.sqrt());
        force += r_vec * force_over_r3;
    }

    force
}

/// handles initial distribution and velocity
fn init_particles() -> Vec<Particle> {
    let mut rng = rand::rng();

    (0..SETTINGS.num_particles)
        .map(|_| {
            // Random spherical distribution
            let r = SETTINGS.arena * (rng.random::<f32>().powf(1.0 / 3.0)); // Avoid center
            let theta = rng.random::<f32>() * 2.0 * std::f32::consts::PI;
            let phi = (rng.random::<f32>() * 2.0 - 1.0).acos();

            let pos = Vec3::new(
                r * phi.sin() * theta.cos(),
                r * phi.sin() * theta.sin(),
                r * phi.cos(),
            );

            // Calculate orbital velocity for a central mass system
            let central_mass = SETTINGS.num_particles as f32 * 5.0; // random numbers go brrr
            let orbital_speed = (SETTINGS.g_const * central_mass / r).sqrt() * SETTINGS.init_vel;
            let tangent = Vec3::new(-pos.y, pos.x, 0.0).normalize_or_zero();
            let vel = tangent * orbital_speed;

            Particle::new(SETTINGS.mass, pos, vel, Vec3::ZERO)
        })
        .collect()
}

pub struct Particle {
    mass: f32,
    pos: Vec3,
    vel: Vec3,
    acc: Vec3,
}

impl Particle {
    pub fn new(mass: f32, pos: Vec3, vel: Vec3, acc: Vec3) -> Particle {
        Particle {
            mass: mass,
            pos: pos,
            vel: vel,
            acc: acc,
        }
    }

    /// New with default values at zero
    pub fn new_zero() -> Particle {
        Particle {
            mass: 1.0,
            pos: Vec3::ZERO,
            vel: Vec3::ZERO,
            acc: Vec3::ZERO,
        }
    }

    /// That these two particles have on each other
    ///
    /// Returns the force vector of influence
    pub fn get_influence(&self, other: &Particle) -> Vec3 {
        const EPSILON_SQ: f32 = 1e-8; // Pre-squared softening

        let r_vec = other.pos - self.pos;
        let r_sq = r_vec.dot(r_vec).max(EPSILON_SQ);

        // Combined magnitude and direction calculation
        let force_over_r3 = SETTINGS.g_const * self.mass * other.mass / (r_sq * r_sq.sqrt());

        r_vec * force_over_r3
    }

    /// Propogate force accumulated over a tick into movement.
    pub fn tick(&mut self, force: &Vec3) {
        // Simple Euler integration (more stable for this system)
        self.acc = force / self.mass;
        self.vel += self.acc * SETTINGS.dt;
        self.pos += self.vel * SETTINGS.dt;
    }
}
