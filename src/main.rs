use bytemuck;
use flate2::Compression;
use flate2::write::GzEncoder;
use glam::Vec3;
use rayon::prelude::*;
use std::io::Write;
use std::sync::{LazyLock, RwLock};
use std::time::Instant;


mod util;
use util::{Settings, init_particles, load_settings};

static SETTINGS: LazyLock<Settings> = LazyLock::new(|| load_settings());
static PARTICLES: LazyLock<RwLock<Vec<Particle>>> = LazyLock::new(|| {
    let particles = init_particles();
    println!("Done with particle init");
    RwLock::new(particles)
});

/// Direct force accumulation - no LUT needed
fn process_frame_group(frame_list: &mut Vec<Vec<Vec3>>, batch_num: usize) {
    for frame in frame_list.iter_mut() {
        let num_particles = SETTINGS.num_particles;
        
        // Single read lock for snapshot
        let particles: Vec<Particle> = PARTICLES.read().unwrap().clone();
        
        // Thread-local force accumulation
        let forces = (0..num_particles)
            .into_par_iter()
            .fold(
                || vec![Vec3::ZERO; num_particles],
                |mut thread_forces, i| {
                    // Only compute j > i (upper triangle)
                    for j in (i + 1)..num_particles {
                        let influence = particles[i].get_influence(&particles[j]);
                        thread_forces[i] += influence;
                        thread_forces[j] -= influence;
                    }
                    thread_forces
                }
            )
            .reduce(
                || vec![Vec3::ZERO; num_particles],
                |mut acc, thread_forces| {
                    for (i, force) in thread_forces.iter().enumerate() {
                        acc[i] += *force;
                    }
                    acc
                }
            );
        
        // Single write lock for update
        {
            let mut particles_mut = PARTICLES.write().unwrap();
            for (idx, (force, out_pos)) in forces.iter().zip(frame.iter_mut()).enumerate() {
                particles_mut[idx].tick(force);
                *out_pos = particles_mut[idx].pos;
            }
        }
    }
    
    let start = Instant::now();
    write_frame_group(frame_list, &batch_num);
    println!("Took to save: {}", start.elapsed().as_secs_f32());
}

// Write batch of frames
fn write_frame_group(frame_list: &mut Vec<Vec<Vec3>>, batch_num: &usize) {
    let filename = SETTINGS
        .out_path
        .join(format!("batch_{:04}.bin.gz", batch_num));
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

#[derive(Clone)]
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

    /// Returns force that `self` experiences from other.
    ///
    /// Returns the force vector of influence
    pub fn get_influence(&self, other: &Particle) -> Vec3 {
        const EPSILON_SQ: f32 = 1e-8; // Pre-squared softening

        let r_vec = other.pos - self.pos;
        let r_sq = (r_vec).dot(r_vec).max(EPSILON_SQ);

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
