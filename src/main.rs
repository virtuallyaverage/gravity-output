use std::sync::{Arc, LazyLock, RwLock};
use std::io::Write;
use rand::prelude::*;
use flate2::write::GzEncoder;
use flate2::Compression;
use bytemuck;
use glam::Vec3;

static PARTICLES: LazyLock<RwLock<Vec<Particle>>> = LazyLock::new(|| {
    let particles = init_particles();
    println!("Done with particle init");
    RwLock::new(particles)
});
static ACC_FORCE: LazyLock<RwLock<Vec<Vec3>>> = LazyLock::new(|| {
    let acc_forces = [Vec3::ZERO; NUM_PARTICLES];
    println!("Done with forces init");
    RwLock::new(acc_forces.to_vec())
});


pub const NUM_PARTICLES: usize = 6000;
pub const FRAMES_TOTAL: usize = 600;
pub const FRAMES_PER_FILE: usize = 50;
pub const DT: f32 = 1.0 / 60.0;
/// radius the particles will be spawned within
const ARENA: f32 = 3000.0;
pub const G_CONST: f32 = 6.67430e-11;

fn main() {
    let mut frame_list = Box::new([[Vec3::ZERO; NUM_PARTICLES]; FRAMES_PER_FILE]);

    for batch in 0..FRAMES_TOTAL {
        process_frame_group(&mut frame_list, batch.clone());
        println!("Done with batch {}, frames: {}",  batch.clone(), batch * FRAMES_PER_FILE);
    }
    
    println!("Hello, world!");
}

/// process a single files worth of frames.
fn process_frame_group(frame_list: &mut Box<[[Vec3; NUM_PARTICLES]; FRAMES_PER_FILE]>, batch_num: usize) {
    for (local_num, frame) in frame_list.iter_mut().enumerate() {
        // accumulate force
        for idx in 0..NUM_PARTICLES - 1 {
            let force = one_particle(idx);
            let mut forces = ACC_FORCE.write().unwrap();
            let current = forces.get_mut(idx).unwrap();
            *current = force;
        }

        // propagate force and store result
        let mut particles = PARTICLES.write().unwrap();
        let forces = ACC_FORCE.read().unwrap();
        for (idx, out_pos) in frame.iter_mut().enumerate() {
            let part = particles.get_mut(idx).unwrap();
            let force = forces.get(idx).unwrap();
            part.tick(force);
            *out_pos = part.pos;
        }
    }

    write_frame_group(frame_list, &batch_num);
}

// Write batch of frames
fn write_frame_group(frame_list: &mut [[Vec3; NUM_PARTICLES]; FRAMES_PER_FILE], batch_num: &usize) {
    let filename = format!("output/batch_{:04}.bin.gz", batch_num);
    let file = std::fs::File::create(filename).unwrap();
    let mut encoder = GzEncoder::new(file, Compression::fast());
    
    // header
    encoder.write_all(&FRAMES_PER_FILE.to_le_bytes()).unwrap();
    encoder.write_all(&NUM_PARTICLES.to_le_bytes()).unwrap();

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
    let mut force = Vec3::ZERO;
    let particles = PARTICLES.read().unwrap();
    let particle = particles.get(idx).unwrap();

    for part in particles.iter() {
        force += particle.get_influence(part);
    }

    return force;
}

/// handles initial distribution and whatnot
fn init_particles() -> Vec<Particle> {
    let mut rng = rand::rng();
    
    (0..NUM_PARTICLES)
        .map(|_| {
            // Random spherical distribution
            let r = ARENA * rng.random::<f32>().powf(1.0/3.0); // Uniform volume distribution
            let theta = rng.random::<f32>() * 2.0 * std::f32::consts::PI;
            let phi = (rng.random::<f32>() * 2.0 - 1.0).acos();
            
            let pos = Vec3::new(
                r * phi.sin() * theta.cos(),
                r * phi.sin() * theta.sin(),
                r * phi.cos(),
            );
            
            // Orbital velocity perpendicular to position (creates rotation)
            let speed = (50.0 / r.sqrt()).min(100.0); // Faster near center
            let tangent = Vec3::new(-pos.y, pos.x, 0.0).normalize_or_zero();
            let vel = tangent * speed;
            
            // Random mass distribution (e.g., 1.0 to 10.0 units)
            let mass = rng.random::<f32>() * 9.0 + 1.0;
            
            Particle::new(10000.0, pos, vel, Vec3::ZERO)
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
        let force_over_r3 = G_CONST * self.mass * other.mass / (r_sq * r_sq.sqrt());

        r_vec * force_over_r3
    }

    /// Propogate force accumulated over a tick into movement.
    pub fn tick(&mut self, force: &Vec3) {
        let new_acc = force / self.mass;
        self.pos += self.vel * DT + 0.5 * self.acc * DT * DT;
        self.vel += 0.5 * (self.acc + new_acc) * DT;
        self.acc = new_acc;
    }
}
