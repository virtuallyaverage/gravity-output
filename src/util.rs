use serde::{Serialize,  Deserialize};
use std::path::PathBuf;
use std::env;
use glam::Vec3;

use super::{Particle, SETTINGS};
use rand::prelude::*;

#[derive(Serialize, Deserialize, Clone)]
pub struct Settings {
    pub num_particles: usize,
    pub frames_total: usize,
    pub frames_per_file: usize,
    pub dt: f32,
    pub arena: f32,
    pub g_const: f32,
    pub mass: f32,
    pub init_vel: f32,
    pub out_path: PathBuf,
}

impl Default for Settings {
    fn default() -> Self {
        Settings {
            num_particles: 12000,
            frames_total: 10000,
            frames_per_file: 100,
            dt: 1.0 / 180.0,
            arena: 100.0,
            g_const: 0.01,
            mass: 1000.,
            init_vel: 4.5,
            out_path: PathBuf::from(""), // initialized properly in load_settings
        }
    }
}

/// handles initial distribution and velocity
pub fn init_particles() -> Vec<Particle> {
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

pub fn load_settings() -> Settings {
    let mut settings = match std::fs::read_to_string("settings.json") {
        Ok(content) => match serde_json::from_str::<Settings>(&content) {
            Ok(settings) => {
                println!("Loaded settings from settings.json");
                settings
            }
            Err(e) => {
                println!("Error parsing settings.json: {}, using defaults", e);
                create_default_settings()
            }
        },
        Err(_) => {
            println!("settings.json not found, creating with default values");
            create_default_settings()
        }
    };

    // resolve path flag
    let args: Vec<String> = env::args().collect();
    let output_path = args
        .windows(2)
        .find(|pair| pair[0] == "--output")
        .map(|pair| PathBuf::from(&pair[1]))
        .unwrap_or_else(|| PathBuf::from("output"));

    // resolve to full path
    let output_path = if output_path.is_absolute() {
        output_path
    } else {
        env::current_dir()
            .unwrap_or_else(|_| PathBuf::from("."))
            .join(output_path)
    };

    settings.out_path = output_path;
    std::fs::create_dir_all(settings.out_path.clone()).unwrap();
    println!("{:?}", settings.out_path);
    return settings;
}

fn create_default_settings() -> Settings {
    let settings = Settings::default();
    match serde_json::to_string_pretty(&settings) {
        Ok(json) => {
            if let Err(e) = std::fs::write("settings.json", json) {
                println!("Warning: Could not create settings.json: {}", e);
            } else {
                println!("Created settings.json with default values");
            }
        }
        Err(e) => println!("Warning: Could not serialize settings: {}", e),
    }
    settings
}
