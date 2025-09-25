use serde::{Serialize,  Deserialize};
use std::path::PathBuf;
use std::env;

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
