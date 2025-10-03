use std::ops::Neg;
use std::sync::atomic::Ordering;
use rayon::iter::{IndexedParallelIterator, ParallelIterator};
use rayon::iter::IntoParallelRefIterator;
use crate::{Particle, FORCES};

pub fn process_forces(particles: &[Particle]) {
    particles.par_iter().enumerate().for_each(|(p1_idx, p1)| {
        particles[p1_idx+1..particles.len()].par_iter().enumerate().for_each(|(p2_idx, p2)| {
            process_pair(p1_idx, p2_idx, p1, p2);
        });
    });
}

pub fn process_pair(p1_idx: usize, p2_idx: usize, p1: &Particle, p2: &Particle) {
    let force = p1.get_influence(p2);
    add_force(p1_idx, (force.x, force.y, force.z));
    add_force(p2_idx, (force.x.neg(), force.y.neg(), force.z.neg()))
}

fn add_force(idx: usize, force: (f32, f32, f32)) {
    let current = &FORCES[idx];
    current.0.fetch_add(force.0, Ordering::Relaxed);
    current.1.fetch_add(force.1, Ordering::Relaxed);
    current.2.fetch_add(force.2, Ordering::Relaxed);
}