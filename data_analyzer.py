"""
Simple debugging for the bin format
"""

import gzip
import struct
import os

def read_gravity_batch(filepath):
    """Read a single batch file and return frame data"""
    frames = []
    
    print(f"Reading: {filepath}")
    
    with gzip.open(filepath, 'rb') as f:
        # Read header
        header_data = f.read(8)
        if len(header_data) < 8:
            print(f"  Error: Could not read header, got {len(header_data)} bytes")
            return frames
        
        frames_per_file = struct.unpack('<I', header_data[:4])[0]
        num_particles = struct.unpack('<I', header_data[4:8])[0]
        
        print(f"  Frames per file: {frames_per_file}")
        print(f"  Particles per frame: {num_particles}")
        
        remaining_data = f.read()
        expected_bytes = frames_per_file * num_particles * 12  # 3 f32 values per particle
        print(f"  Expected data bytes: {expected_bytes}")
        print(f"  Actual data bytes: {len(remaining_data)}")
        
        if len(remaining_data) != expected_bytes:
            print(f"  Warning: Data size mismatch!")
            # Try to determine actual particle count
            if frames_per_file > 0:
                actual_particles = len(remaining_data) // (frames_per_file * 12)
                print(f"  Calculated particles per frame: {actual_particles}")
                num_particles = actual_particles
        
        # Reset file position and skip header again
        f.seek(8)
        
        # Read frame data
        for frame_idx in range(frames_per_file):
            frame_positions = []
            for particle_idx in range(num_particles):
                data = f.read(12)
                if len(data) < 12:
                    print(f"  Warning: Incomplete data at frame {frame_idx}, particle {particle_idx}")
                    break
                x, y, z = struct.unpack('<fff', data)
                frame_positions.append((x, y, z))
            frames.append(frame_positions)
            
            if frame_idx == 0 and len(frame_positions) > 0:  # Show sample data from first frame
                print(f"  Sample positions from frame 0:")
                for i in range(min(5, len(frame_positions))):
                    x, y, z = frame_positions[i]
                    print(f"    Particle {i}: ({x:.2f}, {y:.2f}, {z:.2f})")
    
    return frames

if __name__ == "__main__":
    data_folder = r"output"  # Relative to this script
    
    print("=== Gravity Simulation Data Analysis ===\n")
    
    # Test reading one batch
    print("\n=== Testing Batch Read ===")
    batch_0_file = os.path.join(data_folder, "batch_0000.bin.gz")
    if os.path.exists(batch_0_file):
        frames = read_gravity_batch(batch_0_file)
        print(f"Successfully read {len(frames)} frames")