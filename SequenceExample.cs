using System;
using System.IO;
using System.Numerics;

class Program
{
    static void Main(string[] args)
    {
        // Example usage of the GravitySequenceReader - seamless reading across all batch files
        string folderPath = "output";
        
        if (!Directory.Exists(folderPath))
        {
            Console.WriteLine($"Folder not found: {folderPath}");
            Console.WriteLine("Make sure to run the Rust program first to generate the data files.");
            return;
        }

        try
        {
            using (var sequenceReader = new GravitySequenceReader(folderPath))
            {
                Console.WriteLine($"Sequence loaded successfully!");
                Console.WriteLine($"Found {sequenceReader.TotalBatches} batch files");
                Console.WriteLine($"Estimated total frames: {sequenceReader.EstimateTotalFrames()}");
                Console.WriteLine($"Particles per frame: {sequenceReader.NumParticles}");
                Console.WriteLine();

                // Read and process frames seamlessly across all batch files
                int frameCount = 0;
                while (sequenceReader.HasMoreFrames)
                {
                    Vector3[] frame = sequenceReader.GetNextFrame();
                    
                    if (frame == null)
                        break;

                    frameCount++;
                    
                    // Process the frame data
                    ProcessFrame(frame, frameCount);
                    
                    // Show progress every 25 frames
                    if (frameCount % 25 == 0)
                    {
                        var info = sequenceReader.GetSequenceInfo();
                        Console.WriteLine($"Progress: {info}");
                    }
                }

                Console.WriteLine($"\n=== Final Results ===");
                Console.WriteLine($"Successfully processed {frameCount} frames across all batch files!");
                
                var finalInfo = sequenceReader.GetSequenceInfo();
                Console.WriteLine(finalInfo.GetDetailedString());
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error: {ex.Message}");
        }
    }

    static void ProcessFrame(Vector3[] particles, int frameNumber)
    {
        // Example processing: calculate center of mass and system energy
        if (particles.Length == 0) return;

        Vector3 centerOfMass = Vector3.Zero;
        float minDistance = float.MaxValue;
        float maxDistance = 0f;
        float totalSpread = 0f;

        // Calculate center of mass
        foreach (var particle in particles)
        {
            centerOfMass += particle;
        }
        centerOfMass /= particles.Length;

        // Calculate distances and spread
        foreach (var particle in particles)
        {
            float distance = Vector3.Distance(particle, centerOfMass);
            minDistance = Math.Min(minDistance, distance);
            maxDistance = Math.Max(maxDistance, distance);
            totalSpread += distance;
        }

        float averageDistance = totalSpread / particles.Length;

        // Print statistics for first few frames and periodically
        if (frameNumber <= 5 || frameNumber % 50 == 0)
        {
            Console.WriteLine($"Frame {frameNumber:D4}: " +
                            $"CoM=({centerOfMass.X:F1}, {centerOfMass.Y:F1}, {centerOfMass.Z:F1}) " +
                            $"Avg={averageDistance:F1} Range=[{minDistance:F1}, {maxDistance:F1}]");
        }
    }

    // Example demonstrating frame skipping and batch information
    static void AdvancedUsageExample()
    {
        string folderPath = "output";
        
        using (var reader = new GravitySequenceReader(folderPath))
        {
            Console.WriteLine("=== Advanced Usage Example ===");
            
            // Show all available batch files
            var batchFiles = reader.GetBatchFileNames();
            Console.WriteLine($"Available batch files: {string.Join(", ", batchFiles)}");
            
            // Read first few frames
            Console.WriteLine("\nReading first 3 frames:");
            for (int i = 0; i < 3; i++)
            {
                var frame = reader.GetNextFrame();
                if (frame != null)
                {
                    Console.WriteLine($"Frame {i + 1}: {frame.Length} particles, first particle at {frame[0]}");
                }
            }
            
            // Skip ahead 47 frames (to complete the first batch file)
            Console.WriteLine($"\nSkipping 47 frames...");
            reader.SkipFrames(47);
            
            var info = reader.GetSequenceInfo();
            Console.WriteLine($"Status after skip: {info}");
            
            // Read next frame (should be first frame of second batch)
            var nextFrame = reader.GetNextFrame();
            if (nextFrame != null)
            {
                Console.WriteLine($"Next frame (should be from batch 2): first particle at {nextFrame[0]}");
            }
            
            // Show detailed status
            Console.WriteLine($"\nDetailed Status:");
            Console.WriteLine(reader.GetSequenceInfo().GetDetailedString());
        }
    }

    // Example for reading specific portions of the sequence
    static void TargetedReadingExample()
    {
        string folderPath = "output";
        
        using (var reader = new GravitySequenceReader(folderPath))
        {
            Console.WriteLine("=== Targeted Reading Example ===");
            
            // Skip to middle of sequence
            int totalFrames = reader.EstimateTotalFrames();
            int targetFrame = totalFrames / 2;
            
            Console.WriteLine($"Skipping to frame {targetFrame} (middle of sequence)...");
            reader.SkipFrames(targetFrame - 1);
            
            // Read 10 frames from the middle
            Console.WriteLine("Reading 10 frames from middle:");
            for (int i = 0; i < 10 && reader.HasMoreFrames; i++)
            {
                var frame = reader.GetNextFrame();
                if (frame != null)
                {
                    var info = reader.GetSequenceInfo();
                    Console.WriteLine($"Middle frame {i + 1}: {info.CurrentBatchName} - Frame {info.CurrentFrameInBatch}");
                }
            }
        }
    }
}