using System;
using System.IO;
using System.IO.Compression;
using System.Numerics;
using System.Linq;
using System.Collections.Generic;

public class GravitySequenceReader : IDisposable
{
    private readonly string _folderPath;
    private readonly List<string> _batchFiles;
    private GravityFrameReader _currentReader;
    private int _currentBatchIndex;
    private int _totalFramesRead;
    private bool _disposed = false;

    // Properties for tracking overall progress
    public int TotalBatches => _batchFiles.Count;
    public int CurrentBatch => _currentBatchIndex;
    public int TotalFramesRead => _totalFramesRead;
    public bool HasMoreFrames => _currentBatchIndex < _batchFiles.Count;
    public int? NumParticles { get; private set; }
    public int? FramesPerFile { get; private set; }

    public GravitySequenceReader(string folderPath)
    {
        if (!Directory.Exists(folderPath))
            throw new DirectoryNotFoundException($"Folder not found: {folderPath}");

        _folderPath = folderPath;
        
        // Find all batch files in the folder and sort them
        _batchFiles = Directory.GetFiles(folderPath, "batch_*.bin.gz")
            .OrderBy(f => f)
            .ToList();

        if (_batchFiles.Count == 0)
            throw new InvalidOperationException($"No batch files found in folder: {folderPath}");

        _currentBatchIndex = -1; // Will be set to 0 when first file is opened
        _totalFramesRead = 0;

        Console.WriteLine($"GravitySequenceReader initialized");
        Console.WriteLine($"Found {_batchFiles.Count} batch files in: {folderPath}");
        
        // Load the first file to get metadata
        LoadNextBatch();
    }

    /// <summary>
    /// Seamlessly reads the next frame from the current batch file,
    /// automatically loading the next batch file when the current one is exhausted.
    /// </summary>
    /// <returns>Array of Vector3 positions for all particles in the frame, or null if no more frames in any batch</returns>
    public Vector3[] GetNextFrame()
    {
        if (_disposed)
            throw new ObjectDisposedException(nameof(GravitySequenceReader));

        if (!HasMoreFrames)
            return null;

        // Try to get frame from current reader
        Vector3[] frame = _currentReader?.GetNextFrame();

        // If current reader is exhausted, try to load next batch
        if (frame == null && HasMoreFrames)
        {
            if (LoadNextBatch())
            {
                frame = _currentReader?.GetNextFrame();
            }
        }

        if (frame != null)
        {
            _totalFramesRead++;
        }

        return frame;
    }

    /// <summary>
    /// Loads the next batch file in sequence
    /// </summary>
    /// <returns>True if a new batch was loaded successfully, false if no more batches</returns>
    private bool LoadNextBatch()
    {
        // Dispose current reader if it exists
        _currentReader?.Dispose();
        _currentReader = null;

        // Move to next batch
        _currentBatchIndex++;

        if (_currentBatchIndex >= _batchFiles.Count)
        {
            Console.WriteLine("All batch files have been processed.");
            return false;
        }

        string currentFile = _batchFiles[_currentBatchIndex];
        
        try
        {
            _currentReader = new GravityFrameReader(currentFile);
            
            // Store metadata from first file
            if (!NumParticles.HasValue)
            {
                NumParticles = _currentReader.NumParticles;
                FramesPerFile = _currentReader.FramesPerFile;
            }
            
            Console.WriteLine($"Loaded batch {_currentBatchIndex + 1}/{_batchFiles.Count}: {Path.GetFileName(currentFile)}");
            return true;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Failed to load batch file {currentFile}: {ex.Message}");
            return false;
        }
    }

    /// <summary>
    /// Skips ahead by the specified number of frames across multiple batch files if necessary
    /// </summary>
    /// <param name="framesToSkip">Number of frames to skip</param>
    /// <returns>True if skip was successful, false if reached end of sequence</returns>
    public bool SkipFrames(int framesToSkip)
    {
        if (_disposed)
            throw new ObjectDisposedException(nameof(GravitySequenceReader));

        if (framesToSkip <= 0)
            return true;

        for (int i = 0; i < framesToSkip; i++)
        {
            if (GetNextFrame() == null)
                return false;
        }

        return true;
    }

    /// <summary>
    /// Gets comprehensive information about the current state of the sequence reader
    /// </summary>
    public GravitySequenceInfo GetSequenceInfo()
    {
        var currentReaderInfo = _currentReader?.GetInfo();
        
        return new GravitySequenceInfo
        {
            TotalBatches = TotalBatches,
            CurrentBatch = CurrentBatch + 1, // 1-indexed for display
            CurrentBatchName = _currentBatchIndex >= 0 && _currentBatchIndex < _batchFiles.Count 
                ? Path.GetFileName(_batchFiles[_currentBatchIndex]) 
                : "None",
            TotalFramesRead = TotalFramesRead,
            NumParticles = NumParticles ?? 0,
            FramesPerFile = FramesPerFile ?? 0,
            HasMoreFrames = HasMoreFrames,
            CurrentFrameInBatch = currentReaderInfo?.CurrentFrame ?? 0,
            FramesRemainingInBatch = currentReaderInfo?.HasMoreFrames == true 
                ? (currentReaderInfo.Value.FramesPerFile - currentReaderInfo.Value.CurrentFrame) 
                : 0,
            BatchProgressPercentage = currentReaderInfo?.ProgressPercentage ?? 0f,
            OverallProgressPercentage = TotalBatches > 0 
                ? ((CurrentBatch + (currentReaderInfo?.ProgressPercentage / 100f ?? 0)) / TotalBatches) * 100f 
                : 0f
        };
    }

    /// <summary>
    /// Gets a list of all batch file names in the sequence
    /// </summary>
    public string[] GetBatchFileNames()
    {
        return _batchFiles.Select(Path.GetFileName).ToArray();
    }

    /// <summary>
    /// Estimates the total number of frames across all batch files
    /// (Only accurate if all files have the same frames per file)
    /// </summary>
    public int EstimateTotalFrames()
    {
        if (!FramesPerFile.HasValue)
            return 0;
        
        return TotalBatches * FramesPerFile.Value;
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            _currentReader?.Dispose();
            _disposed = true;
            Console.WriteLine($"GravitySequenceReader disposed. Total frames read: {_totalFramesRead}");
        }
    }
}

/// <summary>
/// Comprehensive information about the gravity sequence reader state
/// </summary>
public struct GravitySequenceInfo
{
    public int TotalBatches { get; set; }
    public int CurrentBatch { get; set; }
    public string CurrentBatchName { get; set; }
    public int TotalFramesRead { get; set; }
    public int NumParticles { get; set; }
    public int FramesPerFile { get; set; }
    public bool HasMoreFrames { get; set; }
    public int CurrentFrameInBatch { get; set; }
    public int FramesRemainingInBatch { get; set; }
    public float BatchProgressPercentage { get; set; }
    public float OverallProgressPercentage { get; set; }

    public override string ToString()
    {
        return $"Batch {CurrentBatch}/{TotalBatches} ({CurrentBatchName}) | " +
               $"Frame {CurrentFrameInBatch}/{FramesPerFile} | " +
               $"Total: {TotalFramesRead} | " +
               $"Progress: {OverallProgressPercentage:F1}%";
    }

    public string GetDetailedString()
    {
        return $"=== Gravity Sequence Status ===\n" +
               $"Current Batch: {CurrentBatch}/{TotalBatches} ({CurrentBatchName})\n" +
               $"Batch Progress: {CurrentFrameInBatch}/{FramesPerFile} ({BatchProgressPercentage:F1}%)\n" +
               $"Overall Progress: {OverallProgressPercentage:F1}%\n" +
               $"Total Frames Read: {TotalFramesRead}\n" +
               $"Particles per Frame: {NumParticles}\n" +
               $"Has More Frames: {HasMoreFrames}";
    }
}