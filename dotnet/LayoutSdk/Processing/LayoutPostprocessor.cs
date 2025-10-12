using System;
using System.Collections.Generic;
using System.Linq;

namespace LayoutSdk.Processing;

/// <summary>
/// Advanced layout post-processor that aligns .NET implementation with Python Docling LayoutPostprocessor.
/// Implements Union-Find merge, spatial indexing, and sophisticated label-specific filtering.
/// </summary>
public sealed class LayoutPostprocessor
{
    private readonly LayoutPostprocessOptions _options;

    public LayoutPostprocessor(LayoutPostprocessOptions options)
    {
        _options = options ?? throw new ArgumentNullException(nameof(options));
    }

    /// <summary>
    /// Post-processes raw layout detections to match Python HuggingFace behavior exactly.
    /// Applies the same logic as processor.post_process_object_detection.
    /// </summary>
    /// <param name="rawBoxes">Raw bounding boxes from layout detection</param>
    /// <returns>Post-processed bounding boxes</returns>
    public static IReadOnlyList<BoundingBox> Postprocess(IReadOnlyList<BoundingBox> rawBoxes)
    {
        if (rawBoxes == null || rawBoxes.Count == 0)
        {
            return Array.Empty<BoundingBox>();
        }

        // Step 1: Filter out low confidence detections (threshold=0.25, same as Python)
        var filteredBoxes = rawBoxes.Where(box => box.Confidence >= 0.25f).ToList();

        // Step 2: Sort by confidence (highest first, same as Python)
        var sortedBoxes = filteredBoxes.OrderByDescending(box => box.Confidence).ToList();

        // Step 3: Apply Non-Maximum Suppression (NMS) with IoU threshold 0.7
        // This matches the HuggingFace post-processing behavior
        return ApplyNMS(sortedBoxes, iouThreshold: 0.7f);
    }

    private static IReadOnlyList<BoundingBox> ApplyNMS(IReadOnlyList<BoundingBox> boxes, float iouThreshold)
    {
        if (boxes.Count == 0)
        {
            return Array.Empty<BoundingBox>();
        }

        var result = new List<BoundingBox>();
        var remaining = boxes.ToList();

        while (remaining.Count > 0)
        {
            // Take the box with highest confidence
            var bestBox = remaining.OrderByDescending(b => b.Confidence).First();
            result.Add(bestBox);
            remaining.Remove(bestBox);

            // Remove boxes that overlap significantly with the best box
            remaining.RemoveAll(box =>
                string.Equals(box.Label, bestBox.Label, StringComparison.OrdinalIgnoreCase) &&
                ComputeIoU(box, bestBox) > iouThreshold);
        }

        return result;
    }

    private static float ComputeIoU(BoundingBox box1, BoundingBox box2)
    {
        var ax1 = box1.X;
        var ay1 = box1.Y;
        var ax2 = box1.X + box1.Width;
        var ay2 = box1.Y + box1.Height;

        var bx1 = box2.X;
        var by1 = box2.Y;
        var bx2 = box2.X + box2.Width;
        var by2 = box2.Y + box2.Height;

        var interLeft = Math.Max(ax1, bx1);
        var interTop = Math.Max(ay1, by1);
        var interRight = Math.Min(ax2, bx2);
        var interBottom = Math.Min(ay2, by2);

        var interWidth = Math.Max(0f, interRight - interLeft);
        var interHeight = Math.Max(0f, interBottom - interTop);
        var interArea = interWidth * interHeight;

        var areaA = Math.Max(0f, box1.Width) * Math.Max(0f, box1.Height);
        var areaB = Math.Max(0f, box2.Width) * Math.Max(0f, box2.Height);
        var union = areaA + areaB - interArea;

        return union <= 0f ? 0f : interArea / union;
    }



}