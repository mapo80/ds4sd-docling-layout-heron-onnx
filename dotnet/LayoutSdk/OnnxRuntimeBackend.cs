using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using LayoutSdk.Inference;
using LayoutSdk.Processing;

namespace LayoutSdk;

internal sealed class OnnxRuntimeBackend : ILayoutBackend, IDisposable
{
    private readonly InferenceSession _session;
    private readonly string _inputName;

    public OnnxRuntimeBackend(string modelPath)
    {
        // Match Python ONNX Runtime configuration exactly
        var options = new SessionOptions
        {
            // Use CPU provider like Python: providers=["CPUExecutionProvider"]
            GraphOptimizationLevel = GraphOptimizationLevel.ORT_DISABLE_ALL, // Disable optimizations to match Python
            ExecutionMode = ExecutionMode.ORT_SEQUENTIAL,
            IntraOpNumThreads = 0,
            InterOpNumThreads = 0  // Let ONNX decide threading
        };

        _session = new InferenceSession(modelPath, options);
        _inputName = _session.InputMetadata.Keys.First();
    }

    public LayoutBackendResult Infer(ImageTensor tensor)
    {
        ArgumentNullException.ThrowIfNull(tensor);

        var dense = new DenseTensor<float>(
            tensor.Buffer.AsMemory(0, tensor.Length),
            new[] { 1, tensor.Channels, tensor.Height, tensor.Width });

        var input = NamedOnnxValue.CreateFromTensor(_inputName, dense);
        using var results = _session.Run(new[] { input });
        (input as IDisposable)?.Dispose();

        var boxes = ParseOutputs(results, tensor.Width, tensor.Height);
        return new LayoutBackendResult(boxes);
    }

    public void Dispose() => _session.Dispose();

    private static IReadOnlyList<BoundingBox> ParseOutputs(
        IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results,
        int imageWidth,
        int imageHeight)
    {
        var logitsOutput = results.FirstOrDefault(r => r.Name == "logits");
        var boxesOutput = results.FirstOrDefault(r => r.Name == "pred_boxes");

        if (logitsOutput == null || boxesOutput == null)
        {
            return Array.Empty<BoundingBox>();
        }

        var logits = logitsOutput.AsTensor<float>();
        var boxes = boxesOutput.AsTensor<float>();

        if (logits.Rank != 3 || boxes.Rank != 3)
        {
            return Array.Empty<BoundingBox>();
        }

        var labelMap = new Dictionary<int, string>
        {
            { 0, "Background" },
            { 1, "Caption" },
            { 2, "Footnote" },
            { 3, "Formula" },
            { 4, "List-item" },
            { 5, "Page-footer" },
            { 6, "Page-header" },
            { 7, "Picture" },
            { 8, "Section-header" },
            { 9, "Table" },
            { 10, "Text" },
            { 11, "Title" },
            { 12, "Document Index" },
            { 13, "Code" },
            { 14, "Checkbox-Selected" },
            { 15, "Checkbox-Unselected" },
            { 16, "Form" },
            { 17, "Key-Value Region" }
        };

        // Match Python: collect all detections first, then filter
        var detections = new List<(BoundingBox Box, float Score, int ClassId)>();
        var numQueries = logits.Dimensions[1];
        var numClasses = logits.Dimensions[2];

        for (var q = 0; q < numQueries; q++)
        {
            var maxLogit = float.NegativeInfinity;
            for (var c = 0; c < numClasses; c++)
            {
                maxLogit = Math.Max(maxLogit, logits[0, q, c]);
            }

            var expSums = 0f;
            var bestExp = 0f;
            var maxClass = 0;
            for (var c = 0; c < numClasses; c++)
            {
                var exp = MathF.Exp(logits[0, q, c] - maxLogit);
                expSums += exp;
                if (exp > bestExp)
                {
                    bestExp = exp;
                    maxClass = c;
                }
            }

            if (expSums <= 0f)
            {
                continue;
            }

            var score = bestExp / expSums;

            var cx = boxes[0, q, 0];
            var cy = boxes[0, q, 1];
            var w = boxes[0, q, 2];
            var h = boxes[0, q, 3];

            var width = w * imageWidth;
            var height = h * imageHeight;
            var x = (cx * imageWidth) - (width / 2f);
            var y = (cy * imageHeight) - (height / 2f);

            x = Math.Clamp(x, 0f, Math.Max(0f, imageWidth - width));
            y = Math.Clamp(y, 0f, Math.Max(0f, imageHeight - height));
            width = Math.Clamp(width, 0f, imageWidth - x);
            height = Math.Clamp(height, 0f, imageHeight - y);

            var label = labelMap.TryGetValue(maxClass, out var mapped) ? mapped : "Unknown";

            // Collect ALL detections (including background and low scores)
            detections.Add((new BoundingBox(x, y, width, height, label), score, maxClass));
        }

        // Apply HuggingFace-style post-processing (match Python exactly)
        return ApplyHuggingFacePostProcessing(detections, imageWidth, imageHeight);
    }

    private static IReadOnlyList<BoundingBox> ApplyHuggingFacePostProcessing(
        List<(BoundingBox Box, float Score, int ClassId)> detections,
        int imageWidth,
        int imageHeight)
    {
        // Step 1: Filter out background and low confidence (match Python threshold=0.25)
        var filtered = detections.Where(d => d.ClassId != 0 && d.Score >= 0.25f).ToList();

        // Step 2: Sort by confidence (highest first)
        var sorted = filtered.OrderByDescending(d => d.Score).ToList();

        // Step 3: Apply Non-Maximum Suppression (NMS) like HuggingFace
        return ApplyNMS(sorted, iouThreshold: 0.7f);
    }

    private static IReadOnlyList<BoundingBox> ApplyNMS(
        List<(BoundingBox Box, float Score, int ClassId)> detections,
        float iouThreshold)
    {
        if (detections.Count == 0)
        {
            return Array.Empty<BoundingBox>();
        }

        var result = new List<BoundingBox>();

        foreach (var candidate in detections)
        {
            var shouldKeep = true;

            foreach (var existing in result)
            {
                // Only suppress if same label
                if (!string.Equals(existing.Label, candidate.Box.Label, StringComparison.OrdinalIgnoreCase))
                {
                    continue;
                }

                if (ComputeIoU(existing, candidate.Box) > iouThreshold)
                {
                    shouldKeep = false;
                    break;
                }
            }

            if (shouldKeep)
            {
                result.Add(candidate.Box);
            }
        }

        return result;
    }

    private static IReadOnlyList<BoundingBox> ApplyNonMaxSuppression(List<(BoundingBox Box, float Score)> detections)
    {
        if (detections.Count == 0)
        {
            return Array.Empty<BoundingBox>();
        }

        var ordered = detections
            .OrderByDescending(d => d.Score)
            .ToList();

        var result = new List<BoundingBox>();

        const float iouThreshold = 0.7f;

        foreach (var candidate in ordered)
        {
            var overlaps = false;
            foreach (var existing in result)
            {
                if (!string.Equals(existing.Label, candidate.Box.Label, StringComparison.OrdinalIgnoreCase))
                {
                    continue;
                }

                if (ComputeIoU(existing, candidate.Box) > iouThreshold)
                {
                    overlaps = true;
                    break;
                }
            }

            if (!overlaps)
            {
                result.Add(candidate.Box);
            }
        }

        return result;
    }

    private static float ComputeIoU(BoundingBox first, BoundingBox second)
    {
        var ax1 = first.X;
        var ay1 = first.Y;
        var ax2 = first.X + first.Width;
        var ay2 = first.Y + first.Height;

        var bx1 = second.X;
        var by1 = second.Y;
        var bx2 = second.X + second.Width;
        var by2 = second.Y + second.Height;

        var interLeft = Math.Max(ax1, bx1);
        var interTop = Math.Max(ay1, by1);
        var interRight = Math.Min(ax2, bx2);
        var interBottom = Math.Min(ay2, by2);

        var interWidth = Math.Max(0f, interRight - interLeft);
        var interHeight = Math.Max(0f, interBottom - interTop);
        var interArea = interWidth * interHeight;

        var areaA = Math.Max(0f, first.Width) * Math.Max(0f, first.Height);
        var areaB = Math.Max(0f, second.Width) * Math.Max(0f, second.Height);
        var union = areaA + areaB - interArea;
        if (union <= 0f)
        {
            return 0f;
        }

        return interArea / union;
    }
}
