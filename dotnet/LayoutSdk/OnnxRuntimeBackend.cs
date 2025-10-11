using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using LayoutSdk.Inference;
using LayoutSdk.Processing;

namespace LayoutSdk;

internal enum OnnxModelFormat
{
    Onnx,
    Ort
}

internal sealed class OnnxRuntimeBackend : ILayoutBackend, IDisposable
{
    private readonly InferenceSession _session;
    private readonly string _inputName;

    public OnnxRuntimeBackend(string modelPath, OnnxModelFormat format = OnnxModelFormat.Onnx)
    {
        using var options = CreateSessionOptions(format);
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

    private static SessionOptions CreateSessionOptions(OnnxModelFormat format)
    {
        var options = new SessionOptions
        {
            GraphOptimizationLevel = format == OnnxModelFormat.Ort
                ? GraphOptimizationLevel.ORT_DISABLE_ALL
                : GraphOptimizationLevel.ORT_ENABLE_ALL,
            ExecutionMode = ExecutionMode.ORT_SEQUENTIAL,
            IntraOpNumThreads = 0,
            InterOpNumThreads = 1
        };

        return options;
    }

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

        const float scoreThreshold = 0.3f;

        var labelThresholds = new Dictionary<string, float>(StringComparer.OrdinalIgnoreCase)
        {
            { "Caption", 0.5f },
            { "Footnote", 0.5f },
            { "Formula", 0.5f },
            { "List-item", 0.5f },
            { "Page-footer", 0.5f },
            { "Page-header", 0.5f },
            { "Picture", 0.5f },
            { "Section-header", 0.4f },
            { "Table", 0.45f },
            { "Text", 0.45f },
            { "Title", 0.4f },
            { "Code", 0.4f },
            { "Checkbox-Selected", 0.4f },
            { "Checkbox-Unselected", 0.4f },
            { "Form", 0.4f },
            { "Key-Value Region", 0.4f },
            { "Document Index", 0.4f }
        };

        var detections = new List<(BoundingBox Box, float Score)>();
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
            if (maxClass == 0 || score < scoreThreshold)
            {
                continue;
            }

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
            var minScore = labelThresholds.TryGetValue(label, out var threshold) ? threshold : scoreThreshold;
            if (score < minScore)
            {
                continue;
            }

            detections.Add((new BoundingBox(x, y, width, height, label), score));
        }

        return ApplyNonMaxSuppression(detections);
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
