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
    /// Post-processes raw layout detections using advanced algorithms.
    /// </summary>
    /// <param name="rawBoxes">Raw bounding boxes from layout detection</param>
    /// <returns>Post-processed bounding boxes</returns>
    public IReadOnlyList<BoundingBox> Postprocess(IReadOnlyList<BoundingBox> rawBoxes)
    {
        if (rawBoxes == null || rawBoxes.Count == 0)
        {
            return Array.Empty<BoundingBox>();
        }

        // Phase 1: Union-Find merge di componenti connessi
        var mergedBoxes = ApplyUnionFindMerge(rawBoxes);

        // Phase 2: Spatial indexing per ottimizzazioni
        var spatialIndex = BuildSpatialIndex(mergedBoxes);

        // Phase 3: Label-specific filtering avanzato
        var filteredBoxes = ApplyLabelSpecificFilters(mergedBoxes, spatialIndex);

        // Phase 4: Wrapper/picture detection intelligente
        var finalBoxes = DetectWrappersAndPictures(filteredBoxes, spatialIndex);

        return finalBoxes;
    }

    /// <summary>
    /// Applies Union-Find algorithm to merge overlapping or connected bounding boxes.
    /// </summary>
    private IReadOnlyList<BoundingBox> ApplyUnionFindMerge(IReadOnlyList<BoundingBox> boxes)
    {
        var unionFind = new UnionFind<BoundingBox>(boxes);
        var mergeThreshold = _options.UnionFindMergeThreshold;

        // Trova componenti connessi usando IoU e distanza spaziale
        for (int i = 0; i < boxes.Count; i++)
        {
            for (int j = i + 1; j < boxes.Count; j++)
            {
                if (ShouldMergeBoxes(boxes[i], boxes[j], mergeThreshold))
                {
                    unionFind.Union(i, j);
                }
            }
        }

        // Crea merged boxes dai gruppi connessi
        var mergedBoxes = new List<BoundingBox>();
        var processedGroups = new HashSet<int>();

        for (int i = 0; i < boxes.Count; i++)
        {
            var groupId = unionFind.Find(i);
            if (processedGroups.Contains(groupId))
            {
                continue;
            }

            var groupBoxes = unionFind.GetGroupBoxes(groupId);
            var mergedBox = MergeBoundingBoxes(groupBoxes);
            mergedBoxes.Add(mergedBox);

            processedGroups.Add(groupId);
        }

        return mergedBoxes;
    }

    /// <summary>
    /// Determina se due bounding box dovrebbero essere mergeati.
    /// </summary>
    private bool ShouldMergeBoxes(BoundingBox box1, BoundingBox box2, float threshold)
    {
        // Calcola IoU
        var iou = CalculateIoU(box1, box2);

        // Calcola distanza spaziale relativa
        var centerDistance = CalculateCenterDistance(box1, box2);
        var avgDimension = (box1.Width + box1.Height + box2.Width + box2.Height) / 4f;
        var relativeDistance = centerDistance / avgDimension;

        // Merge se IoU alto O (stessa label E distanza relativa bassa)
        return iou > threshold ||
               (string.Equals(box1.Label, box2.Label, StringComparison.OrdinalIgnoreCase) &&
                relativeDistance < _options.MaxRelativeDistance);
    }

    /// <summary>
    /// Merges multiple bounding boxes into a single one.
    /// </summary>
    private BoundingBox MergeBoundingBoxes(IReadOnlyList<BoundingBox> boxes)
    {
        if (boxes.Count == 0)
        {
            throw new ArgumentException("Cannot merge empty box collection");
        }

        if (boxes.Count == 1)
        {
            return boxes[0];
        }

        // Trova bounding box complessivo
        float minX = boxes.Min(b => b.X);
        float minY = boxes.Min(b => b.Y);
        float maxX = boxes.Max(b => b.X + b.Width);
        float maxY = boxes.Max(b => b.Y + b.Height);

        var mergedWidth = maxX - minX;
        var mergedHeight = maxY - minY;

        // Usa label del box con confidence più alta (se disponibile)
        var primaryBox = boxes.OrderByDescending(b => b.Confidence).First();

        return new BoundingBox(minX, minY, mergedWidth, mergedHeight, primaryBox.Label, primaryBox.Confidence);
    }

    /// <summary>
    /// Builds spatial index for performance optimization.
    /// </summary>
    private SpatialIndex<BoundingBox> BuildSpatialIndex(IReadOnlyList<BoundingBox> boxes)
    {
        var index = new SpatialIndex<BoundingBox>(_options.SpatialIndexCellSize);

        for (int i = 0; i < boxes.Count; i++)
        {
            index.Insert(boxes[i], i);
        }

        return index;
    }

    /// <summary>
    /// Applies label-specific filtering with advanced logic.
    /// </summary>
    private IReadOnlyList<BoundingBox> ApplyLabelSpecificFilters(
        IReadOnlyList<BoundingBox> boxes,
        SpatialIndex<BoundingBox> spatialIndex)
    {
        var filteredBoxes = new List<BoundingBox>();

        foreach (var box in boxes)
        {
            if (ShouldKeepBoxAfterLabelSpecificFiltering(box, spatialIndex))
            {
                filteredBoxes.Add(box);
            }
        }

        return filteredBoxes;
    }

    /// <summary>
    /// Determines if a box should be kept after label-specific filtering.
    /// </summary>
    private bool ShouldKeepBoxAfterLabelSpecificFiltering(BoundingBox box, SpatialIndex<BoundingBox> spatialIndex)
    {
        // Label-specific thresholds
        if (!_options.LabelThresholds.TryGetValue(box.Label, out var threshold))
        {
            threshold = _options.DefaultThreshold;
        }

        if (box.Confidence < threshold)
        {
            return false;
        }

        // Label-specific size constraints
        if (!_options.LabelSizeConstraints.TryGetValue(box.Label, out var sizeConstraint))
        {
            sizeConstraint = _options.DefaultSizeConstraint;
        }

        if (box.Width < sizeConstraint.MinWidth || box.Height < sizeConstraint.MinHeight ||
            box.Width > sizeConstraint.MaxWidth || box.Height > sizeConstraint.MaxHeight)
        {
            return false;
        }

        // Label-specific spatial relationships
        var nearbyBoxes = spatialIndex.GetNearby(box, _options.SpatialContextRadius);
        return EvaluateSpatialRelationships(box, nearbyBoxes);
    }

    /// <summary>
    /// Detects wrapper and picture relationships.
    /// </summary>
    private IReadOnlyList<BoundingBox> DetectWrappersAndPictures(
        IReadOnlyList<BoundingBox> boxes,
        SpatialIndex<BoundingBox> spatialIndex)
    {
        var resultBoxes = new List<BoundingBox>(boxes);
        var processedIndices = new HashSet<int>();

        for (int i = 0; i < boxes.Count; i++)
        {
            if (processedIndices.Contains(i))
            {
                continue;
            }

            var box = boxes[i];
            var relationships = AnalyzeBoxRelationships(box, spatialIndex);

            if (relationships.IsWrapper)
            {
                // Crea wrapper relationship
                var wrapper = CreateWrapperBox(box, relationships.WrappedBoxes);
                resultBoxes.Add(wrapper);
                processedIndices.UnionWith(relationships.WrappedIndices);
            }
            else if (relationships.IsPicture)
            {
                // Crea picture relationship
                var picture = CreatePictureBox(box, relationships.CaptionBoxes);
                if (picture != box) // Only add if it was modified (caption found)
                {
                    resultBoxes.Add(picture);
                }
            }
        }

        return resultBoxes;
    }

    /// <summary>
    /// Evaluates spatial relationships between boxes.
    /// </summary>
    private bool EvaluateSpatialRelationships(BoundingBox box, IReadOnlyList<BoundingBox> nearbyBoxes)
    {
        // Implementa logica specifica per label
        return box.Label.ToUpperInvariant() switch
        {
            "CAPTION" => EvaluateCaptionRelationships(box, nearbyBoxes),
            "PICTURE" => EvaluatePictureRelationships(box, nearbyBoxes),
            "TABLE" => EvaluateTableRelationships(box, nearbyBoxes),
            _ => true // Altri label: nessuna restrizione spaziale
        };
    }

    /// <summary>
    /// Evaluates caption-specific spatial relationships.
    /// </summary>
    private bool EvaluateCaptionRelationships(BoundingBox caption, IReadOnlyList<BoundingBox> nearbyBoxes)
    {
        // Caption dovrebbe essere vicina a picture/table
        var picturesOrTables = nearbyBoxes.Where(b =>
            string.Equals(b.Label, "Picture", StringComparison.OrdinalIgnoreCase) ||
            string.Equals(b.Label, "Table", StringComparison.OrdinalIgnoreCase)).ToList();

        return picturesOrTables.Count > 0;
    }

    /// <summary>
    /// Evaluates picture-specific spatial relationships.
    /// </summary>
    private bool EvaluatePictureRelationships(BoundingBox picture, IReadOnlyList<BoundingBox> nearbyBoxes)
    {
        // Picture può avere caption vicina
        var captions = nearbyBoxes.Where(b =>
            string.Equals(b.Label, "Caption", StringComparison.OrdinalIgnoreCase)).ToList();

        return true; // Picture è sempre valida, caption è opzionale
    }

    /// <summary>
    /// Evaluates table-specific spatial relationships.
    /// </summary>
    private bool EvaluateTableRelationships(BoundingBox table, IReadOnlyList<BoundingBox> nearbyBoxes)
    {
        // Table può avere caption sopra/sotto
        var captions = nearbyBoxes.Where(b =>
            string.Equals(b.Label, "Caption", StringComparison.OrdinalIgnoreCase)).ToList();

        return true; // Table è sempre valida, caption è opzionale
    }

    /// <summary>
    /// Analyzes relationships between a box and nearby boxes.
    /// </summary>
    private BoxRelationships AnalyzeBoxRelationships(BoundingBox box, SpatialIndex<BoundingBox> spatialIndex)
    {
        var nearbyBoxes = spatialIndex.GetNearby(box, _options.RelationshipAnalysisRadius);
        var relationships = new BoxRelationships();

        foreach (var nearbyBox in nearbyBoxes.Where(b => !ReferenceEquals(b, box)))
        {
            var relationship = AnalyzeBoxPairRelationship(box, nearbyBox);
            relationships.AddRelationship(relationship);
        }

        return relationships;
    }

    /// <summary>
    /// Analyzes relationship between two specific boxes.
    /// </summary>
    private BoxPairRelationship AnalyzeBoxPairRelationship(BoundingBox box1, BoundingBox box2)
    {
        var iou = CalculateIoU(box1, box2);
        var distance = CalculateCenterDistance(box1, box2);
        var verticalOverlap = CalculateVerticalOverlap(box1, box2);
        var horizontalOverlap = CalculateHorizontalOverlap(box1, box2);

        return new BoxPairRelationship
        {
            IoU = iou,
            CenterDistance = distance,
            VerticalOverlap = verticalOverlap,
            HorizontalOverlap = horizontalOverlap,
            Label1 = box1.Label,
            Label2 = box2.Label
        };
    }

    /// <summary>
    /// Creates wrapper box from main box and wrapped boxes.
    /// </summary>
    private BoundingBox CreateWrapperBox(BoundingBox mainBox, IReadOnlyList<BoundingBox> wrappedBoxes)
    {
        var allBoxes = new[] { mainBox }.Concat(wrappedBoxes).ToList();
        var merged = MergeBoundingBoxes(allBoxes);

        return new BoundingBox(
            merged.X, merged.Y, merged.Width, merged.Height,
            "Wrapper", merged.Confidence);
    }

    /// <summary>
    /// Creates picture box with associated caption.
    /// </summary>
    private BoundingBox CreatePictureBox(BoundingBox picture, IReadOnlyList<BoundingBox> captionBoxes)
    {
        if (captionBoxes.Count == 0)
        {
            return picture; // Return original picture if no caption
        }

        // Usa caption più vicina
        var closestCaption = captionBoxes.OrderBy(c => CalculateCenterDistance(picture, c)).First();
        var merged = MergeBoundingBoxes(new[] { picture, closestCaption });

        return new BoundingBox(
            merged.X, merged.Y, merged.Width, merged.Height,
            "Picture", merged.Confidence);
    }

    #region Utility Methods

    private float CalculateIoU(BoundingBox box1, BoundingBox box2)
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

    private float CalculateCenterDistance(BoundingBox box1, BoundingBox box2)
    {
        var center1X = box1.X + box1.Width / 2f;
        var center1Y = box1.Y + box1.Height / 2f;
        var center2X = box2.X + box2.Width / 2f;
        var center2Y = box2.Y + box2.Height / 2f;

        var dx = center1X - center2X;
        var dy = center1Y - center2Y;

        return MathF.Sqrt(dx * dx + dy * dy);
    }

    private float CalculateVerticalOverlap(BoundingBox box1, BoundingBox box2)
    {
        var overlapStart = Math.Max(box1.Y, box2.Y);
        var overlapEnd = Math.Min(box1.Y + box1.Height, box2.Y + box2.Height);
        return Math.Max(0f, overlapEnd - overlapStart);
    }

    private float CalculateHorizontalOverlap(BoundingBox box1, BoundingBox box2)
    {
        var overlapStart = Math.Max(box1.X, box2.X);
        var overlapEnd = Math.Min(box1.X + box1.Width, box2.X + box2.Width);
        return Math.Max(0f, overlapEnd - overlapStart);
    }

    #endregion

    #region Relationship Classes

    private class BoxRelationships
    {
        public bool IsWrapper { get; private set; }
        public bool IsPicture { get; private set; }
        public List<BoundingBox> WrappedBoxes { get; } = new();
        public List<BoundingBox> CaptionBoxes { get; } = new();
        public HashSet<int> WrappedIndices { get; } = new();

        public void AddRelationship(BoxPairRelationship relationship)
        {
            // Analizza relationship e aggiorna stati
            if (IsWrapperRelationship(relationship))
            {
                IsWrapper = true;
            }

            if (IsPictureCaptionRelationship(relationship))
            {
                IsPicture = true;
            }
        }

        private bool IsWrapperRelationship(BoxPairRelationship relationship)
        {
            // Logica per determinare se è una relazione wrapper
            return relationship.IoU > 0.8f &&
                   (relationship.Label1 == "Text" && relationship.Label2 == "Text");
        }

        private bool IsPictureCaptionRelationship(BoxPairRelationship relationship)
        {
            // Logica per determinare se è una relazione picture-caption
            return (relationship.Label1 == "Picture" && relationship.Label2 == "Caption") ||
                   (relationship.Label1 == "Caption" && relationship.Label2 == "Picture");
        }
    }

    private class BoxPairRelationship
    {
        public float IoU { get; set; }
        public float CenterDistance { get; set; }
        public float VerticalOverlap { get; set; }
        public float HorizontalOverlap { get; set; }
        public string Label1 { get; set; } = string.Empty;
        public string Label2 { get; set; } = string.Empty;
    }

    #endregion
}