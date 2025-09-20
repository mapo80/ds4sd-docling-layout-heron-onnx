namespace LayoutSdk;

public enum LayoutRuntime
{
    Onnx,
    Ort,
    OpenVino,

    [System.Obsolete("Use Onnx instead.")]
    OnnxRuntime = Onnx
}

