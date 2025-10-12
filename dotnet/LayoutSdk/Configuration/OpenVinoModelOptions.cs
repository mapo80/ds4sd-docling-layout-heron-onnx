using System;

namespace LayoutSdk.Configuration;

public sealed class OpenVinoModelOptions
{
   public static void EnsureFilesExist()
   {
       throw new NotSupportedException("OpenVINO runtime has been removed. Use only ONNX runtime.");
   }
}
