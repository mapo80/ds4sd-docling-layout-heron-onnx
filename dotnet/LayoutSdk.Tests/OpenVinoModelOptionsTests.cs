using System;
using Xunit;

namespace LayoutSdk.Tests;

public class OpenVinoModelOptionsTests
{
   [Fact]
   public void EnsureFilesExist_ThrowsNotSupportedException()
   {
       Assert.Throws<NotSupportedException>(() => global::LayoutSdk.Configuration.OpenVinoModelOptions.EnsureFilesExist());
   }
}
