using UnityEngine;
using UnityEditor;
using System.IO;

public class ExtractLookupTable : MonoBehaviour
{
    public Texture3D LookupTable;

    [MenuItem("Tools/Extract LAB Lookup Table")]
    public static void Extract()
    {
        Texture3D lut = AssetDatabase.LoadAssetAtPath<Texture3D>("Assets/LookupTexture.asset");

        int size = lut.width; // 256
        var sb = new System.Text.StringBuilder();

        for (int r = 0; r < size; r++)
        {
            for (int g = 0; g < size; g++)
            {
                for (int b = 0; b < size; b++)
                {
                    Color32 c = lut.GetPixel(r, g, b);
                    // r,g,b channels store L,A,B as raw bytes
                    sb.AppendLine($"{c.r},{c.g},{c.b}");
                }
            }
        }

        File.WriteAllText("./Assets/ExtractedLABvals.txt", sb.ToString());
        Debug.Log("Done extracting LAB values.");
    }
}
