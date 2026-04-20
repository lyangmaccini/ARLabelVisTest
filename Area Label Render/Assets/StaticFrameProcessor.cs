using System.Collections.Generic;
using UnityEngine;
using System.Linq;
using System.IO;
using System;
using UnityEngine.Rendering;

#if UNITY_EDITOR
using UnityEditor;
#endif

public class StaticFrameProcessor : MonoBehaviour
{
    [Header("Scene References")]
    public GameObject videoQuad;

    [Header("Textures")]
    public Texture2D backgroundFrame;   // Your test image
    public Texture2D LabelMask;
    public Texture2D BackgroundMask;

    [Header("Compute & Lookup")]
    public ComputeShader cShaderForMask;
    public Texture3D LookupTable;

    [Header("Settings")]
    public bool weightedBins;

    Material mat;
    Texture2D maskedBackgroundAsTex2D;
    RenderTexture backgroundRT;
    int w, h;

    private Queue<AsyncGPUReadbackRequest> requests = new Queue<AsyncGPUReadbackRequest>();
    private Color32[] backgroundDataBuffer;
    int maskBuffer_kernelID;

    List<List<Color32>> ColorHistogram;
    List<Vector3> ColorHistogramBins;
    Color32 backgroundAvg;
    Color32 labelAvg;


    Color32 LAB2RGB(Vector3 LAB)
    {
        double L = LAB[0], A = LAB[1], B = LAB[2];
        double Xr = 95.047, Yr = 100.0, Zr = 108.883;
        double var_Y = (L + 16.0) / 116.0;
        double var_X = A / 500 + var_Y;
        double var_Z = var_Y - B / 200.0;
        if (Math.Pow(var_Y, 3) > 0.008856) var_Y = Math.Pow(var_Y, 3.0); else var_Y = (var_Y - 16.0 / 116.0) / 7.787;
        if (Math.Pow(var_X, 3) > 0.008856) var_X = Math.Pow(var_X, 3.0); else var_X = (var_X - 16.0 / 116.0) / 7.787;
        if (Math.Pow(var_Z, 3) > 0.008856) var_Z = Math.Pow(var_Z, 3.0); else var_Z = (var_Z - 16.0 / 116.0) / 7.787;
        double X = var_X * Xr / 100.0, Y = var_Y * Yr / 100.0, Z = var_Z * Zr / 100.0;
        double var_R = X * 3.2406 + Y * -1.5372 + Z * -0.4986;
        double var_G = X * -0.9689 + Y * 1.8758 + Z * 0.0415;
        double var_B = X * 0.0557 + Y * -0.2040 + Z * 1.0570;
        if (var_R > 0.0031308) var_R = 1.055 * Math.Pow(var_R, 1.0 / 2.4) - 0.055; else var_R = 12.92 * var_R;
        if (var_G > 0.0031308) var_G = 1.055 * Math.Pow(var_G, 1.0 / 2.4) - 0.055; else var_G = 12.92 * var_G;
        if (var_B > 0.0031308) var_B = 1.055 * Math.Pow(var_B, 1.0 / 2.4) - 0.055; else var_B = 12.92 * var_B;
        return new Color32(
            (byte)(int)Math.Max(Math.Min(var_R * 255, 255), 0),
            (byte)(int)Math.Max(Math.Min(var_G * 255, 255), 0),
            (byte)(int)Math.Max(Math.Min(var_B * 255, 255), 0), 255);
    }

    void toTexture2D(RenderTexture rTex, Texture2D target)
    {
        RenderTexture.active = rTex;
        target.ReadPixels(new Rect(0, 0, rTex.width, rTex.height), 0, 0);
        target.Apply();
        RenderTexture.active = null;
    }

    void ApplyMask(Texture2D background, Texture2D mask)
    {
        cShaderForMask.SetInt("image_width", w);
        cShaderForMask.SetInt("image_height", h);
        cShaderForMask.SetTexture(maskBuffer_kernelID, "backgroundScreenshotForSum", background);
        cShaderForMask.SetTexture(maskBuffer_kernelID, "labelScreenshotForSum", mask);
        cShaderForMask.SetTexture(maskBuffer_kernelID, "Result", backgroundRT);
        cShaderForMask.Dispatch(maskBuffer_kernelID, w, h, 1);
        toTexture2D(backgroundRT, maskedBackgroundAsTex2D);
    }

    void FindHistogramAverageColor(Texture2D maskedTex, int granularityMode)
    {
        if (requests.Count < 8)
        {
            requests.Enqueue(AsyncGPUReadback.Request(maskedTex, 0, TextureFormat.RGBA32, (AsyncGPUReadbackRequest req) =>
            {
                if (req.hasError) { Debug.Log("GPU readback error."); requests.Dequeue(); return; }
                if (req.done)
                {
                    req.GetData<Color32>().CopyTo(backgroundDataBuffer);
                    int averageR = 0, averageG = 0, averageB = 0, binSizeSum = 0;

                    for (int i = 0; i < backgroundDataBuffer.Length; i++)
                    {
                        if (backgroundDataBuffer[i].a != 0)
                        {
                            for (int binIdx = 0; binIdx < ColorHistogramBins.Count; binIdx++)
                            {
                                Vector3 bin = ColorHistogramBins[binIdx];
                                if (backgroundDataBuffer[i].r <= bin[0] &&
                                    backgroundDataBuffer[i].g <= bin[1] &&
                                    backgroundDataBuffer[i].b <= bin[2])
                                { ColorHistogram[binIdx].Add(backgroundDataBuffer[i]); break; }
                            }
                        }
                    }

                    ColorHistogram = ColorHistogram.OrderBy(b => b.Count).ToList();

                    for (int b = 0; b < 2; b++)
                    {
                        List<Color32> bin = ColorHistogram[ColorHistogram.Count - 1 - b];
                        foreach (Color32 c in bin) { averageR += c.r; averageG += c.g; averageB += c.b; }
                        binSizeSum += bin.Count;
                    }
                    if (binSizeSum != 0) { averageR /= binSizeSum; averageG /= binSizeSum; averageB /= binSizeSum; }

                    Color32 avg = new Color32((byte)averageR, (byte)averageG, (byte)averageB, 255);
                    if (granularityMode == 2) backgroundAvg = avg; else labelAvg = avg;

                    // Update the shader with the new average
                    Color32 avgToUse = (mat.GetInt("_GranularityMethod") == 2) ? backgroundAvg : labelAvg;
                    mat.SetFloat("_Background_sum_r", avgToUse.r / 255.0f);
                    mat.SetFloat("_Background_sum_g", avgToUse.g / 255.0f);
                    mat.SetFloat("_Background_sum_b", avgToUse.b / 255.0f);

                    for (int b = 0; b < ColorHistogram.Count; b++) ColorHistogram[b].Clear();
                }
                requests.Dequeue();
            }));
        }
    }

    void Start()
    {
        mat = videoQuad.GetComponent<MeshRenderer>().sharedMaterial;

        w = LabelMask.width;
        h = LabelMask.height;

        maskedBackgroundAsTex2D = new Texture2D(w, h, TextureFormat.RGBA32, false);
        backgroundDataBuffer    = new Color32[w * h];

        backgroundRT = RenderTexture.GetTemporary(w, h);
        backgroundRT.enableRandomWrite = true;

        maskBuffer_kernelID = cShaderForMask.FindKernel("CSMain");

        // Assign the static background frame and label mask to the shader
        mat.SetTexture("_MainTex", backgroundFrame);
        mat.SetTexture("_LabelMask", LabelMask);

        // Use existing lookup table
        if (LookupTable == null)
            Debug.LogError("Please assign the LookupTable in the Inspector.");
        else
        {
            Debug.Log("Using existing LookupTable.");
            mat.SetTexture("_CIELAB_LookupTable", LookupTable);
        }

        // Set up histogram bins
        ColorHistogram     = new List<List<Color32>>();
        ColorHistogramBins = new List<Vector3>();
        for (int i = 0; i < 27; i++) ColorHistogram.Add(new List<Color32>());
        for (int r = 85; r <= 255; r += 85)
            for (int g = 85; g <= 255; g += 85)
                for (int b = 85; b <= 255; b += 85)
                    ColorHistogramBins.Add(new Vector3(r, g, b));

        // Run the color processing once on the static frame
        ApplyMask(backgroundFrame, BackgroundMask);
        FindHistogramAverageColor(maskedBackgroundAsTex2D, 2);

        ApplyMask(backgroundFrame, LabelMask);
        FindHistogramAverageColor(maskedBackgroundAsTex2D, 1);
    }

    void OnDestroy()
    {
        RenderTexture.ReleaseTemporary(backgroundRT);
    }
}