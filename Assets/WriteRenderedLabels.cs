using System.Collections.Generic;
using UnityEngine;
using System.Linq;
using System.IO;
using System;
using UnityEngine.Video;

#if UNITY_EDITOR
using UnityEditor;
#endif

public class RenderStereoBackgroundforAreaLabel : MonoBehaviour
{
    [Header("Scene References")]
    public GameObject backgroundAndLabelSphere;
    public Camera mainCamera;

    [Header("Video")]
    public VideoPlayer videoPlayer;
    public RenderTexture videoRenderTexture;

    [Header("Textures")]
    public Texture2D LabelMask;
    public Texture2D BackgroundMask;

    [Header("Compute & Lookup")]
    public ComputeShader cShaderForMask;
    public Texture3D LookupTable;

    [Header("Settings")]
    public bool weightedBins;
    public bool backgroundOrLableChanged;

    // Private fields
    Material backgroundAndLabelSphereMaterial;
    Texture2D currentVideoFrame;
    Texture2D maskedBackgroundAsTex2D;

    RenderTexture backgroundRT;
    int w;
    int h;

    int maskBuffer_kernelID;
    List<Color32> CandidateCIELABVals;

    List<List<Color32>> ColorHistogram;
    List<Vector3> ColorHistogramBins;

    Color32 backgroundAvg;
    Color32 labelAvg;

    // CSV export
    StreamWriter csvWriter;


    // -----------------------------------------------------------------------
    // Color Conversion
    // -----------------------------------------------------------------------

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

        double X = var_X * Xr / 100.0;
        double Y = var_Y * Yr / 100.0;
        double Z = var_Z * Zr / 100.0;

        double var_R = X *  3.2406 + Y * -1.5372 + Z * -0.4986;
        double var_G = X * -0.9689 + Y *  1.8758 + Z *  0.0415;
        double var_B = X *  0.0557 + Y * -0.2040 + Z *  1.0570;

        if (var_R > 0.0031308) var_R = 1.055 * Math.Pow(var_R, 1.0 / 2.4) - 0.055; else var_R = 12.92 * var_R;
        if (var_G > 0.0031308) var_G = 1.055 * Math.Pow(var_G, 1.0 / 2.4) - 0.055; else var_G = 12.92 * var_G;
        if (var_B > 0.0031308) var_B = 1.055 * Math.Pow(var_B, 1.0 / 2.4) - 0.055; else var_B = 12.92 * var_B;

        int finalR = (int)(Math.Max(Math.Min(var_R * 255, 255), 0));
        int finalG = (int)(Math.Max(Math.Min(var_G * 255, 255), 0));
        int finalB = (int)(Math.Max(Math.Min(var_B * 255, 255), 0));

        return new Color32((byte)finalR, (byte)finalG, (byte)finalB, 255);
    }


    // -----------------------------------------------------------------------
    // Masking & Histogram
    // -----------------------------------------------------------------------

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

    Color32 FindHistogramAverageColor(Texture2D maskedTex)
    {
        Color32[] pixels = maskedTex.GetPixels32();

        int averageR = 0, averageG = 0, averageB = 0;

        for (int i = 0; i < pixels.Length; i++)
        {
            if (pixels[i].a != 0)
            {
                for (int binIdx = 0; binIdx < ColorHistogramBins.Count; binIdx++)
                {
                    Vector3 bin = ColorHistogramBins[binIdx];
                    if (pixels[i].r <= bin[0] &&
                        pixels[i].g <= bin[1] &&
                        pixels[i].b <= bin[2])
                    {
                        ColorHistogram[binIdx].Add(pixels[i]);
                        break;
                    }
                }
            }
        }

        ColorHistogram = ColorHistogram.OrderBy(bin => bin.Count).ToList();

        int numBinsToTake = 2;
        int binSizeSum = 0;

        if (!weightedBins)
        {
            for (int b = 0; b < numBinsToTake; b++)
            {
                List<Color32> currentBin = ColorHistogram[ColorHistogram.Count - 1 - b];
                if (currentBin.Count != 0)
                {
                    foreach (Color32 c in currentBin) { averageR += c.r; averageG += c.g; averageB += c.b; }
                    binSizeSum += currentBin.Count;
                }
            }
            if (binSizeSum != 0) { averageR /= binSizeSum; averageG /= binSizeSum; averageB /= binSizeSum; }
        }
        else
        {
            for (int b = 0; b < numBinsToTake; b++)
                binSizeSum += ColorHistogram[ColorHistogram.Count - 1 - b].Count;

            for (int b = 0; b < numBinsToTake; b++)
            {
                int tR = 0, tG = 0, tB = 0;
                List<Color32> currentBin = ColorHistogram[ColorHistogram.Count - 1 - b];
                if (currentBin.Count != 0)
                    foreach (Color32 c in currentBin) { tR += c.r; tG += c.g; tB += c.b; }

                if (binSizeSum != 0)
                {
                    float weight = (float)currentBin.Count / (float)binSizeSum;
                    averageR += (int)(((float)tR / (float)binSizeSum) * weight);
                    averageG += (int)(((float)tG / (float)binSizeSum) * weight);
                    averageB += (int)(((float)tB / (float)binSizeSum) * weight);
                }
            }
        }

        for (int b = 0; b < ColorHistogram.Count; b++)
            ColorHistogram[b].Clear();

        return new Color32((byte)averageR, (byte)averageG, (byte)averageB, 255);
    }

    void CaptureCurrentVideoFrame()
    {
        if (videoRenderTexture == null) return;
        Graphics.Blit(videoRenderTexture, backgroundRT);
        toTexture2D(backgroundRT, currentVideoFrame);
    }

    void OnFrameReady(VideoPlayer vp, long frameIdx)
    {
        CaptureCurrentVideoFrame();

        // Calculate background average color
        ApplyMask(currentVideoFrame, BackgroundMask);
        backgroundAvg = FindHistogramAverageColor(maskedBackgroundAsTex2D);

        // Calculate label region average color
        ApplyMask(currentVideoFrame, LabelMask);
        labelAvg = FindHistogramAverageColor(maskedBackgroundAsTex2D);

        // Determine which average the shader uses as input to the lookup table
        int granularityMethod = backgroundAndLabelSphereMaterial.GetInt("_GranularityMethod");
        Color32 shaderInput = (granularityMethod == 2) ? backgroundAvg : labelAvg;

        // Look up the actual rendered label color from the lookup table
        // This mirrors exactly what the shader does with tex3D(_CIELAB_LookupTable, bgSample.rgb)
        Color32 renderedLabelColor = LookupTable.GetPixel(shaderInput.r, shaderInput.g, shaderInput.b);

        // Update the shader
        backgroundAndLabelSphereMaterial.SetFloat("_Background_sum_r", shaderInput.r / 255.0f);
        backgroundAndLabelSphereMaterial.SetFloat("_Background_sum_g", shaderInput.g / 255.0f);
        backgroundAndLabelSphereMaterial.SetFloat("_Background_sum_b", shaderInput.b / 255.0f);

        // Write one row per frame to CSV
        double frameTime = frameIdx / vp.frameRate;
        if (csvWriter != null)
        {
            csvWriter.WriteLine(
                frameIdx + "," +
                frameTime.ToString("F6") + "," +
                labelAvg.r + "," + labelAvg.g + "," + labelAvg.b + "," +
                backgroundAvg.r + "," + backgroundAvg.g + "," + backgroundAvg.b + "," +
                renderedLabelColor.r + "," + renderedLabelColor.g + "," + renderedLabelColor.b
            );
            csvWriter.Flush();
        }
    }


    // -----------------------------------------------------------------------
    // Lifecycle
    // -----------------------------------------------------------------------

    void Start()
    {
        if (mainCamera == null)
        {
            Debug.LogError("RenderStereoBackgroundforAreaLabel: No camera assigned.");
            return;
        }
        if (videoPlayer == null || videoRenderTexture == null)
        {
            Debug.LogError("RenderStereoBackgroundforAreaLabel: VideoPlayer or RenderTexture not assigned.");
            return;
        }

        backgroundAndLabelSphereMaterial = backgroundAndLabelSphere.GetComponent<MeshRenderer>().sharedMaterial;

        w = LabelMask.width;
        h = LabelMask.height;

        maskedBackgroundAsTex2D = new Texture2D(w, h, TextureFormat.RGBA32, false);
        currentVideoFrame       = new Texture2D(w, h, TextureFormat.RGBA32, false);

        backgroundRT = RenderTexture.GetTemporary(w, h);
        backgroundRT.enableRandomWrite = true;

        maskBuffer_kernelID = cShaderForMask.FindKernel("CSMain");

        videoPlayer.targetTexture = videoRenderTexture;
        videoPlayer.sendFrameReadyEvents = true;
        videoPlayer.frameReady += OnFrameReady;

        backgroundAndLabelSphereMaterial.SetTexture("_MainTex", videoRenderTexture);
        backgroundAndLabelSphereMaterial.SetTexture("_LabelMask", LabelMask);

        // Build or reuse the CIELAB lookup table
        if (LookupTable == null)
        {
            Debug.Log("No LookupTable assigned — generating from .txt files...");

            CandidateCIELABVals = new List<Color32>();
            var labLines = File.ReadLines("./Assets/AllCandidateLABvals_16.txt");
            foreach (var line in labLines)
            {
                string[] parts = line.Split(",");
                Vector3 lab = new Vector3(
                    float.Parse(parts[0], System.Globalization.CultureInfo.InvariantCulture),
                    float.Parse(parts[1], System.Globalization.CultureInfo.InvariantCulture),
                    float.Parse(parts[2], System.Globalization.CultureInfo.InvariantCulture));
                CandidateCIELABVals.Add(LAB2RGB(lab));
            }

            LookupTable = new Texture3D(256, 256, 256, TextureFormat.RGBA32, false);
            int lineCounter = 0;
            var rgbLines = File.ReadLines("./Assets/AllCorrespondingRGBVals_16.txt");
            foreach (var line in rgbLines)
            {
                string[] parts = line.Split(",");
                LookupTable.SetPixel(int.Parse(parts[0]), int.Parse(parts[1]), int.Parse(parts[2]),
                                     CandidateCIELABVals[lineCounter]);
                lineCounter++;
            }
            LookupTable.Apply(updateMipmaps: false);

#if UNITY_EDITOR
            AssetDatabase.CreateAsset(LookupTable, "Assets/NEWLookupTexture_16.asset");
            AssetDatabase.SaveAssetIfDirty(LookupTable);
            Debug.Log("Lookup table saved to Assets/NEWLookupTexture_16.asset.");
#endif
        }
        else
        {
            Debug.Log("Using existing LookupTable assigned in Inspector.");
        }

        backgroundAndLabelSphereMaterial.SetTexture("_CIELAB_LookupTable", LookupTable);

        // Set up 27-bin color histogram
        ColorHistogram     = new List<List<Color32>>();
        ColorHistogramBins = new List<Vector3>();
        for (int i = 0; i < 27; i++) ColorHistogram.Add(new List<Color32>());
        for (int r = 85; r <= 255; r += 85)
            for (int g = 85; g <= 255; g += 85)
                for (int b = 85; b <= 255; b += 85)
                    ColorHistogramBins.Add(new Vector3(r, g, b));

        // Set up CSV export
        string csvPath = Application.dataPath + "/label_colors_export.csv";
        csvWriter = new StreamWriter(csvPath, false);
        csvWriter.WriteLine("frame,time_seconds,label_r,label_g,label_b,background_r,background_g,background_b,rendered_r,rendered_g,rendered_b");
        csvWriter.Flush();
        Debug.Log("Writing color export to: " + csvPath);

        backgroundOrLableChanged = true;
    }

    void OnDestroy()
    {
        if (videoPlayer != null)
            videoPlayer.frameReady -= OnFrameReady;

        RenderTexture.ReleaseTemporary(backgroundRT);

        if (csvWriter != null)
        {
            csvWriter.Close();
            Debug.Log("Color export CSV closed and saved.");
        }
    }
}
