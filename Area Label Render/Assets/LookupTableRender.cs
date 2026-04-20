using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Linq;
using UnityEngine.Networking;
using System.IO;
using System;
using TMPro;
using System.Text;
using System.Linq;
using UnityEngine.UI;
using System.Threading.Tasks;
using UnityEngine.Rendering;
using UnityEditor;

public class LookupTableRender : MonoBehaviour
{
    public Texture3D LookupTable;
    List<Color32> CandidateCIELABVals;

    Color32 LAB2RGB(Vector3 LAB)
    {
        double L = LAB[0];
        double A = LAB[1];
        double B = LAB[2];

        // reference values, D65/2°
        double Xr = 95.047;  
        double Yr = 100.0;
        double Zr = 108.883;

        // first convert LAB to XYZ
        double var_Y = (L + 16.0) / 116.0;
        double var_X = A / 500 + var_Y;
        double var_Z = var_Y - B / 200.0;

        if (Math.Pow(var_Y, 3)  > 0.008856){
            var_Y = Math.Pow(var_Y, 3.0);
        }  
        else{
            var_Y = (var_Y - 16 / 116) / 7.787;
        }
            
        if (Math.Pow(var_X, 3)  > 0.008856){
            var_X = Math.Pow(var_X, 3.0);
        }
        else{
            var_X = (var_X - 16 / 116) / 7.787;
        }
            
        if (Math.Pow(var_Z, 3)  > 0.008856){
            var_Z = Math.Pow(var_Z, 3.0);
        } 
        else{
            var_Z = (var_Z - 16.0 / 116.0) / 7.787;
        }
            
        double X = var_X * Xr;
        double Y = var_Y * Yr;
        double Z = var_Z * Zr;

        // now convert XYZ to RGB
        X /= 100.0;
        Y /= 100.0;
        Z /= 100.0;

        double var_R = var_X *  3.2406 + var_Y * -1.5372 + var_Z * -0.4986;
        double var_G = var_X * -0.9689 + var_Y *  1.8758 + var_Z *  0.0415;
        double var_B = var_X *  0.0557 + var_Y * -0.2040 + var_Z *  1.0570;

        if (var_R > 0.0031308){
            var_R = 1.055 * (Math.Pow(var_R, (1 / 2.4))) - 0.055;
        } 
        else{
            var_R = 12.92 * var_R;
        }
            
        if (var_G > 0.0031308){
            var_G = 1.055 * (Math.Pow(var_G, (1 / 2.4))) - 0.055;
        } 
        else{
            var_G = 12.92 * var_G;
        }
            
        if (var_B > 0.0031308){
            var_B = 1.055 * (Math.Pow(var_B, (1 / 2.4))) - 0.055;
        } 
            
        else{
            var_B = 12.92 * var_B;
        }

        // ensure values are between 0 and 255
        int finalR = (int) (Math.Max(Math.Min(var_R*255, 255), 0));
        int finalG = (int) (Math.Max(Math.Min(var_G*255, 255), 0));
        int finalB = (int) (Math.Max(Math.Min(var_B*255, 255), 0));

        Color32 RGB = new Color32((byte) (finalR), (byte) (finalG), (byte) (finalB), 255);
        return RGB;
    }

    // Start is called before the first frame update
    void Start()
    {
        CandidateCIELABVals = new List<Color32>();
        var linesReadLAB = File.ReadLines("./Assets/AllCandidateLABvals_RGB_1_euclidean.txt");
        foreach (var lineReadLAB in linesReadLAB)
        {
            string[] num = lineReadLAB.Split(",");
            Vector3 currentLAB = new Vector3(float.Parse(num[0], System.Globalization.CultureInfo.InvariantCulture), 
                                            float.Parse(num[1], System.Globalization.CultureInfo.InvariantCulture), 
                                            float.Parse(num[2], System.Globalization.CultureInfo.InvariantCulture));
            Color32 currentLABAsRGB = new Color32((byte) (currentLAB[0]), (byte) (currentLAB[1]), (byte) (currentLAB[2]), 255);
            CandidateCIELABVals.Add(currentLABAsRGB);
        }
        // int lookupTableStepSize = 1; // change for a different step size

        // Make a lookup table (texture3d) with the corresponding LAB-to-RGB converted value at each RGB index
        int lineCounter = 0;
        LookupTable = new Texture3D(256, 256, 256, TextureFormat.RGBA32, false);
        
        // LookupTable.mipCount = 0;
        var linesReadRGB = File.ReadLines("./Assets/AllCorrespondingRGBVals.txt");
       
        foreach (var lineReadRGB in linesReadRGB){
            string[] num = lineReadRGB.Split(",");
            int rIdx = int.Parse(num[0]);
            int gIdx = int.Parse(num[1]);
            int bIdx = int.Parse(num[2]);
            LookupTable.SetPixel(rIdx, gIdx, bIdx, CandidateCIELABVals[lineCounter]);

            //Debug.Log("counter: " + lineCounter);
            //Debug.Log(CandidateCIELABVals[lineCounter]);
            //Debug.Log(LookupTable.GetPixel(rIdx, gIdx, bIdx));
            lineCounter += 1;
            //Debug.Log(LookupTable.GetPixel(rIdx, gIdx, bIdx));
        }

        LookupTable.Apply(updateMipmaps: false);
        AssetDatabase.CreateAsset(LookupTable, $"Assets/RGB_LookupTexture_Euclidean_New.asset");
        AssetDatabase.SaveAssetIfDirty(LookupTable);
    }

    // Update is called once per frame
    void Update()
    {
        
    }
}
