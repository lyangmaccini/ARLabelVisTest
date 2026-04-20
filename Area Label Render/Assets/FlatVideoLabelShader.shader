Shader "Unlit/FlatVideoLabelShader"
{
    Properties
    {
        _MainTex ("Video Frame", 2D) = "white" {}
        _LabelMask ("Label Mask", 2D) = "black" {}

        _ColorMethod ("Color Method", Int) = 4
        _GranularityMethod ("Granularity Method", Int) = 0
        _OpacityLevel ("Label Opacity Level", Range(0, 1)) = 1.0
        _EnableOutline ("Enable Outline", Int) = 1

        _Background_sum_r ("Background_sum_r", Range(0,1)) = 0.1
        _Background_sum_g ("Background_sum_g", Range(0,1)) = 0.1
        _Background_sum_b ("Background_sum_b", Range(0,1)) = 0.1

        _CIELAB_LookupTable ("CIELAB Lookup Table", 3D) = "white" {}
    }

    SubShader
    {
        Tags { "Queue"="Transparent" "RenderType"="Transparent" "IgnoreProjector"="True" }
        ZWrite Off
        Blend SrcAlpha OneMinusSrcAlpha
        Cull Off
        LOD 100

        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #include "UnityCG.cginc"

            sampler2D _MainTex;
            sampler2D _LabelMask;
            float4 _MainTex_ST;

            int _ColorMethod;
            int _GranularityMethod;
            float _OpacityLevel;
            int _EnableOutline;

            float _Background_sum_r;
            float _Background_sum_g;
            float _Background_sum_b;

            sampler3D _CIELAB_LookupTable;

            struct appdata
            {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
            };

            struct v2f
            {
                float4 pos : SV_POSITION;
                float2 uv : TEXCOORD0;
            };

            v2f vert(appdata v)
            {
                v2f o;
                o.pos = UnityObjectToClipPos(v.vertex);
                o.uv = TRANSFORM_TEX(v.uv, _MainTex);
                return o;
            }

            // ---------------------------------------------------------------
            // Color conversion helpers (carried over from original shader)
            // ---------------------------------------------------------------

            float4 RGB2HSV(float4 rgb)
            {
                float R = rgb.r;
                float G = rgb.g;
                float B = rgb.b;
                float var_Min = min(R, min(G, B));
                float var_Max = max(R, max(G, B));
                float del_Max = var_Max - var_Min;
                float H = 0, S = 0;
                float V = var_Max * 100;
                if (del_Max != 0)
                {
                    S = (del_Max / var_Max) * 100;
                    if      (R == var_Max) H = (60 * ((G - B) / del_Max) + 360) % 360;
                    else if (G == var_Max) H = (60 * ((B - R) / del_Max) + 120) % 360;
                    else if (B == var_Max) H = (60 * ((R - G) / del_Max) + 240) % 360;
                }
                return float4(H, S, V, rgb.a);
            }

            float4 HSV2RGB(float4 hsv)
            {
                float H = hsv[0], S = hsv[1], V = hsv[2];
                float s = S / 100, v = V / 100;
                float C = s * v;
                float X = C * (1 - abs((H / 60.0) % 2 - 1));
                float m = v - C;
                float r = 0, g = 0, b = 0;
                if      (H >= 0   && H < 60)  { r = C; g = X; }
                else if (H >= 60  && H < 120) { r = X; g = C; }
                else if (H >= 120 && H < 180) { g = C; b = X; }
                else if (H >= 180 && H < 240) { g = X; b = C; }
                else if (H >= 240 && H < 300) { r = X; b = C; }
                else                          { r = C; b = X; }
                return float4(r + m, g + m, b + m, hsv.a);
            }

            float4 RGB2LAB(float4 RGB)
            {
                float R = RGB.r, G = RGB.g, B = RGB.b;
                float Xr = 95.047, Yr = 100.0, Zr = 108.883;
                float var_R = R, var_G = G, var_B = B;

                if (R > 0.04045) var_R = pow((var_R + 0.055) / 1.055, 2.4);
                else             var_R = var_R / 12.92;
                if (G > 0.04045) var_G = pow((var_G + 0.055) / 1.055, 2.4);
                else             var_G = var_G / 12.92;
                if (B > 0.04045) var_B = pow((var_B + 0.055) / 1.055, 2.4);
                else             var_B = var_B / 12.92;

                var_R *= 100; var_G *= 100; var_B *= 100;

                float X = var_R * 0.4124 + var_G * 0.3576 + var_B * 0.1805;
                float Y = var_R * 0.2126 + var_G * 0.7152 + var_B * 0.0722;
                float Z = var_R * 0.0193 + var_G * 0.1192 + var_B * 0.9505;

                float var_X = X / Xr, var_Y = Y / Yr, var_Z = Z / Zr;

                if (var_X > 0.008856) var_X = pow(var_X, 1/3.0); else var_X = (7.787 * var_X) + (16.0 / 116.0);
                if (var_Y > 0.008856) var_Y = pow(var_Y, 1/3.0); else var_Y = (7.787 * var_Y) + (16.0 / 116.0);
                if (var_Z > 0.008856) var_Z = pow(var_Z, 1/3.0); else var_Z = (7.787 * var_Z) + (16.0 / 116.0);

                return float4((116.0 * var_Y) - 16, 500.0 * (var_X - var_Y), 200.0 * (var_Y - var_Z), RGB.a);
            }

            float4 LAB2RGB(float4 LAB)
            {
                float L = LAB[0], A = LAB[1], B = LAB[2];
                float Xr = 95.047, Yr = 100.0, Zr = 108.883;

                float var_Y = (L + 16.0) / 116.0;
                float var_X = A / 500 + var_Y;
                float var_Z = var_Y - B / 200.0;

                if (pow(var_Y, 3) > 0.008856) var_Y = pow(var_Y, 3.0); else var_Y = (var_Y - 16.0/116.0) / 7.787;
                if (pow(var_X, 3) > 0.008856) var_X = pow(var_X, 3.0); else var_X = (var_X - 16.0/116.0) / 7.787;
                if (pow(var_Z, 3) > 0.008856) var_Z = pow(var_Z, 3.0); else var_Z = (var_Z - 16.0/116.0) / 7.787;

                float X = var_X * Xr / 100.0;
                float Y = var_Y * Yr / 100.0;
                float Z = var_Z * Zr / 100.0;

                float var_R = X *  3.2406 + Y * -1.5372 + Z * -0.4986;
                float var_G = X * -0.9689 + Y *  1.8758 + Z *  0.0415;
                float var_B = X *  0.0557 + Y * -0.2040 + Z *  1.0570;

                if (var_R > 0.0031308) var_R = 1.055 * pow(var_R, 1/2.4) - 0.055; else var_R = 12.92 * var_R;
                if (var_G > 0.0031308) var_G = 1.055 * pow(var_G, 1/2.4) - 0.055; else var_G = 12.92 * var_G;
                if (var_B > 0.0031308) var_B = 1.055 * pow(var_B, 1/2.4) - 0.055; else var_B = 12.92 * var_B;

                return float4(clamp(var_R, 0, 1), clamp(var_G, 0, 1), clamp(var_B, 0, 1), LAB.a);
            }

            // ---------------------------------------------------------------
            // Label color function — same logic as original function_f
            // ---------------------------------------------------------------

            float4 function_f(int method, float4 bgSample)
            {
                float4 col = float4(0, 0, 0, 0);

                // Palette (most contrasting of 8 corners of RGB cube)
                if (method == 1)
                {
                    float maxDistSq = -1;
                    float4 paletteCol = float4(0, 0, 0, 1);
                    for (int i = 0; i < 8; i++)
                    {
                        if (i == 2) continue;
                        float r = i & 1;
                        float g = (i & 2) >> 1;
                        float b = (i & 4) >> 2;
                        float distSq = (bgSample.r - r) * (bgSample.r - r)
                                     + (bgSample.g - g) * (bgSample.g - g)
                                     + (bgSample.b - b) * (bgSample.b - b);
                        if (distSq > maxDistSq) { maxDistSq = distSq; paletteCol = float4(r, g, b, 1); }
                    }
                    col = paletteCol;
                }
                // RGB inversion
                else if (method == 2)
                {
                    col = float4(1.0 - bgSample.r, 1.0 - bgSample.g, 1.0 - bgSample.b, 1.0);
                }
                // HSV inversion
                else if (method == 3)
                {
                    float4 hsv = RGB2HSV(bgSample);
                    hsv[0] = (hsv[0] + 180) % 360;
                    hsv[2] = 100 - hsv[2];
                    col = HSV2RGB(hsv);
                }
                // CIELAB inversion (main method)
                else if (method == 4)
                {
                    col = tex3D(_CIELAB_LookupTable, bgSample.rgb);
                    if (_GranularityMethod == 0)
                    {
                        float4 bgSampleAsLAB   = RGB2LAB(bgSample);
                        float4 backgroundAvg   = float4(_Background_sum_r, _Background_sum_g, _Background_sum_b, 1);
                        float4 backgroundAvgLAB = RGB2LAB(backgroundAvg);
                        float4 avgMaxDistLAB   = RGB2LAB(tex3D(_CIELAB_LookupTable, backgroundAvg.rgb));
                        float4 diff = float4(
                            bgSampleAsLAB[0] - backgroundAvgLAB[0],
                            bgSampleAsLAB[1] - backgroundAvgLAB[1],
                            bgSampleAsLAB[2] - backgroundAvgLAB[2], 1);
                        col = LAB2RGB(float4(
                            avgMaxDistLAB[0] + diff[0],
                            avgMaxDistLAB[1] + diff[1],
                            avgMaxDistLAB[2] + diff[2], 1));
                    }
                }
                // Green label
                else if (method == 5)
                {
                    col = float4(0.0, 1.0, 0.0, 1.0);
                }
                // No label
                else if (method == 6)
                {
                    col = float4(0.0, 0.0, 0.0, 0.0);
                }
                // Blue/gray label
                else if (method == 7)
                {
                    col = float4(0.0, 0.0, 0.7, 1.0);
                }

                return col;
            }

            // ---------------------------------------------------------------
            // Fragment shader
            // ---------------------------------------------------------------

            fixed4 frag(v2f i) : SV_Target
            {
                // Sample the video frame and the static label mask
                fixed4 videoCol  = tex2D(_MainTex,   i.uv);
                fixed4 labelMask = tex2D(_LabelMask, i.uv);

                // Start with the raw video frame as the output
                fixed4 col = videoCol;

                // Determine the background sample to use for color processing
                float4 bgSample = videoCol;
                if (_GranularityMethod != 0)
                {
                    // Per-area or per-background: use precomputed average instead of per-pixel value
                    bgSample = float4(_Background_sum_r, _Background_sum_g, _Background_sum_b, 1);
                }

                // If this pixel is inside the label mask, apply the color method
                if (labelMask.r > 0.5)
                {
                    col = function_f(_ColorMethod, bgSample);
                    col[3] = (_ColorMethod == 6) ? 0.0 : _OpacityLevel;

                    // Sobel outline
                    if (_EnableOutline == 1)
                    {
                        float2 delta = float2(0.002, 0.002);
                        float4 hr = float4(0,0,0,0);
                        float4 vt = float4(0,0,0,0);
                        float filter[3][3] = { {-1,0,1},{-2,0,2},{-1,0,1} };

                        for (int x = -1; x <= 1; x++)
                        {
                            for (int y = -1; y <= 1; y++)
                            {
                                float2 offset = float2(x, y) * delta;
                                float4 pix = tex2D(_LabelMask, i.uv + offset);
                                float4 val = (pix.r > 0.5) ? float4(1,1,1,1) : float4(0,0,0,0);
                                hr += val * filter[x + 1][y + 1];
                                vt += val * filter[y + 1][x + 1];
                            }
                        }

                        float edges = sqrt(dot(hr, hr) + dot(vt, vt));
                        if (edges > 0)
                        {
                            col = (col.r + col.g + col.b < 0.5)
                                ? float4(1, 1, 1, _OpacityLevel)
                                : float4(0, 0, 0, _OpacityLevel);
                        }
                    }
                }

                return col;
            }
            ENDCG
        }
    }
}
