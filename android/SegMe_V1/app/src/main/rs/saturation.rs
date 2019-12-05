/*
 * Copyright (C) 2014 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma version(1)
#pragma rs java_package_name(com.example.android.tflitecamerademo)
#pragma rs_fp_relaxed
#include "rs_graphics.rsh"

const static float3 gMonoMult = {0.299f, 0.587f, 0.114f};

float saturationValue = 0.f;

static inline float smoothstep(float edge0, float edge1, float x)
{
    float value = clamp((x - edge0) / (edge1 - edge0), 0.0f, 1.0f);
    return value * value * (3.0f - 2.0f * value);
}

/*
 * RenderScript kernel that performs smooth and merge operations.
 */

 rs_allocation fgd_alloc;
 rs_allocation mask_alloc;

 uchar4 __attribute__((kernel)) saturation(uchar4 in, uint32_t x, uint32_t y)
 {

     uchar4 i1 = rsGetElementAt_uchar4(fgd_alloc, x, y);
     uchar4 i2 = rsGetElementAt_uchar4(mask_alloc, x, y);

     float4 bgd = rsUnpackColor8888(in);
     float4 fgd = rsUnpackColor8888(i1);
     float4 msk = rsUnpackColor8888(i2);

     float val = smoothstep(0.35,0.45,msk.r);

     float4 out = mix(bgd, fgd, msk);
     return rsPackColorTo8888(out);
 }


