// Copyright 2026 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the COPYING file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS. All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
// -----------------------------------------------------------------------------
//
// AVX-512 variant of methods for the lossless decoder.

#include "src/dsp/dsp.h"

#if defined(WEBP_USE_AVX512)

#include <immintrin.h>
#include <stddef.h>

#include "src/dsp/lossless.h"
#include "src/webp/format_constants.h"
#include "src/webp/types.h"

//------------------------------------------------------------------------------
// Predictor Transform

// Predictor0: ARGB_BLACK.
static void PredictorAdd0_AVX512(const uint32_t* in,
                                 const uint32_t* upper, int num_pixels,
                                 uint32_t* WEBP_RESTRICT out) {
  int i;
  const __m512i black = _mm512_set1_epi32((int)ARGB_BLACK);
  for (i = 0; i + 16 <= num_pixels; i += 16) {
    const __m512i src = _mm512_loadu_si512((const __m512i*)&in[i]);
    const __m512i res = _mm512_add_epi8(src, black);
    _mm512_storeu_si512((__m512i*)&out[i], res);
  }
  if (i != num_pixels) {
    VP8LPredictorsAdd_SSE[0](in + i, NULL, num_pixels - i, out + i);
  }
  (void)upper;
}

// Predictor2: Top. Predictor3: Top-right. Predictor4: Top-left.
#define GENERATE_PREDICTOR_1(X, IN)                                         \
  static void PredictorAdd##X##_AVX512(                                     \
      const uint32_t* in, const uint32_t* upper, int num_pixels,           \
      uint32_t* WEBP_RESTRICT out) {                                        \
    int i;                                                                  \
    for (i = 0; i + 16 <= num_pixels; i += 16) {                           \
      const __m512i src = _mm512_loadu_si512((const __m512i*)&in[i]);       \
      const __m512i other = _mm512_loadu_si512((const __m512i*)&(IN));      \
      const __m512i res = _mm512_add_epi8(src, other);                      \
      _mm512_storeu_si512((__m512i*)&out[i], res);                          \
    }                                                                       \
    if (i != num_pixels) {                                                  \
      VP8LPredictorsAdd_SSE[X](in + i, upper + i, num_pixels - i,           \
                               out + i);                                    \
    }                                                                       \
  }

GENERATE_PREDICTOR_1(2, upper[i])
GENERATE_PREDICTOR_1(3, upper[i + 1])
GENERATE_PREDICTOR_1(4, upper[i - 1])
#undef GENERATE_PREDICTOR_1

// Predictor12: ClampedAddSubtractFull.
#define DO_PRED12(DIFF)                                                 \
  do {                                                                  \
    const __m512i all = _mm512_adds_epi16(L, (DIFF));                   \
    const __m512i alls = _mm512_packus_epi16(all, all);                 \
    const __m512i res = _mm512_add_epi8(src, alls);                     \
    acc = _mm512_alignr_epi32(res, acc, 1);                             \
    L = _mm512_unpacklo_epi8(res, zero);                                \
    (DIFF) = _mm512_alignr_epi32((DIFF), (DIFF), 2);                    \
    src = _mm512_alignr_epi32(src, src, 1);                             \
  } while (0)

static void PredictorAdd12_AVX512(const uint32_t* in,
                                  const uint32_t* upper, int num_pixels,
                                  uint32_t* WEBP_RESTRICT out) {
  int i;
  const __m512i zero = _mm512_setzero_si512();
  __m512i L = _mm512_unpacklo_epi8(
      _mm512_zextsi128_si512(_mm_cvtsi32_si128((int)out[-1])), zero);
  for (i = 0; i + 16 <= num_pixels; i += 16) {
    __m512i src = _mm512_loadu_si512((const __m512i*)&in[i]);
    const __m512i T = _mm512_loadu_si512((const __m512i*)&upper[i]);
    const __m512i TL = _mm512_loadu_si512((const __m512i*)&upper[i - 1]);
    __m512i diff_lo =
        _mm512_sub_epi16(_mm512_unpacklo_epi8(T, zero),
                         _mm512_unpacklo_epi8(TL, zero));
    __m512i diff_hi =
        _mm512_sub_epi16(_mm512_unpackhi_epi8(T, zero),
                         _mm512_unpackhi_epi8(TL, zero));
    __m512i acc = zero;
    DO_PRED12(diff_lo);
    DO_PRED12(diff_lo);
    DO_PRED12(diff_hi);
    DO_PRED12(diff_hi);
    DO_PRED12(diff_lo);
    DO_PRED12(diff_lo);
    DO_PRED12(diff_hi);
    DO_PRED12(diff_hi);
    DO_PRED12(diff_lo);
    DO_PRED12(diff_lo);
    DO_PRED12(diff_hi);
    DO_PRED12(diff_hi);
    DO_PRED12(diff_lo);
    DO_PRED12(diff_lo);
    DO_PRED12(diff_hi);
    DO_PRED12(diff_hi);
    _mm512_storeu_si512((__m512i*)&out[i], acc);
  }
  if (i != num_pixels) {
    VP8LPredictorsAdd_SSE[12](in + i, upper + i, num_pixels - i, out + i);
  }
}
#undef DO_PRED12

//------------------------------------------------------------------------------
// Subtract-Green Transform

static void AddGreenToBlueAndRed_AVX512(const uint32_t* const src,
                                        int num_pixels, uint32_t* dst) {
  int i;
  const __m512i shuffle = _mm512_set_epi8(
      -1, 13, -1, 13, -1, 9, -1, 9, -1, 5, -1, 5, -1, 1, -1, 1,
      -1, 13, -1, 13, -1, 9, -1, 9, -1, 5, -1, 5, -1, 1, -1, 1,
      -1, 13, -1, 13, -1, 9, -1, 9, -1, 5, -1, 5, -1, 1, -1, 1,
      -1, 13, -1, 13, -1, 9, -1, 9, -1, 5, -1, 5, -1, 1, -1, 1);
  for (i = 0; i + 16 <= num_pixels; i += 16) {
    const __m512i in = _mm512_loadu_si512((const __m512i*)&src[i]);
    const __m512i green = _mm512_shuffle_epi8(in, shuffle);
    const __m512i out = _mm512_add_epi8(in, green);
    _mm512_storeu_si512((__m512i*)&dst[i], out);
  }
  if (i != num_pixels) {
    VP8LAddGreenToBlueAndRed_SSE(src + i, num_pixels - i, dst + i);
  }
}

//------------------------------------------------------------------------------
// Color Transform

static void TransformColorInverse_AVX512(const VP8LMultipliers* const m,
                                         const uint32_t* const src,
                                         int num_pixels, uint32_t* dst) {
// sign-extended multiplying constants, pre-shifted by 5.
#define CST(X) (((int16_t)(m->X << 8)) >> 5)
  const __m512i mults_rb = _mm512_set1_epi32(
      (int)((uint32_t)CST(green_to_red) << 16 |
            (CST(green_to_blue) & 0xffff)));
  const __m512i mults_b2 = _mm512_set1_epi32(CST(red_to_blue));
#undef CST
  const __m512i mask_ag = _mm512_set1_epi32((int)0xff00ff00);
  const __m512i perm1 = _mm512_set_epi8(
      13, -1, 13, -1, 9, -1, 9, -1, 5, -1, 5, -1, 1, -1, 1, -1,
      13, -1, 13, -1, 9, -1, 9, -1, 5, -1, 5, -1, 1, -1, 1, -1,
      13, -1, 13, -1, 9, -1, 9, -1, 5, -1, 5, -1, 1, -1, 1, -1,
      13, -1, 13, -1, 9, -1, 9, -1, 5, -1, 5, -1, 1, -1, 1, -1);
  const __m512i perm2 = _mm512_set_epi8(
      -1, -1, 14, -1, -1, -1, 10, -1, -1, -1, 6, -1, -1, -1, 2, -1,
      -1, -1, 14, -1, -1, -1, 10, -1, -1, -1, 6, -1, -1, -1, 2, -1,
      -1, -1, 14, -1, -1, -1, 10, -1, -1, -1, 6, -1, -1, -1, 2, -1,
      -1, -1, 14, -1, -1, -1, 10, -1, -1, -1, 6, -1, -1, -1, 2, -1);
  const __mmask64 keep_ag = _mm512_movepi8_mask(mask_ag);
  int i;
  for (i = 0; i + 16 <= num_pixels; i += 16) {
    const __m512i A = _mm512_loadu_si512((const __m512i*)(src + i));
    const __m512i B = _mm512_shuffle_epi8(A, perm1);
    const __m512i C = _mm512_mulhi_epi16(B, mults_rb);
    const __m512i D = _mm512_add_epi8(A, C);
    const __m512i E = _mm512_shuffle_epi8(D, perm2);
    const __m512i F = _mm512_mulhi_epi16(E, mults_b2);
    const __m512i G = _mm512_add_epi8(D, F);
    const __m512i out = _mm512_mask_blend_epi8(keep_ag, G, A);
    _mm512_storeu_si512((__m512i*)&dst[i], out);
  }
  if (i != num_pixels) {
    VP8LTransformColorInverse_SSE(m, src + i, num_pixels - i, dst + i);
  }
}

//------------------------------------------------------------------------------
// Entry point

extern void VP8LDspInitAVX512(void);

WEBP_TSAN_IGNORE_FUNCTION void VP8LDspInitAVX512(void) {
  VP8LPredictorsAdd[0] = PredictorAdd0_AVX512;
  VP8LPredictorsAdd[2] = PredictorAdd2_AVX512;
  VP8LPredictorsAdd[3] = PredictorAdd3_AVX512;
  VP8LPredictorsAdd[4] = PredictorAdd4_AVX512;
  // These predictors are strictly sequential. The compiler's scalar loop
  // beats the per-pixel SSE helpers on modern wide cores.
  VP8LPredictorsAdd[5] = VP8LPredictorsAdd_C[5];
  VP8LPredictorsAdd[6] = VP8LPredictorsAdd_C[6];
  VP8LPredictorsAdd[7] = VP8LPredictorsAdd_C[7];
  VP8LPredictorsAdd[12] = PredictorAdd12_AVX512;
  VP8LPredictorsAdd[13] = VP8LPredictorsAdd_C[13];
  VP8LAddGreenToBlueAndRed = AddGreenToBlueAndRed_AVX512;
  VP8LTransformColorInverse = TransformColorInverse_AVX512;
}

#else  // !WEBP_USE_AVX512

WEBP_DSP_INIT_STUB(VP8LDspInitAVX512)

#endif  // WEBP_USE_AVX512
