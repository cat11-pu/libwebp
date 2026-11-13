// Copyright 2026 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the COPYING file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS. All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
// -----------------------------------------------------------------------------
//
// Verification and benchmark tool for downsampled decoding
// (WebPDecodeRGBADownsampled() / WebPDecodeRGBDownsampled(), i.e. the
// 'rgba_scaling' decoder option).
//
// For each test bitstream and downsample factor, it checks that the
// downsampled decode is pixel-identical to:
//   full-size decode -> rescale with WebPRescaler
// and reports timing / memory comparisons, as well as the difference of
// the legacy YUV-space scaling path (use_scaling without rgba_scaling).
//
// Usage: downsample_check [file.webp]

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "src/utils/rescaler_utils.h"
#include "src/utils/utils.h"
#include "src/webp/decode.h"
#include "src/webp/encode.h"

static double NowMs(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

// Reference: rescale 'src' (w x h, 'bpp' bytes per pixel) to (sw x sh) with
// WebPRescaler, the same scaler used by the decoder.
static uint8_t* RefRescale(const uint8_t* src, int w, int h, int bpp, int sw,
                           int sh) {
  const int src_stride = w * bpp;
  const int dst_stride = sw * bpp;
  uint8_t* const dst = (uint8_t*)malloc((size_t)dst_stride * sh);
  const size_t work_size =
      (size_t)WebPRescalerWorkSize(w, h, sw, sh, bpp) + WEBP_ALIGN_CST;
  rescaler_t* const work = (rescaler_t*)malloc(work_size);
  WebPRescaler* const scaler = (WebPRescaler*)malloc(sizeof(*scaler));
  int y;
  if (dst == NULL || work == NULL || scaler == NULL ||
      !WebPRescalerInit(scaler, w, h, dst, sw, sh, dst_stride, bpp,
                        (rescaler_t*)WEBP_ALIGN(work))) {
    fprintf(stderr, "RefRescale: init failed\n");
    exit(1);
  }
  for (y = 0; y < h;) {
    y += WebPRescalerImport(scaler, h - y, src + (size_t)y * src_stride,
                            src_stride);
    WebPRescalerExport(scaler);
  }
  free(work);
  free(scaler);
  return dst;
}

typedef uint8_t* (*DecodeFunc)(const uint8_t*, size_t, int*, int*);
typedef uint8_t* (*DecodeDownsampledFunc)(const uint8_t*, size_t, int*, int*,
                                          int);

static int CompareBuffers(const char* name, const uint8_t* a, const uint8_t* b,
                          size_t size) {
  size_t i;
  size_t num_diff = 0;
  int max_diff = 0;
  for (i = 0; i < size; ++i) {
    const int d = abs((int)a[i] - (int)b[i]);
    if (d > 0) {
      ++num_diff;
      if (d > max_diff) max_diff = d;
    }
  }
  printf("  %-46s ", name);
  if (num_diff == 0) {
    printf("IDENTICAL (%zu bytes)\n", size);
    return 1;
  }
  printf("DIFFERS: %zu/%zu bytes, max diff %d\n", num_diff, size, max_diff);
  return 0;
}

// Decodes with the legacy YUV-space scaling path (use_scaling without the
// rgba_scaling option). Used to show that it does NOT match decode+rescale.
static uint8_t* DecodeLegacyScaled(const uint8_t* data, size_t data_size,
                                   int scaled_width, int scaled_height,
                                   WebPDecBuffer* keep) {
  WebPDecoderConfig config;
  if (!WebPInitDecoderConfig(&config)) return NULL;
  config.output.colorspace = MODE_RGBA;
  config.options.use_scaling = 1;
  config.options.scaled_width = scaled_width;
  config.options.scaled_height = scaled_height;
  if (WebPDecode(data, data_size, &config) != VP8_STATUS_OK) {
    WebPFreeDecBuffer(&config.output);
    return NULL;
  }
  *keep = config.output;
  return config.output.u.RGBA.rgba;
}

static int CheckOne(const char* tag, const uint8_t* data, size_t data_size,
                    int factor, DecodeFunc decode,
                    DecodeDownsampledFunc decode_downsampled, int bpp,
                    int check_legacy, int timing_iters) {
  int w = 0, h = 0, dw = 0, dh = 0;
  uint8_t* full = NULL;
  uint8_t* ref = NULL;
  uint8_t* down = NULL;
  int sw, sh;
  size_t num_bytes;
  int ok = 1;
  double t0, t_ref = 0.0, t_down = 0.0;
  int i;

  full = decode(data, data_size, &w, &h);
  if (full == NULL) {
    printf("  %-46s FULL DECODE FAILED\n", tag);
    return 0;
  }
  sw = (w + factor - 1) / factor;
  sh = (h + factor - 1) / factor;
  num_bytes = (size_t)sw * bpp * sh;

  // Correctness.
  ref = RefRescale(full, w, h, bpp, sw, sh);
  down = decode_downsampled(data, data_size, &dw, &dh, factor);
  if (down == NULL || dw != sw || dh != sh) {
    printf("  %-46s DOWNSAMPLED DECODE FAILED (got %dx%d, want %dx%d)\n", tag,
           dw, dh, sw, sh);
    ok = 0;
    goto End;
  }
  ok &= CompareBuffers(tag, ref, down, num_bytes);

  // Show that the legacy YUV-space scaling path is not pixel-identical.
  if (check_legacy) {
    WebPDecBuffer keep;
    uint8_t* const legacy = DecodeLegacyScaled(data, data_size, sw, sh, &keep);
    if (legacy != NULL) {
      char name[128];
      snprintf(name, sizeof(name), "%s [legacy YUV-space scaling]", tag);
      CompareBuffers(name, ref, legacy, num_bytes);
      WebPFreeDecBuffer(&keep);
    }
  }

  // Timing.
  if (timing_iters > 0) {
    t0 = NowMs();
    for (i = 0; i < timing_iters; ++i) {
      int tw, th;
      uint8_t* const tf = decode(data, data_size, &tw, &th);
      uint8_t* const tr = RefRescale(tf, w, h, bpp, sw, sh);
      free(tr);
      WebPFree(tf);
    }
    t_ref = (NowMs() - t0) / timing_iters;
    t0 = NowMs();
    for (i = 0; i < timing_iters; ++i) {
      int tw, th;
      uint8_t* const td = decode_downsampled(data, data_size, &tw, &th, factor);
      WebPFree(td);
    }
    t_down = (NowMs() - t0) / timing_iters;
    printf("  %-46s decode+scale %7.3f ms | downsampled %7.3f ms "
           "(%.2fx faster)\n",
           "", t_ref, t_down, t_ref / t_down);
    printf("  %-46s peak RGBA output: %zu -> %zu bytes (%.1fx less)\n", "",
           (size_t)w * bpp * h, num_bytes,
           (double)((size_t)w * bpp * h) / num_bytes);
  }

 End:
  WebPFree(full);
  free(ref);
  WebPFree(down);
  return ok;
}

// Checks the incremental decoding API (WebPIDecode, fed in chunks) with the
// rgba_scaling option against the decode+rescale reference.
static int CheckIncremental(const char* tag, const uint8_t* data,
                            size_t data_size, int factor) {
  WebPDecoderConfig config;
  uint8_t* full = NULL;
  uint8_t* ref = NULL;
  WebPIDecoder* idec = NULL;
  size_t pos = 0;
  const size_t kChunk = 7;  // deliberately odd chunk size
  int w, h, sw, sh;
  int ok = 1;

  full = WebPDecodeRGBA(data, data_size, &w, &h);
  if (full == NULL || !WebPInitDecoderConfig(&config)) return 0;
  sw = (w + factor - 1) / factor;
  sh = (h + factor - 1) / factor;
  ref = RefRescale(full, w, h, 4, sw, sh);

  config.output.colorspace = MODE_RGBA;
  config.options.use_scaling = 1;
  config.options.rgba_scaling = 1;
  config.options.scaled_width = sw;
  config.options.scaled_height = sh;
  idec = WebPIDecode(NULL, 0, &config);
  if (idec == NULL) {
    ok = 0;
    goto End;
  }
  while (pos < data_size) {
    size_t size = data_size - pos;
    VP8StatusCode status;
    if (size > kChunk) size = kChunk;
    status = WebPIAppend(idec, data + pos, size);
    if (status != VP8_STATUS_OK && status != VP8_STATUS_SUSPENDED) {
      ok = 0;
      goto End;
    }
    pos += size;
  }
  if (config.output.u.RGBA.rgba == NULL) {
    ok = 0;
    goto End;
  }
  {
    char name[128];
    snprintf(name, sizeof(name), "%s [incremental]", tag);
    ok = CompareBuffers(name, ref, config.output.u.RGBA.rgba,
                        (size_t)sw * 4 * sh);
  }

 End:
  WebPIDelete(idec);
  WebPFreeDecBuffer(&config.output);
  WebPFree(full);
  free(ref);
  return ok;
}

// Checks cropping combined with RGBA-space scaling through the advanced
// decoding API: decode -> crop -> rescale must match a single decode with
// use_cropping + use_scaling + rgba_scaling. The reference is a decode with
// the same cropping options (but no scaling), rescaled afterwards: for
// lossy bitstreams cropping happens in YUV space, so cropping the decoded
// RGB picture is not equivalent.
static int CheckCropScale(const char* tag, const uint8_t* data,
                          size_t data_size) {
  WebPDecoderConfig config;
  uint8_t* ref = NULL;
  const int crop_x = 6, crop_y = 4;  // snapped to even for lossy bitstreams
  int w = 0, h = 0;
  int crop_w, crop_h, sw, sh;
  int ok;

  if (!WebPGetInfo(data, data_size, &w, &h)) return 0;
  crop_w = (w - crop_x) * 2 / 3;
  crop_h = (h - crop_y) * 2 / 3;
  if (crop_w <= 0 || crop_h <= 0) {  // image too small for this test
    return 1;
  }
  sw = crop_w / 2;
  sh = crop_h / 2;
  // Reference: decode with cropping, then rescale.
  if (!WebPInitDecoderConfig(&config)) return 0;
  config.output.colorspace = MODE_BGRA;
  config.options.use_cropping = 1;
  config.options.crop_left = crop_x;
  config.options.crop_top = crop_y;
  config.options.crop_width = crop_w;
  config.options.crop_height = crop_h;
  if (WebPDecode(data, data_size, &config) != VP8_STATUS_OK) {
    ok = 0;
    goto End;
  }
  ref = RefRescale(config.output.u.RGBA.rgba, crop_w, crop_h, 4, sw, sh);
  WebPFreeDecBuffer(&config.output);

  // Single decode with cropping + RGBA-space scaling.
  if (!WebPInitDecoderConfig(&config)) {
    ok = 0;
    goto End;
  }
  config.output.colorspace = MODE_BGRA;
  config.options.use_cropping = 1;
  config.options.crop_left = crop_x;
  config.options.crop_top = crop_y;
  config.options.crop_width = crop_w;
  config.options.crop_height = crop_h;
  config.options.use_scaling = 1;
  config.options.rgba_scaling = 1;
  config.options.scaled_width = sw;
  config.options.scaled_height = sh;
  if (!WebPValidateDecoderConfig(&config)) {
    printf("  %-46s INVALID CONFIG\n", tag);
    ok = 0;
    goto End;
  }
  if (WebPDecode(data, data_size, &config) != VP8_STATUS_OK) {
    printf("  %-46s DECODE FAILED\n", tag);
    ok = 0;
    goto End;
  }
  ok = CompareBuffers(tag, ref, config.output.u.RGBA.rgba,
                      (size_t)sw * 4 * sh);

 End:
  WebPFreeDecBuffer(&config.output);
  free(ref);
  return ok;
}

// Generates a synthetic image with smooth gradients, sharp edges, noise and
// a varying alpha channel.
static void GenImage(uint8_t* rgba, int w, int h) {
  int x, y;
  unsigned int seed = 0x1234567u;
  for (y = 0; y < h; ++y) {
    for (x = 0; x < w; ++x) {
      uint8_t* const px = rgba + ((size_t)y * w + x) * 4;
      seed = seed * 1664525u + 1013904223u;
      px[0] = (uint8_t)((x * 255) / (w > 1 ? w - 1 : 1));
      px[1] = (uint8_t)((y * 255) / (h > 1 ? h - 1 : 1));
      px[2] = (uint8_t)(((x / 8) ^ (y / 8)) & 1 ? 255 : 0);  // checkerboard
      px[2] = (uint8_t)((px[2] + (seed >> 24)) / 2);        // + noise
      px[3] =
          (uint8_t)(64 + ((x + y) % 3) * 64 + ((x * y) % 127 > 63 ? 64 : 0));
    }
  }
}

static int CheckImage(const char* name, const uint8_t* rgba, int w, int h,
                      int timing_iters) {
  static const int kFactors[] = {2, 4};
  uint8_t* lossy = NULL;
  uint8_t* lossless = NULL;
  size_t lossy_size = 0, lossless_size = 0;
  int ok = 1;
  size_t f;

  lossy_size = WebPEncodeRGBA(rgba, w, h, w * 4, 75.0f, &lossy);
  lossless_size = WebPEncodeLosslessRGBA(rgba, w, h, w * 4, &lossless);
  if (lossy_size == 0 || lossless_size == 0) {
    printf("  %-46s ENCODE FAILED\n", name);
    ok = 0;
    goto End;
  }
  printf("%s: %dx%d (lossy: %zu bytes, lossless: %zu bytes)\n", name, w, h,
         lossy_size, lossless_size);
  for (f = 0; f < sizeof(kFactors) / sizeof(kFactors[0]); ++f) {
    const int factor = kFactors[f];
    char tag[128];
    snprintf(tag, sizeof(tag), "1/%d RGBA lossy", factor);
    ok &= CheckOne(tag, lossy, lossy_size, factor, WebPDecodeRGBA,
                   WebPDecodeRGBADownsampled, 4, /*check_legacy=*/1,
                   timing_iters);
    snprintf(tag, sizeof(tag), "1/%d RGBA lossless", factor);
    ok &= CheckOne(tag, lossless, lossless_size, factor, WebPDecodeRGBA,
                   WebPDecodeRGBADownsampled, 4, /*check_legacy=*/1,
                   timing_iters);
    snprintf(tag, sizeof(tag), "1/%d RGB lossy", factor);
    ok &= CheckOne(tag, lossy, lossy_size, factor, WebPDecodeRGB,
                   WebPDecodeRGBDownsampled, 3, /*check_legacy=*/0,
                   timing_iters);
    snprintf(tag, sizeof(tag), "1/%d RGB lossless", factor);
    ok &= CheckOne(tag, lossless, lossless_size, factor, WebPDecodeRGB,
                   WebPDecodeRGBDownsampled, 3, /*check_legacy=*/0,
                   timing_iters);
    snprintf(tag, sizeof(tag), "1/%d RGBA lossy", factor);
    ok &= CheckIncremental(tag, lossy, lossy_size, factor);
    snprintf(tag, sizeof(tag), "1/%d RGBA lossless", factor);
    ok &= CheckIncremental(tag, lossless, lossless_size, factor);
  }
  ok &= CheckCropScale("crop+1/2 BGRA lossy", lossy, lossy_size);
  ok &= CheckCropScale("crop+1/2 BGRA lossless", lossless, lossless_size);

 End:
  WebPFree(lossy);
  WebPFree(lossless);
  return ok;
}

static int CheckFile(const char* path, int timing_iters) {
  FILE* const f = fopen(path, "rb");
  uint8_t* data;
  long size;
  int ok = 1;
  size_t k;
  if (f == NULL) {
    fprintf(stderr, "cannot open %s\n", path);
    return 0;
  }
  fseek(f, 0, SEEK_END);
  size = ftell(f);
  fseek(f, 0, SEEK_SET);
  data = (uint8_t*)malloc(size);
  if (data == NULL || fread(data, 1, size, f) != (size_t)size) {
    fprintf(stderr, "cannot read %s\n", path);
    fclose(f);
    free(data);
    return 0;
  }
  fclose(f);
  printf("%s: %ld bytes\n", path, size);
  for (k = 2; k <= 4; k *= 2) {
    char tag[128];
    snprintf(tag, sizeof(tag), "1/%zu RGBA file", k);
    ok &= CheckOne(tag, data, size, (int)k, WebPDecodeRGBA,
                   WebPDecodeRGBADownsampled, 4, /*check_legacy=*/1,
                   timing_iters);
    snprintf(tag, sizeof(tag), "1/%zu RGBA file", k);
    ok &= CheckIncremental(tag, data, size, (int)k);
  }
  ok &= CheckCropScale("crop+1/2 BGRA file", data, size);
  free(data);
  return ok;
}

int main(int argc, char* argv[]) {
  static const struct {
    int w, h;
  } kSizes[] = {
      {1, 1}, {3, 2}, {16, 16}, {17, 1}, {5, 33}, {635, 477}, {640, 480},
  };
  int ok = 1;
  size_t i;
  for (i = 0; i < sizeof(kSizes) / sizeof(kSizes[0]); ++i) {
    const int w = kSizes[i].w;
    const int h = kSizes[i].h;
    uint8_t* const rgba = (uint8_t*)malloc((size_t)w * h * 4);
    char name[64];
    const int big = (w * h > 100000);
    GenImage(rgba, w, h);
    snprintf(name, sizeof(name), "synthetic %dx%d", w, h);
    ok &= CheckImage(name, rgba, w, h, /*timing_iters=*/big ? 30 : 0);
    free(rgba);
  }
  for (i = 1; i < (size_t)argc; ++i) {
    ok &= CheckFile(argv[i], /*timing_iters=*/30);
  }
  printf(ok ? "\nALL CHECKS PASSED\n" : "\nSOME CHECKS FAILED\n");
  return ok ? 0 : 1;
}
