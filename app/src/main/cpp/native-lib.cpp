#include <jni.h>
#include <android/bitmap.h>
#include <algorithm>
#include <cstring>

// Clamps a float in [0, 1] to a uint8 in [0, 255].
inline uint8_t floatToUint8(float val) {
    float scaled = val * 255.0f;
    if (scaled <= 0.0f) return 0;
    if (scaled >= 255.0f) return 255;
    return static_cast<uint8_t>(scaled + 0.5f);
}

// Writes bitmap pixels into a Kotlin FloatArray in NHWC order, normalized to [0, 1].
// The array is the one later passed to TensorBuffer.writeFloat(), since the
// CompiledModel Kotlin API does not expose a raw ByteBuffer write path.
extern "C" JNIEXPORT void JNICALL
Java_eu_kanade_tachiyomi_util_upscale_NativePixelOps_writeBitmapToArrayNHWC(
        JNIEnv* env,
        jobject /* this */,
        jobject bitmap,
        jfloatArray outArray,
        jint arrayPixelOffset,
        jint tileSize
) {
    AndroidBitmapInfo info;
    void* pixels = nullptr;

    if (AndroidBitmap_getInfo(env, bitmap, &info) < 0 || info.format != ANDROID_BITMAP_FORMAT_RGBA_8888) {
        return;
    }
    if (AndroidBitmap_lockPixels(env, bitmap, &pixels) < 0) {
        return;
    }

    auto* src = static_cast<const uint8_t*>(pixels);
    jfloat* dstArray = env->GetFloatArrayElements(outArray, nullptr);
    float* dst = dstArray + (arrayPixelOffset * 3);

    int totalPixels = tileSize * tileSize;
    for (int i = 0; i < totalPixels; ++i) {
        int srcIdx = i * 4;
        int dstIdx = i * 3;
        dst[dstIdx + 0] = src[srcIdx + 0] / 255.0f; // R
        dst[dstIdx + 1] = src[srcIdx + 1] / 255.0f; // G
        dst[dstIdx + 2] = src[srcIdx + 2] / 255.0f; // B
    }

    env->ReleaseFloatArrayElements(outArray, dstArray, 0);
    AndroidBitmap_unlockPixels(env, bitmap);
}

// Same as writeBitmapToArrayNHWC but writes planar (channel-major) order for NCHW models.
extern "C" JNIEXPORT void JNICALL
Java_eu_kanade_tachiyomi_util_upscale_NativePixelOps_writeBitmapToArrayNCHW(
        JNIEnv* env,
        jobject /* this */,
        jobject bitmap,
        jfloatArray outArray,
        jint arrayPixelOffset,
        jint tileSize
) {
    AndroidBitmapInfo info;
    void* pixels = nullptr;

    if (AndroidBitmap_getInfo(env, bitmap, &info) < 0 || info.format != ANDROID_BITMAP_FORMAT_RGBA_8888) {
        return;
    }
    if (AndroidBitmap_lockPixels(env, bitmap, &pixels) < 0) {
        return;
    }

    auto* src = static_cast<const uint8_t*>(pixels);
    jfloat* dstArray = env->GetFloatArrayElements(outArray, nullptr);
    float* dst = dstArray + (arrayPixelOffset * 3);

    int totalPixels = tileSize * tileSize;
    float* rPlane = dst;
    float* gPlane = dst + totalPixels;
    float* bPlane = dst + totalPixels * 2;

    for (int i = 0; i < totalPixels; ++i) {
        int srcIdx = i * 4;
        rPlane[i] = src[srcIdx + 0] / 255.0f;
        gPlane[i] = src[srcIdx + 1] / 255.0f;
        bPlane[i] = src[srcIdx + 2] / 255.0f;
    }

    env->ReleaseFloatArrayElements(outArray, dstArray, 0);
    AndroidBitmap_unlockPixels(env, bitmap);
}

// Reads model output back from a Kotlin FloatArray (as returned by TensorBuffer.readFloat())
// into a reusable output Bitmap, NHWC order.
extern "C" JNIEXPORT void JNICALL
Java_eu_kanade_tachiyomi_util_upscale_NativePixelOps_readArrayToBitmapNHWC(
        JNIEnv* env,
        jobject /* this */,
        jfloatArray inArray,
        jint arrayPixelOffset,
        jobject targetBitmap,
        jint outSize
) {
    AndroidBitmapInfo info;
    void* pixels = nullptr;

    if (AndroidBitmap_getInfo(env, targetBitmap, &info) < 0 || info.format != ANDROID_BITMAP_FORMAT_RGBA_8888) {
        return;
    }
    if (AndroidBitmap_lockPixels(env, targetBitmap, &pixels) < 0) {
        return;
    }

    auto* dst = static_cast<uint8_t*>(pixels);
    jfloat* srcArray = env->GetFloatArrayElements(inArray, nullptr);
    const float* src = srcArray + (arrayPixelOffset * 3);

    int totalPixels = outSize * outSize;
    for (int i = 0; i < totalPixels; ++i) {
        int srcIdx = i * 3;
        int dstIdx = i * 4;
        dst[dstIdx + 0] = floatToUint8(src[srcIdx + 0]); // R
        dst[dstIdx + 1] = floatToUint8(src[srcIdx + 1]); // G
        dst[dstIdx + 2] = floatToUint8(src[srcIdx + 2]); // B
        dst[dstIdx + 3] = 255;                            // Alpha
    }

    // JNI_ABORT: we only read from inArray, no need to copy unmodified data back.
    env->ReleaseFloatArrayElements(inArray, srcArray, JNI_ABORT);
    AndroidBitmap_unlockPixels(env, targetBitmap);
}

// Same as readArrayToBitmapNHWC but reads planar (channel-major) order for NCHW models.
extern "C" JNIEXPORT void JNICALL
Java_eu_kanade_tachiyomi_util_upscale_NativePixelOps_readArrayToBitmapNCHW(
        JNIEnv* env,
        jobject /* this */,
        jfloatArray inArray,
        jint arrayPixelOffset,
        jobject targetBitmap,
        jint outSize
) {
    AndroidBitmapInfo info;
    void* pixels = nullptr;

    if (AndroidBitmap_getInfo(env, targetBitmap, &info) < 0 || info.format != ANDROID_BITMAP_FORMAT_RGBA_8888) {
        return;
    }
    if (AndroidBitmap_lockPixels(env, targetBitmap, &pixels) < 0) {
        return;
    }

    auto* dst = static_cast<uint8_t*>(pixels);
    jfloat* srcArray = env->GetFloatArrayElements(inArray, nullptr);
    const float* src = srcArray + (arrayPixelOffset * 3);

    int totalPixels = outSize * outSize;
    const float* rPlane = src;
    const float* gPlane = src + totalPixels;
    const float* bPlane = src + totalPixels * 2;

    for (int i = 0; i < totalPixels; ++i) {
        int dstIdx = i * 4;
        dst[dstIdx + 0] = floatToUint8(rPlane[i]);
        dst[dstIdx + 1] = floatToUint8(gPlane[i]);
        dst[dstIdx + 2] = floatToUint8(bPlane[i]);
        dst[dstIdx + 3] = 255;
    }

    env->ReleaseFloatArrayElements(inArray, srcArray, JNI_ABORT);
    AndroidBitmap_unlockPixels(env, targetBitmap);
}

// Reads the model's raw pre-PixelShuffle output (48 = 3*scale^2 channels) from 'outArray',
// applies PixelShuffle and the residual skip connection (nearest-neighbor upsample of the
// original tile from 'inArray', added to the shuffled detail) directly into 'targetBitmap'.
// Both PixelShuffle and the skip connection were cut from the traced PyTorch forward before
// export (litert_torch/ML Drift could not compile the resulting 6D Reshape/Transpose pair for
// GPU), so this function reproduces both steps on CPU instead.
extern "C" JNIEXPORT void JNICALL
Java_eu_kanade_tachiyomi_util_upscale_NativePixelOps_readArrayToBitmapPixelShuffle(
        JNIEnv* env,
        jobject /* this */,
        jfloatArray outArray,
        jfloatArray inArray,
        jint arrayPixelOffset,
        jobject targetBitmap,
        jint inTileSize,
        jint scale,
        jboolean isInputNhwc,
        jboolean isOutputNhwc
) {
    AndroidBitmapInfo info;
    void* pixels = nullptr;

    if (AndroidBitmap_getInfo(env, targetBitmap, &info) < 0 || info.format != ANDROID_BITMAP_FORMAT_RGBA_8888) return;
    if (AndroidBitmap_lockPixels(env, targetBitmap, &pixels) < 0) return;

    auto* dst = static_cast<uint8_t*>(pixels);

    jfloat* srcOutArray = env->GetFloatArrayElements(outArray, nullptr);
    jfloat* srcInArray = env->GetFloatArrayElements(inArray, nullptr);

    int numChannels = 3 * scale * scale; // 48 for x4, 12 for x2
    int spatialSize = inTileSize * inTileSize;
    int outSize = inTileSize * scale;

    const float* srcOut = srcOutArray + (arrayPixelOffset * numChannels);
    const float* srcIn = srcInArray + (arrayPixelOffset * 3);

    for (int y = 0; y < inTileSize; ++y) {
        for (int x = 0; x < inTileSize; ++x) {
            int spatialIdx = y * inTileSize + x;
            float baseR, baseG, baseB;

            // 1. Read the base (low-res) pixel, respecting the input tensor layout.
            if (isInputNhwc) {
                int inIdx = spatialIdx * 3;
                baseR = srcIn[inIdx + 0];
                baseG = srcIn[inIdx + 1];
                baseB = srcIn[inIdx + 2];
            } else { // NCHW
                baseR = srcIn[spatialIdx];
                baseG = srcIn[spatialSize + spatialIdx];
                baseB = srcIn[spatialSize * 2 + spatialIdx];
            }

            // 2. Expand the skip connection: same base value replicated across the
            // whole scale x scale output block, equivalent to a nearest-neighbor upsample.
            for (int dy = 0; dy < scale; ++dy) {
                int outY = y * scale + dy;
                for (int dx = 0; dx < scale; ++dx) {
                    int outX = x * scale + dx;

                    // Channel mapping identical to PyTorch's PixelShuffle:
                    // output[c, h*r+i, w*r+j] = input[c*r^2 + i*r + j, h, w]
                    int rChannel = (0 * scale + dy) * scale + dx;
                    int gChannel = (1 * scale + dy) * scale + dx;
                    int bChannel = (2 * scale + dy) * scale + dx;

                    float detR, detG, detB;

                    // 3. Read the detail channels, respecting the output tensor layout.
                    if (isOutputNhwc) {
                        int pixelIdx = spatialIdx * numChannels;
                        detR = srcOut[pixelIdx + rChannel];
                        detG = srcOut[pixelIdx + gChannel];
                        detB = srcOut[pixelIdx + bChannel];
                    } else { // NCHW
                        detR = srcOut[rChannel * spatialSize + spatialIdx];
                        detG = srcOut[gChannel * spatialSize + spatialIdx];
                        detB = srcOut[bChannel * spatialSize + spatialIdx];
                    }

                    // 4. Sum detail + base and write the final pixel.
                    int dstIdx = (outY * outSize + outX) * 4;
                    dst[dstIdx + 0] = floatToUint8(detR + baseR);
                    dst[dstIdx + 1] = floatToUint8(detG + baseG);
                    dst[dstIdx + 2] = floatToUint8(detB + baseB);
                    dst[dstIdx + 3] = 255;
                }
            }
        }
    }

    env->ReleaseFloatArrayElements(outArray, srcOutArray, JNI_ABORT);
    env->ReleaseFloatArrayElements(inArray, srcInArray, JNI_ABORT);
    AndroidBitmap_unlockPixels(env, targetBitmap);
}

extern "C" JNIEXPORT void JNICALL
Java_eu_kanade_tachiyomi_util_upscale_NativePixelOps_readArrayToBitmapPixelShuffleInt8(
        JNIEnv* env,
        jobject /* this */,
        jbyteArray outArray,        // int8 quantized model output
        jfloatArray inArray,        // float32 original tile, unquantized (skip connection base)
        jint arrayPixelOffset,
        jobject targetBitmap,
        jint inTileSize,
        jint scale,
        jboolean isInputNhwc,
        jboolean isOutputNhwc,
        jfloat outputScale,
        jint outputZeroPoint
) {
    AndroidBitmapInfo info;
    void* pixels = nullptr;

    if (AndroidBitmap_getInfo(env, targetBitmap, &info) < 0 || info.format != ANDROID_BITMAP_FORMAT_RGBA_8888) return;
    if (AndroidBitmap_lockPixels(env, targetBitmap, &pixels) < 0) return;

    auto* dst = static_cast<uint8_t*>(pixels);

    jbyte* srcOutArray = env->GetByteArrayElements(outArray, nullptr);
    jfloat* srcInArray = env->GetFloatArrayElements(inArray, nullptr);

    int numChannels = 3 * scale * scale;
    int spatialSize = inTileSize * inTileSize;
    int outSize = inTileSize * scale;

    const jbyte* srcOut = srcOutArray + (arrayPixelOffset * numChannels);
    const float* srcIn = srcInArray + (arrayPixelOffset * 3);

    // Dequantizes one raw int8 model output value back to a real-valued float.
    auto dequantize = [outputScale, outputZeroPoint](jbyte raw) -> float {
        return static_cast<float>(static_cast<int>(raw) - outputZeroPoint) * outputScale;
    };

    for (int y = 0; y < inTileSize; ++y) {
        for (int x = 0; x < inTileSize; ++x) {
            int spatialIdx = y * inTileSize + x;
            float baseR, baseG, baseB;

            if (isInputNhwc) {
                int inIdx = spatialIdx * 3;
                baseR = srcIn[inIdx + 0];
                baseG = srcIn[inIdx + 1];
                baseB = srcIn[inIdx + 2];
            } else {
                baseR = srcIn[spatialIdx];
                baseG = srcIn[spatialSize + spatialIdx];
                baseB = srcIn[spatialSize * 2 + spatialIdx];
            }

            for (int dy = 0; dy < scale; ++dy) {
                int outY = y * scale + dy;
                for (int dx = 0; dx < scale; ++dx) {
                    int outX = x * scale + dx;

                    int rChannel = (0 * scale + dy) * scale + dx;
                    int gChannel = (1 * scale + dy) * scale + dx;
                    int bChannel = (2 * scale + dy) * scale + dx;

                    float detR, detG, detB;
                    if (isOutputNhwc) {
                        int pixelIdx = spatialIdx * numChannels;
                        detR = dequantize(srcOut[pixelIdx + rChannel]);
                        detG = dequantize(srcOut[pixelIdx + gChannel]);
                        detB = dequantize(srcOut[pixelIdx + bChannel]);
                    } else {
                        detR = dequantize(srcOut[rChannel * spatialSize + spatialIdx]);
                        detG = dequantize(srcOut[gChannel * spatialSize + spatialIdx]);
                        detB = dequantize(srcOut[bChannel * spatialSize + spatialIdx]);
                    }

                    int dstIdx = (outY * outSize + outX) * 4;
                    dst[dstIdx + 0] = floatToUint8(detR + baseR);
                    dst[dstIdx + 1] = floatToUint8(detG + baseG);
                    dst[dstIdx + 2] = floatToUint8(detB + baseB);
                    dst[dstIdx + 3] = 255;
                }
            }
        }
    }

    env->ReleaseByteArrayElements(outArray, srcOutArray, JNI_ABORT);
    env->ReleaseFloatArrayElements(inArray, srcInArray, JNI_ABORT);
    AndroidBitmap_unlockPixels(env, targetBitmap);
}
