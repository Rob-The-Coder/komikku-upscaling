package eu.kanade.tachiyomi.util.upscale

import android.graphics.Bitmap

/**
 * JNI bridge for pixel <-> tensor conversion. Operates on plain Kotlin
 * FloatArrays (instead of a direct ByteBuffer) because the CompiledModel
 * Kotlin API's TensorBuffer only exposes writeFloat(FloatArray) / readFloat(),
 * with no raw buffer write path.
 */
object NativePixelOps {
    init {
        System.loadLibrary("aiupscaler")
    }

    /** Writes 'bitmap' pixels into 'outArray' starting at 'arrayPixelOffset', NHWC order, normalized to [0, 1]. */
    external fun writeBitmapToArrayNHWC(
        bitmap: Bitmap,
        outArray: FloatArray,
        arrayPixelOffset: Int,
        tileSize: Int,
    )

    /** Same as writeBitmapToArrayNHWC but writes planar (channel-major) order for NCHW models. */
    external fun writeBitmapToArrayNCHW(
        bitmap: Bitmap,
        outArray: FloatArray,
        arrayPixelOffset: Int,
        tileSize: Int,
    )

    /**
     * Reads the model's raw pre-PixelShuffle output ('outArray') and the original low-res tile
     * ('inArray'), applies PixelShuffle and the residual skip connection, and writes the result
     * into 'targetBitmap'. See native-lib.cpp for the full explanation of why this step moved
     * from the model graph to native code.
     */
    external fun readArrayToBitmapPixelShuffle(
        outArray: FloatArray,
        inArray: FloatArray,
        arrayPixelOffset: Int,
        targetBitmap: Bitmap,
        inTileSize: Int,
        scale: Int,
        isInputNhwc: Boolean,
        isOutputNhwc: Boolean,
    )

    /** Same as readArrayToBitmapPixelShuffle, but 'outArray' holds raw int8 quantized model output, dequantized inline using 'outputScale'/'outputZeroPoint'. */
    external fun readArrayToBitmapPixelShuffleInt8(
        outArray: ByteArray,
        inArray: FloatArray,
        arrayPixelOffset: Int,
        targetBitmap: Bitmap,
        inTileSize: Int,
        scale: Int,
        isInputNhwc: Boolean,
        isOutputNhwc: Boolean,
        outputScale: Float,
        outputZeroPoint: Int,
    )
}
