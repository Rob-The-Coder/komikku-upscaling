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

    /** Reads model output from 'inArray' starting at 'arrayPixelOffset' into 'targetBitmap', NHWC order. */
    external fun readArrayToBitmapNHWC(
        inArray: FloatArray,
        arrayPixelOffset: Int,
        targetBitmap: Bitmap,
        outSize: Int,
    )

    /** Same as readArrayToBitmapNHWC but reads planar (channel-major) order for NCHW models. */
    external fun readArrayToBitmapNCHW(
        inArray: FloatArray,
        arrayPixelOffset: Int,
        targetBitmap: Bitmap,
        outSize: Int,
    )

    external fun readArrayToBitmapPixelShuffle(
        outArray: FloatArray,
        inArray: FloatArray,
        arrayPixelOffset: Int,
        targetBitmap: Bitmap,
        inTileSize: Int,
        scale: Int,
        isInputNhwc: Boolean,
        isOutputNhwc: Boolean
    )
}
