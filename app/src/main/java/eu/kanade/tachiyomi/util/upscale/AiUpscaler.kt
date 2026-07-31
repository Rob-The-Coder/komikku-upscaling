package eu.kanade.tachiyomi.util.upscale

import android.app.Application
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Rect
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel

/**
 * Wrapper per l'inferenza TFLite di Real-ESRGAN x2, con tiling per gestire
 * pagine manga più grandi della dimensione di input fissa del modello.
 */
class AiUpscaler(private val context: Application) {

    private val tileSize = 256
    private val overlap = 16
    private val scale = 2

    private val interpreter: Interpreter by lazy {
        val options = Interpreter.Options()
        try {
            options.addDelegate(GpuDelegate())
        } catch (e: Exception) {
            options.setNumThreads(4)
        }
        Interpreter(loadModelFile(), options)
    }

    private fun loadModelFile(): ByteBuffer {
        val assetFileDescriptor = context.assets.openFd("realesrgan_x2_256.tflite")
        FileInputStream(assetFileDescriptor.fileDescriptor).use { inputStream ->
            val fileChannel = inputStream.channel
            return fileChannel.map(
                FileChannel.MapMode.READ_ONLY,
                assetFileDescriptor.startOffset,
                assetFileDescriptor.declaredLength,
            )
        }
    }

    fun upscale(input: Bitmap): Bitmap {
        val outW = input.width * scale
        val outH = input.height * scale
        val output = Bitmap.createBitmap(outW, outH, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(output)
        val stride = tileSize - overlap

        var y = 0
        while (y < input.height) {
            var x = 0
            while (x < input.width) {
                val tileW = minOf(tileSize, input.width - x)
                val tileH = minOf(tileSize, input.height - y)

                val tile = Bitmap.createBitmap(tileSize, tileSize, Bitmap.Config.ARGB_8888)
                val tileCanvas = Canvas(tile)
                tileCanvas.drawBitmap(
                    input,
                    Rect(x, y, x + tileW, y + tileH),
                    Rect(0, 0, tileW, tileH),
                    null,
                )

                val upscaledTile = runInference(tile)

                // Copia solo la porzione valida (senza il padding oltre i bordi reali)
                canvas.drawBitmap(
                    upscaledTile,
                    Rect(0, 0, tileW * scale, tileH * scale),
                    Rect(x * scale, y * scale, (x + tileW) * scale, (y + tileH) * scale),
                    null,
                )

                tile.recycle()
                upscaledTile.recycle()
                x += stride
            }
            y += stride
        }
        return output
    }

    private fun runInference(tile: Bitmap): Bitmap {
        val inputBuffer = bitmapToByteBuffer(tile)
        val outSize = tileSize * scale
        val outputBuffer = ByteBuffer
            .allocateDirect(4 * outSize * outSize * 3)
            .order(ByteOrder.nativeOrder())

        interpreter.run(inputBuffer, outputBuffer)
        return byteBufferToBitmap(outputBuffer, outSize, outSize)
    }

    private fun bitmapToByteBuffer(bitmap: Bitmap): ByteBuffer {
        val buffer = ByteBuffer
            .allocateDirect(4 * tileSize * tileSize * 3)
            .order(ByteOrder.nativeOrder())
        val pixels = IntArray(tileSize * tileSize)
        bitmap.getPixels(pixels, 0, tileSize, 0, 0, tileSize, tileSize)
        for (pixel in pixels) {
            buffer.putFloat(((pixel shr 16) and 0xFF) / 255.0f) // R
            buffer.putFloat(((pixel shr 8) and 0xFF) / 255.0f)  // G
            buffer.putFloat((pixel and 0xFF) / 255.0f)          // B
        }
        buffer.rewind()
        return buffer
    }

    private fun byteBufferToBitmap(buffer: ByteBuffer, width: Int, height: Int): Bitmap {
        buffer.rewind()
        val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        val pixels = IntArray(width * height)
        for (i in pixels.indices) {
            val r = (buffer.float * 255.0f).toInt().coerceIn(0, 255)
            val g = (buffer.float * 255.0f).toInt().coerceIn(0, 255)
            val b = (buffer.float * 255.0f).toInt().coerceIn(0, 255)
            pixels[i] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
        }
        bitmap.setPixels(pixels, 0, width, 0, 0, width, height)
        return bitmap
    }
}
