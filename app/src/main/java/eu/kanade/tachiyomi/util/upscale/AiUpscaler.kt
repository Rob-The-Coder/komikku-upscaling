package eu.kanade.tachiyomi.util.upscale

import android.app.Application
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Rect
import android.util.Log
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.withContext
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import java.util.concurrent.Executors

/**
 * Wrapper per l'inferenza TFLite di Real-ESRGAN x2, con tiling per gestire
 * pagine manga più grandi della dimensione di input fissa del modello.
 */
class AiUpscaler(private val context: Application) {

    private val tileSize = 256
    private val overlap = 16
    private val scale = 4

    private fun logTensorInfo(interpreter: Interpreter) {
        val inputTensor = interpreter.getInputTensor(0)
        val outputTensor = interpreter.getOutputTensor(0)
        Log.d("AiUpscaler", "Input shape: ${inputTensor.shape().joinToString()}, dtype: ${inputTensor.dataType()}")
        Log.d("AiUpscaler", "Output shape: ${outputTensor.shape().joinToString()}, dtype: ${outputTensor.dataType()}")
    }

    // Un solo thread dedicato: interpreter creato e invocato SEMPRE qui.
    private val inferenceExecutor = Executors.newSingleThreadExecutor { r -> Thread(r, "AiUpscaler-Inference") }
    private val inferenceDispatcher = inferenceExecutor.asCoroutineDispatcher()

    // Creata pigramente, ma la prima creazione avverrà comunque dentro
    // inferenceDispatcher grazie a come la richiamiamo in upscale().
    private var _interpreter: Interpreter? = null
    private fun getOrCreateInterpreter(): Interpreter {
        return _interpreter ?: createInterpreter(useGpu = true)
            ?.also { _interpreter = it }
        ?: createInterpreter(useGpu = false)!!.also { _interpreter = it }
    }

    private fun createInterpreter(useGpu: Boolean): Interpreter? {
        return try {
            val options = Interpreter.Options()
            if (useGpu) {
                options.addDelegate(GpuDelegate())
            } else {
                options.setNumThreads(4)
            }
            Interpreter(loadModelFile(), options)
        } catch (e: Throwable) {
            if (useGpu) null else throw e
        }
    }

    private fun loadModelFile(): ByteBuffer {
        val assetFileDescriptor = context.assets.openFd("fsmangav2_x4_256.tflite")
        FileInputStream(assetFileDescriptor.fileDescriptor).use { inputStream ->
            val fileChannel = inputStream.channel
            return fileChannel.map(
                FileChannel.MapMode.READ_ONLY,
                assetFileDescriptor.startOffset,
                assetFileDescriptor.declaredLength,
            )
        }
    }
    suspend fun upscale(input: Bitmap): Bitmap {
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
                Canvas(tile).drawBitmap(
                    input,
                    Rect(x, y, x + tileW, y + tileH),
                    Rect(0, 0, tileW, tileH),
                    null,
                )

                // Solo QUESTA chiamata va sul thread dedicato: preparazione tile e
                // composizione canvas restano libere di girare in parallelo tra pagine diverse.
                val upscaledTile = withContext(inferenceDispatcher) { runInference(tile) }

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
        val interpreter = getOrCreateInterpreter()

        //logTensorInfo(interpreter)

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

        // Planare invece di interlacciato: prima tutto R, poi tutto G, poi tutto B
        for (pixel in pixels) buffer.putFloat(((pixel shr 16) and 0xFF) / 255.0f) // R
        for (pixel in pixels) buffer.putFloat(((pixel shr 8) and 0xFF) / 255.0f)  // G
        for (pixel in pixels) buffer.putFloat((pixel and 0xFF) / 255.0f)         // B

        buffer.rewind()
        return buffer
    }

    private fun byteBufferToBitmap(buffer: ByteBuffer, width: Int, height: Int): Bitmap {
        buffer.rewind()
        val size = width * height
        val rPlane = FloatArray(size) { buffer.float }
        val gPlane = FloatArray(size) { buffer.float }
        val bPlane = FloatArray(size) { buffer.float }

        val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        val pixels = IntArray(size)
        for (i in 0 until size) {
            val r = (rPlane[i] * 255.0f).toInt().coerceIn(0, 255)
            val g = (gPlane[i] * 255.0f).toInt().coerceIn(0, 255)
            val b = (bPlane[i] * 255.0f).toInt().coerceIn(0, 255)
            pixels[i] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
        }
        bitmap.setPixels(pixels, 0, width, 0, 0, width, height)
        return bitmap
    }
}
