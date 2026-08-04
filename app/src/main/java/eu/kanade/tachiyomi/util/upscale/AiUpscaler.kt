package eu.kanade.tachiyomi.util.upscale

import android.app.Application
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Rect
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.withContext
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
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

    private val tileSize = 512
    private val overlap = 0
    private val scale = 4
    private enum class DelegateMode { GPU, CPU }
    private enum class TensorLayout { NHWC, NCHW }
    private val outSize = tileSize * scale
    private val inputBuffer = ByteBuffer.allocateDirect(4 * tileSize * tileSize * 3).order(ByteOrder.nativeOrder())
    private val outputBuffer = ByteBuffer.allocateDirect(4 * outSize * outSize * 3).order(ByteOrder.nativeOrder())
    private lateinit var inputLayout: TensorLayout
    private lateinit var outputLayout: TensorLayout

    private fun detectLayout(shape: IntArray): TensorLayout {
        // Il canale (valore 3) è all'indice 1 in NCHW, all'indice 3 in NHWC
        return if (shape[1] == 3) TensorLayout.NCHW else TensorLayout.NHWC
    }

    // Chiamala una volta sola subito dopo la creazione dell'interpreter
    private fun detectAndCacheLayouts(interpreter: Interpreter) {
        inputLayout = detectLayout(interpreter.getInputTensor(0).shape())
        outputLayout = detectLayout(interpreter.getOutputTensor(0).shape())
        Log.d("AiUpscaler", "Input layout: $inputLayout, Output layout: $outputLayout")
    }

    // Un solo thread dedicato: interpreter creato e invocato SEMPRE qui.
    private val inferenceExecutor = Executors.newSingleThreadExecutor { r ->
        Thread {
            android.os.Process.setThreadPriority(android.os.Process.THREAD_PRIORITY_BACKGROUND)
            r.run()
        }.apply { name = "AiUpscaler-Inference" }
    }
    private val inferenceDispatcher = inferenceExecutor.asCoroutineDispatcher()

    // Creata pigramente, ma la prima creazione avverrà comunque dentro
    // inferenceDispatcher grazie a come la richiamiamo in upscale().
    private val interpreter: Interpreter by lazy {
        createInterpreter(DelegateMode.GPU)
            ?: createInterpreter(DelegateMode.CPU)!!
    }

    private val compatList = CompatibilityList()
    private fun createInterpreter(mode: DelegateMode): Interpreter? {
        // Per il GPU, controlliamo prima la compatibility list ufficiale:
        // se il device non è in lista, non proviamo nemmeno, risparmiando
        // il costo di un tentativo che sappiamo già fallirebbe o andrebbe male.
        if (mode == DelegateMode.GPU && !compatList.isDelegateSupportedOnThisDevice) {
            return null
        }

        return try {
            val options = Interpreter.Options()
            when (mode) {
                DelegateMode.GPU -> {
                    val gpuOptions = compatList.bestOptionsForThisDevice
                    options.addDelegate(GpuDelegate(gpuOptions))
                }
                DelegateMode.CPU -> options.setNumThreads(4)
            }

            val newInterpreter = Interpreter(loadModelFile(), options)
            Log.d("AiUpscaler", "Creazione interprete con mode=${mode}")
            detectAndCacheLayouts(newInterpreter)

            newInterpreter
        } catch (e: Throwable) {
            Log.w("AiUpscaler", "Creazione interprete GPU fallita", e)
            if (mode == DelegateMode.CPU) throw e else null
        }
    }

    private fun loadModelFile(): ByteBuffer {
        val assetFileDescriptor = context.assets.openFd("realesr_animevideov3_x4_512T_float32.tflite")
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

        val margin = overlap / 2
        val step = tileSize - (margin * 2)

        val tempTile = Bitmap.createBitmap(tileSize, tileSize, Bitmap.Config.ARGB_8888)
        val tempCanvas = Canvas(tempTile)

        var y = 0
        while (y < input.height) {
            val actualY = if (y + tileSize > input.height) maxOf(0, input.height - tileSize) else y

            var x = 0
            while (x < input.width) {
                val actualX = if (x + tileSize > input.width) maxOf(0, input.width - tileSize) else x

                // 1. Estrai sempre un tile nativo 256x256 ancorato
                tempCanvas.drawBitmap(
                    input,
                    Rect(actualX, actualY, actualX + tileSize, actualY + tileSize),
                    Rect(0, 0, tileSize, tileSize),
                    null
                )

                // 2. Inferenza
                val upscaledTile = withContext(inferenceDispatcher) { runInference(tempTile) }

                // 3. Calcola i margini da scartare (se siamo ai bordi assoluti dell'immagine non scartiamo il bordo esterno)
                val cropLeft = if (actualX == 0) 0 else margin * scale
                val cropTop = if (actualY == 0) 0 else margin * scale
                val cropRight = if (actualX + tileSize >= input.width) tileSize * scale else (tileSize - margin) * scale
                val cropBottom = if (actualY + tileSize >= input.height) tileSize * scale else (tileSize - margin) * scale

                val srcRect = Rect(cropLeft, cropTop, cropRight, cropBottom)

                // 4. Mappa le coordinate esatte sulla bitmap di output
                val destLeft = (actualX * scale) + cropLeft
                val destTop = (actualY * scale) + cropTop
                val destRight = (actualX * scale) + cropRight
                val destBottom = (actualY * scale) + cropBottom

                val destRect = Rect(destLeft, destTop, destRight, destBottom)

                canvas.drawBitmap(upscaledTile, srcRect, destRect, null)
                upscaledTile.recycle()

                if (actualX + tileSize >= input.width) break
                x += step
            }
            if (actualY + tileSize >= input.height) break
            y += step
        }
        tempTile.recycle()
        return output
    }


    private fun runInference(tile: Bitmap): Bitmap {
        val activeInterpreter = interpreter

        inputBuffer.clear()
        outputBuffer.clear()

        bitmapToByteBuffer(tile, inputBuffer)

        val t0 = System.currentTimeMillis()
        activeInterpreter.run(inputBuffer, outputBuffer)
        Log.d("AiUpscaler", "Tile inferenza: ${System.currentTimeMillis() - t0}ms")

        return byteBufferToBitmap(outputBuffer, outSize, outSize)
    }

    private fun bitmapToByteBuffer(bitmap: Bitmap, buffer: ByteBuffer) {
        val pixels = IntArray(tileSize * tileSize)
        bitmap.getPixels(pixels, 0, tileSize, 0, 0, tileSize, tileSize)

        when (inputLayout) {
            TensorLayout.NHWC -> {
                for (pixel in pixels) {
                    buffer.putFloat(((pixel shr 16) and 0xFF) / 255.0f) // R
                    buffer.putFloat(((pixel shr 8) and 0xFF) / 255.0f)  // G
                    buffer.putFloat((pixel and 0xFF) / 255.0f)          // B
                }
            }
            TensorLayout.NCHW -> {
                for (c in 0 until 3) {
                    for (pixel in pixels) {
                        val value = when (c) {
                            0 -> (pixel shr 16) and 0xFF
                            1 -> (pixel shr 8) and 0xFF
                            else -> pixel and 0xFF
                        }
                        buffer.putFloat(value / 255.0f)
                    }
                }
            }
        }
        buffer.rewind()
    }

    private fun byteBufferToBitmap(buffer: ByteBuffer, width: Int, height: Int): Bitmap {
        buffer.rewind()
        val size = width * height
        val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        val pixels = IntArray(size)

        when (outputLayout) {
            TensorLayout.NHWC -> {
                for (i in 0 until size) {
                    val r = (buffer.float * 255.0f).toInt().coerceIn(0, 255)
                    val g = (buffer.float * 255.0f).toInt().coerceIn(0, 255)
                    val b = (buffer.float * 255.0f).toInt().coerceIn(0, 255)
                    pixels[i] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
                }
            }
            TensorLayout.NCHW -> {
                val rPlane = FloatArray(size) { buffer.float }
                val gPlane = FloatArray(size) { buffer.float }
                val bPlane = FloatArray(size) { buffer.float }
                for (i in 0 until size) {
                    val r = (rPlane[i] * 255.0f).toInt().coerceIn(0, 255)
                    val g = (gPlane[i] * 255.0f).toInt().coerceIn(0, 255)
                    val b = (bPlane[i] * 255.0f).toInt().coerceIn(0, 255)
                    pixels[i] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
                }
            }
        }
        bitmap.setPixels(pixels, 0, width, 0, 0, width, height)
        return bitmap
    }
}
