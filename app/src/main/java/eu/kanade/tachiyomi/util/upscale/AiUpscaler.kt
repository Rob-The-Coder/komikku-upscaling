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

    private val batchSize = 3
    private val tileSize = 384
    private val overlap = 0
    private val scale = 4
    private enum class DelegateMode { GPU, CPU }
    private enum class TensorLayout { NHWC, NCHW }
    private val outSize = tileSize * scale
    private val inputBuffer = ByteBuffer.allocateDirect(4 * tileSize * tileSize * 3).order(ByteOrder.nativeOrder())
    private val outputBuffer = ByteBuffer.allocateDirect(4 * outSize * outSize * 3).order(ByteOrder.nativeOrder())

    // Buffer dimensionati per l'intero batch, non più per singolo tile
    private val batchInputBuffer =
        ByteBuffer.allocateDirect(batchSize * 4 * tileSize * tileSize * 3).order(ByteOrder.nativeOrder())
    private val batchOutputBuffer =
        ByteBuffer.allocateDirect(batchSize * 4 * outSize * outSize * 3).order(ByteOrder.nativeOrder())

    private val inputTiles = Array(batchSize) {
        Bitmap.createBitmap(tileSize, tileSize, Bitmap.Config.ARGB_8888)
    }
    private val inputCanvases = inputTiles.map { Canvas(it) }

    // Tile di output (2 istanze per batch=2)
    private val reusableOutputTiles = Array(batchSize) {
        Bitmap.createBitmap(outSize, outSize, Bitmap.Config.ARGB_8888)
    }
    private data class TilePos(val x: Int, val y: Int)
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
        Thread(r, "AiUpscaler-Inference")

//        Thread {
//            android.os.Process.setThreadPriority(android.os.Process.THREAD_PRIORITY_BACKGROUND)
//            r.run()
//        }.apply { name = "AiUpscaler-Inference" }
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
        val assetFileDescriptor = context.assets.openFd("realesr_animevideov3_x4_384T_B3_float32.tflite")
        FileInputStream(assetFileDescriptor.fileDescriptor).use { inputStream ->
            val fileChannel = inputStream.channel
            return fileChannel.map(
                FileChannel.MapMode.READ_ONLY,
                assetFileDescriptor.startOffset,
                assetFileDescriptor.declaredLength,
            )
        }
    }

    private fun collectTilePositions(input: Bitmap, step: Int): List<TilePos> {
        val positions = mutableListOf<TilePos>()
        var y = 0
        while (true) {
            val actualY = if (y + tileSize > input.height) maxOf(0, input.height - tileSize) else y
            var x = 0
            while (true) {
                val actualX = if (x + tileSize > input.width) maxOf(0, input.width - tileSize) else x
                positions.add(TilePos(actualX, actualY))
                if (actualX + tileSize >= input.width) break
                x += step
            }
            if (actualY + tileSize >= input.height) break
            y += step
        }
        return positions
    }
//    suspend fun upscale(input: Bitmap): Bitmap {
//        val outW = input.width * scale
//        val outH = input.height * scale
//        val output = Bitmap.createBitmap(outW, outH, Bitmap.Config.ARGB_8888)
//        val canvas = Canvas(output)
//
//        val margin = overlap / 2
//        val step = tileSize - (margin * 2)
//
//        val tempTile = Bitmap.createBitmap(tileSize, tileSize, Bitmap.Config.ARGB_8888)
//        val tempCanvas = Canvas(tempTile)
//
//        var y = 0
//        while (y < input.height) {
//            val actualY = if (y + tileSize > input.height) maxOf(0, input.height - tileSize) else y
//
//            var x = 0
//            while (x < input.width) {
//                val actualX = if (x + tileSize > input.width) maxOf(0, input.width - tileSize) else x
//
//                // 1. Estrai sempre un tile nativo 256x256 ancorato
//                tempCanvas.drawBitmap(
//                    input,
//                    Rect(actualX, actualY, actualX + tileSize, actualY + tileSize),
//                    Rect(0, 0, tileSize, tileSize),
//                    null
//                )
//
//                // 2. Inferenza
//                val upscaledTile = withContext(inferenceDispatcher) { runInference(tempTile) }
//
//                // 3. Calcola i margini da scartare (se siamo ai bordi assoluti dell'immagine non scartiamo il bordo esterno)
//                val cropLeft = if (actualX == 0) 0 else margin * scale
//                val cropTop = if (actualY == 0) 0 else margin * scale
//                val cropRight = if (actualX + tileSize >= input.width) tileSize * scale else (tileSize - margin) * scale
//                val cropBottom = if (actualY + tileSize >= input.height) tileSize * scale else (tileSize - margin) * scale
//
//                val srcRect = Rect(cropLeft, cropTop, cropRight, cropBottom)
//
//                // 4. Mappa le coordinate esatte sulla bitmap di output
//                val destLeft = (actualX * scale) + cropLeft
//                val destTop = (actualY * scale) + cropTop
//                val destRight = (actualX * scale) + cropRight
//                val destBottom = (actualY * scale) + cropBottom
//
//                val destRect = Rect(destLeft, destTop, destRight, destBottom)
//
//                canvas.drawBitmap(upscaledTile, srcRect, destRect, null)
//                upscaledTile.recycle()
//
//                if (actualX + tileSize >= input.width) break
//                x += step
//            }
//            if (actualY + tileSize >= input.height) break
//            y += step
//        }
//        tempTile.recycle()
//        return output
//    }

    suspend fun upscale(input: Bitmap): Bitmap {
        val outW = input.width * scale
        val outH = input.height * scale
        val output = Bitmap.createBitmap(outW, outH, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(output)

        val margin = overlap / 2
        val step = tileSize - (margin * 2)
        val positions = collectTilePositions(input, step)

        positions.chunked(batchSize).forEach { batch ->
            val realCount = batch.size

            // Riutilizziamo le Bitmap e Canvas di input per estrarre i tile
            for (i in 0 until batchSize) {
                val pos = if (i < realCount) batch[i] else batch.last() // Padding duplicando l'ultimo se necessario
                inputCanvases[i].drawBitmap(
                    input,
                    Rect(pos.x, pos.y, pos.x + tileSize, pos.y + tileSize),
                    Rect(0, 0, tileSize, tileSize),
                    null
                )
            }

            // Inferenza nativa C++
            val results = withContext(inferenceDispatcher) { runBatchInference(inputTiles) }

            // Disegno dei soli risultati reali sul canvas finale
            for (i in 0 until realCount) {
                val pos = batch[i]
                val cropLeft = if (pos.x == 0) 0 else margin * scale
                val cropTop = if (pos.y == 0) 0 else margin * scale
                val cropRight = if (pos.x + tileSize >= input.width) tileSize * scale else (tileSize - margin) * scale
                val cropBottom = if (pos.y + tileSize >= input.height) tileSize * scale else (tileSize - margin) * scale

                val srcRect = Rect(cropLeft, cropTop, cropRight, cropBottom)
                val destLeft = (pos.x * scale) + cropLeft
                val destTop = (pos.y * scale) + cropTop
                val destRight = (pos.x * scale) + cropRight
                val destBottom = (pos.y * scale) + cropBottom

                canvas.drawBitmap(results[i], srcRect, Rect(destLeft, destTop, destRight, destBottom), null)
            }
        }
        return output
    }

//    private val reusableOutputTiles = Array(batchSize) {
//        Bitmap.createBitmap(outSize, outSize, Bitmap.Config.ARGB_8888)
//    }

    private fun runBatchInference(tiles: Array<Bitmap>): Array<Bitmap> {
        val activeInterpreter = interpreter

        val t0 = System.currentTimeMillis()

        // 1. Scrittura nativa C++ nei buffer
        batchInputBuffer.clear()
        val tilePixelCount = tileSize * tileSize
        for (i in tiles.indices) {
            val pixelOffset = i * tilePixelCount
            if (inputLayout == TensorLayout.NHWC) {
                NativePixelOps.writeBitmapToBufferNHWC(tiles[i], batchInputBuffer, pixelOffset, tileSize)
            } else {
                NativePixelOps.writeBitmapToBufferNCHW(tiles[i], batchInputBuffer, pixelOffset, tileSize)
            }
        }
        batchInputBuffer.rewind()
        val t1 = System.currentTimeMillis()

        // 2. Run TFLite
        batchOutputBuffer.clear()
        activeInterpreter.run(batchInputBuffer, batchOutputBuffer)
        val t2 = System.currentTimeMillis()

        // 3. Lettura nativa C++ dal buffer alle Bitmap riutilizzabili
        batchOutputBuffer.rewind()
        val outTilePixelCount = outSize * outSize
        for (i in tiles.indices) {
            val pixelOffset = i * outTilePixelCount
            if (outputLayout == TensorLayout.NHWC) {
                NativePixelOps.readBufferToBitmapNHWC(batchOutputBuffer, pixelOffset, reusableOutputTiles[i], outSize)
            } else {
                NativePixelOps.readBufferToBitmapNCHW(batchOutputBuffer, pixelOffset, reusableOutputTiles[i], outSize)
            }
        }
        val t3 = System.currentTimeMillis()

        Log.d("AiUpscaler", "Scrittura Native: ${t1 - t0}ms | TFLite run(): ${t2 - t1}ms | Lettura Native: ${t3 - t2}ms | Totale: ${t3 - t0}ms")

        return reusableOutputTiles

//        val activeInterpreter = interpreter // forza lazy init + layout detection
//
//        batchInputBuffer.clear()
//        tiles.forEach { tile -> writeTileToBuffer(tile, batchInputBuffer) }
//        batchInputBuffer.rewind()
//
//        batchOutputBuffer.clear()
//        val t0 = System.currentTimeMillis()
//        activeInterpreter.run(batchInputBuffer, batchOutputBuffer)
//        Log.d("AiUpscaler", "Batch inferenza (${tiles.size} tile): ${System.currentTimeMillis() - t0}ms")
//        batchOutputBuffer.rewind()
//
//        return tiles.indices.map { readTileFromBuffer(batchOutputBuffer, outSize, outSize) }
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

    // Scrive DIRETTAMENTE nella bitmap riutilizzabile passata, con accesso
    // indicizzato assoluto al buffer — niente FloatArray temporanei intermedi.
    private fun readTileIntoBitmap(buffer: ByteBuffer, byteOffset: Int, width: Int, height: Int, target: Bitmap) {
        val size = width * height
        val pixels = IntArray(size) // questo resta, serve comunque per setPixels in un colpo solo

        when (outputLayout) {
            TensorLayout.NHWC -> {
                for (i in 0 until size) {
                    val off = byteOffset + i * 12
                    val r = (buffer.getFloat(off) * 255.0f).toInt().coerceIn(0, 255)
                    val g = (buffer.getFloat(off + 4) * 255.0f).toInt().coerceIn(0, 255)
                    val b = (buffer.getFloat(off + 8) * 255.0f).toInt().coerceIn(0, 255)
                    pixels[i] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
                }
            }
            TensorLayout.NCHW -> {
                val rOff = byteOffset
                val gOff = byteOffset + size * 4
                val bOff = byteOffset + size * 8
                for (i in 0 until size) {
                    val r = (buffer.getFloat(rOff + i * 4) * 255.0f).toInt().coerceIn(0, 255)
                    val g = (buffer.getFloat(gOff + i * 4) * 255.0f).toInt().coerceIn(0, 255)
                    val b = (buffer.getFloat(bOff + i * 4) * 255.0f).toInt().coerceIn(0, 255)
                    pixels[i] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
                }
            }
        }
        target.setPixels(pixels, 0, width, 0, 0, width, height)
    }

    // Scrive UN tile nel buffer, SENZA clear/rewind interni: la gestione del
    // cursore è responsabilità del chiamante, dato che ora scriviamo N tile
    // di fila nello stesso buffer prima di un unico rewind finale.
    private fun writeTileToBuffer(bitmap: Bitmap, buffer: ByteBuffer) {
        val pixels = IntArray(tileSize * tileSize)
        bitmap.getPixels(pixels, 0, tileSize, 0, 0, tileSize, tileSize)

        when (inputLayout) {
            TensorLayout.NHWC -> {
                for (pixel in pixels) {
                    buffer.putFloat(((pixel shr 16) and 0xFF) / 255.0f)
                    buffer.putFloat(((pixel shr 8) and 0xFF) / 255.0f)
                    buffer.putFloat((pixel and 0xFF) / 255.0f)
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
    }

    // Legge UN tile dal buffer condiviso, avanzando il cursore di lettura;
    // NESSUN rewind interno, va chiamata N volte di fila dopo un unico
    // rewind fatto una volta sola dal chiamante.
    private fun readTileFromBuffer(buffer: ByteBuffer, width: Int, height: Int): Bitmap {
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
