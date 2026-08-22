package eu.kanade.tachiyomi.util.upscale

import android.app.Application
import android.graphics.Bitmap
import android.graphics.BitmapShader
import android.graphics.Canvas
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.Rect
import android.graphics.Shader
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.LiteRtException
import exh.log.xLogD
import exh.log.xLogW
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.withContext
import java.io.File
import java.util.concurrent.Executors

/**
 * Wrapper for LiteRT (CompiledModel API) inference of Real-ESRGAN, with tiling to
 * handle manga pages larger than the model's fixed input size.
 */
class AiUpscaler(
    private val context: Application,
    private val model: UpscaleModel,
    requestedBatchSize: Int,
    requestedOverlap: Int,
) {
    /** Batch variant (batch size + .tflite asset) resolved for the current request. */
    private val variant = model.variantFor(requestedBatchSize)

    /** Number of tiles processed in a single inference call. */
    private val batchSize = variant.batchSize

    /** Hardware target to try when creating the CompiledModel, with GPU -> CPU fallback. */
    private enum class DelegateMode { GPU, CPU }

    /** Side length of a tile produced by the model. */
    private val outSize get() = model.outSize

    /** Side length of a tile fed to the model, content + padding. */
    private val paddedTileSize get() = model.paddedTileSize

    /** Input tensor layout, known statically from the conversion pipeline, not detected at runtime. */
    private val inputLayout get() = model.inputLayout

    /** Output tensor layout, known statically from the conversion pipeline, not detected at runtime. */
    private val outputLayout get() = model.outputLayout

    /*
    Constraints: Odd overlap is truncated to even (margin = overlap/2);
    overlap too large compared to tileContentSize would cause 'step' to collapse to zero or negative in collectTilePositions(),
    causing a loop that never advances. We avoid it by keeping it under the middle of the tile.
     */
    /** Overlap (in pixels, always even) between adjacent tiles, to soften seam artifacts. */
    private val overlap = requestedOverlap.coerceIn(0, model.tileContentSize / 2 - 1).let { it - (it % 2) }

    // Arrays sized for the whole batch, not per single tile
    /** Reusable input buffer for each batch, filled by the native code before every run(). */
    private val batchInputArray by lazy {
        FloatArray(batchSize * paddedTileSize * paddedTileSize * 3)
    }

    /** Reusable input Bitmaps used as extraction targets for each tile, one per batch slot. */
    private val inputTiles by lazy {
        Array(batchSize) { Bitmap.createBitmap(paddedTileSize, paddedTileSize, Bitmap.Config.ARGB_8888) }
    }

    /** Canvases bound to 'inputTiles', used to draw each tile before inference. */
    private val inputCanvases by lazy { inputTiles.map { Canvas(it) } }

    /** Reusable output Bitmaps used as write targets for the results, one per batch slot. */
    private val reusableOutputTiles by lazy {
        Array(batchSize) { Bitmap.createBitmap(outSize, outSize, Bitmap.Config.ARGB_8888) }
    }

    /** Top-left corner of a tile's content area within the source page. */
    private data class TilePos(val x: Int, val y: Int)

    /** Reused Paint for drawing padded tiles via BitmapShader. */
    private val tilePaint = Paint(Paint.FILTER_BITMAP_FLAG)

    // Single dedicated thread: CompiledModel always created and invoked here
    /** Dedicated single-thread executor on which the CompiledModel is always created and invoked. */
    private val inferenceExecutor = Executors.newSingleThreadExecutor { r ->
        Thread {
            android.os.Process.setThreadPriority(android.os.Process.THREAD_PRIORITY_BACKGROUND)
            r.run()
        }.apply { name = "AiUpscaler-Inference" }
    }

    /** Coroutine dispatcher backed by 'inferenceExecutor', used to confine inference to the dedicated thread. */
    private val inferenceDispatcher = inferenceExecutor.asCoroutineDispatcher()

    /** Lazy CompiledModel initialization: tries GPU first, falls back to CPU if GPU is unavailable. */
    private val compiledModelLazy = lazy(LazyThreadSafetyMode.NONE) {
        createCompiledModel(DelegateMode.GPU) ?: createCompiledModel(DelegateMode.CPU)!!
    }
    private val compiledModel by compiledModelLazy

    /** Input buffers pre-allocated by the CompiledModel, reused across inference batches. */
    private val inputBuffersLazy = lazy(LazyThreadSafetyMode.NONE) { compiledModel.createInputBuffers() }

    /** Output buffers pre-allocated by the CompiledModel, reused across inference batches. */
    private val outputBuffersLazy = lazy(LazyThreadSafetyMode.NONE) { compiledModel.createOutputBuffers() }
    private val inputBuffers by inputBuffersLazy
    private val outputBuffers by outputBuffersLazy

    /**
     * Creates a CompiledModel for the requested accelerator. Returns null (instead of
     * propagating the exception) when GPU creation fails, to allow falling back to CPU;
     * CPU mode always propagates, since there is no further fallback available.
     */
    private fun createCompiledModel(mode: DelegateMode): CompiledModel? {
        val accelerator = when (mode) {
            DelegateMode.GPU -> Accelerator.GPU
            DelegateMode.CPU -> Accelerator.CPU
        }

        return try {
            val newModel = CompiledModel.create(
                modelFilePath(),
                CompiledModel.Options(accelerator),
            )
            xLogD("CompiledModel created with accelerator=$accelerator, batch=$batchSize")
            newModel
        } catch (e: LiteRtException) {
            xLogW("CompiledModel creation failed for accelerator=$accelerator", e)
            if (mode == DelegateMode.CPU) throw e else null
        }
    }

    /** Absolute filesystem path of the bundled .tflite file for the current variant. */
    private fun modelFilePath(): String {
        //return File(context.filesDir, "models/${variant.assetFileName}").absolutePath
        val testFileName = "realesr_animevideov3_x4_384T_static.tflite" // Nome del tuo file in assets
        val targetFile = File(context.cacheDir, testFileName)

        try {
            // Copiamo SEMPRE per assicurarci di non usare un file corrotto in cache
            context.assets.open(testFileName).use { input ->
                targetFile.outputStream().use { output ->
                    input.copyTo(output)
                }
            }
            val sizeKb = targetFile.length() / 1024
            xLogD("Modello copiato con successo. Dimensione in cache: $sizeKb KB")

            if (sizeKb < 100) {
                xLogW("ATTENZIONE: Il modello è troppo piccolo ($sizeKb KB)! Probabilmente non è valido.")
            }
        } catch (e: Exception) {
            xLogW("Errore durante la copia del file da assets", e)
        }

        // Copia il file dagli assets alla cache dell'app solo se non esiste già
        return targetFile.absolutePath
    }

    /** Computes the top-left positions of all tiles needed to cover 'input' with stride 'step'. */
    private fun collectTilePositions(input: Bitmap, step: Int): List<TilePos> {
        val positions = mutableListOf<TilePos>()
        val contentSize = model.tileContentSize
        var y = 0
        while (true) {
            val actualY = if (y + contentSize > input.height) maxOf(0, input.height - contentSize) else y
            var x = 0
            while (true) {
                val actualX = if (x + contentSize > input.width) maxOf(0, input.width - contentSize) else x
                positions.add(TilePos(actualX, actualY))
                if (actualX + contentSize >= input.width) break
                x += step
            }
            if (actualY + contentSize >= input.height) break
            y += step
        }
        return positions
    }

    /**
     * Extracts a tile of size 'paddedTileSize' into 'canvas', with
     * 'contentX'/'contentY' as the corner of the actual content (not the padding).
     * If 'model.paddingPerSide == 0' (Real-ESRGAN), behavior unchanged
     * compared to before. If >0 (waifu2x), the margin around the
     * content is taken from real pixels on the page when available;
     * at the true edges of the page, where there is no other content, the bordering
     * pixel is replicated (CLAMP) instead of leaving empty area or reading
     * out of bitmap bounds. Necessary because those pixels are used
     * by the network as a real context for its receptive field before discarding them.
     */
    private fun drawPaddedTile(source: Bitmap, canvas: Canvas, contentX: Int, contentY: Int) {
        val padding = model.paddingPerSide
        val contentSize = model.tileContentSize

        if (padding == 0) {
            canvas.drawBitmap(
                source,
                Rect(contentX, contentY, contentX + contentSize, contentY + contentSize),
                Rect(0, 0, contentSize, contentSize),
                null,
            )
            return
        }

        val shader = BitmapShader(source, Shader.TileMode.CLAMP, Shader.TileMode.CLAMP)
        shader.setLocalMatrix(
            Matrix().apply {
                setTranslate(-(contentX - padding).toFloat(), -(contentY - padding).toFloat())
            },
        )
        tilePaint.shader = shader
        canvas.drawRect(0f, 0f, paddedTileSize.toFloat(), paddedTileSize.toFloat(), tilePaint)
        tilePaint.shader = null
    }

    /** Runs full upscale of 'input', tile by tile in batches, and composes the final result. */
    suspend fun upscale(input: Bitmap): Bitmap {
        val scale = model.scale
        val contentSize = model.tileContentSize

        val outW = input.width * scale
        val outH = input.height * scale
        val output = Bitmap.createBitmap(outW, outH, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(output)

        val margin = overlap / 2
        val step = contentSize - (margin * 2)
        val positions = collectTilePositions(input, step)

        positions.chunked(batchSize).forEach { batch ->
            val realCount = batch.size

            // Reuse input Bitmaps and Canvases to extract tiles
            for (i in 0 until batchSize) {
                val pos = if (i < realCount) batch[i] else batch.last() // Pad by duplicating the last one if needed
                drawPaddedTile(input, inputCanvases[i], pos.x, pos.y)
            }

            // Inference
            val results = withContext(inferenceDispatcher) { runBatchInference(inputTiles) }

            // Draw only the real results onto the final canvas
            for (i in 0 until realCount) {
                val pos = batch[i]
                val cropLeft = if (pos.x == 0) 0 else margin * scale
                val cropTop = if (pos.y == 0) 0 else margin * scale
                val cropRight = if (pos.x + contentSize >= input.width) contentSize * scale else (contentSize - margin) * scale
                val cropBottom = if (pos.y + contentSize >= input.height) contentSize * scale else (contentSize - margin) * scale

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

    /** Runs a single inference on the 'tiles' batch, writing/reading pixels via NativePixelOps. */
    private fun runBatchInference(tiles: Array<Bitmap>): Array<Bitmap> {
        val t0 = System.currentTimeMillis()

        // Native C++ write into the reusable input FloatArray
        val tilePixelCount = paddedTileSize * paddedTileSize
        for (i in tiles.indices) {
            val arrayOffset = i * tilePixelCount
            if (inputLayout == TensorLayout.NHWC) {
                NativePixelOps.writeBitmapToArrayNHWC(tiles[i], batchInputArray, arrayOffset, paddedTileSize)
            } else {
                NativePixelOps.writeBitmapToArrayNCHW(tiles[i], batchInputArray, arrayOffset, paddedTileSize)
            }
        }
        val t1 = System.currentTimeMillis()

        // LiteRT CompiledModel run
        inputBuffers[0].writeFloat(batchInputArray)
        compiledModel.run(inputBuffers, outputBuffers)
        val outputArray = outputBuffers[0].readFloat()
        val t2 = System.currentTimeMillis()

        // Native C++ read from the FloatArray into reusable Bitmaps
        // NUOVA LOGICA: il modello restituisce [Batch, paddedTileSize, paddedTileSize, 3 * scale^2]
        // Il PixelShuffle viene fatto on-the-fly dalla CPU in C++
        for (i in tiles.indices) {
            val arrayOffset = i * tilePixelCount

            NativePixelOps.readArrayToBitmapPixelShuffle(
                outArray = outputArray,
                inArray = batchInputArray,
                arrayPixelOffset = arrayOffset,
                targetBitmap = reusableOutputTiles[i],
                inTileSize = paddedTileSize,
                scale = model.scale,
                isInputNhwc = (inputLayout == TensorLayout.NHWC),
                isOutputNhwc = (outputLayout == TensorLayout.NHWC)
            )
        }
        val t3 = System.currentTimeMillis()

        xLogD("Native write: ${t1 - t0}ms | CompiledModel run(): ${t2 - t1}ms | Native read: ${t3 - t2}ms | Total: ${t3 - t0}ms")

        return reusableOutputTiles
    }

    /** Releases the CompiledModel (if created) and stops the dedicated inference thread. */
    fun close() {
        if (compiledModelLazy.isInitialized()) compiledModel.close()
        inferenceExecutor.shutdown()
    }
}
