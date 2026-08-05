package eu.kanade.tachiyomi.util.upscale

import android.app.Application
import android.content.res.AssetManager
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Rect
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.withContext
import java.util.concurrent.Executors

class NcnnUpscaler(private val context: Application) {

    init {
        System.loadLibrary("ncnnupscaler")
    }

    private val tileSize = 256
    private val overlap = 0
    private val scale = 4
    private val outSize = tileSize * scale

    // JNI Mappings
    private external fun initNcnn(assetManager: AssetManager): Boolean
    private external fun processTile(bitmapIn: Bitmap, bitmapOut: Bitmap)
    private external fun destroyNcnn()

    // Thread dedicato per non bloccare mai la UI
    private val inferenceExecutor = Executors.newSingleThreadExecutor { r ->
        Thread(r, "NcnnUpscaler-Inference")
    }
    private val inferenceDispatcher = inferenceExecutor.asCoroutineDispatcher()

    private var isInitialized = false

    // Riutilizziamo una sola coppia di Bitmap per tutti i tile, azzerando il Garbage Collector
    private val tempInputTile by lazy { Bitmap.createBitmap(tileSize, tileSize, Bitmap.Config.ARGB_8888) }
    private val tempOutputTile by lazy { Bitmap.createBitmap(outSize, outSize, Bitmap.Config.ARGB_8888) }
    private val tempInputCanvas by lazy { Canvas(tempInputTile) }

    private data class TilePos(val x: Int, val y: Int)

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

    suspend fun upscale(input: Bitmap): Bitmap {
        val outW = input.width * scale
        val outH = input.height * scale
        val output = Bitmap.createBitmap(outW, outH, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(output)

        val margin = overlap / 2
        val step = tileSize - (margin * 2)
        val positions = collectTilePositions(input, step)

        withContext(inferenceDispatcher) {
            // Inizializza Vulkan e Modello solo alla prima chiamata
            if (!isInitialized) {
                isInitialized = initNcnn(context.assets)
                if (!isInitialized) error("Impossibile inizializzare ncnn")
            }

            positions.forEach { pos ->
                // 1. Estrai il tile di input nel buffer riutilizzabile
                tempInputCanvas.drawBitmap(
                    input,
                    Rect(pos.x, pos.y, pos.x + tileSize, pos.y + tileSize),
                    Rect(0, 0, tileSize, tileSize),
                    null
                )

                // 2. Esegui l'inferenza C++ sincrona (zero allocazioni)
                processTile(tempInputTile, tempOutputTile)

                // 3. Calcola i bordi di crop e disegna sul canvas finale
                val cropLeft = if (pos.x == 0) 0 else margin * scale
                val cropTop = if (pos.y == 0) 0 else margin * scale
                val cropRight = if (pos.x + tileSize >= input.width) tileSize * scale else (tileSize - margin) * scale
                val cropBottom = if (pos.y + tileSize >= input.height) tileSize * scale else (tileSize - margin) * scale

                val srcRect = Rect(cropLeft, cropTop, cropRight, cropBottom)
                val destLeft = (pos.x * scale) + cropLeft
                val destTop = (pos.y * scale) + cropTop
                val destRight = (pos.x * scale) + cropRight
                val destBottom = (pos.y * scale) + cropBottom

                canvas.drawBitmap(
                    tempOutputTile,
                    srcRect,
                    Rect(destLeft, destTop, destRight, destBottom),
                    null
                )
            }
        }
        return output
    }

    // Chiamalo quando chiudi il reader (es. onDestroy dell'Activity)
    fun destroy() {
        inferenceExecutor.execute {
            if (isInitialized) {
                destroyNcnn()
                isInitialized = false
            }
        }
    }
}
