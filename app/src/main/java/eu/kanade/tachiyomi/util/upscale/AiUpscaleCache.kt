package eu.kanade.tachiyomi.util.upscale

import android.app.Application
import android.graphics.Bitmap
import android.util.Log
import kotlinx.coroutines.sync.Semaphore
import kotlinx.coroutines.sync.withPermit
import okio.BufferedSource
import okio.buffer
import okio.source
import tachiyomi.decoder.ImageDecoder
import uy.kohesive.injekt.Injekt
import uy.kohesive.injekt.api.get
import java.io.File

/**
 * Cache su disco per le pagine già upscalate, chiave = capitolo + indice pagina.
 * Necessaria perché il reader disabilita esplicitamente la cache di Coil per le pagine
 * (vedi ReaderPageImageView), quindi senza questo layer ogni ri-visualizzazione
 * di una pagina già letta rilancerebbe l'inferenza da zero.
 */
object AiUpscaleCache {

    private val context: Application by lazy { Injekt.get() }
    private val upscaler: AiUpscaler by lazy { AiUpscaler(context, UpscaleModel.REALCUGAN_SCALE4X) }
    private val cacheDir: File by lazy {
        File(context.cacheDir, "ai_upscale_cache").apply { mkdirs() }
    }

    // Una sola pipeline di upscaling completa alla volta, per non saturare CPU/GPU
    // quando più pagine sono "in volo" contemporaneamente (viewport + prefetch).
    private val upscaleSemaphore = Semaphore(1)

    suspend fun getOrUpscale(
        chapterId: Long?,
        pageIndex: Int,
        source: BufferedSource,
        targetWidth: Int,
    ): BufferedSource? {
        val file = File(cacheDir, "${chapterId}_$pageIndex.jpg")
        if (file.exists()) {
            return file.source().buffer()
        }

        return upscaleSemaphore.withPermit {
            // Nel frattempo, mentre eravamo in coda, un'altra richiesta per
            // la STESSA pagina potrebbe averla già completata.
            if (file.exists()) return@withPermit file.source().buffer()

            val decoded = try {
                ImageDecoder.newInstance(source.inputStream())?.decode()
            } catch (e: Exception) { null } ?: return@withPermit null

            val resized = if (decoded.width > targetWidth) {
                val scale = targetWidth.toFloat() / decoded.width
                val newHeight = (decoded.height * scale).toInt()
                Bitmap.createScaledBitmap(decoded, targetWidth, newHeight, true).also {
                    if (it !== decoded) decoded.recycle()
                }
            } else {
                decoded
            }

            Log.d("AiUpscaleCache", "Upscaling chapterId: $chapterId, pageIndex: $pageIndex")

            val upscaled = try {
                upscaler.upscale(resized)
            } catch (e: OutOfMemoryError) {
                resized
            }
            if (resized !== decoded && resized !== upscaled) resized.recycle()

            Log.d("AiUpscaleCache", "Upscaling done for chapterId: ${chapterId}, pageIndex: ${pageIndex}")

            try {
                file.outputStream().use { out ->
                    upscaled.compress(Bitmap.CompressFormat.JPEG, 95, out)
                }
                if (upscaled !== decoded && upscaled !== resized) upscaled.recycle()
                file.source().buffer()
            } catch (e: Exception) {
                null
            }
        }
    }
}
