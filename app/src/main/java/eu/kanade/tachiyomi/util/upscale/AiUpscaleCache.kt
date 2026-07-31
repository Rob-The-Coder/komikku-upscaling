package eu.kanade.tachiyomi.util.upscale

import android.app.Application
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import okio.BufferedSource
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
    private val upscaler: AiUpscaler by lazy { AiUpscaler(context) }

    private val cacheDir: File by lazy {
        File(context.cacheDir, "ai_upscale_cache").apply { mkdirs() }
    }

    fun getOrUpscale(chapterId: Long?, pageIndex: Int, source: BufferedSource): Bitmap? {
        val file = File(cacheDir, "${chapterId}_$pageIndex.jpg")

        if (file.exists()) {
            BitmapFactory.decodeFile(file.absolutePath)?.let { return it }
        }

        val original = try {
            ImageDecoder.newInstance(source.inputStream())?.decode()
        } catch (e: Exception) {
            null
        } ?: return null

        val upscaled = upscaler.upscale(original)
        original.recycle()

        try {
            file.outputStream().use { out ->
                upscaled.compress(Bitmap.CompressFormat.PNG, 95, out)
            }
        } catch (e: Exception) {
            // se il salvataggio fallisce va bene lo stesso, mostriamo l'immagine
            // upscalata, semplicemente non sarà cachata
        }

        return upscaled
    }
}
