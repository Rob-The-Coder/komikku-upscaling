package eu.kanade.tachiyomi.util.upscale

import android.util.Log
import eu.kanade.tachiyomi.ui.reader.model.ReaderPage
import eu.kanade.tachiyomi.ui.reader.setting.ReaderPreferences
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import okio.Buffer
import tachiyomi.core.common.util.system.ImageUtil
import uy.kohesive.injekt.Injekt
import uy.kohesive.injekt.api.get
import java.util.concurrent.ConcurrentHashMap

/**
 * Precarica in cache l'upscaling delle N pagine successive a quella corrente,
 * mentre l'utente sta ancora leggendo quella attuale (idle time), invece di
 * partire on-demand esattamente quando la pagina diventa visibile.
 *
 * Scope dedicato, di lunga durata per design (deve sopravvivere al riciclo
 * delle singole holder) — a differenza del bug con GlobalScope visto prima,
 * qui è intenzionale: la concorrenza reale resta comunque limitata dal
 * semaforo già presente in AiUpscaleCache.
 */
object AiUpscalePrefetcher {

    private val prefetchScope = CoroutineScope(SupervisorJob() + Dispatchers.IO)

    // Dedup: evita di rilanciare decode+letura per una pagina già richiesta
    // (l'eventuale duplicato viene comunque bloccato anche a valle, nel
    // controllo file.exists() dentro AiUpscaleCache, ma qui evitiamo di
    // sprecare anche la lettura/decodifica dei byte originali).
    private val requested = ConcurrentHashMap.newKeySet<String>()

    fun schedulePrefetch(current: ReaderPage, aheadCount: Int = 2, targetWidth: Int) {
        val upscalePrefs = Injekt.get<ReaderPreferences>()
        if (!upscalePrefs.aiUpscaleEnabled().get()) return

        val chapter = current.chapter
        val loader = chapter.pageLoader ?: return
        val pages = chapter.pages ?: return

        for (offset in 1..aheadCount) {
            val nextPage = pages.getOrNull(current.index + offset) ?: continue
            val key = "${nextPage.chapter.chapter.id}_${nextPage.index}"
            if (!requested.add(key)) continue

            prefetchScope.launch {
                try {
                    loader.loadPage(nextPage)
                    val streamFn = nextPage.stream ?: return@launch

                    val bytes = withContext(Dispatchers.IO) {
                        streamFn().use { it.readBytes() }
                    }

                    val isAnimated = ImageUtil.isAnimatedAndSupported(Buffer().write(bytes))
                    if (isAnimated) return@launch

                    AiUpscaleCache.getOrUpscale(
                        chapterId = nextPage.chapter.chapter.id,
                        pageIndex = nextPage.index,
                        source = Buffer().write(bytes),
                        targetWidth = targetWidth,
                    )
                } catch (e: Throwable) {
                    Log.w("AiUpscalePrefetch", "Prefetch fallito pagina ${nextPage.index}", e)
                }
            }
        }
    }

    // Da chiamare quando si cambia capitolo, per non far crescere la Set
    // all'infinito durante una sessione di lettura lunga.
    fun clear() {
        requested.clear()
    }
}
