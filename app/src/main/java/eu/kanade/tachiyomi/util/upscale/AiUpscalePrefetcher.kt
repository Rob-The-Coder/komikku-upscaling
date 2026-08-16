package eu.kanade.tachiyomi.util.upscale

import android.util.Log
import eu.kanade.tachiyomi.BuildConfig
import eu.kanade.tachiyomi.source.model.Page
import eu.kanade.tachiyomi.ui.reader.model.ReaderPage
import eu.kanade.tachiyomi.ui.reader.setting.ReaderPreferences
import kotlinx.coroutines.CoroutineExceptionHandler
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import kotlinx.coroutines.withTimeoutOrNull
import okio.Buffer
import tachiyomi.core.common.util.system.ImageUtil
import uy.kohesive.injekt.Injekt
import uy.kohesive.injekt.api.get
import java.util.concurrent.ConcurrentHashMap
import kotlin.time.Duration.Companion.milliseconds

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

    private val exceptionHandler = CoroutineExceptionHandler { _, throwable ->
        if (BuildConfig.DEBUG) Log.e("AiUpscalePrefetch", "fillLoop stopped due to unhandled exception", throwable)
    }
    private val prefetchScope = CoroutineScope(SupervisorJob() + Dispatchers.IO + exceptionHandler)

    // Dedup: evita di rilanciare decode+letura per una pagina già richiesta
    // (l'eventuale duplicato viene comunque bloccato anche a valle, nel
    // controllo file.exists() dentro AiUpscaleCache, ma qui evitiamo di
    // sprecare anche la lettura/decodifica dei byte originali).
    private val requested = ConcurrentHashMap.newKeySet<String>()
    @Volatile private var currentChapterPages: List<ReaderPage>? = null
    @Volatile private var nextChapterProvider: (() -> List<ReaderPage>?)? = null
    @Volatile private var currentIndex: Int = -1
    @Volatile private var aheadCount: Int = 2
    @Volatile private var alreadyCoveredAhead: Int = 0
    @Volatile private var targetWidth: Int = 0
    private var fillJob: Job? = null

    /**
     * Restituisce la pagina all'offset richiesto rispetto alla posizione corrente,
     * attraversando il confine di capitolo se necessario. Se l'offset ricade nel
     * prossimo capitolo ma questo non è ancora caricato (pages == null), ritorna
     * null: il chiamante la riproverà al giro successivo del loop, non è un
     * fallimento definitivo.
     */
    private fun pageAtOffset(offset: Int): ReaderPage? {
        val curPages = currentChapterPages ?: return null
        val idx = currentIndex + offset
        if (idx < curPages.size) return curPages.getOrNull(idx)

        val overflow = idx - curPages.size
        val nextPages = nextChapterProvider?.invoke() ?: return null
        return nextPages.getOrNull(overflow)
    }
    private suspend fun tryFillNextGap(): Boolean {
        if (currentChapterPages == null) {
            if (BuildConfig.DEBUG) Log.d("AiUpscalePrefetch", "tryFillNextGap: currentChapterPages is null")
            return false
        }

        val startOffset = alreadyCoveredAhead + 1
        if (startOffset > aheadCount) return false

        for (offset in startOffset..aheadCount) {
            val nextPage = pageAtOffset(offset) ?: continue
            val key = "${nextPage.chapter.chapter.id}_${nextPage.index}"
            if (requested.contains(key)) continue

            // Risolto per-pagina, non una volta per l'intera finestra: pagine oltre
            // il confine di capitolo appartengono a un pageLoader diverso.
            val loader = nextPage.chapter.pageLoader
            if (loader == null) {
                if (BuildConfig.DEBUG) Log.d("AiUpscalePrefetch", "offset=$offset page ${nextPage.index}: pageLoader not ready (chapter not started)")
                continue
            }

            try {
                prefetchScope.launch(Dispatchers.IO) {
                    try {
                        loader.loadPage(nextPage)
                    } catch (e: Throwable) {
                        if (BuildConfig.DEBUG) Log.w("AiUpscalePrefetch", "failed loadPage for page ${nextPage.index}", e)
                    }
                }

                val readyState = withTimeoutOrNull(15_000.milliseconds) {
                    nextPage.statusFlow.first { it is Page.State.Ready || it is Page.State.Error }
                }

                if (readyState !is Page.State.Ready) {
                    if (BuildConfig.DEBUG) Log.d("AiUpscalePrefetch", "offset=$offset page ${nextPage.index}: not ready within timeout (state=$readyState)")
                    continue
                }

                val streamFn = nextPage.stream
                if (streamFn == null) {
                    if (BuildConfig.DEBUG) Log.w("AiUpscalePrefetch", "offset=$offset page ${nextPage.index}: Ready but null stream")
                    requested.add(key)
                    continue
                }

                if (BuildConfig.DEBUG) Log.d("AiUpscalePrefetch", "offset=$offset page ${nextPage.index}: starting upscale")
                val bytes = withContext(Dispatchers.IO) { streamFn().use { it.readBytes() } }
                if (!ImageUtil.isAnimatedAndSupported(Buffer().write(bytes))) {
                    AiUpscaleCache.getOrUpscale(
                        chapterId = nextPage.chapter.chapter.id,
                        pageIndex = nextPage.index,
                        source = Buffer().write(bytes),
                        targetWidth = targetWidth,
                        priority = UpscalePriorityGate.Priority.PREFETCH,
                    )
                }
                requested.add(key)
                if (BuildConfig.DEBUG) Log.d("AiUpscalePrefetch", "offset=$offset page ${nextPage.index}: completed")
                return true
            } catch (e: Throwable) {
                if (BuildConfig.DEBUG) Log.e("AiUpscalePrefetch", "offset=$offset page ${nextPage.index}: exception", e)
            }
        }
        if (BuildConfig.DEBUG) Log.d("AiUpscalePrefetch", "no processable page (currentIndex=$currentIndex, aheadCount=$aheadCount)")
        return false
    }
    private suspend fun fillLoop() {
        if (BuildConfig.DEBUG) Log.d("AiUpscalePrefetch", "fillLoop started")
        while (currentCoroutineContext().isActive) {
            val processedSomething = try {
                tryFillNextGap()
            } catch (e: Throwable) {
                if (BuildConfig.DEBUG) Log.e("AiUpscalePrefetch", "tryFillNextGap threw an exception", e)
                false
            }
            if (!processedSomething) {
                delay(500.milliseconds)
            }
        }
        if (BuildConfig.DEBUG) Log.d("AiUpscalePrefetch", "fillLoop terminated")
    }

    /**
     * Aggiorna la posizione di lettura corrente. Non avvia direttamente il
     * prefetch di N pagine come prima: aggiorna solo lo stato che il loop
     * continuo (avviato una volta sola) legge ad ogni iterazione. Chiamare
     * ad ogni cambio pagina, non solo alla prima.
     * @param nextChapterProvider fornisce le pagine del capitolo successivo, se
     * disponibili. Passato come lambda (non come lista già risolta) perché il
     * capitolo potrebbe non essere ancora caricato al momento di questa chiamata
     * ma diventarlo mentre il loop continua a girare — la lambda viene rivalutata
     * ad ogni tentativo, non catturata una volta sola.
     */
    fun updatePosition(
        current: ReaderPage,
        aheadCount: Int,
        targetWidth: Int,
        alreadyCoveredAhead: Int = 0,
        nextChapterProvider: () -> List<ReaderPage>? = { null },
    ) {
        val upscalePrefs = Injekt.get<ReaderPreferences>()
        if (!upscalePrefs.aiUpscaleEnabled().get()) return

        currentChapterPages = current.chapter.pages
        this.nextChapterProvider = nextChapterProvider
        currentIndex = current.index
        this.aheadCount = aheadCount
        this.alreadyCoveredAhead = alreadyCoveredAhead
        this.targetWidth = targetWidth

        if (fillJob?.isActive != true) {
            fillJob = prefetchScope.launch { fillLoop() }
        }
    }

    // Da chiamare quando si cambia capitolo, per non far crescere la Set
    // all'infinito durante una sessione di lettura lunga.
    fun clear() {
        requested.clear()
        fillJob?.cancel()
        fillJob = null
        currentChapterPages = null
        nextChapterProvider = null
        currentIndex = -1
    }
}
