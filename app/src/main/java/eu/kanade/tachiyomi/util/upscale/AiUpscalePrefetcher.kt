package eu.kanade.tachiyomi.util.upscale

import android.util.Log
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
import kotlinx.coroutines.supervisorScope
import kotlinx.coroutines.withContext
import kotlinx.coroutines.withTimeoutOrNull
import okio.Buffer
import tachiyomi.core.common.util.lang.launchIO
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

    // Dedup: evita di rilanciare decode+letura per una pagina già richiesta
    // (l'eventuale duplicato viene comunque bloccato anche a valle, nel
    // controllo file.exists() dentro AiUpscaleCache, ma qui evitiamo di
    // sprecare anche la lettura/decodifica dei byte originali).
    private val requested = ConcurrentHashMap.newKeySet<String>()

    @Volatile private var currentPages: List<ReaderPage>? = null
    @Volatile private var currentIndex: Int = -1
    @Volatile private var aheadCount: Int = 2
    @Volatile private var targetWidth: Int = 0
    private var fillJob: Job? = null

    private suspend fun tryFillNextGap(): Boolean {
        val pages = currentPages ?: run {
            Log.d("AiUpscalePrefetch", "tryFillNextGap: currentPages è null")
            return false
        }
        val loader = pages.firstOrNull()?.chapter?.pageLoader ?: run {
            Log.d("AiUpscalePrefetch", "tryFillNextGap: pageLoader è null")
            return false
        }

        for (offset in 1..aheadCount) {
            val nextPage = pages.getOrNull(currentIndex + offset) ?: continue
            val key = "${nextPage.chapter.chapter.id}_${nextPage.index}"
            if (requested.contains(key)) {
                continue
            }

            try {
                prefetchScope.launch(Dispatchers.IO) {
                    try {
                        loader.loadPage(nextPage)
                    } catch (e: Throwable) {
                        Log.w("AiUpscalePrefetch", "loadPage fallita per pagina ${nextPage.index}", e)
                    }
                }   // fire-and-forget vero: lanciato nello scope di lunga durata del prefetcher,
                    // MAI atteso/joinato da questa funzione — esattamente come richiede il commento
                    // in PageLoader.kt ("should be launched asynchronously")

                val readyState = withTimeoutOrNull(2_000.milliseconds) {
                    nextPage.statusFlow.first { it is Page.State.Ready || it is Page.State.Error }
                }

                if (readyState !is Page.State.Ready) {
                    Log.d("AiUpscalePrefetch", "offset=$offset pagina ${nextPage.index}: non pronta entro il timeout (stato=$readyState)")
                    continue
                }

                val streamFn = nextPage.stream
                if (streamFn == null) {
                    Log.w("AiUpscalePrefetch", "offset=$offset pagina ${nextPage.index}: Ready ma stream null")
                    requested.add(key)
                    continue
                }

                Log.d("AiUpscalePrefetch", "offset=$offset pagina ${nextPage.index}: avvio upscale")
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
                Log.d("AiUpscalePrefetch", "offset=$offset pagina ${nextPage.index}: completato")
                return true
            } catch (e: Throwable) {
                Log.e("AiUpscalePrefetch", "offset=$offset pagina ${nextPage.index}: eccezione", e)
            }
        }
        Log.d("AiUpscalePrefetch", "nessuna pagina processabile in questo giro (currentIndex=$currentIndex, aheadCount=$aheadCount)")
        return false
    }

    private val exceptionHandler = CoroutineExceptionHandler { _, throwable ->
        Log.e("AiUpscalePrefetch", "fillLoop terminato per eccezione non gestita", throwable)
    }
    private val prefetchScope = CoroutineScope(SupervisorJob() + Dispatchers.IO + exceptionHandler)

    private suspend fun fillLoop() {
        Log.d("AiUpscalePrefetch", "fillLoop avviato")
        while (currentCoroutineContext().isActive) {
            val processedSomething = try {
                tryFillNextGap()
            } catch (e: Throwable) {
                Log.e("AiUpscalePrefetch", "tryFillNextGap ha lanciato un'eccezione", e)
                false
            }
            if (!processedSomething) {
                delay(500.milliseconds)
            }
        }
        Log.d("AiUpscalePrefetch", "fillLoop terminato")
    }

    /**
     * Aggiorna la posizione di lettura corrente. Non avvia direttamente il
     * prefetch di N pagine come prima: aggiorna solo lo stato che il loop
     * continuo (avviato una volta sola) legge ad ogni iterazione. Chiamare
     * ad ogni cambio pagina, non solo alla prima.
     */
    fun updatePosition(
        current: ReaderPage,
        aheadCount: Int,
        targetWidth: Int,
    ) {
        val upscalePrefs = Injekt.get<ReaderPreferences>()
        val enabled = upscalePrefs.aiUpscaleEnabled().get()
        Log.d("AiUpscalePrefetch", "updatePosition chiamato: enabled=$enabled, index=${current.index}, fillJob.isActive=${fillJob?.isActive}")
        if (!enabled) return

        currentPages = current.chapter.pages
        currentIndex = current.index
        this.aheadCount = aheadCount
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
        currentPages = null
        currentIndex = -1
    }
}
