package eu.kanade.tachiyomi.util.upscale

/**
 * Descrive un modello di upscaling disponibile. `offset` è la quantità di
 * pixel "mangiati" dalla rete per via di convoluzioni senza padding interno
 * (valid convolution) — 0 per modelli same-padding come Real-ESRGAN, >0 per
 * modelli come waifu2x/upconv_7 che restringono il tile attraversando la rete:
 *   output_reale = scale × tileContentSize
 *   input_da_dare_alla_rete = tileContentSize + offset / scale
 *
 * Per aggiungere un nuovo modello in futuro: converti, misura l'offset reale
 * (vedi verifyOffset in fondo al file per come derivarlo empiricamente se non
 * lo conosci dall'architettura), e aggiungi una entry qui — nient'altro nel
 * resto della classe AiUpscaler richiede modifiche.
 */
enum class UpscaleModel(
    val displayName: String,
    val assetFileName: String,
    val scale: Int,
    val batch_size: Int,
    val tileContentSize: Int,
    val offset: Int,
) {
    REALESRGAN_ANIMEVIDEOV3(
        displayName = "Real-ESRGAN (animevideov3, 4x)",
        assetFileName = "realesr_animevideov3_x4_384T_B3_float32.tflite",
        scale = 4,
        tileContentSize = 384,
        batch_size = 3,
        offset = 0,
    ),
    WAIFU2X_SCALE2X(
        displayName = "waifu2x upconv_7 (2x, no denoise)",
        assetFileName = "waifu2x_float32.tflite",
        scale = 2,
        tileContentSize = 384,
        batch_size = 1,
        offset = 28,
    ),
    WAIFU2X_SCALE2X_B3(
        displayName = "waifu2x B3(2x, no denoise)",
        assetFileName = "waifu2x_384T_B3_float32.tflite",
        scale = 2,
        tileContentSize = 384,
        batch_size = 3,
        offset = 28,
    ),
    WAIFU2X_NOISE0_SCALE2X_B3(
        displayName = "waifu2x (2x, denoise leggero)",
        assetFileName = "waifu2x_noise0_384T_B3_float32.tflite",
        scale = 2,
        tileContentSize = 384,
        batch_size = 3,
        offset = 28,
    ),
    WAIFU2X_NOISE1_SCALE2X_B3(
        displayName = "waifu2x (2x, denoise medio)",
        assetFileName = "waifu2x_noise1_384T_B3_float32.tflite",
        scale = 2,
        tileContentSize = 384,
        batch_size = 3,
        offset = 28,
    ),
    CUNET_SCALE2X_B1(
        displayName = "cunet2x (2x)",
        assetFileName = "cunet2x_384T_float32.tflite",
        scale = 2,
        tileContentSize = 384,
        batch_size = 1,
        offset = 72,
    ),
    SWIN_SCALE2X(
        displayName = "swin2x (2x)",
        assetFileName = "swin2x_float32.tflite",
        scale = 2,
        tileContentSize = 384,
        batch_size = 1,
        offset = 32,
    ),
    REALCUGAN_SCALE2X(
        displayName = "realcugan (2x)",
        assetFileName = "realcugan2x_no_denoise_384T_B1_float32.tflite",
        scale = 2,
        tileContentSize = 384,
        batch_size = 1,
        offset = 72,
    ),
    REALCUGAN_SCALE2X_B3(
        displayName = "Real-CUGAN (2x)",
        assetFileName = "realcugan2x_no_denoise_384T_B3_float32.tflite",
        scale = 2,
        tileContentSize = 384,
        batch_size = 3,
        offset = 72,
    ),
    REALCUGAN_SCALE3X(
        displayName = "Real-CUGAN (3x)",
        assetFileName = "realcugan3x_no_denoise_384T_B1_float32.tflite",
        scale = 3,
        tileContentSize = 384,
        batch_size = 1,
        offset = 84,   // = paddingPerSide(14) × 2 × scale(3)
    ),
    REALCUGAN_SCALE3X_B3(
        displayName = "Real-CUGAN (3x)",
        assetFileName = "realcugan3x_no_denoise_384T_B3_float32.tflite",
        scale = 3,
        tileContentSize = 384,
        batch_size = 3,
        offset = 84,   // = paddingPerSide(14) × 2 × scale(3)
    ),
    REALCUGAN_SCALE4X(
        displayName = "Real-CUGAN (4x)",
        assetFileName = "realcugan4x_no_denoise_384T_B1_float32.tflite",
        scale = 4,
        tileContentSize = 384,
        batch_size = 1,
        offset = 152,  // = paddingPerSide(19) × 2 × scale(4)
    ),
    ;

    init {
        require(offset % (2 * scale) == 0) {
            "offset=$offset non divisibile per 2*scale=${2 * scale}: il padding per lato non sarebbe un intero, ricontrolla il valore per $name"
        }
    }

    /** Pixel di padding da aggiungere per lato quando si estrae il tile dalla pagina sorgente. */
    val paddingPerSide: Int get() = offset / (2 * scale)

    /** Dimensione reale del tile da dare in input al modello (contenuto + padding). */
    val paddedTileSize: Int get() = tileContentSize + 2 * paddingPerSide

    /** Dimensione dell'output prodotto dal modello per un singolo tile. */
    val outSize: Int get() = scale * tileContentSize
}
