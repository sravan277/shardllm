package com.dllm.mesh.data

/**
 * Single source of truth for the served model's layer topology.
 *
 * WHY centralised: the layer count is a property of the *model*, not of any one
 * screen, but it was previously copy-pasted as the literal `28` / `0` / `27` in
 * three places (worker capabilities, the usage strip, the chat split row). When
 * the served model changes shape those copies drift apart and the UI starts
 * claiming a device owns all 28 layers when the coordinator assigned 32, or
 * offers a range the planner can never satisfy. One file, one truth.
 *
 * These are NOT measurements of this phone — they describe the model being
 * served. Nothing here is inferred at runtime.
 */
object ModelTopology {

    /**
     * Total decoder layers of the served model (28 = the Qwen3-0.6B shape the
     * coordinator currently serves). Used as the `capabilities.layers` figure
     * the backend `/v1/plan` planner divides across workers.
     */
    const val TOTAL_LAYERS: Int = 28

    /** First layer index. Kept explicit so an off-by-one is visible at the call site. */
    const val LAYER_START: Int = 0

    /** Last layer index, inclusive. */
    const val LAYER_END: Int = TOTAL_LAYERS - 1

    /**
     * Every layer index this build knows about, ascending.
     *
     * Used where the app must state the whole model is being covered: a phone
     * that is the only worker offers the full range, and the UI says "all
     * layers" only when the distinct assigned set really is [LAYER_START]
     * through [LAYER_END].
     */
    fun allLayers(): List<Int> = (LAYER_START..LAYER_END).toList()
}

/**
 * Compresses an expanded layer list into human ranges: `[0,1,2,5,6]` -> `"0–2, 5–6"`.
 *
 * WHY it lives here: this exact function was copy-pasted into both the chat
 * split row and the usage strip, so the two screens could disagree about how a
 * plan renders. One implementation means a plan always reads identically
 * everywhere. Empty input renders `"none"` (never an empty string, which would
 * read as a rendering bug).
 */
fun layerRangeText(layers: List<Int>): String {
    if (layers.isEmpty()) return "none"
    val sorted = layers.distinct().sorted()
    val ranges = ArrayList<String>()
    var start = sorted[0]
    var prev = sorted[0]
    for (i in 1..sorted.size) {
        // Int.MIN_VALUE sentinel closes the final run without a special case.
        val current = if (i < sorted.size) sorted[i] else Int.MIN_VALUE
        if (current == prev + 1) {
            prev = current
            continue
        }
        ranges.add(if (start == prev) "$start" else "$start–$prev")
        if (i < sorted.size) {
            start = current
            prev = current
        }
    }
    return ranges.joinToString(", ")
}

/**
 * True when [layers] covers the whole model exactly once — the single-worker
 * fast path. Compared against [ModelTopology] so the "all layers" claim can
 * never be true for a stale hardcoded range.
 */
fun coversWholeModel(layers: Collection<Int>): Boolean =
    layers.size == ModelTopology.TOTAL_LAYERS &&
        layers.minOrNull() == ModelTopology.LAYER_START &&
        layers.maxOrNull() == ModelTopology.LAYER_END
