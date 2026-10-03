package com.dllm.mesh.net

import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.view.CameraController
import androidx.camera.view.LifecycleCameraController
import androidx.camera.view.PreviewView
import androidx.camera.mlkit.vision.MlKitAnalyzer
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import com.google.mlkit.vision.barcode.BarcodeScannerOptions
import com.dllm.mesh.ui.MeshColors
import com.google.mlkit.vision.barcode.BarcodeScanning
import com.google.mlkit.vision.barcode.common.Barcode
import org.json.JSONObject

data class JoinPayload(
    val url: String,
    val nodeId: String,
    val fingerprint: String,
)

/**
 * Phase 0 join payload. Real pairing crypto (signatures, verify code,
 * allow-list) lands in Phase 3 — fingerprint is a placeholder until then.
 */
fun buildJoinPayload(
    coordinatorUrl: String,
    nodeId: String,
    fingerprint: String = "TODO-phase3",
): String = JSONObject()
    .put("v", 1)
    .put("url", coordinatorUrl)
    .put("node_id", nodeId)
    .put("fingerprint", fingerprint)
    .toString()

fun parseJoinPayload(raw: String): JoinPayload? = runCatching {
    val obj = JSONObject(raw.trim())
    val url = obj.optString("url").trim()
    if (url.isEmpty()) return@runCatching null
    JoinPayload(
        url = url,
        nodeId = obj.optString("node_id"),
        fingerprint = obj.optString("fingerprint"),
    )
}.getOrNull()

/**
 * Show-join-info screen. No zxing dependency in Phase 0, so this draws a
 * selectable-text fallback card instead of a QR bitmap (TODO Phase 3).
 */
@Composable
fun QrShowScreen(payload: String, modifier: Modifier = Modifier) {
    Column(modifier = modifier.fillMaxSize().padding(16.dp)) {
        Text("Show this to a new device", color = MeshColors.Text)
        Spacer(Modifier.height(8.dp))
        Card(colors = CardDefaults.cardColors(containerColor = MeshColors.Panel)) {
            SelectionContainer {
                Text(
                    text = payload,
                    modifier = Modifier.padding(16.dp),
                    color = MeshColors.Teal,
                    fontFamily = FontFamily.Monospace,
                )
            }
        }
        Spacer(Modifier.height(8.dp))
        Text(
            "QR bitmap lands in Phase 3. Until then, copy this text to the joining device.",
            color = MeshColors.Muted,
        )
    }
}

/**
 * CameraX + ML Kit QR scan screen (camera-view 1.3.4 + barcode-scanning 17.3.0).
 *
 * Caller MUST hold android.permission.CAMERA before showing this. Includes a
 * paste fallback for emulators / denied permission.
 */
@Composable
fun QrScanScreen(
    onScanned: (String) -> Unit,
    onCancel: () -> Unit,
    modifier: Modifier = Modifier,
) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    var manual by remember { mutableStateOf("") }
    var scanned by remember { mutableStateOf(false) }

    val barcodeScanner = remember {
        val options = BarcodeScannerOptions.Builder()
            .setBarcodeFormats(Barcode.FORMAT_QR_CODE)
            .build()
        BarcodeScanning.getClient(options)
    }
    val controller = remember {
        LifecycleCameraController(context).apply {
            cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA
            setEnabledUseCases(CameraController.IMAGE_ANALYSIS)
            imageAnalysisBackpressureStrategy = ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST
        }
    }
    val analyzer = remember {
        MlKitAnalyzer(
            listOf(barcodeScanner),
            CameraController.COORDINATE_SYSTEM_VIEW_REFERENCED,
            ContextCompat.getMainExecutor(context),
        ) { result ->
            val barcodes = result.getValue(barcodeScanner)
            val raw = barcodes?.firstOrNull()?.rawValue
            if (!scanned && !raw.isNullOrEmpty()) {
                scanned = true
                onScanned(raw)
            }
        }
    }

    DisposableEffect(controller, analyzer, lifecycleOwner) {
        controller.setImageAnalysisAnalyzer(ContextCompat.getMainExecutor(context), analyzer)
        controller.bindToLifecycle(lifecycleOwner)
        onDispose {
            runCatching { controller.unbind() }
            runCatching { barcodeScanner.close() }
        }
    }

    Column(modifier = modifier.fillMaxSize().padding(16.dp)) {
        Text("Scan coordinator code", color = MeshColors.Text)
        Spacer(Modifier.height(8.dp))
        Box(modifier = Modifier.fillMaxWidth().height(320.dp)) {
            AndroidView(
                factory = { ctx -> PreviewView(ctx).apply { this.controller = controller } },
                modifier = Modifier.fillMaxSize(),
            )
        }
        Spacer(Modifier.height(12.dp))
        OutlinedTextField(
            value = manual,
            onValueChange = { manual = it },
            label = { Text("Or paste join text") },
            modifier = Modifier.fillMaxWidth(),
        )
        Spacer(Modifier.height(8.dp))
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Button(
                onClick = { if (manual.isNotBlank()) onScanned(manual.trim()) },
                enabled = manual.isNotBlank(),
            ) { Text("Use pasted text") }
            OutlinedButton(onClick = onCancel) { Text("Cancel") }
        }
    }
}
