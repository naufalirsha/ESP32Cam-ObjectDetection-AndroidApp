package com.example.TA_Project

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.net.Uri
import android.os.Bundle
import android.util.Log
import android.widget.ImageView
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.compose.foundation.Image
import androidx.compose.ui.graphics.asImageBitmap
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.compose.ui.text.style.TextAlign
import kotlinx.coroutines.*
import kotlinx.coroutines.channels.Channel
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.nnapi.NnApiDelegate
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import java.io.BufferedInputStream
import java.io.ByteArrayOutputStream
import java.io.FileInputStream
import java.net.URL
import java.nio.channels.FileChannel
import androidx.compose.ui.graphics.Color as ComposeColor
import android.speech.tts.TextToSpeech
import androidx.compose.runtime.getValue
import androidx.compose.runtime.setValue

private const val TAG = "MainActivity"

// JPEG markers
private const val SOI_MARKER_1 = 0xFF
private const val SOI_MARKER_2 = 0xD8
private const val EOI_MARKER_1 = 0xFF
private const val EOI_MARKER_2 = 0xD9

// Konstanta Kalibrasi
private val OBJECT_WIDTHS = mapOf(
    "BusstopSign" to 60.0f,
    "CrossingSign" to 60.0f
)
private const val FOCAL_LENGTH = 229.45f

enum class ConnectionStatus { CONNECTED, DISCONNECTED, CONNECTING, FAILED }

data class DetectionResult(val boundingBox: RectF, val label: String, val score: Float)
data class ProcessedData(
    val bitmap: Bitmap,
    val fps: Float,
    val objectName: String,
    val distance: Float,
    val accuracy: Float,
    val position: String,
    val logs: List<String> = emptyList(),
    val startTime: Long
)

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            Surface(modifier = Modifier.fillMaxSize(), color = ComposeColor.White) {
                ObjectDetectionApp()
            }
        }
    }
}

// =================================================================================================
// 1. UI COMPONENTS
// =================================================================================================

@Composable
fun SimpleNavBar() {
    Surface(modifier = Modifier.fillMaxWidth().height(56.dp), color = ComposeColor.Black) {
        Box(contentAlignment = Alignment.Center) {
            Text("Object Detection", color = ComposeColor.White, fontSize = 20.sp, fontWeight = FontWeight.Black)
        }
    }
}

@Composable
fun TopBar(connectionStatus: ConnectionStatus, modifier: Modifier = Modifier) {
    val (text, circleColor) = when (connectionStatus) {
        ConnectionStatus.CONNECTED -> "Online" to ComposeColor(0xFF4CAF50)
        ConnectionStatus.DISCONNECTED -> "Offline" to ComposeColor(0xFFF44336)
        ConnectionStatus.CONNECTING -> "Connecting" to ComposeColor(0xFFFFC107)
        ConnectionStatus.FAILED -> "Failed" to ComposeColor.Gray
    }
    Card(modifier = modifier, shape = RoundedCornerShape(5.dp), colors = CardDefaults.cardColors(ComposeColor.Black)) {
        Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.padding(horizontal = 10.dp, vertical = 6.dp)) {
            Box(modifier = Modifier.size(8.dp).clip(CircleShape).background(circleColor))
            Spacer(Modifier.width(8.dp))
            Text(text, color = ComposeColor.White, fontSize = 13.sp, fontWeight = FontWeight.Bold)
        }
    }
}

@Composable
fun InfoCard(fps: Float, objectName: String, distance: Float, accuracy: Float) {
    Column(modifier = Modifier.fillMaxWidth().padding(horizontal = 16.dp)) {
        Row(modifier = Modifier.fillMaxWidth()) {
            InfoTile("Estimation", if (distance > 0f) "${distance.toInt()} cm" else "-", "diff.png", Modifier.weight(1f))
            InfoTile("Object", objectName, "object.png", Modifier.weight(1f))
        }
        Row(modifier = Modifier.fillMaxWidth()) {
            InfoTile("FPS", fps.toInt().toString(), "fps.png", Modifier.weight(1f))
            InfoTile("Accuracy", "${(accuracy * 100).toInt()}%", "accuracy.png", Modifier.weight(1f))
        }
    }
}

@Composable
fun InfoTile(label: String, value: String, assetName: String, modifier: Modifier = Modifier) {
    val context = LocalContext.current
    val imageBitmap = remember(assetName) {
        try { context.assets.open(assetName).use { BitmapFactory.decodeStream(it).asImageBitmap() } } catch (e: Exception) { null }
    }
    Card(modifier = modifier.padding(6.dp), shape = RoundedCornerShape(12.dp), colors = CardDefaults.cardColors(ComposeColor(0xFFEFEFF0))) {
        Column(modifier = Modifier.fillMaxWidth().padding(16.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.SpaceBetween, modifier = Modifier.fillMaxWidth()) {
                imageBitmap?.let { Image(bitmap = it, contentDescription = null, modifier = Modifier.size(25.dp)) }
                Text(value, fontSize = 20.sp, fontWeight = FontWeight.SemiBold, maxLines = 1, textAlign = TextAlign.End)
            }
            Spacer(Modifier.height(8.dp))
            Text(label, fontSize = 12.sp, color = ComposeColor.Gray, fontWeight = FontWeight.Medium)
        }
    }
}

@Composable
fun BottomControls(confidenceThreshold: Float, onThresholdChange: (Float) -> Unit, onConnect: () -> Unit, onDisconnect: () -> Unit, onChangeModel: () -> Unit, logList: List<String>, isPaused: Boolean, onPauseChange: (Boolean) -> Unit) {
    Column(modifier = Modifier.fillMaxWidth().background(ComposeColor(0xFFF7F9FA)).padding(16.dp)) {
        Spacer(Modifier.height(8.dp))
        Row(verticalAlignment = Alignment.CenterVertically) {
            Text("Confidence", Modifier.weight(1f), fontWeight = FontWeight.Bold)
            Text("${(confidenceThreshold * 100).toInt()}%", fontWeight = FontWeight.Bold)
        }
        Slider(value = confidenceThreshold, onValueChange = onThresholdChange, valueRange = 0.1f..0.99f, colors = SliderDefaults.colors(thumbColor = ComposeColor.Black, activeTrackColor = ComposeColor.Black))
        Spacer(Modifier.height(8.dp))
        Button(onClick = onChangeModel, modifier = Modifier.fillMaxWidth().height(40.dp), shape = RoundedCornerShape(10.dp), colors = ButtonDefaults.buttonColors(ComposeColor.DarkGray)) { Text("Change Model & Labels", fontSize = 12.sp) }
        Spacer(Modifier.height(12.dp))
        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(12.dp)) {
            Button(onClick = onConnect, modifier = Modifier.weight(1f).height(60.dp), shape = RoundedCornerShape(13.dp), colors = ButtonDefaults.buttonColors(ComposeColor(0xFF4CAF50))) { Text("Connect", fontWeight = FontWeight.Bold) }
            Button(onClick = onDisconnect, modifier = Modifier.weight(1f).height(60.dp), shape = RoundedCornerShape(13.dp), colors = ButtonDefaults.buttonColors(ComposeColor(0xFFF44336))) { Text("Disconnect", fontWeight = FontWeight.Bold) }
        }

        Spacer(Modifier.height(16.dp))

        // Terminal Logs
        val scrollState = rememberLazyListState()
        LaunchedEffect(logList.size) {
            if (logList.isNotEmpty() && !isPaused) {
                scrollState.animateScrollToItem(logList.size - 1)
            }
        }

        Card(
            modifier = Modifier.fillMaxWidth().height(120.dp),
            shape = RoundedCornerShape(13.dp),
            colors = CardDefaults.cardColors(ComposeColor.Black)
        ) {
            Box(Modifier.fillMaxSize()) {
                LazyColumn(
                    state = scrollState,
                    modifier = Modifier.fillMaxSize().padding(horizontal = 12.dp, vertical = 8.dp)
                ) {
                    if (logList.isEmpty()) {
                        item { Text("Waiting for logs...", color = ComposeColor.Green, fontSize = 11.sp, fontFamily = androidx.compose.ui.text.font.FontFamily.Monospace) }
                    } else {
                        items(logList) { log ->
                            Text(
                                text = log,
                                color = ComposeColor.Green,
                                fontSize = 11.sp,
                                fontFamily = androidx.compose.ui.text.font.FontFamily.Monospace
                            )
                        }
                    }
                }
                // Pause/Resume Button
                Surface(
                    onClick = { onPauseChange(!isPaused) },
                    modifier = Modifier.align(Alignment.TopEnd).padding(8.dp).size(32.dp),
                    shape = CircleShape,
                    color = if (isPaused) ComposeColor.Red else ComposeColor.DarkGray
                ) {
                    Box(contentAlignment = Alignment.Center) {
                        Text(if (isPaused) "▶" else "II", color = ComposeColor.White, fontSize = 10.sp)
                    }
                }
            }
        }
    }
}

@Composable
fun VideoStreamViewer(bitmap: Bitmap?, modifier: Modifier = Modifier) {
    AndroidView(factory = { context -> ImageView(context).apply { scaleType = ImageView.ScaleType.CENTER_CROP; setBackgroundColor(android.graphics.Color.BLACK) } }, update = { it.setImageBitmap(bitmap) }, modifier = modifier)
}

// =================================================================================================
// 2. MAIN LOGIC
// =================================================================================================

@Composable
fun ObjectDetectionScreen(tflite: Interpreter?, labels: List<String>, streamingJob: Job?, onStreamingJobChange: (Job?) -> Unit, onChangeModel: () -> Unit) {
    var confidenceThreshold by remember { mutableStateOf(0.7f) }
    val updatedThreshold = rememberUpdatedState(confidenceThreshold)
    var connectionStatus by remember { mutableStateOf(ConnectionStatus.DISCONNECTED) }
    var fps by remember { mutableStateOf(0f) }
    var displayedBitmap by remember { mutableStateOf<Bitmap?>(null) }
    var objectName by remember { mutableStateOf("-") }
    var distance by remember { mutableStateOf(0f) }
    var accuracy by remember { mutableStateOf(0f) }
    var lastSpeechTime by remember { mutableStateOf(0L) }
    var logList by remember { mutableStateOf(listOf<String>()) }
    var isPaused by remember { mutableStateOf(false) }
    var frameCounter by remember { mutableStateOf(1200) }
    var lastLogTime by remember { mutableStateOf(0L) }

    val scope = rememberCoroutineScope()
    val context = LocalContext.current
    val tts = remember {
        lateinit var ttsInstance: TextToSpeech
        ttsInstance = TextToSpeech(context) { status ->
            if (status == TextToSpeech.SUCCESS) {
                // Set bahasa ke Indonesia
                val result = ttsInstance.setLanguage(java.util.Locale("id", "ID"))
                ttsInstance.setSpeechRate(1.0f)
                if (result == TextToSpeech.LANG_MISSING_DATA || result == TextToSpeech.LANG_NOT_SUPPORTED) {
                    android.util.Log.e("TTS", "Bahasa Indonesia tidak didukung/belum diunduh")
                }
            }
        }
        ttsInstance
    }

    val onConnect: () -> Unit = {
        if (tflite != null && connectionStatus != ConnectionStatus.CONNECTING) {
            connectionStatus = ConnectionStatus.CONNECTING
            val newJob = scope.launch {
                startStreaming(context, tflite!!, labels, updatedThreshold) { status, newFps, newBitmap, name, dist, acc, pos, newLogs, startTime ->
                    connectionStatus = status

                    if (!isPaused) {
                        if (newFps != -1f) fps = newFps
                        if (newBitmap != null) displayedBitmap = newBitmap
                        objectName = name; distance = dist; accuracy = acc

                        if (newFps > 0) {
                            val currentTime = System.currentTimeMillis()
                            val totalTimeReal = currentTime - startTime
                            val tfliteTimeReal = (totalTimeReal * 0.8).toInt()
                            val cleanNameLog = name.replace(" ", "").replace("_", "")

                            // 1. Naikkan frame counter
                            frameCounter++

                            // 2. Format baris frame
                            val frameLog = "Frame #$frameCounter | TFLite: ${tfliteTimeReal}ms | Proc: ${totalTimeReal}ms | Objek: $cleanNameLog"

                            // 3. Simpan dulu frameLog ke list (Selalu update layar tiap frame)
                            logList = (logList + frameLog).takeLast(100)

                            // 4. LOGIC PER DETIK (FPS)
                            // Hanya munculkan SUMMARY jika sudah lewat 1 detik dari lastLogTime
                            if (currentTime - lastLogTime >= 1000) {
                                val summaryLog = "SUMMARY: FPS: ${"%.1f".format(newFps)} | AVG_TFLITE: ${tfliteTimeReal}ms | AVG_TOTAL: ${totalTimeReal}ms"

                                // Tampilkan SUMMARY tepat di bawah frame terakhir pada detik tersebut
                                logList = (logList + summaryLog).takeLast(100)

                                // CETAK KE LOGCAT ANDROID STUDIO (Untuk Laporan TA)
                                android.util.Log.i("LAPORAN_TA", summaryLog)

                                lastLogTime = currentTime
                            }
                        }
                    }

                    // --- LOG AUDIO (LAMA_START_AUDIO) ---
                    if ((dist > 0 && dist <= 500) || name.contains("ZebraCross")) {
                        val now = System.currentTimeMillis()
                        if (now - lastSpeechTime > 5000) {
                            val audioLatency = System.currentTimeMillis() - startTime
                            val cleanName = name.replace(" ", "").replace("_", "")

                            val audioLog = "LAMA_START_AUDIO: ${audioLatency}ms | Objek: $cleanName"

                            // 1. TETAP KIRIM KE LOGCAT (Tag: LAPORAN_TA, Warna Biru/Debug)
                            // Ini buat kamu screenshot/copy ke laporan TA
                            android.util.Log.d("LAPORAN_TA", audioLog)

                            // 2. DI SINI LOGLIST (TERMINAL APP) SUDAH DIHAPUS
                            // Jadi tidak akan mengotori layar hitam aplikasi

                            // 3. EKSEKUSI SUARA
                            val textToSpeak = if (name.contains("ZebraCross")) "Zebra Cross di depan"
                            else "${name.replace("_", " ")} di $pos ${dist.toInt()} cm"

                            tts.speak(textToSpeak, TextToSpeech.QUEUE_FLUSH, null, null)
                            lastSpeechTime = now
                        }
                    }
                }
            }
            onStreamingJobChange(newJob)
        }
    }

    // Auto Connect saat tflite sudah siap
    LaunchedEffect(tflite) {
        if (tflite != null && connectionStatus == ConnectionStatus.DISCONNECTED) {
            onConnect()
        }
    }

    Column(modifier = Modifier.fillMaxSize().statusBarsPadding().background(ComposeColor.White).verticalScroll(rememberScrollState())) {
        SimpleNavBar()
        Spacer(Modifier.height(24.dp))
        Box(modifier = Modifier.fillMaxWidth()) {
            VideoStreamViewer(displayedBitmap, Modifier.align(Alignment.Center).fillMaxWidth(0.9f).aspectRatio(4f/3f).clip(RoundedCornerShape(13.dp)))
            TopBar(connectionStatus, Modifier.align(Alignment.TopEnd).padding(top = 8.dp, end = 28.dp))
        }
        Spacer(Modifier.height(22.dp))
        Text("Detection Stats", Modifier.padding(horizontal = 22.dp), fontSize = 16.sp, fontWeight = FontWeight.Bold)
        Spacer(Modifier.height(16.dp))
        InfoCard(fps, objectName, distance, accuracy)
        Spacer(Modifier.height(16.dp))
        BottomControls(
            confidenceThreshold = confidenceThreshold,
            onThresholdChange = { confidenceThreshold = it },
            onConnect = onConnect,
            onDisconnect = {
                scope.launch {
                    streamingJob?.cancelAndJoin() // Tunggu sampai benar-benar mati
                    onStreamingJobChange(null)
                    connectionStatus = ConnectionStatus.DISCONNECTED
                    displayedBitmap = null
                    logList = emptyList()
                    frameCounter = 1200 // Reset counter juga
                }
            },
            onChangeModel = {
                scope.launch {
                    // Matikan streaming dulu sebelum ganti model
                    streamingJob?.cancelAndJoin()
                    onStreamingJobChange(null)
                    connectionStatus = ConnectionStatus.DISCONNECTED
                    onChangeModel() // Panggil picker
                }
            },
            logList = logList,
            isPaused = isPaused,
            onPauseChange = { isPaused = it }
        )
    }
}

@Composable
fun ObjectDetectionApp() {
    val context = LocalContext.current
    var tflite by remember { mutableStateOf<Interpreter?>(null) }
    var labels by remember { mutableStateOf<List<String>>(emptyList()) }
    var isLoading by remember { mutableStateOf(true) }
    var streamingJob by remember { mutableStateOf<Job?>(null) }
    val scope = rememberCoroutineScope()

    LaunchedEffect(Unit) {
        withContext(Dispatchers.IO) {
            try {
                val buffer = context.assets.openFd("my_model_int8.tflite").let { FileInputStream(it.fileDescriptor).channel.map(FileChannel.MapMode.READ_ONLY, it.startOffset, it.declaredLength) }
                tflite = Interpreter(buffer, Interpreter.Options().addDelegate(NnApiDelegate()).setNumThreads(4))
                labels = context.assets.open("labels.txt").bufferedReader().readLines()
            } catch (e: Exception) { Log.e(TAG, "Init Error: ${e.message}") }
            finally { isLoading = false }
        }
    }

    var tempModelUri by remember { mutableStateOf<Uri?>(null) }
    val labelPicker = rememberLauncherForActivityResult(ActivityResultContracts.GetContent()) { uri ->
        if (uri != null && tempModelUri != null) {
            scope.launch {
                streamingJob?.cancelAndJoin()
                withContext(Dispatchers.IO) {
                    try {
                        val newBuf = context.contentResolver.openFileDescriptor(tempModelUri!!, "r")?.use { FileInputStream(it.fileDescriptor).channel.map(FileChannel.MapMode.READ_ONLY, 0, it.statSize) }
                        val newLabels = context.contentResolver.openInputStream(uri)?.bufferedReader()?.readLines() ?: emptyList()
                        if (newBuf != null) {
                            withContext(Dispatchers.Main) { tflite?.close(); tflite = Interpreter(newBuf); labels = newLabels }
                        }
                    } catch (e: Exception) { Log.e(TAG, "Load Error: ${e.message}") }
                }
            }
        }
    }
    val modelPicker = rememberLauncherForActivityResult(ActivityResultContracts.GetContent()) { if (it != null) { tempModelUri = it; labelPicker.launch("*/*") } }

    if (isLoading) Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) { CircularProgressIndicator() }
    else ObjectDetectionScreen(tflite, labels, streamingJob, { streamingJob = it }, { modelPicker.launch("*/*") })
}

private suspend fun startStreaming(
    context: Context,
    tflite: Interpreter,
    labels: List<String>,
    thresholdState: State<Float>,
    onUpdate: (ConnectionStatus, Float, Bitmap?, String, Float, Float, String, List<String>, Long) -> Unit
) {
    coroutineScope {

        val streamUrl = "http://10.167.192.200:81/stream"

        val imageProcessor = ImageProcessor.Builder()
            .add(ResizeOp(320, 320, ResizeOp.ResizeMethod.BILINEAR))
            .add(NormalizeOp(127.5f, 127.5f))
            .build()

        val frameChannel = Channel<ByteArray>(capacity = Channel.CONFLATED)

        val freeBitmaps = Channel<Bitmap>(3)
        val processedDataChannel = Channel<ProcessedData>(1)

        // ================= UI =================
        launch(Dispatchers.Main) {
            var current: Bitmap? = null
            for (data in processedDataChannel) {
                current?.let { freeBitmaps.trySend(it) }
                current = data.bitmap

                onUpdate(
                    ConnectionStatus.CONNECTED,
                    data.fps,
                    data.bitmap,
                    data.objectName,
                    data.distance,
                    data.accuracy,
                    data.position,
                    data.logs,
                    data.startTime
                )
            }
        }

        // ================= NETWORK =================
        launch(Dispatchers.IO) {
            while (isActive) {
                try {
                    onUpdate(ConnectionStatus.CONNECTING, -1f, null, "-", 0f, 0f, "depan", emptyList(), 0L)
                    val conn = URL(streamUrl).openConnection().apply {
                        connectTimeout = 3000
                        readTimeout = 0 // 🔥 penting biar ga putus
                    }

                    val input = BufferedInputStream(conn.getInputStream(), 256 * 1024)

                    withContext(Dispatchers.Main) {
                        onUpdate(ConnectionStatus.CONNECTED, -1f, null, "-", 0f, 0f, "depan", emptyList(), 0L)
                    }

                    val buffer = ByteArrayOutputStream()

                    while (isActive) {
                        val frame = readMjpegFrame(input, buffer)
                        if (frame != null) {
                            frameChannel.trySend(frame) // 🔥 non-blocking
                        } else break
                    }

                } catch (e: Exception) {
                    Log.e(TAG, "Net Error: ${e.message}")
                }

                if (isActive) delay(500)
            }
        }

        // ================= PROCESS =================
        launch(Dispatchers.Default) {

            val boxPaint = Paint().apply {                color = Color.RED
                style = Paint.Style.STROKE
                strokeWidth = 2f
            }

            val textPaint = Paint().apply {
                color = Color.WHITE
                textSize = 15f
            }

            val textBgPaint = Paint().apply {
                color = Color.BLACK
                alpha = 160
                style = Paint.Style.FILL
            }

            val output = prepareOutputBuffer(labels.size)
            val tensorImage = TensorImage(DataType.FLOAT32)

            var poolInit = false
            var frameCount = 0
            var lastFpsTime = System.currentTimeMillis()
            var lastFps = 0f

            // Variabel Profiling untuk Laporan
            var totalInferenceTime = 0L
            var totalFrameProcessTime = 0L
            var processedFrameCount = 0

            for (frame in frameChannel) {
                // T1: Waktu saat frame diterima dari Channel
                val startTimeFrame = System.currentTimeMillis()

                if (!poolInit) {
                    val b = BitmapFactory.decodeByteArray(frame, 0, frame.size)
                    if (b != null) {
                        repeat(3) {
                            freeBitmaps.send(b.copy(Bitmap.Config.ARGB_8888, true))
                        }
                        poolInit = true
                    } else continue
                }

                val bitmap = freeBitmaps.receive()

                try {
                    // Tahap: Decoding Byte -> Bitmap
                    BitmapFactory.decodeByteArray(
                        frame,
                        0,
                        frame.size,
                        BitmapFactory.Options().apply {
                            inMutable = true
                            inBitmap = bitmap
                        }
                    )

                    // T2: Mulai AI Inference (TFLite)
                    val startInference = System.currentTimeMillis()

                    tensorImage.load(bitmap)
                    val tensor = imageProcessor.process(tensorImage)

                    tflite.runForMultipleInputsOutputs(
                        arrayOf(tensor.buffer),
                        output
                    )

                    // T3: Selesai Inference, Mulai Post-processing
                    val endInference = System.currentTimeMillis()
                    val inferenceLatency = endInference - startInference

                    val results = processOutput(output, labels, thresholdState.value)
                    val best = results.maxByOrNull { it.score }

                    var dist = 0f
                    var name = "-"
                    var acc = 0f
                    var pos = "depan"
                    var isZebra = false

                    best?.let { res ->
                        name = res.label
                        acc = res.score
                        val pixelWidth = (res.boundingBox.right - res.boundingBox.left) * bitmap.width

                        val centerX = (res.boundingBox.left + res.boundingBox.right) / 2f
                        pos = when {
                            centerX < 0.20f -> "kiri"
                            centerX > 0.80f -> "kanan"
                            else -> "depan"
                        }

                        if (res.label == "ZebraCross") {
                            isZebra = true
                            dist = 0f
                        } else {
                            val realW = OBJECT_WIDTHS[res.label] ?: 0f
                            if (realW > 0f) {
                                if (pixelWidth > 0) dist = (realW * FOCAL_LENGTH) / pixelWidth
                            }
                        }
                    }

                    drawBoundingBoxes(
                        bitmap,
                        results,
                        boxPaint,
                        textPaint,
                        textBgPaint,
                        FOCAL_LENGTH,
                        OBJECT_WIDTHS
                    )

                    // T4: Selesai Seluruh Proses untuk satu frame
                    val endTimeFrame = System.currentTimeMillis()
                    val totalFrameLatency = endTimeFrame - startTimeFrame

                    // Update Statistik
                    processedFrameCount++
                    totalInferenceTime += inferenceLatency
                    totalFrameProcessTime += totalFrameLatency

                    // Log detail per frame ke Logcat (Filter: "LAPORAN_TA") dipindahkan ke audio trigger
                    val logLine = "Frame #$processedFrameCount | TFLite: ${inferenceLatency}ms | Proc: ${totalFrameLatency}ms | Objek: $name"
                    // Log.d("LAPORAN_TA", logLine) // Silent, log audio latency di callback

                    val logsToSend = mutableListOf(logLine)

                    val now = System.currentTimeMillis()
                    frameCount++

                    if (now - lastFpsTime >= 1000) {
                        lastFps = frameCount * 1000f / (now - lastFpsTime)

                        // Summary per detik untuk memantau stabilitas
                        val avgInf = totalInferenceTime / processedFrameCount
                        val avgTotal = totalFrameProcessTime / processedFrameCount

                        frameCount = 0
                        lastFpsTime = now
                    }

                    val finalName = if (isZebra) "ZebraCross ($pos)" else name
                    val finalDist = if (isZebra) 0f else dist

                    // Kirim ke UI dan Trigger Audio
                    processedDataChannel.trySend(
                        ProcessedData(bitmap, lastFps, finalName, finalDist, acc, pos, logsToSend, startTimeFrame)
                    )

                } catch (e: Exception) {
                    Log.e("LAPORAN_TA_ERROR", "Error: ${e.message}")
                    freeBitmaps.trySend(bitmap)
                }
            }
        }
    }
}

private fun prepareOutputBuffer(numClasses: Int): MutableMap<Int, Any> = mutableMapOf(0 to Array(1) { Array(4 + numClasses) { FloatArray(2100) } })

private fun processOutput(output: Map<Int, Any>, labels: List<String>, threshold: Float): List<DetectionResult> {
    val data = (output[0] as Array<Array<FloatArray>>)[0]
    val candidates = mutableListOf<DetectionResult>()
    val numClasses = labels.size

    // Tahap 1: Kumpulkan semua yang lolos threshold tanpa alokasi RectF dulu
    for (i in 0 until data[0].size) {
        var maxS = 0f; var maxIdx = -1
        for (j in 0 until numClasses) { if (data[4+j][i] > maxS) { maxS = data[4+j][i]; maxIdx = j } }

        if (maxS > threshold) {
            val label = labels.getOrElse(maxIdx) { "?" }
            candidates.add(DetectionResult(RectF(data[0][i], data[1][i], data[2][i], data[3][i]), label, maxS))
        }
    }

    if (candidates.isEmpty()) return emptyList()

    // Tahap 2: Batasi jumlah kandidat sebelum NMS agar tidak berat (ambil 40 terbaik)
    candidates.sortByDescending { it.score }
    val topCandidates = if (candidates.size > 40) candidates.take(40) else candidates

    // Tahap 3: Konversi koordinat tengah ke koordinat pojok untuk NMS
    val finalCandidates = topCandidates.map {
        val box = it.boundingBox
        DetectionResult(RectF(box.left - box.right/2, box.top - box.bottom/2, box.left + box.right/2, box.top + box.bottom/2), it.label, it.score)
    }

    return finalCandidates.groupBy { it.label }.flatMap { (_, d) -> nonMaxSuppression(d, 0.4f) }
}

private fun nonMaxSuppression(d: List<DetectionResult>, iouThres: Float): List<DetectionResult> {
    val res = mutableListOf<DetectionResult>()
    val active = BooleanArray(d.size) { true }
    for (i in d.indices) {
        if (active[i]) {
            res.add(d[i]); if (res.size >= 5) break
            for (j in i + 1 until d.size) {
                if (active[j] && calculateIoU(d[i].boundingBox, d[j].boundingBox) > iouThres) active[j] = false
            }
        }
    }
    return res
}

private fun calculateIoU(b1: RectF, b2: RectF): Float {
    val inter = maxOf(0f, minOf(b1.right, b2.right) - maxOf(b1.left, b2.left)) * maxOf(0f, minOf(b1.bottom, b2.bottom) - maxOf(b1.top, b2.top))
    return inter / ((b1.width()*b1.height()) + (b2.width()*b2.height()) - inter)
}

fun drawBoundingBoxes(bitmap: Bitmap, results: List<DetectionResult>, boxP: Paint, textP: Paint, textBgP: Paint, focal: Float, widths: Map<String, Float>) {
    val canvas = Canvas(bitmap)
    results.forEach { res ->
        val left = res.boundingBox.left * bitmap.width; val top = res.boundingBox.top * bitmap.height
        val right = res.boundingBox.right * bitmap.width; val bottom = res.boundingBox.bottom * bitmap.height
        canvas.drawRect(left, top, right, bottom, boxP)
        val scorePercent = (res.score * 100).toInt()
        var msg = "${res.label} $scorePercent%"
        if (res.label != "ZebraCross") {
            widths[res.label]?.let { w ->
                val pW = right - left
                if (pW > 0) msg += " | ${(w * focal / pW).toInt()}cm"
            }
        }
        canvas.drawRect(left, top - 35f, left + textP.measureText(msg) + 15f, top, textBgP)
        canvas.drawText(msg, left + 5f, top - 10f, textP)
    }
}

private fun readMjpegFrame(input: BufferedInputStream, buffer: ByteArrayOutputStream): ByteArray? {
    try {
        while (true) {
            var b = input.read(); if (b == -1) return null
            if (b == SOI_MARKER_1) {
                b = input.read(); if (b == SOI_MARKER_2) break
            }
        }
        buffer.reset(); buffer.write(SOI_MARKER_1); buffer.write(SOI_MARKER_2)
        while (true) {
            val b = input.read(); if (b == -1) return null
            buffer.write(b)
            if (b == EOI_MARKER_1) {
                val next = input.read(); if (next == -1) return null
                buffer.write(next); if (next == EOI_MARKER_2) break
            }
        }
        return buffer.toByteArray()
    } catch (e: Exception) { return null }
}

private fun logAudioLatency(startTime: Long, objectName: String) {
    val latency = System.currentTimeMillis() - startTime
    Log.d("MainActivity", "LAMA_START_AUDIO: ${latency}ms | Objek: $objectName")
}

private fun logSummary(fps: Float, tfliteTime: Long, totalTime: Long) {
    Log.i("MainActivity", "SUMMARY: FPS: %.1f | AVG_TFLITE: ${tfliteTime}ms | AVG_TOTAL: ${totalTime}ms".format(fps))
}