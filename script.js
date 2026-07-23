/* ============================================================================
   SCRIPT.JS - Logika Frontend Aplikasi Deteksi Penyakit Selada Hidroponik
   ============================================================================
   Berisi 2 bagian utama:
     1. FITUR UPLOAD GAMBAR  -> kirim file ke /predict/image
     2. FITUR WEBCAM REALTIME -> kirim frame berkala ke /predict/frame
   ============================================================================ */

// ============================================================================
// BAGIAN 0: NAVIGASI TAB
// ============================================================================
const tabButtons = document.querySelectorAll(".tab-btn");
const tabContents = document.querySelectorAll(".tab-content");

tabButtons.forEach((button) => {
  button.addEventListener("click", () => {
    const targetTabId = button.getAttribute("data-tab");

    tabButtons.forEach((btn) => btn.classList.remove("active"));
    tabContents.forEach((content) => content.classList.remove("active"));

    button.classList.add("active");
    document.getElementById(targetTabId).classList.add("active");

    // Jika pindah keluar dari tab webcam, hentikan webcam otomatis
    if (targetTabId !== "tab-webcam" && isWebcamRunning) {
      stopWebcam();
    }
  });
});

// ============================================================================
// BAGIAN 1: FITUR UPLOAD GAMBAR
// ============================================================================

const uploadForm = document.getElementById("upload-form");
const imageInput = document.getElementById("image-input");
const previewImage = document.getElementById("preview-image");
const resultImage = document.getElementById("result-image");
const uploadLoading = document.getElementById("upload-loading");
const detectionListItems = document.getElementById("detection-list-items");

// Menampilkan preview gambar yang dipilih user sebelum dikirim ke server
imageInput.addEventListener("change", () => {
  const file = imageInput.files[0];
  if (file) {
    previewImage.src = URL.createObjectURL(file);
  }
});

uploadForm.addEventListener("submit", async (event) => {
  event.preventDefault();

  const file = imageInput.files[0];
  if (!file) {
    alert("Silakan pilih gambar terlebih dahulu.");
    return;
  }

  // Validasi ekstensi file di sisi frontend (validasi utama tetap di backend)
  const allowedExtensions = ["jpg", "jpeg", "png"];
  const fileExtension = file.name.split(".").pop().toLowerCase();
  if (!allowedExtensions.includes(fileExtension)) {
    alert("Format file tidak didukung. Gunakan JPG, JPEG, atau PNG.");
    return;
  }

  const formData = new FormData();
  formData.append("image", file);

  uploadLoading.classList.remove("hidden");
  resultImage.src = "";
  clearDetectionList(detectionListItems);

  try {
    const response = await fetch("/predict/image", {
      method: "POST",
      body: formData,
    });

    const data = await response.json();

    if (!data.success) {
      alert("Gagal melakukan deteksi: " + data.error);
      return;
    }

    // Menampilkan gambar hasil deteksi (sudah ada bounding box dari backend)
    resultImage.src = data.result_image;

    // Menampilkan daftar deteksi dalam bentuk teks
    renderDetectionList(detectionListItems, data.detections);
  } catch (error) {
    console.error(error);
    alert("Terjadi kesalahan saat menghubungi server.");
  } finally {
    uploadLoading.classList.add("hidden");
  }
});

/**
 * Menampilkan daftar hasil deteksi (nama kelas + confidence) dalam bentuk <li>.
 * Fungsi ini dipakai ulang oleh fitur upload maupun webcam.
 */
function renderDetectionList(listElement, detections) {
  clearDetectionList(listElement);

  if (!detections || detections.length === 0) {
    const emptyItem = document.createElement("li");
    emptyItem.className = "empty-state";
    emptyItem.textContent = "Tidak ada penyakit terdeteksi pada gambar ini.";
    listElement.appendChild(emptyItem);
    return;
  }

  detections.forEach((detection) => {
    const listItem = document.createElement("li");

    const labelSpan = document.createElement("span");
    labelSpan.textContent = detection.class_name;

    const confidenceSpan = document.createElement("span");
    confidenceSpan.className = "confidence-badge";
    confidenceSpan.textContent = detection.confidence.toFixed(1) + "%";

    listItem.appendChild(labelSpan);
    listItem.appendChild(confidenceSpan);
    listElement.appendChild(listItem);
  });
}

function clearDetectionList(listElement) {
  listElement.innerHTML = "";
}

// ============================================================================
// BAGIAN 2: FITUR UPLOAD VIDEO
// ============================================================================

const videoForm = document.getElementById("video-form");
const videoInput = document.getElementById("video-input");
const videoLoading = document.getElementById("video-loading");
const videoPreviewArea = document.getElementById("video-preview-area");
const videoPreview = document.getElementById("video-preview");
const videoResultArea = document.getElementById("video-result-area");
const videoResult = document.getElementById("video-result");
const videoDownloadLink = document.getElementById("video-download-link");
const videoSummaryArea = document.getElementById("video-summary-area");
const videoSummaryList = document.getElementById("video-summary-list");
const videoStatsText = document.getElementById("video-stats-text");
const videoSamplesArea = document.getElementById("video-samples-area");
const videoSamplesGrid = document.getElementById("video-samples-grid");

// Tampilkan preview video asli saat file dipilih
videoInput.addEventListener("change", () => {
  const file = videoInput.files[0];
  if (file) {
    videoPreview.src = URL.createObjectURL(file);
    videoPreviewArea.classList.remove("hidden");

    // Sembunyikan hasil deteksi lama kalau ada
    videoResultArea.classList.add("hidden");
    videoSamplesArea.classList.add("hidden");
    videoSummaryArea.classList.add("hidden");
  }
});

videoForm.addEventListener("submit", async (event) => {
  event.preventDefault();

  const file = videoInput.files[0];
  if (!file) {
    alert("Silakan pilih file video terlebih dahulu.");
    return;
  }

  // Validasi ekstensi di frontend
  const allowedVideoExts = ["mp4", "avi", "mov", "mkv"];
  const fileExt = file.name.split(".").pop().toLowerCase();
  if (!allowedVideoExts.includes(fileExt)) {
    alert("Format video tidak didukung.\nGunakan MP4, AVI, MOV, atau MKV.");
    return;
  }

  const formData = new FormData();
  formData.append("video", file);

  // Tampilkan loading, sembunyikan hasil lama
  videoLoading.classList.remove("hidden");
  videoResultArea.classList.add("hidden");
  videoSummaryArea.classList.add("hidden");
  document.getElementById("video-btn").disabled = true;

  try {
    // Kirim video ke backend untuk diproses
    // Catatan: ini bisa memakan waktu lama tergantung panjang video
    const response = await fetch("/predict/video", {
      method: "POST",
      body: formData,
    });

    const data = await response.json();

    if (!data.success) {
      alert("Gagal memproses video: " + data.error);
      return;
    }

    // Tampilkan video hasil deteksi dari server
    const videoUrl = `/download/video/${data.output_filename}`;
    videoResult.src = videoUrl;
    videoDownloadLink.href = videoUrl;
    videoDownloadLink.download = data.output_filename;
    videoResultArea.classList.remove("hidden");

    // Tampilkan 4 sample frame contoh hasil deteksi
    renderSampleFrames(data.sample_frames);

    // Tampilkan ringkasan statistik deteksi
    videoStatsText.textContent = `Total frame diproses: ${data.total_frames} frame  |  FPS: ${data.fps}`;

    renderVideoSummary(data.summary, data.total_frames);
    videoSummaryArea.classList.remove("hidden");

    // Scroll ke hasil
    videoResultArea.scrollIntoView({ behavior: "smooth" });
  } catch (error) {
    console.error(error);
    alert(
      "Terjadi kesalahan saat menghubungi server.\nPastikan server Flask masih berjalan.",
    );
  } finally {
    videoLoading.classList.add("hidden");
    document.getElementById("video-btn").disabled = false;
  }
});

/**
 * Menampilkan grid 4 foto contoh frame hasil deteksi dari video.
 * Setiap foto diambil dari titik berbeda di sepanjang durasi video.
 */
function renderSampleFrames(sampleFrames) {
  videoSamplesGrid.innerHTML = "";

  if (!sampleFrames || sampleFrames.length === 0) {
    videoSamplesArea.classList.add("hidden");
    return;
  }

  const labels = ["📍 Awal", "📍 Tengah-Awal", "📍 Tengah-Akhir", "📍 Akhir"];

  sampleFrames.forEach((base64Img, index) => {
    const box = document.createElement("div");
    box.className = "sample-frame-box";

    const img = document.createElement("img");
    img.src = base64Img;
    img.alt = `Sample frame ${index + 1}`;

    const label = document.createElement("div");
    label.className = "sample-frame-label";
    label.textContent = labels[index] || `Frame ${index + 1}`;

    box.appendChild(img);
    box.appendChild(label);
    videoSamplesGrid.appendChild(box);
  });

  videoSamplesArea.classList.remove("hidden");
}

/**
 * Menampilkan ringkasan deteksi video: nama kelas + jumlah frame terdeteksi
 * + persentase frame video yang mengandung deteksi tersebut.
 */
function renderVideoSummary(summary, totalFrames) {
  videoSummaryList.innerHTML = "";

  if (!summary || Object.keys(summary).length === 0) {
    const li = document.createElement("li");
    li.className = "empty-state";
    li.textContent = "Tidak ada penyakit terdeteksi di seluruh video.";
    videoSummaryList.appendChild(li);
    return;
  }

  for (const [className, frameCount] of Object.entries(summary)) {
    const persen = ((frameCount / totalFrames) * 100).toFixed(1);
    const li = document.createElement("li");

    const labelSpan = document.createElement("span");
    labelSpan.textContent = `${className}  —  ${frameCount} frame`;

    const badgeSpan = document.createElement("span");
    badgeSpan.className = "confidence-badge";
    badgeSpan.textContent = `${persen}% durasi video`;

    li.appendChild(labelSpan);
    li.appendChild(badgeSpan);
    videoSummaryList.appendChild(li);
  }
}

// ============================================================================
// BAGIAN 3: FITUR WEBCAM REAL-TIME
// ============================================================================

const startWebcamBtn = document.getElementById("start-webcam-btn");
const stopWebcamBtn = document.getElementById("stop-webcam-btn");
const webcamVideo = document.getElementById("webcam-video");
const webcamCanvas = document.getElementById("webcam-canvas");
const webcamStatus = document.getElementById("webcam-status");
const webcamDetectionList = document.getElementById("webcam-detection-list");
const canvasContext = webcamCanvas.getContext("2d");

let webcamStream = null;
let isWebcamRunning = false;
let detectionIntervalId = null;
let isSendingFrame = false; // mencegah pengiriman frame menumpuk jika server lambat

// Interval pengiriman frame ke server (ms). Semakin kecil = makin real-time,
// tapi makin berat untuk server. Silakan disesuaikan untuk skripsi (mis. 300-500ms).
const FRAME_SEND_INTERVAL_MS = 400;

startWebcamBtn.addEventListener("click", startWebcam);
stopWebcamBtn.addEventListener("click", stopWebcam);

async function startWebcam() {
  try {
    // Cek apakah browser mendukung getUserMedia sama sekali
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      alert(
        "Browser Anda tidak mendukung akses kamera.\nGunakan Chrome atau Edge versi terbaru.",
      );
      return;
    }

    // Meminta izin akses webcam ke browser.
    // Tidak memaksa resolusi tertentu (video: true) agar kompatibel
    // dengan semua jenis kamera (built-in laptop, webcam USB, dll).
    webcamStream = await navigator.mediaDevices.getUserMedia({
      video: true,
      audio: false,
    });

    webcamVideo.srcObject = webcamStream;
    await webcamVideo.play();

    // Tunggu metadata video (lebar & tinggi) sudah tersedia
    await new Promise((resolve) => {
      if (webcamVideo.readyState >= 2) return resolve();
      webcamVideo.addEventListener("loadeddata", resolve, { once: true });
    });

    // Menyesuaikan ukuran canvas dengan ukuran video asli dari kamera
    webcamCanvas.width = webcamVideo.videoWidth || 640;
    webcamCanvas.height = webcamVideo.videoHeight || 480;

    isWebcamRunning = true;
    startWebcamBtn.disabled = true;
    stopWebcamBtn.disabled = false;
    webcamStatus.textContent =
      "🟢 Webcam aktif. Mendeteksi penyakit secara real-time...";

    // Loop untuk menggambar frame video ke canvas terus-menerus (untuk tampilan halus)
    drawVideoLoop();

    // Loop terpisah untuk mengirim frame ke backend secara berkala
    detectionIntervalId = setInterval(
      sendFrameForDetection,
      FRAME_SEND_INTERVAL_MS,
    );
  } catch (error) {
    console.error("Error webcam:", error.name, error.message);

    // Pesan error yang spesifik per jenis kesalahan
    const pesanError = {
      NotAllowedError:
        "Izin kamera ditolak.\n\n" +
        "Cara mengizinkan:\n" +
        "1. Klik ikon 🔒 di sebelah kiri address bar browser\n" +
        "2. Cari 'Camera' → ubah ke 'Allow'\n" +
        "3. Refresh halaman ini, lalu coba lagi.",
      NotFoundError:
        "Kamera tidak ditemukan di perangkat ini.\n" +
        "Pastikan webcam sudah terhubung dan tidak dinonaktifkan di Device Manager.",
      NotReadableError:
        "Kamera sedang digunakan oleh aplikasi lain.\n" +
        "Tutup dulu aplikasi seperti Zoom, Teams, OBS, atau aplikasi kamera lainnya, lalu coba lagi.",
      OverconstrainedError:
        "Kamera tidak mendukung pengaturan yang diminta.\n" +
        "Coba refresh halaman dan klik Mulai Webcam lagi.",
    };

    const pesan =
      pesanError[error.name] ||
      `Tidak dapat mengakses kamera.\nError: ${error.name} - ${error.message}`;

    alert(pesan);
  }
}

function stopWebcam() {
  isWebcamRunning = false;

  if (detectionIntervalId) {
    clearInterval(detectionIntervalId);
    detectionIntervalId = null;
  }

  if (webcamStream) {
    webcamStream.getTracks().forEach((track) => track.stop());
    webcamStream = null;
  }

  canvasContext.clearRect(0, 0, webcamCanvas.width, webcamCanvas.height);

  startWebcamBtn.disabled = false;
  stopWebcamBtn.disabled = true;
  webcamStatus.textContent = "⏹️ Webcam dihentikan.";

  clearDetectionList(webcamDetectionList);
}

/**
 * Menggambar ulang frame video ke canvas secara terus-menerus (~60fps)
 * supaya video terlihat halus. Bounding box akan ditambahkan di atas frame
 * ini setiap kali hasil deteksi baru diterima dari server.
 */
let lastDetections = [];

function drawVideoLoop() {
  if (!isWebcamRunning) return;

  canvasContext.drawImage(
    webcamVideo,
    0,
    0,
    webcamCanvas.width,
    webcamCanvas.height,
  );
  drawBoundingBoxes(lastDetections);

  requestAnimationFrame(drawVideoLoop);
}

/**
 * Mengambil 1 frame dari video, mengirimkannya ke backend Flask dalam
 * format base64 JPEG, lalu menyimpan hasil deteksi untuk digambar
 * di atas canvas oleh drawVideoLoop().
 */
async function sendFrameForDetection() {
  if (!isWebcamRunning || isSendingFrame) return;

  isSendingFrame = true;

  try {
    // Membuat canvas sementara untuk mengambil snapshot frame video
    const snapshotCanvas = document.createElement("canvas");
    snapshotCanvas.width = webcamCanvas.width;
    snapshotCanvas.height = webcamCanvas.height;
    const snapshotContext = snapshotCanvas.getContext("2d");
    snapshotContext.drawImage(
      webcamVideo,
      0,
      0,
      snapshotCanvas.width,
      snapshotCanvas.height,
    );

    // Mengubah snapshot menjadi base64 JPEG (kualitas 0.7 untuk mempercepat transfer)
    const frameBase64 = snapshotCanvas.toDataURL("image/jpeg", 0.7);

    const response = await fetch("/predict/frame", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ frame: frameBase64 }),
    });

    const data = await response.json();

    if (data.success) {
      lastDetections = data.detections;
      renderDetectionList(webcamDetectionList, lastDetections);
    }
  } catch (error) {
    console.error("Gagal mengirim frame ke server:", error);
  } finally {
    isSendingFrame = false;
  }
}

/**
 * Menggambar bounding box, label kelas, dan confidence di atas canvas video.
 * Koordinat box yang diterima dari backend (x1, y1, x2, y2) berbasis ukuran
 * gambar asli yang dikirim, sehingga harus sesuai dengan ukuran canvas.
 */
function drawBoundingBoxes(detections) {
  if (!detections || detections.length === 0) return;

  detections.forEach((detection) => {
    const [x1, y1, x2, y2] = detection.box;
    const boxWidth = x2 - x1;
    const boxHeight = y2 - y1;
    const label = `${detection.class_name} ${detection.confidence.toFixed(1)}%`;

    // Kotak deteksi
    canvasContext.strokeStyle = "#00c853";
    canvasContext.lineWidth = 3;
    canvasContext.strokeRect(x1, y1, boxWidth, boxHeight);

    // Background label
    canvasContext.font = "16px Segoe UI, Arial";
    const textWidth = canvasContext.measureText(label).width;
    canvasContext.fillStyle = "#00c853";
    canvasContext.fillRect(x1, Math.max(y1 - 24, 0), textWidth + 10, 24);

    // Teks label
    canvasContext.fillStyle = "#ffffff";
    canvasContext.fillText(label, x1 + 5, Math.max(y1 - 6, 16));
  });
}

// Menghentikan webcam otomatis jika user menutup/refresh halaman
window.addEventListener("beforeunload", () => {
  if (isWebcamRunning) {
    stopWebcam();
  }
});
