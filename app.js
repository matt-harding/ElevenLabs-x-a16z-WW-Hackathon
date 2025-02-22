let mediaRecorder;
let audioChunks = [];
let relevantQuery = "";

const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const deepInvestBtn = document.getElementById("deepInvestBtn");
const transcriptsDiv = document.getElementById("transcripts");
const resultContent = document.getElementById("resultContent");

startBtn.addEventListener("click", startRecording);
stopBtn.addEventListener("click", stopRecording);
deepInvestBtn.addEventListener("click", performDeepInvestigation);

function startRecording() {
  navigator.mediaDevices.getUserMedia({ audio: true })
    .then(stream => {
      audioChunks = [];
      mediaRecorder = new MediaRecorder(stream);
      mediaRecorder.ondataavailable = e => {
        audioChunks.push(e.data);
      };
      mediaRecorder.start();

      startBtn.disabled = true;
      stopBtn.disabled = false;
      deepInvestBtn.classList.add("hidden");
      deepInvestBtn.classList.remove("green");
      relevantQuery = "";
      transcriptsDiv.textContent = "";
      resultContent.textContent = "";
    })
    .catch(err => console.error("Mic error:", err));
}

function stopRecording() {
  mediaRecorder.stop();
  mediaRecorder.onstop = () => {
    startBtn.disabled = false;
    stopBtn.disabled = true;

    // Combine the audio chunks
    const audioBlob = new Blob(audioChunks, { type: "audio/webm" });
    const formData = new FormData();
    formData.append("audio", audioBlob, "user_audio.webm");

    fetch("/api/transcribe-audio", {
      method: "POST",
      body: formData
    })
    .then(res => res.json())
    .then(data => {
      if (data.error) {
        transcriptsDiv.textContent = "Error: " + data.error;
        return;
      }
      // Display final transcript
      transcriptsDiv.textContent = data.transcript || "";

      if (data.is_relevant) {
        relevantQuery = data.suggested_query;
        deepInvestBtn.classList.remove("hidden");
        deepInvestBtn.classList.add("green");
      }
    })
    .catch(err => {
      console.error("Transcribe error:", err);
      transcriptsDiv.textContent = "An error occurred.";
    });
  };
}

function performDeepInvestigation() {
  if (!relevantQuery.trim()) {
    alert("No relevant query found or it's not relevant!");
    return;
  }
  fetch("/api/perplexity-search", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query: relevantQuery })
  })
  .then(res => res.json())
  .then(data => {
    if (data.error) {
      resultContent.textContent = "Error: " + data.error;
    } else {
      resultContent.textContent = data.content;
    }
  })
  .catch(err => {
    console.error("DeepInvest error:", err);
    resultContent.textContent = "Error calling Perplexity.";
  });
}
