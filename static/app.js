const startSpeechBtn = document.getElementById("startSpeechBtn");
const stopSpeechBtn = document.getElementById("stopSpeechBtn");
const transcriptsDiv = document.getElementById("transcripts");
const resultsDiv = document.getElementById("results");

let recognizedSegments = [];
let lastUsedIndex = 0;
let intervalId = null;
let keepListening = false;
let analysisInProgress = false;
let recognition = null;

startSpeechBtn.addEventListener("click", startBrowserSTT);
stopSpeechBtn.addEventListener("click", stopBrowserSTT);

function startBrowserSTT() {
  console.log("[CLIENT] Starting indefinite speech recognition...");

  if (!("webkitSpeechRecognition" in window)) {
    alert("Your browser doesn't support Web Speech API. Please use Chrome/Edge.");
    return;
  }

  transcriptsDiv.textContent = "";
  resultsDiv.textContent = "";
  recognizedSegments = [];
  lastUsedIndex = 0;
  keepListening = true;
  analysisInProgress = false;

  recognition = createRecognitionInstance();
  recognition.start();

  startSpeechBtn.disabled = true;
  stopSpeechBtn.disabled = false;

  intervalId = setInterval(checkNewSegments, 2000);
}

function createRecognitionInstance() {
  const rec = new webkitSpeechRecognition();
  rec.continuous = true;
  rec.interimResults = true;
  rec.lang = "en-US";

  rec.onresult = (event) => {
    let finalSoFar = "";
    let interim = "";

    for (let i = event.resultIndex; i < event.results.length; i++) {
      const transcriptChunk = event.results[i][0].transcript;
      if (event.results[i].isFinal) {
        recognizedSegments.push({ text: transcriptChunk.trim() });
        finalSoFar += transcriptChunk + " ";
      } else {
        interim += transcriptChunk;
      }
    }

    const allFinalText = recognizedSegments.map(s => s.text).join(" ");
    transcriptsDiv.textContent = allFinalText + " [ " + interim + " ]";
  };

  rec.onerror = (event) => {
    console.error("[CLIENT] Recognition error:", event.error);
  };

  rec.onend = () => {
    console.log("[CLIENT] Speech recognition ended. keepListening =", keepListening);
    if (keepListening) {
      console.log("[CLIENT] Auto-restarting speech recognition...");
      rec.start();
    }
  };

  return rec;
}

function stopBrowserSTT() {
  console.log("[CLIENT] Stopping indefinite speech recognition...");
  keepListening = false;

  if (recognition) {
    recognition.stop();
    recognition = null;
  }

  startSpeechBtn.disabled = false;
  stopSpeechBtn.disabled = true;

  if (intervalId) {
    clearInterval(intervalId);
    intervalId = null;
  }
}

function checkNewSegments() {
  if (analysisInProgress) {
    console.log("[CLIENT] Last analysis still in progress, skipping this cycle...");
    return;
  }

  if (lastUsedIndex >= recognizedSegments.length) {
    console.log("[CLIENT] No new final text to analyze this cycle.");
    return;
  }

  const newSegments = recognizedSegments.slice(lastUsedIndex);
  lastUsedIndex = recognizedSegments.length;

  let chunkText = newSegments.map(s => s.text).join(" ").trim();
  console.log("[CLIENT] Raw new chunk:", chunkText);

  // limit to 25 words
  const words = chunkText.split(/\s+/).filter(Boolean);
  if (words.length > 25) {
    chunkText = words.slice(-25).join(" ");
    console.log("[CLIENT] Truncated to last 25 words ->", chunkText);
  }

  if (!chunkText) {
    console.log("[CLIENT] chunk is empty after trimming, skip.");
    return;
  }

  analysisInProgress = true;

  console.log("[CLIENT] Sending chunk to server:", chunkText);
  fetch("/api/process-chunk", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ chunk: chunkText })
  })
    .then(res => res.json())
    .then(data => {
      console.log("[CLIENT] /api/process-chunk response:", data);
      displayChunkResult(data);
    })
    .catch(err => {
      console.error("[CLIENT] Error calling /api/process-chunk:", err);
      displayChunkResult({
        chunk_text: chunkText,
        openai_response: "Error calling /api/process-chunk"
      });
    })
    .finally(() => {
      analysisInProgress = false;
    });
}

function displayChunkResult(data) {
  const blockDiv = document.createElement("div");
  blockDiv.style.border = "1px solid #ccc";
  blockDiv.style.marginBottom = "10px";
  blockDiv.style.padding = "10px";

  const chunkPara = document.createElement("p");
  chunkPara.innerHTML = `<strong>Chunk:</strong> ${data.chunk_text}`;
  blockDiv.appendChild(chunkPara);

  const respPara = document.createElement("p");
  respPara.innerHTML = `<strong>One-sentence Prompt:</strong> ${data.openai_response}`;
  blockDiv.appendChild(respPara);

  resultsDiv.prepend(blockDiv);
}
