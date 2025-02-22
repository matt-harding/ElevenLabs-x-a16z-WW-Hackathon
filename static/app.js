// DOM Elements
const startSpeechBtn = document.getElementById("startSpeechBtn");
const stopSpeechBtn = document.getElementById("stopSpeechBtn");
const transcriptsDiv = document.getElementById("transcripts");
const resultsDiv = document.getElementById("results");

let recognition = null;

// We'll store final recognized segments in an array, each with { text, timestamp }
let recognizedSegments = [];
let intervalId = null;  // For our 3-second interval

// Add event listeners
startSpeechBtn.addEventListener("click", startBrowserSTT);
stopSpeechBtn.addEventListener("click", stopBrowserSTT);

function startBrowserSTT() {
  console.log("[CLIENT] Starting browser speech recognition...");

  if (!("webkitSpeechRecognition" in window)) {
    alert("Your browser does not support the Web Speech API. Please use Chrome or Edge.");
    return;
  }

  // Clear any old data
  transcriptsDiv.textContent = "";
  resultsDiv.textContent = "";
  recognizedSegments = [];

  recognition = new webkitSpeechRecognition();
  recognition.continuous = true;
  recognition.interimResults = true;
  recognition.lang = "en-US";

  recognition.onresult = (event) => {
    let finalSoFar = "";
    let interim = "";

    // Gather recognized speech
    for (let i = event.resultIndex; i < event.results.length; i++) {
      const transcriptChunk = event.results[i][0].transcript;
      if (event.results[i].isFinal) {
        // We got a final segment
        recognizedSegments.push({
          text: transcriptChunk.trim(),
          timestamp: Date.now()
        });
        finalSoFar += transcriptChunk + " ";
      } else {
        // An interim piece
        interim += transcriptChunk;
      }
    }

    // Show real-time transcript: combine all final + current interim
    const allFinalText = recognizedSegments.map(seg => seg.text).join(" ");
    transcriptsDiv.textContent = allFinalText + " [ " + interim + " ]";
  };

  recognition.onerror = (event) => {
    console.error("[CLIENT] Speech Recognition Error:", event.error);
  };

  recognition.onend = () => {
    console.log("[CLIENT] Speech recognition ended.");
    // Usually fired when user stops speaking or we call .stop()
  };

  recognition.start();
  startSpeechBtn.disabled = true;
  stopSpeechBtn.disabled = false;

  // Start checking every 3 seconds for new segments
  intervalId = setInterval(() => {
    checkLast3Seconds();
  }, 3000);
}

function stopBrowserSTT() {
  console.log("[CLIENT] Stopping speech recognition...");

  if (recognition) {
    recognition.stop();
    recognition = null;
  }
  startSpeechBtn.disabled = false;
  stopSpeechBtn.disabled = true;

  // Clear the interval
  if (intervalId) {
    clearInterval(intervalId);
    intervalId = null;
  }
}

// ------------------ EVERY 3 SECONDS: CHECK LAST 3s OF SPEECH ------------------
function checkLast3Seconds() {
  if (!recognizedSegments.length) return;

  const now = Date.now();
  // We'll gather text that arrived in the last 3 seconds
  let chunkText = "";
  for (let seg of recognizedSegments) {
    // If within last 3s
    if (now - seg.timestamp <= 3000) {
      chunkText += seg.text + " ";
    }
  }

  chunkText = chunkText.trim();
  if (!chunkText) {
    // No new final text in the last 3s
    return;
  }

  console.log("[CLIENT] Checking chunk (last 3s):", chunkText);

  fetch("/api/process-chunk", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ chunk: chunkText })
  })
    .then(res => res.json())
    .then(data => {
      console.log("[CLIENT] /api/process-chunk response:", data);
      if (data.is_relevant) {
        // Show the relevant query and perplexity result
        // We'll just append it to "results" so user can see ongoing findings
        const p = document.createElement("p");
        p.innerHTML = `
          <strong>Relevant Query:</strong> ${data.query}<br />
          <strong>Perplexity:</strong> ${data.perplexity_result || "No data"}
        `;
        // Insert at the top or bottom, your choice:
        resultsDiv.prepend(p);
      }
    })
    .catch(err => {
      console.error("[CLIENT] Error calling /api/process-chunk:", err);
    });
}
