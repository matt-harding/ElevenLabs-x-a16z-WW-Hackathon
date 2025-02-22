// -------------- DOM ELEMENTS --------------
const startSpeechBtn = document.getElementById("startSpeechBtn");
const stopSpeechBtn = document.getElementById("stopSpeechBtn");
const transcriptsDiv = document.getElementById("transcripts");
const resultsDiv = document.getElementById("results");

let recognition = null;
let finalTranscript = "";

// -------------- EVENT LISTENERS --------------
startSpeechBtn.addEventListener("click", startBrowserSTT);
stopSpeechBtn.addEventListener("click", stopBrowserSTT);

// -------------- WEB SPEECH API --------------
function startBrowserSTT() {
  console.log("[CLIENT] Starting browser speech recognition...");
  // Check for browser support
  if (!("webkitSpeechRecognition" in window)) {
    alert("Your browser does not support the Web Speech API. Please use Chrome or Edge.");
    return;
  }

  // Reset transcripts and results
  transcriptsDiv.textContent = "";
  resultsDiv.textContent = "";
  finalTranscript = "";

  recognition = new webkitSpeechRecognition();
  recognition.continuous = true;
  recognition.interimResults = true;
  recognition.lang = "en-US";

  recognition.onresult = (event) => {
    let interim = "";
    for (let i = event.resultIndex; i < event.results.length; i++) {
      const transcriptChunk = event.results[i][0].transcript;
      if (event.results[i].isFinal) {
        // Add to final transcript
        finalTranscript += transcriptChunk + " ";
      } else {
        // Build an interim transcript
        interim += transcriptChunk;
      }
    }
    // Show combined text: final + interim
    transcriptsDiv.textContent = finalTranscript + " [ " + interim + " ]";
  };

  recognition.onerror = (event) => {
    console.error("[CLIENT] Speech Recognition Error:", event.error);
  };

  recognition.onend = () => {
    console.log("[CLIENT] Speech recognition ended.");
  };

  recognition.start();
  startSpeechBtn.disabled = true;
  stopSpeechBtn.disabled = false;
}

function stopBrowserSTT() {
  console.log("[CLIENT] Stopping speech recognition...");
  if (recognition) {
    recognition.stop();
    recognition = null;
  }
  stopSpeechBtn.disabled = true;
  startSpeechBtn.disabled = false;

  // Clean up the final transcript text
  finalTranscript = finalTranscript.trim();
  transcriptsDiv.textContent = finalTranscript;

  if (finalTranscript) {
    console.log("[CLIENT] Sending final transcript to server:", finalTranscript);
    resultsDiv.textContent = "Analyzing text with OpenAI + Perplexity...";

    fetch("/api/process-text", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: finalTranscript })
    })
      .then((res) => res.json())
      .then((data) => {
        console.log("[CLIENT] Received response from server:", data);
        if (!data.is_relevant) {
          resultsDiv.textContent = "NOT RELEVANT";
          return;
        }

        // If relevant, show the query and results from Perplexity
        const query = data.query;
        const perplexityText = data.perplexity_result || "No Perplexity data";

        resultsDiv.innerHTML = `
          <p><strong>Relevant Query:</strong> ${query}</p>
          <p><strong>Perplexity Response:</strong> ${perplexityText}</p>
        `;
      })
      .catch((err) => {
        console.error("[CLIENT] Error analyzing text:", err);
        resultsDiv.textContent = "Error analyzing text.";
      });
  }
}
