let recognizing = false;
let recognition;
let transcriptBuffer = "";

// Check for webkit speech recognition
if (!('webkitSpeechRecognition' in window)) {
  alert("Your browser does not support Speech Recognition. Please use Chrome.");
} else {
  recognition = new webkitSpeechRecognition();
  recognition.continuous = true;
  recognition.interimResults = true;

  recognition.onstart = () => {
    recognizing = true;
    document.getElementById("startSpeechBtn").disabled = true;
    document.getElementById("stopSpeechBtn").disabled = false;
    console.log("Speech recognition started");
  };

  recognition.onresult = (event) => {
    let interimTranscript = "";
    for (let i = event.resultIndex; i < event.results.length; i++) {
      let transcript = event.results[i][0].transcript;
      if (event.results[i].isFinal) {
        transcriptBuffer += transcript + " ";
      } else {
        interimTranscript += transcript;
      }
    }
    // Display the transcripts + interim
    document.getElementById("transcripts").textContent =
      transcriptBuffer + "\n[Interim]: " + interimTranscript;
  };

  recognition.onerror = (e) => {
    console.log("Speech recognition error: ", e);
  };

  recognition.onend = () => {
    recognizing = false;
    document.getElementById("startSpeechBtn").disabled = false;
    document.getElementById("stopSpeechBtn").disabled = true;
    console.log("Speech recognition stopped");
  };
}

// Start speech recognition
document.getElementById("startSpeechBtn").addEventListener("click", () => {
  if (!recognizing) recognition.start();
});

// Stop speech recognition
document.getElementById("stopSpeechBtn").addEventListener("click", () => {
  if (recognizing) recognition.stop();
});

// Periodically send the collected transcript to the server
setInterval(() => {
  if (!transcriptBuffer.trim()) return; // Skip if empty

  const chunk = transcriptBuffer.trim();
  transcriptBuffer = ""; // Reset local buffer

  // Call our Flask API
  fetch("/api/process-chunk", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ chunk })
  })
    .then((res) => res.json())
    .then((data) => {
      console.log("OpenAI Response:", data);
      const resultsDiv = document.getElementById("results");
      resultsDiv.textContent += `\n[Model: ${data.model}] ${data.openai_response}`;
    })
    .catch((err) => console.error("Error calling /api/process-chunk:", err));
}, 2000);
