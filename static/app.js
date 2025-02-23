let recognizing = false;
let recognition;
let transcriptBuffer = "";

// Check for webkit speech
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

// Start speech
document.getElementById("startSpeechBtn").addEventListener("click", () => {
  if (!recognizing) recognition.start();
});

// Stop speech
document.getElementById("stopSpeechBtn").addEventListener("click", () => {
  if (recognizing) recognition.stop();
});

// Periodically send the transcripts to /api/process-chunk
setInterval(() => {
  if (!transcriptBuffer.trim()) return;

  const chunk = transcriptBuffer.trim();
  transcriptBuffer = ""; // reset local buffer

  fetch("/api/process-chunk", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ chunk })
  })
    .then(res => res.json())
    .then(data => {
      console.log("OpenAI (gpt-4o) Response:", data);
      // Show it in #results
      const resultsDiv = document.getElementById("results");
      resultsDiv.textContent += `\n[Model: ${data.model}] ${data.openai_response}`;

      // *** Automatically populate the Perplexity prompt input with the new GPT response ***
      const perplexityPrompt = document.getElementById("perplexityPrompt");
      perplexityPrompt.value = data.openai_response;

    })
    .catch(err => console.error("Error calling /api/process-chunk:", err));
}, 2000);

// -------------------------------------------
// Perplexity Integration (non-streaming)
// -------------------------------------------
const perplexityChatBtn = document.getElementById("perplexityChatBtn");
const perplexityOutput = document.getElementById("perplexityOutput");

perplexityChatBtn.addEventListener("click", () => {
  const prompt = document.getElementById("perplexityPrompt").value.trim();
  if (!prompt) {
    perplexityOutput.textContent = "No prompt to investigate.";
    return;
  }

  perplexityOutput.textContent = "Investigating with Perplexity...";

  fetch("/api/perplexity-chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prompt })
  })
    .then(res => res.json())
    .then(data => {
      if (data.error) {
        perplexityOutput.textContent = "Error: " + data.error;
      } else {
        perplexityOutput.textContent = data.perplexity_response;
      }
    })
    .catch(err => {
      perplexityOutput.textContent = "Error calling /api/perplexity-chat.";
      console.error(err);
    });
});
