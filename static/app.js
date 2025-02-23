let recognizing = false;
let recognition;
let transcriptBuffer = "";

// Check for speech recognition
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

// Start/Stop speech
document.getElementById("startSpeechBtn").addEventListener("click", () => {
  if (!recognizing) recognition.start();
});

document.getElementById("stopSpeechBtn").addEventListener("click", () => {
  if (recognizing) recognition.stop();
});

// Periodically send transcripts to OpenAI route
setInterval(() => {
  if (!transcriptBuffer.trim()) return;

  const chunk = transcriptBuffer.trim();
  transcriptBuffer = "";

  fetch("/api/process-chunk", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ chunk })
  })
    .then(res => res.json())
    .then(data => {
      console.log("OpenAI (gpt-4o) Response:", data);
      const resultsDiv = document.getElementById("results");
      resultsDiv.textContent += `\n[Model: ${data.model}] ${data.openai_response}`;
    })
    .catch(err => console.error("Error calling /api/process-chunk:", err));
}, 2000);

// ---------------------------------------------
// Perplexity Integration (Non-Streaming)
// ---------------------------------------------
const perplexityChatBtn = document.getElementById("perplexityChatBtn");
const perplexityPrompt = document.getElementById("perplexityPrompt");
const perplexityOutput = document.getElementById("perplexityOutput");

perplexityChatBtn.addEventListener("click", () => {
  const prompt = perplexityPrompt.value.trim();
  if (!prompt) {
    perplexityOutput.textContent = "Please enter a question before clicking send!";
    return;
  }

  perplexityOutput.textContent = "Loading Perplexity response...";

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
        // Show the Perplexity text
        perplexityOutput.textContent = data.perplexity_response;
      }
    })
    .catch(err => {
      perplexityOutput.textContent = "Error calling /api/perplexity-chat.";
      console.error(err);
    });
});

