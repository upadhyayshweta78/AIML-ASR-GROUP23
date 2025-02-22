document.addEventListener("DOMContentLoaded", () => {
  const transcribeButton = document.getElementById("transcribeButton");
  const recordButton = document.getElementById("recordButton");
  const stopButton = document.getElementById("stopButton");
  const audioFileInput = document.getElementById("audioFile");
  const transcriptionDiv = document.getElementById("transcription");
  const completedTranscriptionDiv = document.getElementById("completedTranscription");
  const completedTranscriptionText = document.getElementById("completedTranscriptionText");
  const loadingDiv = document.getElementById("loading");

  let mediaRecorder;
  let audioChunks = [];

  // Enable transcribe button when a file is selected
  audioFileInput.addEventListener("change", () => {
    transcribeButton.disabled = !audioFileInput.files.length;
  });

  // Transcribe audio file
  transcribeButton.addEventListener("click", async () => {
    const file = audioFileInput.files[0];
    if (file) {
      const formData = new FormData();
      formData.append("file", file);

      loadingDiv.style.display = "block";
      transcriptionDiv.style.display = "none";
      completedTranscriptionDiv.style.display = "none";

      try {
        const response = await fetch("http://localhost:8000/transcribe-file", {
          method: "POST",
          body: formData,
        });
        const data = await response.json();
        loadingDiv.style.display = "none";
        transcriptionDiv.style.display = "block";
        transcriptionDiv.innerText = `Transcription: \n${data.transcription}`;
      } catch (error) {
        loadingDiv.style.display = "none";
        transcriptionDiv.style.display = "block";
        transcriptionDiv.innerText = "Error during transcription.";
        console.error(error);
      }
    }
  });

  // Handle audio recording
  let audioStream;

  recordButton.addEventListener("click", async () => {
    audioChunks = [];
    try {
      audioStream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRecorder = new MediaRecorder(audioStream);

      mediaRecorder.ondataavailable = (event) => {
        audioChunks.push(event.data);
      };

      mediaRecorder.onstop = () => {
        const audioBlob = new Blob(audioChunks, { type: "audio/wav" });
        const audioUrl = URL.createObjectURL(audioBlob);

        // Send the recorded audio to the backend via WebSocket
        const ws = new WebSocket("ws://localhost:8000/ws/transcribe");
        ws.onopen = () => {
          const reader = new FileReader();
          reader.onloadend = () => {
            const buffer = reader.result;
            ws.send(buffer);
          };
          reader.readAsArrayBuffer(audioBlob);
        };

        ws.onmessage = (event) => {
          const data = JSON.parse(event.data);
          completedTranscriptionText.innerText = `Transcription: \n${data.text}`;
          completedTranscriptionDiv.style.display = "block";
        };

        ws.onerror = (error) => {
          completedTranscriptionText.innerText = "Error during transcription.";
          completedTranscriptionDiv.style.display = "block";
          console.error(error);
        };
      };

      mediaRecorder.start();
      recordButton.disabled = true;
      stopButton.disabled = false;
    } catch (error) {
      console.error("Error accessing microphone:", error);
    }
  });

  // Stop recording and process audio
  stopButton.addEventListener("click", () => {
    mediaRecorder.stop();
    audioStream.getTracks().forEach((track) => track.stop());
    stopButton.disabled = true;
    recordButton.disabled = false;
  });
});
