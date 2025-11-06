<template>
  <div class="app">
    <Toast />
    <div class="hero">
      <!-- Background lung illustration -->
      <img src="/lungs.png" alt="Lungs" class="lung-img" />

      <!-- Upload + Prediction Panel -->
      <div class="panel">
        <h2>AI Lung Analyzer</h2>
        <p>Upload a chest X-ray image to detect signs of Pneumonia.</p>

        <FileUpload
          name="file"
          accept="image/*"
          :maxFileSize="2000000"
          customUpload
          @uploader="onUpload"
          chooseLabel="Select Image"
          class="upload-box"
        >
          <template #empty>
            <div class="flex flex-col items-center justify-center">
              <i class="pi pi-cloud-upload text-4xl text-blue-400 mb-2" />
              <p>Click or drag an image here</p>
            </div>
          </template>
        </FileUpload>

        <div v-if="previewUrl" class="preview mt-3">
          <img :src="previewUrl" alt="Preview" />
        </div>

        <div v-if="loading" class="mt-3">
          <ProgressBar mode="indeterminate" style="height: 8px;" />
          <p class="mt-2 text-sm text-blue-200">Analyzing image...</p>
        </div>

        <div v-if="result" class="result mt-4">
          <h3>Prediction Result</h3>
          <p><b>Label:</b> {{ result.label }}</p>
          <p><b>Confidence:</b> {{ (result.probability * 100).toFixed(2) }}%</p>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref } from "vue";
import { useToast } from "primevue/usetoast";

const toast = useToast();
const previewUrl = ref(null);
const result = ref(null);
const loading = ref(false);

const onUpload = async (event) => {
  const file = event.files[0];
  if (!file) return;

  previewUrl.value = URL.createObjectURL(file);
  loading.value = true;

  try {
    const formData = new FormData();
    formData.append("file", file);

    const res = await fetch("http://127.0.0.1:8000/predict", {
      method: "POST",
      body: formData,
    });

    if (!res.ok) {
      throw new Error("Prediction failed");
    }

    const data = await res.json();
    result.value = data;
    toast.add({
      severity: "success",
      summary: "Prediction Complete",
      detail: `${data.label} (${(data.probability * 100).toFixed(2)}%)`,
      life: 4000,
    });
  } catch (e) {
    toast.add({ severity: "error", summary: "Error", detail: e.message, life: 3000 });
  } finally {
    loading.value = false;
  }
};
</script>

<style scoped>
.app {
  background: linear-gradient(135deg, #001f3f, #000307);
  min-height: 100vh;
  display: flex;
  justify-content: center;
  align-items: center;
  color: white;
  overflow: hidden;
  font-family: 'Segoe UI', sans-serif;
}

.hero {
  display: flex;
  flex-direction: row;
  align-items: center;
  gap: 3rem;
  text-align: left;
}

.lung-img {
  width: 350px;
  filter: drop-shadow(0 0 25px #00bfff);
  animation: pulse 3s infinite alternate;
}

@keyframes pulse {
  from {
    opacity: 0.8;
    transform: scale(1);
  }
  to {
    opacity: 1;
    transform: scale(1.05);
  }
}

.panel {
  background: rgba(255, 255, 255, 0.1);
  padding: 2rem;
  border-radius: 20px;
  backdrop-filter: blur(10px);
  width: 400px;
}

.upload-box {
  margin-top: 1rem;
}

.preview img {
  width: 100%;
  border-radius: 10px;
  margin-top: 0.5rem;
  box-shadow: 0 0 10px rgba(0, 191, 255, 0.4);
}

.result {
  background: rgba(0, 0, 0, 0.3);
  padding: 1rem;
  border-radius: 10px;
}
</style>
