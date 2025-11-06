<!-- App.vue -->
<template>
  <div class="card p-4">
    <!-- Toast for notifications -->
    <Toast />

    <!-- Your customized FileUpload component -->
    <FileUpload
      name="demo[]"
      url="/api/upload"
      @upload="onTemplatedUpload"
      :multiple="true"
      accept="image/*"
      :maxFileSize="1000000"
      @select="onSelectedFiles"
    >
      <!-- Header slot -->
      <template #header="{ chooseCallback, uploadCallback, clearCallback, files }">
        <div class="flex flex-wrap justify-between items-center gap-4">
          <div class="flex gap-2">
            <Button @click="chooseCallback" icon="pi pi-images" rounded outlined severity="secondary" />
            <Button @click="uploadEvent(uploadCallback)" icon="pi pi-cloud-upload" rounded outlined severity="success" :disabled="!files || files.length === 0" />
            <Button @click="clearCallback()" icon="pi pi-times" rounded outlined severity="danger" />
          </div>
          <ProgressBar :value="totalSizePercent" showValue="false" class="w-full md:w-20rem h-1 md:ml-auto">
            <span class="whitespace-nowrap">{{ totalSize }}B / 1MB</span>
          </ProgressBar>
        </div>
      </template>

      <!-- Content slot for file previews -->
      <template #content="{ files, uploadedFiles, removeUploadedFileCallback, removeFileCallback }">
        <div class="flex flex-col gap-8 pt-4">
          <div v-if="files.length > 0">
            <h5>Pending</h5>
            <div class="flex flex-wrap gap-4">
              <div v-for="(file, index) of files" :key="file.name + file.type + file.size" class="p-4 rounded border flex flex-col items-center gap-2 border-surface">
                <img :src="file.objectURL" :alt="file.name" width="100" height="50" />
                <span class="font-semibold max-w-60 whitespace-nowrap overflow-hidden">{{ file.name }}</span>
                <div>{{ formatSize(file.size) }}</div>
                <Badge value="Pending" severity="warn" />
                <Button icon="pi pi-times" @click="onRemoveTemplatingFile(file, removeFileCallback, index)" rounded severity="danger" outlined />
              </div>
            </div>
          </div>

          <div v-if="uploadedFiles.length > 0">
            <h5>Completed</h5>
            <div class="flex flex-wrap gap-4">
              <div v-for="(file, index) of uploadedFiles" :key="file.name + file.type + file.size" class="p-4 rounded border flex flex-col items-center gap-2 border-surface">
                <img :src="file.objectURL" :alt="file.name" width="100" height="50" />
                <span class="font-semibold max-w-60 whitespace-nowrap overflow-hidden">{{ file.name }}</span>
                <div>{{ formatSize(file.size) }}</div>
                <Badge value="Completed" severity="success" class="mt-2" />
                <Button icon="pi pi-times" @click="removeUploadedFileCallback(index)" rounded severity="danger" outlined />
              </div>
            </div>
          </div>
        </div>
      </template>

      <!-- Empty slot -->
      <template #empty>
        <div class="flex flex-col items-center justify-center p-4">
          <i class="pi pi-cloud-upload text-4xl text-muted" />
          <p class="mt-4 mb-0">Drag and drop files to here to upload.</p>
        </div>
      </template>
    </FileUpload>
  </div>
</template>

<script setup>
import { ref } from 'vue';
import { useToast } from 'primevue/usetoast';

const toast = useToast();

const totalSize = ref(0);
const totalSizePercent = ref(0);
const files = ref([]);

const onSelectedFiles = (event) => {
  files.value = event.files;
  files.value.forEach((file) => {
    totalSize.value += file.size;
  });
  totalSizePercent.value = Math.min((totalSize.value / 10), 100);
};

const uploadEvent = (callback) => {
  totalSizePercent.value = Math.min((totalSize.value / 10), 100);
  callback();
};

const onTemplatedUpload = () => {
  toast.add({ severity: 'info', summary: 'Success', detail: 'File Uploaded', life: 3000 });
  // Reset after upload
  totalSize.value = 0;
  totalSizePercent.value = 0;
  files.value = [];
};

const onRemoveTemplatingFile = (file, removeFileCallback, index) => {
  removeFileCallback(index);
  totalSize.value -= file.size;
  totalSizePercent.value = Math.min((totalSize.value / 10), 100);
};

const formatSize = (bytes) => {
  const k = 1024;
  const dm = 2;
  if (bytes === 0) return '0 B';
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  const sizes = ['B', 'KB', 'MB', 'GB'];
  const size = parseFloat((bytes / Math.pow(k, i)).toFixed(dm));
  return `${size} ${sizes[i]}`;
};
</script>