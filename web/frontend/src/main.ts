import { createApp } from "vue";
import App from "./App.vue";

import PrimeVue from "primevue/config";
import Aura from "@primeuix/themes/aura";
import "primeicons/primeicons.css";

// 👇 ADD THIS
import ToastService from "primevue/toastservice";

const app = createApp(App);

app.use(PrimeVue, {
  theme: {
    preset: Aura,
  },
});

// 👇 ADD THIS
app.use(ToastService);

// Register PrimeVue components globally if desired
import Button from "primevue/button";
import Toast from "primevue/toast";
import FileUpload from "primevue/fileupload";
import ProgressBar from "primevue/progressbar";
import Badge from "primevue/badge";

app.component("Button", Button);
app.component("Toast", Toast);
app.component("FileUpload", FileUpload);
app.component("ProgressBar", ProgressBar);
app.component("Badge", Badge);

app.mount("#app");
