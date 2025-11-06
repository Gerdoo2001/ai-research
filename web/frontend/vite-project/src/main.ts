// main.js
import { createApp } from 'vue';
import App from './App.vue';

import PrimeVue from 'primevue/config';

// Import the desired theme, e.g., Aura
import Aura from '@primeuix/themes/aura';

// Import core PrimeVue styles
import 'primevue/resources/primevue.min.css';

// Import the theme CSS
import '@primeuix/themes/aura/theme.css';

// Import PrimeIcons for icons used in buttons
import 'primeicons/primeicons.css';

const app = createApp(App);

app.use(PrimeVue, {
  theme: {
    preset: Aura
  }
});

// Register PrimeVue components globally if desired
import Button from 'primevue/button';
import Toast from 'primevue/toast';
import FileUpload from 'primevue/fileupload';
import ProgressBar from 'primevue/progressbar';
import Badge from 'primevue/badge';

app.component('Button', Button);
app.component('Toast', Toast);
app.component('FileUpload', FileUpload);
app.component('ProgressBar', ProgressBar);
app.component('Badge', Badge);

app.mount('#app');