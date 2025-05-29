import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  test: {
    environment: 'jsdom', // Gunakan jsdom untuk menguji komponen Reaact
    setupFiles: ['./src/test/setupTests.js'], // Tambahkan file setup untuk pengujian
    globals: true, // Gunakan variabel global seperti window dan document
  },
})
