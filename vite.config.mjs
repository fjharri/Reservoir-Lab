import {defineConfig} from 'vite';
// Optional developer preview; production is the unbuilt public/ directory.
export default defineConfig({root:'public',publicDir:false,server:{host:'0.0.0.0',allowedHosts:['terminal.local']}});
