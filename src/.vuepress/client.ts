import { defineClientConfig } from "vuepress/client";

import MazeGame from "./components/MazeGame.vue";

export default defineClientConfig({
  enhance: ({ app }) => {
    app.component("MazeGame", MazeGame);
  },
});
