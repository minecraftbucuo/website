import { defineClientConfig } from "vuepress/client";

import MazeGame from "./components/MazeGame.vue";
import HanoiGame from "./components/HanoiGame.vue";

export default defineClientConfig({
  enhance: ({ app }) => {
    app.component("MazeGame", MazeGame);
    app.component("HanoiGame", HanoiGame);
  },
});
