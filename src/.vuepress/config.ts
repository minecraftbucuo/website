import { defineUserConfig } from "vuepress";

import theme from "./theme.js";

export default defineUserConfig({
  base: "/website/",

  lang: "zh-CN",
  title: "我的世界不错",
  description: "莫道桑榆晚，为霞尚满天。",

  head: [
    ["link", { rel: "icon", type: "image/svg+xml", href: "/website/logo.svg" }],
  ],

  theme,
  
  // 和 PWA 一起启用
  // shouldPrefetch: false,
});