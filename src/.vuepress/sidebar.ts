import { sidebar } from "vuepress-theme-hope";

export default sidebar({
  "/": [
    "",
    "/算法相关/",
    "/about.md",
    // {
    //   text: "案例",
    //   icon: "laptop-code",
    //   prefix: "demo/",
    //   link: "demo/",
    //   children: "structure",
    // },
    // {
    //   text: "文档",
    //   icon: "book",
    //   prefix: "guide/",
    //   children: "structure",
    // },
  ],
  "/算法相关": [
    {
      text: "算法相关",
      icon: "laptop-code",
      link: "",
      children: [
        {
          text: "算法模板",
          icon: "laptop-code",
          link: "算法模板.md",
        },
        {
          text: "算法刷题记录",
          icon: "laptop-code",
          link: "算法做题记录/",
        },
      ],
    },
  ]
});
