import { sidebar } from "vuepress-theme-hope";

export default sidebar({
  "/": [
    "",
    "/算法相关/",
    "/技术相关/",
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
          text: "算法模板初稿",
          icon: "laptop-code",
          link: "算法模板.md",
        },
        {
          text: "算法刷题记录",
          icon: "laptop-code",
          link: "算法刷题记录/",
          children: [
            {
              text: "2025暑假算法刷题记录",
              icon: "laptop-code",
              link: "算法刷题记录/2025暑假算法刷题记录.md",
            },
            {
              text: "大二上算法刷题记录",
              icon: "laptop-code",
              link: "算法刷题记录/大二上算法刷题记录.md",
            },
          ],
        },
        {
          text: "学长的算法模版",
          icon: "laptop-code",
          link: "NoTeamName-XCPC-v1.0.0.md",
        },
      ],
    },
  ]
});
