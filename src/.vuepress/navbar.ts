import { navbar } from "vuepress-theme-hope";

export default navbar([
  { text: "项目主页", icon: "house", link: "/" },
  { text: "算法相关", icon: "flag-checkered", link: "/算法相关/" },
  { text: "技术相关", icon: "screwdriver-wrench", link: "/技术相关/" },
  { text: "数学相关", icon: "infinity", link: "/数学相关/" },
  { text: "关于本站", icon: "book", link: "/about.md" },
  // {
  //   text: "指南",
  //   icon: "lightbulb",
  //   prefix: "/guide/",
  //   children: [
  //     {
  //       text: "Bar",
  //       icon: "lightbulb",
  //       prefix: "bar/",
  //       children: ["baz", { text: "...", icon: "ellipsis", link: "" }],
  //     },
  //     {
  //       text: "Foo",
  //       icon: "lightbulb",
  //       prefix: "foo/",
  //       children: ["ray", { text: "...", icon: "ellipsis", link: "" }],
  //     },
  //   ],
  // },
  // {
  //   text: "V2 文档",
  //   icon: "book",
  //   link: "https://theme-hope.vuejs.press/zh/",
  // },
]);
