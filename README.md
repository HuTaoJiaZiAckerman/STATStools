<link rel="stylesheet" href="./static/css/main.css">
<div align="center">

[English](README.en.md) | 简体中文

<a href="https://hellogithub.com/repository/Achuan-2/SlideSCI" target="_blank"><img src="./static/img/ApacheArrow.png" alt="Featured｜HelloGitHub" style="width: 250px; height: 54px;" width="250" height="54" /></a>

</div>
<div class="social-badges">
  <div class="custom-social-badge"> <!-- GitHub -->
    <span>
      <div class="widget widget-lg"><a class="btn" href="https://github.com/apache/arrow" rel="noopener noopener" target="_blank" aria-label="Star Apache Arrow on GitHub"><svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewbox="0 0 16 16" fill="none" role="img" aria-hidden="true"><path d="M8 0c4.42 0 8 3.58 8 8a8.013 8.013 0 0 1-5.45 7.59c-.4.08-.55-.17-.55-.38 0-.27.01-1.13.01-2.2 0-.75-.25-1.23-.54-1.48 1.78-.2 3.65-.88 3.65-3.95 0-.88-.31-1.59-.82-2.15.08-.2.36-1.02-.08-2.12 0 0-.67-.22-2.2.82-.64-.18-1.32-.27-2-.27-.68 0-1.36.09-2 .27-1.53-1.03-2.2-.82-2.2-.82-.44 1.1-.16 1.92-.08 2.12-.51.56-.82 1.28-.82 2.15 0 3.06 1.86 3.75 3.64 3.95-.23.2-.44.55-.51 1.07-.46.21-1.61.55-2.33-.66-.15-.24-.6-.83-1.23-.82-.67.01-.27.38.01.53.34.19.73.9.82 1.13.16.45.68 1.31 2.69.94 0 .67.01 1.3.01 1.49 0 .21-.15.45-.55.38A7.995 7.995 0 0 1 0 8c0-4.42 3.58-8 8-8Z" fill="#000"></path></svg> <span>Star</span></a></div>
    </span>
  </div>
  <div class="custom-social-badge"> <!-- LinkedIn -->
    <span>
      <div class="widget widget-lg"><a class="btn" href="https://www.linkedin.com/company/apache-arrow/" rel="noopener noopener" target="_blank" aria-label="Follow Apache Arrow on LinkedIn"><svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewbox="0 0 16 16" fill="none" role="img" aria-hidden="true"><path d="M14.5455 0H1.45455C0.650909 0 0 0.650909 0 1.45455V14.5455C0 15.3491 0.650909 16 1.45455 16H14.5455C15.3491 16 16 15.3491 16 14.5455V1.45455C16 0.650909 15.3491 0 14.5455 0ZM5.05746 13.0909H2.912V6.18764H5.05746V13.0909ZM3.96291 5.20073C3.27127 5.20073 2.712 4.64 2.712 3.94982C2.712 3.25964 3.272 2.69964 3.96291 2.69964C4.65236 2.69964 5.21309 3.26036 5.21309 3.94982C5.21309 4.64 4.65236 5.20073 3.96291 5.20073ZM13.0938 13.0909H10.9498V9.73382C10.9498 8.93309 10.9353 7.90327 9.83491 7.90327C8.71855 7.90327 8.54691 8.77527 8.54691 9.67564V13.0909H6.40291V6.18764H8.46109V7.13091H8.49018C8.77673 6.58836 9.47636 6.016 10.52 6.016C12.6924 6.016 13.0938 7.44582 13.0938 9.30473V13.0909V13.0909Z" fill="#000"></path></svg> <span>Follow</span></a></div>
    </span>
  </div>
  <div class="custom-social-badge"> <!-- BlueSky -->
    <span>
      <div class="widget widget-lg"><a class="btn" href="https://bsky.app/profile/arrow.apache.org" rel="noopener noopener" target="_blank" aria-label="Follow Apache Arrow on BlueSky"><svg xmlns="http://www.w3.org/2000/svg" fill="none" viewbox="0 0 600 530" width="16" height="16" role="img" aria-hidden="true"><path fill="#000" d="M407.8 294.7c-3.3-.4-6.7-.8-10-1.3c3.4 .4 6.7 .9 10 1.3zM288 227.1C261.9 176.4 190.9 81.9 124.9 35.3C61.6-9.4 37.5-1.7 21.6 5.5C3.3 13.8 0 41.9 0 58.4S9.1 194 15 213.9c19.5 65.7 89.1 87.9 153.2 80.7c3.3-.5 6.6-.9 10-1.4c-3.3 .5-6.6 1-10 1.4C74.3 308.6-9.1 342.8 100.3 464.5C220.6 589.1 265.1 437.8 288 361.1c22.9 76.7 49.2 222.5 185.6 103.4c102.4-103.4 28.1-156-65.8-169.9c-3.3-.4-6.7-.8-10-1.3c3.4 .4 6.7 .9 10 1.3c64.1 7.1 133.6-15.1 153.2-80.7C566.9 194 576 75 576 58.4s-3.3-44.7-21.6-52.9c-15.8-7.1-40-14.9-103.2 29.8C385.1 81.9 314.1 176.4 288 227.1z"></path></svg><span>Follow</span></a></div>
    </span>
  </div>
</div>


## 1 ❤️ 工具说明

首先查看工具列表：`statstools -h`，专为科学计算设计，全面处理 AppchrArrow的工具包，支持Bayes混合效应模型计算方差、组间方差、组内方差、可重复性（R=组间方差/组内方差）。

举个例子：这个工具可以查看parquet格式的文件：`statstools show_parquet -i stat.parquet -n 10`
<img alt="image" src="./static/img/statstools_show_parquet-h.png" style="width: 849px;" />​

## 2 📝 开发背景

不知道有没有人和我一样，对PPT积怨已久😡：

💔​**不能添加图片标题**：图片没法像word一样直接添加图片标题，只能手动插入文本框，对齐半天还歪七扭八！

💔​**不能复制元素的位置粘贴给另一个元素**：不同页PPT的类似元素要保持同一个位置，只能复制粘贴再修改，无法直接复制粘贴位置

💔**不能图片自动排列整齐：**  插入多张图片后，想要多行多列整齐排列？要么一张一张手动拖动，对齐到天荒地老！要么先一列列水平对齐再垂直对齐

💔**不能插入代码块：**  只能从外部编辑器（如VSCode）或专门网站复制粘贴，或者截图、生成图片粘贴代码块，有点麻烦

💔**不能插入latex数学公式：**  现在我基本上靠ai来识别和生成数学公式，公式都是latex数学公式格式，不方便直接粘贴到PPT里

……

市面上的ppt插件花里胡哨的功能一大堆，没几个能用得上。对我而言，每周要做研究生科研进展工作汇报，要的就是快速插入内容、做出内容清晰的PPT，不追求太美观。

在AI的帮助下，很快就把这些痛点功能都开发出来了 ！真的成就感满满！（这个插件99%的代码都是ai生成的，感谢AI老师！）

本着开源的精神，这个插件也在Github上开源了，欢迎大家给我点小星星！

Github地址：[https://github.com/Achuan-2/SlideSCI](https://github.com/Achuan-2/SlideSCI)

## 3 ✨ 主要功能

- **批量添加图片标题**：支持批量选中图片后，批量在图片下方添加居中图题，支持设置图片和图题是否自动编组  
  ​<img alt="" src="https://s2.loli.net/2025/08/29/OoXlgpGdrtx2bEP.png" />
- **批量添加图片标签**：用于科研绘图，可选择添加的标签模板（`A`、`a`、`A)`、`a)`、`1`、`1)`），默认添加的标签字体为`Arial`<img alt="" src="assets/network-asset-68747470733a2f2f666173746c792e6a7364656c6976722e6e65742f67682f41636875616e2d322f5069634265642f6173736574732f50697850696e5f323032352d30312d32345f31382d33342d31312d323032352d30312d32342e706e6-20250417103743-vry4kvj.png" />
- **图片自动排列**：可以自动排列多张图片，支持设置排序方式、排列方式、每列多少张图片、列间距多少、行间距多少（默认为空，为列间距大小）、图片宽高

  - **排序方式**：

    - 根据位置排序：可以根据用户手动粗排的图片位置，来自动判断图片的排列顺序。
    - 根据多选顺序：根据用户的多选选择顺序来排列
  - **排列方式**：

    - **列最大宽度占位排列**：按每列的最大宽度来占位排列，以保持表格布局，适用于科研绘图的图片排版
    - **统一高度排列**：默认统一图片的高度进行排列，如果不设置图片高度，则使用第一张图片的高度来统一图片高度
    - **统一宽度瀑布流**：默认统一图片的宽度进行排列，如果不设置图片宽度，则使用第一张图片的宽度来统一图片宽度  
      ​<img alt="" src="https://s2.loli.net/2025/08/29/RmxjZpTzGDL8evP.png" />
- **复制粘贴格式功能**  
  ​<img alt="" src="https://s2.loli.net/2025/08/29/3fjcw5KWbsLJkAo.png" />

  - **复制粘贴格式**：可以复制形状、文字的格式，粘贴给其他元素。
  - **复制位置和粘贴元素位置**：可以复制多个元素的位置，粘贴给其他元素（可以多选复制和粘贴！），可以用来让不同页的PPT的多个元素位置一致，或者让同一页的不同元素都是一个位置（可以先排好一组元素，用这个功能让另一组元素自动排好，再调整位置）。  
    ​<img alt="" src="https://s2.loli.net/2025/08/29/q5vblI3nrDhewJ6.gif" />  
      
    ​<img alt="" src="https://s2.loli.net/2025/08/29/u19w8IMkOYjE3rz.gif" />
  - **复制和粘贴元素宽高**：支持多选同时粘贴宽高，快速统一图片宽高
- **支持插入代码块自动高亮**  
  ​<img alt="" src="https://s2.loli.net/2025/08/29/jbSgDfnP69eZopV.png" />

  - **支持代码语言高亮列表**：matlab、python、r、js、html、css、csharp、Fortran
  - **支持切换黑白背景色**：默认是黑色背景色，切换为背景色，只需要点击「代码黑色背景色」按钮取消激活状态即可
- **支持插入latex数学公式**  
  ​「插入Markdown」可以快速插入文字与数学混排
  <img width="1926" height="1106" alt="image" src="https://github.com/user-attachments/assets/8d345baf-41af-473d-a182-3c7d2864c155" />

  「插入LaTeX文字」单独插入可编辑的数学公式

  <img width="1889" height="1203" alt="image" src="https://github.com/user-attachments/assets/d33eba81-4241-4ec6-bc73-057087be8a5e" />

  如果数学公式比较复杂，PPT原生不支持，可以选择「插入LaTeX svg」，支持更多latex公式样式，只需要简单配置下node.js环境即可使用，而不需要像IguanaTeX等插件除了安装LaTeX环境还要安装很多插件才能实现latex转svg，本插件的安装简单很多
  <img width="1756" height="1202" alt="PixPin_2025-09-30_10-21-00" src="https://github.com/user-attachments/assets/4d7732b2-47da-41e1-bb22-ddfceab4604b" />
  插入的svg的图片提示文字会保留原始LaTeX代码，方便修改后再插入
  <img width="1768" height="979" alt="PixPin_2025-09-30_10-25-48" src="https://github.com/user-attachments/assets/b5bbeaf1-8b12-44b1-9664-4591741e83ea" />



  > 配置「插入LaTeX svg」功能的方法
  >
  > - 配置[Node.js](https://nodejs.org/zh-cn/)环境
  > - 进入插件安装文件夹的`latex-converter`文件夹，默认为`%APPDATA%\Achuan-2\SlideSCI\latex-converter`
  > - 打开命令行，运行`npm install`安装环境
  > - 配置完成，可以使用「插入LaTeX svg」功能


- **支持插入Markdown文本**：可以直接把markdown整篇笔记一口气全部粘贴到PPT里！并按原文顺序排列！  
  ​<img alt="" src="https://s2.loli.net/2025/08/29/MPKOgWonijCsl4D.png" />

  - 支持行内格式：加粗、下划线、上标、下标、斜体、链接、行级代码、行级数学公式
  - 支持块级格式：标题、列表、代码块、表格、数学公式、引述块

    - 列表特别处理：

      - 保留列表悬挂缩进：常规粘贴html中的列表到PPT里会丢失悬挂缩进，本插件粘贴列表可以保留悬挂缩进
      - 支持任务列表粘贴，列表项符号会转化为☑和☐，代表完成和未完成
    - 代码块特别处理：

      - 独立文本框，支持设置黑白背景高亮配色，支持PPT直接再编辑
    - 表格特别处理：

      - 默认限制500px宽度，添加1pt黑色边框
    - 数学公式特别处理：

      - 独立文本框，支持PPT直接再编辑
    - 引述块特别处理：

      - 独立文本框，添加黑色边框

## 4 🪟 支持环境

插件在Windows11 使用[Visual Studio Tools For Office](https://www.visualstudio.com/de/vs/office-tools/) 和C#语言开发，专为Microsoft Powerpoint设计，兼容安装到WPS（注：WPS版本不支持插入latex数学公式、插入markdown笔记，强行使用会造成卡死）

> 插件只支持Windows端，不支持Mac端（因为Windows端的PPT插件和Mac端插件开发架构不一样，市面上很多的PPT插件，Windows端和Mac端都是独立开发的界面）

## 5 🖥️ 安装方法

下载本插件Github页面[Release](https://github.com/Achuan-2/my_ppt_plugin/releases)中的exe文件，双击安装即可

注意：安装时需要先退出PPT，否则PPT不会即时加载该插件

需要的安装环境（一般插件安装时会自动提示安装）

- 运行环境1：[Microsoft .Net Framework 4.0或更高](https://www.microsoft.com/zh-cn/download/details.aspx?id=17718)
- 运行环境2：[Microsoft Visual Studio 2010 Tools for Office Runtime](https://www.microsoft.com/zh-cn/download/details.aspx?id=105522)

如果插件安装后无法正常运行、没有在PPT中显示，或者在开发工具→COM加载项里勾选插件，提示“未加载。加载COM加载项时出现运行错误”，请尝试安装上面的环境依赖。

## 6 📝 使用介绍视频

- [让AI帮我我开发了一个PPT插件！支持一键添加图片标题，复制粘贴位置、一键图片对齐、一键插入代码块、一键插入latex公式！_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV15YcmeSEbq/?spm_id_from=0.0.homepage.video_card.click&vd_source=b4a1fcb6dce305e26d8d16d9cbb71304)
- [可能是全网第一个支持插入Markdown到PPT的PPT插件_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1VXcZe2EyK/?spm_id_from=0.0.homepage.video_card.click)

## 7 🤝参与开发

我是如何开发PPT插件的分享：[如何开发一个PPT插件：使用VSTO开发](https://mp.weixin.qq.com/s/LpOw5tdbHTbC1FsG4seEOw)

## 8 ❓ 常见问题

- **如何把插件的功能添加到PPT的快捷工具栏？**   
  按钮右击，添加到「快速访问工具栏」即可  
  ​<img alt="" src="https://s2.loli.net/2025/08/29/RdxoUpzIcZMTv6u.png" />  
  可以把「快速访问工具栏」放在下方，更方便使用

- **插入latex公式，显示不正常？**

  - 「插入latex文字」按钮由于调用的是PPT自带的功能，比较适合插入简单的数学公式，插入复杂的latex请使用「插入latex svg」
  - PPT特殊latex语法举例见：[#7](https://github.com/Achuan-2/SlideSCI/issues/7)
- **如何及时获取插件版本更新**

  Github有关注功能，关注本项目的repo的release动态，当新版本发布后，Github就会自动发邮件通知
  <img width="662" height="425" alt="PixPin_2025-10-14_20-54-44" src="https://github.com/user-attachments/assets/646144ed-8ed1-47d5-afb3-889dd0c2bfdf" />

  <img width="516" height="392" alt="PixPin_2025-10-14_20-55-32" src="https://github.com/user-attachments/assets/9eefb02d-e57f-4193-a331-18ca696745e9" />



## 9 ❤️ 用爱发电

开源与创作不易，如果喜欢我的作品，欢迎给我赞赏，这会激励我继续维护项目和持续创作新项目。

开源不等于免费，我开源的目的是分享与交流学习，而不是免费给别人打工。开源的代码和插件、软件，首先都是写给自己用，顺道分享出来的，而不是为了给别人用而写。个人时间和精力有限，我不会免费帮忙实现用户提的各种功能请求、免费帮别人答疑解惑，在优先考虑个人需求的前提下，再考虑赞赏用户的使用答疑、功能建议。不考虑非赞赏用户提的需求。

累积赞赏50元的朋友如果想加我微信，可以发邮件到achuan-2@outlook.com来进行好友申请（赞赏达不到50元的，我不会回复邮件和加好友哦，因为不想当免费客服）

<img alt="" src="https://s2.loli.net/2025/08/29/tI4HCGzql17nr2D.png" />

## 10 👨💻 问题反馈

如果使用过程中遇到问题，可以通过以下方式反馈:

1. 在 GitHub 上提 [Issue](https://github.com/Achuan-2/SlideSCI/issues)
2. 发送邮件到: [achuan-2@outlook.com](mailto:achuan-2@outlook.com)

## 11 🔍 参考项目与致谢

- [jph00/latex-ppt](https://github.com/jph00/latex-ppt): LaTeX in PowerPoint 支持
- [Markdig](https://github.com/xoofx/markdig): Markdown 解析支持
- [MathJax](https://github.com/mathjax/MathJax): 数学公式转SVG支持
- 感谢 Visual Studio Tools For Office 提供的开发支持
- 感谢所有提供建议反馈以及捐赠赞赏的用户

## 12 📄 特别说明

- 本仓库发布的`SlideSCI`项目中涉及的任何脚本，仅用于测试和学习研究，禁止用于商业用途，不能保证其合法性，准确性，完整性和有效性，请根据情况自行判断。
- ​`本人` 对任何脚本问题概不负责，包括但不限于由任何脚本错误导致的任何损失或损害.
- 未经授权，请勿将`SlideSCI`项目的任何内容用于商业或非法目的，否则后果自负。
- 以任何方式查看此项目的人或直接或间接使用`SlideSCI`项目的任何脚本的使用者都应仔细阅读此声明。`本人` 保留随时更改或补充此免责声明的权利。一旦使用并复制了任何相关脚本或`SlideSCI`项目，则视为您已接受此免责声明。
- 本项目遵循`AGPL-3.0 License`协议，如果本特别声明与`AGPL-3.0 License`协议有冲突之处，以本特别声明为准。



> 您使用或者复制了本仓库且本人制作的任何代码或项目，则视为`已接受`此声明，请仔细阅读  
> 您在本声明未发出之时点使用或者复制了本仓库且本人制作的任何代码或项目且此时还在使用，则视为`已接受`此声明，请仔细阅读