# 深度符号音乐生成的三类tokenization表示及其可视化综述

## 摘要

符号音乐进入生成模型之前，必须先选定表示。本文不做音乐生成的全面综述，只讨论符号音乐的 tokenization策略：按模型实际接受的对象的颗粒度来分类，并用同一段单音轨谱例做对照可视化，依次改写成三类模型实际接受的对象。第一类的一步仍对应一个音或音的一个属性，音符级事件序列和 ABC 记谱都在这里。第二类把音乐写成时间轴与音高或音轨对齐的二维表，单通道或多通道均可，模型在这张表上或在潜空间里去噪。第三类的一步是可学习的码，覆盖比一个音更长的片段，码本身没有音名。学习任务随对象而变：音符级符号上可以做时序预测或掩码扩散，卷帘上做二维去噪或掩码预测，粗码上再做序列预测。本文开发了可视化工作把这三种对象放到同一段谱上对照。更贴合音乐语义的粗颗粒单位，是第三类已经在做的事，也是后续值得继续追究的方向。

**关键词：** 符号音乐，token 化，钢琴卷帘，扩散模型，残差量化，ABC 记谱

---

## 1. 引言

既有综述已经从不同轴向整理过这一领域。Ji 等把深度音乐生成分成乐谱、演奏和音频三级：乐谱只作为离散记号，只记音高与时值，演奏层再加入力度和微小时间偏移，最终音频层面才反映为连续声波 [Ji et al., 2020]。本文只讨论符号音乐，但第一类内部仍沿用这个差别：MIDI-like 记演奏动作，REMI 与 ABC 更接近乐谱。网格不在类内另定，见第 2 节。Le 等把符号序列分成时间片和事件两类，事件再分成原子 token 与复合 token，并指出把一个音符的多个属性排成前后相继的 token，会在时间顺序之外再引入一层人为顺序 [Le et al., 2025]。Li 等则按存储形式列出事件、钢琴卷帘、序列、文本和图 [Li et al., 2025]。Ma 等在音乐基础模型综述里把分词器分成手工词表和学习式词表，并指出波形、MIDI 与 ABC 都还没有被同一种基础模型充分覆盖 [Ma et al., 2024]。

上述分类按表示本身或按模态来组织。本文延续已有研究，按模型实际接受的对象来组织，把既有tokenization方案分为三类：

- 第一类：音符级事件序列。一步是一个音或音的一个属性，预处理只做量化并排成序列。下一符号预测和掩码扩散都在这一类。
- 第二类：把卷帘当作图像。卷积网络和扩散模型在二维表上，或在其连续潜空间里去噪或修补。
- 第三类：粗颗粒离散码。码字不表示音高，下游预测码本索引，解码器再把码还原为 MIDI。

三类用同一段谱例对照。谱例是两小节的单音轨，4/4，120 BPM，共六个音。图 1、图 3 至图 5 里的音符保持不变，变的是模型接受的对象。图 2 只把 E4 的起音推迟 30 ms。图 1 是这段谱在十六分音符拍网格上的卷帘，作为底本。图 2 放在第 2 节，看同一处 30 ms 的起音偏差量化为节拍网前后的区别。图 3 把同一段写成第一类的事件符号，比较不同打包方式下的序列步数。图 4 把同一段写成第二类的多通道卷帘，时间轴仍用十六分音符。图 5 把同一段写成 GETScore，纵轴是音轨。第三类的码覆盖的片段长于这段谱，图 6 的 token 个数因此取自各原论文，而不是从这段谱编码得到。整首曲子有多长，token 就有多少，不同曲子的总数不能放在一起比。本文仅比这一段谱在不同写法下有多少步。时间格子在第 2 节先定好。第一类音符级事件序列在同一个格子上比较怎么把音排成序列：一个属性占一步，或把几个属性并成一步。

ABC 放在第一类。它的字符是音名、时值或小节线，和 REMI 里的音高、时值、小节线同属音符级事实，载体换成了记谱文本。字节对编码也放在第一类：它叠在已有事件词表上，合并高频片段，按合并表展开仍是这些音符级事件。第三类的一步是量化得到的码，覆盖比一个音更长的片段，码本身没有音名，要按更长的音乐片段重建。

![同一段两小节的钢琴卷帘](figures/fig1_pianoroll.svg)

**图 1.** 对照谱例的钢琴卷帘。4/4，120 BPM，两小节，横轴每格一个十六分音符。C4 与 E4 同时起奏，接着 G4、A4、B4，第 1 小节第 4 拍休止，第 2 小节为 C5。力度未在图中标出。图 3 至图 5 使用同一段音。图 2 只推迟 E4 的起音。

---

## 2. 时间网格与节拍网格

绝对时间网格和节拍网格是两类音乐内容对齐时间轴的方式，本文所列出的三类tokenization策略先选定最小量化单位格子有多宽，是以绝对时间为单位，比如10ms，还是以音乐实际的乐谱组织进行。格子不同，后面的步数和列数不能放在一起比。第三类的码不直接写出毫秒或拍位；量化之前，片段仍落在下面两种格子之一上。

第一种，时间网格按绝对时间划分。MAESTRO 的 MIDI 是比赛中 Disklavier 在演奏时由琴键传感器记下的，约 200 小时。音符时间来自传感器，不是从音频转录识别得来的 [Hawthorne et al., 2019]。Music Transformer 在这套时间上用 10 ms 一档的时间移位，从 10 ms 到 1 s，只在有事件时写入 [Huang et al., 2019]。GiantMIDI-Piano 从录音自动转录出 10855 首古典钢琴，共 1237 小时，约九成是现场演奏。起音、松键、力度和踏板是转录模型的估计，文件里没有拍、拍号和乐谱 [Kong et al., 2022]。ATEPP 同样从录音转录，得到 11742 条演奏、约 1000 小时，约一半曲目另有 MusicXML乐谱。写出的仍是带毫秒时间的 MIDI，音符信息是模型估计的，不是琴键传感器记下的 [Zhang et al., 2022]。

第二种，节拍网格按拍切分。拍一旦对齐，偏离拍点的时值就按量化线和距离和量化策略进入对应的量化网格。POP909 是909 首流行钢琴改编，标了拍、和弦与调式；最小拍网格取每拍四格，或十六分音符，在4/4拍的拍号下两者的量化策略是等价的。POP909内部乐曲BPM按真实乐曲标注。[Wang et al., 2020]。TheoryTab 整理出的 10148 首流行主旋律加和弦，格子也按拍写 [Yeh, n.d.]。PDMX 从 MuseScore 收集 25 万余首公共领域 MusicXML，音符位置是乐谱上的拍位 [Long et al., 2024]。MusicXML 按小节记音名、时值和拍号，模型不直接读这种乐谱。

Lakh MIDI 常被放进大规模符号训练。发布的 v0.1 有 176581 个不重复文件，从公网抓取后按校验和去重。其中 45129 个能对上百万歌曲数据集里的一条录音 [Raffel, 2016]。这些音符来自网上已有的 MIDI。个别文件可能原本就是别人做的记谱，库内的时间仍不统一：有的已经按节拍网格量化，有的保留速度曲线，有的没有速度标注，仍按绝对时间网格对齐。整库不能算成一种网格。



| 语料 | 时间从哪来 | 落到哪种网格 |
|---|---|---|
| MAESTRO | 电钢琴直接记录，约 200 小时 | 时间网格，毫秒 |
| GiantMIDI-Piano | 录音自动转录，10855 首，1237 小时；无拍、无乐谱 | 时间网格 |
| ATEPP | 录音自动转录，11742 条，约 1000 小时；约一半另有乐谱 | 时间网格 |
| Lakh MIDI | 公网 MIDI，176581 个文件 | 不统一，须逐文件判断 |
| POP909 | 流行钢琴改编，标了拍、和弦、调 | 节拍网格 |
| ASAP | 乐谱与演奏对齐，标了拍和下拍 | 两种网格的对应 |
| TheoryTab、PDMX | 记谱 | 节拍网格 |

毫秒要收成拍，先得有拍点。文件里的速度、录音上跟踪到的拍、乐谱上对齐的拍，人工打点测速都可用来确定乐曲的真实bpm，但也都会错，精度不高，文件写了速度BPM参数时，一拍是 $60/\mathrm{BPM}$ 秒。每拍切 $N$ 格，时刻 $t$（秒）在格子上的连续位置是

$$
u = N \cdot \mathrm{BPM} \cdot t / 60.
$$

收成整数格时，$u$ 可以向下取整，也可以四舍五入。BPM 与拍点一致、且 $t=0$ 落在某一拍上时，第 $k$ 拍的绝对时刻是 $t=k\cdot 60/\mathrm{BPM}$。代入得 $u=N\cdot\mathrm{BPM}\cdot(k\cdot 60/\mathrm{BPM})/60=Nk$，是整数，正好在第 $k$ 拍。不在节拍格线上的音，$u$ 不是整数。

4/4 里 $N=4$ 就是十六分音符。MIDI 不直接记秒，式子里的 $t$ 要用文件写下的速度从 tick 计算得出；MIDI文件中途改变了速度，则必须用新速度计算。比如演奏在渐慢、文件却始终只有一个速度标记时，渐慢部分计算得出的 $u$ 会快于实际拍点。

有乐谱时，谱上的音已经按拍记好，把 MusicXML 导出成 MIDI 就已经在节拍网格上。而ASAP 同时给谱和演奏：谱仍按拍记，另外标明谱上的第几拍落在演奏的哪一毫秒。得到的演奏MIDI文件仍然是绝对时间网格 [Foscarin et al., 2020]。

演奏里的毫秒一旦收进拍格子，原来的精确绝对时间信息就消失了，写回演奏还得另给每一格多长。图 2 固定在 120 BPM、每拍四格，一格 125 ms。量化前 E4 比拍点晚 30 ms；30 ms 小于半格 62.5 ms，向下取整和四舍五入都把它收到这一拍上，这 30 ms 就没有了。

![30 毫秒偏差在量化前后](figures/fig2_clocks.svg)

**图 2.** 起音收到十六分音符网格。120 BPM，一格 125 ms，半格 62.5 ms。(a) 量化前，C4 在 0 ms，E4 在 30 ms。(b) 量化后，两音都在 0 ms。虚线为 E4 原来的 30 ms。

---

## 3. 第一类：音符级事件序列

本类的一步是一个音，或音的一个属性：音高、时值、力度、拍位、小节线，也可以是 ABC 里的一个字符。这些符号排成一条全序序列。预处理是读取midi文件进行量化并排列，不另训编码器。一首曲子常常因此拉成数千步。第三类也通常在深度序列学习模型上预测，但一步是没有音名的码。本类与它的分界是：符号展开之后仍能读成音符级事实。

学习任务不由这一类别固定。常见的任务是自回归预测下一个符号，骨干多为 decoder-only Transformer，mamba及其混合架构。同一条序列也可以做掩码扩散：把符号换成[mask]，再预测原来的符号。已经揭开的位置保持不动，每步揭开的位置数量由时间表决定。

网格已在第 2 节选定。MIDI-like 和 TSD 用时间网格，REMI 用节拍网格。

3.1 MIDI-like

Oore 等用按下、松开、力度和时间移位记钢琴 [Oore et al., 2020]，Music Transformer 沿用这套词表 [Huang et al., 2019]。同时响起的音接连写按下，中间不放时间移位。词表记的是一架单轨钢琴。力度和毫秒级时值得以保留。序列的时间参数是绝对时间，不是拍点，节拍也不在词表里。Music Transformer 用相对注意力，在长序列上表示事件之间的相对距离。松开必须和前面某一次按下配对。模型收敛之后，仍可能生成没有配上的松开，或同音高上重复的按下，这样的序列不能稳定地还原成可播放 MIDI。

3.2 REMI 与 TSD

REMI 和 TSD 都用时值换掉松开，模型不用再把一次松开配回前面的按下。差别在时钟。

Pop Music Transformer 的骨干仍是 Transformer-XL，改的是词表 [Huang and Yang, 2020]。REMI 用 Bar 和 Position 换掉时间移位：每小节一条 Bar，小节内 16 个拍位。落在同一拍位的音共用这个 Position，每个音仍各自写音高、力度和时值。音头和音尾都吸附到这 16 个拍位上；时值符号按三十二分音符计数，一个十六分音符写成 duration（2）。节拍进了词表，节奏比 MIDI-like 稳。速度另用 Tempo 符号，可以中途改变。拍位按一小节 16 格来设，不支持中途改变拍号，比如把4/4拍换成3/4拍。REMI一小节的拍位数是固定的。

TSD 同样用音高、力度和时值记一个音，时间仍用时间移位，不写小节和拍位 [Fradet et al., 2023]。节拍要模型从这些移位里自己学。

两种写法都是一个属性一步，序列仍随属性变长。词表还可以按乐种再加事件。MusicMamba 面向中国传统旋律，在 REMI 上加入音符类型和调式事件，合称 REMI-M [Chen et al., 2025]。音符类型区分调式音和过渡音，调式事件标出调式的起点、终点和类型。预测的仍是下一个事件。Mamba 一支处理旋律和调式，Transformer 一支处理全局，再由双向 Mamba 合并。词表粒度不变。

3.3 把多事件按音符打包

前面的一步只写音符的一个属性，一首歌因此有数千步。复合词从 REMI 出发，把共同定义同一个事件的相邻 token 放到同一步 [Hsiao et al., 2021]。音符的一步只填音高、时值和力度；拍位、和弦和速度另成一步。模型每一步都输出同一组字段，用不到的填 `[ignore]`：音符步里的拍位、和弦和速度是空的，节拍步里的音高、时值和力度是空的。图 3 由 MidiTok 画出，这个空字段在它的词表里写作 `Ignore_None`。族标记标明这一步是哪一种。各字段各有一个预测头，同一步并行抽样，一个头看不到另一个头刚抽到的值。序列按事件计数。他们用这套表示生成整首流行钢琴。

Octuple 的一步也有多个字段，但一步只对应一个音符，小节、拍位和速度写在这个音符里，不再单独占一步 [Zeng et al., 2021]。八个字段是拍号、速度、小节、拍位、乐器、音高、时值和力度，拼成一个向量。拍号写在字段里，中途改变可以留下。原论文用它做掩码预训练和理解。长度约为 REMI 的四分之一、复合词的一半。

复合词和 Octuple 把多步收成一步，字节对编码叠在这些事件词表之上，同样把相邻的token组合并成为一个符号，不另起一种表示 [Fradet et al., 2023]。它统计已有 token 流里最常相邻的片段，合并成新符号，序列因此更短。合并表可逆，展开后仍是音高、时值或拍位；不另训编码器，也没有重建损失。新符号可以跨过若干事件，边界由搭配决定，并不对齐小节或乐句。

3.4 ABC 记谱

ABC 把乐谱写成一行文字。图 1 以八分音符为基本时值，写成 `[CE]2 G2 AB z2 | c2 z6 |`：方括号里的音同时发声，字母后的数字是时值的倍数 [Walshaw, 2011]。ChatMusician 不另建音乐词表，把整段字符串交给 LLaMA 的分词器，仍预测下一个 token，所以放在第一类 [Yuan et al., 2024]。

ChatMusician 把 ABC 用于民谣和传统曲，正文写旋律 [Yuan et al., 2024]。和弦名放在引号里、写在音的左侧，只含音名、可选的升降号、和弦类型和可选的低音，不写用哪些音、按什么节奏。播放器按和弦名标注自动生成伴奏，织体由播放器决定，不写在ABC记谱字符串里 [Walshaw, 2011]。

音名、时值和方括号里的同时音展开后仍是音符级事件，留在本类。和弦名展开不成唯一的奏法。力度、毫秒和踏板默认不写。分词按子词切，一个 token 可以落在音符中间。文字和乐谱用同一个模型，因此chatmusic可以方便的使用纯文本大模型进行ABC记谱的符号音乐的生成，并覆盖和弦编配，音乐理解等多种下游任务，然而ABC记谱法仍然存在局限性，对于古典钢琴里的复杂织体、复音和力度， ABC记谱不能无损保留 MIDI 里的这些信息。

和写成 MIDI 的同一批曲子比，ABC 用到的 token 更少。Yuan 等从训练集抽了 1000 首来数：ABC 交给 LLaMA 分词器，平均每首 288.21 个 token、每秒 5.16 个；转成 MIDI 后按 MIDI-like 计数，是每首 728.60 个 token、每秒 12.42 个，取整大约是每首 288 对 729。同一批再按 REMI 计数，是每首 753.41 个 token、每秒 12.84 个 [Yuan et al., 2024]。反复段用 `|:`、`:|` 只写一遍，字符串先短一截；LLaMA 再按子词切，一个 token 里可以有好几个字符，MIDI-like 和 REMI 则一步一个事件。图 3 记的是图 1 这段谱的事件步数，这组数字是 1000 首整曲的平均 token 数。

![音符级符号与序列步数](figures/fig2_layer1_tokens.svg)

**图 3.** 图 1 的 MIDI 由 MidiTok 分词后原样画出，没有改写成音名。Hsiao 等把一步里用不到的字段写作 `[ignore]`，MidiTok 的词表写成 `Ignore_None`，是同一个空位。(a) MIDI-like。(b) REMI。(c) 复合词，一行是一步。(d) Octuple，一行是一个音符。(e) ABC 记谱，不是这套分词器的输出。(f) 各方案的 token 个数，虚线为音符数 6。分词时打开了和弦、速度、乐器和拍号。

3.5 适用模型

表示策略决定token的内容是什么，模型决定怎么读取这个midi文件信息，两者可以进行简略对照。

自回归符号预测沿序列从左到右，每步增加一个符号，上下文是已经写好的前缀。Music Transformer 与 Pop Music Transformer 用 Transformer。MusicMamba 把 Mamba 和 Transformer 分成两支，预测的仍是下一个事件。

掩码扩散一次看整条序列。网络给还盖着掩码的位置预测原来的符号，每步只揭开其中一部分，已经揭开的位置保持不动。SCHmUBERT 用 Transformer 做去噪器。4/4 每小节 16 步，送入掩码扩散的是沿时间排开的一维索引序列。64 小节的旋律是 1024 个整数，没有新事件的步也留在序列里；音高写在整数里。送入模型的是这一条一维序列，没有时间对音高的二维轴 [Plasser et al., 2023]。SMDIM 的符号是 REMI。去噪块称为 MFA：每个块依次是一层 Mamba、一层前馈和一层自注意力，全文堆 8 个块 [Xu et al., 2026]。作者把每个块里只放一层注意力，与 Jamba 一类混合模型里稀疏放置的注意力放在一起讨论。每步揭开多少个位置由扩散的时间表决定，揭开哪几个位置是在仍被遮住的位置里另抽的。两条模型的输出都还是音符级符号，所以留在本类。

| 表示 | 取舍特点 |
|---|---|
| MIDI-like | 保留力度和毫秒；节拍不在词表里，松开可能配不上 |
| REMI、TSD | 都用时值换掉松开；REMI 写小节和拍位，TSD 仍用绝对时间 |
| REMI-M | 在 REMI 上增加调式符号；一步仍是事件 |
| 复合词、Octuple | 一步同时预测多个属性，长度降到事件或音符；同一步里的属性看不到刚采样的值 |
| 字节对编码 | 叠在已有事件词表上，展开后仍是音高、时值或拍位 |
| ABC | 旋律与和弦名在同一字符串里，分词用现成语言模型的词表；伴奏织体由播放器按和弦名生成 |
| 时间片 | 16 步/小节排成一维索引序列，音高写在整数里；掩码扩散，去噪器是 Transformer |
| REMI | 符号仍是 REMI，学习改成掩码扩散；已揭开的符号保持，去噪块是一层 Mamba、一层前馈、一层注意力 |

3.1 的 MIDI-like 和 3.2 的 REMI 原词表记的是钢琴上的音。多件乐器的文件按这套词表来写时，各轨的音按时间接进同一条序列，乐器不写进符号。MidiTok 是实现 MIDI-like、REMI、TSD、复合词和 Octuple 的软件包，需要多轨时可以在词表里加上 Program，乐器仍写在同一条序列里，无需在预处理阶段按音轨拆分分别进行 tokenization [Fradet et al., 2021]。MuMIDI 在一组音前面放一个 Track，内容分类是旋律、鼓、钢琴、弦乐、吉他和贝斯；换轨就再写一个 Track，下游仍预测下一个符号 [Ren et al., 2020]。Octuple 的一步是一个音符，乐器是八个字段之一，0 到 127 是普通乐器，128 是鼓。论文的例子里，钢琴和吉他的音按时间先后排在同一条序列里 [Zeng et al., 2021]。复合词的无条件生成只写钢琴，序列里没有音轨标记。条件生成才加音轨标记，用来分开给定的主旋律谱和要生成的钢琴 [Hsiao et al., 2021]。

---

## 4. 第二类：把卷帘当图像



第二类把音乐写成音高对时间的表。横轴是时间格，纵轴是音高，同一时刻发声的音占同一列，模型不用再把按下和松开配成一对。图 4 的对照谱例是一条钢琴轨；多件乐器可以是每件一张卷帘，也可以把音轨本身当作表的一轴。卷帘本身是一个稀疏的二维图像。

一个通道不够用。若只用一个二值通道标记某个音高是否正在响，同音高上紧挨着的两个音会连成一个长音，音符起点也无法识别到。因此，如果仅用二值图像来表示钢琴卷帘，卷帘至少拆成两个通道：一个标记起音，一个标记持续。Polyffusion 与 Proffusion 用的就是这对二值通道。图 4(a) 把持续和起音画在同一张图里，(b)–(d) 再拆出力度、起音和踏板，用来对照通道可以怎么分。Huang 等在演奏数据上用的三个通道是卷帘帧、起音和踏板，与图 4 的力度通道不是同一套 [Huang et al., 2024]。

早期模型直接在这张表上生成若干小节。MidiNet 用卷积生成旋律，MuseGAN 为贝斯、鼓、吉他、钢琴和弦乐各放一张卷帘，表示多通道生成 [Yang et al., 2017; Dong et al., 2018]。Coconet 在四声部的格子上反复改写被遮住的位置 [Huang et al., 2017]。画布通常只有数小节，由于钢琴卷帘的稀疏特性，表中大多数格子为 0。

4.1 可视化

![二维卷帘的两种画法](figures/fig3_layer2_roll.svg)

**图 4.** 图 1 的二维卷帘，横轴为十六分音符。(a) 绿色为持续，红色为起音。(b) 力度，C4 与 E4 为 80，着色较浅，其余为 90。(c) 起音，只占起点一格。(d) 踏板，这段谱全为 0。

4.2 去噪与读回 MIDI

直接在卷帘上去噪时，正向过程往格子的数值上加高斯噪声，网络预测去噪后的表。第一类里的离散扩散对象不同：那里的序列是 REMI 一类符号，噪声是把符号换成掩码，网络预测词表里的原符号，揭开之后就是离散事件。这里网络输出的仍是每个格子的连续值。

连续值需要按条件激活成音符，才能写入 MIDI。二值通道按阈值截成 0 或 1：起音通道给出起点，持续通道给出这个音占哪些格。两条通道不一致时，事件对不齐。只有持续、没有起音，音符缺少起点，直接丢弃；起音之后持续中间断开，一个音会被切成几段；相邻的同音高音符如果没有起音通道的起音音符，会粘成一个较长音。力度或踏板通道同样要把连续值收成可写入 MIDI 的等级。扩散步骤本身不检查这些关系。

Polyffusion 接受的对象是一张固定长度的卷帘，一次 8 小节 [Min et al., 2023]。按 4/4、每拍四格计算，8 小节为 32 拍，时间轴有 128 格；音高取 MIDI 0 到 127，共 128 格。起音和持续分成两路，形状为 2×128×128。模型对这张图去噪。无条件采样时，卷帘图从纯噪声开始，去噪结束即得到这 8 小节。

长于 8 小节时，仍用同一尺寸的图。已写好的小节留在格子上，尚未写的格子空出，再对整张图去噪，空出的部分写成新的音。论文的例子先给出 4 小节，每次在图中保留 4 小节、补写后面 4 小节，重复五次，得到 24 小节。已有的音保留为格子上的数值。

和弦与织体是去噪之前给定的条件，模型采样时不生成这组向量，也不从这组向量解码出音符。8 小节的和弦由已训练的编码器表示为 512 维向量，织体按每 2 小节 256 维、四段连成 1024 维，作为条件送入网络。去噪得到的是上述卷帘。起音和持续两路再按阈值读成音符，这一段中的音都在这张图上。

Proffusion 沿用这套 2×128×128 的拍网格 [Zhang et al., 2025]。U-Net 的编码和解码加入 Transformer-Mamba 块，和弦同样事先给定：8 小节的和弦进行编码为 512 维向量后送入网络。生成结果仍是这张卷帘，音符从卷帘读出。

画布再拉长、仍留在像素上时，Hawley 把卷帘画成图，用图像扩散在图上修补 [Hawley, 2024]。所用的 Hourglass Diffusion Transformer 是 Crowson 等提出的图像模型 [Crowson et al., 2024]。时间格若细到 10 ms，还要带上踏板，格子数过大。Huang 等先用变分自编码器把约 1.28 秒的三通道卷帘 3×128×128 压成 4×16×16 的连续潜变量，再把八段拼成 10.24 秒，交给 DiT 去噪 [Huang et al., 2024; Peebles and Xie, 2023]。潜变量是实数，没有词表，也没有码本索引，模型接受的仍是第二类里的连续对象。风格只是一个类别嵌入。比 10.24 秒更长时，Huang 等用 DiffCollage 在同一次去噪里取重叠的短窗并合并分数 [Huang et al., 2024]。解码器先回到卷帘，再按阈值激活成音符。像素上的去噪和潜空间里的去噪，输出都是连续值。

GETMusic 的二维表不是时间对音高的卷帘 [Lv et al., 2023]。横轴是时间，一列为一个十六分音符；纵轴是音轨，每条音轨占两行，一行写音高符号，一行写时值符号。同一轨上同时响起的几个音合成一个音高符号，时值取这组里出现最多的一个。格子里是词表中的离散符号。要生成的音轨上，这些符号被换成掩码，离散扩散预测原来的符号；作为条件留下的音轨保持原样。符号可以直接读成音高和时值，不再把连续值截成 0 或 1。图 5 是图 1 这六个音写成的这张表。

![GETScore：音轨对时间](figures/fig5_getscore.svg)

**图 5.** 图 1 的 GETScore，横轴每列一个十六分音符 [Lv et al., 2023]。(a) 钢琴。音高与时值只写在起音列，时值为十六分音符个数；其余列为 padding。C4 与 E4 合成一个音高符号。(b) 未使用音轨，两行皆为 [EMPTY]。图中写音名是为了可读；模型存储的是该音轨词表中的编号。力度不进入该表。

按本文的分法，它仍属第二类：模型接受的是一张按时间和音轨对齐的表。它和第三类有联系的一点是，格子里已经是离散符号，扩散预测的是词表里的原符号，而不是格子上的连续数值。它不进入第三类。第三类的码没有音名，覆盖比一个音更长的片段，要等解码器重建。这里的符号带有音高和时值，对应一种乐器的一个时间格上的音。

卷帘的每一列都占位。列的步长用第 2 节的两种网格：POP909 上按已标出的拍取格，Polyffusion 用每拍四格 [Min et al., 2023]；MAESTRO 上，Huang 等用 10 ms 一列，Zhang 等用 32 Hz，大约 31 ms 一列 [Huang et al., 2024; Zhang et al., 2024]。第 3.1 节的时间移位只写在有事件的地方。

本类的多轨不写进一条事件序列，也不在 token 字段里标乐器。MuseGAN 把贝斯、鼓、吉他、钢琴和弦乐各写成一张卷帘，五张卷帘沿时间对齐，模型一次看这五路 [Dong et al., 2018]。GETMusic 的音轨是同一张表的纵轴 [Lv et al., 2023]。本节前面的起音通道和持续通道是同一张卷帘里的两路数值，用来分开音头和持续，不是两件乐器。



---

## 5. 第三类：在粗码上再做序列学习

前两类的一步仍能对上音或格子。第一类因此把一首曲子拉成很长的事件序列，第二类则在一张大部分为 0 的表上去噪。第三类先把一段比一个音更长的音乐压成离散码。下游仍是序列模型，预测的是码本索引。索引没有音名，音要等解码器重建，量化会损失细节。

5.1 可视化

![三种粗颗粒离散码](figures/fig4_layer3_codes.svg)

**图 6.** 第三类编号序列的长度，个数取自原文，不是图 1 的编码结果。(a) 8 小节卷帘收成 32 个编号。(b) 一条钢琴卷帘收成 1408 个编号，与 (a) 共用横轴。(c) 一小节收成 8 或 16 个残差编号，单位是小节，不与 (a)(b) 比长短。

5.2 码覆盖的片段

三篇都让下游去预测离散索引，差别在量化之前的片段有多长，以及索引能不能对齐小节。

Han 等把 8 小节的低音和鼓卷帘量化成 32 个编号，码本有 512 项，用 LSTM 预测这些编号，再解码回卷帘 [Han et al., 2022]。这是较早把生成放到离散码上的做法。32 个索引摊在整段循环上，一个码覆盖的是循环里的若干时间步，单位还不是小节，索引也不能读成音名。

Zhang 等把长钢琴卷帘量化成 1408 个索引，用离散扩散 Transformer 在索引上生成，并用作曲家风格做条件 [Zhang et al., 2024]。这里的离散扩散和第一类的 SMDIM 一样，对象都是一条符号序列。Zhang 等的正向过程除了换成掩码，还可以把符号随机换成码本里的其他项。网络预测的是原来的码本索引。索引不能展开成音高或时值，要等量化解码器回到卷帘。一条卷帘仍是 1408 个索引。

MuseTok 从一小节的 REMI+ 出发，用残差向量量化（RQ-VAE）把该小节写成一叠编号 [Huang et al., 2025]。它仍是向量量化变分自编码器的做法：编码器、多层码本和解码器一起训练。小模型每小节 8 个编号，每个码本 1024 项；大模型每小节 16 个编号，每个码本 2048 项。靠前的码本更接近节奏和旋律轮廓，靠后的码本更随绝对音高变化。Transformer 预测这些编号，解码器再还原成这一小节的 REMI+ 事件。一小节对应这一叠编号，解不出 8 小节卷帘。同一套编号既用于续写，也用于和弦、织体和情绪一类的理解。重建有损失。

5.3 适用模型

| 方案 | 量化前 | 下游模型 | 取舍 |
|---|---|---|---|
| Han 等，VQ-VAE | 8 小节低音和鼓的卷帘 | LSTM 预测 32 个编号 | 编号来自 512 项码本；不能读成音名 |
| Zhang 等，VQ-VAE | 32 Hz 钢琴卷帘 | 离散扩散 Transformer | 一条卷帘编成 1408 个编号，码本 128 项 |
| MuseTok，RQ-VAE | 一小节的 REMI+ | Transformer 预测一叠残差编号 | 小模型每小节 8 个编号，大模型 16 个；解码回事件 |

Han 等和 Zhang 等没有给表示起名，量化用的都是向量量化变分自编码器，上表写作 VQ-VAE。编码器把一段音乐映成向量，每个向量换成训练得到的码本里最近的一项，下游预测的是这项的编号。编号没有音名，要等解码器重建，重建有损失 [Han et al., 2022; Zhang et al., 2024]。

字节对编码的做法见第 3.3 节，展开后仍是音符级事件，留在第一类。连续的小节潜变量，例如 MuseMorphose，学到的也比一个音符粗，但下游预测的不是离散码 [Wu and Yang, 2023]，因此不进入这一类。Huang 等（2024）的变分自编码器同样输出连续潜变量，去噪之后回到卷帘，归在第二类。

本类的码索引上没有乐器字段，也不按音轨拆成多条序列。Han 等在量化前把低音和鼓放进同一张卷帘：低音取 C1 到 B4，同时有多个音时只留最低的；鼓收成九件，与低音拼成 57 行，8 小节、每小节 16 格 [Han et al., 2022]。Zhang 等量化的是 MAESTRO 钢琴，材料里只有一件乐器 [Zhang et al., 2024]。MuseTok 把一小节事件收成一叠码，码号本身不标乐器 [Huang et al., 2025]。

---

## 6. 结语

图 1、图 3 至图 5 把前两类放在同一段六个音上：音不变，变的是模型接受的对象。图 2 只推迟 E4 的起音。网格在第 2 节先选定。第三类在图 6，码数来自原文。事件序列长，就先把节拍写进词表，再把多个属性收成一步。常见的学习范式是下一符号预测；同一条序列上也可以做掩码扩散。符号级事件这一类的tokenization预处理的工作仍只是文件读取与量化，不另训编码器。在一条序列里同时处理音符级事件和节奏级事件表示较为复杂，第二类把二维钢琴卷帘交给卷积和扩散，换来空间上的复调，代价是卷帘稀疏，长于固定画布时要分段再生成。再进一步，量化单位放在小节或多音符事件上，模型通过训练编码器再量化成更粗颗粒度的离散码，让序列模型去预测码，重建时有信息丢失。近来的模型多半仍接受这两类对象：MusicMamba 与 SMDIM 仍采用 REMI 或其扩展，Proffusion 吃卷帘；改变主要落在骨干上，Mamba 与注意力混合，事件序列上也可以做离散扩散 [Chen et al., 2025; Xu et al., 2026; Zhang et al., 2025]。


更贴合音乐语义层级的通用表示仍在开发。波形、MIDI 与 ABC 还没有被同一种基础模型充分覆盖 [Ma et al., 2024]。往后更需要关注第三类里可自适应学习的离散码：码本在数据上训练，量化单位贴近拍、小节或更长的按音乐语义内容划分的片段，同一套编号既能生成，也能用于和弦与织体一类的音乐内容理解。MuseTok 已按小节把 REMI+ 收成残差编号，是这一方向上的一例 [Huang et al., 2025]。

---

## 参考文献

- Chen, J., Tang, X., Xie, T., Wang, J., Dong, W., Shi, B. (2025). MusicMamba: A Dual-Feature Modeling Approach for Generating Chinese Traditional Music with Modal Precision. *ICASSP*. arXiv:2409.02421.
- Crowson, K., Baumann, S. A., Birch, A., Abraham, T. M., Kaplan, D. Z., Shippole, E. (2024). Scalable High-Resolution Pixel-Space Image Synthesis with Hourglass Diffusion Transformers. arXiv:2401.11605.
- Dong, H.-W., Hsiao, W.-Y., Yang, L.-C., Yang, Y.-H. (2018). MuseGAN: Multi-track Sequential Generative Adversarial Networks for Symbolic Music Generation and Accompaniment. *AAAI*. arXiv:1709.06298.
- Fradet, N., Briot, J.-P., Chhel, F., El Fallah Seghrouchni, A., Gutowski, N. (2021). MidiTok: A Python Package for MIDI File Tokenization. *ISMIR Late-Breaking Demo*.
- Fradet, N., Gutowski, N., Chhel, F., Briot, J.-P. (2023). Byte Pair Encoding for Symbolic Music. *EMNLP*, pp. 2001–2020. arXiv:2301.11975.
- Foscarin, F., McLeod, A., Rigaux, P., Jacquemard, F., Sakai, M. (2020). ASAP: A Dataset of Aligned Scores and Performances for Piano Transcription. *ISMIR*, pp. 534–541.
- Han, S., Ihm, H., Lee, M., Lim, W. (2022). Symbolic Music Loop Generation with Neural Discrete Representations. *ISMIR*, pp. 403–410. arXiv:2208.05605.
- Hawley, S. H. (2024). Pictures Of MIDI: Controlled Music Generation via Graphical Prompts for Image-Based Diffusion Inpainting. arXiv:2407.01499.
- Hawthorne, C., Stasyuk, A., Roberts, A., Simon, I., Huang, C.-Z. A., Dieleman, S., Elsen, E., Engel, J., Eck, D. (2019). Enabling Factorized Piano Music Modeling and Generation with the MAESTRO Dataset. *ICLR*. arXiv:1810.12247.
- Hsiao, W.-Y., Liu, J.-Y., Yeh, Y.-C., Yang, Y.-H. (2021). Compound Word Transformer: Learning to Compose Full-Song Music over Dynamic Directed Hypergraphs. *AAAI*. arXiv:2101.02402.
- Huang, C.-Z. A., Cooijmans, T., Roberts, A., Courville, A., Eck, D. (2017). Counterpoint by Convolution. *ISMIR*. arXiv:1903.07227.
- Huang, C.-Z. A., Vaswani, A., Uszkoreit, J., Simon, I., Hawthorne, C., Shazeer, N., Dai, A. M., Hoffman, M. D., Dinculescu, M., Eck, D. (2019). Music Transformer. *ICLR*. arXiv:1809.04281.
- Huang, J., Novack, Z., Long, P., Hou, Y., Chen, K., Berg-Kirkpatrick, T., McAuley, J. (2025). MuseTok: Symbolic Music Tokenization for Generation and Semantic Understanding. arXiv:2510.16273.
- Huang, Y., Ghatare, A., Liu, Y., Hu, Z., Zhang, Q., Sastry, C. S., Gururani, S., Oore, S., Yue, Y. (2024). Symbolic Music Generation with Non-Differentiable Rule Guided Diffusion. *ICML*. arXiv:2402.14285.
- Huang, Y.-S., Yang, Y.-H. (2020). Pop Music Transformer: Beat-Based Modeling and Generation of Expressive Pop Piano Compositions. *ACM MM*. arXiv:2002.00212.
- Ji, S., Luo, J., Yang, X. (2020). A Comprehensive Survey on Deep Music Generation: Multi-level Representations, Algorithms, Evaluations, and Future Directions. arXiv:2011.06801.
- Kong, Q., Li, B., Chen, J., Wang, Y. (2022). GiantMIDI-Piano: A Large-Scale MIDI Dataset for Classical Piano Music. *Transactions of the International Society for Music Information Retrieval*, 5(1), 87–98. https://doi.org/10.5334/tismir.80
- Le, D.-V.-T., Bigo, L., Keller, M., Herremans, D. (2025). Natural Language Processing Methods for Symbolic Music Generation and Information Retrieval: A Survey. *ACM Computing Surveys*. arXiv:2402.17467.
- Li, S., Ji, S., Wang, Z., Wu, S., Yu, J., Zhang, K. (2025). A Survey on Music Generation from Single-Modal, Cross-Modal and Multi-Modal Perspectives.
- Long, P., Novack, Z., Berg-Kirkpatrick, T., McAuley, J. (2024). PDMX: A Large-Scale Public Domain MusicXML Dataset for Symbolic Music Processing. arXiv:2409.10831.
- Lv, A., Tan, X., Lu, P., Ye, W., Zhang, S., Bian, J., Yan, R. (2023). GETMusic: Generating Music Tracks with a Unified Representation and Diffusion Framework. arXiv:2305.10841.
- Ma, Y., Øland, A., Ragni, A., et al. (2024). Foundation Models for Music: A Survey. arXiv:2408.14340.
- Min, L., Jiang, J., Xia, G., Zhao, J. (2023). Polyffusion: A Diffusion Model for Polyphonic Score Generation with Internal and External Controls. *ISMIR*. arXiv:2307.10304.
- Oore, S., Simon, I., Dieleman, S., Eck, D., Simonyan, K. (2020). This Time with Feeling: Learning Expressive Musical Performance. *Neural Computing and Applications*. arXiv:1808.03715.
- Peebles, W., Xie, S. (2023). Scalable Diffusion Models with Transformers. *ICCV*.
- Plasser, M., Peter, S., Widmer, G. (2023). Discrete Diffusion Probabilistic Models for Symbolic Music Generation. arXiv:2305.09489.
- Raffel, C. (2016). Learning-Based Methods for Comparing Sequences, with Applications to Audio-to-MIDI Alignment and Matching. PhD thesis, Columbia University.
- Ren, Y., He, J., Tan, X., Qin, T., Zhao, Z., Liu, T.-Y. (2020). PopMAG: Pop Music Accompaniment Generation. *ACM MM*, pp. 1198–1206.
- Walshaw, C. (2011). The abc Music Standard 2.1. https://abcnotation.com/wiki/abc:standard:v2.1
- Wang, Z., Chen, K., Jiang, J., Zhang, Y., Xu, M., Dai, S., Xia, G. (2020). POP909: A Pop-song Dataset for Music Arrangement Generation. *ISMIR*. arXiv:2008.07142.
- Wu, S.-L., Yang, Y.-H. (2023). MuseMorphose: Full-Song and Fine-Grained Piano Music Style Transfer with One Transformer VAE. *IEEE/ACM Transactions on Audio, Speech, and Language Processing*.
- Xu, J., Tang, X., Yang, H., Zhang, H., Yuan, S., Chen, J., Xie, T., Wang, J., Yu, J., Xiang, G. (2026). Efficient Long-Sequence Diffusion Modeling for Symbolic Music Generation. arXiv:2603.00576.
- Yang, L.-C., Chou, S.-Y., Yang, Y.-H. (2017). MidiNet: A Convolutional Generative Adversarial Network for Symbolic-Domain Music Generation. *ISMIR*.
- Yeh, Y.-C. (n.d.). Lead Sheet Dataset. https://github.com/wayne391/lead-sheet-dataset
- Yuan, R., Lin, H., Wang, Y., Tian, Z., Wu, S., et al. (2024). ChatMusician: Understanding and Generating Music Intrinsically with LLM. *Findings of ACL*. arXiv:2402.16153.
- Zeng, M., Tan, X., Wang, R., Ju, Z., Qin, T., Liu, T.-Y. (2021). MusicBERT: Symbolic Music Understanding with Large-Scale Pre-Training. *Findings of ACL-IJCNLP*. arXiv:2106.05630.
- Zhang, H., Tang, J., Rafee, S. R. M., Dixon, S., Fazekas, G. (2022). ATEPP: A Dataset of Automatically Transcribed Expressive Piano Performance. *ISMIR*, pp. 446–453.
- Zhang, J., Fazekas, G., Saitis, C. (2024). Composer Style-specific Symbolic Music Generation Using Vector Quantized Discrete Diffusion Models. *IEEE MLSP*. arXiv:2310.14044.
- Zhang, J., Fazekas, G., Saitis, C. (2025). Mamba-Diffusion Model with Learnable Wavelet for Controllable Symbolic Music Generation. arXiv:2505.03314.
