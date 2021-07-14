## RNN

Han Li

### Example Slot Filling

ticket booking system : 在智能客服、智能订票系统中，往往会需要slot filling技术，它会分析用户说出的语句，将时间、地址等有效的关键词填到对应的槽上，并过滤掉无效的词语

输入 :  I would like to arrive <font color=red>Shanghai </font> on <font color=blue>November $2^{nd}$</font>

这就是一个slot 包含了

Destination : Shanghai

time arrival : November $2^{nd}$

如何解决这个问题？

可以用FeedForward Network

Input: a word. 需要把一个词汇用一个vector表示, 可以使用

1-of-N(one hot)word. 但是有可能有些词汇是未知的，所以要加一个other类

word hashing或者是word vector. word-hashing中不是简单的hash计算value 而是搞了一些单词组合，计算是否在该单词中

Ouput : 判断该word属于每个slot的probability distribution

<center><image src="./image/rnn-word-hashing.jpeg" width="60%"></center>

​	但是这里会有问题：该神经网络会先处理“arrive”和“leave”这两个词汇，然后再处理“Taipei”，这时对NN来说，输入是相同的，它没有办法区分出“Taipei”是出发地还是目的地. 因此我们的网络需要记忆

<center><image src="./image/rnn-intro.jpeg" width="60%"></center>

### Idea of RNN

现在每个输入的x后，hidden layer每次产生的output $a_1$、$a_2$，都会被存到memory里，下一次有input的时候，这些neuron就不仅会考虑新输入的$x_1$、$x_2$，还会考虑存放在memory中的$a_1$、$a_2$



<center><image src="./image/rnn.jpeg" width="60%"></center>

以上图为例 Input Sequence : $\left[\begin{array}{l}{1\\ 1}\end{array} \right] \left[\begin{array}{l}{1\\ 1}\end{array} \right] \left[\begin{array}{l}{2\\ 2}\end{array} \right]$... 

假设所有weight初始都为1 bias为0 ， **store的要初始化，假设现在初始化为0**

第一次输入 $\left[\begin{array}{l}{1\\ 1}\end{array} \right]$ 后，hidden layer的output + store 的输出为 $\left[\begin{array}{l}{2\\ 2}\end{array} \right]$ , 因此最终的output为 $\left[\begin{array}{l}{4\\ 4}\end{array} \right]$, 同时更新store为 $\left[\begin{array}{l}{2\\ 2}\end{array} \right]$

第二次输入 $\left[\begin{array}{l}{1\\ 1}\end{array} \right]$ 后，hidden layer的output  的输出为 $\left[\begin{array}{l}{2\\ 2}\end{array} \right]$, 然后加上store(注意不是一一对应，而是一个store和所有的output的对应) , 变为 $\left[\begin{array}{l}{6\\ 6}\end{array} \right]$ , 因此最终的output为 $\left[\begin{array}{l}{12\\ 12}\end{array} \right]$, 同时更新store为 $\left[\begin{array}{l}{6\\ 6}\end{array} \right]$

因此对应RNN来说，就算相同的输入，输出仍可能不同。



### Slot RNN

每次输入一个input(是一个word而不是一组word) $x^i$， 然后得到hidden layer的output $a^i$, 以及最终的output $y_i$, 下一个输入的 $x^{i+1}$在计算 $a^{i+1}$的时候会用到 $a^i$, 以此类推.

<center><image src="./image/rnnslot.jpeg" width="60%"></center>

此时，即使输入同样是“Taipei”，我们依旧可以根据前文的“leave”或“arrive”来得到不一样的输出。

当然，可以使网络更deep，因为就会有多层的memory。

<center><image src="./image/rnnslot2.jpeg" width="60%"></center>

### Elman/Jordan Network

上面的是Elman Network，是把hidden layer的output存储起来

Jordan NetWord则是把最终的output加进来

据说Jordan的效果比Elman好，实际上Elman是经过了筛选，但是每层筛选的结果无法控制，我们可能真正需要的最终的筛选结果

<center><image src="./image/rnnEleman.jpeg" width="60%"></center>



### Bidirection RNN

刚刚是从句首读到句尾，先读 $x^t$，再读 $x^{t+1}$。这样实际上后面的input对之前的影响就丢失了，因此出现了Bidirectional RNN。

同时正向读如和逆向读入。

<center><image src="./image/rnnBi.jpeg" width="60%"></center>



### LSTM

> 上面的RNN是最基础的版本，我们可以随时写memory，也可以随时读。目前比较常用的是LSTM( Long Short-term Memory)
>
> 为什么-是在short和term之间呢 ? 因为这指的是一个 Long maintain的short-term momory。我们主要记得的还是前一步的，而不是记得很多步之前的（只是有可能会被存储好久）

**Input Gate**

我们不再能随心所欲的**写memory**了，这里有一个input gate，它打开时我们才能来写memory。而input gate打开与否也是network所学习的

**Output Gate**

决定了是否能够**读取**memory。

**Forget Gate**

决定了是否忘记掉memory中的值(重置)



<center><image src="./image/rnn_gate.jpeg" width="60%"></center>

LSTM有4个input和1个output

- 4个input: 想要被存到memory cell里的值+操控input gate的信号+操控output gate的信号+操控forget gate的信号
- 1个output: 想要从memory cell中被读取的值



**Momery Cell**

$z$： 存入cell的signal

$z_i$ : 控制input gate

$z_f$ : 控制forget gate

$z_o$ : 控制output gate

$a$ : 最终的output

假设现在 $z$ 输入，经过activation function f 得到 g(z), 每个 $z_k, k \in \left\{ i, f, o \right\}$ 也经过f得到 $f(z_k)$

> f通常使用sigmoid function, $f \in [0, 1]$， 判断是否打开

然后 $g(z) \times f(z_i)$, 然后把原先存放在cell中的$c$与$f(z_f)$相乘得到$cf(z_f)$，两者相加得到存在memory中的新值$c'=g(z)\cdot f(z_i)+cf(z_f)$ 

> $f(z_i) = 0$ 代表关闭，则z的输入就没有影响, $f(z_f) = 0$ 关闭代表忘记原来的值（有违背直觉）

然后 $c'$ 经过h得到 $h(c')$ 

最终的output $a = h(c')f(z_o)$

<center><image src="./image/rnn_lstm.jpeg" width="60%"></center>



**LSTM Example**

$x_1$、$x_2$、$x_3$是输入序列的三个维度（注意不是x1, x2, x3先后输入），$y$是输出序列

* $x_2 = 1$ , 则$x_1$ 写入memory
* $x_2=-1$，将memory里的值清零
* $x_3=1$，将memory里的值输出

比如这里第二个时刻x2=1, 则将x1写入memory变为3，第六个时刻x3=1，输出memory的值

<center><image src="./image/rnn_lstm_ex1.jpeg" width="60%"></center>



这里input gate当x2为1的时候就是打开的，forget gate基本都是打开的，除非x2是个很大的负值. ouput gate当x3 = 1时打开。

<center><image src="./image/rnn_lstm_ex2.jpeg" width="60%"></center>



**LSTM Structure**

LSTM实际上就是一个Neuron。

现在假设只有2个neuron。Input $x_1$ 和 $x_2$，每个input经过两个weight输入个一个gate.

和传统的不同点在于：**对于每个neuron需要产生4个signal，因此会是原来的四倍参数。**

下图原本每个input只要连1个 +, 现在要4个了. 

<center><image src="./image/rnn_lstm3.jpeg" width="60%"></center>



**LSTM与RNN**

假设我们现在有一整排的LSTM作为neuron，每个LSTM的cell里都存了一个scalar值，把所有的scalar连接起来就组成了一个vector $c^{t-1}$。

在时间点$t$，输入了一个vector $x^t$，它会乘上一个matrix，通过转换得到$z$，而$z$的每个dimension就代表了操控每个LSTM的输入值( 将input转为LSTM个数的dimension)，同理经过不同的转换得到$z^i$、$z^f$和$z^o$，得到操控每个LSTM的门信号. 因此输入 x 会经过4个transform得到4个input。

<center><image src="./image/rnn_lstm4.jpeg" width="60%"></center>

$f(z^f)$与上一个时间点的cell值$c^{t-1}$相乘，并加到经过input gate的输入$g(z)\cdot f(z^i)$上，得到这个时刻cell中的值$c^t$，最终再乘上output gate的信号$f(z^o)$，得到输出$y^t$



<center><image src="./image/rnn_lstm5.jpeg" width="60%"></center>



**Final LSTM**

最终版本的LSTM不止是使用多个neuron LSTM，而是把上一层的 hidden layer output $h^t$ 和 memory $c^t$ 与 $x^{t+1}$ 一起当作 t+1 时刻的 input

<center><image src="./image/rnn_lstm6.jpeg" width="60%"></center>

**Multiple Layer LSTM**

事实上LSTM基本上都会叠多层，如下图所示，左边两个LSTM代表了两层叠加，右边两个则是它们在下一个时间点的状态

GRU只有2个门, simpleRNN就是最初讲的没用 LSTM的。

GRU的效果和LSTM差不多，但是少了1/4的参数

<center><image src="./image/rnn_lstm7.jpeg" width="60%"></center>



## Learning Target

<<<<<<< HEAD
> 如何训练RNN

### Loss Function

仍然是Slot Filling. 给我们一些sentence，然后给sentence一些label. 然后输出 $y_i$ 与映射到slot的reference vector求交叉熵，比如“Taipei”对应到的是“dest”这个slot，则reference vector在“dest”位置上值为1，其余维度值为0.  **Loss Function 就是交叉熵之和**。

 但是在扔 Tapei之前一定要把 arrive先输入，不能把sentence打乱丢入.



<center><image src="./image/rnn_loss.png" width="60%"></center>

### Training

> 使用Gradient Decent

和DNN一样，使用梯度下降，进行 Backpropagation.  但是这里不是简单的BackPropagation，此处提出了一个

**Backpropagation through time(BPTT)**

<center><image src="./image/rnn_bptt.png" width="60%"></center>

不幸的是，RNN的traning是比较难的。一般来说Loss应该是随training下降的，但是对于RNN，有可能loss剧烈抖动，并且会在某个时刻跳到无穷大，导致程序运行失败

<center><image src="./image/rnn_train.png" width="60%"></center>



### Error Surface

RNN的随参数变化的error surface是非常陡峭的. 有的地方很平坦有的地方很陡峭。

假设我们从橙色开始使用gradient decent，一不小心跳到悬崖上面，loss就会巨大。更不妙的是，一下跳到了悬崖上，gradient暴增，同时learning rate很大，结果参数就爆炸了，出现了nan。

怎么办呢？

使用clipping。就是当gradient超过一个阈值时，就设置gradient为阈值。

<center><image src="./image/rnn_error.png" width="60%"></center>

但是为什么RNN会这样？

有人说是因为sigmoid，但是实际上不是的，使用Relu仍然会有这个问题。而且使用Relu会使RNN效果变差。

这里有一个很直观的解释。我们将一个参数进行小小的变化看output的变化就可以预测出gradient大概的大小。



假设network输入是 [1, 0, 0, ...]。 输出是 $w^{999}$, 因为别的input都是0没有用，只有第一个input在一直传播。

- 当$w$从1->1.01，得到的$y^{1000}$就从1变到了20000，这表示$w$的梯度很大，需要调低学习率
- 当$w$从0.99->0.01，则$y^{1000}$几乎没有变化，这表示$w$的梯度很小，需要调高学习率
- 从中可以看出gradient时大时小，error surface很崎岖，尤其是在$w=1$的周围，gradient几乎是突变的，这让我们很难去调整learning rate

<center><image src="./image/rnn_prob.png" width="60%"></center>

实际上本质上是因为 RNN把同样的操作在训练过程中，不同的时间点不断使用。

从memory接到neuron输入的参数$w$，在不同的时间点被反复使用，$w$的变化有时候可能对RNN的输出没有影响，而一旦产生影响，经过长时间的不断累积，该影响就会被放得无限大，因此RNN经常会遇到这两个问题：

- 梯度消失(gradient vanishing)，一直在梯度平缓的地方停滞不前
- 梯度爆炸(gradient explode)，梯度的更新步伐迈得太大导致直接飞出有效区间



### Helpful Techniques

**Long Short-term Memory(LSTM)**

可以解决gradient vanishing，它会把error surface上那些比较平坦的地方拿掉。

为什么要把RNN换成LSTM？A：LSTM可以解决梯度消失的问题

Q：为什么LSTM能够解决梯度消失的问题？

A：RNN和LSTM对memory的处理其实是不一样的：

- 在RNN中，每个新的时间点，memory里的旧值都会被新值所覆盖
- 在LSTM中，每个新的时间点，memory里的值会乘上$f(g_f)$与新值相加

对RNN来说，$w$对memory的影响每次都会被清除，而对LSTM来说，除非forget gate被打开，否则$w$对memory的影响就不会被清除，而是一直累加保留，因此它不会有梯度消失的问题。实际上最早的LSTM并没有forget gate，就是为了解决梯度消失的问题。

但是不能解决gradient explode。因此我们可以放心的把 LR 设置的很小。

<center><image src="./image/rnn_sol.png" width="60%"></center>

GRU只有两个gate，需要的参数量比LSTM少，鲁棒性比LSTM好，不容易过拟合，它的基本精神是旧的不去，新的不来，GRU会把input gate和forget gate连起来，当forget gate把memory里的值清空时，input gate才会打开，再放入新的值。只有清空memory的值才能放入新的值。

此外，还有很多技术可以用来处理梯度消失的问题，比如Clockwise RNN、SCRN等

<center><image src="./image/rnn_sol2.png" width="60%"></center>



### More application

> 在slot filling中，input一个vector输出也是一个vector，但是实际上还有很多别的应用

* Many to one

Input vector sequence, output only one vector.



#### Sentiment Analysis 

判断文章是Postive和Negative. 

<center><image src="./image/rnn_ap1.png" width="60%"></center>



**Key Term Extraction**

把document当作一个sequence, 通过最后一个时刻的输出当作输出

<center><image src="./image/rnn_ap2.png" width="60%"></center>

* Many to Many

Input和output都是sequence，但是output更短



**Speech Recognition**

以语音识别为例，输入是一段声音信号，每隔一小段时间就用1个vector来表示，因此输入为vector sequence，而输出则是character vector

如果依旧使用Slot Filling的方法，只能做到每个vector对应1个输出的character，识别结果就像是下图中的“好好好棒棒棒棒棒”，但这不是我们想要的，可以使用Trimming的技术把重复内容消去，剩下“好棒”

<center><image src="./image/rnn_ap3.png" width="60%"></center>

但“好棒”和“好棒棒”实际上是不一样的，如何区分呢？

需要用到CTC算法，它的基本思想是，输出不只是字符，还要填充NULL，输出的时候去掉NULL就可以得到连词的效果.

<center><image src="./image/rnn_ap4.png" width="60%"></center>

CTC如何训练呢？

训练的时候不要告诉你哪些声音对应的是 “好”，哪些是 null。因此我们罗列所有的可能都放进去训练。

<center><image src="./image/rnn_ap5.png" width="60%"></center>



#### Sq2Sq

Input和Output都是Sq，但是长度并不确定。

**Machine Translation**

假设输入是 machine learning，要翻译成中文。最后一个时刻的memory就包含了所有的information ，接下来让RNN输出，就会得到“机”，把“机”当做input，并读取memory里的值，就会输出“器”，依次类推，这个RNN甚至会一直输出，不知道什么时候会停止

<center><image src="./image/rnn_ap6.png" width="60%"></center>

怎样才能让机器停止输出呢？

可以多加一个叫做“断”的symbol “===”，当输出到这个symbol时，机器就停止输出

<center><image src="./image/rnn_ap7.png" width="60%"></center>

现在google正在试直接将一种语言的语音转成别的语言的文字，不做语音识别的转换。



#### Beyond Sequence

**Syntactic Parsing Tree**

让machine看一个句子，得到一颗结构树

<center><image src="./image/rnn_ap8.png" width="60%"></center>



#### Sq2Sq Auto-Encoder

为了理解一句话的meaning，如果用bag-of-word来表示一篇文章，就很容易丢失词语之间的联系，丢失语序上的信息

比如“白血球消灭了感染病”和“感染病消灭了白血球”，两者bag-of-word是相同的，但语义却是完全相反的

<center><image src="./image/rnn_ap9.png" width="60%"></center>

这里就可以使用Seq2Seq Autoencoder，在考虑了语序的情况下，把文章编码成vector，只需要把RNN当做编码器和解码器即可

我们输入word sequence，通过RNN变成embedded vector，再通过另一个RNN解压回去，如果能够得到一模一样的句子，则压缩后的vector就代表了这篇文章中最重要的信息

<center><image src="./image/rnn_ap10.png" width="60%"></center>

这个结构甚至可以被层次化，我们可以对句子的几个部分分别做vector的转换，最后合并起来得到整个句子的vector

<center><image src="./image/rnn_ap11.png" width="60%"></center>



##### Seq2Seq for Auto-encoder Speech

Seq2Seq autoencoder还可以用在语音处理上，它可以把一段语音信号编码成vector

这种方法可以把声音信号都转化为低维的vecotr，并通过计算相似度来做语音搜索

<center><image src="./image/rnn_ap13.png" width="60%"></center>

怎么训练？

经过一个RNN以后，存在memory中的就是最终的information。

但是只有Encoder没用，还得有一个Decoder。Encoder中的Memory中的当作input给Decoder，希望Decoder的结果能够和Input越像越好。

<center><image src="./image/rnn_ap14.png" width="60%"></center>

**Demo : Chat Bot**

收集很多的对话，把Q当作Encoder的Input，A当作Decoder的Output



### Attention-Based Model

除了RNN之外，Attention-based Model也用到了memory的思想

机器会有自己的记忆池，神经网络通过操控读写头去读或者写指定位置的信息，这个过程跟图灵机很像，因此也被称为neural turing machine

<center><image src="./image/rnn_ap15.png" width="60%"></center>

<center><image src="./image/rnn_ap16.png" width="60%"></center>
=======
>>>>>>> 773f7ea6bfb7f61c0c1a16cf3d1b70d8801540f3
