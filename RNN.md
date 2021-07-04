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

