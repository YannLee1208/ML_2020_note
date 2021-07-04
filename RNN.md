## RNN

Han Li

### Example Application

* Slot Filling
  * ticket booking system
  * 在智能客服、智能订票系统中，往往会需要slot filling技术，它会分析用户说出的语句，将时间、地址等有效的关键词填到对应的槽上，并过滤掉无效的词语
    * 输入 :  I would like to arrive <font color=red>Shanghai </font> on <font color=blue>November $2^{nd}$</font>
    * 这就是一个slot。包含
      * Destination : Shanghai
      * time arrival : November $2^{nd}$
  * 如何解决？
    1. 可以用FeedForward Network
       * Input: a word
       * 