## Data

> pytorch中加载数据的顺序是：
> ①创建一个dataset对象
> ②创建一个dataloader对象
> ③循环dataloader对象，将data,label拿到模型中去训练



### Dataset类

> 在 `torch.utis.data` 下， 用来构建自己的DataSet类，以DataSet作为父类，并且重写 `__init__`,  `__getitem__`,  `__len__` 方法

`__init__`

初始化信息， 传入参数，构建数据集

可以加入是否transform / 具体的transform对象

`__getitem__(self, index)` 

传入如何获取数据

`__len__`

返回数据长度



### Transform

> `torchvision.transforms`  下，用来 data augmentation

`transfroms.Compose([])` : 传入transform列表

最后一个有`ToTenser()`

**针对图片常用的**

`ToPILImage()` , `RandomHorizontalFlip()`, `RandomRotation(angle)`

**注意**

Data Augumentation 只针对 train set做，validation和test都不做。

但是validation的 `ToPILImage` 和  `ToTenser` 也是要用的



### DataLoader

> 和DataSet对象配合使用获取数据

`dataset `  :  传入dataset对象

`shuffle` : 是否打乱数据

一般train打乱，避免数据的顺序影响训练效果。validation不打乱，因为是测试，没必要打乱

`batch_size` : batch size

`collate_fn` : 传入函数句柄，操作每个batch。





## Model

> pytorch中对于一般的序列模型，直接使用torch.nn.Sequential类及可以实现，这点类似于keras，但是更多的时候面对复杂的模型，比如：多输入多输出、多分支模型、跨层连接模型、带有自定义层的模型等，就需要自己来定义一个模型了。



### 自定义Module

自定义module需要继承 `nn.Module` 类

**继承实现** 

`__init__`  ： **把网络中具有可学习参数的层（如全连接层、卷积层等）放在构造函数__init__()中**

- 不具有参数的层（如ReLU、dropout、BatchNormanation层）可以在 `__init__` 中，如果不具有可学习参数的层不放在构造函数`__init__`里面，则在forward方法里面可以使用 nn.functional.relu 来代替
- `super(MyNet, self).__init__()  `  继承父类的构造函数

`forward(self, x)` ： 提供如何将输入的x forward计算

**通过nn.Sequential来包装层**

```python
self.conv_block = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2))
```

在每一个包装的块里面，各个层是没有名称的，默认按照0、1、2、3、4来排序

**通过OrderedDict中的元组方式打包层，同时可以给各层命名**

```python
from collections import OrderedDict
self.conv_block = nn.Sequential(
            OrderedDict(
                [
                    ("conv1", nn.Conv2d(3, 32, 3, 1, 1)),
                    ("relu1", nn.ReLU()),
                    ("pool", nn.MaxPool2d(2))
                ]
            ))
```

**通过add_module添加层，还可以给不同的层命名**

```python
self.conv_block=torch.nn.Sequential()
self.conv_block.add_module("conv1",torch.nn.Conv2d(3, 32, 3, 1, 1))
self.conv_block.add_module("relu1",torch.nn.ReLU())
self.conv_block.add_module("pool1",torch.nn.MaxPool2d(2))
```



### Module和Sequence的区别

> Sequential类虽然继承自Module类，二者有相似部分，但是也有很多不同的部分，集中体现在：
>
> - Sequenrial类实现了整数索引，故而可以使用model[index] 这样的方式获取一个层
> - Module类没有实现整数索引，不能够通过整数索引来获得层，那该怎么办呢？它提供了几个主要的方法，如下

```python
def children(self):
 
def named_children(self):
 
def modules(self):
 
def named_modules(self, memo=None, prefix=''):
 
'''
注意：这几个方法返回的都是一个Iterator迭代器，所以可以通过for循环访问，当然也可以通过next
'''
```

**model.chilren() 与 model.named_children**

（1）model.children()和model.named_children()方法返回的是迭代器iterator；

（2）model.children():每一次迭代返回的每一个元素实际上是 Sequential 类型,而Sequential类型又可以使用下标index索引来获取每一个Sequenrial 里面的具体层，比如conv层、dense层等。

（3）model.named_children():每一次迭代返回的每一个元素实际上是 一个元组类型，元组的第一个元素是名称，第二个元素就是对应的层或者是Sequential。


**model.modules()与model.named_modules()**

（1）model.modules()和model.named_modules()方法返回的是迭代器iterator；

（2）model的modules()方法和named_modules()方法都会将整个模型的所有构成（包括包装层、单独的层、自定义层等）由浅入深依次遍历出来,但是：

modules()返回的每一个元素是直接返回的层对象本身
named_modules()返回的每一个元素是一个元组，第一个元素是名称，第二个元素才是层对象本身

### 常用Module

**CNN**

`torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)`

`torch.nn.MaxPool2d(kernel_size, stride, padding)`

