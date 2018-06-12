<center><h2>sopt 简介</h2></center>
&nbsp;&nbsp;&nbsp;&nbsp;**sopt**是simple optimization的简称，目前我已经将代码托管到pypi,地址是[sopt](https://pypi.org/project/sopt/),可以直接通过pip命令下载安装使用,由于项目只依赖numpy,所以在windows和linux环境下安装都很方便，直接`pip install sopt`即可。项目的github地址是[sopt](https://github.com/Lyrichu/sopt)。目前sopt包含的优化方法如下:
- 遗传算法(Genetic Algorithm,GA)
- 粒子群算法(Particle Swarm Optimization,PSO)
- 模拟退火算法(Simulated Anealing,SA)
- 随机游走算法(Random Walk):
- 梯度下降法(Gradient Descent,GD)
- 动量优化算法(Momentum)
- 自适应梯度算法(AdaGrad)
- RMSProp
- Adam
由于只是一个初步的版本，后续如果有时间的话，会加上更多的优化算法进去。目前所有的优化算法暂时只支持**连续实函数**的优化;除了基于梯度的几个优化算法,GA、PSO、SA以及Random Walk都同时支持**无约束优化**，**线性约束优化**以及**非线性约束优化**。具体我会在下面详细说明。
<center><h2>sopt 使用详解以及实例演示</h2></center>
<h3>1.SGA 使用</h3>
&nbsp;&nbsp;&nbsp;&nbsp;SGA是Simple Genetic Algorithm的简称，是最基本的遗传算法。其编码方式采用**二进制编码**,选择方法采用**轮盘赌法**,交叉方法采用**单点交叉**,变异方式采用**均匀变异**,默认是求函数的**最小值**。下面是sopt中**SGA**的一个简单使用实例:
```python
from sopt.SGA import SGA
from math import sin

def func1(x):
    return (x[0]-1)**2 + (sin(x[1])-0.5)**4 + 2

if __name__ == '__main__':
    sga = SGA.SGA(func = func1,func_type = 'min',variables_num = 2,
        lower_bound = 0,upper_bound = 2,generations = 20,
        binary_code_length = 10)
    # run SGA
    sga.run()
    # show the SGA optimization result in figure
    sga.save_plot()
    # print the result
    sga.show_result()
```
运行结果如下:
```
-------------------- SGA config is: --------------------
lower_bound:[0, 0]
generations:20
cross_rate:0.7
variables_num:2
mutation_rate:0.1
func_type:min
upper_bound:[2, 2]
population_size:100
func:<function func1 at 0x7f3d2311b158>
binary_code_length:10
-------------------- SGA caculation result is: --------------------
global best generation index/total generations:3/20
global best point:[1.00488759 0.45356794]
global best target:2.00003849823336
```
用图像展示为图1所示:
<center><img src="http://t1.aixinxi.net/o_1cfk6qi2e1vok1vpvpf7jir1fvra.png-w.jpg"></center>
<center>图1 SGA 运行结果</center>
上面定义的目标函数为$f(x_1,x_2)=(x_1-1)^2+(sin(x_2)-0.5)^4+2$,其中$0<x_1,x_2<2$,函数的最小值点为(1,0.5236),最小值为2,与寻找到的最小值点(1.00488759 0.45356794)以及最小值2.00003849823336是非常接近的。
SGA类的全部参数定义如下:
- lower_bound:一个int或者float的数字，或者是长度为`variables_num`的ndarray,表示每个变量的下界,如果是一个数字的话，我们认为所有的下界都是一样的(必填);
- upper_bound:一个int或者float的数字，或者是长度为`variables_num`的ndarray,表示每个变量的上界,如果是一个数字的话，我们认为所有的上界都是一样的(必填);
- variables_num:表示目标函数变量的个数(必填);
- func(必填):表示目标函数;
- cross_rate:GA的交叉率,默认为0.7;
- mutation_rate:变异率,默认为0.1;
- population_size:种群数目,默认为100;
- generations:进化代数,默认是200;
- binary_code_length:GA二进制编码的长度,默认是10;
- func_type:函数优化类型,求最小值取值为'min',求最大值取值为'max',默认是'min';
<h3>2. GA 使用</h3>
&nbsp;&nbsp;&nbsp;&nbsp;GA是相比SGA更加一般的遗传算法实现，其对于编码方式，选择方式，交叉方式以变异方式等会有更多的支持。首先还是看一个例子:
```python 
from sopt.GA.GA import GA
from sopt.util.functions import *
from sopt.util.ga_config import *
from sopt.util.constraints import *

class TestGA:
    def __init__(self):
        self.func = quadratic11
        self.func_type = quadratic11_func_type
        self.variables_num = quadratic11_variables_num
        self.lower_bound = quadratic11_lower_bound
        self.upper_bound = quadratic11_upper_bound
        self.cross_rate = 0.8
        self.mutation_rate = 0.05
        self.generations = 200
        self.population_size = 100
        self.binary_code_length = 20
        self.cross_rate_exp = 1
        self.mutation_rate_exp = 1
        self.code_type = code_type.binary
        self.cross_code = False
        self.select_method = select_method.proportion
        self.rank_select_probs = None
        self.tournament_num = 2
        self.cross_method = cross_method.uniform
        self.arithmetic_cross_alpha = 0.1
        self.arithmetic_cross_exp = 1
        self.mutation_method = mutation_method.uniform
        self.none_uniform_mutation_rate = 1
        #self.complex_constraints = [constraints1,constraints2,constraints3]
        self.complex_constraints = None
        self.complex_constraints_method = complex_constraints_method.penalty
        self.complex_constraints_C = 1e6
        self.M = 1e8
        self.GA = GA(**self.__dict__)

    def test(self):
        start_time = time()
        self.GA.run()
        print("GA costs %.4f seconds!" % (time()-start_time))
        self.GA.save_plot()
        self.GA.show_result()



if __name__ == '__main__':
    TestGA().test()
```
上面代码的运行结果为:
```
GA costs 6.8320 seconds!
-------------------- GA config is: --------------------
func:<function quadratic11 at 0x7f998927bd08>
code_type:binary
complex_constraints:None
global_generations_step:200
cross_method:uniform
mutation_method:uniform
cross_rate:0.8
lower_bound:[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
tournament_num:2
variables_num:11
complex_constraints_method:penalty
none_uniform_mutation_rate:1
population_size:100
mutation_rate:0.05
generations:200
arithmetic_cross_alpha:0.1
func_type:min
mutation_rate_exp:1
cross_rate_exp:1
arithmetic_cross_exp:1
M:100000000.0
select_method:proportion
upper_bound:[11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11]
cross_code:False
binary_code_length:20
complex_constraints_C:1000000.0
-------------------- GA caculation result is: --------------------
global best target generation index/total generations:149/200
global best point:[ 1.07431037  2.41401426  2.4807906   4.36291634  4.90653029  6.13753427
  6.58147963  7.7370479   9.42957347 10.46616122 10.87151134]
global best target:2.2685208668743204
```
&nbsp;&nbsp;&nbsp;&nbsp;其中`sopt.util.functions`预定义了一些测试函数,`quadratic11`是一个11元变量的二次函数，函数原型为:$quadratic11(x_1,...,x_{11})=(x_1-1)^2 +(x_2-2)^2 + ... +(x_{11}-11)^2 + 1$,其中$1 \le x_1,...,x_{11} \le 11$,函数的最小值点在(1,2,...,11)处取得，最小值为1。另外还定义了其他几个测试函数为:
- quadratic50:和quadratic11类似,只是变量个数变为50,取值范围变为1-50;
- quadratic100:和quadratic11类似,只是变量个数变为100,取值范围变为1-100;
- quadratic500:和quadratic11类似,只是变量个数变为500,取值范围变为1-500;
- quadratic1000:和quadratic1000类似,只是变量个数变为1000,取值范围变为1-1000;
- Rosenbrock:函数原型为:$Rosenbrock(x_1,x_2)=100(x_1^2-x_2)^2+(1-x_1)^2$,其中$-2.048 \le x_1,x_2 \le 2.048$,这个函数有很多局部极小值点，函数在(-2.048,-2.048)处取得最大值;
- shubert2函数:函数定义为,$shubert2(x_1,x_2)=\prod_{i=1}^{2}(\sum_{j=1}^{5}(jcos((j+1)*x_i+j))$,其中$-10 \le x_1,x_2 \le 10$,函数有760个局部极小值点,全局极小值为-186.731,这里为了将适应度函数变换为正，我们在原函数的基础上加上1000,这样函数的全局极小值点就变为813.269;
- shubert函数，将shubert2函数中的两个变量拓展为n个变量($\prod_{i=1}^{n}$)就可以得到一般的shubert函数了。
&nbsp;&nbsp;&nbsp;&nbsp;对于每一个优化方法，我们都预定了一个形如`xx_config`的模块，其中定义了该优化方法的一些常用默认参数，比如`ga_config`中就定义了一些GA的一些常用优化参数，`ga_config.basic_config`定义了一些基础参数设置，比如`basic_config.generations`是一个默认进化代数,`basic_config.mutation_rate`是默认的变异参数；而`ga_config.cross_method`则预定义了所有支持的交叉方法,比如`cross_method.uniform`表示均匀交叉,`cross_method.one_point`表示单点交叉等;`ga_config.mutation_method`等也是类似的。有了这些预定义变量，可以免去我们手动输入很多参数取值以及传入方法字符串的麻烦(有时候可能会写错)。
&nbsp;&nbsp;&nbsp;&nbsp;观察上面的运行结果，我们发现对于quadratic11函数，最终找到的全局极小值为2.2685208668743204,和真实的全局极小值点1已经比较接近了；运行耗时6秒多，似乎有些长，这是因为我们种群数目设为100，进化代数设为200,比较大，而且又是采用二进制20位编码，再加上python脚本语言的运行效率问题，所以稍慢。为了做一个对比,这里我们将目标函数从quadratic11变为Rosenbrock函数，其他参数设置保持不变，得到的结果如下:
```
GA costs 1.7245 seconds!
-------------------- GA config is: --------------------
population_size:100
lower_bound:[-2.048, -2.048]
mutation_rate_exp:1
select_method:proportion
code_type:binary
global_generations_step:200
generations:200
mutation_method:uniform
complex_constraints_method:penalty
binary_code_length:20
cross_method:uniform
arithmetic_cross_alpha:0.1
func:<function Rosenbrock at 0x7fe5fd538a60>
upper_bound:[2.048, 2.048]
cross_code:False
mutation_rate:0.05
cross_rate_exp:1
complex_constraints_C:1000000.0
cross_rate:0.8
variables_num:2
M:100000000.0
complex_constraints:None
none_uniform_mutation_rate:1
tournament_num:2
arithmetic_cross_exp:1
func_type:max
-------------------- GA caculation result is: --------------------
global best target generation index/total generations:75/200
global best point:[-2.04776953 -2.04537109]
global best target:3901.4655271502425
```
图2是每代最优值的计算结果:
<center><img src="http://t1.aixinxi.net/o_1cfke3r4d1moe11q1sbo12b41ncua.png-w.jpg"></center>
<center>图2 GA Rosenbrock 函数运行200代寻优结果</center>
替换成Rosenbrock函数之后，函数的运行时间从6秒多减少到1秒多(函数变量个数明显减少了),这说明GA的运行时间与目标函数的变量个数是显著相关的。最后找到的全局极大值点为(-2.04776953,-2.04537109)和真实全局极大值点(-2.048,-2.048)已经很接近了。
&nbsp;&nbsp;&nbsp;&nbsp;SGA类是GA类的父类，所以GA类和SGA类有一些公共的属性，比如`generations`,`population_size`,`func_type`等。和SGA一样的参数这里就不再列举了，如下是GA特有的一些参数:
- cross_rate_exp:cross_rate指数递增的参数取值,即变异率按照$r_t=r_0 \beta^t$,这里$r_t$表示t次迭代的交叉率,$r_0$是初始的交叉率，$\beta$就是这里的`cross_rate_exp`,默认取值为1,一般设置为一个比1稍大的数字，比如1.0001等;
- mutation_rate_exp:mutation_rate指数递增参数取值,具体含义和`cross_rate_exp`类似;
- code_type:采用什么样的编码方式,有三种取值:'binary'表示二进制编码(默认),'gray'表示采用格雷编码,'real'表示采用实数编码。其中格雷编码是一种改进的二进制编码，其优点在于进行交叉、变异等操作时，改变了染色体基因的某几个位置，二进制编码对应的实数值可能会发生非常大的变化，而格雷编码一般不会。二进制编码与格雷编码的转换可以参考附录;
- cross_code:这是一个布尔值,表示是否对二进制编码或格雷码采用交叉编码的方式。具体说明参考附录;
- select_method:选择操作方法，有如下的6种取值:
1. 'proportion':表示比例选择(轮盘赌法)
2. 'keep_best':表示保留最优个体
3. 'determinate_sampling':表示确定性采样
4. 'rssr':remainder stochastic sampling with replacement的缩写，表示无放回余数随机选择
5. 'rank':表示排序选择
6. 'stochastic_tournament':表示随机锦标赛法(以上6种选择方式详细说明请参考附录)
- rank_select_probs:仅当select_method取值为'rank'时，该参数才起作用，表示采用排序选择时，适应度从低到高每个个体被选择的概率，默认为None,采用$p_i=\frac{i}{\sum_{j=1}^{n}j},i=1,2,...,n$的方式计算，其中$n$表示种群个数,$p_i$表示第$i$个个体被选择的概率，所有概率之和为1，如果取值不为None,你应该传入一个size为`population_size`的ndarray,所有元素按照递增排序，数组和为1;
- tournament_num:如果select_method取值为'stochastic_tournament',该值表示锦标赛竞争者数量;
- cross_method:交叉方法,有如下4种取值:
1. 'one_point'：表示单点交叉
2. 'two_point':表示双点交叉
3. 'uniform':表示均匀交叉
4. 'arithmetic':表示算术交叉,只有当编码采用实数编码时，才能使用算术交叉
- arithmetic_cross_alpha:算术交叉的系数，算术交叉的公式为:$x_{new1}=\alpha x_1 + (1-\alpha)x_2,x_{new2}=\alpha x_2 + (1-\alpha)x_1$,这里的$\alpha$即表示`arithmetic_cross_alpha`,其默认值为0.1;
- 'arithmetic_cross_exp':算术交叉系数按照指数递增，即$\alpha_t = \alpha r^t$,这里$\alpha_t$表示第t代的算术交叉系数,$\alpha$是初始交叉系数,$r$即是这里的`arithmetic_cross_exp`,默认取值为1，一般取一个比1稍大的数，比如1.0001,t是进化代数;
- 'mutation_method':变异方法，有如下5种取值:
1. simple:即简单变异，随机选择一个染色体上的基因，按照变异概率进行变异
2. uniform:即均匀变异,每个染色体上的每个基因都按照变异概率进行变异
3. boundary:边界变异,每个基因进行变异以后的取值只能去边界值,比如说各以0.5的概率取得其上界和下界，这种变异方式仅适用于实数编码的情况，一般在目标函数的最优点靠近边界时使用
4. none_uniform:非均匀变异，我们不是取均匀分布的随机数去替换原来的基因，而是在原来基因的基础上做一点微小的随机扰动，扰动以后的结果作为新的基因值，具体来说，我们采用的是这样的变异方式:1)if random(0,1) = 0,$x_k^{'} = x_k + \bigtriangleup (t,U_{max}^{k}-x_k)$;2)if random(0,1) = 1,$x_k{'} = x_k - \bigtriangleup (t,x_k -U_{min}{k})$ 其中 $\bigtriangleup(t,y)=y(1-r^{(1-t/T)b})$,$r$是0-1均匀分布的一个随机数，$T$是最大进化代数，$b$是一个系统参数，表示随机扰动对于进化代数的依赖长度，默认值为1
5. gaussian:高斯变异,即使用高斯分布去替代均匀分布来进行变异,由高斯分布的特点可以知道，这种变异方式也是在原个体区域附近的某个局部区域进行重点搜索，这里高斯分布的均值$\mu$定义为$\mu = \frac{U_{min}^{k}+U_{max}^{k}}{2}$,标准差$\sigma=\frac{U_{max}^{k}-U_{min}^{k}}{6}$
- none_uniform_mutation_rate:即均匀分布`none_uniform`中定义的系统参数$b$;
- complex_constraints:复杂约束,默认值为None,即没有复杂约束,只有简单的边界约束,如果有复杂约束，其取值应该是$\[func_1,func_2,...,func_n\]$,
其中$func_i$表示第$i$个复杂约束函数名，比如对于一个复杂约束函数$func_1:x_1^2+x_2^2 < 3$,其复杂约束函数应该这样定义:
```
def func1(x):
	x1 = x[0]
	x2 = x[1]
	return x1**2 + x2**2 - 3
```
- complex_constraints_method:复杂约束求解的方法,默认是`penalty`即惩罚函数法，暂时不支持其他的求解方式;
- complex_constraints_C:采用`penalty`求解复杂约束的系数$C$,比如对于某一个约束$x_1^2+x_2^2 < 3$,GA在求解过程中，违反了该约束,即解满足$x_1^2+x_2^2 \ge 3$,那么我们对目标函数增加一个惩罚项: $C(x_1^2+x_2^2-3)$,$C$一般取一个很大的正数，默认值为$10^6$。
&nbsp;&nbsp;&nbsp;&nbsp;GA类的参数很多可以调节，所以相比之下使用起来更加麻烦，特别是对于复杂函数的寻优，默认参数未必是最好的，可能需要你根据目标函数的特点自行尝试，选择最优的参数组合。
<h3>3. PSO 使用</h3>
&nbsp;&nbsp;&nbsp;&nbsp;PSO算法的原理可以参考我之前的博客[C语言实现粒子群算法(PSO)一](http://www.cnblogs.com/lyrichu/p/6151272.html)和[C语言实现粒子群算法（PSO）二](http://www.cnblogs.com/lyrichu/p/6151293.html),这里不再赘述。还是先看一个实例代码:
```python
from time import time
from sopt.util.functions import *
from sopt.util.pso_config import *
from sopt.PSO.PSO import PSO
from sopt.util.constraints import *

class TestPSO:
    def __init__(self):
        self.func = quadratic11
        self.func_type = quadratic11_func_type
        self.variables_num = quadratic11_variables_num
        self.lower_bound = quadratic11_lower_bound
        self.upper_bound = quadratic11_upper_bound
        self.c1 = basic_config.c1
        self.c2 = basic_config.c2
        self.generations = 200
        self.population_size = 100
        self.vmax = 1
        self.vmin = -1
        self.w = 1
        self.w_start = 0.9
        self.w_end = 0.4
        self.w_method = pso_w_method.linear_decrease
        #self.complex_constraints = [constraints1,constraints2,constraints3]
        self.complex_constraints = None
        self.complex_constraints_method = complex_constraints_method.loop
        self.PSO = PSO(**self.__dict__)

    def test(self):
        start_time = time()
        self.PSO.run()
        print("PSO costs %.4f seconds!" %(time()-start_time))
        self.PSO.save_plot()
        self.PSO.show_result()


if __name__ == '__main__':
    TestPSO().test()
```
运行结果为:
```
PSO costs 1.1731 seconds!
-------------------- PSO config is: --------------------
complex_constraints_method:loop
c1:1.49445
lower_bound:[1 1 1 1 1 1 1 1 1 1 1]
w_end:0.4
w_method:linear_decrease
complex_constraints:None
func:<function quadratic11 at 0x7f1ddb81a510>
upper_bound:[11 11 11 11 11 11 11 11 11 11 11]
generations:200
func_type:min
w:1
c2:1.49445
w_start:0.9
population_size:100
vmin:-1
vmax:1
variables_num:11
-------------------- PSO calculation result is: --------------------
global best generation index/total generations: 198/200
global best point: [ 1.          1.99999999  2.99999999  4.          5.          6.
  7.00000001  7.99999999  9.00000001 10.00000001 11.        ]
global best target: 1.0
```
<center><img src="http://t1.aixinxi.net/o_1cfos44ji1hl41k2316rte9n1crga.png-w.jpg"></center>
<center>图3 PSO 求解 quadratic11 200代运行结果</center>
上面的代码意图应该是非常明显的,目标函数是$quadratic11$,最终求得的最小值点几乎就是全局最小值点。下面是PSO类中所有参数的具体定义:
- variables_num:变量个数(必填);
- lower_bound:一个int或者float的数字，或者是长度为`variables_num`的ndarray,表示每个变量的下界,如果是一个数字的话，我们认为所有的下界都是一样的(必填);
- upper_bound:一个int或者float的数字,或者是长度为`variables_num`的ndarray,表示每个变量的上界,如果是一个数字的话，我们认为所有的上界都是一样的(必填);
- func:目标函数名(必填);
- func_type:函数优化类型,求最小值取值为'min',求最大值取值为'max',默认是'min';
- c1:PSO 参数c1,默认值为1.49445;
- c2:PSO 参数c2,默认值为1.49445;
- generations:进化代数,默认值为100;
- population_size:种群数量,默认为50;
- vmax:粒子最大速度,默认值为1;
- vmin:粒子最小速度,默认值为-1;
- w:粒子惯性权重,默认为1;
- w_start:如果不是使用恒定的惯性权重话,比如使用权重递减策略,w_start表示初始权重;
- w_end:相应的,w_end表示末尾权重;
- w_method:权重递减的方式,有如下5种取值:
1. 'constant':权重恒定
2. 'linear_decrease':线性递减,即$w_t = w_{end} + (w_{start} - w_{end})\frac{(T-t)}{T}$,其中$T$表示最大进化代数,$w_t$表示第$t$代权重
3. 'square1_decrease':第一种平方递减,即:$w_t = w_{start} - (w_{start}-w_{end})(\frac{t}{T})^2$
4. 'square2_decrease':第二种平方递减,即：$w_t = w_{start} - (w_{start}-w_{end})(\frac{2t}{T}-(\frac{t}{T})^2)$
5. 'exp_decrease':指数递减,即:$w_t = w_{end}(\frac{w_{start}}{w_{end}})^{\frac{1}{1+\frac{10t}{T}}}$
- complex_constraints:复杂约束，默认为None,具体含义参考GA类的complex_constraints
- complex_constraints_method:求解复杂约束的方法，默认为'loop',即如果解不满足复杂约束，则再次随机产生解，直到满足约束,暂时不支持其他的求解方式。
<h3>4. SA 使用</h3>
&nbsp;&nbsp;&nbsp;&nbsp;SA,即模拟退火算法，是基于概率的一种寻优方法，具体原理可以参考我的博客[模拟退火算法（SA）求解TSP 问题（C语言实现）](http://www.cnblogs.com/lyrichu/p/6688459.html)。sopt中SA的使用实例如下:
```python
from time import time
from sopt.util.functions import *
from sopt.Optimizers.SA import SA
from sopt.util.sa_config import *
from sopt.util.constraints import *

class TestSA:
    def __init__(self):
        self.func = Rosenbrock
        self.func_type = Rosenbrock_func_type
        self.variables_num = Rosenbrock_variables_num
        self.lower_bound = Rosenbrock_lower_bound
        self.upper_bound = Rosenbrock_upper_bound
        self.T_start = 100
        self.T_end = 1e-6
        self.q = 0.9
        self.L = 100
        self.init_pos = None
        #self.complex_constraints = [constraints1,constraints2,constraints3]
        self.complex_constraints_method = complex_constraints_method.loop
        self.SA = SA(**self.__dict__)


    def test(self):
        start_time = time()
        self.SA.run()
        print("SA costs %.4f seconds!" %(time()-start_time))
        self.SA.save_plot()
        self.SA.show_result()


if __name__ == '__main__':
    TestSA().test()
```
运行结果如下:
```
SA costs 0.2039 seconds!
-------------------- SA config is: --------------------
func_type:max
complex_constraints:None
q:0.9
complex_constraints_method:loop
T_start:100
steps:17500
T_end:1e-06
L:100
func:<function Rosenbrock at 0x7f8261799048>
variables_num:2
lower_bound:[-2.048 -2.048]
init_pos:[-2.03887265 -2.02503927]
upper_bound:[2.048 2.048]
-------------------- SA calculation result is: --------------------
global best generation index/total generations:2126/17500
global best point: [-2.03887265 -2.02503927]
global best target: 3830.997799328349
```
<center><img src="http://t1.aixinxi.net/o_1cfoujjbsodjl0i1kmu8egaeea.png-w.jpg"></center>
<center>图4 SA求解Rosenbrock函数</center>
SA类的具体参数含义如下:
- variables_num:变量个数(必填);
- lower_bound:一个int或者float的数字，或者是长度为`variables_num`的ndarray,表示每个变量的下界,如果是一个数字的话，我们认为所有的下界都是一样的(必填);
- upper_bound:一个int或者float的数字,或者是长度为`variables_num`的ndarray,表示每个变量的上界,如果是一个数字的话，我们认为所有的上界都是一样的(必填);
- func:目标函数名(必填);
- func_type:函数优化类型,求最小值取值为'min',求最大值取值为'max',默认是'min';
- T_start:初始温度,默认值为100;
- T_end:末尾温度,默认值为$10^{-6}$;
- q:退火系数,默认值为0.9,一般取一个在[0.9,1)之间的数字;
- L:每个温度时的迭代次数，即链长,默认值为100;
- init_pos:初始解的取值，默认值为None,此时初始解是随机生成的,或者你也可以指定一个用ndarray表示的初始解;
- complex_constraints:复杂约束，默认为None,表示没有约束，具体定义同GA 的 complex_constraints;
- complex_constraints_method:求解复杂约束的方法，默认为'loop',即如果解不满足复杂约束，则再次随机产生解，直到满足约束,暂时不支持其他的求解方式。
<h3>5. Random Walk 使用</h3>
&nbsp;&nbsp;&nbsp;&nbsp;Random Walk 即随机游走算法，是一种全局随机搜索算法，具体原理可以参考我的博客[介绍一个全局最优化的方法：随机游走算法(Random Walk)](http://www.cnblogs.com/lyrichu/p/7209529.html)。sopt中Random Walk的使用实例如下:
```python 
from time import time
from sopt.util.functions import *
from sopt.util.constraints import *
from sopt.util.random_walk_config import *
from sopt.Optimizers.RandomWalk import RandomWalk

class TestRandomWalk:
    def __init__(self):
        self.func = quadratic50
        self.func_type = quadratic50_func_type
        self.variables_num = quadratic50_variables_num
        self.lower_bound = quadratic50_lower_bound
        self.upper_bound = quadratic50_upper_bound
        self.generations = 100
        self.init_step = 10
        self.eps = 1e-4
        self.vectors_num = 10
        self.init_pos = None
        # self.complex_constraints = [constraints1,constraints2,constraints3]
        self.complex_constraints = None
        self.complex_constraints_method = complex_constraints_method.loop
        self.RandomWalk = RandomWalk(**self.__dict__)

    def test(self):
        start_time = time()
        self.RandomWalk.random_walk()
        print("random walk costs %.4f seconds!" %(time() - start_time))
        self.RandomWalk.save_plot()
        self.RandomWalk.show_result()


if __name__ == '__main__':
    TestRandomWalk().test()


```
运行结果为:
```
Finish 1 random walk!
Finish 2 random walk!
Finish 3 random walk!
Finish 4 random walk!
Finish 5 random walk!
Finish 6 random walk!
Finish 7 random walk!
Finish 8 random walk!
Finish 9 random walk!
Finish 10 random walk!
Finish 11 random walk!
Finish 12 random walk!
Finish 13 random walk!
Finish 14 random walk!
Finish 15 random walk!
Finish 16 random walk!
Finish 17 random walk!
random walk costs 1.0647 seconds!
-------------------- random walk config is: --------------------
init_step:10
eps:0.0001
generations_nums:9042
lower_bound:[1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1]
complex_constraints_method:loop
walk_nums:17
complex_constraints:None
vectors_num:10
upper_bound:[50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50
 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50
 50 50]
variables_num:50
func:<function quadratic50 at 0x7f51555a1840>
generations:100
func_type:min
-------------------- random walk caculation result is: --------------------
global best generation index/total generations:8942/9042
global best point is: [ 1.00004803  1.99999419  3.00000569  3.99998558  5.00002455  5.99999255
  6.99992476  7.99992864  9.00000401  9.99994717 10.99998155 12.00002429
 13.0000035  13.99998567 15.00000421 16.00001454 16.99997252 17.99998041
 19.00002491 20.00003141 21.00004182 21.99998565 22.99997668 23.99999821
 24.99995881 25.99999359 27.00000443 28.00005117 28.99998132 30.00004136
 31.00002021 32.00000616 33.00000678 34.00005423 35.00001799 36.00000051
 37.00002749 38.00000203 39.00007087 39.9999964  41.00004432 42.0000158
 42.99992991 43.99995352 44.99997267 46.00003533 46.9999834  47.99996778
 49.00002904 50.        ]
global best target is: 1.0000000528527013
```
<center><img src="http://t1.aixinxi.net/o_1cfp1962fbhc1ha4191a1ik2ib8a.png-w.jpg"></center>
<center>图5 Random Walk 求解quadratic50</center>
经过实验发现,Random Walk 具有非常强的全局寻优能力，对于quadratic50这种具有50个变量的复杂目标函数，它也可以很快找到其全局最优点，而且运行速度也很快。RandomWalk类的具体参数含义如下:
- variables_num:变量个数(必填);
- lower_bound:一个int或者float的数字，或者是长度为`variables_num`的ndarray,表示每个变量的下界,如果是一个数字的话，我们认为所有的下界都是一样的(必填);
- upper_bound:一个int或者float的数字,或者是长度为`variables_num`的ndarray,表示每个变量的上界,如果是一个数字的话，我们认为所有的上界都是一样的(必填);
- func:目标函数名(必填);
- func_type:函数优化类型,求最小值取值为'min',求最大值取值为'max',默认是'min';
- generations:每个step的最大迭代次数,默认是100;
- init_step:初始步长(step)，默认值为10.0;
- eps:终止迭代的步长,默认为$10^{-4}$;
- vectors_num:使用 improved_random_walk时随机产生向量的个数,默认为10;
- init_pos:初始解的取值，默认值为None,此时初始解是随机生成的,或者你也可以指定一个用ndarray表示的初始解;
- complex_constraints:复杂约束，默认为None,表示没有约束，具体定义同GA 的 complex_constraints;
- complex_constraints_method:求解复杂约束的方法，默认为'loop',即如果解不满足复杂约束，则再次随机产生解，直到满足约束,暂时不支持其他的求解方式。
RandomWalk 除了提供基本的random_walk函数之外，还提供了一个更加强大的improved_random_walk函数，后者的全局寻优能力要更强。
<h3>6. 求解带复杂约束的目标函数</h3>
&nbsp;&nbsp;&nbsp;&nbsp;上面所述的各种优化方法求解的都是变量仅有简单边界约束(形如$a \le x_i \le b$),下面介绍如何使用各种优化方法求解带有复杂约束条件的目标函数。其实，求解方法也非常简单，以GA为例，下面的例子即对Rosenbrock函数求解了带有三个复杂约束条件的最优值:
```python
from time import time
from sopt.GA.GA import GA
from sopt.util.functions import *
from sopt.util.ga_config import *
from sopt.util.constraints import *

class TestGA:
    def __init__(self):
        self.func = Rosenbrock
        self.func_type = Rosenbrock_func_type
        self.variables_num = Rosenbrock_variables_num
        self.lower_bound = Rosenbrock_lower_bound
        self.upper_bound = Rosenbrock_upper_bound
        self.cross_rate = 0.8
        self.mutation_rate = 0.1
        self.generations = 300
        self.population_size = 200
        self.binary_code_length = 20
        self.cross_rate_exp = 1
        self.mutation_rate_exp = 1
        self.code_type = code_type.real 
        self.cross_code = False
        self.select_method = select_method.proportion
        self.rank_select_probs = None
        self.tournament_num = 2
        self.cross_method = cross_method.uniform
        self.arithmetic_cross_alpha = 0.1
        self.arithmetic_cross_exp = 1
        self.mutation_method = mutation_method.uniform
        self.none_uniform_mutation_rate = 1
        self.complex_constraints = [constraints1,constraints2,constraints3]
        self.complex_constraints_method = complex_constraints_method.penalty
        self.complex_constraints_C = 1e8
        self.M = 1e8
        self.GA = GA(**self.__dict__)

    def test(self):
        start_time = time()
        self.GA.run()
        print("GA costs %.4f seconds!" % (time()-start_time))
        self.GA.save_plot()
        self.GA.show_result()



if __name__ == '__main__':
    TestGA().test()
```
运行结果如下:
```
GA costs 1.9957 seconds!
-------------------- GA config is: --------------------
lower_bound:[-2.048, -2.048]
cross_code:False
complex_constraints_method:penalty
mutation_method:uniform
mutation_rate:0.1
mutation_rate_exp:1
cross_rate:0.8
upper_bound:[2.048, 2.048]
arithmetic_cross_exp:1
variables_num:2
generations:300
tournament_num:2
select_method:proportion
func_type:max
complex_constraints_C:100000000.0
cross_method:uniform
complex_constraints:[<function constraints1 at 0x7f5efe2e8d08>, <function constraints2 at 0x7f5efe2e8d90>, <function constraints3 at 0x7f5efe2e8e18>]
func:<function Rosenbrock at 0x7f5efe2e87b8>
none_uniform_mutation_rate:1
cross_rate_exp:1
code_type:real
M:100000000.0
binary_code_length:20
global_generations_step:300
population_size:200
arithmetic_cross_alpha:0.1
-------------------- GA caculation result is: --------------------
global best target generation index/total generations:226/300
global best point:[ 1.7182846  -1.74504313]
global best target:2207.2089435117955
```
<center><img src="http://t1.aixinxi.net/o_1cfp2mucq1usd1mcd1ueu1qte11s7a.png-w.jpg"></center>
<center>图6 GA求解带有三个复杂约束的Rosenbrock函数</center>
上面的constraints1,constraints2,constraints3是三个预定义的约束条件函数，其定义分别为:$constraints1:x_1^2 + x_2^2 - 6 \le 0$;$constraints2:x_1 + x_2 \le 0$;$constraints3:-2-x_1 - x_2 \le 0$,函数原型为:
```python
def constraints1(x):
    x1 = x[0]
    x2 = x[1]
    return x1**2 + x2**2 -3

def constraints2(x):
    x1 = x[0]
    x2 = x[1]
    return x1+x2

def constraints3(x):
    x1 = x[0]
    x2 = x[1]
    return -2 -x1 -x2
```
其实观察可以发现，上面的代码和原始的GA实例代码唯一的区别，就是其增加了`self.complex_constraints = [constraints1,constraints2,constraints3]`这样一句，对于其他的优化方法，其都定义了`complex_constraints`和`complex_constraints_method`这两个属性，只要传入相应的约束条件函数列表以及求解约束条件的方法就可以求解带复杂约束的目标函数了。比如我们再用Random Walk求解和上面一样的带三个约束的Rosenbrock函数，代码及运行结果如下:
```python
from time import time
from sopt.util.functions import *
from sopt.util.constraints import *
from sopt.util.random_walk_config import *
from sopt.Optimizers.RandomWalk import RandomWalk

class TestRandomWalk:
    def __init__(self):
        self.func = Rosenbrock
        self.func_type = Rosenbrock_func_type
        self.variables_num = Rosenbrock_variables_num
        self.lower_bound = Rosenbrock_lower_bound
        self.upper_bound = Rosenbrock_upper_bound
        self.generations = 100
        self.init_step = 10
        self.eps = 1e-4
        self.vectors_num = 10
        self.init_pos = None
        self.complex_constraints = [constraints1,constraints2,constraints3]
        self.complex_constraints_method = complex_constraints_method.loop
        self.RandomWalk = RandomWalk(**self.__dict__)

    def test(self):
        start_time = time()
        self.RandomWalk.random_walk()
        print("random walk costs %.4f seconds!" %(time() - start_time))
        self.RandomWalk.save_plot()
        self.RandomWalk.show_result()


if __name__ == '__main__':
    TestRandomWalk().test()
```
运行结果:
```
Finish 1 random walk!
Finish 2 random walk!
Finish 3 random walk!
Finish 4 random walk!
Finish 5 random walk!
Finish 6 random walk!
Finish 7 random walk!
Finish 8 random walk!
Finish 9 random walk!
Finish 10 random walk!
Finish 11 random walk!
Finish 12 random walk!
Finish 13 random walk!
Finish 14 random walk!
Finish 15 random walk!
Finish 16 random walk!
Finish 17 random walk!
random walk costs 0.1543 seconds!
-------------------- random walk config is: --------------------
eps:0.0001
func_type:max
lower_bound:[-2.048 -2.048]
upper_bound:[2.048 2.048]
init_step:10
vectors_num:10
func:<function Rosenbrock at 0x7f547fc952f0>
variables_num:2
walk_nums:17
complex_constraints_method:loop
generations:100
generations_nums:2191
complex_constraints:[<function constraints1 at 0x7f547fc95bf8>, <function constraints2 at 0x7f547fc95c80>, <function constraints3 at 0x7f547fc95d08>]
-------------------- random walk caculation result is: --------------------
global best generation index/total generations:2091/2191
global best point is: [-2.41416736  0.41430367]
global best target is: 2942.6882849234585
```
<center><img src="http://t1.aixinxi.net/o_1cfp3vhoqisu1ea21lte1mgo7m4a.png-w.jpg"></center>
<center>图7 Random Walk求解带有三个复杂约束的Rosenbrock函数</center>
可以发现Random Walk 求解得到的最优解要比GA好，而且运行时间更快，经过实验发现，在所有的优化方法中，不论是求解带复杂约束还是不带复杂约束条件的目标函数，求解效果大体上排序是:Random Walk > PSO > GA > SA 。所以当你在求解具体问题时，不妨多试几种优化方法，然后择优选择。
<h3>7. 基于梯度的系列优化方法</h3>
&nbsp;&nbsp;&nbsp;&nbsp;上面所述的各种优化方法，比如GA,PSO,SA等都是基于随机搜索的优化算法，其计算是不依赖于目标函数的具体形式，也不需要知道其梯度的，更加传统的优化算法是基于梯度的算法，比如经典的梯度下降(上升)法(Gradient Descent)以及其一系列变种。下面就简要介绍sopt中GD,Momentum,AdaGrad,RMSProp以及Adam的实现。关于这些基于梯度的优化算法的具体原理，可以参考我之前的一篇博文[深度学习中常用的优化方法](http://www.cnblogs.com/lyrichu/p/8940363.html)。另外需要注意，以下所述的基于梯度的各种优化算法，一般都是用在无约束优化问题里面的，如果是有约束的问题，请选择上面其他的优化算法。下面是GradientDescent的一个使用实例:
```python 
from time import time
from sopt.util.gradients_config import *
from sopt.util.functions import *
from sopt.Optimizers.Gradients import GradientDescent


class TestGradientDescent:
    def __init__(self):
        self.func = quadratic50
        self.func_type = quadratic50_func_type
        self.variables_num = quadratic50_variables_num
        self.init_variables = None
        self.lr = 1e-3
        self.epochs = 5000
        self.GradientDescent = GradientDescent(**self.__dict__)

    def test(self):
        start_time = time()
        self.GradientDescent.run()
        print("Gradient Descent costs %.4f seconds!" %(time()-start_time))
        self.GradientDescent.save_plot()
        self.GradientDescent.show_result()



if __name__ == '__main__':
    TestGradientDescent().test()
```
运行结果为:
```
Gradient Descent costs 14.3231 seconds!
-------------------- Gradient Descent config is: --------------------
func_type:min
variables_num:50
func:<function quadratic50 at 0x7f74e737b620>
epochs:5000
lr:0.001
-------------------- Gradient Descent caculation result is: --------------------
global best epoch/total epochs:4999/5000
global best point: [ 0.9999524   1.99991045  2.99984898  3.9998496   4.99977767  5.9997246
  6.99967516  7.99964102  8.99958143  9.99951782 10.99947879 11.99944665
 12.99942492 13.99935192 14.99932708 15.99925856 16.99923686 17.99921689
 18.99911527 19.9991255  20.99908968 21.99899699 22.99899622 23.99887832
 24.99883597 25.99885616 26.99881394 27.99869772 28.99869349 29.9986766
 30.99861142 31.99851987 32.998556   33.99849351 34.99845985 35.99836731
 36.99832444 37.99831792 38.99821067 39.99816567 40.99814951 41.99808199
 42.99808161 43.99806655 44.99801207 45.99794449 46.99788003 47.99785468
 48.99780825 49.99771656]
global best target: 1.0000867498727912
```
<center><img src="http://t1.aixinxi.net/o_1cfp9tnhjvll2tvvtiojn14ima.png-w.jpg"></center>
<center>图7 GradientDescent 求解quadratic50</center>
下面简要说明以下GradientDescent,Momentum等类中的主要参数,像`func`,`variables_num`等含义已经解释很多次了，不再赘述，这里主要介绍各类特有的一些参数。
1. GradientDescent类:
- lr:学习率,默认是$10^{-3}$;
- epochs:迭代次数，默认是100；
2. Momentum类:
- lr:学习率,默认是$10^{-3}$;
- beta:Momentum的$\beta$参数，默认是0.9;
- epochs:迭代次数，默认是100；
3. AdaGrad类:
- lr:学习率,默认是$10^{-3}$;
- eps:极小的一个正数(防止分母为0),默认为$10^{-8}$;
- epochs:迭代次数，默认是100；
4. RMSProp类:
- lr:学习率,默认是$10^{-3}$;
- beta:RMSProp的$\beta$参数，默认是0.9;
- eps:极小的一个正数(防止分母为0),默认为$10^{-8}$;
- epochs:迭代次数，默认是100；
5. Adam类:
- lr:学习率,默认是$10^{-3}$;
- beta1:Adam的$\beta_1$参数，默认是0.5;
- beta2:Adam的$\beta_2$参数，默认是0.9;
- eps:极小的一个正数(防止分母为0),默认为$10^{-8}$;
- epochs:迭代次数，默认是100；
这里额外说一句,Adam原文作者给出的最优超参数为$\beta_1 = 0.9,\beta_2 = 0.999$,但是我在实际调试过程中发现,$\beta_1 = 0.5,\beta_2 = 0.9$才能达到比较好的效果，具体原因目前不是很清楚。如果想要学习各个优化函数类的具体用法，还可以使用`from sopt.test import *`导入预定义的示例测试，来运行观察结果。比如下面的代码就运行了PSO的一个示例测试:
```python 
from sopt.test import test_PSO
test_PSO.TestPSO().test()
```
结果:
```
PSO costs 3.4806 seconds!
-------------------- PSO config is: --------------------
lower_bound:[1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1]
generations:200
vmin:-1
func:<function quadratic50 at 0x7fd3bf37d8c8>
w_method:linear_decrease
func_type:min
population_size:100
w_start:0.9
complex_constraints_method:loop
vmax:1
c2:1.49445
upper_bound:[50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50
 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50]
variables_num:50
c1:1.49445
w:1
complex_constraints:None
w_end:0.4
-------------------- PSO calculation result is: --------------------
global best generation index/total generations: 199/200
global best point: [  1.           1.           2.94484328   4.13875216   5.00293498
   6.13124759   6.99713025   7.92116383   8.87648843  10.02066994
  11.0758768   12.02240279  13.01125368  13.98010373  14.98063168
  15.97776149  17.11878537  18.00246112  18.14780887  20.00637617
  21.00223704  22.00689373  23.14823218  24.0002456   24.98672157
  25.99141686  27.02112321  28.01540506  29.05403155  30.07304888
  31.00414822  32.00982867  32.99444884  33.9114213   34.96631157
  36.22871824  37.0015616   37.98907918  39.01245751  40.1371835
  41.0182043   42.07768102  42.87178292  43.93687997  45.05786395
  46.03778693  47.07913415  50.          48.9964866   50.        ]
global best target: 6.95906097685
```
<h2>附录</h2>
&nbsp;&nbsp;&nbsp;&nbsp;附录会简要说明上文提到的一些概念，待有空更新。。。
<h2>To do ...</h2>
1. GA的并行化计算
2. 小生境GA
3. GA,PSO等求解离散函数最优解
4. GA,PSO等实际应用举例
5. 其他的一些最优化方法,比如牛顿法，拟牛顿法等
6. 其他 
