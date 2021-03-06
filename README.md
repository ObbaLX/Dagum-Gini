# -*- coding: utf-8 -*-
"""
Created on Sat Jan  8 13:19:08 2022

@author: Codeonce
"""

# 项目简介
(1)Dagum（1997）在文章《A new approach to the decomposition of the Gini income inequality ratio》中将传统Gini系数分解为三个组成部分，即组内差距、组间差距以及超变密度
(2)William Griffiths(2008)指出Dagum（1997）提出的分解基尼系数的新方法用于测量组间和组内的不平等贡献，其分解结果与文献中常用的传统分解方法相同。因此，尽管“Dagum Gini”为不平等性的度量提供了新的解释和新的见解，但它并没有为分解的不同组成部分带来新的表达式。因此，在计算组间净贡献时，我所使用的是William Griffiths所说的传统的计算方法，因为在这个公式里没有积分运算，相对来说更容易实现，同时我的计算结果也验证了所谓Dagum分解，其结果和传统分解是一样的。

# 项目输入
数据输入参考传统的使用excel宏计算基尼系数的数据格式，主要分为['index', 'year', 'group', 'value']四列，value即为我们关注的差距，比如收入数据，group是对应的分组.
注意：数据的构建形式要和输入要求严格一致，尤其是列名，同时，数据要求不能有缺失值，根据基尼系数的要求，数据也不能有负值。
示例\

  index	year	group	value\
  0	   2011	      1 	0.7784\
  1	   2011	      1	    0.6208\
  2	   2011	      3	    0.4486\
  3	   2011	      2	    0.4822


# 项目输出
如果完美运行，会输出一个表格，如下\
总体	  1	      2	         3	1-2	1-3     2-3	组内   组间	  超变密度\
2011	0.2344	0.2651	0.1914	0.1628	0.2909	0.2902	0.1828	28.31%	43.63%	28.06%\
2012	0.2348	0.2662	0.1972	0.1569	0.2957	0.2889	0.1833	28.14%	44.35%	27.51%\
2013	0.2396	0.2641	0.207	0.1507	0.3023	0.3038	0.1866	27.37%	48.10%	24.53%\
2014	0.2365	0.2605	0.2048	0.1481	0.2998	0.3000	0.1834	27.33%	49.59%	23.08%\
2015	0.2438	0.2753	0.202	0.1559	0.3144	0.3082	0.1858	27.36%	48.03%	24.61%\
2016	0.255	0.2822	0.1946	0.1868	0.3185	0.3234	0.1961	27.89%	46.29%	25.82%\
2017	0.2739	0.3264	0.1965	0.199	0.3445	0.3506	0.2058	27.92%	44.51%	27.57%\
2018	0.27	0.3257	0.1897	0.1962	0.343	0.3458	0.2005	27.88%	45.81%	26.31%\
2019	0.2712	0.3253	0.1862	0.2031	0.3396	0.3454	0.2044	28.04%	45.15%	26.81%

这是大部分期刊要求的输出内容。
1.总体即总体基尼系数的大小。

2.“1、2、3”是组，即组内基尼系数。

3.“1-2”、“1-3”、“2-3”是组间基尼系数，即组1和2之间，组1和3之间，组2和3之间。

4.“组内、组间、超变密度”则展示了各个部分贡献率的大小。

# 参考文献
