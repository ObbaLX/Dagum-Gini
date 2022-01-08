# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 21:40:59 2021

@author: Codeonce

desc: Dagum基尼系数分解
"""
import numpy as np
import pandas as pd


class DagumGini:
    """Dagum基尼系数分解"""
    
    def __init__(self, data):
        self.data = data
        self.μ = self.data['value'].mean() # 样本均值
        self.group_num = self.data.groupby('group').count() # 每组的个体数量
        self.λi = self.group_num / self.group_num.sum() # 每组个体数量的比重
        self.λi = self.λi.rename(columns={'value': 'λ'})
        self.μi = self.data.groupby('group').mean() # 每组得分的均值
        self.μi = self.μi.rename(columns={'value': 'μ'})
        self.si = self.data.groupby('group').sum().apply(lambda x: x / self.data['value'].sum()) # 每组得分均值在总得分中的比重
        self.si = self.si.rename(columns={'value': 's'})
        
    def gini_w(self):
        """within Gini index"""
        # 先计算Gii
        self.gini_w = 0
        self.Gini_ij = pd.DataFrame(index=self.group_num.index, columns=self.group_num.index)
        for group in self.group_num.index:
            group_i = self.data.loc[self.data.group == group, 'value']
            total = 0
            for i in group_i:
                diff = abs(np.array([i]) - np.array(group_i)).sum()
                total += diff
                
            gini_ii = total / (2 * pow(self.group_num.loc[group,][0], 2) * self.μi.loc[group,][0])
            self.Gini_ij.loc[group, group] = gini_ii
        # 计算Gw
            self.gini_w += self.λi.loc[group,][0] * self.si.loc[group,][0] * gini_ii
        # 返回组间差距和组间基尼系数    
        return self.gini_w, self.Gini_ij
    
    def gini_gb(self):
        """Gross Contribution betweengroups"""
        self.gini_gb = 0
        for i in self.group_num.index:
            # 删除当前组，剩下的就是其他组
            g_ls = list(self.group_num.index)
            g_ls.remove(i)
            # 当前组
            group_i = self.data.loc[self.data.group == i, 'value']
            # 建立字典，存放其他组
            og_dic = {}
            for k in g_ls:
                other_group = self.data.loc[(self.data.group == k), 'value']
                og_dic[k] = other_group
                
            
            for j in og_dic.keys():
                sum_ = 0
                for value in group_i:
                    diff = abs(np.array([value]) - np.array(og_dic[j])).sum()
                    sum_ += diff
                    
                g_ij = sum_ / ((self.μi.loc[i,][0] + self.μi.loc[j,][0]) * \
                               self.group_num.loc[i,][0] * self.group_num.loc[j,][0])
                self.Gini_ij.loc[i, j] = g_ij
                
                # 累加计算G_gb
                y = (self.λi.loc[j,][0] * self.si.loc[i,][0] + self.λi.loc[i,][0] * \
                     self.si.loc[j,][0]) * g_ij
                self.gini_gb += y / 2
                
        return self.gini_gb, self.Gini_ij

    
    def gini_nb(self):
        """Net Contribution of the Gini between"""
        λ_μ = pd.concat([self.λi, self.μi], axis=1)

        multi_array = np.array([])
        subtrac_array = np.array([])
        for index in range(self.group_num.shape[0] - 1):
            index += 1
            array1 = np.array(λ_μ.loc[index, 'λ']) * np.array(λ_μ.loc[index+1:, 'λ'])
            array2 = abs(np.array(λ_μ.loc[index, 'μ']) - np.array(λ_μ.loc[index+1:, 'μ']))

            multi_array = np.append(multi_array, array1)
            subtrac_array = np.append(subtrac_array, array2)
        #  Net Contribution of the Gini between   
        self.gini_nb = (multi_array * subtrac_array).sum() / self.μ
        
        return self.gini_nb
    
    def gini_t(self):
        """Transvariation between"""
        self.gini_t = self.gini_gb - self.gini_nb
        
        return self.gini_t

def calculate_dg(data):
    """
    以表格形式输出Dagum基尼系数分解结果
    
    Parameters:
    -----------
    data: 分解Dagum 基尼系数的标准数据格式，index，group，value.
    """
    years = data['year'].drop_duplicates()
    gini_table = pd.DataFrame()
    for year in years:
        df = data.loc[data.year == year][['group', 'value']]
        
        dg = DagumGini(df)
        gini_w, Gini_ij = dg.gini_w()
        gini_gb, Gini_ij = dg.gini_gb()
        gini_nb = dg.gini_nb()
        gini_t = dg.gini_t()
        gini_sum = gini_w + gini_gb
        
        gini_table.loc[year, '总体'] = round(gini_sum, 4)
        # 汇报组内基尼系数
        for index in Gini_ij.index:
            gini_table.loc[year, index] = round(Gini_ij.loc[index, index], 4)
        # 汇报组间基尼系数
        for i in Gini_ij.index:
            for j in Gini_ij.index:
                if i != j:
                    gini_table.loc[year, '{}-{}'.format(i, j)] = round(Gini_ij.loc[i, j], 4)
        gini_table.loc[year, '组内'] = '{:.2%}'.format(gini_w / gini_sum)
        gini_table.loc[year, '组间'] = '{:.2%}'.format(gini_nb / gini_sum)
        gini_table.loc[year, '超变密度'] = '{:.2%}'.format(gini_t / gini_sum)
    # 返回汇总表格    
    return gini_table

if __name__ == '__main__':
    data = pd.read_excel('./example.xlsx', index_col=0)
    gini_table = calculate_dg(data)

