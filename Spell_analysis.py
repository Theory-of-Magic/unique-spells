import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
import pandas as pd
import itertools

data = pd.read_csv("output.csv")

names = data['name']
level = data['level']
school = data['school']
conc = data['concentration']
verbal = data['verbal']
somatic = data['somatic']
material = data['material']
material_consumed = data['material_consumed']
material_cp_cost = data['material_cp_cost']
area_types = data['area_types']
ability_check = data['ability_checks']
saving_throw = data['saving_throws']
cantrip_scale = data['cantrip_scaling']
dtype = data['damage']
dice = data['dice']
sigma = data['sigma']
a = data['a']
range_ = data['range']

cmap = cm.get_cmap("viridis")

n = []
s = []
m = []

for d in dice:
    d = d.split("d")
    #print(dice)
    n.append(int(d[0]))
    d = d[1].split("+")
    s.append(int(d[0]))
    m.append(int(d[1]))

spells = np.vstack([names,level,school,range_,conc,verbal,somatic,material,
                    material_consumed,material_cp_cost,area_types,
                    ability_check,saving_throw,cantrip_scale,
                    n,s,m,dtype,sigma,a]).T

YN_set = ["N","Y"]
school_set = list(set(school))
area_set = list(set(area_types))
ability_set = list(set(ability_check))
saving_throw_set = list(set(saving_throw))
dset = list(set(dtype))
range_set = list(set(range_))
names_coord = []
level_coord = []
school_coord = []
conc_coord = []
verbal_coord = []
somatic_coord = []
material_coord = []
material_consumed_coord = []
material_cp_cost_coord = []
area_types_coord = []
ability_check_coord = []
saving_throw_coord = []
cantrip_scale_coord = []
dtype_coord = []
n_coord = []
s_coord = []
m_coord = []
sigma_coord = []
a_coord = []
range_coord = []

spells_coord = []
for i in np.arange(len(names)):
    names_coord.append(i)
    level_coord.append(int(level[i]))
    school_coord.append(school_set.index(school[i]))
    conc_coord.append(YN_set.index(conc[i]))
    verbal_coord.append(YN_set.index(verbal[i]))
    somatic_coord.append(YN_set.index(somatic[i]))
    material_coord.append(YN_set.index(material[i]))
    material_consumed_coord.append(YN_set.index(material_consumed[i]))
    material_cp_cost_coord.append(int(material_cp_cost[i]))
    area_types_coord.append(area_set.index(area_types[i]))
    ability_check_coord.append(ability_set.index(ability_check[i]))
    saving_throw_coord.append(saving_throw_set.index(saving_throw[i]))
    cantrip_scale_coord.append(int(cantrip_scale[i]))
    dtype_coord.append(dset.index(dtype[i]))
    n_coord.append(int(n[i]))
    s_coord.append(int(s[i]))
    m_coord.append(int(m[i]))
    sigma_coord.append(round(float(sigma[i]),3))
    a_coord.append(round(float(a[i]),3))
    range_coord.append(range_set.index(range_[i]))
    spells_coord.append([names_coord[-1],#0
                         level_coord[-1],#1
                         school_coord[-1],#2
                         range_coord[-1],#3
                       conc_coord[-1],#4
                         verbal_coord[-1],#5
                         somatic_coord[-1],#6
                       material_coord[-1],#7
                         material_consumed_coord[-1],#8
                       material_cp_cost_coord[-1],#9
                         area_types_coord[-1],#10
                       ability_check_coord[-1],#11
                         ability_check_coord[-1],#12
                       saving_throw_coord[-1],#13
                         cantrip_scale_coord[-1],#14
                        dtype_coord[-1],#15
                         n_coord[-1],#16
                        s_coord[-1],#17
                         m_coord[-1],#18
                         sigma_coord[-1],#19
                         a_coord[-1]])#20
                   
                        
                        
spells_coord = np.array(spells_coord)
#print(spells_coord)
#print(spells_coord.shape)
#input()

#~~~~~~~~~~~

#included = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
#~~~~~~~~~~~

def check(included,names_return = False):
    
    Big_zeros = []

    selections = []
    for i in included:
        sel = spells_coord[:,i]
        selections.append(sel)
    selections = np.vstack(selections).T
    spells_copy = spells_coord.copy()

    for j in np.arange(len(selections)):
        sels = selections - selections[j]
        sels = np.sum(np.abs(sels),axis = 1)
        zeros = []
        for i in np.arange(len(sels)):
            
            if sels[i] == 0:
                zeros.append(i)
        Big_zeros.append(zeros)

    fails = 0
    for bz in Big_zeros:
        #print(bz)
        if len(bz) >1:
            zero_names = []
            zero_level = []
            for z in bz:
                if names[z] not in zero_names:
                    
                    zero_names.append(names[z])
                    zero_level.append(level[z])
                    
            if len(zero_names) != 1:
                #print(zero_names)
                for j in np.arange(len(zero_names)):
                    fails+= 1
                    #if names_return == True:
                        #print(zero_names[j])
                    #print(zero_names[j],zero_level[j])
                    #print(zn)
                
                #print("---------------")
    
    if names_return == True:
        return(fails,zero_names)
    else:
        return(fails)



#these are the other fields to be tested
stuff = [4,5,6,7,8,9,10,11,12,13,14,15]
results = []

test = 0
for L in range(0,len(stuff)+1):
    #print(L)
    #R = []
    for subset in itertools.combinations(stuff,L):
        #1 and 3 always seem to matter so best add them as well as n,s,m
        subset = [1,3]+list(subset)+[16,17,18]
        if len(subset) > 1:
            test+=1
            r = check(subset)
            #R.append(r)
            results.append([subset,r])
            if r ==0:
                print("r is 0: ", subset)
            elif r<= 10:
                #print("r is M< 10: ", subset)
                if r == 0:
                    print("r is 0: ", subset)
            if test%100 == 0:
                pass
                #print("------")
                #print("iteration: ",test)
                #print("r: ",r)
                #print("subset: ",subset)
                #plt.plot(np.R,".")
                #plt.yscale('log')
                #plt.savefig("progress2.png")
                
                #plt.clf()"""
"""trials = [[1, 3,6, 10, 15, 19,20]]
for t in trials:
    print(t,check(t,True))
for r in results:
    if r[0] <10:
        print(r)"""
