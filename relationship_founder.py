import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import cm
from scipy.optimize import curve_fit

data = pd.read_csv("output.csv")
#print(data.keys())
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


def straight_line(x,m,c):
    return(m*x+c)
def expo(x,A,B,C):
    return(C+A*np.exp(B*x))

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
#print(range_set)
#input()

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

selection = []
for sc in spells_coord:
    ss = 0
    if sc[1] == 0:
        if sc[14] == 1:
            ss = 1
        elif sc[14] == 5:
            ss = 3
        elif sc[14] == 11:
            ss = 6
        elif sc[14] == 17:
            ss = 9
    else:
        if sc[14] == 0:
            ss = sc[1]
        else:
            ss = sc[14]
    selection.append(np.concatenate((sc,[ss])))
    
    #if sc[14] == 0:
        #selection.append(sc)

print(dset)
#print(len(dset))
selection = np.array(selection)
damage1 = (0.5*(selection[:,17]+1)*selection[:,16]+selection[:,18])
#plt.xlabel("Spell Slot")
#plt.ylabel("Average Damage")
#plt.plot(selection[:,-1],damage1,".")
#plt.show()

d0m=[]
d0c=[]
sigm=[]
sigc=[]
am = []
ac=[]
for i in np.arange(len(dset)):
    
    dtype_in_question = i
    #print(dset[i])
    sel2 = []
    selection = np.array(selection)

    for s in selection:
        if s[15] == dtype_in_question:
            sel2.append(s)

    sel2 = np.array(sel2)
    S = sel2[:,6]
    colors_S = cmap(S)[:,:3]
    #print(colors_S)
    #input()
    mean_sigma = np.mean(sel2[:,19])
    lvl_sig = []
    lvl_A = []
    for lvl in range(0,10):
        temp = []
        temp2 = []
        for s2 in sel2:
            if s2[-1] == lvl:
                temp.append(s2[19])
                temp2.append(s2[20])
        #print(temp)
        #input()
        if len(temp) == 0:
            temp = [0]
            temp2 = [0]
        lvl_sig.append(np.mean(temp))
        lvl_A.append(np.mean(temp2))
    #print(lvl_sig)
    damage = (0.5*(sel2[:,17]+1)*sel2[:,16]+sel2[:,18])
    if len(damage) >1:
        popt,pcov = curve_fit(straight_line,sel2[:,-1],damage)

        fit_x = np.arange(0,10)
        fit_d = straight_line(fit_x,popt[0],popt[1])
        max_fit = fit_d + lvl_sig
        min_fit = fit_d - lvl_sig
    
    #plt.subplot(4,4,i+1)
    plt.subplot(1,3,1)
    plt.title("Dist. Mean")
    plt.xlabel("Spell Level",fontsize = 6)
    plt.ylabel("Average_Damage",fontsize = 6)
    plt.scatter(sel2[:,-1],damage,label = dset[i],c = colors_S)
    
    #plt.errorbar(sel2[:,-1],damage,yerr = sel2[:,19],fmt = ",")
    if len(damage) >1:
        plt.plot(fit_x,fit_d,label = f'fit: m = {popt[0]:.2f} c = {popt[1]:.2f}')
        plt.fill_between(fit_x,min_fit,max_fit,alpha = 0.2)
    plt.legend()
   # plt.show()
    
    """for j in np.arange(len(lvl_sig)):
        if lvl_sig[i] == 0:
            lvl_sig[i] = np.nan"""
    fit_x2 = np.linspace(0,9,100)
    if len(damage) >1:
        try:
            popt2,pcov2 = curve_fit(straight_line,fit_x,lvl_sig)
            
        except:
            popt2 = [0,0]
        
        fit2 = straight_line(fit_x2,popt2[0],popt2[1])
    plt.subplot(1,3,2)
    plt.plot(fit_x,lvl_sig,".",label = "spell dist. sigma")
    if len(damage) >1:
        plt.plot(fit_x2,fit2,label = f'fit: m = {popt2[0]:.2f} c = {popt2[1]:.2f}')

    plt.title("Dist. Sigma")
    plt.xlabel("Spell Level",fontsize = 6)
    plt.ylabel("Mean Sigma",fontsize = 6)
    plt.legend(fontsize = 6)


    if len(damage) >1:
        lvl_A = np.nan_to_num(lvl_A)
        try:
            
            popt3,pcov3 = curve_fit(straight_line,fit_x,lvl_A)
        except Exception as e:
            print(lvl_A)
            print(e)
            popt3 = [0,0]
        fit3 = straight_line(fit_x2,popt3[0],popt3[1])
    plt.subplot(133)
    plt.title("Dist. Amplitude")
    plt.xlabel("Spell Level")
    plt.ylabel("Distribution Amplitude")
    plt.plot(fit_x,lvl_A,".",label = "dist. Amp")
    if len(damage)>1:
        plt.plot(fit_x2,fit3,label = f'fit: m = {popt3[0]:.2f} c = {popt3[1]:.2f}')
    plt.show()
    #plt.clf()

    d0m.append(popt[0])
    d0c.append(popt[1])
    sigm.append(popt2[0])
    sigc.append(popt2[1])
    am.append(popt3[0])
    ac.append(popt3[1])
    #print(f'\n----{dset[i]}----\nAverage_Damage: m = {popt[0]:.2f} c = {popt[1]:.2f}\nDistribution Variance: m = {popt2[0]:.2f} c = {popt2[1]:.2f}\nDistribution Amplitude: m = {popt3[0]:.2f} c = {popt3[1]:.2f}\n')


print(d0m)
d0m=np.array(d0m)
d0c=np.array(d0c)
sigm=np.array(sigm)
sigc=np.array(sigc)
am = np.array(am) 
ac=np.array(ac)

rels = np.vstack((d0m,d0c,sigm,sigc,am,ac))
rels = pd.DataFrame(rels,columns = dset,index = ["D0_M","D0_c","Sigma_m","Sigma_c","A_m","A_c"])

rels.to_pickle("relationships.pkl")
"""coord_df = pd.DataFrame(spells_coord,columns = ['name', 'level', 'school', 'range', 'concentration', 'verbal',
                                               'somatic', 'material', 'material_consumed', 'material_cp_cost',
                                                'area_types', 'ability_checks', 'ability_checks', 'saving_throws', 'cantrip_scaling',
                                                'dtype','n','s','m', 'sigma', 'a'])

coord_df.to_pickle("coord_data.pkl")"""


"""selected_data = np.vstack((level_coord,range_coord,n_coord,s_coord,m_coord,area_types_coord,somatic_coord,dtype_coord)).T
df2 = pd.DataFrame(selected_data,columns=['level','range','n','s','m','area_type','S','damage_type'])
df2.to_pickle("Selected_Data.pkl")     """

"""data2 = pd.read_pickle("coord_data.pkl")
selected_data = pd.read_pickle("Selected_data.pkl")
plt.matshow(selected_data.corr())
plt.show()
plt.matshow(data2.corr())
plt.xticks(range(len(list(data2.keys()))),list(data2.keys()),rotation = 90,fontsize=8)
plt.yticks(range(len(list(data2.keys()))),list(data2.keys()),fontsize=8)
plt.show()"""
