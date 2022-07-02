import numpy as np
import matplotlib.pyplot as plt
from numpy import inf
import pandas as pd

def translate(s,n,m):
    D0 = 0.5*(s+1)*n+m
    sig = np.sqrt(n*(s**2-1)/12)
    A = D0/(sig*np.sqrt(2*np.pi))
    
    return(D0,A,sig)

def create_dice_data():
    N = np.arange(1,21)
    #S = np.arange(2,21)
    S = np.array([2,4,6,8,12,20])
    M = [0,30,40]

    D0 = []
    A = []
    Sig = []

    n_coords = []
    s_coords = []
    m_coords = []

    for n in N:
        for s in S:
            for m in M:
                d0,a,sig = translate(s,n,m)
                D0.append(d0)
                A.append(a)
                Sig.append(sig)
                s_coords.append(s)
                n_coords.append(n)
                m_coords.append(m)
                
    D0 = np.array(D0)
    A = np.array(A)
    Sig = np.array(Sig)
    m_coords = np.array(m_coords)
    n_coords = np.array(n_coords)
    s_coords = np.array(s_coords)

    final = np.vstack((n_coords,s_coords,m_coords,D0,A,Sig)).T

    df = pd.DataFrame(final,columns = ["n","s","m","d0","a","sig"])
    print(df)
    df.to_pickle("dice_gauss_relations.pkl")

def get_dice_rels():
    data = pd.read_pickle("dice_gauss_relations.pkl")
    data_array = np.array(data)
    return(data_array)

def inverse_translate(A,D0,Sig):
    data = get_dice_rels()
    differ = np.array([0,0,0,D0,A,Sig])
    diff = data-differ
    diff = np.abs(diff)
    identifier = np.sum(diff[:,3:],axis = 1)

    i = list(identifier).index(min(identifier))
    n,s,m,d0,a,sig = data[i]
    print(f"closest match: n: {n}, s: {s}, m: {m}")
    print(f'checks: D0: {D0-d0:.3f}, A: {A-a:.3f}, sig: {Sig-sig:.3f}')
    error = np.mean([D0-d0,A-a,Sig-sig])
    return(n,s,m,error)

def predict_average_damage(dtype,level):
    dtype_data = pd.read_pickle("relationships.pkl")
    factors = dtype_data[dtype]
    md,cd,mo,co,ma,ca = factors
    A = ma*level+ca
    o = mo*level+ma
    D = md*level+cd
    n,s,m,e = inverse_translate(A,D,o)
    print(f"On average a Level {level} {dtype} spell will do: {n:.1g}d{s:.1g}+{m:.1g} damage")
    return(n,s,m,e)

def change_type(spell_name,spell_slot,new_type):
    data_coord = pd.read_pickle("coord_data.pkl")
    data = pd.read_csv("output.csv")
    names = data["name"]
    Indices = []
    for i in np.arange(len(list(names))):
        
        if names[i].lower() == spell_name.lower():
            #print(names[i])
            Indices.append(i)
    #print(Indices)
    options = []
    for i in Indices:
        options.append(np.array(data_coord)[i])
    #print(options)
    choice = []
    for sc in options:
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
        
        if ss == spell_slot:
            choice = np.concatenate((sc,[ss]))
            
    dtype = data['damage'][int(choice[0])]

    level = choice[21]
    dtype_data = pd.read_pickle("relationships.pkl")
    factors = dtype_data[dtype]
    md,cd,mo,co,ma,ca = factors
    A = ma*level+ca
    o = mo*level+ma
    D = md*level+cd
    A_ = choice[20]
    o_ = choice[19]
    D0 = 0.5*(choice[17]+1)*choice[16]+choice[18]

    A_f = A/A_
    o_f = o/o_
    D_f = D/D0

    factors2 = dtype_data[new_type]
    md,cd,mo,co,ma,ca = factors
    A = ma*level+ca
    o = mo*level+ma
    D = md*level+cd
    new_a = A*A_f
    new_o = o*o_f
    new_d = D*D_f

    n,s,m,e = inverse_translate(new_a,new_d,new_o)
    return(n,s,m,e)

#create_dice_data()

def convert_spell():
    og_spell = input("What is the original spell name? ")
    spell_slot = int(input("What level spell slot are you using? "))
    new_type = input("What type of damage are you converting to? ")
    n,s,m,e = change_type(og_spell,spell_slot,new_type)

    print()
    print(f'level {spell_slot} {og_spell} in the {new_type} damage type is: {int(n)}d{int(s)} + {int(m)}\nerror of {e:.2f}')
    
convert_spell()

