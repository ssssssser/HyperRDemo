### hyper function
###TODO: add comment for function input and output (if need)
import copy
import time
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pulp

def get_combination(lst,tuplelst):
    i=0
    new_tuplelst=[]
    if len(tuplelst)==0:
        l=lst[0]
        for v in l:
            new_tuplelst.append([v])
        if len(lst)>1:
            return get_combination(lst[1:],new_tuplelst)
        else:
            return new_tuplelst
    

    currlst=lst[0]
    for l in tuplelst:
        
        for v in currlst:
            newl=copy.deepcopy(l)
            newl.append(v)
            new_tuplelst.append(newl)
        
    if len(lst)>1:
        return get_combination(lst[1:],new_tuplelst)
    else:
        return new_tuplelst

def get_C_set(df,C):
    lst=[]
    for Cvar in C:
        lst.append(list(set(list(df[Cvar]))))
        
    combination_lst= (get_combination(lst,[]))
    
    return combination_lst

def get_val(row,target,target_val):
    i=0
    while i<len(target):
        if not float(row[target[i]])==float(target_val[i]):
            return 0
        i+=1
    return 1

def train_regression(df,conditional,conditional_values,target,target_val):
    new_lst=[]
    count=0
    for index,row in df.iterrows():
        new_lst.append(get_val(row,target,target_val))
        if new_lst[-1]==1:
            count+=1
    if len(conditional)==0:
        return count*1.0/df.shape[0]
    if len(list(set(new_lst)))==1:
        if new_lst[0]==1:
            return 1
        else:
            return 0
        
    if len(conditional)>0:
        X=df[conditional]
    else:
        X=df
    
    ###CHANGE
    regr = RandomForestRegressor(random_state=0)
    #regr = LogisticRegression(random_state=0)
    regr.fit(X.values, new_lst)
    #print('regr is:', regr)
    return regr

def train_regression_raw(df,conditional,conditional_values,AT):
    new_lst=[]
    count=0
    '''
    for index,row in df.iterrows():
        new_lst.append(get_val(row,target,target_val))
        if new_lst[-1]==1:
            count+=1
    if len(conditional)==0:
        return count*1.0/df.shape[0]
    if len(list(set(new_lst)))==1:
        if new_lst[0]==1:
            return 1
        else:
            return 0
    '''
    if len(conditional)>0:
        X=df[conditional]
    else:
        X=df
    ###CHANGE
    regr = RandomForestRegressor(random_state=0)
    #regr = LinearRegression()#random_state=0)
    regr.fit(X, df[AT])
    return regr

def get_prob_o_regression(df,conditional,conditional_values,target,target_val):
    new_lst=[]
    count=0
    for index,row in df.iterrows():
        new_lst.append(get_val(row,target,target_val))
        if new_lst[-1]==1:
            count+=1
    if len(conditional)==0:
        return count*1.0/df.shape[0]
    if len(list(set(new_lst)))==1:
        if new_lst[0]==1:
            return 1
        else:
            return 0
        
    if len(conditional)>0:
        X=df[conditional]
    else:
        X=df
    start = time.process_time()

    regr = RandomForestRegressor(random_state=0)
    
    #regr = LogisticRegression(random_state=0)
    regr.fit(X, new_lst)
    #print("timesssssssss",time.process_time() - start)

    #print (regr.coef_.tolist())
    #print (regr.predict_proba([conditional_values]),"ASDFDS")
    print ("heeeeeere")
    #return (regr.predict([conditional_values].values)[0])
    #CHANGE HERE
    return (regr.predict([conditional_values])[0])
    #return(regr.predict_proba([conditional_values])[0][1])

def get_query_output(df,q_type,AT,prelst,prevallst,postlst,postvallst,Ac,c,g_Ac_lst,interference, blocks):
    #interference is set of attributes of other tuples in a block that affect current tuple's attribute
    #blocks are list of lists
    
    #Identify all attributes which are used for regression and add as columns 
    backdoor={'brand':[],'color':['brand'],'category':['brand'],'quality':['brand'],'price':['brand','quality'],
          'sentiment':['brand','category','quality'],'rating':['brand','category','quality','price','color','sentiment']}
            
    #print (len(sub_df),len(sub_intervene))
    if q_type=='count':
        conditioning_set=prelst
        #        intervention=
        backdoorlst=[]
        for attr in Ac:
            backdoorlst.extend(backdoor[attr])
        backdoorlst=list(set(backdoorlst))
        if len(backdoorlst)>0:
            backdoorvals=get_C_set(df,backdoorlst)
            #print(backdoorvals)
        else:
            backdoorvals=[]
        total_prob=0
        regr=''
        iter=0
        for backdoorvallst in backdoorvals:
            conditioning_set=[]
            conditioning_set.extend(prelst)
            conditioning_set.extend(Ac)
            conditioning_set.extend(backdoorlst)

            conditioning_val=[]
            conditioning_val.extend(prevallst)
            conditioning_val.extend(c)
            conditioning_val.extend(backdoorvallst)

            #print ("conditioning set",conditioning_set,conditioning_val)
            #print("post condition",postlst,postvallst)
            if iter==0:
                start = time.process_time()

                regr=train_regression(df,conditioning_set,conditioning_val,postlst,postvallst)
                #print("time",time.process_time() - start)
            #print (conditioning_val)
            print(regr)
            pogivenck= regr.predict([conditioning_val])[0]#(get_prob_o_regression(df,conditioning_set,conditioning_val,postlst,postvallst))
            #print("this",prelst,prevallst,backdoorlst,backdoorvallst)
            pcgivenk = (get_prob_o_regression(df,prelst,prevallst,backdoorlst,backdoorvallst))
            #print (pogivenck,pcgivenk)
            total_prob+=pogivenck * pcgivenk
            iter+=1
            
        print("final prob is ",total_prob)
        #print (iter)
        return total_prob
    if q_type=='avg':
        
        conditioning_set=prelst
        #        intervention=
        backdoorlst=[]
        for attr in Ac:
            backdoorlst.extend(backdoor[attr])
        backdoorlst=list(set(backdoorlst))
        if len(backdoorlst)>0:
            backdoorvals=get_C_set(df,backdoorlst)
            print(backdoorvals)
        else:
            backdoorvals=[[]]
        total_prob=0
        regr=''
        iter=0
        print (backdoorvals)
        
        
        for backdoorvallst in backdoorvals:
            
            conditioning_set=[]
            conditioning_set.extend(prelst)
            conditioning_set.extend(Ac)
            conditioning_set.extend(backdoorlst)

            conditioning_val=[]
            conditioning_val.extend(prevallst)
            conditioning_val.extend(c)
            conditioning_val.extend(backdoorvallst)

            #print ("conditioning set",conditioning_set,conditioning_val, AT)
            if iter==0:
                regr=train_regression_raw(df,conditioning_set,conditioning_val,AT)
                
            pogivenck= regr.predict([conditioning_val])[0]#(get_prob_o_regression(df,conditioning_set,conditioning_val,postlst,postvallst))
            #print("this",prelst,prevallst,backdoorlst,backdoorvallst)
            pcgivenk = (get_prob_o_regression(df,prelst,prevallst,backdoorlst,backdoorvallst))
            #print (pogivenck,pcgivenk)
            total_prob+=pogivenck * pcgivenk
            iter+=1
            
        #print("final prob is ",total_prob)
        return total_prob


def convert_cate_features(df, cate_cols):
    le_dict = {}
    for col in cate_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        le_dict[col] = le
    return df, le_dict

def groupby_output(df,q_type,AT,prelst,prevallst,postlst,postvallst,Ac,c,g_Ac_lst,interference, blocks, group_attr):
    #convert categorical features
    cate_cols = ['category','brand','color']
    df_new, le_dict = convert_cate_features(df, cate_cols)

    ### get outputs group by attribute
    print(le_dict)
    group_attr_vals = list(set(df_new[group_attr]))
    le = le_dict[group_attr]
    for i in range(len(prelst)):
        pre = prelst[i]
        preval_cate = prevallst[i]
        if pre in cate_cols:
            preval = le_dict[pre].transform([preval_cate])[0]
            prevallst[i] = preval

    score_ls = []
    for val in group_attr_vals:
        ###TO DO: postlst & postvallst
        score_val = get_query_output(df,q_type,AT,prelst,prevallst,postlst.append(group_attr),postvallst.append(val),Ac,c,g_Ac_lst,interference, blocks)#,{0:[1,2]})
        val_name = le.classes_[val]
        #print(val_name)
        score_ls.append([val_name,score_val])
    return score_ls

def vary_output(df,q_type,AT,prelst,prevallst,postlst,postvallst,Ac,const_ls, g_Ac_lst,interference, blocks):
    #convert categorical features
    cate_cols = ['category','brand','color']
    df_new, le_dict = convert_cate_features(df, cate_cols)
    print('df',df)
    print('df_new', df_new)
    ### get outputs group by attribute
    #print(le_dict)
    for i in range(len(prelst)):
        pre = prelst[i]
        preval_cate = prevallst[i]
        if pre in cate_cols:
            preval = le_dict[pre].transform([preval_cate])[0]
            prevallst[i] = preval
    #group_attr_vals = list(set(df_new[group_attr]))
    #le = le_dict[group_attr]
    score_ls = []
    for const in const_ls:
        ###TO DO: postlst & postvallst
        score_val = get_query_output(df_new,q_type,AT,prelst,prevallst,postlst,postvallst.append,Ac,[const],g_Ac_lst,interference, blocks)#,{0:[1,2]})
        #print(val_name)
        score_ls.append([const,score_val])
    return score_ls

### function for how-to
###TO CHANGE
def optimize_top_5_bucketized(df, get_query_output, c_from, c_to, num_bins):
    # Create bins
    bins = np.linspace(c_from, c_to, num_bins, endpoint=True)
    

    # Define the problem
    prob = pulp.LpProblem("Top5_C_Optimization_Bucketized", pulp.LpMaximize)

    # Define variables
    c_vars = [pulp.LpVariable(f"c_{i}", lowBound=0, upBound=1, cat="Binary") for i in range(num_bins)]

    # Define the objective function
    AT_values = [get_query_output(df,'avg','rating',[],[],[],[],['price'],[bins[i]],['*'],'',{})for i in range(num_bins)]

    prob += pulp.lpSum([c_vars[i] * AT_values[i] for i in range(num_bins)])

    # Constraint: Only select 5 values of c
    prob += pulp.lpSum(c_vars) == 5

    # Solve the problem
    prob.solve()

    # Get the optimal solution
    optimal_bin_indices = [i for i in range(num_bins) if c_vars[i].varValue > 0.5]
    optimal_bin_ranges = [(bins[i], bins[i+1]) for i in optimal_bin_indices]

    return optimal_bin_ranges, pulp.value(prob.objective)