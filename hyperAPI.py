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
from joblib import Parallel, delayed

#hardcoded (need to grab this from a config file (worry about later))
backdoor={'brand':[],'color':['brand'],'category':['brand'],'quality':['brand'],'price':['brand','quality'],
          'sentiment':['brand','category','quality'],'rating':['brand','category','quality','price','color','sentiment']}


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
        return [count*1.0/df.shape[0]]
    #print ("regression classifier",count,df.shape,count*1.0/df.shape[0], target, target_val, conditional)
    score=count*1.0/df.shape[0]
    #newlst = [i * score for i in new_lst]
    #new_lst=newlst

    if len(list(set(new_lst)))==1:
        if new_lst[0]==1:
            return [1]
        else:
            return [0]
        
    if len(conditional)>0:
        X=df[conditional]
    else:
        X=df
    
    ###CHANGE
    #regr = RandomForestRegressor(random_state=0)
    regr = LinearRegression()#random_state=0)
    #print (new_lst)
    #print(X)
    regr.fit(X.values, new_lst)
    #print('regr is:', regr)
    return regr

def train_regression_raw(df,conditional,conditional_values,AT,postlst,postvallst):
    new_lst=[]
    count=0
    
    #print (postlst,postvallst,AT)
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
    #regr = RandomForestRegressor(random_state=0)
    regr = LinearRegression()#random_state=0)
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

    #regr = RandomForestRegressor(random_state=0)
    
    regr = LinearRegression()#(random_state=0)
    regr.fit(X, new_lst)
    #print("timesssssssss",time.process_time() - start)

    #print (regr.coef_.tolist())
    #print (regr.predict_proba([conditional_values]),"ASDFDS")
    #print ("heeeeeere")
    #return (regr.predict([conditional_values].values)[0])
    #CHANGE HERE
    return (regr.predict([conditional_values])[0])
    #return(regr.predict_proba([conditional_values])[0][1])
  

#for what-if
def get_query_output(df,q_type,AT,prelst,prevallst,postlst,postvallst,Ac,c):
    #interference is set of attributes of other tuples in a block that affect current tuple's attribute
    #blocks are list of lists
    
    #Identify all attributes which are used for regression and add as columns 
    
            
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
            #print(regr)
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
        Acindex= prelst.index(Ac[0])
        #print (Acindex,prelst[:Acindex],prelst[Acindex+1:])
        newprelst = prelst[:Acindex]
        newprelst.extend(prelst[Acindex+1:])
        newprevallst = prevallst[:Acindex]
        newprevallst.extend(prevallst[Acindex+1:])
        prelst=newprelst
        prevallst=newprevallst
        
        conditioning_set=prelst
        #        intervention=
        backdoorlst=[]
        for attr in Ac:
            backdoorlst.extend(backdoor[attr])
        backdoorlst=list(set(backdoorlst) - set(prelst))
        if len(backdoorlst)>0:
            backdoorvals=get_C_set(df,backdoorlst)
            #print(backdoorvals)
        else:
            backdoorvals=[[]]
        total_prob=0
        regr=''
        iter=0
        #print (backdoorvals)
        
        #print ("AT is ",AT, set(df[AT].values))
        AT_domain = list(set(df[AT].values))
        for val in AT_domain:

            iter=0
            check1=0
            
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
                currpostlst=copy.deepcopy(postlst)
                currpostvallst=copy.deepcopy(postvallst)
                currpostlst.append(AT)
                currpostvallst.append(val)
                if iter==0:
                    regr=train_regression(df,conditioning_set,conditioning_val,currpostlst,currpostvallst)
                    
                    #regr=train_regression_raw(df,conditioning_set,conditioning_val,AT,postlst,postvallst)#train_regression_raw(df,conditioning_set,conditioning_val,AT,postlst,postvallst)
                try:
                    pogivenck= regr.predict([conditioning_val])[0]#(get_prob_o_regression(df,conditioning_set,conditioning_val,postlst,postvallst))
                except:
                    pogivenck = regr[0]
                #print("this",prelst,prevallst,backdoorlst,backdoorvallst)
                pcgivenk = (get_prob_o_regression(df,prelst,prevallst,backdoorlst,backdoorvallst))
                #print ("adding this value",val,pogivenck,pcgivenk)
                check1+=pogivenck
                #print ("check1 old,new",check1-pogivenck,check1,conditioning_val,conditioning_set)
                total_prob+= val * pogivenck * pcgivenk
                iter+=1
            #print ("for current value",val,total_prob,check1)
            
        #print("final prob is ",total_prob)
        return total_prob



def convert_cate_features(df, cate_cols):
    le_dict = {}
    for col in cate_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        le_dict[col] = le
    return df, le_dict




###NEW: get weighted average score (current version)

def get_weighted_avg_output2(df,q_type,AT,prelst,prevallst,postlst,postvallst,Ac,c_sign, c_const):
    
    
    update_attr = Ac[0]
    #prelst.append(update_attr)
    #print(update_attr)
    #print(type(df))
    grouped_df = df.groupby(update_attr)
    
    scores = []
    weights = []
    #print(set(list(df['rating'].values)))
    new_val = 0
    for old_val, df_group in grouped_df:
        if c_sign == '+':
            new_val = old_val + c_const
        elif c_sign == 'x':
            new_val = old_val * c_const
        #print('old_val:',old_val)
        #print('new_val:',new_val)
        #print(prelst)
        #print(prevallst)
        #group_score = get_query_output(df,q_type,AT,prelst.append(update_attr),prevallst.append(old_val),postlst,postvallst,Ac,[c])
        group_score = get_query_output(df,q_type,AT,prelst+[update_attr],prevallst+[old_val],postlst,postvallst,Ac,[new_val])
        group_weight = len(df_group)
        
        scores.append(group_score)
        weights.append(group_weight)
        #print('score:', group_score)
        #print('weight:', group_weight)

    weighted_avg_score = np.average(scores, weights=weights)
    #return scores, weights
    return weighted_avg_score
    #, scores, weights


#parallel
def get_weighted_avg_output(df,q_type,AT,prelst,prevallst,postlst,postvallst,Ac,c_sign, c_const):
    update_attr = Ac[0]
    prelst.append(update_attr)
    
    grouped_df = df.groupby(update_attr)
    
    scores = []
    weights = []
    #print(set(list(df['rating'].values)))
    if c_sign=='+':
        results = Parallel(n_jobs=5)(delayed(get_query_output)(df,q_type,AT,prelst,prevallst+[old_val],postlst,postvallst,Ac,[old_val+c_const]) for old_val, df_group in grouped_df)
    if c_sign=='x':
        results = Parallel(n_jobs=5)(delayed(get_query_output)(df,q_type,AT,prelst,prevallst+[old_val],postlst,postvallst,Ac,[old_val * c_const]) for old_val, df_group in grouped_df)
    print (results)
    
    for old_val, df_group in grouped_df:
        #if c_sign == '+':
        #    new_val = old_val + c_const
        #elif c_sign == 'x':
        #    new_val = old_val * c_const
        #print('old_val:',old_val)
        #print('new_val:',new_val)
        #print(prelst)
        #print(prevallst)
        #group_score = get_query_output(df,q_type,AT,prelst.append(update_attr),prevallst.append(old_val),postlst,postvallst,Ac,[c])
        #group_score = get_query_output(df,q_type,AT,prelst,prevallst+[old_val],postlst,postvallst,Ac,[new_val])
        group_weight = len(df_group)
        
        #scores.append(group_score)
        weights.append(group_weight)
        #print('score:', group_score)
        #print('weight:', group_weight)
    #scores = results
    weighted_avg_score = np.average(results, weights=weights)
    #return scores, weights
    return weighted_avg_score


def bucketize_price(price):
    bucket_size = 100
    bucket_center = 50
    return (price // bucket_size) * bucket_size + bucket_center

def bucketize_rating(rating):
    bucket_size = 0.5
    bucket_center = 0.25
    return (rating // bucket_size) * bucket_size - bucket_center

#change: do bucketize for quality
def bucketize_quality(quality):
    bucket_size = 0.5
    bucket_center = 0.25
    return (quality // bucket_size) * bucket_size - bucket_center

def groupby_output(df,le_dict,q_type,AT,prelst,prevallst,postlst,postvallst,Ac,c_sign,c_const,group_attr):
    ### get outputs group by attribute
    #print(le_dict)

    
        group_attr_vals = list(set(df[group_attr])) #already in numeric
        le = le_dict[group_attr]
        for i in range(len(prelst)):
            pre = prelst[i]
            preval_cate = prevallst[i]
            if pre in le_dict.keys():
                preval = le_dict[pre].transform([preval_cate])[0]
                prevallst[i] = preval
        print(group_attr_vals)
        score_ls = []
        for val in group_attr_vals:
            score_val = get_weighted_avg_output(df,q_type,AT,prelst+[group_attr],prevallst+[val],postlst,postvallst,Ac,c_sign,c_const)
            #score_val = get_query_output(df,q_type,AT,prelst,prevallst,postlst.append(group_attr),postvallst.append(val),Ac,c,g_Ac_lst,interference, blocks)#,{0:[1,2]})
            #print(score_val)
            val_name = le.classes_[val]
            print(val_name)
            score_ls.append([val_name,score_val])
        return score_ls


def vary_output(df,le_dict, q_type,AT,prelst,prevallst,postlst,postvallst,Ac,c_sign,const_ls):

    ### get outputs group by attribute
    #print(le_dict)
    for i in range(len(prelst)):
        pre = prelst[i]
        preval_cate = prevallst[i]
        if pre in le_dict.keys():
            preval = le_dict[pre].transform([preval_cate])[0]
            prevallst[i] = preval

    score_ls = []
    results = Parallel(n_jobs=5)(delayed(get_weighted_avg_output)(df,q_type,AT,prelst,prevallst,postlst,postvallst,Ac,c_sign,const) for const in const_ls)

    for i in range(len(const_ls)):
        score_ls.append([const_ls[i],results[i]])
    '''
    for const in const_ls:
        print('1111111111111111')
        #print(df,q_type,AT,prelst,prevallst,postlst,postvallst,Ac,c_sign,const)
        score_val = get_weighted_avg_output(df,q_type,AT,prelst,prevallst,postlst,postvallst,Ac,c_sign,const)
        # score_val = get_query_output(df_new,q_type,AT,prelst,prevallst,postlst,postvallst.append,Ac,[const],g_Ac_lst,interference, blocks)#,{0:[1,2]})
        #print(val_name)
        score_ls.append([const,score_val])
    '''
    return score_ls



### function for how-to
def optimize_top_5_bucketized(df, q_type, AT,prelst,prevallst,postlst,postvallst,Ac,c_sign,c_from, c_to, bin_width=0.1):
    # Calculate number of bins based on range and bin width
    num_bins = int(np.ceil((c_to - c_from) / bin_width))

    # Create bins
    bins = np.linspace(c_from, c_to, num_bins + 1)

    # Define the problem
    prob = pulp.LpProblem("Top5_C_Optimization_Bucketized", pulp.LpMaximize)

    # Define variables
    c_vars = [pulp.LpVariable(f"c_{i}", lowBound=0, upBound=1, cat="Binary") for i in range(num_bins)]

    # Define the objective function
    #AT_values = [get_weighted_avg_output(df, q_type, AT, prelst,prevallst,postlst,postvallst,Ac,c_sign,bins[i])for i in range(num_bins)]
    AT_values = Parallel(n_jobs=-1)(delayed(get_weighted_avg_output)(df, q_type, AT, prelst, prevallst, postlst, postvallst, Ac, c_sign, bins[i]) for i in range(num_bins))


    prob += pulp.lpSum([c_vars[i] * AT_values[i] for i in range(num_bins)])

    # Constraint: Only select 5 values of c
    prob += pulp.lpSum(c_vars) == 5

    # Constraint: Select values within range [c_from, c_to]
    prob += pulp.lpSum([c_vars[i] for i in range(num_bins) if bins[i] >= c_from and bins[i+1] <= c_to]) == 5

    # Solve the problem
    prob.solve()

    # Get the top 5 values and corresponding objective values
    values = []
    objectives = []
    for i in range(num_bins):
        if bins[i] >= c_from and bins[i+1] <= c_to:
            value = bins[i]
            values.append(value)
            objectives.append(AT_values[i] * c_vars[i].varValue)
    top_indices = np.argsort(objectives)[-5:]
    top_values = [values[i] for i in top_indices][::-1]
    top_objectives = [objectives[i] for i in top_indices][::-1]

    return top_values, top_objectives

#how to code with category attributes
def optimization(df,le_dict, q_type, AT,prelst,prevallst,postlst,postvallst,Ac,c_sign,c_from, c_to, bin_width=0.1):

    ### get outputs group by attribute

    for i in range(len(prelst)):
        pre = prelst[i]
        preval_cate = prevallst[i]
        if pre in le_dict.keys():
            preval = le_dict[pre].transform([preval_cate])[0]
            prevallst[i] = preval
    #print(df,q_type, AT,prelst,prevallst,postlst,postvallst,Ac,c_sign,c_from, c_to,bin_width)
    top_values, top_objectives = optimize_top_5_bucketized(df,q_type, AT,prelst,prevallst,postlst,postvallst,Ac,c_sign,c_from, c_to,bin_width)
    return top_values,top_objectives