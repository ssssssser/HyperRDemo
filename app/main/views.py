import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
#from matplotlib.figure import Figure
import seaborn as sns
import pandas as pd
import numpy as np
from flask import render_template, request, redirect, url_for, make_response, current_app, flash, session, jsonify, Flask, send_file
import json
from .forms import InputWhatIfForm, InputHowToForm

from . import main

from .. import db

from ..models import Product, Review
from flask_login import login_required
from datetime import date
from sqlalchemy import extract
import datetime
import sqlite3
import time
import io
import random
from flask import Response
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import hyperAPI
import sqlparse
from sqlparse.sql import Identifier, IdentifierList, Function

RERUN = False #used so that if you run code after run aggregate query, it won't delete the graph shown. This should be better optimized in future
RERUN2 = False
attr_list, items = None, None
vary_updates=False


def get_relevant_table(form):
    conn = sqlite3.connect('data-dev.sqlite')

    query = form.use.data
    print('query',query)
    cursor = conn.execute(query)
    attributes = cursor.description
    if attributes:
        attr_list = []
        for attr in attributes:
            attr_list.append(attr[0])

        items = cursor.fetchall()
    else:
        attr_list = None
        items = None
    conn.close()
    return attr_list,items


#function get df for hyper function, change get data from sqlite
def get_tuple():
    #conn = sqlite3.connect('data-dev.sqlite')
    #query = 'SELECT * FROM amazon_product, amazon_review WHERE amazon_product.pid = amazon_review.pid'
    df = pd.read_csv('db/amazon_merge_smalldata.csv')
    return df

#helper function#not use now
def get_updated_value(update_sign, update_attrs, update_const, df):
    if update_sign == 'add':
        after_update_val = df[update_attrs].mean() + update_const
    else: #if update_sign is multiply
        after_update_val = df[update_attrs].mean()* update_const
    return after_update_val

#parse sql query to get aggregate function type and attribute name for update
def parse_sql_query(query):
    parsed_query = sqlparse.parse(query)[0]
    results = []
    def process_token(token):
        if isinstance(token, Identifier):
            for child_token in token.tokens:
                if isinstance(child_token, Function):
                    function_name = child_token.get_name()
                    attribute_name = child_token.get_parameters()[0].value
                    attribute_name = attribute_name.split('.')[-1]
                    results.append((function_name.upper(), attribute_name))
                    break

    for token in parsed_query.tokens:
        if isinstance(token, IdentifierList):
            for identifier in token.get_identifiers():
                process_token(identifier)
        else:
            process_token(token)

    return results


#parse function for when and for vary update dropdown
def parse_sql_query2(query):
    parsed_query = sqlparse.parse(query)[0]
    results = []

    def process_token(token):
        if isinstance(token, Identifier):
            for child_token in token.tokens:
                if isinstance(child_token, Function):
                    function_name = child_token.get_name()
                    attribute_name = child_token.get_parameters()[0].value
                    attribute_name = attribute_name.split('.')[-1]
                    results.append((function_name.upper(), attribute_name))
                    break

    for token in parsed_query.tokens:
        if isinstance(token, IdentifierList):
            for identifier in token.get_identifiers():
                process_token(identifier)
                # Add the identifier name to the results list
                results.append(identifier.get_real_name())
        if token.value.upper() == 'FROM':
            break
        else:
            process_token(token)

    return results

#function used to split when and for conditions:
def split_condition(data):
    #input: FOR/WHEN data from UI
    #output: attr_list and attr_val_list
    if data:
        data = data.split('AND')
        #print(data)
        preval = []
        prevallst_cate = []
        for i in data:
            i_lst = i.split('=')
            #print(i_lst)
            preval.append(i_lst[0].strip())
            prevallst_cate.append(i_lst[1].strip("'"))
        return preval, prevallst_cate
    else:
        return [],[]

def get_bar_plot(attr_x,attr_y,attr_list, items,isUpdate):
    #print("hello")
    df = pd.DataFrame(columns = attr_list,data=items)
    sns.set(style="darkgrid", font_scale=2)
    plt.figure(figsize=(8,4))
    #fig = Figure()
    #ax = fig.subplots()
    #print(attr_list)
    #print(items)
    #print(df)
    ax = sns.barplot(x=attr_x, y=attr_y,
                    data=df,
                    #palette='Set2',
                    color = "#00BFFF",
                    errwidth=0)
    #plt.xlabel('AVG(Rtng)')
    # plt.xlabel("Type")

    #set items label
    #unique_y_attrs = df[attr_y].unique() #get's the unique labels
    #print(unique_y_attrs)
    #ax.set_yticklabels(unique_y_attrs, fontsize=20)
    #df[attr_y] = df[attr_y].astype("category")
    # ax.set_yticks(yticks)
    # ax.set_yticklabels(df[attr_y].unique(), fontsize=20)
    
    #xticks = ax.get_xticks()
    #ax.set_xticklabels(xticks, fontsize=20)
    
    #ax.set_xlabel(attr_x, fontsize=20)
    # ax.set_ylabel(attr_y, fontsize=20)

    # yticks = [str(int(y)) for y in ax.get_yticks()]
    # ax.set_yticklabels(yticks, fontsize=20)

    #gfg.set(xlabel ="GFG X", ylabel = "GFG Y", title ='some title')
    
    '''
    We need the 'Type' attribute when sending in the update attribute data
    so we can distinguish between the pre and the post values
    THIS TYPE ATTRIBUTE WILL COME WHEN WE CONNECT TO HYPER API
    This is commented because we need the data from Hyper API to make this function work
    '''

    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig('app/static/bar_graph.jpg', dpi=500)
    print("generate new graph")
    return fig
    # plt.close(fig)    
    #fig.show()
    # if isUpdate:
    #     ax = sns.barplot(x=attr_x, y=attr_y, hue='Type',
    #             data=df,
    #             #color = "#86bdf7", 
    #             palette=['Set2','#86bdf7'],
    #             errwidth=0)
    #     fig = ax.get_figure()
    #     fig.tight_layout()
    #     fig.savefig('app/static/bar_update_graph.jpg', dpi=500)
    #     print("generate new graph")
    #     return fig


def get_update_bar_plot(attr_x,attr_y,attr_list, items, score_ls):
    print("hello: update bar plot")
    df_graph = pd.DataFrame(columns = attr_list,data=items)
    df_graph['type'] = 'PRE'
    df_graph2 = pd.DataFrame(columns=attr_list,data=score_ls)
    df_graph2['type'] = 'POST'
    df_graph = df_graph.append(df_graph2)
    #print('df_graph here:',df_graph)
    sns.set(style="darkgrid", font_scale=2)

    plt.figure(figsize=(8,4))
    #fig = Figure()
    #ax = fig.subplots()
    #print(items)
    print(attr_list)
    print(items)
    #print(df)
    colors = ["#00BFFF", "#FFA07A"]
    ax2 = sns.barplot(x=attr_y, y=attr_x,hue='type',
                    data=df_graph,
                    #palette='Set2',
                    palette = colors,
                    errwidth=0)
    
     # Move legend to the bottom
    handles, labels = ax2.get_legend_handles_labels()
    plt.legend(handles=handles[0:2], labels=labels[0:2],
               bbox_to_anchor=(0.5, 1.15), loc='upper center', ncol=2)
    #plt.xlabel('AVG(Rtng)')
    # plt.xlabel("Type")

    #set items label
    #unique_y_attrs = df[attr_y].unique() #get's the unique labels
    #print(unique_y_attrs)
    #ax.set_yticklabels(unique_y_attrs, fontsize=20)
    #df[attr_y] = df[attr_y].astype("category")
    # ax.set_yticks(yticks)
    # ax.set_yticklabels(df[attr_y].unique(), fontsize=20)
    
    #xticks = ax.get_xticks()
    #ax.set_xticklabels(xticks, fontsize=20)
    
    #ax.set_xlabel(attr_x, fontsize=20)
    # ax.set_ylabel(attr_y, fontsize=20)

    # yticks = [str(int(y)) for y in ax.get_yticks()]
    # ax.set_yticklabels(yticks, fontsize=20)

    #gfg.set(xlabel ="GFG X", ylabel = "GFG Y", title ='some title')
    
    '''
    We need the 'Type' attribute when sending in the update attribute data
    so we can distinguish between the pre and the post values
    THIS TYPE ATTRIBUTE WILL COME WHEN WE CONNECT TO HYPER API
    This is commented because we need the data from Hyper API to make this function work
    '''
    #print('breakpoint')
    fig2 = ax2.get_figure()
    fig2.tight_layout()
    fig2.savefig('app/static/update_bar_graph.jpg', dpi=500)
    #print("generate new graph complete!!!!!!!!")
    return fig2
    # plt.close(fig)    
    #fig.show()
    # if isUpdate:
    #     ax = sns.barplot(x=attr_x, y=attr_y, hue='Type',
    #             data=df,
    #             #color = "#86bdf7", 
    #             palette=['Set2','#86bdf7'],
    #             errwidth=0)
    #     fig = ax.get_figure()
    #     fig.tight_layout()
    #     fig.savefig('app/static/bar_update_graph.jpg', dpi=500)
    #     print("generate new graph")
    #     return fig

'''OLD COMMENT:this function is the updated bar plot when updating the query
should be able to take in multiple inputs "i.e update attribute boxes"
so it needs to work for vary updates and overall updates (sql based or value based)
*****IMPLEMENT ACTUAL FUNCTIONALLITY OF UPDATING QUERY WHEN CONNECTING THE HYPER API***********
'''
'''
NEW UPDATE TO COMMENT:This is not used but i'm keeping it for future if we do want to make a seperate function
update query has been added to the earlier function with a boolean isUpdate parameter
'''

def get_vary_update_bar_plot(df_graph, update_attrs_vary, attr_x):
    plt.figure(figsize=(8,4))
    ax = sns.lineplot(data=df_graph,x = update_attrs_vary,y=attr_x, marker = 'o', palette='Set2')
    #out_filename = 'line_graph1.jpg'
    #plt.xlabel('Price (x)')
    ax.set_xlabel('Updates of '+update_attrs_vary)
    ax.set_ylabel(attr_x)
    # if selected_value != 'blank':
    #     ax.set_title(f'Vary updates view for group"{selected_value}"')
    #ax.legend(title='Legend')
    fig = ax.get_figure()

    # Add labels to data points
    # for x, y in zip(df_graph[update_attrs_vary], df_graph[attr_x]):
    #     ax.text(x, y, f'{y:.2f}', ha='center', va='bottom')
    fig.tight_layout()
    fig.savefig('app/static/line_graph.jpg', dpi=500)
    print("generate new graph")
    return fig


# def get_updateQuery_bar_plot(attr_x,attr_y,attr_list, items):
#     df = pd.DataFrame(columns = attr_list,data=items)
#     plt.figure(figsize=(8,4))
#     #fig = Figure()
#     #ax = fig.subplots()
#     ax = sns.barplot(x=attr_x, y=attr_y,
#                     data=df,
#                     palette='Set2',
#                     errwidth=0)
#     #plt.xlabel('AVG(Rtng)')
#     # plt.xlabel("Type")
#     fig = ax.get_figure()
#     fig.tight_layout()
#     fig.savefig('app/static/bar_graph.jpg', dpi=500)
#     print("generate new graph")
#     retur

df = get_tuple()
cate_cols = ['category','brand','color']
df, le_dict = hyperAPI.convert_cate_features(df,cate_cols)

@main.route('/')
def index():
    return render_template('index.html')

# error handling
# @main.errorhandler(500)
# def server_error(error):
#     return render_template('index.html'), 500

@main.route('/bar_plot.png')
def bar_plot_png():
    global attr_list, items
    #attr_list, items = get_relevant_table(form)
    #print(attr_list, "look here")
    attr_x = attr_list[1]
    attr_y = attr_list[0]
    #print('run sub function:'+str(attr_x))
    #form = InputWhatIfForm()
    #if 'run_relevant' in request.form:
    #print('RUN Agg query')
    #print(attr_x,attr_y,attr_list,items)
    fig = get_bar_plot(attr_x, attr_y, attr_list, items,False)
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')

#not use now
@main.route('/update_bar_plot.png')
def update_bar_plot_png():
    global attr_list, items, score_ls
    #print('what is attr_list:', attr_list)
    #attr_list, items = get_relevant_table(form)
    #print(attr_list, "look here")
    attr_x = attr_list[1]
    attr_y = attr_list[0]
    #print('run sub function:'+str(attr_x))
    #form = InputWhatIfForm()
    #if 'run_relevant' in request.form:
    #print('RUN update query plot')
    fig2 = get_update_bar_plot(attr_x, attr_y, attr_list, items,score_ls)
    #print('break?')
    output = io.BytesIO()
    #print('break?')
    FigureCanvas(fig2).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')

@main.route('/causal_graph_popup.jpg')
def image():
    return send_file('static/causal_graph_new.jpg')

@main.route('/query_input_what_if', methods=['GET', 'POST'])
def query_input_what_if():
    form = InputWhatIfForm()
    print("we are in query input what if")
    #print(request.form)
    global RERUN, RERUN2
    global attr_list, items,score_ls
    global df, le_dict
    #casual_graph = True
    final_run = False
    if RERUN == False:
        causal_graph = False
    else: 
        causal_graph = True
    #for specify constraints -> causal graph
    if RERUN2 == False:
        specConst = False
    else:
        specConst =True

    #overall=True
    #vary_updates=False
    global vary_updates
    show_vary_updates=False


    #agg_query = False

    #now it means run the aggregate query and plot
    #run_relevant = run aggregate query
    if 'run_relevant' in request.form:
        print('RUN Agg query')
        try:
            attr_list, items = get_relevant_table(form)
        except:
            error_msg = "Bad input query, try the sample query"
            return render_template('query_input_what_if.html', form=form, error=error_msg)
        if (attr_list==None or items == None):
            error_msg = "Bad input query, try the sample query"
            return render_template('query_input_what_if.html', form=form, error=error_msg)
        #print(attr_list, "look here")
        attr_x = attr_list[1]
        attr_y = attr_list[0]
    
        #get_bar_plot(attr_x,attr_y, attr_list, items,False)
        # if attr_list:
        #     form.output_attrs.choices = [(attr,"POST(" +str(attr)+")") for attr in attr_list]
        #     form.update_attrs.choices = [(attr,"POST(" +str(attr)+")") for attr in attr_list]
        session['attr_list']=attr_list
        session['items'] = items
        # session['default_plot'] = True 
        #print("session attr list:",session['attr_list'])
        
        session['causal_graph'] = True #when run the aggregate query, generate a default bar plot(automatically choose the)
        causal_graph = session.get('causal_graph', None)
        RERUN = True
        #agg_query = True
        # return render_template('query_input_what_if.html', form=form,
        #     causal_graph=causal_graph, )

        query = form.use.data
        update_button = parse_sql_query2(query)

        session['update_button'] = update_button
        #print('update_button',update_button)
        dropdown_attr = update_button[0]
        vary_dropdown = df[dropdown_attr].unique()
        if dropdown_attr in ['category','brand','color']:
            vary_dropdown = le_dict[dropdown_attr].inverse_transform(vary_dropdown)
        session['vary_dropdown'] = tuple(vary_dropdown)

    elif 'run' in request.form:
        start = time.time()
        print('RUN button')
        
        #print(attr_list)
        try:
            update_attr_list = session['attr_list']
            update_items = session['items']
            #print(update_attr_list)
            attr_x = update_attr_list[1]
            attr_y = update_attr_list[0]

            update_button = session.get('update_button', None)
            
            q_type = update_button[1][0].lower()
            target_attr = update_button[1][1].lower()
        except:
            error_msg = "Bad update attribute input, try the sample update"
            return render_template('query_input_what_if.html', form=form, errorUpdate=error_msg)
        #generate default_plot

        #update_attr_x = form.update_attrs.data
        #update_attr_y = form.output_attrs.data
        #if show overall updates
        if vary_updates == False:
            print('vary_updates == False (we are in overall updates)')
            #need to add API
            ###TODO: add subfunction for this part
            #df = get_tuple()
            #df = pd.DataFrame(df)
            try:
                update_attrs = request.form.get('update_attrs')
                update_const = float(request.form.get('update_const'))
            
                update_sign = request.form.get('update_sign')
                #after_update_val = get_updated_value(update_sign, update_attrs, update_const, df)
            
                when_data = request.form.get('when')
                #when_data = when_data.split('AND')
            except:
                error_msg = "Bad update attribute input, try the sample update"
                return render_template('query_input_what_if.html', form=form, errorUpdate=error_msg)
            print(when_data)
            # preval = []
            # prevallst_cate = []
            # if when_data != None:
            #     for i in when_data:
            #         i_lst = i.split('=')
            #         print(i_lst)
            #         preval.append(i_lst[0].strip())
            #         prevallst_cate.append(i_lst[1].strip("'"))
            preval,prevallst_cate = split_condition(when_data)
            #print('preval:',preval)
            #print('prevallst:',prevallst_cate)
            #print("over here pal")
            #check if data on price and quality is okay??
            #print(df) #good (gives the whole merged database bucketized)
            #print(attr_x) #this rtng(i want this as "rating")
            #print(preval) #['brand']
            #print(prevallst_cate) #['"Asus"'] we don't use this??
            #print(update_attrs) #price
            #print(update_const) #1.2 (i.e pre price (*)(+) 1.2)
            #print(update_sign)
            
            #obviously need to be changed to not be hardcoded
            # cate_cols = ['category','brand','color'] 
            # newdf,ledict = hyperAPI.convert_cate_features(df,cate_cols) #no idea what ledict is
            #print(newdf)
            #score_ls = hyperAPI.get_weighted_avg_output(df, 'avg',attr_x,preval,prevallst_cate,[],[],[update_attrs],[after_update_val],['*'],'',{},'category')
            #print(type(newdf))
            #newdf = pd.DataFrame(newdf) 
            #obviously fix hardcoded parts
            #print('paramter:',df, le_dict, q_type,attr_x,preval,prevallst_cate,[],[],[update_attrs],update_sign,update_const,attr_y)
            try:
                score_ls = hyperAPI.groupby_output(df, le_dict, q_type,target_attr,preval,prevallst_cate,[],[],[update_attrs],update_sign,update_const,attr_y)
            except:
                error_msg = "Bad update attribute input, try the sample update"
                return render_template('query_input_what_if.html', form=form, errorUpdate=error_msg)
            print('scores are:', score_ls)

            get_update_bar_plot(update_attr_list[0],update_attr_list[1], update_attr_list, update_items,score_ls)
            session['final_run'] = True
            #return a value that relays to output a certain graph with the run button
            #attr_list = session.get('attr_list', None)
            #items = session.get('items', None)
            #default_plot = session.get('items', None)

            final_run = session.get('final_run',None)
        #if show vary updates
        elif vary_updates == True:
            print('vary_updates == True')
            #df = get_tuple()
            try:
                update_attrs_vary = request.form.get('update_attrs_vary')
                update_const_vary_from = float(request.form.get('update_const_vary_from'))
                update_const_vary_to = float(request.form.get('update_const_vary_to'))

                update_sign_vary = request.form.get('update_sign_vary')
                selected_value = request.form.get('vary_dropdown')
                print('selected_value',selected_value.lower())
                # after_update_val_from = get_updated_value(update_sign_vary, update_attrs_vary,update_const_vary_from,df)
                # after_update_val_to = get_updated_value(update_sign_vary, update_attrs_vary,update_const_vary_to,df)
                ###To Change:
                #after_update_val_ls = np.linspace(after_update_val_from, after_update_val_to, num=10, endpoint=True)
                #update_const_val_ls = np.linspace(update_const_vary_from,update_const_vary_to,num=10,endpoint=True)
                update_const_val_ls = np.linspace(update_const_vary_from,update_const_vary_to,num=5) #not include endpoint
                #print(after_update_val_ls)

                when_data = request.form.get('when')
            except:
                error_msg = "Bad vary update input, try the sample update"
                return render_template('query_input_what_if.html', form=form, errorVary=error_msg)
            preval,prevallst_cate = split_condition(when_data)
            #print(df,le_dict,q_type,attr_x,preval,prevallst_cate,[],[],[update_attrs_vary],update_sign_vary,update_const_val_ls)
            try:
                if selected_value != 'blank':
                    dropdown_attr = update_button[0].lower()
                    print('ADDDDD')
                    #print(df,le_dict,q_type,attr_x,preval+[dropdown_attr],prevallst_cate+[selected_value],[],[],[update_attrs_vary],update_sign_vary,update_const_val_ls)
                    score_ls = hyperAPI.vary_output(df,le_dict,q_type,target_attr,preval+[dropdown_attr],prevallst_cate+[selected_value],[],[],[update_attrs_vary],update_sign_vary,update_const_val_ls)
                else:
                    score_ls = hyperAPI.vary_output(df,le_dict,q_type,target_attr,preval,prevallst_cate,[],[],[update_attrs_vary],update_sign_vary,update_const_val_ls)
            except:
                error_msg = "Bad vary update input, try the sample update"
                return render_template('query_input_what_if.html', form=form, errorVary=error_msg)
            print('scores are:', score_ls)
            df_graph = pd.DataFrame(data=score_ls,columns=[update_attrs_vary,attr_x])
            get_vary_update_bar_plot(df_graph, update_attrs_vary, attr_x)
            session['show_vary_updates']=True
            vary_updates = session.get('vary_updates', None)
            show_vary_updates = session.get('show_vary_updates', None)
            

        
        end = time.time()
        print('time:', end-start)
    elif 'overall' in request.form:
        #overall = True
        # session['overall'] = True
         session['vary_updates'] = False
        # overall = session.get('overall', None)
         vary_updates = session.get('vary_updates', None)
    elif "vary_updates" in request.form:
        #overall=False       
        # session['overall'] = False
        session['vary_updates'] = True
        # overall = session.get('overall', None)
        vary_updates = session.get('vary_updates', None)
    elif "show_vary_updates" in request.form:
        session['show_vary_updates']=True
        print('sucess')
        vary_updates = session.get('vary_updates', None)
        show_vary_updates = session.get('show_vary_updates', None)
        #print(show_vary_updates)


    elif "specify_constraints" in request.form:
        session['specify_constraints']=True
        print('sucess')
        specConst = session.get('specify_constraints', None)
        #show_vary_updates = session.get('show_vary_updates', None)
        #print(specConst)
        RERUN2 = True


    if form.is_submitted():
        #causal_graph = session.get("causal_graph", None)
        #print(casual_graph)
        #attr_list = session.get('attr_list', None)
        #print(attr_list) #what is attr_list and where do we get it from?
        #items = session.get('items', None)
        #default_plot = session.get('items', None)
        #final_run = session.get('final_run',None)
        form.set_vary_dropdown_choices(session.get('vary_dropdown', ''))
        # return render_template('query_input_what_if.html', form = form,
        #     causal_graph=casual_graph, attr_list = attr_list, items = items, len_item = len(attr_list), default_plot = default_plot,final_run=final_run)
        return render_template('query_input_what_if.html', form = form, causal_graph=causal_graph, final_run=final_run, vary_updates=vary_updates, show_vary_updates = show_vary_updates,specConst= specConst)
    else:
        print('wrong')
        return render_template('query_input_what_if.html', form = form)

@main.route('/query_input_how_to', methods=['GET', 'POST'])
def query_input_how_to():
    form  = InputHowToForm()
    print("we are in query input how to")
    global RERUN
    global attr_list, items,score_ls
    global df, le_dict
    #casual_graph = True
    final_run = False
    if RERUN == False:
        causal_graph = False
    else: 
        causal_graph = True
   

    #overall=True
    #vary_updates=False
    #global vary_updates
    #show_vary_updates=False
    
    result_ls = []
    result_columns = []

    #now it means run the aggregate query and plot
    #run_relevant = run aggregate query
    if 'run_relevant' in request.form:
        try:
            attr_list, items = get_relevant_table(form)
        except:
            error_msg = "Bad input query, try the sample query"
            return render_template('query_input_how_to.html', form=form, errorHow=error_msg)
        #print(attr_list, "look here")
        if (attr_list==None or items == None):
            error_msg = "Bad input query, try the sample query"
            return render_template('query_input_how_to.html', form=form, errorHow=error_msg)
        #attr_x = attr_list[1]
        #attr_y = attr_list[0]
    
        #get_bar_plot(attr_x,attr_y, attr_list, items,False)
        # if attr_list:
        #     form.output_attrs.choices = [(attr,"POST(" +str(attr)+")") for attr in attr_list]
        #     form.update_attrs.choices = [(attr,"POST(" +str(attr)+")") for attr in attr_list]
        session['attr_list']=attr_list
        session['items'] = items
        # session['default_plot'] = True 
        #print("session attr list:",session['attr_list'])
        
        session['causal_graph'] = True #when run the aggregate query, generate a default bar plot(automatically choose the)
        causal_graph = session.get('causal_graph', None)
        RERUN = True
    
        #agg_query = True
        # return render_template('query_input_what_if.html', form=form,
        #     causal_graph=causal_graph, )

        ###TODO: let `avg` and `rating` button change with query input
        query = form.use.data
        update_button2 = parse_sql_query(query) #update_button[0][0]='AVG', update_buytton[0][1] = 'rating'
        print('update_button',update_button2)
        session['update_button2'] = update_button2
    
    elif 'run' in request.form:
        print('RUN button')
        update_button2 = session.get('update_button2', None)
        #print('update_button',update_button)
        #update_attr_list = session['attr_list']
        #update_items = session['items']
        #print(update_attr_list)
        #attr_x is the objective variable
        #attr_x = update_attr_list[1]
        #attr_y = update_attr_list[0]
        if update_button2:
            q_type = update_button2[0][0].lower()
            AT = update_button2[0][1].lower()

        #df = get_tuple()
        update_attr = request.form.get('update_attrs')
        #print('update_attr_print',update_attr)

        update_const_from = request.form.get('update_const_from')
        if update_const_from:
            update_const_from = float(update_const_from)
        else:
            update_const_from = None
        
            if update_const_from:
                update_const_from = float(update_const_from)
            else:
                update_const_from = None
            

        try:
            update_attr = request.form.get('update_attrs')
            update_const_from = request.form.get('update_const_from')
            if update_const_from:
                update_const_from = float(update_const_from)
            else:
                update_const_from = None
            update_const_to = request.form.get('update_const_to')
            if update_const_to:
                update_const_to = float(update_const_to)
            else:
                update_const_to = None
            
            update_sign = request.form.get('update_sign')

            update_attr2 = request.form.get('update_attrs2')
            #print('update_attr2',update_attr2)

            if update_attr2 !='blank':
                update_const_from2 = request.form.get('update_const_from2')
                if update_const_from2:
                    update_const_from2 = float(update_const_from2)
                else:
                    update_const_from2 = None
                update_const_to2 = request.form.get('update_const_to2')
                if update_const_to2:
                    update_const_to2 = float(update_const_to2)
                else:
                    update_const_to2 = None
                update_sign2 = request.form.get('update_sign2')
                update_attr_ls = [update_attr,update_attr2]
                update_const_from_ls = [update_const_from,update_const_from2]
                update_const_to_ls = [update_const_to,update_const_to2]
                update_sign_ls = [update_sign,update_sign2]
            else:
                update_attr_ls = [update_attr]
                update_const_from_ls = [update_const_from]
                update_const_to_ls = [update_const_to]
                update_sign_ls = [update_sign]

        except:
            error_msg = "Bad update attribute input, try the sample update"
            return render_template('query_input_how_to.html', form=form, errorConstraint=error_msg)
        try:    
            for_data = request.form.get('for_condition')
            for_preval, for_prevallst_cate = split_condition(for_data)
        except:
            error_msg = "Bad update attribute input, try the sample update"
            return render_template('query_input_how_to.html', form=form, errorObjective=error_msg)
        try:    
            when_data = request.form.get('when')
            when_preval, when_prevallst_cate = split_condition(when_data)
        except:
            error_msg = "Bad update attribute input, try the sample update"
            return render_template('query_input_how_to.html', form=form, errorConstraint=error_msg)
        preval = when_preval+for_preval
        prevallst = when_prevallst_cate + for_prevallst_cate
        
        #print('preval:', preval)
        #print('prevallst:',prevallst)

        start = time.time()
        try:
        #print(df, le_dict, q_type, AT,preval,prevallst,[],[],update_attr_ls,update_sign_ls,update_const_from_ls, update_const_to_ls)
            top_attrs, top_values, top_objectives = hyperAPI.optimization_multiple(df, le_dict, q_type, AT,preval,prevallst,[],[],update_attr_ls,update_sign_ls,update_const_from_ls, update_const_to_ls) #can adjust binwidth
        except:
            error_msg = "Bad update attribute input, try the sample update"
            return render_template('query_input_how_to.html', form=form, errorConstraint=error_msg)
        print(top_attrs, top_values, top_objectives)
        result_columns = ['Rank','Attributes','To Value','New Objective Value']
        result_ls = []
        for i in range(5):
            result_i = [i+1,top_attrs[i], str(round(top_values[i],1))+update_sign+'PRE('+top_attrs[i]+')',round(top_objectives[i],4)]
            result_ls.append(result_i)
        
        end=time.time()
        print('time:',end-start)

        session['final_run'] = True

        
        final_run = session.get('final_run',None)
       



    if form.is_submitted():
        #causal_graph = session.get("causal_graph", None)
        #print(casual_graph)
        #attr_list = session.get('attr_list', None)
        #print(attr_list) #what is attr_list and where do we get it from?
        #items = session.get('items', None)
        #default_plot = session.get('items', None)
        #final_run = session.get('final_run',None)
        
        # return render_template('query_input_what_if.html', form = form,
        #     causal_graph=casual_graph, attr_list = attr_list, items = items, len_item = len(attr_list), default_plot = default_plot,final_run=final_run)
        update_button2 = session.get('update_button2',[])
        return render_template('query_input_how_to.html', form = form, causal_graph = causal_graph, final_run=final_run, result_columns=result_columns, result_ls = result_ls, update_button2=update_button2)
    else:
        print('wrong')
        return render_template('query_input_how_to.html', form = form)