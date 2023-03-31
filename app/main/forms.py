from flask_wtf import FlaskForm
from wtforms import StringField, SelectField, SubmitField, FloatField, SelectMultipleField, TextAreaField
from wtforms.validators import DataRequired, InputRequired, NumberRange, Regexp, Optional, Email
from wtforms.fields import DateField
#from ..models import Branch, Employee, Client, Loan
#from ..models import *


class InputWhatIfForm(FlaskForm):
    database = SelectField('Database', choices = [
         ('amazon_product','Amazon Product'),
         ('adult_income','Adult Income'),
    ])
    base_tables = SubmitField('BaseTables',validators=None)

    #groupby query part
    use_table = SelectField('USE Table', choices=[
        ('product','Product'),
        ('review','Review'),
        ('custom','Custom'),
    ])
    #not use now
    use = TextAreaField('USE', default="")
    run_relevant = SubmitField('Run Aggregate Query',validators=None)

    sample_query = SubmitField('Sample Query', validators=None)  # add sample query button



    #dropdown menu field for placeholder query
    #OUTPUT part
    output_type = SelectField('OutputType', choices = [
        ('avg','AVG'),
        ('sum','SUM'),
        ('count','COUNT')
    ])
    output_attrs = SelectField('OutputAttrs',choices= [
        ('post_rtng','POST(Rtng)'),
        ('post_brand','POST(Brand)'),
        ('post_senti','POST(Senti)'),
        ('post_price','POST(Price)')
    ])
    #dynamic button
    #output_attrs = SelectField('OutputAttrs',coerce=str)
    
    #UPDATE part
    #overall update
    update_attrs = SelectField('UpdateAttrs',choices=[
        ('blank',''),
        ('brand','POST(Brand)'),
        ('price','POST(Price)'),
        ('category','POST(Category)'),
        ('quality','POST(Quality)'),
        ('color','POST(Color)'),
        ('ratng','POST(Rtng)'),
        ('senti','POST(Senti)')
    ])
    #update_attrs = SelectField('UpdateAttrs', coerce=str)
    update_const = FloatField('UpdateConst')
    update_sign = SelectField('UpdateSign', choices=[
        ('+','+'),
        ('x','x')
    ])


    #vary upate part
        #overall update
    update_attrs_vary = SelectField('UpdateAttrs',choices=[
        ('blank',''),
        ('brand','POST(Brand)'),
        ('price','POST(Price)'),
        ('category','POST(Category)'),
        ('quality','POST(Quality)'),
        ('color','POST(Color)'),
        ('ratng','POST(Rtng)'),
        ('senti','POST(Senti)')
    ])
    update_const_vary_from = FloatField('UpdateConst')
    update_const_vary_to = FloatField('UpdateConst')

    update_sign_vary = SelectField('UpdateSign', choices=[
        ('+','+'),
        ('x','x')
    ])
    '''
    for_attrs = SelectField('ForAttrs',choices=[
        ('rtng','Rtng'),
        ('brand','Brand'),
        ('senti','Senti'),
        ('price','Price')
    ])
    for_comp = SelectField('ForComp', choices=[
        ('bigger','>'),
        ('equal','='),
        ('smaller','<'),
        ('bigger_equal','>='),
        ('smaller_equal','<=')
    ])
    for_const = FloatField('ForConst')

    
    show_attrs = SelectField('ShowAttrs',choices=[
        ('rtng','Rtng'),
        ('brand','Brand'),
        ('senti','Senti'),
        ('price','Price')
    ])
    show_comp = SelectField('ShowComp', choices=[
        ('bigger','>'),
        ('equal','='),
        ('smaller','<'),
        ('bigger_equal','>='),
        ('smaller_equal','<=')
    ])
    show_const = FloatField('ShowConst')
    '''


    #need to change: to dynamic choices when connect to database
    output = TextAreaField('OUTPUT',default = "")
    #output_for = SubmitField('+ FOR')
    #need to add: +FOR function
    #need to add: PRE(xxx)
    #need to add: +WHEN function
    when = TextAreaField('WHEN', default="")
    
    run = SubmitField('Update Output')
    sample_update = SubmitField('Sample Update')

    overall = SubmitField('Overall Updates', validators=None)
    vary_updates = SubmitField('Vary Updates', validators=None)
    show_vary_updates = SubmitField('Show Vary Updates', validators=None)
    specify_constraints = SubmitField('Specify Constraints', validators=None)

class InputHowToForm(FlaskForm):
    database = SelectField('Database', choices = [
         ('amazon_product','Amazon Product'),
         ('adult_income','Adult Income'),
    ])
    #base_tables = SubmitField('BaseTables',validators=None)

    #groupby query part
    use = TextAreaField('USE', default="")
    run_relevant = SubmitField('Run Aggregate Query',validators=None)

    sample_query = SubmitField('Sample Query', validators=None)  # add sample query button

    #UPDATE part
    update_attrs = SelectField('UpdateAttrs',choices=[
        ('blank',''),
        ('brand','POST(Brand)'),
        ('price','POST(Price)'),
        ('category','POST(Category)'),
        ('quality','POST(Quality)'),
        ('color','POST(Color)'),
        ('ratng','POST(Rating)'),
        ('senti','POST(senti)')
    ])
    #update_attrs = SelectField('UpdateAttrs', coerce=str)
    update_const_from = FloatField('UpdateConst')
    update_const_to = FloatField('UpdateConst')
    update_sign = SelectField('UpdateSign', choices=[
        ('+','+'),
        ('x','x')
    ])
    #the meaning of "."

    #OBJECTIVE part
    objective = SelectField('objective', choices=[
        ('maximize','Maxmize'),
        ('minimize','minimize')
    ])

    for_attrs = SelectField('ForAttrs',choices=[
        ('rtng','Rtng'),
        ('brand','Brand'),
        ('senti','Senti'),
        ('price','Price')
    ])
    for_comp = SelectField('ForComp', choices=[
        ('bigger','>'),
        ('equal','='),
        ('smaller','<'),
        ('bigger_equal','>='),
        ('smaller_equal','<=')
    ])
    for_const = FloatField('ForConst')


    show_attrs = SelectField('ShowAttrs',choices=[
        ('rtng','Rtng'),
        ('brand','Brand'),
        ('senti','Senti'),
        ('price','Price')
    ])
    show_comp = SelectField('ShowComp', choices=[
        ('bigger','>'),
        ('equal','='),
        ('smaller','<'),
        ('bigger_equal','>='),
        ('smaller_equal','<=')
    ])
    show_const = FloatField('ShowConst')



    #need to change: to dynamic choices when connect to database
    output = TextAreaField('OUTPUT',default = "")
    #output_for = SubmitField('+ FOR')
    #need to add: +FOR function
    #need to add: PRE(xxx)
    #need to add: +WHEN function
    when = TextAreaField('WHEN', default="")
    for_condition = TextAreaField('FOR', default="")

    #dropdown menu field for placeholder query
    #OUTPUT part
    output_type = SelectField('OutputType', choices = [
        ('avg','AVG'),
        ('sum','SUM'),
        ('count','COUNT')
    ])
    output_attrs = SelectField('OutputAttrs',choices= [
        ('post_rtng','POST(Rtng)'),
        ('post_brand','POST(Brand)'),
        ('post_senti','POST(Senti)'),
        ('post_price','POST(Price)')
    ])
    #dynamic button
    #output_attrs = SelectField('OutputAttrs',coerce=str)
    
    
    
    run = SubmitField('RUN')

    overall = SubmitField('Overall Updates', validators=None)
    vary_updates = SubmitField('Vary Updates', validators=None)
    show_vary_updates = SubmitField('Show Vary Updates', validators=None)
    specify_constraints = SubmitField('Specify Constraints', validators=None)