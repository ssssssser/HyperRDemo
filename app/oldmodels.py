
# from datetime import date
# from datetime import datetime
# from flask import current_app
# from werkzeug.security import generate_password_hash, check_password_hash
# from flask_login import UserMixin, AnonymousUserMixin
from . import db


#create database tables 

class Product(db.Model):
    __tablename__ = 'amazon_product'

    #tuple_id = db.Column(db.String(30), unique=True)
    pid = db.Column(db.Integer, primary_key=True)
    category = db.Column(db.String(30))
    brand = db.Column(db.String(30))
    color = db.Column(db.String(30))
    quality = db.Column(db.Float)
    price = db.Column(db.Float)
    
    
    reviews = db.relationship('Review', back_populates='product')

class Review(db.Model):
    __tablename__ = 'amazon_review'

    #tuple_id = db.Column(db.String(30), primary_key=True)
    pid = db.Column(db.Integer, db.ForeignKey('amazon_product.pid'))
    review_id = db.Column(db.Integer,primary_key=True)
    rating = db.Column(db.Integer)
    sentiment = db.Column(db.Float)

    product = db.relationship("Product", back_populates="reviews")
