from . import db 
class Product(db.Model):
    __tablename__ = 'amazon_product'

    pid = db.Column(db.Integer(primary_key=True))
    category = db.Column(db.String(30))
    brand = db.Column(db.String(30))
    color = db.Column(db.String(30))
    quality = db.Column(db.Float)
    price = db.Column(db.Float)

    reviews = db.relationship('Review', back_populates="product")

class Review(db.Model):
    __tablename__ = 'amazon_review'

    pid = db.Column(db.Integer(db.ForeignKey('amazon_product.pid')))
    review_id = db.Column(db.Integer(primary_key=True))
    rating = db.Column(db.Integer)
    sentiment = db.Column(db.Float)

    product = db.relationship('Product', back_populates="reviews")

