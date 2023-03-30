import csv
from app.models import Product,Review
from app import db

def import_csv():
    with open('db/amazon_product.csv', 'r') as f:
        reader = csv.reader(f)
        next(reader)  # skip header row
        for row in reader:
            # create a new object of MyModel for each row in the CSV
            my_model = Product(pid=row[0], category=row[1], brand=row[2], color=row[3], quality=row[4], price=row[5])
            db.session.add(my_model)
    
    with open('db/amazon_review.csv', 'r') as f:
        reader = csv.reader(f)
        next(reader)  # skip header row
        for row in reader:
            # create a new object of MyModel for each row in the CSV
            my_model = Review(pid=row[0], review_id=row[1], rating=row[2], sentiment=row[3])
            db.session.add(my_model)
    # commit all changes to the database
    db.session.commit()