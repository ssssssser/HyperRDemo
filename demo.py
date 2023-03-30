import os
from flask_migrate import Migrate
from app import create_app, db
from app.models import Product, Review


#主文件，启动和管理项目

app = create_app(os.getenv('FLASK_CONFIG') or 'default')
migrate = Migrate(app, db)



@app.shell_context_processor
def make_shell_context():
    return dict(db=db)

# if __name__ =="__main__":
#     app.run(debug=True, port=8080)
#     #app.run()
