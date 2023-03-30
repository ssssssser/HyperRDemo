import yaml

with open('models_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

with open('app/models.py', 'w') as f:
    f.write("from . import db \n")
    for table_name, table_config in config.items():
        f.write(f"class {table_config['class_name']}(db.Model):\n")
        f.write(f"    __tablename__ = '{table_name}'\n\n")
        for column_name, column_type in table_config['columns'].items():
            f.write(f"    {column_name} = db.Column({column_type})\n")
        f.write("\n")

        for relationship_name, relationship_config in table_config.get('relationships', {}).items():
            f.write(f"    {relationship_name} = db.relationship('{relationship_config['model']}', {relationship_config.get('options', '')})\n\n")
