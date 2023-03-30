#!/usr/bin/env bash

rm -r migrations

# sqlite3 data-dev.sqlite <<EOF
# .echo on
# PRAGMA foreign_keys=OFF;
# BEGIN TRANSACTION;
# SELECT 'DROP TABLE IF EXISTS ' || name || ';' FROM sqlite_master WHERE type = 'table';
# COMMIT;
# VACUUM;
# PRAGMA foreign_keys=ON;
# EOF

# HAVE TO MANUALLY DELETE THE TABLES, WANT TO FIGURE A WAY TO GET RID OF THIS
# Path to the SQLite database file
DBFILE="data-dev.sqlite"

# Get a list of all table names in the database
TABLES=$(sqlite3 "$DBFILE" ".tables")

# Loop through each table name and drop it
for TABLE in $TABLES; do
  sqlite3 "$DBFILE" "DROP TABLE $TABLE;"
done

#create the models.py based off of models_config file
rm app/models.py
python createModels.py

## database create and migration
flask db init
flask db migrate
flask db upgrade


## initial records to database   
sqlite3 data-dev.sqlite -cmd ".mode csv" \
"delete from amazon_product" \
".import db/amazon_product.csv amazon_product" \
"delete from amazon_review" \
".import db/amazon_review.csv amazon_review" \


# username for log in is admin; password is 1234567
# "delete from system_user" \
# "insert into system_user values(1,'admin','pbkdf2:sha256:260000\$cWVvrPVgMXzn8I6E\$9cf23719edcb252f928ef52fe95fd6ca94beb5c50580e088e789cb01c1204d0b')" \

