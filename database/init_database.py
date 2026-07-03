import database.database_manager as database
import database.seeding as seeding

print('Initializing database...')
database.create_database()

print('Database initialized.')
print('Initializing seeding...')
seeding.seed()

print('Seeding done.')