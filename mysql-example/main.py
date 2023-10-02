#pip install mysql-connector-python
#imports

import mysql.connector
from getpass import getpass
from mysql.connector import connect, Error

#establish a connection, prompt for user credentials

try:
    with connect(
        host="localhost",
        user=input("Enter username: "),
        password=getpass("Enter password: "),
    ) as connection:
        print(connection)
except Error as e:
    print(e)

#create a cursor to execute SQL queries
cursor = connection.cursor()

#create database (car dealership)

try:
    with connect(
        host="localhost",
        user=input("Enter username: "),
        password=getpass("Enter password: "),
    ) as connection:
        create_db_query = "CREATE DATABASE car_dealership"
        with connection.cursor() as cursor:
            cursor.execute(create_db_query)
except Error as e:
    print(e)

##connect to created database

try:
    with connect(
        host="localhost",
        user=input("Enter username: "),
        password=getpass("Enter password: "),
        database="car_dealership"
    ) as connection:
        print(connection)
except Error as e:
    print(e)

#design schema

#create table after connecting to database (for cars)

create_new_table = """
CREATE TABLE cars(
    id INT AUTO_INCREMENT PRIMARY KEY,
    make VARCHAR(10),
    model VARCHAR(10),
    year YEAR,
)
"""
with connection.cursor() as cursor:
    cursor.execute(create_new_table)
    connection.commit()

#create another table (sales_consultants)

create_new_table = """
CREATE TABLE sales_consultants(
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(50,
    hired_since DATE,
    cars_sold INT,
)
"""
with connection.cursor() as cursor:
    cursor.execute(create_new_table)
    connection.commit()

#create a table for service

create_new_table = """
CREATE TABLE service(
    id INT AUTO_INCREMENT PRIMARY KEY,
    vin VARCHAR(100),
    scheduled_date DATE,
    service_performed VARCHAR(100),
    ready char(1),
)
"""
with connection.cursor() as cursor:
    cursor.execute(create_new_table)
    connection.commit()

#read records using SELECT

select_car_dealership_query = "SELECT * FROM cars LIMIT 5"
with connection.cursor() as cursor:
    cursor.execute(select_car_dealership_query)
    result = cursor.fetchall()
    for row in result:
        print(row)

#using WHERE to filter results

select_cars_query = """
SELECT make, model
FROM cars
WHERE make = Ford AND model = F-150 AND year > 2015,
"""
with connection.cursor() as cursor:
    cursor.execute(select_cars_query)
    for id in cursor.fetchall():
        print(id)