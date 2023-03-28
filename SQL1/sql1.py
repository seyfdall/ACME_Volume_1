# sql1.py
"""Volume 1: SQL 1 (Introduction).
<Name> Dallin Seyfried
<Class> 001
<Date> 3/28/2023
"""

import sqlite3 as sql
import csv


# Problems 1, 2, and 4
def student_db(db_file="students.db", student_info="student_info.csv",
                                      student_grades="student_grades.csv"):
    """Connect to the database db_file (or create it if it doesn’t exist).
    Drop the tables MajorInfo, CourseInfo, StudentInfo, and StudentGrades from
    the database (if they exist). Recreate the following (empty) tables in the
    database with the specified columns.

        - MajorInfo: MajorID (integers) and MajorName (strings).
        - CourseInfo: CourseID (integers) and CourseName (strings).
        - StudentInfo: StudentID (integers), StudentName (strings), and
            MajorID (integers).
        - StudentGrades: StudentID (integers), CourseID (integers), and
            Grade (strings).

    Next, populate the new tables with the following data and the data in
    the specified 'student_info' 'student_grades' files.

                MajorInfo                         CourseInfo
            MajorID | MajorName               CourseID | CourseName
            -------------------               ---------------------
                1   | Math                        1    | Calculus
                2   | Science                     2    | English
                3   | Writing                     3    | Pottery
                4   | Art                         4    | History

    Finally, in the StudentInfo table, replace values of −1 in the MajorID
    column with NULL values.

    Parameters:
        db_file (str): The name of the database file.
        student_info (str): The name of a csv file containing data for the
            StudentInfo table.
        student_grades (str): The name of a csv file containing data for the
            StudentGrades table.
    """
    # Establish a connection to the database/create it if it doesn't exist
    try:
        with sql.connect(db_file) as conn:
            cur = conn.cursor()

            # Drop tables
            cur.execute("DROP TABLE IF EXISTS MajorInfo")
            cur.execute("DROP TABLE IF EXISTS CourseInfo")
            cur.execute("DROP TABLE IF EXISTS StudentInfo")
            cur.execute("DROP TABLE IF EXISTS StudentInfo")

            # Create tables
            cur.execute("CREATE TABLE IF NOT EXISTS MajorInfo (MajorID INTEGER, MajorName TEXT)")
            cur.execute("CREATE TABLE IF NOT EXISTS CourseInfo (CourseID INTEGER, CourseName TEXT)")
            cur.execute("CREATE TABLE IF NOT EXISTS StudentInfo (StudentID INTEGER, StudentName TEXT, MajorID INTEGER)")
            cur.execute("CREATE TABLE IF NOT EXISTS StudentGrades (StudentID INTEGER, CourseID INTEGER, Grade TEXT)")

            # Populate the new tables
            major_info_rows = [(1, 'Math'), (2, 'Science'), (3, 'Writing'), (4, 'Art')]
            course_info_rows = [(1, 'Calculus'), (2, 'English'), (3, 'Pottery'), (4, 'History')]
            cur.executemany("INSERT INTO MajorInfo(MajorID, MajorName) VALUES(?, ?);", major_info_rows)
            cur.executemany("INSERT INTO CourseInfo(CourseID, CourseName) VALUES(?, ?);", course_info_rows)

            # Read in student info
            with open(student_info, 'r') as infile:
                student_info_rows = list(csv.reader(infile))
                cur.executemany("INSERT INTO StudentInfo(StudentID, StudentName, MajorID) VALUES(?, ?, ?);", student_info_rows)

            # Read in student grades csv
            with open(student_grades, 'r') as infile:
                student_grades_rows = list(csv.reader(infile))
                cur.executemany("INSERT INTO StudentGrades(StudentID, CourseID, Grade) VALUES(?, ?, ?);", student_grades_rows)


    finally:
        conn.close()


# Test Problem student_db
def test_student_db():
    student_db()
    print('\n')

    # Test updated tables
    with sql.connect("students.db") as conn:
        cur = conn.cursor()
        cur.execute("SELECT * FROM StudentInfo")
        print([d[0] for d in cur.description])

        for row in cur.execute("SELECT * FROM MajorInfo;"):
            print(row)


# Problems 3 and 4
def earthquakes_db(db_file="earthquakes.db", data_file="us_earthquakes.csv"):
    """Connect to the database db_file (or create it if it doesn’t exist).
    Drop the USEarthquakes table if it already exists, then create a new
    USEarthquakes table with schema
    (Year, Month, Day, Hour, Minute, Second, Latitude, Longitude, Magnitude).
    Populate the table with the data from 'data_file'.

    For the Minute, Hour, Second, and Day columns in the USEarthquakes table,
    change all zero values to NULL. These are values where the data originally
    was not provided.

    Parameters:
        db_file (str): The name of the database file.
        data_file (str): The name of a csv file containing data for the
            USEarthquakes table.
    """
    # Establish a connection to the database/create it if it doesn't exist
    try:
        with sql.connect(db_file) as conn:
            cur = conn.cursor()

        # Reset Database
        cur.execute("DROP TABLE IF EXISTS USEarthquakes")
        cur.execute("CREATE TABLE IF NOT EXISTS USEarthquakes (Year INTEGER, Month INTEGER, Day INTEGER, Hour INTEGER, "
                    "Minute INTEGER, Second INTEGER, Latitude REAL, Longitude REAL, Magnitude REAL)")

        # Read in earthquake data
        with open(data_file, 'r') as infile:
            data_rows = list(csv.reader(infile))
            cur.executemany("INSERT INTO USEarthquakes (Year, Month, Day, Hour, "
                    "Minute, Second, Latitude, Longitude, Magnitude) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?);", data_rows)
            # Delete rows with magnitude of 0
            cur.execute("DELETE FROM USEarthquakes WHERE Magnitude <= 0;")
    finally:
        conn.close()


# Test Earthquakes db
def test_earthquakes_db():
    earthquakes_db()
    print('\n')

    # Test updated tables
    with sql.connect("earthquakes.db") as conn:
        cur = conn.cursor()

        for row in cur.execute("SELECT * FROM USEarthquakes;"):
            print(row)


# Problem 5
def prob5(db_file="students.db"):
    """Query the database for all tuples of the form (StudentName, CourseName)
    where that student has an 'A' or 'A+'' grade in that course. Return the
    list of tuples.

    Parameters:
        db_file (str): the name of the database to connect to.

    Returns:
        (list): the complete result set for the query.
    """
    raise NotImplementedError("Problem 5 Incomplete")


# Problem 6
def prob6(db_file="earthquakes.db"):
    """Create a single figure with two subplots: a histogram of the magnitudes
    of the earthquakes from 1800-1900, and a histogram of the magnitudes of the
    earthquakes from 1900-2000. Also calculate and return the average magnitude
    of all of the earthquakes in the database.

    Parameters:
        db_file (str): the name of the database to connect to.

    Returns:
        (float): The average magnitude of all earthquakes in the database.
    """
    raise NotImplementedError("Problem 6 Incomplete")
