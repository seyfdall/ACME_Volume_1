# solutions.py
"""Volume 1: SQL 2.
<Name> Dallin Seyfried
<Class> 001
<Date> 04/10/2023
"""

import sqlite3 as sql


# Problem 1
def prob1(db_file="students.db"):
    """Query the database for the list of the names of students who have a
    'B' grade in any course. Return the list.

    Parameters:
        db_file (str): the name of the database to connect to.

    Returns:
        (list): a list of strings, each of which is a student name.
    """
    # Establish a connection to the database
    b_students = []
    try:
        with sql.connect(db_file) as conn:
            cur = conn.cursor()
            # Get the names of students who have a grade B in any class
            b_students = cur.execute("SELECT SI.StudentName "
                                     "FROM StudentInfo as SI INNER JOIN StudentGrades as SG "
                                     "ON SI.StudentID == SG.StudentID "
                                     "WHERE SG.Grade == 'B';").fetchall()
            b_students = [student for tuple in b_students for student in tuple]
    finally:
        conn.close()

    return b_students


# Test problem 1
def test_prob1():
    print('\n')
    print(prob1())


# Problem 2
def prob2(db_file="students.db"):
    """Query the database for all tuples of the form (Name, MajorName, Grade)
    where 'Name' is a student's name and 'Grade' is their grade in Calculus.
    Only include results for students that are actually taking Calculus, but
    be careful not to exclude students who haven't declared a major.

    Parameters:
        db_file (str): the name of the database to connect to.

    Returns:
        (list): the complete result set for the query.
    """
    # Establish a connection to the database
    calc_students = []
    try:
        with sql.connect(db_file) as conn:
            cur = conn.cursor()
            # Get the tuple (name, major, grade) from the database in Calculus
            calc_students = cur.execute("SELECT SI.StudentName, MI.MajorName, SG.Grade "
                                        "FROM StudentInfo as SI LEFT OUTER JOIN MajorInfo as MI "
                                        "ON SI.MajorID == MI.MajorID "
                                        "INNER JOIN StudentGrades as SG "
                                        "ON SI.StudentID == SG.StudentID "
                                        "INNER JOIN CourseInfo as CI "
                                        "ON SG.CourseID == CI.CourseID "
                                        "WHERE CI.CourseName == 'Calculus';").fetchall()
    finally:
        conn.close()

    return calc_students


# Test Problem 2
def test_prob2():
    print('\n')
    print(prob2())


# Problem 3
def prob3(db_file="students.db"):
    """Query the given database for tuples of the form (MajorName, N) where N
    is the number of students in the specified major. Sort the results in
    descending order by the counts N, then in alphabetic order by MajorName.

    Parameters:
        db_file (str): the name of the database to connect to.

    Returns:
        (list): the complete result set for the query.
    """
    # Establish a connection to the database/create it if it doesn't exist
    major_students = []
    try:
        with sql.connect(db_file) as conn:
            cur = conn.cursor()
            # Get the tuple (MajorName, N) where N is the number of students in the specified major
            major_students = cur.execute("SELECT MI.MajorName, COUNT(*) as num_students "
                                         "FROM StudentInfo as SI LEFT OUTER JOIN MajorInfo as MI "
                                         "ON SI.MajorID == MI.MajorID "
                                         "GROUP BY MI.MajorID "
                                         "ORDER BY num_students DESC, MI.MajorName ASC;").fetchall()
    finally:
        conn.close()

    return major_students


# Test Problem 3
def test_prob3():
    print('\n')
    print(prob3())


# Problem 4
def prob4(db_file="students.db"):
    """Query the database for tuples of the form (StudentName, N, GPA) where N
    is the number of courses that the specified student is in and 'GPA' is the
    grade point average of the specified student according to the following
    point system.

        A+, A  = 4.0    B  = 3.0    C  = 2.0    D  = 1.0
            A- = 3.7    B- = 2.7    C- = 1.7    D- = 0.7
            B+ = 3.4    C+ = 2.4    D+ = 1.4

    Order the results from greatest GPA to least.

    Parameters:
        db_file (str): the name of the database to connect to.

    Returns:
        (list): the complete result set for the query.
    """
    # Establish a connection to the database/create it if it doesn't exist
    student_info = []
    try:
        with sql.connect(db_file) as conn:
            cur = conn.cursor()
            # Get the tuple (StudentName, N, GPA) where N is number of courses student is in
            student_info = cur.execute("SELECT SI.StudentName, COUNT(*), AVG(SG.gradeisa) "
                                         "FROM ("
                                            "SELECT StudentID, CASE Grade "
                                                "WHEN 'A+' THEN 4.0 "
                                                "WHEN 'A' THEN 4.0 "
                                                "WHEN 'A-' THEN 3.7 "
                                                "WHEN 'B+' THEN 3.4 "
                                                "WHEN 'B' THEN 3.0 "
                                                "WHEN 'B-' THEN 2.7 "
                                                "WHEN 'C+' THEN 2.4 "
                                                "WHEN 'C' THEN 2.0 "
                                                "WHEN 'C-' THEN 1.7 "
                                                "WHEN 'D+' THEN 1.4 "
                                                "WHEN 'D' THEN 1.0 "
                                                "WHEN 'D-' THEN 0.7 END as gradeisa "
                                            "FROM StudentGrades) as SG "
                                         "INNER JOIN StudentInfo as SI "
                                         "ON SG.StudentID == SI.StudentID "
                                         "GROUP BY SG.StudentID "
                                         "ORDER BY avg(gradeisa) DESC;").fetchall()
    finally:
        conn.close()

    return student_info


# Test Problem 4
def test_prob4():
    print('\n')
    print(prob4())


# Problem 5
def prob5(db_file="mystery_database.db"):
    """Use what you've learned about SQL to identify the outlier in the mystery
    database.

    Parameters:
        db_file (str): the name of the database to connect to.

    Returns:
        (list): outlier's name, outlier's ID number, outlier's eye color, outlier's height
    """
    raise NotImplementedError("Problem 5 Incomplete")
