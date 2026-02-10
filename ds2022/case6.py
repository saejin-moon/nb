# /// script
# dependencies = ["SQLAlchemy", "pandas"]
# ///

import marimo

__generated_with = "0.19.5"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Crash Course: SQL Joins with SQLite, SQLAlchemy, and pandas

    This notebook:
    - Creates a small **SQLite** database.
    - Uses **SQLAlchemy** for the engine/connection.
    - Uses **pandas** to run queries and display results.
    - Demonstrates **INNER**, **LEFT**,and **CROSS** joins.

    > Tip: RIGHT and FULL OUTER JOIN are not natively supported in SQLite. We’ll emulate them.
    """)
    return


@app.cell
def _():
    # packages added via marimo's package management: SQLAlchemy pandas !pip -q install SQLAlchemy pandas
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In SQLAlchemy 1.4, `future=True` opts you into the SQLAlchemy 2.0-style behavior. It flips a bunch of defaults so you're using the newer API patterns. We also turn off loggin by setting `echo=False`
    """)
    return


@app.cell
def _():
    import pandas as pd
    from sqlalchemy import create_engine

    engine = create_engine("sqlite:////content/sql_joins_demo.db", echo=False, future=True)

    # SELECT -> DataFrame
    def q(sql: str) -> pd.DataFrame:
        return pd.read_sql(sql, engine)

    # DDL / INSERT / UPDATE / DELETE -> no return
    def exec_sql(sql: str) -> None:
        with engine.begin() as conn:
            conn.exec_driver_sql(sql)
    return create_engine, exec_sql, pd, q


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1) Create a tiny database

    We’ll make three tables:
    - `authors(id, name)`
    - `books(id, title, author_id, pub_year)` — some books have a missing/unknown author
    - `sales(book_id, qty)` — only some books have sales rows
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Database structure

    ```
    +---------+           +---------+           +-------+
    | authors |           | books   |           | sales |
    +----+----+           +---+-----+           +---+---+
    | id | PK |<--+       | id|PK   |<--+    +--|book_id|FK
    |name|    |   |       |title    |   |    |  | qty   |
    +----+----+   |    +--|author_id|   +----+  +-------+
                  |    |  |pub_year |
                  |    |  +---------+
                  |    |
                  +----+  (authors.id = books.author_id)
    ```
    """)
    return


@app.cell
def _(display, exec_sql, q):
    # DDL (one statement per call)
    exec_sql("DROP TABLE IF EXISTS sales;")
    exec_sql("DROP TABLE IF EXISTS books;")
    exec_sql("DROP TABLE IF EXISTS authors;")

    exec_sql("""
    CREATE TABLE authors (
      id INTEGER PRIMARY KEY,
      name TEXT NOT NULL
    );
    """)

    exec_sql("""
    CREATE TABLE books (
      id INTEGER PRIMARY KEY,
      title TEXT NOT NULL,
      author_id INTEGER,
      pub_year INTEGER,
      FOREIGN KEY(author_id) REFERENCES authors(id)
    );
    """)

    exec_sql("""
    CREATE TABLE sales (
      book_id INTEGER,
      qty INTEGER,
      FOREIGN KEY(book_id) REFERENCES books(id)
    );
    """)

    exec_sql("""
    INSERT INTO authors (id, name) VALUES
      (1, 'Ada Lovelace'),
      (2, 'Grace Hopper'),
      (3, 'Donald Knuth'),
      (4, 'Leslie Lamport');      -- This author will have no books
    """)

    exec_sql("""
    INSERT INTO books (id, title, author_id, pub_year) VALUES
      (10, 'Analytical Engine Notes', 1, 1843),
      (11, 'Compilers 101', 2, 1952),
      (12, 'The Art of Computer Programming', 3, 1968),
      (13, 'Distributed Systems Sketches', NULL, 1978),  -- unknown author
      (14, 'Future of Typesetting', 3, 1977);
    """)

    exec_sql("""INSERT INTO sales (book_id, qty) VALUES
      (10, 120),
      (11, 75),
      (12, 250);
    """)



    print("Database created at /content/sql_joins_demo.db")
    print("Authors:")
    display(q("SELECT * FROM authors"))
    print("Books:")
    display(q("SELECT * FROM books"))
    print("Sales:")
    display(q("SELECT * FROM sales"))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2) INNER JOIN
    Return rows that match in **both** tables.
    """)
    return


@app.cell
def _(display, q):
    sql = '''
    SELECT b.id AS book_id, b.title, a.name AS author, b.pub_year
    FROM books b
    INNER JOIN authors a ON b.author_id = a.id
    ORDER BY b.id;
    '''
    display(q(sql))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3) LEFT JOIN
    Return **all** rows from the left table and matching rows from the right (or `NULL` if no match).
    """)
    return


@app.cell
def _(display, q):
    sql_1 = '\nSELECT a.id AS author_id, a.name AS author, b.id AS book_id, b.title\nFROM authors a\nLEFT JOIN books b ON b.author_id = a.id\nORDER BY a.id, b.id;\n'
    display(q(sql_1))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4) CROSS JOIN
    Cartesian product of the two tables (every row from A paired with every row from B). Use with care.
    """)
    return


@app.cell
def _(display, q):
    sql_2 = '\nSELECT a.name AS author, b.title AS book_title\nFROM authors a\nCROSS JOIN books b\nORDER BY a.name, b.id\nLIMIT 12;\n'
    display(q(sql_2))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Task: Create a Local SQLite Database for a Many-to-Many Schema

    You are given the following conceptual schema (Students–Courses with a join table Enrollments):
    ```
    +-------------------+             +--------------------+             +-------------------+
    |     Students      |             |     Enrollments    |             |      Courses      |
    +-------------------+             +--------------------+             +-------------------+
    | PK student_id ----+----------+  | PK enrollment_id   |   +-------> | PK course_id      |
    | name              |          +->| FK student_id      |   |         | title             |
    | major             |             | FK course_id ----------+         | credits           |
    +-------------------+             | grade              |             +-------------------+
                                      +--------------------+
    ```

    ## Your job
    Write the **SQLite-compatible SQL DDL** statements to create a local database (tables + constraints) that matches this structure.

    ### Requirements
    1. **Tables**
       - `Students(student_id, name, major)`
       - `Courses(course_id, title, credits)`
       - `Enrollments(enrollment_id, student_id, course_id, grade)`

    2. **Keys & Relationships**
       - Primary keys on: `Students.student_id`, `Courses.course_id`, `Enrollments.enrollment_id`.
       - Foreign keys:
         - `Enrollments.student_id` → `Students.student_id`
         - `Enrollments.course_id` → `Courses.course_id`
       - Enforce foreign keys (SQLite requires this).

    3. **Constraints (use reasonable types and checks)**
       - `name` and `title` must be `NOT NULL`.
       - `credits` is an integer and must be within a reasonable range (e.g., 1–6).
       - `grade` is optional, but if present it should be one of typical letter grades (e.g., `A, B, C, D, F, W, I`) using a `CHECK`
       ```sql
       --snip--
          course_id     INTEGER NOT NULL,
          grade         TEXT CHECK(grade IN ('A', 'B', 'C', 'D', 'F', 'W', 'I')),
        --snip--
        ```
       - Prevent duplicate enrollment of the same student in the same course (add a unique constraint on `(student_id, course_id)`).

       ```sql
         --snip--
          UNIQUE (student_id, course_id),
        --snip--
       ```

    4. **SQLite specifics**
       - Use `INTEGER PRIMARY KEY` for rowid PKs in SQLite.

    ### Deliverables
    - Upload a link to your Colab notebook to gradescope.  The TA will run your colab notebook and grade results. Because there isn't autograder, TA office hour time this week maybe reduced to make time for grading. 10 minutes per student X 120 student 12000 minutes for 20 hours of TA time.

    ### Notes/Hints
    - SQLite data types are flexible; use `INTEGER` and `TEXT` where appropriate.
    - To test FK enforcement in SQLite CLI or Python

    **Do not include any INSERTs**—this task is schema design only.
    """)
    return


@app.cell
def _(create_engine, pd):
    engine_2 = create_engine('sqlite:///assignment.db', echo=False, future=True)

    def q_1(sql: str) -> pd.DataFrame:
    # create a new engine that connects to local sql query data
        return pd.read_sql(sql, engine_2)

    # SELECT -> DataFrame
    def exec_sql_1(sql: str) -> None:
        with engine_2.begin() as conn:
    # DDL / INSERT / UPDATE / DELETE -> no return
            conn.exec_driver_sql(sql)
    return exec_sql_1, q_1


@app.cell
def _(exec_sql_1):
    # Write you code a sql query here. Once you are down
    exec_sql_1('PRAGMA foreign_keys=ON;')
    exec_sql_1('DROP TABLE IF EXISTS Enrollments;')
    exec_sql_1('DROP TABLE IF EXISTS Courses;')
    exec_sql_1('DROP TABLE IF EXISTS Students;')
    exec_sql_1('\nCREATE TABLE Students (\n    student_id INTEGER PRIMARY KEY,\n    name TEXT NOT NULL,\n    major TEXT\n);\n')
    exec_sql_1('\nCREATE TABLE Courses (\n    course_id INTEGER PRIMARY KEY,\n    title TEXT NOT NULL,\n    credits INTEGER CHECK(credits BETWEEN 1 AND 6)\n);\n')
    exec_sql_1("\nCREATE TABLE Enrollments (\n    enrollment_id INTEGER PRIMARY KEY,\n    student_id INTEGER,\n    course_id INTEGER,\n    grade TEXT CHECK(grade IN ('A', 'B', 'C', 'D', 'F', 'W', 'I', 'A-', 'B+')),\n    FOREIGN KEY (student_id) REFERENCES Students (student_id),\n    FOREIGN KEY (course_id) REFERENCES Courses (course_id),\n    UNIQUE (student_id, course_id)\n);\n")
    print('Database and tables created successfully.')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Write the sql queries that  will insert the following data into tables

    **Students**
    | student_id | name        | major        |
    |-----------:|-------------|--------------|
    | 1001       | Alice Smith | CS           |
    | 1002       | Bob Lee     | Data Science |

    **Courses**
    | course_id | title                 | credits |
    |---------:|------------------------|--------:|
    | 501      | Databases              | 3       |
    | 502      | Intro to Data Science  | 4       |

    **Enrollments**
    | enrollment_id | student_id | course_id | grade |
    |--------------:|-----------:|----------:|-------|
    | 1             | 1001       | 501       | A     |
    | 2             | 1001       | 502       | B+    |
    | 3             | 1002       | 501       | A-    |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #Write the sql queries
    """)
    return


@app.cell
def _(exec_sql_1):
    # Start writing the sql queries here.
    exec_sql_1("INSERT INTO Students (student_id, name, major) VALUES\n  (1001, 'Alice Smith', 'CS'),\n  (1002, 'Bob Lee', 'Data Science');\n")
    exec_sql_1("INSERT INTO Courses (course_id, title, credits) VALUES\n  (501, 'Databases', 3),\n  (502, 'Intro to Data Science', 4);\n")
    exec_sql_1("INSERT INTO Enrollments (enrollment_id, student_id, course_id, grade) VALUES\n  (1, 1001, 501, 'A'),\n  (2, 1001, 502, 'B+'),\n  (3, 1002, 501, 'A-');\n")
    print('Data inserted successfully.')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #Question 1.
    Write an SQL query to list every student who is enrolled in at least one course. **The results should show every course student is enrolled in**   Show each student's `student_id`, `name`, the `course_id`, and their `grade`.
    """)
    return


@app.cell
def _(display, q_1):
    # Write your code here.
    query1 = '\nSELECT s.student_id, s.name, e.course_id, e.grade\nFROM Students s\nINNER JOIN Enrollments e ON s.student_id = e.student_id;\n'
    display(q_1(query1))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #Question 2
    Write an SQL query that lists **all students**, even those not enrolled in any course.  For students without enrollments, the course fields should show as `NULL`.
    """)
    return


@app.cell
def _(display, exec_sql_1, q_1):
    # Write your code here.
    exec_sql_1("INSERT OR IGNORE INTO Students (student_id, name, major) VALUES (1003, 'Charlie Brown', 'Philosophy');")
    query2 = '\nSELECT s.student_id, s.name, e.course_id, e.grade\nFROM Students s\nLEFT JOIN Enrollments e ON s.student_id = e.student_id;\n'
    display(q_1(query2))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Question 3
    Write an SQL query that lists **all students**, even those not enrolled in any course.   For students without enrollments, the course fields should show as `NULL`.
    """)
    return


@app.cell
def _(display, q_1):
    #Write your code here.
    # This question is identical to Q2, but we can enhance it by joining all three tables to get the course title.
    query3 = '\nSELECT s.student_id, s.name, c.title AS course_title, e.grade\nFROM Students s\nLEFT JOIN Enrollments e ON s.student_id = e.student_id\nLEFT JOIN Courses c ON e.course_id = c.course_id;\n'
    display(q_1(query3))
    return


@app.cell
def _():
    #Write your code here.
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Understanding `GROUP BY` in SQLite

    The `GROUP BY` clause in SQL is used to **aggregate rows** that share the same values in one or more columns.

    It's often paired with **aggregate functions** like:
    - `COUNT()` — how many rows per group
    - `AVG()` — average value
    - `SUM()` — total value
    - `MIN()` / `MAX()` — smallest or largest value

    ---

    ##  Basic Example

    Imagine a table `Enrollments`:

    | student_id | course_id | grade |
    |-------------|------------|--------|
    | 1 | 101 | A |
    | 1 | 102 | B |
    | 2 | 101 | A |
    | 3 | 103 | C |
    | 3 | 104 | B |

    Let's count how many courses each student is enrolled in:

    ```sql
    SELECT
      student_id,
      COUNT(course_id) AS n_courses
    FROM Enrollments
    GROUP BY student_id;
    ```

    **Result:**

    | student_id | n_courses |
    |-------------|------------|
    | 1 | 2 |
    | 2 | 1 |
    | 3 | 2 |

    This collapses multiple rows per student into one row per `student_id`.

    ---

    ## Grouping with Multiple Columns

    You can group by **more than one column**.
    For example, counting how many students earned each grade per course:

    ```sql
    SELECT
      course_id,
      grade,
      COUNT(student_id) AS n_students
    FROM Enrollments
    GROUP BY course_id, grade
    ORDER BY course_id;
    ```

    This groups by both `course_id` and `grade`—so you get one row per unique combination.

    ---

    ## Using Aggregates with `JOIN`s

    Combine `GROUP BY` with joins to count enrollments per student name:

    ```sql
    SELECT
      s.name,
      COUNT(e.course_id) AS n_courses
    FROM Students AS s
    LEFT JOIN Enrollments AS e
      ON e.student_id = s.student_id
    GROUP BY s.name
    ORDER BY n_courses DESC;
    ```

    Students with no enrollments will still appear (because of the `LEFT JOIN`),
    but their count will be `0`.

    ---

    ## SQLite’s Special Behavior

    SQLite is **more lenient** than most databases:
    - It allows selecting columns **not listed in the GROUP BY** clause.
    - In that case, it returns an **arbitrary value** from within each group — not guaranteed or deterministic.

    Example (unsafe pattern):

    ```sql
    SELECT student_id, grade
    FROM Enrollments
    GROUP BY student_id;
    ```

    SQLite won’t throw an error, but `grade` will be **one random grade** per student.
    In other databases (like PostgreSQL or MySQL in strict mode), this would **fail**.

    **Best practice:**
    Only select grouped columns or use aggregate functions.

    ---

    ## Summary

    | Concept | Example | Behavior |
    |----------|----------|-----------|
    | Basic grouping | `GROUP BY student_id` | Aggregates rows per student |
    | Multiple keys | `GROUP BY course_id, grade` | One row per unique combo |
    | With `JOIN` | `LEFT JOIN` then `GROUP BY` | Works across related tables |
    | Non-aggregated select |  Allowed in SQLite but nondeterministic | Avoid! |

    ---

     **Tip:**
    Use `GROUP BY` to summarize large datasets, but remember that in SQLite,
    results of non-aggregated columns not in `GROUP BY` are **undefined** — always aggregate or group properly!
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Question 4
    Write an SQL query to count how many courses each student is enrolled in.  The result should include the students name, id and course count.
    Include students with zero enrollments, showing `0` for `n_courses`.
    """)
    return


@app.cell
def _(display, q_1):
    #Write your code here.
    query4 = '\nSELECT s.student_id, s.name, COUNT(e.course_id) AS n_courses\nFROM Students s\nLEFT JOIN Enrollments e ON s.student_id = e.student_id\nGROUP BY s.student_id, s.name\nORDER BY s.student_id;\n'
    display(q_1(query4))
    return


if __name__ == "__main__":
    app.run()
