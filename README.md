 # Exam-Scheduling-system 
 
 Project in smart system 
Optimize university exam schedules to reduce student conflicts and evenly 
distribute workload using genetic algorithms.


# Output:Optimized exam timetable with minimal student conflicts and even distribution of workload. 

# 1. The Dataset
This dataset is designed for modeling and solving the exam scheduling problem in universities. It contains multiple types of information covering course details, students, instructors, classrooms, and exam timeslots. This enables analysis of exam conflicts and optimization of the final schedule using techniques such as genetic algorithms or other optimization methods.

2. Dataset Contents
The dataset consists of several CSV files, including:

courses.csv
Contains course data such as:

Course ID (course_id)

Course name (course_name)

Academic department (department)

Credit hours (credits)

students.csv
Student information, including:

Student ID (student_id)

Personal or academic details (e.g., year of study)

instructors.csv
Instructor data:

Instructor ID (instructor_id)

Name or other details

classrooms.csv
Classroom information, such as:

Classroom ID (classroom_id)

Capacity (capacity)

Building name (building_name)

Room number (room_number)

timeslots.csv
Details of exam timeslots:

Timeslot ID (timeslot_id)

Day (day)

Start time (start_time)

End time (end_time)

schedule.csv
Initial exam schedule or booking data:

Links course, student, instructor, classroom, and timeslot

 # 3. Importance of the Data
Conflict Analysis:
The data allows examination of the number of students enrolled in multiple courses to determine which exams should not be scheduled simultaneously.

Resource Allocation:
Classroom data and capacities help assign appropriate rooms for each exam based on student numbers.

Time Distribution:
Timeslot data facilitates distributing exams across different time periods to reduce conflicts and student overload.

Decision Support:
Using this data with optimization algorithms helps produce a practical and balanced exam schedule minimizing logistical issues.


  # Project Objectives:
Create a Conflict-Free Exam Schedule : 
➤ Ensure that no student has overlapping exams at the same time.

 Optimize Resource Utilization :
➤ Distribute exams efficiently across classrooms to avoid overcrowding and underuse.

 Minimize Student Stress :
➤ Spread exams over different days and times so students are not overloaded on a single day.

 Consider Course Sizes :
➤ Assign appropriate classrooms based on course size (small, medium, large), avoiding small rooms for large courses.

 Ensure Balanced Scheduling of Large Courses :
➤ Avoid scheduling multiple large courses simultaneously to reduce pressure on facilities and students.

 Prevent Exceeding Room Capacity :
➤ Make sure the number of students does not exceed the capacity of the assigned classroom for safety and practicality.

 Generate Multiple Solutions and Find the Best Using Genetic Algorithm :
➤ Explore many possible schedules and select the optimal ones based on multiple criteria (conflicts, balance, stress, capacity).

Produce a Final Schedule That Can Be Exported and Reviewed :
➤ Create a final exam schedule that can be saved as a CSV file and analyzed or visualized.

Provide Visual Reports of the Schedule
➤ Present graphs showing exam distribution by day, student load, and classroom usage for better understanding.

