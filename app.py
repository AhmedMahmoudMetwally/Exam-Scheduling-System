import os
import pandas as pd
import numpy as np
import random
from collections import defaultdict
from deap import base, creator, tools, algorithms
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
import streamlit as st

# Ignore seaborn warnings to keep the output clean
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module='seaborn')

class ExamScheduler:
    """
    A class to handle exam scheduling using a Genetic Algorithm.
    It loads data, preprocesses it, builds a conflict matrix,
    sets up the GA, runs optimization, and generates a final schedule.
    """
    def __init__(self):
        self.classrooms = None
        self.courses = None
        self.instructors = None
        self.schedule = None
        self.students = None
        self.timeslots = None
        self.exam_data = None
        self.unique_courses = None
        self.n_courses = 0
        self.course_to_idx = {}
        self.conflict_matrix = None
        self.toolbox = None # Will be set up after data is loaded

    def load_data(self, data_dict):
        """
        Loads data from a dictionary of pandas DataFrames (typically from Streamlit file uploaders).
        Args:
            data_dict (dict): A dictionary where keys are filenames (e.g., 'classrooms.csv')
                              and values are the corresponding pandas DataFrames.
        Raises:
            ValueError: If there's an error loading data or required columns are missing.
        Returns:
            bool: True if data is loaded successfully.
        """
        try:
            self.classrooms = data_dict['classrooms.csv']
            self.courses = data_dict['courses.csv']
            self.instructors = data_dict['instructors.csv']
            self.schedule = data_dict['schedule.csv']
            self.students = data_dict['students.csv']
            self.timeslots = data_dict['timeslots.csv']
            
            # Convert timeslot times to datetime.time objects for proper comparison
            if 'start_time' in self.timeslots.columns and 'end_time' in self.timeslots.columns:
                self.timeslots['start_time'] = pd.to_datetime(self.timeslots['start_time'], format='%H:%M').dt.time
                self.timeslots['end_time'] = pd.to_datetime(self.timeslots['end_time'], format='%H:%M').dt.time
            else:
                raise ValueError("Timeslots CSV must contain 'start_time' and 'end_time' columns.")
            return True
        except Exception as e:
            # Propagate the error up to the Streamlit app to display it
            raise ValueError(f"Error loading data: {e}")

    def preprocess_data(self):
        """
        Preprocesses the loaded data by merging various dataframes
        and adding useful derived columns for scheduling.
        Raises:
            ValueError: If essential dataframes are not loaded or required columns are missing.
        """
        # Ensure all dataframes are loaded before preprocessing
        if not all([df is not None for df in [self.schedule, self.courses, self.students,
                                            self.classrooms, self.timeslots, self.instructors]]):
            raise ValueError("All dataframes must be loaded before preprocessing.")

        # Define required columns for each dataframe to ensure data integrity
        required_cols = {
            'schedule': ['course_id', 'student_id', 'classroom_id', 'timeslot_id', 'instructor_id'],
            'courses': ['course_id', 'credits'],
            'students': ['student_id'],
            'classrooms': ['classroom_id', 'capacity'],
            'timeslots': ['timeslot_id', 'start_time', 'end_time'], # start_time and end_time already converted
            'instructors': ['instructor_id']
        }

        # Check for missing required columns in each loaded DataFrame
        for df_name, cols in required_cols.items():
            df = getattr(self, df_name) # Get the dataframe object by its name
            if not all(col in df.columns for col in cols):
                missing_cols = set(cols) - set(df.columns)
                raise ValueError(f"Missing required columns in {df_name}.csv: {', '.join(missing_cols)}")

        # Merge dataframes to create a comprehensive exam data table
        self.exam_data = self.schedule.merge(
            self.courses, on='course_id'
        ).merge(
            self.students, on='student_id'
        ).merge(
            self.classrooms, on='classroom_id'
        ).merge(
            self.timeslots, on='timeslot_id'
        ).merge(
            self.instructors, on='instructor_id'
        )

        # Add derived columns: exam duration and course type
        self.exam_data['exam_duration'] = pd.to_numeric(self.exam_data['credits']) * 30  # 30 minutes per credit hour

        # Process student year if available (optional column)
        if 'year' in self.exam_data.columns:
            year_mapping = {'Freshman': 1, 'Sophomore': 2, 'Junior': 3, 'Senior': 4}
            self.exam_data['student_year'] = self.exam_data['year'].map(year_mapping)

        # Classify courses by student count (size)
        student_counts = self.exam_data.groupby('course_id')['student_id'].nunique()
        self.exam_data['course_size'] = self.exam_data['course_id'].map(student_counts)
        self.exam_data['course_type'] = np.where(
            self.exam_data['course_size'] > 100, 'Large',
            np.where(self.exam_data['course_size'] > 50, 'Medium', 'Small')
        )

    def build_conflict_matrix(self):
        """
        Builds a square conflict matrix where matrix[i][j] represents
        the number of shared students between course i and course j.
        Raises:
            ValueError: If exam_data has not been preprocessed.
        """
        if self.exam_data is None:
            raise ValueError("Exam data must be preprocessed before building conflict matrix.")

        self.unique_courses = self.exam_data['course_id'].unique()
        self.n_courses = len(self.unique_courses)
        # Create a mapping from course ID to its index in the conflict matrix
        self.course_to_idx = {course: idx for idx, course in enumerate(self.unique_courses)}

        self.conflict_matrix = np.zeros((self.n_courses, self.n_courses), dtype=int)

        # Map students to the set of courses they are enrolled in
        student_courses = defaultdict(set)
        for student, group in self.exam_data.groupby('student_id'):
            student_courses[student] = set(group['course_id'])

        # Populate the conflict matrix based on shared students
        for student, courses in student_courses.items():
            courses_list = list(courses)
            for i in range(len(courses_list)):
                for j in range(i + 1, len(courses_list)):
                    idx1 = self.course_to_idx[courses_list[i]]
                    idx2 = self.course_to_idx[courses_list[j]]
                    self.conflict_matrix[idx1][idx2] += 1
                    self.conflict_matrix[idx2][idx1] += 1 # Symmetric matrix

    def setup_genetic_algorithm(self):
        """
        Initializes the DEAP genetic algorithm components:
        fitness, individual, population, evaluation, crossover, mutation, and selection.
        Raises:
            ValueError: If conflict matrix has not been built.
        """
        if self.conflict_matrix is None:
            raise ValueError("Conflict matrix must be built before setting up GA.")
        
        # Create DEAP types only once to avoid re-creation errors in Streamlit
        if not hasattr(creator, "FitnessMin"):
            creator.create("FitnessMin", base.Fitness, weights=(-1.0, -0.5, -0.3, -0.2)) # Minimize all objectives
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMin)

        self.toolbox = base.Toolbox()

        # Define how to create a single individual (a schedule for all courses)
        def create_individual():
            individual = []
            all_rooms = self.classrooms['classroom_id'].tolist()
            # Identify large rooms for large courses
            big_rooms = self.classrooms[self.classrooms['capacity'] >= 100]['classroom_id'].tolist()
            all_timeslots = self.timeslots['timeslot_id'].tolist()

            for course_id in self.unique_courses:
                course_data_rows = self.exam_data[self.exam_data['course_id'] == course_id]
                course_data = course_data_rows.iloc[0] if not course_data_rows.empty else {'course_type': 'Small'}

                # Assign classroom: prefer big rooms for large courses
                room = random.choice(big_rooms) if course_data['course_type'] == 'Large' and big_rooms else random.choice(all_rooms)
                timeslot = random.choice(all_timeslots)
                individual.append((room, timeslot))
            return creator.Individual(individual)

        # Register GA components with the toolbox
        self.toolbox.register("individual", tools.initIterate, creator.Individual, create_individual)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("evaluate", self.evaluate_schedule)
        self.toolbox.register("mate", self.custom_crossover)
        self.toolbox.register("mutate", self.custom_mutation)
        self.toolbox.register("select", tools.selNSGA2) # NSGA-II for multi-objective optimization

    def custom_crossover(self, ind1, ind2):
        """
        Performs a two-point crossover on two individuals (schedules).
        Args:
            ind1 (deap.creator.Individual): The first individual.
            ind2 (deap.creator.Individual): The second individual.
        Returns:
            tuple: A tuple containing the two altered individuals.
        """
        tools.cxTwoPoint(ind1, ind2)
        return ind1, ind2

    def custom_mutation(self, individual):
        """
        Mutates an individual (schedule) by randomly changing the classroom or timeslot
        for some courses.
        Args:
            individual (deap.creator.Individual): The individual to be mutated.
        Returns:
            tuple: A tuple containing the mutated individual.
        """
        all_rooms = self.classrooms['classroom_id'].tolist()
        big_rooms = self.classrooms[self.classrooms['capacity'] >= 100]['classroom_id'].tolist()
        all_timeslots = self.timeslots['timeslot_id'].tolist()

        for i in range(len(individual)):
            if random.random() < 0.1: # 10% chance to mutate each course's assignment
                course_id = self.unique_courses[i]
                course_data_rows = self.exam_data[self.exam_data['course_id'] == course_id]
                course_data = course_data_rows.iloc[0] if not course_data_rows.empty else {'course_type': 'Small'}

                # Mutate classroom, respecting large course preference
                new_room = random.choice(big_rooms) if course_data['course_type'] == 'Large' and big_rooms else random.choice(all_rooms)
                # Mutate timeslot
                new_timeslot = random.choice(all_timeslots)
                individual[i] = (new_room, new_timeslot)
        return individual,

    def evaluate_schedule(self, individual):
        """
        Evaluates the fitness of a given schedule (individual).
        It calculates multiple objectives: student conflicts, classroom usage balance,
        large course distribution penalty, and classroom capacity utilization penalty.
        Args:
            individual (deap.creator.Individual): The schedule to evaluate.
        Returns:
            tuple: A tuple of fitness values (conflicts, room_balance, large_courses_penalty, capacity_penalty).
        """
        conflicts = 0
        # Group courses by (room, timeslot) to identify concurrent exams
        schedule = defaultdict(list) # Key: (room, timeslot), Value: list of course indices
        for i, (room, timeslot) in enumerate(individual):
            schedule[(room, timeslot)].append(i)

        # HARD CONSTRAINT: No two exams in the same room at the same time
        for courses_in_slot_indices in schedule.values():
            if len(courses_in_slot_indices) > 1:
                # Assign a very high penalty if multiple exams are assigned to the same room/timeslot
                conflicts += (len(courses_in_slot_indices) - 1) * 1000 

            # SOFT CONSTRAINT: Conflicts due to shared students
            for i in range(len(courses_in_slot_indices)):
                for j in range(i + 1, len(courses_in_slot_indices)):
                    # Add penalty based on number of shared students between conflicting courses
                    conflicts += self.conflict_matrix[courses_in_slot_indices[i]][courses_in_slot_indices[j]] * 10 

        # SOFT CONSTRAINT: Classroom usage balance (minimize standard deviation of usage)
        room_usage = defaultdict(int)
        for (room, _), courses in schedule.items():
            room_usage[room] += len(courses)
        
        # Ensure all classrooms are considered, even if unused, to correctly calculate balance
        all_classroom_ids = self.classrooms['classroom_id'].tolist()
        for room_id in all_classroom_ids:
            room_usage.setdefault(room_id, 0) 
        
        room_balance = np.std(list(room_usage.values())) if room_usage else 0

        # SOFT CONSTRAINT: Large courses distribution (avoid too many large courses in one timeslot)
        large_courses_penalty = 0
        large_courses_ids = self.exam_data[self.exam_data['course_type'] == 'Large']['course_id'].unique()
        # Get indices of large courses in the individual
        large_indices = [self.course_to_idx[course] for course in large_courses_ids if course in self.course_to_idx]

        large_distribution_by_timeslot = defaultdict(int)
        for idx in large_indices:
            if idx < len(individual): # Ensure index is valid within the current individual
                _, timeslot = individual[idx]
                large_distribution_by_timeslot[timeslot] += 1

        # Penalize if more than one large course is scheduled in the same timeslot
        for count in large_distribution_by_timeslot.values():
            if count > 1:
                large_courses_penalty += (count - 1) * 50

        # SOFT CONSTRAINT: Classroom capacity utilization (penalize exceeding capacity)
        capacity_penalty = 0
        for (room, timeslot), courses_indices in schedule.items():
            room_capacity = self.classrooms[self.classrooms['classroom_id'] == room]['capacity'].values[0]
            
            total_students_in_slot = 0
            for course_idx in courses_indices:
                course_id = self.unique_courses[course_idx]
                # Get unique student count for this specific course
                num_students_for_course = self.exam_data[self.exam_data['course_id'] == course_id]['student_id'].nunique()
                total_students_in_slot += num_students_for_course
            
            # Penalize if total students assigned to a room exceed its capacity
            if total_students_in_slot > room_capacity:
                capacity_penalty += (total_students_in_slot - room_capacity) * 20

        return conflicts, room_balance, large_courses_penalty, capacity_penalty

    def optimize(self, ngen=100, pop_size=50, cxpb=0.7, mutpb=0.2):
        """
        Runs the genetic algorithm optimization process.
        Args:
            ngen (int): Number of generations.
            pop_size (int): Population size.
            cxpb (float): Crossover probability.
            mutpb (float): Mutation probability.
        Returns:
            tuple: The best individual (schedule) found and the statistics log.
        """
        pop = self.toolbox.population(n=pop_size)
        hof = tools.HallOfFame(1) # Stores the best individual found
        stats = tools.Statistics(lambda ind: ind.fitness.values) # Collects fitness statistics
        stats.register("avg", np.mean)
        stats.register("min", np.min)
        stats.register("max", np.max)

        # Run the simple genetic algorithm
        pop, log = algorithms.eaSimple(
            pop, self.toolbox, cxpb=cxpb, mutpb=mutpb, ngen=ngen,
            stats=stats, halloffame=hof, verbose=False # verbose=False to suppress console output
        )
        return hof[0], log # Return the best individual and the statistics log

    def generate_final_schedule(self, best_ind):
        """
        Converts the best individual (optimized schedule) into a human-readable
        pandas DataFrame.
        Args:
            best_ind (deap.creator.Individual): The best individual (schedule) found by the GA.
        Returns:
            pandas.DataFrame: A DataFrame representing the final optimized exam schedule.
        """
        schedule_df = pd.DataFrame({
            'course_id': self.unique_courses,
            'assigned_room': [room for room, _ in best_ind],
            'assigned_timeslot': [timeslot for _, timeslot in best_ind]
        })

        # Merge with original data to get more details about courses, timeslots, and classrooms
        final_schedule = schedule_df.merge(
            self.courses, on='course_id'
        ).merge(
            self.timeslots, left_on='assigned_timeslot', right_on='timeslot_id'
        ).merge(
            self.classrooms, left_on='assigned_room', right_on='classroom_id'
        )

        # Add student counts per course to the final schedule
        student_counts = self.exam_data.groupby('course_id')['student_id'].nunique()
        final_schedule['num_students'] = final_schedule['course_id'].map(student_counts)

        # Define desired column order for display
        columns_order = [
            'course_id', 'course_name', 'department', 'credits',
            'day', 'start_time', 'end_time', 'building_name', 'room_number',
            'capacity', 'num_students', 'room_type'
        ]

        # Filter for available columns and sort the schedule
        available_columns = [col for col in columns_order if col in final_schedule.columns]
        return final_schedule[available_columns].sort_values(['day', 'start_time'])

    def visualize_results(self, final_schedule):
        """
        Generates matplotlib figures for visualizing the schedule distribution.
        Args:
            final_schedule (pandas.DataFrame): The final optimized schedule.
        Returns:
            tuple: A tuple of matplotlib Figure objects (fig_day_dist, fig_building_dist, fig_student_dist).
        """
        if final_schedule.empty:
            return None, None, None # Return None if no schedule to plot

        # Plot 1: Course Distribution by Day
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        if 'day' in final_schedule.columns:
            day_counts = final_schedule['day'].value_counts().sort_index()
            sns.barplot(x=day_counts.index, y=day_counts.values, ax=ax1, palette='viridis')
            ax1.set_title("Course Distribution by Day")
            ax1.set_ylabel("Number of Courses")
        plt.close(fig1) # Close the figure to prevent it from displaying automatically

        # Plot 2: Course Distribution by Building (Classroom Usage)
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        if 'building_name' in final_schedule.columns:
            room_usage = final_schedule.groupby('building_name').size().sort_values(ascending=False)
            sns.barplot(x=room_usage.values, y=room_usage.index, ax=ax2, palette='magma')
            ax2.set_title("Course Distribution by Building")
            ax2.set_xlabel("Number of Courses")
        plt.close(fig2)

        # Plot 3: Student Distribution by Day
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        if 'day' in final_schedule.columns and 'num_students' in final_schedule.columns:
            student_dist = final_schedule.groupby('day')['num_students'].sum().sort_index()
            sns.barplot(x=student_dist.index, y=student_dist.values, ax=ax3, palette='cividis')
            ax3.set_title("Student Distribution by Day")
            ax3.set_ylabel("Number of Students")
        plt.close(fig3)
        
        return fig1, fig2, fig3

    def run_optimization(self, ngen, pop_size, cxpb, mutpb):
        """
        Executes the entire optimization workflow: data loading, preprocessing,
        conflict matrix building, GA setup, and optimization run.
        It returns all results (status, messages, schedule, plots, log) for Streamlit display.
        Args:
            ngen (int): Number of generations for GA.
            pop_size (int): Population size for GA.
            cxpb (float): Crossover probability for GA.
            mutpb (float): Mutation probability for GA.
        Returns:
            tuple: (status_str, messages_str, final_schedule_df, fig1, fig2, fig3, log_obj)
                   status_str: "Success" or "Error"
                   messages_str: A string containing status/error messages
                   final_schedule_df: The optimized schedule DataFrame or empty DataFrame on error
                   fig1, fig2, fig3: Matplotlib Figure objects or None
                   log_obj: DEAP statistics log or None
        """
        status_messages = [] # List to collect messages for display

        status_messages.append("Starting exam schedule optimization...")
        # Display initial stats if data is loaded
        status_messages.append(f"Number of courses: {len(self.unique_courses) if self.unique_courses is not None else 'N/A'}")
        status_messages.append(f"Number of students: {len(self.students['student_id'].unique()) if self.students is not None else 'N/A'}")
        status_messages.append(f"Number of classrooms: {len(self.classrooms) if self.classrooms is not None else 'N/A'}")
        status_messages.append(f"Number of timeslots: {len(self.timeslots) if self.timeslots is not None else 'N/A'}")

        # Check for large classrooms and issue a warning if none exist
        big_rooms = self.classrooms[self.classrooms['capacity'] >= 100] if self.classrooms is not None else pd.DataFrame()
        if len(big_rooms) == 0:
            status_messages.append("Warning: No large classrooms (capacity ≥ 100) found in data.")

        try:
            # Run the genetic algorithm optimization
            best_ind, log = self.optimize(ngen=ngen, pop_size=pop_size, cxpb=cxpb, mutpb=mutpb)
            # Generate the final readable schedule from the best individual
            final_schedule = self.generate_final_schedule(best_ind)

            status_messages.append("Optimization completed successfully!")
            # Display the fitness of the best schedule
            if hasattr(best_ind.fitness, 'values'):
                status_messages.append(f"Best schedule has {best_ind.fitness.values[0]:.0f} student conflicts.")

            # Generate visualization figures
            fig_day_dist, fig_building_dist, fig_student_dist = self.visualize_results(final_schedule)

            return "Success", "\n".join(status_messages), final_schedule, fig_day_dist, fig_building_dist, fig_student_dist, log
        except Exception as e:
            # Catch any error during the optimization process and return error status
            status_messages.append(f"Error during optimization: {e}")
            return "Error", "\n".join(status_messages), pd.DataFrame(), None, None, None, None


# --- Streamlit Application ---
# Configure the Streamlit page settings
st.set_page_config(
    page_title="Exam Scheduling with Genetic Algorithm", # Title shown in the browser tab
    page_icon="🗓️", # Icon for the browser tab
    layout="wide", # Use a wide layout for better content display
    initial_sidebar_state="expanded" # Sidebar is expanded by default
)

# Main title and description of the application
st.title("📚 Exam Scheduling using Genetic Algorithm")
st.markdown("Upload your university data files to generate an optimized exam schedule.")

# --- File Upload Section (in the sidebar) ---
st.sidebar.header("Upload Data Files")
uploaded_files = {}
# List of all required CSV files
required_files = ["classrooms.csv", "courses.csv", "instructors.csv", "schedule.csv", "students.csv", "timeslots.csv"]

# Create a file uploader for each required file
for file_name in required_files:
    uploaded_file = st.sidebar.file_uploader(f"Upload {file_name}", type=['csv'], key=file_name)
    if uploaded_file:
        uploaded_files[file_name] = pd.read_csv(uploaded_file)
    else:
        uploaded_files[file_name] = None # Keep as None if file is not uploaded

# Check if all required files have been uploaded
all_files_uploaded = all(uploaded_files[f] is not None for f in required_files)

# --- Session State Initialization ---
# Initialize session state variables to persist data across Streamlit reruns
if 'scheduler' not in st.session_state:
    st.session_state.scheduler = ExamScheduler()

if 'final_schedule' not in st.session_state:
    st.session_state.final_schedule = pd.DataFrame()

if 'optimization_log' not in st.session_state:
    st.session_state.optimization_log = None

# --- Optimization Parameters (in the sidebar) ---
st.sidebar.header("Genetic Algorithm Parameters")
num_generations = st.sidebar.slider("Number of Generations (ngen)", 50, 500, 100, 50)
population_size = st.sidebar.slider("Population Size (pop_size)", 20, 200, 50, 10)
crossover_prob = st.sidebar.slider("Crossover Probability (cxpb)", 0.5, 1.0, 0.7, 0.05)
mutation_prob = st.sidebar.slider("Mutation Probability (mutpb)", 0.05, 0.5, 0.2, 0.05)


# --- Main Application Logic ---
st.header("Run Optimization")

# Display a warning if not all files are uploaded
if not all_files_uploaded:
    st.warning("Please upload all required CSV files in the sidebar to start the optimization.")
else:
    # Button to trigger the optimization process
    if st.button("Start Exam Scheduling", key="run_optimization_button"):
        # Show a spinner while the optimization is running
        with st.spinner("Loading data, preprocessing, and running optimization... This might take a while."):
            try:
                # Re-initialize the scheduler to ensure a clean state for a new run
                st.session_state.scheduler = ExamScheduler() 
                # Load data from uploaded files into the scheduler
                if st.session_state.scheduler.load_data(uploaded_files):
                    # Perform preprocessing, build conflict matrix, and set up GA
                    st.session_state.scheduler.preprocess_data()
                    st.session_state.scheduler.build_conflict_matrix()
                    st.session_state.scheduler.setup_genetic_algorithm()

                    # Run the optimization and get results
                    status, messages, final_schedule_df, fig1, fig2, fig3, log = \
                        st.session_state.scheduler.run_optimization(
                            ngen=num_generations,
                            pop_size=population_size,
                            cxpb=crossover_prob,
                            mutpb=mutation_prob
                        )

                    if status == "Success":
                        st.success("Scheduling completed successfully!")
                        st.write(messages) # Display status messages
                        st.session_state.final_schedule = final_schedule_df # Store the final schedule
                        st.session_state.optimization_log = log # Store the optimization log

                        # Display generated plots
                        st.subheader("Analysis Results")
                        if fig1:
                            st.pyplot(fig1)
                        if fig2:
                            st.pyplot(fig2)
                        if fig3:
                            st.pyplot(fig3)
                        
                        # Display optimization progress plot if log exists
                        if log:
                            gen = log.select("gen") # Generations
                            min_fitness = log.select("min") # Minimum fitness values
                            avg_fitness = log.select("avg") # Average fitness values

                            fig_progress, ax_progress = plt.subplots(figsize=(10, 6))
                            ax_progress.plot(gen, min_fitness, label="Minimum Fitness")
                            ax_progress.plot(gen, avg_fitness, label="Average Fitness")
                            ax_progress.set_xlabel("Generation")
                            ax_progress.set_ylabel("Fitness Value")
                            ax_progress.set_title("Optimization Progress (Fitness over Generations)")
                            ax_progress.legend()
                            st.pyplot(fig_progress)
                            plt.close(fig_progress) # Close the figure to free memory

                    else:
                        st.error("An error occurred during scheduling.")
                        st.error(messages) # Display error messages
                        st.session_state.final_schedule = pd.DataFrame() # Clear previous schedule
                        st.session_state.optimization_log = None

                else:
                    st.error("Data loading failed. Please check your files.")
                    st.session_state.final_schedule = pd.DataFrame()
                    st.session_state.optimization_log = None

            except Exception as e:
                # Catch any unexpected errors during the overall process
                st.error(f"An unexpected error occurred: {e}")
                st.session_state.final_schedule = pd.DataFrame()
                st.session_state.optimization_log = None

# --- Display Final Schedule ---
# Only display the schedule and download button if a schedule exists
if not st.session_state.final_schedule.empty:
    st.subheader("Optimized Exam Schedule")
    st.dataframe(st.session_state.final_schedule) # Display the schedule DataFrame

    # Provide a download button for the generated schedule
    csv_output = st.session_state.final_schedule.to_csv(index=False, encoding='utf-8-sig')
    st.download_button(
        label="Download Optimized Schedule (CSV)",
        data=csv_output,
        file_name=f"optimized_schedule_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
    )
else:
    st.info("No schedule generated yet. Please upload files and run the optimization.")

st.markdown("---")
st.markdown("This application was developed using Genetic Algorithms and Streamlit.")