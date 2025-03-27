from flask import Flask, render_template, request, redirect, flash, session, url_for, jsonify
import joblib
import numpy as np
from flask_mysqldb import MySQL
from MySQLdb.cursors import DictCursor
import MySQLdb
import os
import pandas as pd
from werkzeug.security import generate_password_hash
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


app = Flask(__name__)

app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'manas'

# Load and preprocess the dataset
def load_and_prepare_data(file_path):
    data = pd.read_csv(file_path)
    data['combinedText'] = data['questionTitle'] + " " + data['questionText'] + " " + data['topic']
    return data

# Train the TF-IDF model
def train_chatbot(data):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(data['combinedText'])
    return vectorizer, tfidf_matrix

# Generate a response
def get_response(user_query, data, vectorizer, tfidf_matrix):
    user_vector = vectorizer.transform([user_query])
    similarity_scores = cosine_similarity(user_vector, tfidf_matrix)
    best_match_idx = similarity_scores.argmax()
    best_match_score = similarity_scores.max()

    if best_match_score > 0.1:  # Threshold for relevance
        question = data.loc[best_match_idx, 'questionTitle']
        answer = data.loc[best_match_idx, 'answerText']
        topic = data.loc[best_match_idx, 'topic']
        return {
            "topic": topic,
            "question": question,
            "answer": answer
        }
    else:
        return {
            "error": "I'm sorry, I couldn't find a relevant response. Could you rephrase your query?"
        }
# File paths for models and encoders
fomo_model_filename = 'fomo_new/fomo_model.pkl'
fomo_encoder_filename = 'fomo_new/label_encoder.pkl'



adhd_model_filename = r'adhd_new\adhd_impulsivity_model.pkl'
adhd_encoder_filename = r'adhd_new\label_encoder.pkl'


anxiety_model_filename = r'dass_new\anxiety\anxiety_model.pkl'
anxiety_target_encoder_filename = r'dass_new\anxiety\target_encoder.pkl'

stress_model_filename = r'dass_new\stress\stress_model.pkl'
stress_target_encoder_filename = r'dass_new\stress\target_encoder.pkl'

depression_model_filename = r'dass_new\depression\depression_model.pkl'
depression_target_encoder_filename = r'dass_new\depression\target_encoder.pkl'

@app.route('/')
def home():
   return render_template('index.html')

@app.route('/home')
def homepage():
    return render_template('index.html')

file_path = '20200325_counsel_chat.csv'
data = load_and_prepare_data(file_path)
vectorizer, tfidf_matrix = train_chatbot(data)

@app.route('/chat')
def home2():
    return render_template('chat.html')

@app.route('/get_response', methods=['POST'])
def chat():
    user_query = request.json.get('query', '')
    if not user_query:
        return jsonify({"error": "Query cannot be empty."})

    response = get_response(user_query, data, vectorizer, tfidf_matrix)
    return jsonify(response)


@app.route('/parent')
def servicepage2():
    return render_template('parent_login.html')


@app.route('/view_high_severity')
def view_high_severity():
    return render_template('view_high_severity.html')


@app.route('/high_adhd')
def high_adhd():
    if 'admin_logged_in' not in session:
        flash("Please log in as admin to access this page.", "danger")
        return redirect('/admin_login')

    cursor = mysql.connection.cursor(DictCursor)
    cursor.execute("SELECT  student_name, student_id, class, section, ADHD FROM student where ADHD='High Impulsivity/Restlessness'")
    students = cursor.fetchall()
    cursor.close()

    return render_template('high_adhd.html', students=students)

@app.route('/high_depression')
def high_depression():
    if 'admin_logged_in' not in session:
        flash("Please log in as admin to access this page.", "danger")
        return redirect('/admin_login')

    cursor = mysql.connection.cursor(DictCursor)
    cursor.execute("SELECT  student_name, student_id, class, section, depression FROM student where depression='depressed'")
    students = cursor.fetchall()
    cursor.close()

    return render_template('high_depression.html', students=students)


@app.route('/high_anxiety')
def high_anxiety():
    if 'admin_logged_in' not in session:
        flash("Please log in as admin to access this page.", "danger")
        return redirect('/admin_login')

    cursor = mysql.connection.cursor(DictCursor)
    cursor.execute("SELECT  student_name, student_id, class, section, anxiety FROM student where anxiety=' Anxiety'")
    students = cursor.fetchall()
    cursor.close()

    return render_template('high_anxiety.html', students=students)

@app.route('/high_stress')
def high_stress():
    if 'admin_logged_in' not in session:
        flash("Please log in as admin to access this page.", "danger")
        return redirect('/admin_login')

    cursor = mysql.connection.cursor(DictCursor)
    cursor.execute("SELECT  student_name, student_id, class, section, stress FROM student where stress='Stressed'")
    students = cursor.fetchall()
    cursor.close()

    return render_template('high_stress.html', students=students)

@app.route('/high_fomo')
def high_fomo():
    if 'admin_logged_in' not in session:
        flash("Please log in as admin to access this page.", "danger")
        return redirect('/admin_login')

    cursor = mysql.connection.cursor(DictCursor)
    cursor.execute("SELECT  student_name, student_id, class, section, FOMO FROM student where FOMO='High FOMO'")
    students = cursor.fetchall()
    cursor.close()

    return render_template('high_FOMO.html', students=students)

@app.route('/parentreg')
def parreg():
    return render_template('parent_register.html')

@app.route('/adminreg')
def admreg():
    return render_template('admin_register.html')

@app.route('/studentreg')
def stureg():
    return render_template('student_register.html')

@app.route('/about')
def aboutpage():
    return render_template('about.html')


@app.route('/services')
def servicespage():
    return render_template('services.html')

@app.route('/contact')
def contactpage():
    return render_template('contact.html')

# Initialize MySQL
mysql = MySQL(app)
@app.route('/register', methods=['POST','GET'])
def register():
    if request.method == 'POST':
        student_name = request.form['student-name']
        student_id = request.form['student-id']
        class_name = request.form['class']
        section = request.form['section']
        age = request.form['age']
        weight = request.form['weight']
        contact = request.form['contact']
        password = request.form['password']
        confirm_password = request.form['confirm-password']

        # Check if passwords match
        if password != confirm_password:
            flash('Passwords do not match', 'error')
            return redirect('/register')

        # Database operation
        try:
            cur = mysql.connection.cursor()
            cur.execute('''
                INSERT INTO student (student_name, student_id, class, section, age, weight_kg, contact_number, password, confirm_password)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ''', (student_name, student_id, class_name, section, age, weight, contact, password, confirm_password))
            mysql.connection.commit()
            cur.close()

            flash('Student registered successfully!', 'success')
            return redirect('/login')

        except MySQLdb.IntegrityError as e:
            if e.args[0] == 1062:  # Error code for duplicate entry
                flash(f"Student ID '{student_id}' is already registered. Please use a different ID.", 'error')
            else:
                flash('An error occurred while registering the student. Please try again.', 'error')
            return redirect('/register')

    return render_template('student_register.html')
    
@app.route('/register_parent', methods=['GET', 'POST'])
def register_parent():
    if request.method == 'POST':
        # Get form data
        parent_name = request.form['parent_name']
        student_id = request.form['student_id']
        contact_number = request.form['contact_number']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        if password != confirm_password:
            flash('Passwords do not match!', 'danger')
            return redirect('/register_parent')

        # Check if the student is registered
        cursor = mysql.connection.cursor()
        cursor.execute("SELECT * FROM student WHERE student_id = %s", (student_id,))
        student = cursor.fetchone()

        if not student:
            flash('Student ID not found. Register the student first.', 'danger')
            return redirect('/register_parent')

        cursor.execute("""
    INSERT INTO parent (parent_name, student_id, contact_number, email, password, confirm_password)
    VALUES (%s, %s, %s, %s, %s, %s)
""", (parent_name, student_id, contact_number, email, password, confirm_password))

        mysql.connection.commit()
        cursor.close()

        flash('Parent registration successful!', 'success')
        return redirect('/parent_login')

    return render_template('parent_register.html')

        
# Admin login route
@app.route('/admin_login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        if username == 'admin' and password == 'admin123':  # Hardcoded credentials
            session['admin_logged_in'] = True
            flash('Admin login successful!', 'success')
            return redirect('/admin_dashboard')
        else:
            flash('Invalid credentials!', 'danger')
            return redirect('/admin_login')

    return render_template('admin_login.html')

# Admin dashboard with student lookup
@app.route('/admin_dashboard', methods=['GET', 'POST'])
def admin_dashboard():
    if 'admin_logged_in' not in session:
        flash("Please log in as admin to access the dashboard.", "danger")
        return redirect('/admin_login')
    
    student = None

    if request.method == 'POST':
        student_id = request.form['student_id']
        cursor = mysql.connection.cursor(DictCursor)
        cursor.execute("SELECT * FROM student WHERE student_id = %s", (student_id,))
        student = cursor.fetchone()
        cursor.close()

        if not student:
            flash(f"No student found with ID {student_id}", "danger")

    return render_template('admin_dashboard.html', student=student)

# View all students in a table format
@app.route('/view_all_students', methods=['GET'])
def view_all_students():
    if 'admin_logged_in' not in session:
        flash("Please log in as admin to access this page.", "danger")
        return redirect('/admin_login')

    cursor = mysql.connection.cursor(DictCursor)
    cursor.execute("SELECT  student_name, student_id, class, section, age, weight_kg, contact_number, FOMO, ADHD, feedback, stress, depression, anxiety FROM student")
    students = cursor.fetchall()
    cursor.close()

    return render_template('view_all_students.html', students=students)


@app.route('/update_feedback', methods=['GET', 'POST'])
def update_feedback():
    if 'admin_logged_in' not in session:
        flash("Please log in as admin to access this page.", "danger")
        return redirect('/admin_login')

    if request.method == 'POST':
        student_id = request.form['student_id']
        feedback = request.form['feedback']

        # Update the feedback in the database
        cursor = mysql.connection.cursor()
        cursor.execute("UPDATE student SET feedback = %s WHERE student_id = %s", (feedback, student_id))
        mysql.connection.commit()
        cursor.close()

        flash('Feedback updated successfully!', 'success')
        return redirect('/admin_dashboard')

    # Assuming you want to display student details when the route is accessed by GET
    student_id = request.args.get('student_id')
    cursor = mysql.connection.cursor()
    cursor.execute("SELECT * FROM student WHERE student_id = %s", (student_id,))
    student = cursor.fetchone()
    cursor.close()

    return render_template('view_student.html', student=student)

@app.route('/view_student', methods=['GET', 'POST'])
def view_student():
    if 'admin_logged_in' not in session:
        flash("Please log in as admin to access this page.", "danger")
        return redirect('/admin_login')

    student = None

    if request.method == 'POST':
        student_id = request.form.get('student_id', '').strip()

        if not student_id:
            flash("Student ID is required!", "danger")
            return redirect('/view_student')  # Redirect to the same page

        cursor = mysql.connection.cursor(DictCursor)
        cursor.execute("""
            SELECT  student_name, student_id, class, section, age, weight_kg, contact_number, 
                   FOMO, ADHD, feedback, stress, depression, anxiety
            FROM student
            WHERE student_id = %s
        """, (student_id,))
        student = cursor.fetchone()
        cursor.close()

        if not student:
            flash("No student found with the provided ID.", "danger")
            return redirect('/view_student')  # Redirect to the same page

    return render_template('view_student.html', student=student)


@app.route('/parent_login', methods=['GET', 'POST'])
def parent_login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        # Validate parent credentials
        cursor = mysql.connection.cursor(DictCursor)
        cursor.execute("SELECT * FROM parent WHERE email = %s AND password = %s", (email, password))
        parent = cursor.fetchone()
        cursor.close()

        if parent:
            session['parent_logged_in'] = True
            session['student_id'] = parent['student_id']
            flash("Login successful!", "success")
            return redirect('/parent_dashboard')
        else:
            flash("Invalid email or password!", "danger")
            return redirect('/parent_login')

    return render_template('parent_login.html')

# Parent dashboard to view student details
@app.route('/parent_dashboard', methods=['GET'])
def parent_dashboard():
    if 'parent_logged_in' not in session:
        flash("Please log in to access the dashboard.", "danger")
        return redirect('/parent_login')

    student_id = session.get('student_id')
    
    # Fetch student details based on student_id
    cursor = mysql.connection.cursor(DictCursor)
    cursor.execute("""
        SELECT student_name, student_id, class, section, age, weight_kg, contact_number, created_at, FOMO, ADHD, feedback, stress, depression, anxiety
        FROM student
        WHERE student_id = %s
    """, (student_id,))
    student = cursor.fetchone()
    cursor.close()

    if not student:
        flash("Student details not found.", "danger")
        return redirect('/parent_login')

    return render_template('parent_dashboard.html', student=student)


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Database connection
        cursor = mysql.connection.cursor(DictCursor)

        # Verify credentials
        query = """
        SELECT student_id FROM student
        WHERE student_name = %s AND password = %s
        """
        cursor.execute(query, (username, password))
        result = cursor.fetchone()

        cursor.close()
        

        if result:
            # Start session and redirect to home
            session['student_id'] = result['student_id']
            return redirect(url_for('home1'))
        else:
            flash("Invalid username or password. Please try again.", "danger")
            return redirect(url_for('login'))  # Redirect to login page after flash message

    return render_template('login.html')


@app.route('/home1')
def home1():
    if 'student_id' in session:
        return render_template('home.html')
    return redirect(url_for('login'))

# Route for the FoMO questionnaire page
@app.route('/predictinfo')
def predictinfo():
    return render_template('fomo.html')

# Route to handle form submission for FoMO questionnaire
@app.route('/submit', methods=['POST'])
def submit():
    # Load the FoMO model and label encoder
    model = joblib.load(fomo_model_filename)
    label_encoder = joblib.load(fomo_encoder_filename)

    # Retrieve answers to FoMO questions (q1 to q10)
    answers = []
    for i in range(1, 11):
        answers.append(request.form.get(f'q{i}'))

    # Convert the answers to a DataFrame to match the model's input column names (Q1, Q2, ..., Q10)
    df_input = pd.DataFrame([answers], columns=[f'Q{i}' for i in range(1, 11)])

    # Make prediction using the trained model
    y_pred_encoded = model.predict(df_input)
    y_pred = label_encoder.inverse_transform(y_pred_encoded)

    # Display the predicted result
    result = f"{y_pred[0]}"

    # Check if the user is logged in (retrieve student_id from session)
    if 'student_id' in session:
        student_id = session['student_id']

        # Update the FOMO column in the database for the logged-in student
        cursor = mysql.connection.cursor()

        try:
            # Debugging: Print query and parameters
            print(f"Executing query: UPDATE student SET FOMO = '{result}' WHERE student_id = '{student_id}'")

            # Update query
            query = """
            UPDATE student
            SET FOMO = %s
            WHERE student_id = %s
            """
            cursor.execute(query, (result, student_id))
            
            # Commit the changes
            mysql.connection.commit()

            # Set success message
            response_message = f"FOMO value '{result}' updated successfully for student ID {student_id}."
        except Exception as e:
            # Handle database errors
            print(f"Error: {e}")  # Log the error
            mysql.connection.rollback()  # Rollback changes on error
            response_message = f"Error updating FOMO value: {e}"
        finally:
            cursor.close()

        # Render home page with alert message
        return render_template('home.html', alert_message=response_message)
    else:
        # If no student session is found
        return render_template('home.html', alert_message="No active session found. Please log in.")


# Directly show the Stress questionnaire page
answer_mapping = {
    "Applied to me very much": 3,
    "Applied to me a considerable degree": 2,
    "Applied to me to some degree": 1,
    "Did not apply to me at all": 0
}

@app.route('/dass_dashboard')
def dass_dashboard():
    return render_template('dass_dashboard.html')

# Route to directly show the Stress questionnaire page
@app.route('/service_stress')
def service_stress():
    return render_template('stress_form.html')

# Route to handle form submission for the Stress questionnaire
@app.route('/submit_stress', methods=['POST'])
def submit_stress():
    try:
        # Load the Stress model and target encoder
        model = joblib.load(stress_model_filename)
        target_encoder = joblib.load(stress_target_encoder_filename)

        # Retrieve answers to Stress questions (q1 to q7)
        answers = []
        for i in range(1, 8):  # From q1 to q7
            answer = request.form.get(f'q{i}')
            if not answer:  # Handle empty answers
                answer = 'Did not apply to me at all'  # Default to 'Did not apply to me at all'
            answers.append(answer)

        # Debug: Check if answers are as expected
        print(f"Answers: {answers}")

        # Convert the answers to numerical values using the mapping
        encoded_answers = [answer_mapping[answer] for answer in answers]

        # Debug: Check encoded answers
        print(f"Encoded Answers: {encoded_answers}")

        # Flatten to 1D array for prediction (model expects 2D input)
        encoded_answers = np.array(encoded_answers).reshape(1, -1)

        # Make prediction using the trained model
        y_pred_encoded = model.predict(encoded_answers)  # The model expects 2D input
        print(f"Prediction Output (Encoded): {y_pred_encoded}")

        # Inverse transform the prediction to get the category (Stress level)
        y_pred = target_encoder.inverse_transform(y_pred_encoded)

        # Final result: Predicted Stress Category
        result = f"{y_pred[0]}"
        print(f"Final Predicted Result: {result}")

        # Check if the student is logged in
        if 'student_id' in session:
            student_id = session['student_id']

            # Update the database for the logged-in student
            cursor = mysql.connection.cursor()

            try:
                # Debugging: Print the query and parameters
                print(f"Executing query: UPDATE student SET Stress = '{result}' WHERE student_id = '{student_id}'")

                # Update the student record with the stress value
                query = """
                UPDATE student
                SET Stress = %s
                WHERE student_id = %s
                """
                cursor.execute(query, (result, student_id))
                
                # Commit the changes to the database
                mysql.connection.commit()

                # Set success message
                alert_message = f"Stress level '{y_pred[0]}' updated successfully for student ID {student_id}."
            except Exception as e:
                # Handle any database errors
                print(f"Database error: {e}")  # Log the error
                mysql.connection.rollback()  # Rollback changes on error
                alert_message = f"Error updating Stress value: {e}"
            finally:
                cursor.close()

            # Render the home page with the alert message
            return render_template('home.html', alert_message=alert_message)
        else:
            # If no student session is found
            alert_message = "No active session found. Please log in."
            return render_template('home.html', alert_message=alert_message)

    except Exception as e:
        # Handle general errors
        print(f"Error in submit_stress: {e}")
        alert_message = f"An error occurred: {e}"
        return render_template('home.html', alert_message=alert_message)



@app.route('/service_anxiety')
def service_anxiety():
    return render_template('anxiety_form.html')


# Route for the Anxiety questionnaire page
@app.route('/submit_anxiety', methods=['POST'])
def submit_anxiety():
    try:
        # Load the Anxiety model and target encoder
        model = joblib.load(anxiety_model_filename)
        target_encoder = joblib.load(anxiety_target_encoder_filename)

        # Retrieve answers to Anxiety questions (q1 to q7)
        answers = []
        for i in range(1, 8):  # From q1 to q7
            answer = request.form.get(f'q{i}')
            if not answer:  # Handle empty answers
                answer = 'Did not apply to me at all'  # Default to 'Did not apply to me at all'
            answers.append(answer)

        # Debug: Check if answers are as expected
        print(f"Answers: {answers}")

        # Convert the answers to numerical values using the mapping
        encoded_answers = [answer_mapping[answer] for answer in answers]

        # Debug: Check encoded answers
        print(f"Encoded Answers: {encoded_answers}")

        # Flatten to 1D array for prediction (model expects 2D input)
        encoded_answers = np.array(encoded_answers).reshape(1, -1)

        # Make prediction using the trained model
        y_pred_encoded = model.predict(encoded_answers)  # The model expects 2D input
        print(f"Prediction Output (Encoded): {y_pred_encoded}")

        # Inverse transform the prediction to get the category (Anxiety level)
        y_pred = target_encoder.inverse_transform(y_pred_encoded)

        # Final result: Predicted Anxiety Category
        result = f" {y_pred[0]}"
        print(f"Final Predicted Result: {result}")

        # Check if the student is logged in
        if 'student_id' in session:
            student_id = session['student_id']

            # Update the database for the logged-in student
            cursor = mysql.connection.cursor()

            try:
                # Debugging: Print the query and parameters
                print(f"Executing query: UPDATE student SET anxiety = '{result}' WHERE student_id = '{student_id}'")

                # Update the student record with the anxiety value
                query = """
                UPDATE student
                SET anxiety = %s
                WHERE student_id = %s
                """
                cursor.execute(query, (result, student_id))
                
                # Commit the changes to the database
                mysql.connection.commit()

                # Set success message
                response_message = f"Anxiety level '{y_pred[0]}' updated successfully for student ID {student_id}."
            except Exception as e:
                # Handle any database errors
                print(f"Database error: {e}")  # Log the error
                mysql.connection.rollback()  # Rollback changes on error
                response_message = f"Error updating anxiety value: {e}"
            finally:
                cursor.close()

            # Store the final result in a text file (append mode)
            with open("anxiety_results.txt", "a") as file:
                file.write(result + "\n")

            # Set the alert message
            alert_message = f"Anxiety value '{y_pred[0]}' updated successfully."

            # Render the home page with the alert message
            return render_template('dass_dashboard.html', alert_message=alert_message)
        else:
            # If no student session is found
            return render_template('dass_dashboard.html', alert_message="No active session found. Please log in.", status_code=403)

    except Exception as e:
        # Log the error message
        print(f"Error in submit_anxiety: {str(e)}")
        alert_message = f"An error occurred: {e}"
        return render_template('dass_dashboard.html', alert_message=alert_message)




# Route for the Depression questionnaire page

@app.route('/service_depression')
def depression():
    return render_template('depression.html')  # Render the Depression questionnaire form

@app.route('/submit_depression', methods=['POST'])
def submit_depression():
    try:
        # Load the Depression model and target encoder
        model = joblib.load(depression_model_filename)
        target_encoder = joblib.load(depression_target_encoder_filename)

        # Retrieve answers to Depression questions (q1 to q7)
        answers = []
        for i in range(1, 8):  # From q1 to q7
            answer = request.form.get(f'q{i}')
            if not answer:  # Handle missing answers
                answer = 'Not Answered'  # Default to 'Not Answered' if missing
            answers.append(answer)

        # Debugging: Print the answers
        print(f"Answers: {answers}")

        # Encode the answers using a predefined response mapping (ensure this aligns with model training)
        response_mapping = {
            "Did not apply to me at all": 0,
            "Applied to me to some degree": 1,
            "Applied to me a considerable degree": 2,
            "Applied to me very much": 3
        }
        
        # Encode the answers
        encoded_answers = [response_mapping.get(answer, -1) for answer in answers]  # Encoding answers

        # Debugging: Print the encoded answers
        print(f"Encoded Answers: {encoded_answers}")

        # Convert answers to a DataFrame (ensure the column names match the model's expected input)
        df_input = pd.DataFrame([encoded_answers], columns=[f'Q{i}' for i in range(1, 8)])

        # Debugging: Print the DataFrame for prediction
        print(f"DataFrame for prediction: {df_input}")

        # Make prediction using the trained model
        y_pred_encoded = model.predict(df_input)
        y_pred = target_encoder.inverse_transform(y_pred_encoded)

        # Final result: Predicted Depression Category
        result = f"{y_pred[0]}"

        # Check if student session exists
        if 'student_id' in session:
            student_id = session['student_id']

            # Update the database for the logged-in student
            cursor = mysql.connection.cursor()

            try:
                # Debugging: Print the query and parameters
                print(f"Executing query: UPDATE student SET depression = '{result}' WHERE student_id = '{student_id}'")

                # Update the student record with the depression value
                query = """
                UPDATE student
                SET depression = %s
                WHERE student_id = %s
                """
                cursor.execute(query, (result, student_id))
                
                # Commit the changes to the database
                mysql.connection.commit()

                # Set success message
                alert_message = f"Depression level '{y_pred[0]}' updated successfully for student ID {student_id}."
            except Exception as e:
                # Handle any database errors
                print(f"Database error: {e}")  # Log the error
                mysql.connection.rollback()  # Rollback changes on error
                alert_message = f"Error updating depression value: {e}"
            finally:
                cursor.close()

            # Store the final result in a text file (append mode)
            with open("depression_results.txt", "a") as file:
                file.write(result + "\n")

            # Render the home page with the alert message
            return render_template('dass_dashboard.html', alert_message=alert_message)
        else:
            # If no student session is found
            return render_template('dass_dashboard.html', alert_message="No active session found. Please log in.", status_code=403)

    except Exception as e:
        # Log the error message
        print(f"Error in submit_depression: {str(e)}")
        alert_message = f"An error occurred: {e}"
        return render_template('home.html', alert_message=alert_message)


@app.route('/currency')
def adhd():
    # Check if the student is logged in
    if 'student_id' not in session:
        return redirect(url_for('login'))  # Redirect to login page if no session found
    return render_template('adhd.html')

# Route to handle form submission for ADHD questionnaire
@app.route('/submit_adhd', methods=['POST'])  # This route will handle the form submission
def submit_adhd():
    # Load the ADHD model and label encoder
    model = joblib.load(adhd_model_filename)
    label_encoder = joblib.load(adhd_encoder_filename)

    # Retrieve answers to ADHD questions (q1 to q18)
    answers = []
    for i in range(1, 19):  # From q1 to q18
        answer = request.form.get(f'q{i}')
        if answer:
            answers.append(int(answer))  # Ensure answers are integers
        else:
            answers.append(0)  # Handle missing answers as 0 (or other default value)

    # Calculate Total_Score (if necessary, depending on your model's expected features)
    total_score = sum(answers)  # Adjust if needed based on how Total_Score is calculated

    # Add Total_Score to the answers
    answers.append(total_score)

    # Check the expected feature names from the trained model
    print("Expected Features:", model.feature_names_in_)

    # Convert the answers to a DataFrame to match the model's input column names
    df_input = pd.DataFrame([answers], columns=[f'Q{i}' for i in range(1, 19)] + ['Total_Score'])

    # Make sure df_input has the exact same columns as model expects
    df_input = df_input[model.feature_names_in_]  # Reorder columns to match

    # Make prediction using the trained model
    y_pred_encoded = model.predict(df_input)
    y_pred = label_encoder.inverse_transform(y_pred_encoded)

    # Final result: Predicted ADHD Category
    result = f" {y_pred[0]}"

    # Store the result in the database if the student is logged in
    if 'student_id' in session:
        student_id = session['student_id']

        cursor = mysql.connection.cursor()

        try:
            # Debugging: Print query and parameters
            print(f"Executing query: UPDATE student SET ADHD = '{result}' WHERE student_id = '{student_id}'")

            # Update query
            query = """
            UPDATE student
            SET ADHD = %s
            WHERE student_id = %s
            """
            cursor.execute(query, (result, student_id))
            
            # Commit the changes
            mysql.connection.commit()

            # Success message
            response_message = f"ADHD value '{result}' updated successfully for student_id {student_id}."
        except Exception as e:
            # Handle database errors
            print(f"Error: {e}")  # Log the error
            mysql.connection.rollback()  # Rollback changes on error
            response_message = f"Error updating ADHD value: {e}"
        finally:
            cursor.close()

        # Store the final result in a text file (append mode)
        with open("adhd_results.txt", "a") as file:
            file.write(result + "\n")

        # Redirect to home page or another form
        return redirect(url_for('home1'))  # Ensure the session is still valid and return to home

    else:
        # If no student session is found
        return "No active session found. Please log in.", 403
    
@app.route('/logout')
def logout():
    # Remove the user from the session to log out
    session.pop('user', None)
    return redirect(url_for('home'))
from flask import render_template, request, redirect, url_for, flash
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

@app.route('/visualisation')
def visualisation():
    return render_template('visualisation.html')

# Route for FOMO visualization
@app.route('/fomo_visualisation', methods=['GET'])
def fomo_visualisation():
    # Ensure you get the data from the database
    cursor = mysql.connection.cursor(DictCursor)
    cursor.execute("SELECT FOMO FROM student")  # Fetch FOMO category data from the student table
    students = cursor.fetchall()
    cursor.close()

    # Count occurrences of each FOMO category
    low_fomo_count = 0
  
    high_fomo_count = 0

    for student in students:
        fomo_category = student['FOMO']  # Assuming FOMO is stored as a string like 'Low FOMO', 'Moderate FOMO', 'High FOMO'
        
        if fomo_category == 'Low FoMO':
            low_fomo_count += 1
  
        elif fomo_category == 'High FoMO':
            high_fomo_count += 1

    # Pie Chart
    pie_labels = ['Low FoMO', 'High FoMO']
    pie_sizes = [low_fomo_count, high_fomo_count]
    pie_colors = ['#99a4ae', '#5b6b7b']

    fig_pie, ax_pie = plt.subplots()
    ax_pie.pie(pie_sizes, labels=pie_labels, colors=pie_colors, autopct='%1.1f%%', startangle=90)
    ax_pie.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    
    pie_img = io.BytesIO()
    fig_pie.savefig(pie_img, format='png')
    pie_img.seek(0)
    pie_plot_url = base64.b64encode(pie_img.getvalue()).decode('utf8')

    # Bar Plot (FOMO distribution across groups)
    categories = ['Low FoMO',  'High FoMO']
    bar_heights = [low_fomo_count,  high_fomo_count]
    
    fig_bar, ax_bar = plt.subplots()
    ax_bar.bar(categories, bar_heights, color=['#99a4ae', '#5b6b7b'])
    ax_bar.set_title('FOMO Distribution')
    ax_bar.set_xlabel('FOMO Levels')
    ax_bar.set_ylabel('Number of Students')

    bar_img = io.BytesIO()
    fig_bar.savefig(bar_img, format='png')
    bar_img.seek(0)
    bar_plot_url = base64.b64encode(bar_img.getvalue()).decode('utf8')

   

    

    # Render the fomo_visualisation.html template and pass the plot URLs
    return render_template('fomo_visualisation.html', 
                           pie_plot_url=pie_plot_url,
                           bar_plot_url=bar_plot_url)




# Route for FOMO visualization
@app.route('/adhd_visualisation', methods=['GET'])
def adhd_visualisation():
    # Ensure you get the data from the database
    cursor = mysql.connection.cursor(DictCursor)
    cursor.execute("SELECT ADHD FROM student")  # Fetch FOMO category data from the student table
    students = cursor.fetchall()
    cursor.close()

    # Count occurrences of each FOMO category
    low_ADHD_count = 0
  
    high_ADHD_count = 0

    for student in students:
        ADHD_category = student['ADHD']  # Assuming FOMO is stored as a string like 'Low FOMO', 'Moderate FOMO', 'High FOMO'
        
        if ADHD_category == 'Low Impulsivity/Restlessness':
            low_ADHD_count += 1
  
        elif ADHD_category == 'High Impulsivity/Restlessness':
            high_ADHD_count += 1

    # Pie Chart
    pie_labels = ['Low Impulsivity/Restlessness', 'High Impulsivity/Restlessness']
    pie_sizes = [low_ADHD_count, high_ADHD_count]
    pie_colors = ['#99a4ae', '#5b6b7b']

    fig_pie, ax_pie = plt.subplots()
    ax_pie.pie(pie_sizes, labels=pie_labels, colors=pie_colors, autopct='%1.1f%%', startangle=90)
    ax_pie.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    
    pie_img = io.BytesIO()
    fig_pie.savefig(pie_img, format='png')
    pie_img.seek(0)
    pie_plot_url = base64.b64encode(pie_img.getvalue()).decode('utf8')

    # Bar Plot (FOMO distribution across groups)
    # Horizontal Bar Plot (ADHD distribution across groups)
    # Bar Plot (ADHD distribution across groups)
    categories = ['Low Impulsivity/Restlessness', 'High Impulsivity/Restlessness']
    bar_heights = [low_ADHD_count, high_ADHD_count]

    fig_bar, ax_bar = plt.subplots()
    ax_bar.bar(categories, bar_heights, color=['#99a4ae', '#5b6b7b'])
    ax_bar.set_title('ADHD Distribution')
    ax_bar.set_xlabel('ADHD Levels')
    ax_bar.set_ylabel('Number of Students')

    bar_img = io.BytesIO()
    fig_bar.savefig(bar_img, format='png')
    bar_img.seek(0)
    bar_plot_url = base64.b64encode(bar_img.getvalue()).decode('utf8')


    # Render the fomo_visualisation.html template and pass the plot URLs
    return render_template('adhd_visualisation.html', 
                           pie_plot_url=pie_plot_url,
                           bar_plot_url=bar_plot_url)



# Route for FOMO visualization
@app.route('/depression_visualisation', methods=['GET'])
def depression_visualisation():
    # Ensure you get the data from the database
    cursor = mysql.connection.cursor(DictCursor)
    cursor.execute("SELECT depression FROM student")  # Fetch FOMO category data from the student table
    students = cursor.fetchall()
    cursor.close()

    # Count occurrences of each FOMO category
    low_depression_count = 0
  
    high_depression_count = 0

    for student in students:
        depression_category = student['depression']  # Assuming FOMO is stored as a string like 'Low FOMO', 'Moderate FOMO', 'High FOMO'
        
        if depression_category == 'Not Depressed':
            low_depression_count += 1
  
        elif depression_category == 'Depressed':
            high_depression_count += 1

    # Pie Chart
    pie_labels = ['Not Depressed', 'Depressed']
    pie_sizes = [low_depression_count, high_depression_count]
    pie_colors =['#99a4ae', '#5b6b7b']

    fig_pie, ax_pie = plt.subplots()
    ax_pie.pie(pie_sizes, labels=pie_labels, colors=pie_colors, autopct='%1.1f%%', startangle=90)
    ax_pie.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    
    pie_img = io.BytesIO()
    fig_pie.savefig(pie_img, format='png')
    pie_img.seek(0)
    pie_plot_url = base64.b64encode(pie_img.getvalue()).decode('utf8')

    # Bar Plot (FOMO distribution across groups)
    categories = ['Not Depressed', 'Depressed']
    bar_heights = [low_depression_count, high_depression_count]
    
    fig_bar, ax_bar = plt.subplots()
    ax_bar.bar(categories, bar_heights, color=['#99a4ae', '#5b6b7b'])
    ax_bar.set_title('Depression Distribution')
    ax_bar.set_xlabel('Depression Levels')
    ax_bar.set_ylabel('Number of Students')

    bar_img = io.BytesIO()
    fig_bar.savefig(bar_img, format='png')
    bar_img.seek(0)
    bar_plot_url = base64.b64encode(bar_img.getvalue()).decode('utf8')
    

    # Render the fomo_visualisation.html template and pass the plot URLs
    return render_template('depression_visualisation.html', 
                           pie_plot_url=pie_plot_url,
                           bar_plot_url=bar_plot_url)




# Route for FOMO visualization
@app.route('/anxiety_visualisation', methods=['GET'])
def anxiety_visualisation():
    # Ensure you get the data from the database
    cursor = mysql.connection.cursor(DictCursor)
    cursor.execute("SELECT anxiety FROM student")  # Fetch FOMO category data from the student table
    students = cursor.fetchall()
    cursor.close()

    # Count occurrences of each FOMO category
    low_anxiety_count = 0
  
    high_anxiety_count = 0

    for student in students:
        anxiety_category = student['anxiety']  # Assuming FOMO is stored as a string like 'Low FOMO', 'Moderate FOMO', 'High FOMO'
        
        if anxiety_category == ' Not Anxiety':
            low_anxiety_count += 1
  
        elif anxiety_category == ' Anxiety':
            high_anxiety_count += 1

    # Pie Chart
    pie_labels = ['Not Anxiety', 'Anxiety']
    pie_sizes = [low_anxiety_count, high_anxiety_count]
    pie_colors = ['#99a4ae', '#5b6b7b']

    fig_pie, ax_pie = plt.subplots()
    ax_pie.pie(pie_sizes, labels=pie_labels, colors=pie_colors, autopct='%1.1f%%', startangle=90)
    ax_pie.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    
    pie_img = io.BytesIO()
    fig_pie.savefig(pie_img, format='png')
    pie_img.seek(0)
    pie_plot_url = base64.b64encode(pie_img.getvalue()).decode('utf8')

    # Bar Plot (FOMO distribution across groups)
    categories = ['Not Anxiety', 'Anxiety']
    bar_heights = [low_anxiety_count, high_anxiety_count]
    
    fig_bar, ax_bar = plt.subplots()
    ax_bar.bar(categories, bar_heights, color=['#99a4ae', '#5b6b7b'])
    ax_bar.set_title('anxiety Distribution')
    ax_bar.set_xlabel('anxiety Levels')
    ax_bar.set_ylabel('Number of Students')

    bar_img = io.BytesIO()
    fig_bar.savefig(bar_img, format='png')
    bar_img.seek(0)
    bar_plot_url = base64.b64encode(bar_img.getvalue()).decode('utf8')
    

    # Render the fomo_visualisation.html template and pass the plot URLs
    return render_template('anxiety_visualisation.html', 
                           pie_plot_url=pie_plot_url,
                           bar_plot_url=bar_plot_url)





# Route for FOMO visualization
@app.route('/stress_visualisation', methods=['GET'])
def stress_visualisation():
    # Ensure you get the data from the database
    cursor = mysql.connection.cursor(DictCursor)
    cursor.execute("SELECT stress FROM student")  # Fetch FOMO category data from the student table
    students = cursor.fetchall()
    cursor.close()

    # Count occurrences of each FOMO category
    low_stress_count = 0
  
    high_stress_count = 0

    for student in students:
        stress_category = student['stress']  # Assuming FOMO is stored as a string like 'Low FOMO', 'Moderate FOMO', 'High FOMO'
        
        if stress_category == 'Not Stressed':
            low_stress_count += 1
  
        elif stress_category == 'Stressed':
            high_stress_count += 1

    # Pie Chart
    pie_labels = ['Not stressed', 'Stressed']
    pie_sizes = [low_stress_count, high_stress_count]
    pie_colors = ['#99a4ae', '#5b6b7b']

    fig_pie, ax_pie = plt.subplots()
    ax_pie.pie(pie_sizes, labels=pie_labels, colors=pie_colors, autopct='%1.1f%%', startangle=90)
    ax_pie.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    
    pie_img = io.BytesIO()
    fig_pie.savefig(pie_img, format='png')
    pie_img.seek(0)
    pie_plot_url = base64.b64encode(pie_img.getvalue()).decode('utf8')

    # Bar Plot (FOMO distribution across groups)
    categories = ['Not stressed', 'Stressed']
    bar_heights = [low_stress_count, high_stress_count]
    
    fig_bar, ax_bar = plt.subplots()
    ax_bar.bar(categories, bar_heights, color=['#99a4ae', '#5b6b7b'])
    ax_bar.set_title('Stress Distribution')
    ax_bar.set_xlabel('Stress Levels')
    ax_bar.set_ylabel('Number of Students')

    bar_img = io.BytesIO()
    fig_bar.savefig(bar_img, format='png')
    bar_img.seek(0)
    bar_plot_url = base64.b64encode(bar_img.getvalue()).decode('utf8')
    

    # Render the fomo_visualisation.html template and pass the plot URLs
    return render_template('stress_visualisation.html', 
                           pie_plot_url=pie_plot_url,
                           bar_plot_url=bar_plot_url)



if __name__ == '__main__':
    app.secret_key = os.urandom(12)
    app.run(debug=True)

