from django.shortcuts import render, redirect
from app.models import Smart
from django.contrib import messages
from django.contrib.auth import logout
from django.http import JsonResponse  # Add this import
import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split 
from xgboost import XGBClassifier 
from sklearn.preprocessing import LabelEncoder 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import google.generativeai as genai
import json
import os

# Configure Gemini API (add your API key in settings or environment variable)
# You can get API key from: https://makersuite.google.com/app/apikey
GEMINI_API_KEY = "AIzaSyDc-tVHVQfM6WUfhBRVkTYfAAVuf-GxWJg"  # Or set in settings.py
genai.configure(api_key=GEMINI_API_KEY)

# Create your views here.
def index(request):
    return render(request, 'index.html')

def about(request):
    return render(request, 'about.html')

def contact(request):
    if request.method == "POST":
        first_name = request.POST.get('first_name')
        last_name = request.POST.get('last_name')
        email = request.POST.get('email')
        phone = request.POST.get('phone')
        message = request.POST.get('comments')
        print(first_name, last_name, email, phone, message)
        messages.success(request, f"Your Message sent Successfully")
    return render(request, 'contact.html')

def register(request):
    if request.method == 'POST':
        name = request.POST.get('name')
        email = request.POST.get('email')
        password = request.POST.get('password')
        cpassword = request.POST.get('cpassword')

        print(name, email, password, cpassword)
        if password == cpassword:
            if Smart.objects.filter(email=email).exists():
                messages.error(request, f"Your {email} Email Id already exists, Try another Id")
                return render(request, 'index.html')
            query = Smart(name=name, email=email, password=password)
            query.save()
            messages.success(request, f"Your Email Id registered successfully")
        else:
            messages.error(request, f"Your password does not match, Try Again!")
    return render(request, 'index.html')

def login(request):
    if request.method == "POST":
        email = request.POST.get('email')
        password = request.POST.get('password')
        user = Smart.objects.filter(email=email).first()
        if user:
            if user.password == password:
                return render(request, 'home.html')
            else:
                messages.error(request, f"Your password is incorrect! Try Again")
                return render(request, 'index.html')
        else:
            messages.error(request, f"User with this email does not exist.")
            return render(request, 'index.html')
        
    return render(request, 'index.html')

def home(request):
    return render(request, 'home.html')

def custom_logout(request):
    logout(request)
    return render(request, 'index.html')

def view(request):
    global df
    if request.method == 'POST':
        g = int(request.POST['num'])
        file_path = (r'app/DATASET/dataset9000.csv')
        df = pd.read_csv(file_path)
        col = df.head(g).to_html()
        return render(request, 'view.html', {'table': col})
    return render(request, 'view.html')

def predict(request):
    if request.method == "POST":
        file_path = (r'app/DATASET/dataset9000.csv')
        df = pd.read_csv(file_path)
        df.drop_duplicates(inplace=True)
        original_columns = df.select_dtypes(include='object').columns

        # Initialize LabelEncoder
        label_encoders = {}

        # Apply LabelEncoder to each categorical variable
        for col in original_columns:
            label_encoders[col] = LabelEncoder()
            df[col] = label_encoders[col].fit_transform(df[col])

        # Data splitting
        x = df.drop(['Role'], axis=1)
        y = df['Role']

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

        # Get user input from the form
        form_data = [
            float(request.POST.get(field)) for field in [
                'database_fundamentals', 'computer_architecture', 'distributed_computing_systems', 'cyber_security',
                'networking', 'software_development', 'Programming_skills', 'project_management', 'computer_forensics',
                'technical_communication', 'ai_ml', 'software_engineering', 'business_analysis', 'communication_skills',
                'data_science', 'troubleshooting_skills', 'graphics_desgining'
            ]
        ]
        
        # Train the model
        model = RandomForestClassifier()
        model.fit(x_train, y_train)

        # Get probabilities and predictions
        probabilities = model.predict_proba([form_data])[0]
        predicted_labels = model.classes_

        # Create a list of tuples (label, probability) and sort by probability
        label_probabilities = [(predicted_labels[i], probabilities[i]) for i in range(len(predicted_labels))]
        label_probabilities.sort(key=lambda x: x[1], reverse=True)

        # Get top 3 predictions with images and probabilities
        job_titles = [
            "AI ML Specialist", "API Specialist", "Application Support Engineer", "Business Analyst", "Customer Service Executive",
            "Cyber Security Specialist", "Data Scientist", "Database Administrator", "Graphics Designer", "Hardware Engineer",
            "Helpdesk Engineer", "Information Security Specialist", "Networking Engineer", "Project Manager", "Software Developer",
            "Software tester", "Technical Writer"
        ]
        job_images = [
            "https://www.onlc.com/blog/wp-content/uploads/2023/12/it-professionals-using-artificial-intelligence-aug-2024-07-04-01-21-27-utc.jpg",
            "https://codinix.com/uploads/products/267-1137-1744354040.png",
            "https://avahr.com/wp-content/uploads/2023/03/application-support-engineer-job-description-template.jpg",
            "https://onlinedegrees.sandiego.edu/wp-content/uploads/2022/09/business-analyst-vs-data-analyst-vs-data-scientist.jpg",
            "https://www.libm.co.uk/wp-content/uploads/2021/09/Customer-Service-Executive-Diploma.jpg",
            "https://woz-u.com/wp-content/uploads/2023/03/cybersecurity-spcialists.jpg",
            "http://thedatascientist.com/wp-content/uploads/2025/08/Data-Science.jpg",
            "https://onlinedegrees.sandiego.edu/wp-content/uploads/2023/06/five-steps-to-start-career-database-admin.jpg",
            "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTC5nDZc9RA-ZcRGoGQ_yLGj_QKzCdYQP9aXA&s",
            "https://res.cloudinary.com/highereducation/images/f_auto,q_auto/v1675809054/ComputerScience.org/day-to-day-hardware-engineer/day-to-day-hardware-engineer.jpg?_i=AA",
            "https://d3njjcbhbojbot.cloudfront.net/api/utilities/v1/imageproxy/https://images.ctfassets.net/wp1lcwdav1p1/6RYYfl482kuXpfD52Vg7aO/64f351d91851eb9eab19b93ce17d207f/GettyImages-1270220777.jpeg?w=1500&h=680&q=60&fit=fill&f=faces&fm=jpg&fl=progressive&auto=format%2Ccompress&dpr=1&w=1000",
            "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSPA1MWtYSzeURulG4JPxHqHYifAje5zgRO-A&s",
            "https://www.zippia.com/_next/image/?url=https%3A%2F%2Fmedia.zippia.com%2Fjob-title%2Fimages%2Fnetwork-engineer%2Fnetwork-engineer-3.webp&w=3840&q=75",
            "https://www.worldcampus.psu.edu/sites/default/files/2020-06/What-Does-a-Project-Manager-Do-Alt.jpg",
            "https://www.training.com.au/wp-content/uploads/software-developer.jpg",
            "https://jessup.edu/wp-content/uploads/2023/12/Why-Pursue-a-Career-in-Software-Testing.jpg",
            "https://writingstudio.com/wp-content/uploads/2021/07/hire-technical-writer.jpg"
        ]

        # Prepare the top 3 predictions with their corresponding images
        top_3_predictions = [
            {
                "job_title": job_titles[label_probabilities[i][0]],
                "probability": int(50 + label_probabilities[i][1] * 50),
                "image": job_images[label_probabilities[i][0]]
            }
            for i in range(3)
        ]

        # Store the predictions in session and redirect to result page
        request.session['predictions'] = top_3_predictions
        return redirect('result')
    
    return render(request, 'predict.html')

def result(request):
    # Get predictions from session
    top_3_predictions = request.session.get('predictions', [])
    if not top_3_predictions:
        # If no results, redirect to predict page
        return redirect('predict')
    
    return render(request, 'result.html', {'top_3_predictions': top_3_predictions})

def model(request):
    msg = ''
    file_path = (r'app/DATASET/dataset9000.csv')
    df = pd.read_csv(file_path)
    df.drop_duplicates(inplace=True)
    original_columns = df.select_dtypes(include='object').columns

    # Initialize LabelEncoder
    label_encoders = {}

    # Apply LabelEncoder to each categorical variable
    for col in original_columns:
        label_encoders[col] = LabelEncoder()
        df[col] = label_encoders[col].fit_transform(df[col])
    
    # Data splitting
    x = df.drop(['Role'], axis=1)
    y = df['Role']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    if request.method == 'POST':
        model_type = request.POST.get('algorithm')
     
        if model_type == '1':
            re = RandomForestClassifier()
            re.fit(x_train, y_train)
            re_pred = re.predict(x_test)
            ac = accuracy_score(y_test, re_pred)
            precision = precision_score(y_test, re_pred, average='weighted')
            recall = recall_score(y_test, re_pred, average='weighted')
            f1 = f1_score(y_test, re_pred, average='weighted')
            acc_percent = round(ac*100, 2)
            msg = f'Accuracy of RandomForest: {acc_percent}%\n'
            msg += f'Precision: {precision:.4f}\n'
            msg += f'Recall: {recall:.4f}\n'
            msg += f'F1-score: {f1:.4f}\n'
            return render(request, 'model.html', {'msg': msg})
        elif model_type == '2':
            xgb = XGBClassifier(max_depth=5, random_state=42)
            xgb.fit(x_train, y_train)
            de_pred = xgb.predict(x_test)
            ac1 = accuracy_score(y_test, de_pred)
            precision = precision_score(y_test, de_pred, average='weighted')
            recall = recall_score(y_test, de_pred, average='weighted')
            f1 = f1_score(y_test, de_pred, average='weighted')
            acc_percent = round(ac1*100, 2)
            msg = f'Accuracy of XGBOOST Algorithm: {acc_percent}%\n'
            msg += f'Precision: {precision:.4f}\n'
            msg += f'Recall: {recall:.4f}\n'
            msg += f'F1-score: {f1:.4f}\n'
            return render(request, 'model.html', {'msg': msg})
        elif model_type == '3':
            de = DecisionTreeClassifier()
            de.fit(x_train, y_train)
            de_pred = de.predict(x_test)
            ac1 = accuracy_score(y_test, de_pred)
            precision = precision_score(y_test, de_pred, average='weighted')
            recall = recall_score(y_test, de_pred, average='weighted')
            f1 = f1_score(y_test, de_pred, average='weighted')
            acc_percent = round(ac1*100, 2)
            msg = f'Accuracy of Decision Tree Algorithm: {acc_percent}%\n'
            msg += f'Precision: {precision:.4f}\n'
            msg += f'Recall: {recall:.4f}\n'
            msg += f'F1-score: {f1:.4f}\n'
            return render(request, 'model.html', {'msg': msg})
        elif model_type == '4':
            sv = SVC()
            sv.fit(x_train, y_train)
            de_pred = sv.predict(x_test)
            ac1 = accuracy_score(y_test, de_pred)
            precision = precision_score(y_test, de_pred, average='weighted')
            recall = recall_score(y_test, de_pred, average='weighted')
            f1 = f1_score(y_test, de_pred, average='weighted')
            acc_percent = round(ac1*100, 2)
            msg = f'Accuracy of SVM Algorithm: {acc_percent}%\n'
            msg += f'Precision: {precision:.4f}\n'
            msg += f'Recall: {recall:.4f}\n'
            msg += f'F1-score: {f1:.4f}\n'
            return render(request, 'model.html', {'msg': msg})
   
    return render(request, 'model.html')

def chatbot_interface(request):
    # Get predictions from session
    top_3_predictions = request.session.get('predictions', [])
    if not top_3_predictions:
        return redirect('predict')
    
    return render(request, 'chatbot.html', {'top_3_predictions': top_3_predictions})

def ask_gemini(request):
    if request.method == 'POST':
        try:
            selected_job = request.POST.get('selected_job')
            selected_question = request.POST.get('selected_question')
            
            if not selected_job or not selected_question:
                return JsonResponse({'error': 'Please select both job and question'})
            
            # Get user's skill levels from session if available
            user_skills = request.session.get('user_skills', {})
            
            # Define the questions and their prompts
            question_prompts = {
                '1': f"""Create a detailed 6-month step-by-step learning roadmap for becoming a {selected_job}:
                
                Format as:
                **Month 1-2: Foundation Phase**
                - [ ] Skill 1 to learn
                - [ ] Skill 2 to practice
                - [ ] Project: Build a basic project
                
                **Month 3-4: Intermediate Phase**
                - [ ] Advanced skill 1
                - [ ] Tool mastery
                - [ ] Project: Intermediate project
                
                **Month 5-6: Advanced & Job Ready**
                - [ ] Portfolio development
                - [ ] Interview preparation
                - [ ] Final capstone project
                
                Include time allocation per week and recommended resources.""",
                
                '2': f"""List ALL the skills needed for a {selected_job} career:
                
                **Technical Skills:**
                - Programming languages required
                - Tools and frameworks
                - Technologies to master
                
                **Soft Skills:**
                - Communication skills needed
                - Teamwork and collaboration
                - Problem-solving abilities
                
                **Industry-Specific Skills:**
                - Domain knowledge
                - Industry standards
                - Best practices
                
                Format as bullet points with importance level (High/Medium/Low).""",
                
                '3': f"""Provide comprehensive e-learning platforms and resources for learning skills for {selected_job}:
                
                For each platform provide:
                **Platform Name:**
                - **Best Courses:** List 2-3 specific course names
                - **URL:** Direct links to courses
                - **Cost:** Free/Premium/Subscription
                - **Skill Level:** Beginner/Intermediate/Advanced
                - **Duration:** Estimated completion time
                - **Certificate:** Yes/No
                
                Include platforms like: Coursera, Udemy, edX, Pluralsight, LinkedIn Learning, YouTube channels, freeCodeCamp, etc.
                Provide actual course names and approximate pricing.""",
                
                '4': f"""Explain why {selected_job} is specifically recommended based on the user's skill assessment:
                
                Analyze why this career path matches well with the user's profile. Consider:
                - Which of the user's strongest skills align with this job
                - Growth opportunities in this field
                - Market demand and future prospects
                - Salary potential compared to other options
                - Work-life balance aspects
                - Entry barriers and how to overcome them
                
                Give specific reasons why this is a good match.""",
                
                '5': f"""What salary can I expect as a {selected_job}?
                - Entry-level salary range (0-2 years)
                - Mid-career salary range (3-8 years)
                - Senior-level salary range (8+ years)
                - Factors affecting salary (location, company, skills)
                - Geographic variations (US, Europe, India, etc.)
                - Bonus and benefits structure""",
                
                '6': f"""List the key industry certifications for {selected_job}:
                - Certification name
                - Issuing organization
                - Prerequisites needed
                - Exam format and duration
                - Validity period
                - Approximate cost
                - Career benefits
                - Difficulty level""",
                
                '7': f"""What are the day-to-day responsibilities of a {selected_job}?
                - Morning routine and tasks
                - Afternoon responsibilities
                - Weekly meetings and reviews
                - Monthly/Quarterly goals
                - Team collaboration aspects
                - Tools used regularly
                - Common challenges faced
                - Performance metrics""",
                
                '8': f"""What are the growth opportunities for a {selected_job}?
                - Career progression path (Junior → Mid → Senior → Lead)
                - Management vs technical track
                - Specialization options
                - Industry transitions possible
                - Entrepreneur opportunities
                - 5-year and 10-year outlook""",
                
                '9': f"""How to prepare for {selected_job} interviews?
                - Common technical interview questions
                - Behavioral questions to expect
                - Coding challenges (if applicable)
                - System design questions
                - Portfolio presentation tips
                - Salary negotiation strategies
                - Mock interview resources""",
                
                '10': f"""What are the current industry trends for {selected_job}?
                - Emerging technologies to watch
                - Market demand changes
                - Remote work opportunities
                - Industry 4.0/5.0 impact
                - Future-proof skills
                - Companies leading in this field"""
            }
            
            if selected_question in question_prompts:
                prompt = question_prompts[selected_question]
            else:
                # Handle custom question more naturally
                prompt = f"You are an AI Career Counselor specializing in {selected_job}. The user asked: '{selected_question}'. Respond appropriately and helpfully. If it's a greeting or simple query, keep the response concise. For detailed questions, use markdown format with headings and bullet points where suitable."
            
            # Initialize Gemini model - using a valid model name
            model = genai.GenerativeModel('gemini-2.5-flash')
            response = model.generate_content(prompt)
            
            return JsonResponse({
                'answer': response.text,
                'job': selected_job,
                'question': selected_question
            })
            
        except Exception as e:
            return JsonResponse({'error': str(e)})
    
    return JsonResponse({'error': 'Invalid request method'})