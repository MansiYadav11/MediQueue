from flask import Flask, request, render_template_string, session, redirect, url_for, send_from_directory
from flask_socketio import SocketIO, emit
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import pytz
import os
import uuid
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///mediqueue.db' 
app.config['SECRET_KEY'] = "SET_YOUR_SECRECT_KEY"

# Configure upload settings
UPLOAD_FOLDER = 'static/payments'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'pdf'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5MB max file size

# Doctor login credentials
DOCTOR_USERNAME = "doctor"
DOCTOR_PASSWORD = "mediqueue123"

db = SQLAlchemy(app)
socketio = SocketIO(app, cors_allowed_origins="*")
IST = pytz.timezone('Asia/Kolkata')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Database Models
class Patient(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    phone = db.Column(db.String(20), nullable=False)

class Doctor(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    specialty = db.Column(db.String(100), nullable=False)

class Appointment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey('patient.id'))
    doctor_id = db.Column(db.Integer, db.ForeignKey('doctor.id'))
    token_number = db.Column(db.Integer)
    symptoms = db.Column(db.Text)
    status = db.Column(db.String(20), default='waiting')
    payment_screenshot = db.Column(db.String(500))  # Added for payment screenshot

class EmergencyAlert(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    patient_name = db.Column(db.String(100), nullable=False)
    patient_phone = db.Column(db.String(20), nullable=False)
    symptoms = db.Column(db.Text, nullable=False)
    status = db.Column(db.String(20), default='active')
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(IST))

class Prescription(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    appointment_id = db.Column(db.Integer, db.ForeignKey('appointment.id'))
    medicines = db.Column(db.Text, nullable=False)
    instructions = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(IST))

# Authentication Decorator
def doctor_login_required(f):
    def decorated_function(*args, **kwargs):
        if not session.get('doctor_logged_in'):
            return redirect(url_for('doctor_login'))
        return f(*args, **kwargs)
    decorated_function.__name__ = f.__name__
    return decorated_function

# ===== AI RECOMMENDATION FUNCTIONS =====

def fallback_recommendation(symptoms: str) -> str:
    """
    Rule-based fallback system when AI model fails
    Updated to match existing doctor specialties
    """
    symptoms_lower = symptoms.lower()
    
    # Map to existing doctor specialties
    symptom_mapping = {
        # Emergency/Critical
        'chest pain': 'Heart Attack',
        'sharp chest pain': 'Heart Attack',
        'difficulty breathing': 'Heart Attack',
        'shortness of breath': 'Heart Attack',
        'sudden weakness': 'Stroke',
        'slurred speech': 'Stroke',
        'severe pain': 'Kidney Stones',
        
        # Respiratory
        'cough': 'Asthma',
        'wheezing': 'Asthma', 
        'asthma': 'Asthma',
        'breathing problem': 'Asthma',
        
        # Skin
        'skin rash': 'Psoriasis',
        'red patches': 'Psoriasis',
        'itching skin': 'Psoriasis',
        'dry skin': 'Psoriasis',
        
        # Neurological
        'headache': 'Migraine',
        'migraine': 'Migraine',
        'dizziness': 'Migraine',
        
        # Gastrointestinal
        'stomach pain': 'Gastritis',
        'abdominal pain': 'Gastritis',
        'nausea': 'Gastritis',
        'vomiting': 'Gastritis',
        
        # Kidney
        'kidney pain': 'Chronic Kidney Disease',
        'back pain': 'Chronic Kidney Disease',
        'urinary problems': 'Chronic Kidney Disease',
        
        # Blood
        'fatigue': 'Anemia',
        'weakness': 'Anemia',
        'pale skin': 'Anemia',
        
        # Joints
        'joint pain': 'Osteoarthritis',
        'swelling joints': 'Osteoarthritis',
        'stiffness': 'Osteoarthritis',
        
        # Infectious
        'fever': 'Chickenpox',
        'blisters': 'Chickenpox',
        'rash': 'Chickenpox',
        
        # Diabetes
        'thirst': 'Diabetes',
        'frequent urination': 'Diabetes',
        
        # Veins
        'swollen veins': 'Varicose Veins',
        'leg pain': 'Varicose Veins',
        
        # Hypertension
        'high blood pressure': 'Hypertension',
        'blood pressure': 'Hypertension',
        
        # COVID
        'covid': 'COVID-19',
        'loss of taste': 'COVID-19',
        'loss of smell': 'COVID-19',
        
        # TB
        'tuberculosis': 'Tuberculosis',
        'blood in sputum': 'Tuberculosis',
        
        # Allergy
        'sneezing': 'Allergy',
        'runny nose': 'Allergy',
        'allergic': 'Allergy',
        
        # Mental Health
        'depression': 'Depression',
        'sadness': 'Depression',
        'anxiety': 'Depression'
    }
    
    # Check for exact matches first
    for symptom, specialty in symptom_mapping.items():
        if symptom in symptoms_lower:
            return specialty
    
    # Word-based matching
    words = symptoms_lower.split()
    for word in words:
        for symptom, specialty in symptom_mapping.items():
            if word in symptom and len(word) > 4:
                return specialty
    
    return 'General Physician'


def recommend_specialty(symptoms):
    """
    Enhanced function: Uses ACTUAL trained DistilBERT model
    """
    print(f"\n🔍 Enhanced AI analyzing symptoms: '{symptoms}'")
    
    try:
        from model_predictor_enhanced import ai_recommend
        ai_result = ai_recommend(symptoms)
        
        if ai_result and ai_result.get('success'):
            specialty = ai_result['condition']
            confidence = ai_result.get('confidence', 0)
            is_emergency = ai_result.get('emergency', False)
            
            print(f"✅ AI Success: {specialty} (Confidence: {confidence:.2f})")
            
            # Add emergency flag
            display_specialty = specialty
            if is_emergency:
                display_specialty = f"🚨 {specialty}"
            
            return display_specialty
        else:
            print(f"❌ AI Low confidence: {ai_result.get('message', 'Unknown reason')}")
            # Show confidence scores for debugging
            if 'suggestions' in ai_result:
                for suggestion in ai_result['suggestions']:
                    print(f"   - {suggestion['condition']}: {suggestion['confidence']:.2f}")
            
    except ImportError as e:
        print(f"❌ Enhanced AI Model not found: {e} - using fallback")
    except Exception as e:
        print(f"❌ Enhanced AI Model error: {e} - using fallback")
    
    # Fallback to your existing rule-based system
    print("🔄 Using fallback recommendation system")
    fallback_result = fallback_recommendation(symptoms)
    print(f"📋 Fallback result: {fallback_result}")
    
    return fallback_result

# ===== ROUTES =====

@app.route('/')
def home():
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>MediQueue+ - Smart Healthcare</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body { 
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                color: #333;
            }
            .container { 
                max-width: 1200px; 
                margin: 0 auto; 
                padding: 20px; 
            }
            .header {
                text-align: center;
                padding: 60px 20px;
                color: white;
            }
            .header h1 {
                font-size: 3.5em;
                margin-bottom: 10px;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
            }
            .header p {
                font-size: 1.3em;
                opacity: 0.9;
                margin-bottom: 40px;
            }
            .nav-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 25px;
                margin: 40px 0;
            }
            .nav-card {
                background: white;
                padding: 40px 30px;
                border-radius: 20px;
                text-align: center;
                text-decoration: none;
                color: #333;
                box-shadow: 0 15px 35px rgba(0,0,0,0.1);
                transition: all 0.3s ease;
                border: 3px solid transparent;
            }
            .nav-card:hover {
                transform: translateY(-10px);
                box-shadow: 0 20px 40px rgba(0,0,0,0.2);
                border-color: #667eea;
            }
            .nav-card h3 {
                font-size: 1.8em;
                margin-bottom: 15px;
                color: #2c3e50;
            }
            .nav-card p {
                color: white;
                line-height: 1.6;
            }
            .patient-card { background: linear-gradient(135deg, #27ae60, #2ecc71); color: white; }
            .doctor-card { background: linear-gradient(135deg, #74b9ff, #0984e3); color: white; }
            .emergency-card { background: linear-gradient(135deg, #ff7675, #d63031); color: white; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🏥 MediQueue+</h1>
                <p>Advanced Healthcare Management System</p>
            </div>
            
            <div class="nav-grid">
                <a href="/patient" class="nav-card patient-card">
                    <h3>👤 Patient Portal</h3>
                    <p>Get AI-powered doctor recommendations and book appointments</p>
                </a>
                
                <a href="/doctor-login" class="nav-card doctor-card">
                    <h3>👨‍⚕️ Doctor Dashboard</h3>
                    <p>Secure access to patient management and prescriptions</p>
                </a>
                
                <a href="/sos" class="nav-card emergency-card">
                    <h3>🚨 Emergency SOS</h3>
                    <p>Immediate medical attention for critical situations</p>
                </a>
            </div>
        </div>
    </body>
    </html>
    '''

@app.route('/patient')
def patient():
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Patient Portal - MediQueue+</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body { 
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%);
                min-height: 100vh;
                color: #333;
            }
            .container { 
                max-width: 800px; 
                margin: 0 auto; 
                padding: 40px 20px; 
            }
            .card {
                background: white;
                padding: 50px;
                border-radius: 20px;
                box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            }
            .back-btn { 
                display: inline-flex;
                align-items: center;
                padding: 12px 25px; 
                background: #95a5a6; 
                color: white; 
                text-decoration: none; 
                border-radius: 10px; 
                margin-bottom: 30px;
                font-weight: 600;
            }
            h1 { 
                color: #2c3e50; 
                margin-bottom: 10px;
                font-size: 2.5em;
            }
            .subtitle {
                color: #7f8c8d;
                margin-bottom: 40px;
                font-size: 1.2em;
            }
            .form-group {
                margin-bottom: 30px;
            }
            label {
                display: block;
                margin-bottom: 12px;
                font-weight: 600;
                color: #2c3e50;
                font-size: 1.1em;
            }
            textarea {
                width: 100%;
                padding: 20px;
                border: 2px solid #e0e6ed;
                border-radius: 12px;
                font-size: 16px;
                transition: all 0.3s ease;
                box-sizing: border-box;
                font-family: inherit;
                resize: vertical;
                min-height: 150px;
            }
            textarea:focus {
                border-color: #3498db;
                outline: none;
                box-shadow: 0 0 0 3px rgba(52, 152, 219, 0.1);
            }
            button { 
                background: linear-gradient(135deg, #3498db, #2980b9); 
                color: white; 
                padding: 18px 45px; 
                border: none; 
                border-radius: 12px; 
                font-size: 18px;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.3s ease;
                width: 100%;
            }
            button:hover { 
                transform: translateY(-2px);
                box-shadow: 0 10px 25px rgba(52, 152, 219, 0.3);
            }
        </style>
    </head>
    <body>
        <div class="container">
            <a href="/" class="back-btn">← Back to Home</a>
            
            <div class="card">
                <h1>👤 Patient Portal</h1>
                <p class="subtitle">Describe your symptoms for AI-powered doctor matching</p>
                
                <form method="POST" action="/recommend">
                    <div class="form-group">
                        <label for="symptoms">Describe Your Symptoms in Detail:</label>
                        <textarea name="symptoms" placeholder="Example: I have been experiencing sharp chest pain for the last 2 hours, accompanied by shortness of breath and dizziness..." required></textarea>
                    </div>
                    
                    <button type="submit">🔍 Analyze Symptoms & Find Doctor</button>
                </form>
            </div>
        </div>
    </body>
    </html>
    '''

@app.route('/recommend', methods=['POST'])
def recommend():
    symptoms = request.form['symptoms']
    specialty = recommend_specialty(symptoms)
    
    # Clean the specialty for matching (remove emojis, etc.)
    clean_specialty = specialty.replace('🚨', '').strip()
    
    print(f"🔍 Looking for doctors with specialty: '{clean_specialty}'")
    
    # Emergency warning for critical symptoms
    emergency_warning = ""
    critical_symptoms = ['chest pain', 'shortness of breath', 'severe pain', 'unconscious', 'heart attack']
    
    if any(symptom in symptoms.lower() for symptom in critical_symptoms):
        emergency_warning = '''
        <div style="background: #f8d7da; color: #721c24; padding: 20px; border-radius: 10px; 
                   border: 2px solid #f5c6cb; margin: 20px 0; text-align: center;">
            <h3 style="color: #721c24; margin-top: 0;">🚨 MEDICAL EMERGENCY WARNING</h3>
            <p><strong>Your symptoms may indicate a serious medical condition!</strong></p>
            <p>Please seek immediate medical attention or call emergency services.</p>
            <p style="font-size: 1.2em; margin: 15px 0;">📞 Call Emergency: 108 or 112</p>
        </div>
        '''
    
    # Find matching doctors
    doctors = Doctor.query.filter_by(specialty=clean_specialty).all()
    
    print(f"✅ Found {len(doctors)} doctors for specialty: {clean_specialty}")
    
    doctors_html = ""
    for doctor in doctors:
        doctors_html += f'''
        <div style="background: #ecf0f1; padding: 20px; margin: 15px 0; border-radius: 10px;">
            <h3>👨‍⚕️ Dr. {doctor.name}</h3>
            <p><strong>Specialty:</strong> {doctor.specialty}</p>
            
            <form method="POST" action="/register" enctype="multipart/form-data">
                <input type="hidden" name="doctor_id" value="{doctor.id}">
                <input type="hidden" name="symptoms" value="{symptoms}">
                
                <div style="background: white; padding: 15px; border-radius: 5px; margin: 15px 0;">
                    <h4>📝 Book Appointment</h4>
                    <input type="text" name="name" placeholder="Your Full Name" required style="padding: 10px; margin: 5px; width: 200px; border: 1px solid #bdc3c7; border-radius: 5px;">
                    <input type="text" name="phone" placeholder="Phone Number" required style="padding: 10px; margin: 5px; width: 200px; border: 1px solid #bdc3c7; border-radius: 5px;">
                    <br>
                    
                    <div style="margin: 10px 5px; padding: 10px; background: #f8f9fa; border-radius: 5px; width: 400px;">
                        <label style="display: block; margin-bottom: 8px; font-weight: bold; color: #2c3e50;">
                            💳 Upload Payment Screenshot (Required):
                        </label>
                        <input type="file" name="payment_screenshot" accept="image/*,.pdf" required style="padding: 8px; width: 100%;">
                        <p style="font-size: 12px; color: #666; margin-top: 5px;">Accepted: JPG, PNG, PDF (Max: 5MB)</p>
                    </div>
                    
                    <button type="submit" style="background: #27ae60; color: white; padding: 12px 25px; margin-top: 10px; border: none; border-radius: 5px; cursor: pointer;">🎫 Book Appointment & Get Token</button>
                </div>
            </form>
        </div>
        '''
    
    return f'''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Doctor Recommendation - MediQueue+</title>
        <style>
            body {{ font-family: Arial; margin: 40px; background: #f0f8ff; }}
            .container {{ max-width: 800px; margin: 0 auto; background: white; padding: 40px; border-radius: 15px; box-shadow: 0 10px 30px rgba(0,0,0,0.1); }}
        </style>
    </head>
    <body>
        <div class="container">
            <a href="/patient" style="padding: 10px 20px; background: #95a5a6; color: white; text-decoration: none; border-radius: 5px; margin-bottom: 20px; display: inline-block;">← Back to Symptoms</a>
            
            {emergency_warning}
            
            <div style="background: #e8f4f8; padding: 20px; border-radius: 10px; margin: 20px 0; border-left: 4px solid #3498db;">
                <h3 style="color: #2c3e50; margin-top: 0;">Your Symptoms:</h3>
                <p style="color: #5d6d7e;">"{symptoms}"</p>
            </div>
            
            <div style="background: #d4edda; padding: 20px; border-radius: 10px; margin: 20px 0; border-left: 4px solid #27ae60;">
                <h3 style="color: #155724; margin-top: 0;">✅ AI Recommendation: {specialty}</h3>
                <p style="color: #155724;">Based on our analysis of your symptoms</p>
            </div>
            
            <h2>Available Doctors:</h2>
            {doctors_html if doctors_html else '<p>No doctors available in this specialty.</p>'}
        </div>
    </body>
    </html>
    '''

@app.route('/register', methods=['POST'])
def register():
    try:
        name = request.form['name']
        phone = request.form['phone']
        doctor_id = request.form['doctor_id']
        symptoms = request.form['symptoms']
        
        # Check if file was uploaded
        if 'payment_screenshot' not in request.files:
            return '''
            <!DOCTYPE html>
            <html>
            <body>
                <div style="text-align: center; padding: 50px;">
                    <h2 style="color: #e74c3c;">❌ Payment Screenshot Required</h2>
                    <p>Please upload payment confirmation screenshot</p>
                    <a href="/patient" style="padding: 10px 20px; background: #3498db; color: white; text-decoration: none; border-radius: 5px;">Go Back</a>
                </div>
            </body>
            </html>
            ''', 400
        
        file = request.files['payment_screenshot']
        
        if file.filename == '':
            return '''
            <!DOCTYPE html>
            <html>
            <body>
                <div style="text-align: center; padding: 50px;">
                    <h2 style="color: #e74c3c;">❌ No file selected</h2>
                    <p>Please select a payment screenshot to upload</p>
                    <a href="/patient" style="padding: 10px 20px; background: #3498db; color: white; text-decoration: none; border-radius: 5px;">Go Back</a>
                </div>
            </body>
            </html>
            ''', 400
        
        # Handle file upload
        screenshot_filename = None
        if file and allowed_file(file.filename):
            # Create upload folder if not exists
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            
            # Generate unique filename
            filename = secure_filename(file.filename)
            unique_filename = f"{uuid.uuid4().hex}_{filename}"
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(file_path)
            screenshot_filename = unique_filename
        else:
            return '''
            <!DOCTYPE html>
            <html>
            <body>
                <div style="text-align: center; padding: 50px;">
                    <h2 style="color: #e74c3c;">❌ Invalid File Type</h2>
                    <p>Please upload only image files (PNG, JPG, JPEG, GIF) or PDF</p>
                    <a href="/patient" style="padding: 10px 20px; background: #3498db; color: white; text-decoration: none; border-radius: 5px;">Go Back</a>
                </div>
            </body>
            </html>
            ''', 400
        
        # Save patient
        patient = Patient(name=name, phone=phone)
        db.session.add(patient)
        db.session.commit()
        
        # Get last token for THIS SPECIFIC DOCTOR (only waiting appointments)
        last_token = Appointment.query.filter_by(doctor_id=doctor_id, status='waiting').count()
        token_number = last_token + 1
        
        # Create appointment with payment screenshot
        appointment = Appointment(
            patient_id=patient.id,
            doctor_id=doctor_id,
            token_number=token_number,
            symptoms=symptoms,
            payment_screenshot=screenshot_filename
        )
        db.session.add(appointment)
        db.session.commit()
        
        return f'''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Appointment Confirmed - MediQueue+</title>
            <style>
                body {{ font-family: Arial; margin: 40px; background: #f0f8ff; }}
                .container {{ max-width: 500px; margin: 0 auto; background: white; padding: 40px; border-radius: 15px; box-shadow: 0 10px 30px rgba(0,0,0,0.1); text-align: center; }}
                .payment-note {{ background: #e8f4f8; padding: 15px; border-radius: 8px; margin: 20px 0; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h2 style="color: #27ae60;">✅ Appointment Booked!</h2>
                <div class="payment-note">
                    <p>📱 <strong>Payment screenshot uploaded successfully</strong></p>
                </div>
                <h1 style="color: #e74c3c; font-size: 48px;">Token #{token_number}</h1>
                <p><strong>Patient:</strong> {name}</p>
                <p>Please wait for your token to be called.</p>
                <a href="/" style="display: inline-block; padding: 10px 20px; background: #3498db; color: white; text-decoration: none; border-radius: 5px;">Back to Home</a>
            </div>
        </body>
        </html>
        '''
            
    except Exception as e:
        print(f"Registration error: {e}")
        return f'''
        <!DOCTYPE html>
        <html>
        <body>
            <div style="text-align: center; padding: 50px;">
                <h2 style="color: #e74c3c;">❌ Registration Failed</h2>
                <p>Error: {str(e)}</p>
                <a href="/patient" style="padding: 10px 20px; background: #3498db; color: white; text-decoration: none; border-radius: 5px;">Go Back</a>
            </div>
        </body>
        </html>
        ''', 500

# ===== DOCTOR AUTHENTICATION =====

@app.route('/doctor-login')
def doctor_login():
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Doctor Login - MediQueue+</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body { 
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                display: flex;
                align-items: center;
                justify-content: center;
            }
            .login-container { 
                max-width: 400px; 
                background: white; 
                padding: 50px 40px; 
                border-radius: 20px; 
                box-shadow: 0 20px 40px rgba(0,0,0,0.2);
                text-align: center;
            }
            .login-icon {
                font-size: 3em;
                margin-bottom: 20px;
                color: #3498db;
            }
            h1 { 
                color: #2c3e50; 
                margin-bottom: 10px;
            }
            .subtitle {
                color: #7f8c8d;
                margin-bottom: 40px;
            }
            .form-group {
                margin-bottom: 25px;
                text-align: left;
            }
            label {
                display: block;
                margin-bottom: 10px;
                font-weight: 600;
                color: #2c3e50;
            }
            input {
                width: 100%;
                padding: 15px;
                border: 2px solid #e0e6ed;
                border-radius: 10px;
                font-size: 16px;
                box-sizing: border-box;
                transition: all 0.3s ease;
            }
            input:focus {
                border-color: #3498db;
                outline: none;
                box-shadow: 0 0 0 3px rgba(52, 152, 219, 0.1);
            }
            button { 
                background: linear-gradient(135deg, #3498db, #2980b9); 
                color: white; 
                padding: 18px 40px; 
                border: none; 
                border-radius: 10px; 
                font-size: 16px;
                font-weight: 600;
                cursor: pointer;
                width: 100%;
                transition: all 0.3s ease;
            }
            button:hover { 
                transform: translateY(-2px);
                box-shadow: 0 10px 25px rgba(52, 152, 219, 0.3);
            }
            .back-btn {
                display: inline-block;
                margin-top: 25px;
                color: #3498db;
                text-decoration: none;
                font-weight: 600;
            }
            .security-note {
                background: #e8f4f8;
                padding: 15px;
                border-radius: 8px;
                margin-top: 20px;
                color: #2c3e50;
                font-size: 14px;
                border-left: 4px solid #3498db;
            }
        </style>
    </head>
    <body>
        <div class="login-container">
            <div class="login-icon">👨‍⚕️</div>
            <h1>Doctor Login</h1>
            <div class="subtitle">Secure Access to Medical Dashboard</div>
            
            <form method="POST" action="/doctor-login">
                <div class="form-group">
                    <label for="username">Username:</label>
                    <input type="text" id="username" name="username" required>
                </div>
                
                <div class="form-group">
                    <label for="password">Password:</label>
                    <input type="password" id="password" name="password" required>
                </div>
                
                <button type="submit">🔐 Login to Dashboard</button>
            </form>
            
            <div class="security-note">
                <strong>🔒 Secure Access Only</strong><br>
                Authorized medical personnel only. Contact administration for credentials.
            </div>
            
            <a href="/" class="back-btn">← Back to Home</a>
        </div>
    </body>
    </html>
    '''

@app.route('/doctor-login', methods=['POST'])
def doctor_login_post():
    username = request.form['username']
    password = request.form['password']
    
    if username == DOCTOR_USERNAME and password == DOCTOR_PASSWORD:
        session['doctor_logged_in'] = True
        return redirect(url_for('doctor_dashboard'))
    else:
        return '''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Login Failed - MediQueue+</title>
            <style>
                body { font-family: Arial; margin: 40px; background: #f0f8ff; }
                .container { max-width: 400px; margin: 0 auto; background: white; padding: 40px; border-radius: 15px; box-shadow: 0 10px 30px rgba(0,0,0,0.1); text-align: center; }
            </style>
        </head>
        <body>
            <div class="container">
                <h2 style="color: #e74c3c;">❌ Invalid Credentials</h2>
                <p>Please check your username and password and try again.</p>
                <a href="/doctor-login" style="display: inline-block; padding: 10px 20px; background: #3498db; color: white; text-decoration: none; border-radius: 5px; margin: 10px;">Try Again</a>
                <a href="/" style="display: inline-block; padding: 10px 20px; background: #95a5a6; color: white; text-decoration: none; border-radius: 5px; margin: 10px;">Back to Home</a>
            </div>
        </body>
        </html>
        '''

@app.route('/doctor-logout')
def doctor_logout():
    session.pop('doctor_logged_in', None)
    return redirect(url_for('home'))

# ===== TOKEN RESET FUNCTIONALITY =====

@app.route('/reset-tokens', methods=['POST'])
@doctor_login_required
def reset_tokens():
    """Reset all token numbers to 1, 2, 3... PER DOCTOR"""
    try:
        # Get all doctors
        doctors = Doctor.query.all()
        
        for doctor in doctors:
            # Get waiting appointments for THIS DOCTOR ONLY
            appointments = Appointment.query.filter_by(
                doctor_id=doctor.id, 
                status='waiting'
            ).order_by(Appointment.id).all()
            
            # Reset token numbers 1, 2, 3... for this doctor
            for i, appointment in enumerate(appointments, 1):
                appointment.token_number = i
        
        db.session.commit()
        
        return '''
        <script>
            alert("✅ All tokens reset successfully (per doctor)!");
            window.location.href = "/doctor";
        </script>
        '''
        
    except Exception as e:
        return f"Error resetting tokens: {str(e)}", 500

# ===== DOCTOR DASHBOARD =====

@app.route('/doctor')
@doctor_login_required
def doctor_dashboard():
    appointments = Appointment.query.filter_by(status='waiting').all()
    emergencies = EmergencyAlert.query.filter_by(status='active').all()
    
    queue_html = ""
    for appt in appointments:
        patient = Patient.query.get(appt.patient_id)
        doctor = Doctor.query.get(appt.doctor_id)
        
        # Check if payment screenshot exists
        payment_info = ""
        if appt.payment_screenshot:
            payment_info = f'''
            <div style="margin-top: 10px;">
                <a href="/view-payment/{appt.id}" target="_blank" style="background: #9b59b6; color: white; padding: 6px 12px; text-decoration: none; border-radius: 4px; font-size: 14px;">
                    💰 View Payment
                </a>
            </div>
            '''
        
        queue_html += f'''
        <div style="background: #ecf0f1; padding: 20px; margin: 15px 0; border-radius: 10px;">
            <h3>Token #{appt.token_number} - {patient.name}</h3>
            <p><strong>Phone:</strong> {patient.phone}</p>
            <p><strong>Symptoms:</strong> {appt.symptoms}</p>
            <p><strong>Doctor:</strong> Dr. {doctor.name}</p>
            {payment_info}
            <div style="margin-top: 15px;">
                <a href="/complete-appointment/{appt.id}" style="background: #27ae60; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px; margin-right: 10px;">Mark Completed</a>
                <a href="/create-prescription/{appt.id}" style="background: #3498db; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px;">Generate Prescription</a>
            </div>
        </div>
        '''
    
    emergencies_html = ""
    for emergency in emergencies:
        formatted_time = emergency.created_at.strftime("%d-%m-%Y %I:%M %p")
        emergencies_html += f'''
        <div style="background: #f8d7da; padding: 20px; margin: 15px 0; border-radius: 10px; border-left: 4px solid #e74c3c;">
            <h3 style="color: #721c24;">🚨 EMERGENCY - {emergency.patient_name}</h3>
            <p><strong>Phone:</strong> {emergency.patient_phone}</p>
            <p><strong>Emergency:</strong> {emergency.symptoms}</p>
            <p><strong>Time:</strong> {formatted_time}</p>
            <button onclick="handleEmergency({emergency.id}, 'accepted')" style="background: #f39c12; color: white; padding: 10px 20px; border: none; border-radius: 5px; margin-right: 10px; cursor: pointer;">Accept Emergency</button>
            <button onclick="handleEmergency({emergency.id}, 'completed')" style="background: #95a5a6; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer;">Mark Handled</button>
        </div>
        '''
    
    return f'''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Doctor Dashboard - MediQueue+</title>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.1/socket.io.js"></script>
        <style>
            body {{ font-family: Arial; margin: 40px; background: #f0f8ff; }}
            .container {{ max-width: 1200px; margin: 0 auto; }}
            .header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 30px; }}
            .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 30px; }}
            .stat-card {{ background: white; padding: 20px; border-radius: 10px; text-align: center; box-shadow: 0 5px 15px rgba(0,0,0,0.1); }}
            .dashboard {{ display: grid; grid-template-columns: 1fr 1fr; gap: 30px; }}
            .section {{ background: white; padding: 25px; border-radius: 10px; box-shadow: 0 5px 15px rgba(0,0,0,0.1); }}
            .reset-btn {{ 
                background: linear-gradient(135deg, #e74c3c, #c0392b); 
                color: white; 
                padding: 12px 24px; 
                border: none; 
                border-radius: 8px; 
                font-size: 16px;
                font-weight: bold;
                cursor: pointer;
                transition: all 0.3s ease;
                width: 100%;
                margin-top: 10px;
            }}
            .reset-btn:hover {{
                transform: translateY(-2px);
                box-shadow: 0 5px 15px rgba(231, 76, 60, 0.4);
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>👨‍⚕️ Doctor Dashboard</h1>
                <div>
                    <a href="/" style="padding: 10px 20px; background: #3498db; color: white; text-decoration: none; border-radius: 5px; margin-right: 10px;">Home</a>
                    <a href="/doctor-logout" style="padding: 10px 20px; background: #e74c3c; color: white; text-decoration: none; border-radius: 5px;">Logout</a>
                </div>
            </div>
            
            <div class="stats">
                <div class="stat-card">
                    <h3 style="margin: 0; font-size: 24px;">{len(appointments)}</h3>
                    <p>Patients in Queue</p>
                </div>
                <div class="stat-card">
                    <h3 style="margin: 0; font-size: 24px;">{len(emergencies)}</h3>
                    <p>Active Emergencies</p>
                </div>
                <div class="stat-card">
                    <h3 style="margin: 0; font-size: 24px;">🚨</h3>
                    <p>Token Management</p>
                    <form method="POST" action="/reset-tokens" onsubmit="return confirm('Reset ALL token numbers? This will renumber all waiting patients PER DOCTOR.');">
                        <button type="submit" class="reset-btn">🔄 Reset Token Numbers</button>
                    </form>
                </div>
            </div>
            
            <div class="dashboard">
                <div class="section">
                    <h2>🔄 Patient Queue</h2>
                    {queue_html if queue_html else '<p>No patients in queue.</p>'}
                </div>
                
                <div class="section">
                    <h2>🚨 Emergency Alerts</h2>
                    {emergencies_html if emergencies_html else '<p>No active emergencies.</p>'}
                </div>
            </div>
        </div>

        <script>
            const socket = io();
            socket.on('new_emergency', function(data) {{
                alert('🚨 New Emergency: ' + data.patient_name);
                location.reload();
            }});
            
            function handleEmergency(emergencyId, action) {{
                fetch('/handle-emergency/' + emergencyId + '/' + action, {{ method: 'POST' }})
                .then(() => location.reload());
            }}
        </script>
    </body>
    </html>
    '''

# ===== VIEW PAYMENT SCREENSHOT =====

@app.route('/view-payment/<int:appointment_id>')
@doctor_login_required
def view_payment(appointment_id):
    """View payment screenshot for an appointment"""
    try:
        appointment = Appointment.query.get(appointment_id)
        if not appointment:
            return "Appointment not found", 404
        
        patient = Patient.query.get(appointment.patient_id)
        doctor = Doctor.query.get(appointment.doctor_id)
        
        if not appointment.payment_screenshot:
            return '''
            <!DOCTYPE html>
            <html>
            <body>
                <div style="text-align: center; padding: 50px;">
                    <h3>No Payment Record Found</h3>
                    <p>This appointment has no payment information</p>
                    <button onclick="window.history.back()" style="padding: 10px 20px; background: #3498db; color: white; border: none; border-radius: 5px; cursor: pointer;">Go Back</button>
                </div>
            </body>
            </html>
            '''
        
        # Check file type
        file_ext = appointment.payment_screenshot.rsplit('.', 1)[-1].lower() if '.' in appointment.payment_screenshot else ''
        
        if file_ext == 'pdf':
            file_display = f'''
            <embed src="/static/payments/{appointment.payment_screenshot}" width="100%" height="500px" type="application/pdf">
            '''
        else:
            file_display = f'''
            <img src="/static/payments/{appointment.payment_screenshot}" alt="Payment Screenshot" style="max-width: 100%; max-height: 500px; border: 1px solid #ddd; border-radius: 5px;">
            '''
        
        return f'''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Payment Verification - MediQueue+</title>
            <style>
                body {{ font-family: Arial; margin: 40px; background: #f0f8ff; }}
                .container {{ max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 15px; box-shadow: 0 10px 30px rgba(0,0,0,0.1); }}
                .patient-info {{ background: #e8f4f8; padding: 20px; border-radius: 10px; margin-bottom: 20px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <a href="/doctor" style="padding: 10px 20px; background: #95a5a6; color: white; text-decoration: none; border-radius: 5px; margin-bottom: 20px; display: inline-block;">← Back to Dashboard</a>
                
                <h2>💰 Payment Verification</h2>
                
                <div class="patient-info">
                    <h3>Appointment Details</h3>
                    <p><strong>Patient:</strong> {patient.name}</p>
                    <p><strong>Phone:</strong> {patient.phone}</p>
                    <p><strong>Doctor:</strong> Dr. {doctor.name} ({doctor.specialty})</p>
                    <p><strong>Token #:</strong> {appointment.token_number}</p>
                </div>
                
                <div>
                    <h3>Payment Screenshot</h3>
                    <div style="background: #f8f9fa; padding: 20px; border-radius: 10px; text-align: center;">
                        {file_display}
                    </div>
                </div>
                
                <div style="margin-top: 25px; text-align: center;">
                    <a href="/doctor" style="background: #95a5a6; color: white; padding: 12px 25px; text-decoration: none; border-radius: 5px; display: inline-block;">← Back to Dashboard</a>
                </div>
            </div>
        </body>
        </html>
        '''
        
    except Exception as e:
        return f"Error viewing payment: {str(e)}", 500

# ===== PRESCRIPTION ROUTES =====

@app.route('/create-prescription/<int:appointment_id>')
@doctor_login_required
def create_prescription(appointment_id):
    appointment = Appointment.query.get(appointment_id)
    if not appointment:
        return "Appointment not found", 404
        
    patient = Patient.query.get(appointment.patient_id)
    doctor = Doctor.query.get(appointment.doctor_id)
    
    return f'''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Create Prescription - MediQueue+</title>
        <style>
            body {{ font-family: Arial; margin: 40px; background: #f0f8ff; }}
            .container {{ max-width: 800px; margin: 0 auto; background: white; padding: 40px; border-radius: 15px; box-shadow: 0 10px 30px rgba(0,0,0,0.1); }}
        </style>
    </head>
    <body>
        <div class="container">
            <a href="/doctor" style="padding: 10px 20px; background: #95a5a6; color: white; text-decoration: none; border-radius: 5px; margin-bottom: 20px; display: inline-block;">← Back to Dashboard</a>
            
            <h1>💊 Create Prescription</h1>
            
            <div style="background: #e8f4f8; padding: 20px; border-radius: 10px; margin-bottom: 30px;">
                <h3>Patient Information</h3>
                <p><strong>Name:</strong> {patient.name}</p>
                <p><strong>Phone:</strong> {patient.phone}</p>
                <p><strong>Doctor:</strong> Dr. {doctor.name}</p>
                <p><strong>Diagnosis:</strong> {appointment.symptoms}</p>
            </div>
            
            <form method="POST" action="/save-prescription/{appointment_id}">
                <div style="margin-bottom: 20px;">
                    <label><strong>Prescribed Medicines:</strong></label>
                    <textarea name="medicines" rows="8" style="width: 100%; padding: 15px; border: 2px solid #ddd; border-radius: 8px; font-size: 16px;" placeholder="Enter prescribed medicines with dosage and instructions..." required></textarea>
                </div>
                
                <div style="margin-bottom: 20px;">
                    <label><strong>Additional Instructions:</strong></label>
                    <textarea name="instructions" rows="4" style="width: 100%; padding: 15px; border: 2px solid #ddd; border-radius: 8px; font-size: 16px;" placeholder="Any additional instructions for the patient..."></textarea>
                </div>
                
                <button type="submit" style="background: #27ae60; color: white; padding: 15px 30px; border: none; border-radius: 8px; font-size: 16px; cursor: pointer;">Generate Prescription</button>
                <a href="/doctor" style="background: #95a5a6; color: white; padding: 15px 30px; border-radius: 8px; text-decoration: none; display: inline-block; margin-left: 10px;">Cancel</a>
            </form>
        </div>
    </body>
    </html>
    '''

@app.route('/save-prescription/<int:appointment_id>', methods=['POST'])
@doctor_login_required
def save_prescription(appointment_id):
    try:
        medicines = request.form['medicines']
        instructions = request.form.get('instructions', '')
        
        prescription = Prescription(
            appointment_id=appointment_id,
            medicines=medicines,
            instructions=instructions
        )
        db.session.add(prescription)
        db.session.commit()
        
        return redirect(f'/prescription/{appointment_id}')
        
    except Exception as e:
        return f"Error saving prescription: {str(e)}", 500

@app.route('/prescription/<int:appointment_id>')
@doctor_login_required
def view_prescription(appointment_id):
    try:
        appointment = Appointment.query.get(appointment_id)
        if not appointment:
            return "Appointment not found", 404
            
        patient = Patient.query.get(appointment.patient_id)
        doctor = Doctor.query.get(appointment.doctor_id)
        prescription = Prescription.query.filter_by(appointment_id=appointment_id).first()
        
        if not prescription:
            return redirect(url_for('create_prescription', appointment_id=appointment_id))
        
        current_time = datetime.now(IST).strftime("%d-%m-%Y %I:%M:%S %p")
        
        return f'''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Prescription - MediQueue+</title>
            <style>
                body {{ font-family: Arial; margin: 40px; background: white; }}
                .container {{ max-width: 600px; margin: 0 auto; }}
                .prescription-header {{ text-align: center; border-bottom: 2px solid #3498db; padding-bottom: 20px; margin-bottom: 30px; }}
                .prescription-body {{ margin: 30px 0; }}
                .section {{ margin-bottom: 25px; }}
                .medicines {{ background: #f8f9fa; padding: 20px; border-radius: 8px; }}
                .footer {{ margin-top: 40px; text-align: center; color: #7f8c8d; font-size: 14px; }}
                @media print {{ 
                    .no-print {{ display: none; }} 
                    body {{ margin: 20px; }}
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="prescription-header">
                    <h1 style="color: #2c3e50; margin-bottom: 5px;">🏥 MediQueue+</h1>
                    <h2 style="color: #3498db; margin-top: 0;">MEDICAL PRESCRIPTION</h2>
                </div>
                
                <div class="prescription-body">
                    <div class="section">
                        <h3>Patient Details</h3>
                        <p><strong>Name:</strong> {patient.name}</p>
                        <p><strong>Phone:</strong> {patient.phone}</p>
                        <p><strong>Date:</strong> {current_time}</p>
                    </div>
                    
                    <div class="section">
                        <h3>Diagnosis</h3>
                        <p>{appointment.symptoms}</p>
                    </div>
                    
                    <div class="section">
                        <h3>Prescribed Medicines</h3>
                        <div class="medicines">
                            {prescription.medicines.replace(chr(10), '<br>')}
                        </div>
                    </div>
                    
                    {f'''<div class="section">
                        <h3>Additional Instructions</h3>
                        <p>{prescription.instructions.replace(chr(10), '<br>')}</p>
                    </div>''' if prescription.instructions else ''}
                    
                    <div class="section">
                        <h3>Doctor's Signature</h3>
                        <p><strong>Dr. {doctor.name}</strong></p>
                        <p>{doctor.specialty}</p>
                    </div>
                </div>
                
                <div class="footer">
                    <p>This is a computer-generated prescription. No physical signature required.</p>
                    <p>MediQueue+ - Smart Healthcare Management System</p>
                </div>
                
                <div class="no-print" style="margin-top: 30px; text-align: center;">
                    <button onclick="window.print()" style="background: #3498db; color: white; padding: 12px 25px; border: none; border-radius: 5px; cursor: pointer; margin-right: 10px;">🖨️ Print Prescription</button>
                    <a href="/doctor" style="background: #95a5a6; color: white; padding: 12px 25px; text-decoration: none; border-radius: 5px; display: inline-block;">← Back to Dashboard</a>
                </div>
            </div>
        </body>
        </html>
        '''
        
    except Exception as e:
        return f"Error loading prescription: {str(e)}", 500

# ===== EMERGENCY ROUTES =====

@app.route('/sos')
def sos():
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Emergency SOS - MediQueue+</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body { 
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                background: linear-gradient(135deg, #ff7675 0%, #d63031 100%);
                min-height: 100vh;
                color: white;
            }
            .container { 
                max-width: 600px; 
                margin: 0 auto; 
                padding: 40px 20px; 
            }
            .back-btn { 
                display: inline-flex;
                align-items: center;
                padding: 12px 25px; 
                background: rgba(255,255,255,0.2); 
                color: white; 
                text-decoration: none; 
                border-radius: 10px; 
                margin-bottom: 30px;
                font-weight: 600;
                backdrop-filter: blur(10px);
            }
            .emergency-card {
                background: rgba(255,255,255,0.95);
                padding: 50px 40px;
                border-radius: 20px;
                text-align: center;
                color: #2c3e50;
                box-shadow: 0 20px 40px rgba(0,0,0,0.2);
            }
            .emergency-icon {
                font-size: 4em;
                margin-bottom: 20px;
            }
            h1 { 
                color: #e74c3c;
                margin-bottom: 20px;
                font-size: 2.5em;
            }
            .warning {
                background: #fff3cd;
                color: #856404;
                padding: 20px;
                border-radius: 10px;
                margin: 25px 0;
                border-left: 4px solid #ffc107;
            }
            .form-group {
                margin-bottom: 25px;
                text-align: left;
            }
            label {
                display: block;
                margin-bottom: 10px;
                font-weight: 600;
                color: #2c3e50;
            }
            input, textarea {
                width: 100%;
                padding: 15px;
                border: 2px solid #e0e6ed;
                border-radius: 10px;
                font-size: 16px;
                box-sizing: border-box;
                transition: all 0.3s ease;
            }
            input:focus, textarea:focus {
                border-color: #e74c3c;
                outline: none;
                box-shadow: 0 0 0 3px rgba(231, 76, 60, 0.1);
            }
            .sos-button {
                background: linear-gradient(135deg, #e74c3c, #c0392b);
                color: white;
                padding: 20px 50px;
                border: none;
                border-radius: 15px;
                font-size: 20px;
                font-weight: bold;
                cursor: pointer;
                transition: all 0.3s ease;
                width: 100%;
                margin-top: 20px;
            }
            .sos-button:hover {
                transform: scale(1.05);
                box-shadow: 0 15px 30px rgba(231, 76, 60, 0.4);
            }
        </style>
    </head>
    <body>
        <div class="container">
            <a href="/" class="back-btn">← Back to Home</a>
            
            <div class="emergency-card">
                <div class="emergency-icon">🚨</div>
                <h1>EMERGENCY SOS</h1>
                <p style="font-size: 1.2em; margin-bottom: 30px; color: #7f8c8d;">
                    Immediate medical attention required. Please provide details below.
                </p>
                
                <div class="warning">
                    <strong>⚠️ CRITICAL WARNING:</strong> This service is for genuine emergencies only. 
                    In case of life-threatening situations, call your local emergency number immediately.
                </div>
                
                <!-- EMERGENCY FORM - NO PAYMENT REQUIRED -->
                <form method="POST" action="/trigger-emergency">
                    <div class="form-group">
                        <label for="patient_name">Your Name:</label>
                        <input type="text" id="patient_name" name="patient_name" required placeholder="Enter your full name">
                    </div>
                    
                    <div class="form-group">
                        <label for="patient_phone">Phone Number:</label>
                        <input type="text" id="patient_phone" name="patient_phone" required placeholder="Emergency contact number">
                    </div>
                    
                    <div class="form-group">
                        <label for="symptoms">Emergency Description:</label>
                        <textarea id="symptoms" name="symptoms" rows="4" required placeholder="Describe the emergency situation in detail..."></textarea>
                    </div>
                    
                    <button type="submit" class="sos-button">
                        🚨 TRIGGER EMERGENCY ALERT
                    </button>
                </form>
            </div>
        </div>
    </body>
    </html>
    '''

@app.route('/trigger-emergency', methods=['POST'])
def trigger_emergency():
    try:
        patient_name = request.form['patient_name']
        patient_phone = request.form['patient_phone']
        symptoms = request.form['symptoms']
        
        emergency = EmergencyAlert(
            patient_name=patient_name,
            patient_phone=patient_phone,
            symptoms=symptoms
        )
        db.session.add(emergency)
        db.session.commit()
        
        # Emit socket event for real-time notification
        socketio.emit('new_emergency', {
            'patient_name': patient_name,
            'symptoms': symptoms,
            'timestamp': datetime.now(IST).strftime("%I:%M %p")
        })
        
        return '''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Emergency Alert Sent - MediQueue+</title>
            <style>
                body { font-family: Arial; margin: 40px; background: #f0f8ff; }
                .container { max-width: 500px; margin: 0 auto; background: white; padding: 40px; border-radius: 15px; box-shadow: 0 10px 30px rgba(0,0,0,0.1); text-align: center; }
            </style>
        </head>
        <body>
            <div class="container">
                <div style="font-size: 48px; margin-bottom: 20px;">✅</div>
                <h2 style="color: #27ae60;">Emergency Alert Sent!</h2>
                <p>Medical staff has been notified and will respond immediately.</p>
                <p><strong>Please stay on the line and wait for assistance.</strong></p>
                <a href="/" style="display: inline-block; padding: 12px 25px; background: #3498db; color: white; text-decoration: none; border-radius: 8px; margin-top: 20px;">Back to Home</a>
            </div>
        </body>
        </html>
        '''
        
    except Exception as e:
        return f"Error triggering emergency: {str(e)}", 500

@app.route('/handle-emergency/<int:emergency_id>/<action>', methods=['POST'])
@doctor_login_required
def handle_emergency(emergency_id, action):
    try:
        emergency = EmergencyAlert.query.get(emergency_id)
        if emergency:
            if action == 'accepted':
                emergency.status = 'accepted'
            elif action == 'completed':
                emergency.status = 'completed'
            db.session.commit()
        return '', 200
    except Exception as e:
        return str(e), 500

@app.route('/complete-appointment/<int:appointment_id>')
@doctor_login_required
def complete_appointment(appointment_id):
    try:
        appointment = Appointment.query.get(appointment_id)
        if appointment:
            appointment.status = 'completed'
            db.session.commit()
        return redirect(url_for('doctor_dashboard'))
    except Exception as e:
        return f"Error completing appointment: {str(e)}", 500

# ===== DEBUG ROUTES =====

@app.route('/debug-doctors')
def debug_doctors():
    doctors = Doctor.query.all()
    doctor_list = "<h2>Current Doctors in Database:</h2><ul>"
    for doctor in doctors:
        doctor_list += f"<li>Dr. {doctor.name} - {doctor.specialty}</li>"
    doctor_list += "</ul>"
    return doctor_list

@app.route('/debug-specialty/<specialty>')
def debug_specialty(specialty):
    """Debug route to check if doctors exist for a specific specialty"""
    doctors = Doctor.query.filter_by(specialty=specialty).all()
    result = f"<h2>Doctors with specialty: '{specialty}'</h2>"
    if doctors:
        result += "<ul>"
        for doctor in doctors:
            result += f"<li>Dr. {doctor.name} - {doctor.specialty} (ID: {doctor.id})</li>"
        result += "</ul>"
    else:
        result += "<p>No doctors found!</p>"
    
    # Also show all doctors for reference
    all_doctors = Doctor.query.all()
    result += "<h3>All Doctors:</h3><ul>"
    for doc in all_doctors:
        result += f"<li>Dr. {doc.name} - {doc.specialty}</li>"
    result += "</ul>"
    
    return result

# ===== SOCKET EVENTS =====

@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

# ===== STATIC FILES =====

@app.route('/static/payments/<filename>')
def serve_payment_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# ===== INITIALIZE DATABASE =====

with app.app_context():
    db.create_all()
    if Doctor.query.count() == 0:
        doctors = [
            Doctor(name="Sharma", specialty="Psoriasis"),
            Doctor(name="Patel", specialty="Varicose Veins"),
            Doctor(name="Kumar", specialty="Asthma"),
            Doctor(name="Gupta", specialty="Chronic Kidney Disease"),
            Doctor(name="Singh", specialty="Migraine"),
            Doctor(name="Reddy", specialty="Gastritis"),
            Doctor(name="Yadav", specialty="Anemia"),
            Doctor(name="Pal", specialty="Osteoarthritis"),
            Doctor(name="Gour", specialty="Chickenpox"),
            Doctor(name="Verma", specialty="Diabetes"),
            Doctor(name="Tiwari", specialty="Hypertension"),
            Doctor(name="Mishra", specialty="General Physician"),
            Doctor(name="Saxena", specialty="COVID-19"),
            Doctor(name="Thakur", specialty="Tuberculosis"),
            Doctor(name="Dubey", specialty="Allergy"),
            Doctor(name="Shukla", specialty="Depression"),
            Doctor(name="Bajpai", specialty="Heart Attack"),
            Doctor(name="Khanna", specialty="Stroke"),
            Doctor(name="Mehra", specialty="Kidney Stones"),
            Doctor(name="Joshi", specialty="General Physician"),
        ]
        db.session.add_all(doctors)
        db.session.commit()
        print("✅ Database initialized with sample doctors")

# ===== RUN APPLICATION =====

if __name__ == '__main__':
    # Create upload folder if it doesn't exist
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)