# AquaRisk – Intelligent Risk Prediction for Water-Borne Disease Outbreaks

## Project Structure
```
water_disease_app/
├── app.py                  ← Main Flask application
├── requirements.txt        ← Python dependencies
├── templates/
│   ├── base.html           ← Sidebar layout (shared)
│   ├── home.html           ← Landing page (public)
│   ├── login.html          ← User + Admin login
│   ├── dashboard.html      ← Stats, charts, quick actions
│   ├── predict.html        ← Prediction form + Confusion Matrix + Feature Importance
│   ├── alerts.html         ← Active + Resolved alerts
│   ├── reports.html        ← Analytics & classification report
│   ├── admin.html          ← Admin panel (admin role only)
│   └── error.html          ← Error page
└── static/                 ← (optional for extra CSS/JS/images)
```

## Setup & Run (Windows)

### Step 1 – Create virtual environment
```
cd water_disease_app
python -m venv venv
venv\Scripts\activate
```

### Step 2 – Install dependencies
```
pip install -r requirements.txt
```

### Step 3 – Run the app
```
python app.py
```

### Step 4 – Open in browser
```
http://localhost:5000
```

## Login Credentials
| Username | Password | Role  |
|----------|----------|-------|
| admin    | admin123 | Admin |
| user1    | user123  | User  |
| user2    | user456  | User  |

## Features
- **Home Page** – Animated landing with feature overview and workflow
- **Login** – Dual tab (User/Admin) with demo credential hints
- **Dashboard** – 8 stat cards, risk distribution chart, accuracy comparison, region & season charts
- **Predict** – 12-parameter form, RF or DT selection, risk result with confidence, probabilities, confusion matrix, feature importance
- **Alerts** – Auto-generated for High/Critical predictions; resolve from panel
- **Reports** – Seasonal/regional bar charts, classification report table, model accuracy progress bars
- **Admin Panel** – User list, recent alerts, system info (admin only)
- **Logout** – Clears session

## ML Details
- **Dataset**: 1000 synthetic records generated on startup
- **Features**: temperature, rainfall, pH, turbidity, dissolved oxygen, coliform count, population density, sanitation index, flood occurrence, region, season, water source
- **Target**: Risk Level (Low / Medium / High / Critical)
- **Models**: Random Forest (100 estimators) + Decision Tree (max_depth=8)
- **Split**: 80% train, 20% test
