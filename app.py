from flask import Flask, render_template, request, redirect, url_for, session, jsonify, send_file
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import json
import os
import pickle
import datetime
import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from functools import wraps
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
app.secret_key = 'waterborne_risk_secret_2024'

# ─── USERS ────────────────────────────────────────────────────────────────────
USERS = {
    'admin': {'password': 'admin123', 'role': 'admin', 'name': 'Admin User'},
    'user1': {'password': 'user123',  'role': 'user',  'name': 'John Doe'},
    'user2': {'password': 'user456',  'role': 'user',  'name': 'Jane Smith'},
}

# ─── ALERTS STORE ─────────────────────────────────────────────────────────────
ALERTS = []

# ─── AUTH DECORATORS ──────────────────────────────────────────────────────────
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'username' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated

def admin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'username' not in session:
            return redirect(url_for('login'))
        if session.get('role') != 'admin':
            return render_template('error.html', message='Admin access required.')
        return f(*args, **kwargs)
    return decorated

# ─── DATASET GENERATION ───────────────────────────────────────────────────────
def generate_dataset():
    np.random.seed(42)
    n = 1000
    regions = ['Hyderabad', 'Mumbai', 'Delhi', 'Chennai', 'Kolkata',
               'Bangalore', 'Pune', 'Ahmedabad', 'Jaipur', 'Lucknow']
    seasons = ['Summer', 'Monsoon', 'Winter', 'Spring']

    data = {
        'region': np.random.choice(regions, n),
        'season': np.random.choice(seasons, n),
        'temperature': np.random.uniform(15, 45, n).round(1),
        'rainfall_mm': np.random.uniform(0, 300, n).round(1),
        'ph_level': np.random.uniform(5.5, 9.5, n).round(2),
        'turbidity_ntu': np.random.uniform(0, 50, n).round(2),
        'dissolved_oxygen': np.random.uniform(1, 14, n).round(2),
        'coliform_count': np.random.uniform(0, 500, n).round(0),
        'population_density': np.random.randint(100, 15000, n),
        'sanitation_index': np.random.uniform(0, 10, n).round(2),
        'flood_occurrence': np.random.randint(0, 2, n),
        'water_source': np.random.choice(['River', 'Well', 'Municipal', 'Borewell', 'Lake'], n),
    }

    df = pd.DataFrame(data)

    # Risk label logic
    risk_score = (
        (df['coliform_count'] > 200).astype(int) * 3 +
        (df['turbidity_ntu'] > 20).astype(int) * 2 +
        (df['ph_level'] < 6.5).astype(int) * 2 +
        (df['ph_level'] > 8.5).astype(int) * 1 +
        (df['dissolved_oxygen'] < 4).astype(int) * 2 +
        (df['rainfall_mm'] > 150).astype(int) * 2 +
        (df['temperature'] > 35).astype(int) * 1 +
        (df['flood_occurrence'] == 1).astype(int) * 3 +
        (df['sanitation_index'] < 4).astype(int) * 2 +
        (df['population_density'] > 8000).astype(int) * 1
    )

    df['risk_level'] = pd.cut(risk_score, bins=[-1, 3, 6, 10, 100],
                               labels=['Low', 'Medium', 'High', 'Critical'])
    return df

# ─── MODEL TRAINING ───────────────────────────────────────────────────────────
def train_models():
    df = generate_dataset()
    le_region  = LabelEncoder()
    le_season  = LabelEncoder()
    le_source  = LabelEncoder()
    le_risk    = LabelEncoder()

    df['region_enc']      = le_region.fit_transform(df['region'])
    df['season_enc']      = le_season.fit_transform(df['season'])
    df['water_source_enc']= le_source.fit_transform(df['water_source'])
    df['risk_enc']        = le_risk.fit_transform(df['risk_level'])

    features = ['temperature','rainfall_mm','ph_level','turbidity_ntu',
                'dissolved_oxygen','coliform_count','population_density',
                'sanitation_index','flood_occurrence',
                'region_enc','season_enc','water_source_enc']

    X = df[features]
    y = df['risk_enc']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf  = RandomForestClassifier(n_estimators=100, random_state=42)
    dt  = DecisionTreeClassifier(max_depth=8, random_state=42)
    rf.fit(X_train, y_train)
    dt.fit(X_train, y_train)

    models = {
        'rf': rf, 'dt': dt,
        'le_region': le_region, 'le_season': le_season,
        'le_source': le_source, 'le_risk': le_risk,
        'features': features,
        'X_test': X_test, 'y_test': y_test,
        'df': df
    }
    return models

MODELS = train_models()

# ─── PLOT HELPER ──────────────────────────────────────────────────────────────
def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=110,
                facecolor='#0d1117', edgecolor='none')
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_b64

def get_confusion_matrix_plot(model_name='rf'):
    model = MODELS[model_name]
    y_pred = model.predict(MODELS['X_test'])
    y_test = MODELS['y_test']
    cm = confusion_matrix(y_test, y_pred)
    labels = MODELS['le_risk'].classes_

    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#0d1117')

    sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd',
                xticklabels=labels, yticklabels=labels,
                ax=ax, linewidths=0.5, linecolor='#1e2a3a',
                annot_kws={'size': 13, 'weight': 'bold', 'color': 'white'})

    ax.set_title('Confusion Matrix', color='#00d4ff', fontsize=15, pad=12)
    ax.set_xlabel('Predicted Label', color='#8892a4', fontsize=11)
    ax.set_ylabel('True Label', color='#8892a4', fontsize=11)
    ax.tick_params(colors='#8892a4')
    for spine in ax.spines.values():
        spine.set_edgecolor('#1e2a3a')

    return fig_to_base64(fig)

def get_feature_importance_plot(model_name='rf'):
    model = MODELS[model_name]
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        return None

    feat_names = MODELS['features']
    sorted_idx = np.argsort(importances)[::-1]
    sorted_names = [feat_names[i].replace('_', ' ').title() for i in sorted_idx]
    sorted_vals  = importances[sorted_idx]

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#0d1117')

    colors = plt.cm.YlOrRd(np.linspace(0.3, 0.9, len(sorted_vals)))
    bars = ax.barh(sorted_names[::-1], sorted_vals[::-1], color=colors[::-1],
                   edgecolor='none', height=0.65)

    ax.set_title('Feature Importance', color='#00d4ff', fontsize=15, pad=12)
    ax.set_xlabel('Importance Score', color='#8892a4', fontsize=11)
    ax.tick_params(colors='#8892a4', labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor('#1e2a3a')
    ax.xaxis.grid(True, color='#1e2a3a', linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)

    return fig_to_base64(fig)

def get_risk_distribution_plot():
    df = MODELS['df']
    counts = df['risk_level'].value_counts()
    colors_map = {'Low': '#22c55e', 'Medium': '#f59e0b',
                  'High': '#f97316', 'Critical': '#ef4444'}
    fig, ax = plt.subplots(figsize=(6, 4))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#0d1117')

    bars = ax.bar(counts.index, counts.values,
                  color=[colors_map.get(c, '#00d4ff') for c in counts.index],
                  edgecolor='none', width=0.55)

    ax.set_title('Risk Level Distribution', color='#00d4ff', fontsize=14, pad=10)
    ax.set_ylabel('Count', color='#8892a4', fontsize=10)
    ax.tick_params(colors='#8892a4')
    for spine in ax.spines.values():
        spine.set_edgecolor('#1e2a3a')
    ax.yaxis.grid(True, color='#1e2a3a', linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)

    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 5,
                str(int(h)), ha='center', va='bottom',
                color='white', fontsize=9, fontweight='bold')

    return fig_to_base64(fig)

def get_accuracy_comparison():
    rf_acc = accuracy_score(MODELS['y_test'], MODELS['rf'].predict(MODELS['X_test']))
    dt_acc = accuracy_score(MODELS['y_test'], MODELS['dt'].predict(MODELS['X_test']))

    fig, ax = plt.subplots(figsize=(5, 4))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#0d1117')

    bars = ax.bar(['Random Forest', 'Decision Tree'],
                  [rf_acc * 100, dt_acc * 100],
                  color=['#00d4ff', '#f59e0b'], edgecolor='none', width=0.45)

    ax.set_title('Model Accuracy Comparison', color='#00d4ff', fontsize=14, pad=10)
    ax.set_ylabel('Accuracy (%)', color='#8892a4', fontsize=10)
    ax.set_ylim(0, 110)
    ax.tick_params(colors='#8892a4')
    for spine in ax.spines.values():
        spine.set_edgecolor('#1e2a3a')
    ax.yaxis.grid(True, color='#1e2a3a', linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)

    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 1,
                f'{h:.1f}%', ha='center', va='bottom',
                color='white', fontsize=10, fontweight='bold')

    return fig_to_base64(fig), round(rf_acc * 100, 2), round(dt_acc * 100, 2)

# ─── ROUTES ───────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()
        if username in USERS and USERS[username]['password'] == password:
            session['username'] = username
            session['role']     = USERS[username]['role']
            session['name']     = USERS[username]['name']
            return redirect(url_for('dashboard'))
        error = 'Invalid username or password.'
    return render_template('login.html', error=error)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/dashboard')
@login_required
def dashboard():
    df = MODELS['df']
    risk_counts = df['risk_level'].value_counts().to_dict()
    rf_acc = accuracy_score(MODELS['y_test'], MODELS['rf'].predict(MODELS['X_test']))
    dt_acc = accuracy_score(MODELS['y_test'], MODELS['dt'].predict(MODELS['X_test']))

    region_risk = df.groupby('region')['risk_level'].apply(
        lambda x: (x.isin(['High','Critical'])).sum()
    ).sort_values(ascending=False).to_dict()

    season_risk = df.groupby('season')['risk_level'].apply(
        lambda x: (x.isin(['High','Critical'])).sum()
    ).to_dict()

    dist_plot = get_risk_distribution_plot()
    acc_plot, rf_pct, dt_pct = get_accuracy_comparison()

    stats = {
        'total_records': len(df),
        'critical_cases': risk_counts.get('Critical', 0),
        'high_cases': risk_counts.get('High', 0),
        'medium_cases': risk_counts.get('Medium', 0),
        'low_cases': risk_counts.get('Low', 0),
        'rf_accuracy': round(rf_acc * 100, 2),
        'dt_accuracy': round(dt_acc * 100, 2),
        'active_alerts': len([a for a in ALERTS if not a.get('resolved', False)]),
    }

    return render_template('dashboard.html',
                           stats=stats,
                           region_risk=json.dumps(region_risk),
                           season_risk=json.dumps(season_risk),
                           dist_plot=dist_plot,
                           acc_plot=acc_plot)

@app.route('/predict', methods=['GET', 'POST'])
@login_required
def predict():
    result = None
    cm_plot = None
    fi_plot = None
    model_name = 'rf'

    if request.method == 'POST':
        model_name = request.form.get('model', 'rf')
        try:
            region   = request.form.get('region')
            season   = request.form.get('season')
            w_source = request.form.get('water_source')
            temp     = float(request.form.get('temperature'))
            rainfall = float(request.form.get('rainfall_mm'))
            ph       = float(request.form.get('ph_level'))
            turb     = float(request.form.get('turbidity_ntu'))
            do_val   = float(request.form.get('dissolved_oxygen'))
            coliform = float(request.form.get('coliform_count'))
            pop_den  = float(request.form.get('population_density'))
            sanit    = float(request.form.get('sanitation_index'))
            flood    = int(request.form.get('flood_occurrence'))

            le_region = MODELS['le_region']
            le_season = MODELS['le_season']
            le_source = MODELS['le_source']
            le_risk   = MODELS['le_risk']

            r_enc = le_region.transform([region])[0] if region in le_region.classes_ else 0
            s_enc = le_season.transform([season])[0] if season in le_season.classes_ else 0
            ws_enc= le_source.transform([w_source])[0] if w_source in le_source.classes_ else 0

            inp = np.array([[temp, rainfall, ph, turb, do_val, coliform,
                             pop_den, sanit, flood, r_enc, s_enc, ws_enc]])

            model = MODELS[model_name]
            pred_enc  = model.predict(inp)[0]
            pred_prob = model.predict_proba(inp)[0]
            risk_label = le_risk.inverse_transform([pred_enc])[0]
            confidence = round(max(pred_prob) * 100, 1)

            risk_colors = {
                'Low': '#22c55e', 'Medium': '#f59e0b',
                'High': '#f97316', 'Critical': '#ef4444'
            }

            result = {
                'risk_level': risk_label,
                'confidence': confidence,
                'color': risk_colors.get(risk_label, '#00d4ff'),
                'probabilities': {
                    le_risk.classes_[i]: round(p * 100, 1)
                    for i, p in enumerate(pred_prob)
                },
                'model_used': 'Random Forest' if model_name == 'rf' else 'Decision Tree',
                'region': region,
                'season': season,
            }

            # Auto-generate alert for High/Critical
            if risk_label in ['High', 'Critical']:
                ALERTS.append({
                    'id': len(ALERTS) + 1,
                    'region': region,
                    'risk_level': risk_label,
                    'message': f'{risk_label} risk detected in {region} during {season}.',
                    'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'resolved': False,
                    'triggered_by': session.get('name'),
                })

        except Exception as e:
            result = {'error': str(e)}

    cm_plot = get_confusion_matrix_plot(model_name)
    fi_plot = get_feature_importance_plot(model_name)

    regions     = list(MODELS['le_region'].classes_)
    seasons     = list(MODELS['le_season'].classes_)
    water_srcs  = list(MODELS['le_source'].classes_)

    return render_template('predict.html',
                           result=result,
                           cm_plot=cm_plot,
                           fi_plot=fi_plot,
                           regions=regions,
                           seasons=seasons,
                           water_sources=water_srcs,
                           model_name=model_name)

@app.route('/alerts')
@login_required
def alerts():
    active   = [a for a in ALERTS if not a.get('resolved', False)]
    resolved = [a for a in ALERTS if a.get('resolved', False)]
    return render_template('alerts.html', active=active, resolved=resolved)

@app.route('/resolve_alert/<int:alert_id>')
@login_required
def resolve_alert(alert_id):
    for a in ALERTS:
        if a['id'] == alert_id:
            a['resolved'] = True
            a['resolved_at'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            a['resolved_by'] = session.get('name')
            break
    return redirect(url_for('alerts'))

@app.route('/reports')
@login_required
def reports():
    df = MODELS['df']
    region_summary = df.groupby('region')['risk_level'].value_counts().unstack(fill_value=0).to_dict()
    season_summary = df.groupby('season')['risk_level'].value_counts().unstack(fill_value=0)

    rf_acc = accuracy_score(MODELS['y_test'], MODELS['rf'].predict(MODELS['X_test']))
    dt_acc = accuracy_score(MODELS['y_test'], MODELS['dt'].predict(MODELS['X_test']))

    cr = classification_report(
        MODELS['y_test'],
        MODELS['rf'].predict(MODELS['X_test']),
        target_names=MODELS['le_risk'].classes_,
        output_dict=True
    )

    season_data = {}
    for season in df['season'].unique():
        sub = df[df['season'] == season]['risk_level'].value_counts()
        season_data[season] = sub.to_dict()

    region_data = {}
    for region in df['region'].unique():
        sub = df[df['region'] == region]['risk_level'].value_counts()
        region_data[region] = sub.to_dict()

    return render_template('reports.html',
                           rf_acc=round(rf_acc*100,2),
                           dt_acc=round(dt_acc*100,2),
                           classification_report=cr,
                           season_data=json.dumps(season_data),
                           region_data=json.dumps(region_data),
                           total_alerts=len(ALERTS),
                           resolved_alerts=len([a for a in ALERTS if a.get('resolved')]),
                           total_records=len(df))

@app.route('/admin')
@admin_required
def admin():
    users_list = [{'username': u, **v} for u, v in USERS.items()]
    df = MODELS['df']
    stats = {
        'total_users': len(USERS),
        'total_records': len(df),
        'total_alerts': len(ALERTS),
        'active_alerts': len([a for a in ALERTS if not a.get('resolved', False)]),
        'rf_accuracy': round(accuracy_score(MODELS['y_test'],
                             MODELS['rf'].predict(MODELS['X_test'])) * 100, 2),
        'dt_accuracy': round(accuracy_score(MODELS['y_test'],
                             MODELS['dt'].predict(MODELS['X_test'])) * 100, 2),
    }
    return render_template('admin.html', users=users_list, stats=stats, alerts=ALERTS[-10:])

@app.route('/api/stats')
@login_required
def api_stats():
    df = MODELS['df']
    risk_counts = df['risk_level'].value_counts().to_dict()
    return jsonify({
        'risk_distribution': risk_counts,
        'total': len(df),
        'active_alerts': len([a for a in ALERTS if not a.get('resolved')])
    })

if __name__ == '__main__':
   import os
port = int(os.environ.get('PORT', 5000))
app.run(debug=False, host='0.0.0.0', port=port)
