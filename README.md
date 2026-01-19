# EV-Bus Intelligent Fleet Analytics (EV_Bus_Predictor_ML)

An intelligent, Machine Learning-driven platform designed for fixed-route EV bus fleets. This system leverages historical battery telemetry to predict trip feasibility, monitor battery aging (SOH), and optimize fleet readiness to prevent mid-route breakdowns.

# 🚀 Live Demo

Model View & Prediction Engine: View on Streamlit: http://ev-bus-fleet-battery-ai.streamlit.app/

# Dashboard Gallery

# Fleet Command Center

The central dashboard providing real-time KPIs including Operational Readiness, Battery Health (SOH), and Energy Demand Trends.

# Trip Forecast Engine

AI-simulated battery usage based on environmental factors (like temperature) and route load. It provides a Predicted End SoC and a Recommended Speed.

# Route Monitor & Geospatial View

Live tracking of the fleet with telemetry updates and status alerts (Critical/Good) mapped across the city.

# Technical Stack

**Frontend:** Crafted a responsive UI using HTML5, CSS3, and JavaScript for high-fidelity, real-time data visualization and interactive mapping.

**Backend:** Powered by Python and Flask, providing a robust RESTful API for seamless data flow between the ML models and the UI.

**Machine Learning:** Implemented and compared Linear Regression and Random Forest Regressor models to achieve high-accuracy battery state and discharge predictions.

**Database:** Utilized a MongoDB Cluster (Atlas) for scalable, high-performance storage of historical telemetry and fleet logs.

**Deployment:** Model preview and interactive testing hosted via Streamlit.

# System Workflow

The project follows a modular 4-step pipeline to ensure data integrity and prediction accuracy:

- Model Development: Data preprocessing and training in Jupyter Notebooks.

- Backend Integration: Flask API connects the trained model to the live database.

- API & JS Logic: Fetches real-time telemetry and processes ML results.

- Frontend Implementation: Displays interactive charts, route risks, and fleet alerts.

# Key Features

- Predictive Maintenance: Monitor State of Health (SOH) and maintenance status to move from reactive to proactive fleet care.

- Intelligent Trip Forecast: Input route data and environmental conditions (e.g., Extreme Cold) to see a predicted battery discharge curve.

- CO2 Savings Tracker: Real-time calculation of environmental impact based on EV mileage.

- Fleet Logs: Filterable records of bus status, maintenance due dates, and detected issues.

# Installation & Setup

Clone the Repository:

git clone https://github.com/yash-s29/EV-Bus-Intelligent-Fleet-Analytics.git

Install Dependencies:

pip install -r requirements.txt

Environment Configuration: Create a .env file in the root directory and add your MongoDB credentials:

MONGODB_URI=mongodb+srv://<username>:<password>@cluster.mongodb.net/

Run the Flask App:

python app.py

# Acknowledgments
Developed during the Advanced Course Training & Internship Program by Edunet Foundation in collaboration with Shell Market Pvt Ltd.

