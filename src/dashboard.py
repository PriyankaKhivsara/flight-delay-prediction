"""
Streamlit dashboard for flight delay prediction system
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import json
from datetime import datetime, timedelta
import time

from predict import FlightDelayPredictor
from config import API_CONFIG


# Page configuration
st.set_page_config(
    page_title="Flight Delay Prediction Dashboard",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .status-high {
        color: #d32f2f;
        font-weight: bold;
    }
    .status-medium {
        color: #f57f17;
        font-weight: bold;
    }
    .status-low {
        color: #388e3c;
        font-weight: bold;
    }
    .header-style {
        font-size: 2rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


class FlightDashboard:
    """Main dashboard class"""
    
    def __init__(self):
        self.api_base_url = f"http://localhost:{API_CONFIG['port']}"
        
    def render_header(self):
        """Render dashboard header"""
        st.markdown('<div class="header-style">✈️ Flight Delay Prediction Dashboard</div>', 
                   unsafe_allow_html=True)
        st.markdown("Real-time weather-related flight delay predictions")
        st.markdown("---")
    
    def render_sidebar(self):
        """Render sidebar with controls"""
        st.sidebar.header("🛠️ Controls")
        
        # Prediction settings
        st.sidebar.subheader("Single Flight Prediction")
        
        origin = st.sidebar.selectbox(
            "Origin Airport",
            ["JFK", "LAX", "ORD", "ATL", "DFW", "DEN", "SFO", "LAS", "PHX", "IAH"],
            index=0
        )
        
        destination = st.sidebar.selectbox(
            "Destination Airport", 
            ["LAX", "JFK", "ORD", "ATL", "DFW", "DEN", "SFO", "LAS", "PHX", "IAH"],
            index=1
        )
        
        date = st.sidebar.date_input(
            "Flight Date",
            value=datetime.now().date(),
            min_value=datetime.now().date(),
            max_value=datetime.now().date() + timedelta(days=30)
        )
        
        dep_time = st.sidebar.time_input(
            "Departure Time",
            value=datetime.now().time()
        )
        
        weather_type = st.sidebar.selectbox(
            "Weather Condition",
            ["Clear", "Partly Cloudy", "Cloudy", "Rain", "Snow", "Thunderstorm", "Fog"]
        )
        
        precipitation = st.sidebar.slider(
            "Precipitation (inches)",
            min_value=0.0,
            max_value=2.0,
            value=0.0,
            step=0.1
        )
        
        if st.sidebar.button("🔮 Predict Delay", type="primary"):
            self.make_prediction(origin, destination, date, dep_time, weather_type, precipitation)
        
        st.sidebar.markdown("---")
        
        # Dashboard controls
        st.sidebar.subheader("Dashboard Settings")
        
        auto_refresh = st.sidebar.checkbox("Auto Refresh", value=False)
        if auto_refresh:
            refresh_interval = st.sidebar.slider("Refresh Interval (seconds)", 5, 60, 30)
            if st.sidebar.button("Refresh Now"):
                st.experimental_rerun()
        
        return {
            "origin": origin,
            "destination": destination,
            "date": date,
            "dep_time": dep_time,
            "weather_type": weather_type,
            "precipitation": precipitation
        }
    
    def make_prediction(self, origin, destination, date, dep_time, weather_type, precipitation):
        """Make API call for prediction"""
        try:
            # Convert time to 24-hour format
            dep_time_int = dep_time.hour * 100 + dep_time.minute
            
            # Prepare request data
            request_data = {
                "origin": origin,
                "destination": destination,
                "date": date.strftime("%Y-%m-%d"),
                "crs_dep_time": dep_time_int,
                "weather_type": weather_type,
                "precipitation_in": precipitation
            }
            
            # Make API call (fallback to local prediction if API unavailable)
            try:
                response = requests.post(f"{self.api_base_url}/predict", json=request_data, timeout=5)
                if response.status_code == 200:
                    result = response.json()
                else:
                    raise Exception(f"API error: {response.status_code}")
            except:
                # Fallback to local prediction
                predictor = FlightDelayPredictor()
                result = predictor.predict_single_flight(request_data)
            
            # Store result in session state
            st.session_state.last_prediction = result
            st.session_state.last_request = request_data
            
        except Exception as e:
            st.error(f"Prediction failed: {e}")
    
    def render_prediction_results(self):
        """Render prediction results section"""
        if hasattr(st.session_state, 'last_prediction'):
            result = st.session_state.last_prediction
            request = st.session_state.last_request
            
            st.subheader("🔮 Latest Prediction")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    label="Route",
                    value=f"{request['origin']} → {request['destination']}"
                )
            
            with col2:
                # Color code based on prediction
                if result['prediction'] == 'High':
                    color = "🔴"
                elif result['prediction'] == 'Medium':
                    color = "🟡"
                else:
                    color = "🟢"
                
                st.metric(
                    label="Delay Risk",
                    value=f"{color} {result['prediction']}"
                )
            
            with col3:
                st.metric(
                    label="Probability",
                    value=f"{result['probability']:.1%}"
                )
            
            # Key factors
            if 'key_factors' in result:
                st.subheader("📊 Key Factors")
                factors_df = pd.DataFrame(result['key_factors'])
                
                fig = px.bar(
                    factors_df,
                    x='impact',
                    y='feature',
                    orientation='h',
                    title="Factors Contributing to Prediction",
                    color='impact',
                    color_continuous_scale='RdYlGn_r'
                )
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
    
    def render_flight_status_board(self):
        """Render flight status board"""
        st.subheader("📋 Flight Status Board")
        
        # Generate sample flight data
        sample_flights = self.generate_sample_flights()
        
        # Color mapping for status
        def color_status(val):
            if val == 'High':
                return 'background-color: #ffebee'
            elif val == 'Medium':
                return 'background-color: #fff8e1'
            else:
                return 'background-color: #e8f5e8'
        
        # Display flights table
        styled_df = sample_flights.style.applymap(color_status, subset=['Prediction'])
        st.dataframe(styled_df, use_container_width=True)
        
        # Summary statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_flights = len(sample_flights)
            st.metric("Total Flights", total_flights)
        
        with col2:
            high_risk = len(sample_flights[sample_flights['Prediction'] == 'High'])
            st.metric("High Risk", high_risk, delta=f"{high_risk/total_flights:.1%}")
        
        with col3:
            medium_risk = len(sample_flights[sample_flights['Prediction'] == 'Medium'])
            st.metric("Medium Risk", medium_risk, delta=f"{medium_risk/total_flights:.1%}")
        
        with col4:
            low_risk = len(sample_flights[sample_flights['Prediction'] == 'Low'])
            st.metric("Low Risk", low_risk, delta=f"{low_risk/total_flights:.1%}")
    
    def render_weather_panel(self):
        """Render weather impact panel"""
        st.subheader("🌤️ Weather Impact Panel")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Precipitation gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=0.4,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Precipitation (inches)"},
                delta={'reference': 0.2},
                gauge={
                    'axis': {'range': [None, 2]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 0.1], 'color': "lightgray"},
                        {'range': [0.1, 0.5], 'color': "yellow"},
                        {'range': [0.5, 1.0], 'color': "orange"},
                        {'range': [1.0, 2.0], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 1.0
                    }
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Weather severity indicators
            weather_conditions = {
                "Snow": 85,
                "Thunderstorm": 78,
                "Rain": 45,
                "Fog": 35,
                "Cloudy": 15,
                "Clear": 5
            }
            
            weather_df = pd.DataFrame(list(weather_conditions.items()), 
                                    columns=['Condition', 'Delay_Risk'])
            
            fig = px.bar(
                weather_df,
                x='Delay_Risk',
                y='Condition',
                orientation='h',
                title="Weather Condition Delay Risk (%)",
                color='Delay_Risk',
                color_continuous_scale='RdYlGn_r'
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
    
    def render_historical_trends(self):
        """Render historical trends chart"""
        st.subheader("📈 Historical Delay Trends (30 Days)")
        
        # Generate sample historical data
        dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
        historical_data = []
        
        for date in dates:
            # Simulate seasonal patterns
            base_prob = 0.2 + 0.3 * np.sin(2 * np.pi * date.dayofyear / 365)
            weather_factor = np.random.normal(0, 0.1)
            weekend_factor = 0.1 if date.weekday() >= 5 else 0
            
            delay_prob = max(0, min(1, base_prob + weather_factor + weekend_factor))
            
            historical_data.append({
                'Date': date,
                'Delay_Probability': delay_prob,
                'Weather_Factor': weather_factor,
                'Is_Weekend': date.weekday() >= 5
            })
        
        hist_df = pd.DataFrame(historical_data)
        
        # Create subplot with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add delay probability line
        fig.add_trace(
            go.Scatter(
                x=hist_df['Date'],
                y=hist_df['Delay_Probability'],
                mode='lines+markers',
                name='Delay Probability',
                line=dict(color='red', width=2)
            ),
            secondary_y=False,
        )
        
        # Add weather factor bars
        fig.add_trace(
            go.Scatter(
                x=hist_df['Date'],
                y=hist_df['Weather_Factor'],
                mode='lines',
                name='Weather Impact',
                line=dict(color='blue', width=1),
                opacity=0.6
            ),
            secondary_y=True,
        )
        
        # Set y-axes titles
        fig.update_yaxes(title_text="Delay Probability", secondary_y=False)
        fig.update_yaxes(title_text="Weather Impact", secondary_y=True)
        
        fig.update_layout(
            title="30-Day Delay Probability Trend",
            xaxis_title="Date",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Summary statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            avg_delay = hist_df['Delay_Probability'].mean()
            st.metric("Average Delay Risk", f"{avg_delay:.1%}")
        
        with col2:
            max_delay = hist_df['Delay_Probability'].max()
            max_date = hist_df.loc[hist_df['Delay_Probability'].idxmax(), 'Date']
            st.metric("Highest Risk Day", f"{max_delay:.1%}", 
                     delta=max_date.strftime("%m/%d"))
        
        with col3:
            weekend_avg = hist_df[hist_df['Is_Weekend']]['Delay_Probability'].mean()
            weekday_avg = hist_df[~hist_df['Is_Weekend']]['Delay_Probability'].mean()
            weekend_diff = weekend_avg - weekday_avg
            st.metric("Weekend vs Weekday", f"{weekend_diff:+.1%}", 
                     delta="Weekend Effect")
    
    def generate_sample_flights(self):
        """Generate sample flight data for status board"""
        airports = ["JFK", "LAX", "ORD", "ATL", "DFW", "DEN", "SFO", "LAS", "PHX", "IAH"]
        airlines = ["AA", "DL", "UA", "WN", "B6"]
        
        flights = []
        for i in range(15):
            origin = np.random.choice(airports)
            dest = np.random.choice([a for a in airports if a != origin])
            
            # Simulate departure times
            dep_hour = np.random.randint(6, 23)
            dep_min = np.random.choice([0, 15, 30, 45])
            dep_time = f"{dep_hour:02d}:{dep_min:02d}"
            
            # Simulate delay probability based on various factors
            base_prob = 0.3
            weather_prob = np.random.uniform(0, 0.4)
            delay_prob = min(1.0, base_prob + weather_prob)
            
            # Classify prediction
            if delay_prob < 0.3:
                prediction = "Low"
            elif delay_prob < 0.7:
                prediction = "Medium"
            else:
                prediction = "High"
            
            flights.append({
                "Flight": f"{np.random.choice(airlines)}{np.random.randint(100, 9999)}",
                "Origin": origin,
                "Dest": dest,
                "Scheduled": dep_time,
                "Prediction": prediction,
                "Probability": f"{delay_prob:.1%}"
            })
        
        return pd.DataFrame(flights)
    
    def render_system_status(self):
        """Render system status section"""
        st.subheader("⚙️ System Status")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            try:
                # Check API health
                response = requests.get(f"{self.api_base_url}/health", timeout=2)
                if response.status_code == 200:
                    st.success("🟢 API Online")
                else:
                    st.error("🔴 API Error")
            except:
                st.warning("🟡 API Offline (Local Mode)")
        
        with col2:
            # Model status
            st.info("🔵 Model Loaded")
        
        with col3:
            # Performance metrics
            st.metric("Target F1 Score", "0.78")
    
    def run(self):
        """Main dashboard function"""
        self.render_header()
        
        # Sidebar
        settings = self.render_sidebar()
        
        # Main content
        self.render_prediction_results()
        
        # Create tabs for different sections
        tab1, tab2, tab3, tab4 = st.tabs(["📋 Flight Status", "🌤️ Weather Panel", 
                                          "📈 Historical Trends", "⚙️ System Status"])
        
        with tab1:
            self.render_flight_status_board()
        
        with tab2:
            self.render_weather_panel()
        
        with tab3:
            self.render_historical_trends()
        
        with tab4:
            self.render_system_status()
        
        # Footer
        st.markdown("---")
        st.markdown("Built with Streamlit • Flight Delay Prediction System v1.0")


def main():
    """Main function to run the dashboard"""
    dashboard = FlightDashboard()
    dashboard.run()


if __name__ == "__main__":
    main()