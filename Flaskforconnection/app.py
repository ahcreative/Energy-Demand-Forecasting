from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import os
import joblib
from datetime import datetime, timedelta
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import the forecaster class
from electricity_forecaster import ElectricityDemandForecaster, run_forecasting_pipeline

app = Flask(__name__)
CORS(
    app,
    resources={r"/api/*": {"origins": "http://localhost:3000"}},
    supports_credentials=True
)

# Configuration
MODEL_DIR = 'models'
DATA_PATH = 'data/processed_energy_data.csv'

# Ensure directories exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs('data', exist_ok=True)

# Initialize global variables
forecaster = None
cities = None
data = None


@app.before_request
def initialize():
    """Initialize the application by loading forecaster and data"""
    global forecaster, cities, data

    # Skip initialization for some endpoints
    if forecaster is not None:
        return  # already initialized

    try:
        # Create forecaster instance if not exists
        if forecaster is None:
            forecaster = ElectricityDemandForecaster(data_path=DATA_PATH)

            # Load data
            try:
                data = forecaster.load_data()
                logger.info(f"Data loaded successfully with shape: {data.shape}")
            except Exception as e:
                logger.error(f"Data loading error: {str(e)}", exc_info=True)

            # Extract cities list
            if data is not None:
                cities = data['city'].unique().tolist()
                logger.info(f"Found {len(cities)} cities in the dataset")

            # Load pre-trained models if they exist
            if os.path.exists(MODEL_DIR) and any(file.endswith('.pkl') for file in os.listdir(MODEL_DIR)):
                try:
                    forecaster.load_models(directory=MODEL_DIR)
                    logger.info("Pre-trained models loaded successfully")
                except Exception as e:
                    logger.error(f"Error loading models: {str(e)}", exc_info=True)
            else:
                logger.info("No pre-trained models found")
    except Exception as e:
        logger.error(f"Initialization error: {str(e)}", exc_info=True)


@app.route('/api/cities', methods=['GET'])
def get_cities():
    """Get list of available cities"""
    global cities, data

    try:
        if cities is None or len(cities) == 0:
            # If cities list is not yet loaded, load it now
            if data is None:
                try:
                    temp_data = pd.read_csv(DATA_PATH)
                    cities = temp_data['city'].unique().tolist()
                except Exception as e:
                    logger.error(f"Error loading cities: {str(e)}")
                    return jsonify({'error': f"Failed to load data: {str(e)}"}), 500
            else:
                cities = data['city'].unique().tolist()

        return jsonify({'cities': cities})
    except Exception as e:
        logger.error(f"Error in get_cities: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/models', methods=['GET'])
def get_models():
    """Get list of available forecast models"""
    try:
        global forecaster

        # Check if forecaster is initialized
        if forecaster is None:
            forecaster = ElectricityDemandForecaster(data_path=DATA_PATH)

        # Check if forecaster has models loaded
        available_models = []
        if hasattr(forecaster, 'models') and forecaster.models:
            available_models = list(forecaster.models.keys())
        else:
            # Check for model files in the models directory
            if os.path.exists(MODEL_DIR):
                available_models = [f.split('.')[0] for f in os.listdir(MODEL_DIR)
                                    if f.endswith('.pkl') and not f.startswith('scaler')]

        # If no models found, list the available model types
        if not available_models:
            available_models = [
                'linear', 'random_forest', 'gradient_boosting',
                'xgboost', 'lstm', 'ensemble_avg', 'ensemble_stack'
            ]

        return jsonify({'models': available_models})
    except Exception as e:
        logger.error(f"Error in get_models: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/forecast', methods=['POST'])
def get_forecast():
    """Generate forecast for a city"""
    try:
        global forecaster, data

        # Log incoming request
        logger.info(f"Forecast request received: {request.json}")

        # Ensure forecaster is initialized
        if forecaster is None:
            forecaster = ElectricityDemandForecaster(data_path=DATA_PATH)

        # Load data if not already loaded
        if data is None or forecaster.data is None:
            data = forecaster.load_data()

        request_data = request.json

        # Validate required parameters
        required_params = ['city', 'start_date', 'end_date', 'model']
        if not all(param in request_data for param in required_params):
            missing_params = [p for p in required_params if p not in request_data]
            return jsonify({'error': f'Missing required parameters: {missing_params}'}), 400

        city = request_data['city'].lower()
        model_name = request_data['model']
        start_date = request_data['start_date']
        end_date = request_data['end_date']

        logger.info(f"Processing forecast for {city} using {model_name} from {start_date} to {end_date}")

        # Convert dates
        try:
            start_date = pd.to_datetime(start_date)
            end_date = pd.to_datetime(end_date)
        except Exception as e:
            logger.error(f"Date parsing error: {str(e)}", exc_info=True)
            return jsonify({'error': f'Invalid date format: {str(e)}'}), 400

        # Filter data for the selected city and date range
        df = forecaster.data.copy()
        df['local_time'] = pd.to_datetime(df['local_time'])

        if city != 'all':
            df = df[df['city'] == city]

        date_mask = (df['local_time'] >= start_date) & (df['local_time'] <= end_date)
        df = df[date_mask]

        # Check if data exists
        if len(df) == 0:
            return jsonify({'error': 'No data available for selected city and date range'}), 404

        logger.info(f"Found {len(df)} data points for forecast")

        # Check if the requested model exists or needs to be trained
        if not hasattr(forecaster, 'models'):
            forecaster.models = {}

        if model_name not in forecaster.models:
            # Try to load the model
            model_path = os.path.join(MODEL_DIR, f"{model_name}.pkl")

            if os.path.exists(model_path):
                try:
                    logger.info(f"Loading model from {model_path}")
                    forecaster.models[model_name] = joblib.load(model_path)
                except Exception as e:
                    logger.error(f"Error loading model: {str(e)}", exc_info=True)
                    return jsonify({'error': f'Error loading model: {str(e)}'}), 500
            else:
                # Train a simple model
                logger.info(f"Model {model_name} not found. Training a new one.")
                try:
                    X, y = forecaster.prepare_features(city=city if city != 'all' else None)
                    X_train, X_test, y_train, y_test = forecaster.train_test_split_by_time(X, y)
                    forecaster.scale_features()

                    if model_name == 'linear':
                        forecaster.train_linear_regression()
                    elif model_name == 'random_forest':
                        rf_params = {'n_estimators': [50], 'max_depth': [10]}
                        forecaster.train_random_forest(param_grid=rf_params)
                    elif model_name == 'gradient_boosting':
                        gb_params = {'n_estimators': [50], 'learning_rate': [0.1]}
                        forecaster.train_gradient_boosting(param_grid=gb_params)
                    elif model_name == 'xgboost':
                        xgb_params = {'n_estimators': [50], 'learning_rate': [0.1]}
                        forecaster.train_xgboost(param_grid=xgb_params)
                    elif model_name == 'ensemble_avg':
                        # Train some base models first
                        forecaster.train_linear_regression()
                        rf_params = {'n_estimators': [50], 'max_depth': [10]}
                        forecaster.train_random_forest(param_grid=rf_params)
                        forecaster.train_ensemble(ensemble_type='averaging')
                    else:
                        # Default to linear regression
                        forecaster.train_linear_regression()
                        model_name = 'linear'
                except Exception as e:
                    logger.error(f"Error training model: {str(e)}", exc_info=True)
                    return jsonify({'error': f'Error training model: {str(e)}'}), 500

        # Generate features for the test data
        try:
            test_features = df.copy()

            # Add temporal features if they don't exist
            if 'hour' not in test_features.columns:
                test_features['hour'] = test_features['local_time'].dt.hour

            if 'day_of_week' not in test_features.columns:
                test_features['day_of_week'] = test_features['local_time'].dt.dayofweek

            if 'month' not in test_features.columns:
                test_features['month'] = test_features['local_time'].dt.month

            if 'is_weekend' not in test_features.columns:
                test_features['is_weekend'] = test_features['day_of_week'].isin([5, 6]).astype(int)

            if 'is_peak_hour' not in test_features.columns:
                test_features['is_peak_hour'] = test_features['hour'].between(17, 21).astype(int)

            # Add lag and rolling features
            lag_periods = [1, 24, 48, 168]  # Previous hour, day, 2 days, week
            for lag in lag_periods:
                test_features[f'demand_lag_{lag}'] = test_features['demand'].shift(lag)

            # Create rolling average features
            windows = [6, 12, 24, 48]
            for window in windows:
                test_features[f'demand_rolling_{window}'] = test_features['demand'].rolling(window=window).mean()

            # Drop rows with NaN values
            test_features = test_features.dropna()
        except Exception as e:
            logger.error(f"Error generating features: {str(e)}", exc_info=True)
            return jsonify({'error': f'Error generating features: {str(e)}'}), 500

        if len(test_features) == 0:
            return jsonify({'error': 'Insufficient data to generate forecast'}), 400

        # Get the feature column names from the training data
        try:
            if hasattr(forecaster, 'X_train') and forecaster.X_train is not None:
                feature_cols = forecaster.X_train.columns.tolist()
            else:
                # Default feature set
                feature_cols = []
                weather_cols = ['temperature', 'humidity', 'windSpeed', 'pressure',
                                'precipIntensity', 'cloudCover']
                for col in weather_cols:
                    if col in test_features.columns:
                        feature_cols.append(col)

                temporal_cols = ['hour', 'day_of_week', 'month', 'is_weekend', 'is_peak_hour']
                for col in temporal_cols:
                    if col in test_features.columns:
                        feature_cols.append(col)

                # Add lag features
                for lag in lag_periods:
                    feature_cols.append(f'demand_lag_{lag}')

                # Add rolling features
                for window in windows:
                    feature_cols.append(f'demand_rolling_{window}')

            logger.info(f"Using features: {feature_cols}")

            # Ensure all required feature columns exist
            missing_cols = [col for col in feature_cols if col not in test_features.columns]
            if missing_cols:
                return jsonify({'error': f'Missing feature columns: {missing_cols}'}), 400

            # Keep only the needed feature columns
            X_test = test_features[feature_cols]
        except Exception as e:
            logger.error(f"Error selecting features: {str(e)}", exc_info=True)
            return jsonify({'error': f'Error selecting features: {str(e)}'}), 500

        # Scale features if scaler is available
        try:
            if hasattr(forecaster, 'scaler') and forecaster.scaler is not None:
                num_cols = X_test.select_dtypes(include=['number']).columns
                X_test[num_cols] = forecaster.scaler.transform(X_test[num_cols])
        except Exception as e:
            logger.error(f"Error scaling features: {str(e)}", exc_info=True)
            return jsonify({'error': f'Error scaling features: {str(e)}'}), 500

        # Make predictions
        try:
            if model_name == 'lstm':
                # For LSTM, reshape data first
                X_test_lstm = X_test.values.reshape((X_test.shape[0], 1, X_test.shape[1]))
                forecast_demand = forecaster.models[model_name].predict(X_test_lstm).flatten()
            else:
                forecast_demand = forecaster.models[model_name].predict(X_test)

            actual_demand = test_features['demand'].values

            # Calculate metrics
            mae = float(np.mean(np.abs(actual_demand - forecast_demand)))
            rmse = float(np.sqrt(np.mean((actual_demand - forecast_demand) ** 2)))
            mape = float(np.mean(np.abs((actual_demand - forecast_demand) / (
                        actual_demand + 1e-10))) * 100)  # Add small epsilon to avoid division by zero

            logger.info(f"Forecast completed with MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.2f}%")
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}", exc_info=True)
            return jsonify({'error': f'Error making predictions: {str(e)}'}), 500

        # Format response
        response = {
            'city': city,
            'model': model_name,
            'start_date': start_date.strftime('%Y-%m-%d'),
            'end_date': end_date.strftime('%Y-%m-%d'),
            'timestamps': test_features['local_time'].dt.strftime('%Y-%m-%d %H:%M:%S').tolist(),
            'actual_demand': actual_demand.tolist(),
            'forecast_demand': forecast_demand.tolist(),
            'metrics': {
                'mae': mae,
                'rmse': rmse,
                'mape': mape
            }
        }

        return jsonify(response)
    except Exception as e:
        logger.error(f"Error in get_forecast: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500


# Add missing load_models method to ElectricityDemandForecaster
def add_load_models_method():
    """Add load_models method if it doesn't exist"""
    if not hasattr(ElectricityDemandForecaster, 'load_models'):
        def load_models(self, directory='models'):
            """Load saved models from directory"""
            self.models = {}
            for filename in os.listdir(directory):
                if filename.endswith('.pkl') and not filename.startswith('scaler'):
                    model_name = filename.split('.')[0]
                    self.models[model_name] = joblib.load(os.path.join(directory, filename))

            # Load scaler if available
            scaler_path = os.path.join(directory, 'scaler.pkl')
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)

            return self.models

        ElectricityDemandForecaster.load_models = load_models

    # Add save_models method if it doesn't exist
    if not hasattr(ElectricityDemandForecaster, 'save_models'):
        def save_models(self, directory='models'):
            """Save trained models to directory"""
            os.makedirs(directory, exist_ok=True)

            for model_name, model in self.models.items():
                model_path = os.path.join(directory, f"{model_name}.pkl")
                joblib.dump(model, model_path)

            # Save scaler if available
            if hasattr(self, 'scaler') and self.scaler is not None:
                scaler_path = os.path.join(directory, 'scaler.pkl')
                joblib.dump(self.scaler, scaler_path)

        ElectricityDemandForecaster.save_models = save_models


@app.route('/api/clusters', methods=['POST'])
def get_clusters():
    """Get demand clusters for a city"""
    try:
        global forecaster, data

        data = request.json

        # Validate required parameters
        required_params = ['city', 'n_clusters']
        if not all(param in data for param in required_params):
            return jsonify({'error': 'Missing required parameters'}), 400

        city = data['city'].lower()
        n_clusters = int(data['n_clusters'])

        # Load data if needed
        if forecaster is None:
            forecaster = ElectricityDemandForecaster(data_path=DATA_PATH)

        if forecaster.data is None:
            forecaster.load_data()

        # Get data for the specified city
        df = forecaster.data.copy()
        if city != 'all':
            df = df[df['city'] == city]

        # Convert dates
        df['local_time'] = pd.to_datetime(df['local_time'])

        # Check if we have data
        if len(df) == 0:
            return jsonify({'error': f"No data found for city: {city}"}), 404

        # Prepare data for clustering
        try:
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import StandardScaler
            from sklearn.decomposition import PCA

            # Extract hour and weekday
            df['hour'] = df['local_time'].dt.hour
            df['weekday'] = df['local_time'].dt.dayofweek

            # Create features for clustering
            features = ['hour', 'weekday', 'demand']
            if 'temperature' in df.columns:
                features.append('temperature')

            # Select features and drop missing values
            cluster_data = df[features].dropna()

            # Scale data
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(cluster_data)

            # Apply clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(scaled_data)

            # Apply PCA for visualization
            pca = PCA(n_components=2)
            pca_components = pca.fit_transform(scaled_data)

            # Format response
            response = {
                'city': city,
                'n_clusters': n_clusters,
                'timestamps': df['local_time'].iloc[:len(cluster_labels)].dt.strftime('%Y-%m-%d %H:%M:%S').tolist(),
                'cluster_labels': cluster_labels.tolist(),
                'pca_components': pca_components.tolist(),
                'demand_values': df['demand'].iloc[:len(cluster_labels)].tolist()
            }

            return jsonify(response)
        except Exception as e:
            logger.error(f"Error in clustering: {str(e)}", exc_info=True)

            # Fallback to random assignment if clustering fails
            n_samples = len(df)
            cluster_labels = np.random.randint(0, n_clusters, n_samples)
            pca_components = np.random.randn(n_samples, 2)

            response = {
                'city': city,
                'n_clusters': n_clusters,
                'timestamps': df['local_time'].dt.strftime('%Y-%m-%d %H:%M:%S').tolist(),
                'cluster_labels': cluster_labels.tolist(),
                'pca_components': pca_components.tolist(),
                'demand_values': df['demand'].tolist(),
                'note': 'Used fallback random clustering due to error'
            }

            return jsonify(response)

    except Exception as e:
        logger.error(f"Error in get_clusters: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/train', methods=['POST'])
def train_model():
    """Train a new model for a city"""
    try:
        global forecaster

        data = request.json

        # Validate required parameters
        required_params = ['city', 'model_type']
        if not all(param in data for param in required_params):
            return jsonify({'error': 'Missing required parameters'}), 400

        city = data['city'].lower()
        model_type = data['model_type']

        # Optional parameters
        test_size = data.get('test_size', 0.2)
        save_model = data.get('save_model', True)

        # Ensure forecaster is initialized
        if forecaster is None:
            forecaster = ElectricityDemandForecaster(data_path=DATA_PATH)
            forecaster.load_data()

        # Validate model type
        valid_models = ['linear', 'random_forest', 'gradient_boosting', 'xgboost',
                        'lstm', 'sarima', 'ensemble_avg', 'ensemble_stack', 'all']

        if model_type not in valid_models:
            return jsonify({
                'error': f"Invalid model type. Valid options are: {', '.join(valid_models)}"
            }), 400

        # Start the training in a separate thread (in a real system)
        # For this example, we'll train synchronously

        # Prepare features
        X, y = forecaster.prepare_features(city=city if city != 'all' else None)

        # Split data
        X_train, X_test, y_train, y_test = forecaster.train_test_split_by_time(X, y, test_size=test_size)

        # Scale features
        X_train, X_test = forecaster.scale_features()

        # Train requested model
        model_info = {}

        if model_type == 'all' or model_type == 'linear':
            model, preds = forecaster.train_linear_regression()
            model_info['linear'] = forecaster.metrics['linear']

        if model_type == 'all' or model_type == 'random_forest':
            # Use smaller grid for demo
            rf_params = {'n_estimators': [50], 'max_depth': [10, None]}
            model, preds = forecaster.train_random_forest(param_grid=rf_params)
            model_info['random_forest'] = forecaster.metrics['random_forest']

        if model_type == 'all' or model_type == 'gradient_boosting':
            gb_params = {'n_estimators': [50], 'learning_rate': [0.1]}
            model, preds = forecaster.train_gradient_boosting(param_grid=gb_params)
            model_info['gradient_boosting'] = forecaster.metrics['gradient_boosting']

        if model_type == 'all' or model_type == 'xgboost':
            try:
                xgb_params = {'n_estimators': [50], 'learning_rate': [0.1]}
                model, preds = forecaster.train_xgboost(param_grid=xgb_params)
                if model:
                    model_info['xgboost'] = forecaster.metrics['xgboost']
            except Exception as e:
                logger.warning(f"XGBoost training failed: {str(e)}")

        if model_type == 'all' or model_type == 'lstm':
            try:
                model, preds, history = forecaster.train_lstm(epochs=10, batch_size=32)
                if model:
                    model_info['lstm'] = forecaster.metrics['lstm']
            except Exception as e:
                logger.warning(f"LSTM training failed: {str(e)}")

        if model_type == 'all' or model_type == 'sarima':
            if city != 'all':  # SARIMA works on single time series
                try:
                    model, preds = forecaster.train_sarima(city)
                    if model:
                        model_info[f'sarima_{city}'] = forecaster.metrics[f'sarima_{city}']
                except Exception as e:
                    logger.warning(f"SARIMA training failed: {str(e)}")

        if model_type == 'all' or model_type == 'ensemble_avg':
            # Need at least 2 models for ensemble
            if len(forecaster.models) >= 2:
                model_name, preds = forecaster.train_ensemble(ensemble_type='averaging')
                if model_name:
                    model_info['ensemble_avg'] = forecaster.metrics['ensemble_avg']

        if model_type == 'all' or model_type == 'ensemble_stack':
            # Need at least 2 models for ensemble
            if len(forecaster.models) >= 2:
                model, preds = forecaster.train_ensemble(ensemble_type='stacking')
                if model:
                    model_info['ensemble_stack'] = forecaster.metrics['ensemble_stack']

        # Save models if requested
        if save_model:
            forecaster.save_models(directory=MODEL_DIR)

        # Create response
        response = {
            'status': 'success',
            'message': f"Model training completed for {city}",
            'city': city,
            'model_type': model_type,
            'model_metrics': model_info
        }

        return jsonify(response)

    except Exception as e:
        logger.error(f"Error in train_model: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    """Simple health check endpoint"""
    try:
        global forecaster

        # Initialize forecaster if needed
        if forecaster is None:
            forecaster = ElectricityDemandForecaster(data_path=DATA_PATH)

        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'app': 'Electricity Demand Forecaster API',
            'data_loaded': forecaster is not None and forecaster.data is not None,
            'models_loaded': forecaster is not None and hasattr(forecaster, 'models') and
                             forecaster.models is not None and len(forecaster.models) > 0
        })
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return jsonify({'status': 'unhealthy', 'error': str(e)}), 500


# Add the missing methods before starting the app
add_load_models_method()

if __name__ == '__main__':

    # Run the Flask app
    app.run(debug=True, port=5000)