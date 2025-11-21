<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Energy Demand Forecasting System</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
            line-height: 1.6;
            color: #24292e;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 40px 20px;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 12px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
            overflow: hidden;
        }

        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 60px 40px;
            text-align: center;
        }

        .header h1 {
            font-size: 3em;
            margin-bottom: 15px;
            font-weight: 700;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.2);
        }

        .header p {
            font-size: 1.3em;
            opacity: 0.95;
            margin-bottom: 30px;
        }

        .badges {
            display: flex;
            gap: 10px;
            justify-content: center;
            flex-wrap: wrap;
        }

        .badge {
            background: rgba(255, 255, 255, 0.2);
            backdrop-filter: blur(10px);
            padding: 8px 16px;
            border-radius: 20px;
            font-size: 0.9em;
            border: 1px solid rgba(255, 255, 255, 0.3);
        }

        .content {
            padding: 40px;
        }

        .section {
            margin-bottom: 50px;
        }

        .section h2 {
            font-size: 2em;
            color: #667eea;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 3px solid #667eea;
        }

        .section h3 {
            font-size: 1.5em;
            color: #764ba2;
            margin: 25px 0 15px 0;
        }

        .section p {
            margin-bottom: 15px;
            color: #586069;
            line-height: 1.8;
        }

        .features-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 25px;
            margin-top: 30px;
        }

        .feature-card {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            padding: 25px;
            border-radius: 10px;
            border-left: 4px solid #667eea;
            transition: transform 0.3s, box-shadow 0.3s;
        }

        .feature-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
        }

        .feature-card h4 {
            color: #667eea;
            font-size: 1.2em;
            margin-bottom: 10px;
        }

        .feature-card p {
            color: #586069;
            font-size: 0.95em;
            margin: 0;
        }

        .folder-structure {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 25px;
            margin-top: 30px;
        }

        .folder-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
            transition: transform 0.3s;
        }

        .folder-card:hover {
            transform: scale(1.05);
        }

        .folder-card h3 {
            color: white;
            font-size: 1.5em;
            margin-bottom: 15px;
            border-bottom: 2px solid rgba(255, 255, 255, 0.3);
            padding-bottom: 10px;
        }

        .folder-card ul {
            list-style: none;
            padding-left: 0;
        }

        .folder-card li {
            padding: 8px 0;
            padding-left: 25px;
            position: relative;
            color: rgba(255, 255, 255, 0.9);
        }

        .folder-card li:before {
            content: "📁";
            position: absolute;
            left: 0;
        }

        .code-block {
            background: #f6f8fa;
            border: 1px solid #e1e4e8;
            border-radius: 6px;
            padding: 20px;
            margin: 20px 0;
            overflow-x: auto;
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
        }

        .code-block pre {
            margin: 0;
            color: #24292e;
        }

        .tech-stack {
            display: flex;
            flex-wrap: wrap;
            gap: 15px;
            margin-top: 20px;
        }

        .tech-badge {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 10px 20px;
            border-radius: 25px;
            font-weight: 600;
            box-shadow: 0 4px 10px rgba(102, 126, 234, 0.3);
        }

        .installation-steps {
            counter-reset: step-counter;
        }

        .step {
            background: #f6f8fa;
            padding: 25px;
            margin: 20px 0;
            border-radius: 8px;
            border-left: 4px solid #667eea;
            position: relative;
            padding-left: 70px;
        }

        .step:before {
            counter-increment: step-counter;
            content: counter(step-counter);
            position: absolute;
            left: 20px;
            top: 50%;
            transform: translateY(-50%);
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            width: 40px;
            height: 40px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            font-size: 1.2em;
        }

        .api-table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background: white;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
            border-radius: 8px;
            overflow: hidden;
        }

        .api-table th {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px;
            text-align: left;
            font-weight: 600;
        }

        .api-table td {
            padding: 15px;
            border-bottom: 1px solid #e1e4e8;
        }

        .api-table tr:last-child td {
            border-bottom: none;
        }

        .api-table tr:hover {
            background: #f6f8fa;
        }

        .footer {
            background: #24292e;
            color: white;
            padding: 30px 40px;
            text-align: center;
        }

        .footer a {
            color: #667eea;
            text-decoration: none;
            font-weight: 600;
        }

        .footer a:hover {
            text-decoration: underline;
        }

        .highlight {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            font-weight: 700;
        }

        @media (max-width: 768px) {
            .header h1 {
                font-size: 2em;
            }

            .content {
                padding: 20px;
            }

            .folder-structure {
                grid-template-columns: 1fr;
            }

            .step {
                padding-left: 70px;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>⚡ Energy Demand Forecasting System</h1>
            <p>AI-Powered Electricity Demand Prediction & Analysis Platform</p>
            <div class="badges">
                <span class="badge">Next.js 14</span>
                <span class="badge">Python 3.10+</span>
                <span class="badge">Flask</span>
                <span class="badge">TensorFlow</span>
                <span class="badge">XGBoost</span>
                <span class="badge">Machine Learning</span>
            </div>
        </div>

        <div class="content">
            <section class="section">
                <h2>📖 Overview</h2>
                <p>
                    A comprehensive full-stack application for forecasting electricity demand using advanced machine learning techniques. 
                    This system combines data preprocessing, clustering analysis, and multiple forecasting models to provide accurate 
                    predictions for energy consumption patterns across multiple cities.
                </p>
                <p>
                    The platform integrates weather data, temporal patterns, and historical demand to deliver actionable insights 
                    for energy management and grid optimization.
                </p>
            </section>

            <section class="section">
                <h2>✨ Key Features</h2>
                <div class="features-grid">
                    <div class="feature-card">
                        <h4>🤖 Multiple ML Models</h4>
                        <p>Linear Regression, Random Forest, Gradient Boosting, XGBoost, LSTM, and SARIMA models for comprehensive forecasting</p>
                    </div>
                    <div class="feature-card">
                        <h4>🔍 Advanced Clustering</h4>
                        <p>K-Means, DBSCAN, and Hierarchical clustering for demand pattern analysis</p>
                    </div>
                    <div class="feature-card">
                        <h4>📊 Data Visualization</h4>
                        <p>Interactive charts, PCA/t-SNE visualizations, and comprehensive dashboards</p>
                    </div>
                    <div class="feature-card">
                        <h4>🌦️ Weather Integration</h4>
                        <p>Real-time weather data processing for accurate demand correlation</p>
                    </div>
                    <div class="feature-card">
                        <h4>🎯 Ensemble Methods</h4>
                        <p>Averaging and stacking ensemble techniques for improved accuracy</p>
                    </div>
                    <div class="feature-card">
                        <h4>⚡ Real-time Predictions</h4>
                        <p>Fast API responses with optimized model serving</p>
                    </div>
                </div>
            </section>

            <section class="section">
                <h2>🏗️ Project Structure</h2>
                <div class="folder-structure">
                    <div class="folder-card">
                        <h3>📱 Nextjs Frontend</h3>
                        <ul>
                            <li>Modern React-based UI</li>
                            <li>Interactive data visualizations</li>
                            <li>Real-time forecast displays</li>
                            <li>Responsive design</li>
                            <li>Chart.js integrations</li>
                        </ul>
                    </div>
                    <div class="folder-card">
                        <h3>🔧 Backend</h3>
                        <ul>
                            <li>Data preprocessing pipeline</li>
                            <li>ML model training scripts</li>
                            <li>Clustering algorithms</li>
                            <li>Feature engineering</li>
                            <li>Model evaluation tools</li>
                        </ul>
                    </div>
                    <div class="folder-card">
                        <h3>🔌 Flaskforconnection</h3>
                        <ul>
                            <li>RESTful API endpoints</li>
                            <li>Model serving layer</li>
                            <li>Request handling</li>
                            <li>CORS configuration</li>
                            <li>Error handling</li>
                        </ul>
                    </div>
                </div>
            </section>

            <section class="section">
                <h2>🛠️ Technology Stack</h2>
                <div class="tech-stack">
                    <span class="tech-badge">Next.js 14</span>
                    <span class="tech-badge">React</span>
                    <span class="tech-badge">Python 3.10+</span>
                    <span class="tech-badge">Flask</span>
                    <span class="tech-badge">TensorFlow</span>
                    <span class="tech-badge">XGBoost</span>
                    <span class="tech-badge">scikit-learn</span>
                    <span class="tech-badge">Pandas</span>
                    <span class="tech-badge">NumPy</span>
                    <span class="tech-badge">Matplotlib</span>
                    <span class="tech-badge">Seaborn</span>
                </div>
            </section>

            <section class="section">
                <h2>🚀 Installation & Setup</h2>
                <div class="installation-steps">
                    <div class="step">
                        <h3>Clone the Repository</h3>
                        <div class="code-block">
<pre>git clone https://github.com/yourusername/energy-demand-forecasting.git
cd energy-demand-forecasting</pre>
                        </div>
                    </div>

                    <div class="step">
                        <h3>Setup Backend</h3>
                        <div class="code-block">
<pre>cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Run preprocessing
python preprocessing_and_merging.py

# Run clustering analysis
python clustering.py

# Train forecasting models
python electricity_forecaster.py</pre>
                        </div>
                    </div>

                    <div class="step">
                        <h3>Setup Flask API</h3>
                        <div class="code-block">
<pre>cd ../Flaskforconnection
pip install flask flask-cors joblib pandas numpy scikit-learn

# Start Flask server
python app.py
# Server runs on http://localhost:5000</pre>
                        </div>
                    </div>

                    <div class="step">
                        <h3>Setup Frontend</h3>
                        <div class="code-block">
<pre>cd ../Nextjs\ Frontend
npm install
# or
yarn install

# Start development server
npm run dev
# or
yarn dev

# Open http://localhost:3000</pre>
                        </div>
                    </div>
                </div>
            </section>

            <section class="section">
                <h2>📡 API Endpoints</h2>
                <table class="api-table">
                    <thead>
                        <tr>
                            <th>Endpoint</th>
                            <th>Method</th>
                            <th>Description</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td><code>/api/forecast</code></td>
                            <td>POST</td>
                            <td>Get demand forecast for specified parameters</td>
                        </tr>
                        <tr>
                            <td><code>/api/clusters</code></td>
                            <td>GET</td>
                            <td>Retrieve clustering analysis results</td>
                        </tr>
                        <tr>
                            <td><code>/api/models</code></td>
                            <td>GET</td>
                            <td>List available forecasting models</td>
                        </tr>
                        <tr>
                            <td><code>/api/metrics</code></td>
                            <td>GET</td>
                            <td>Get model performance metrics</td>
                        </tr>
                        <tr>
                            <td><code>/api/cities</code></td>
                            <td>GET</td>
                            <td>List available cities for forecasting</td>
                        </tr>
                    </tbody>
                </table>
            </section>

            <section class="section">
                <h2>📊 Data Pipeline</h2>
                <h3>1. Data Preprocessing</h3>
                <p>
                    <span class="highlight">preprocessing_and_merging.py</span> handles data loading, cleaning, merging multiple sources, 
                    feature engineering, normalization, and anomaly detection.
                </p>

                <h3>2. Clustering Analysis</h3>
                <p>
                    <span class="highlight">clustering.py</span> performs dimensionality reduction (PCA, t-SNE), applies K-Means, DBSCAN, 
                    and hierarchical clustering, and generates comprehensive visualizations.
                </p>

                <h3>3. Forecasting Models</h3>
                <p>
                    <span class="highlight">electricity_forecaster.py</span> trains multiple models including Linear Regression, Random Forest, 
                    Gradient Boosting, XGBoost, LSTM, SARIMA, and ensemble methods.
                </p>
            </section>

            <section class="section">
                <h2>📈 Model Performance</h2>
                <p>The system evaluates models using multiple metrics:</p>
                <ul style="margin-left: 30px; color: #586069;">
                    <li><strong>MAE (Mean Absolute Error):</strong> Average prediction error</li>
                    <li><strong>RMSE (Root Mean Square Error):</strong> Penalizes larger errors</li>
                    <li><strong>MAPE (Mean Absolute Percentage Error):</strong> Percentage-based accuracy</li>
                    <li><strong>Silhouette Score:</strong> Cluster quality measurement</li>
                </ul>
            </section>

            <section class="section">
                <h2>🤝 Contributing</h2>
                <p>Contributions are welcome! Please follow these steps:</p>
                <div class="code-block">
<pre>1. Fork the repository
2. Create a feature branch (git checkout -b feature/AmazingFeature)
3. Commit your changes (git commit -m 'Add AmazingFeature')
4. Push to the branch (git push origin feature/AmazingFeature)
5. Open a Pull Request</pre>
                </div>
            </section>

            <section class="section">
                <h2>📄 License</h2>
                <p>This project is licensed under the MIT License - see the LICENSE file for details.</p>
            </section>

            <section class="section">
                <h2>👥 Authors</h2>
                <p>Your Name - <a href="https://github.com/yourusername" style="color: #667eea;">@yourusername</a></p>
            </section>

            <section class="section">
                <h2>🙏 Acknowledgments</h2>
                <ul style="margin-left: 30px; color: #586069;">
                    <li>Weather data providers</li>
                    <li>Open-source ML libraries</li>
                    <li>Energy grid datasets</li>
                    <li>Research papers on time series forecasting</li>
                </ul>
            </section>
        </div>

        <div class="footer">
            <p>⭐ If you find this project helpful, please give it a star on <a href="https://github.com/yourusername/energy-demand-forecasting">GitHub</a></p>
            <p style="margin-top: 10px; opacity: 0.8;">Made with ❤️ for sustainable energy management</p>
        </div>
    </div>
</body>
</html>
