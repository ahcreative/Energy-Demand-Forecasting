"use client";
import React, { useState, useEffect } from "react";
import {
  Container,
  Row,
  Col,
  Form,
  Button,
  Spinner,
  Card,
  Tab,
  Tabs,
  Alert,
} from "react-bootstrap";
import DatePicker from "react-datepicker";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";
import { ScatterChart, Scatter, ZAxis } from "recharts";

// API URL - Replace with your backend URL
const API_URL = "http://127.0.0.1:5000/api/";

function App() {
  // State for form inputs
  const [cities, setCities] = useState([]);
  const [models, setModels] = useState([]);
  const [selectedCity, setSelectedCity] = useState("");
  const [selectedModel, setSelectedModel] = useState("");
  const [startDate, setStartDate] = useState(
    new Date(Date.now() - 7 * 24 * 60 * 60 * 1000)
  ); // 1 week ago
  const [endDate, setEndDate] = useState(new Date());
  const [numClusters, setNumClusters] = useState(3);

  // State for API data
  const [forecastData, setForecastData] = useState(null);
  const [clusterData, setClusterData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Fetch cities on component mount
  useEffect(() => {
    const fetchCities = async () => {
      try {
        const response = await fetch(`${API_URL}/cities`);
        const data = await response.json();
        setCities(data.cities);
        if (data.cities.length > 0) {
          setSelectedCity(data.cities[0]);
        }
      } catch (err) {
        console.error("Error fetching cities:", err);
        setError("Failed to fetch cities. Please try again later.");
      }
    };

    const fetchModels = async () => {
      try {
        const response = await fetch(`${API_URL}/models`);
        const data = await response.json();
        setModels(data.models);
        if (data.models.length > 0) {
          setSelectedModel(data.models[0]);
        }
      } catch (err) {
        console.error("Error fetching models:", err);
        setError("Failed to fetch models. Please try again later.");
      }
    };

    fetchCities();
    fetchModels();
  }, []);

  // Function to fetch forecast data
  const fetchForecast = async () => {
    if (!selectedCity || !selectedModel) {
      setError("Please select a city and model");
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const response = await fetch(`${API_URL}/forecast`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          city: selectedCity,
          model: selectedModel,
          start_date: startDate.toISOString().split("T")[0],
          end_date: endDate.toISOString().split("T")[0],
        }),
      });

      const data = await response.json();

      if (response.ok) {
        // Process data for charts
        const chartData = data.timestamps.map((timestamp, index) => ({
          timestamp,
          actual: data.actual_demand[index],
          forecast: data.forecast_demand[index],
        }));

        setForecastData({
          chartData,
          metrics: data.metrics,
          city: data.city,
          model: data.model,
        });
      } else {
        setError(data.error || "Failed to fetch forecast data");
      }
    } catch (err) {
      console.error("Error fetching forecast:", err);
      setError("Failed to fetch forecast data. Please try again later.");
    } finally {
      setLoading(false);
    }
  };

  // Function to fetch clustering data
  const fetchClusters = async () => {
    if (!selectedCity) {
      setError("Please select a city");
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const response = await fetch(`${API_URL}/clusters`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          city: selectedCity,
          n_clusters: numClusters,
        }),
      });

      const data = await response.json();

      if (response.ok) {
        // Process data for charts
        const clusterChartData = data.timestamps.map((timestamp, index) => ({
          timestamp,
          cluster: data.cluster_labels[index],
          x: data.pca_components[index][0],
          y: data.pca_components[index][1],
          demand: data.demand_values[index],
        }));

        setClusterData({
          chartData: clusterChartData,
          numClusters: data.n_clusters,
          city: data.city,
        });
      } else {
        setError(data.error || "Failed to fetch cluster data");
      }
    } catch (err) {
      console.error("Error fetching clusters:", err);
      setError("Failed to fetch cluster data. Please try again later.");
    } finally {
      setLoading(false);
    }
  };

  // Handle form submit
  const handleSubmit = (e) => {
    e.preventDefault();
    fetchForecast();
    fetchClusters();
  };

  // Custom colors for clusters
  const clusterColors = [
    "#8884d8",
    "#82ca9d",
    "#ffc658",
    "#ff8042",
    "#0088fe",
    "#00C49F",
    "#FFBB28",
    "#FF8042",
    "#a4de6c",
    "#d0ed57",
  ];

  return (
    <Container fluid className="app-container">
      <Row className="header mb-4">
        <Col>
          <h1>Electricity Demand Analysis</h1>
          <p>Analyze and forecast electricity demand patterns across cities</p>
        </Col>
      </Row>

      {error && (
        <Row className="mb-3">
          <Col>
            <Alert variant="danger">{error}</Alert>
          </Col>
        </Row>
      )}

      <Row>
        <Col md={3}>
          <Card className="mb-4">
            <Card.Header>Analysis Parameters</Card.Header>
            <Card.Body>
              <Form onSubmit={handleSubmit}>
                <Form.Group className="mb-3">
                  <Form.Label>City</Form.Label>
                  <Form.Control
                    as="select"
                    value={selectedCity}
                    onChange={(e) => setSelectedCity(e.target.value)}
                    required
                  >
                    <option value="">Select City</option>
                    {cities.map((city) => (
                      <option key={city} value={city}>
                        {city.charAt(0).toUpperCase() + city.slice(1)}
                      </option>
                    ))}
                  </Form.Control>
                </Form.Group>

                <Form.Group className="mb-3">
                  <Form.Label>Start Date</Form.Label>
                  <DatePicker
                    selected={startDate}
                    onChange={(date) => setStartDate(date)}
                    className="form-control"
                    dateFormat="yyyy-MM-dd"
                    maxDate={endDate}
                    required
                  />
                </Form.Group>

                <Form.Group className="mb-3">
                  <Form.Label>End Date</Form.Label>
                  <DatePicker
                    selected={endDate}
                    onChange={(date) => setEndDate(date)}
                    className="form-control"
                    dateFormat="yyyy-MM-dd"
                    minDate={startDate}
                    maxDate={new Date()}
                    required
                  />
                </Form.Group>

                <Form.Group className="mb-3">
                  <Form.Label>Forecast Model</Form.Label>
                  <Form.Control
                    as="select"
                    value={selectedModel}
                    onChange={(e) => setSelectedModel(e.target.value)}
                    required
                  >
                    <option value="">Select Model</option>
                    {models.map((model) => (
                      <option key={model} value={model}>
                        {model.replace("_", " ").charAt(0).toUpperCase() +
                          model.replace("_", " ").slice(1)}
                      </option>
                    ))}
                  </Form.Control>
                </Form.Group>

                <Form.Group className="mb-3">
                  <Form.Label>Number of Clusters (K)</Form.Label>
                  <Form.Control
                    type="range"
                    min={2}
                    max={10}
                    value={numClusters}
                    onChange={(e) => setNumClusters(parseInt(e.target.value))}
                  />
                  <div className="d-flex justify-content-between">
                    <span>2</span>
                    <span>{numClusters}</span>
                    <span>10</span>
                  </div>
                </Form.Group>

                <Button
                  variant="primary"
                  type="submit"
                  className="w-100"
                  disabled={loading}
                >
                  {loading ? (
                    <>
                      <Spinner
                        as="span"
                        animation="border"
                        size="sm"
                        role="status"
                        aria-hidden="true"
                      />
                      <span className="ms-2">Loading...</span>
                    </>
                  ) : (
                    "Analyze Data"
                  )}
                </Button>
              </Form>
            </Card.Body>
          </Card>

          <Card className="mb-4">
            <Card.Header>Help & Documentation</Card.Header>
            <Card.Body>
              <h5>How to Use:</h5>
              <ol>
                <li>Select a city from the dropdown.</li>
                <li>Choose a date range for analysis.</li>
                <li>Select a forecast model.</li>
                <li>Adjust the number of clusters if needed.</li>
                <li>Click Analyze Data to generate results.</li>
              </ol>

              <h5>About the Analysis:</h5>
              <p>This tool performs two types of analysis:</p>
              <ul>
                <li>
                  <strong>Clustering:</strong> Groups similar electricity demand
                  patterns.
                </li>
                <li>
                  <strong>Forecasting:</strong> Predicts future electricity
                  demand based on historical data and weather patterns.
                </li>
              </ul>

              <h5>Metrics:</h5>
              <ul>
                <li>
                  <strong>MAE:</strong> Mean Absolute Error - Average absolute
                  difference between predictions and actual values.
                </li>
                <li>
                  <strong>RMSE:</strong> Root Mean Square Error - Square root of
                  the average squared differences.
                </li>
                <li>
                  <strong>MAPE:</strong> Mean Absolute Percentage Error -
                  Percentage error relative to actual values.
                </li>
              </ul>
            </Card.Body>
          </Card>
        </Col>

        <Col md={9}>
          <Tabs defaultActiveKey="forecast" id="analysis-tabs" className="mb-3">
            <Tab eventKey="forecast" title="Demand Forecast">
              {forecastData ? (
                <>
                  <Card className="mb-4">
                    <Card.Header>
                      <h4>Electricity Demand Forecast</h4>
                      <h6>
                        {forecastData.city.charAt(0).toUpperCase() +
                          forecastData.city.slice(1)}{" "}
                        -
                        {forecastData.model
                          .replace("_", " ")
                          .charAt(0)
                          .toUpperCase() +
                          forecastData.model.replace("_", " ").slice(1)}
                      </h6>
                    </Card.Header>
                    <Card.Body>
                      <ResponsiveContainer width="100%" height={400}>
                        <LineChart
                          data={forecastData.chartData}
                          margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                        >
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis
                            dataKey="timestamp"
                            tickFormatter={(tick) =>
                              new Date(tick).toLocaleDateString()
                            }
                            interval={Math.floor(
                              forecastData.chartData.length / 10
                            )}
                          />
                          <YAxis
                            label={{
                              value: "Demand (MW)",
                              angle: -90,
                              position: "insideLeft",
                            }}
                          />
                          <Tooltip
                            formatter={(value) => [
                              Number(value).toFixed(2),
                              "",
                            ]}
                            labelFormatter={(label) =>
                              new Date(label).toLocaleString()
                            }
                          />
                          <Legend />
                          <Line
                            type="monotone"
                            dataKey="actual"
                            stroke="#8884d8"
                            name="Actual Demand"
                            dot={false}
                          />
                          <Line
                            type="monotone"
                            dataKey="forecast"
                            stroke="#82ca9d"
                            name="Forecast Demand"
                            dot={false}
                          />
                        </LineChart>
                      </ResponsiveContainer>
                    </Card.Body>
                  </Card>

                  <Row>
                    <Col md={4}>
                      <Card className="metric-card">
                        <Card.Body>
                          <h5>MAE</h5>
                          <h3>{forecastData.metrics.mae.toFixed(2)}</h3>
                          <p>Mean Absolute Error</p>
                        </Card.Body>
                      </Card>
                    </Col>
                    <Col md={4}>
                      <Card className="metric-card">
                        <Card.Body>
                          <h5>RMSE</h5>
                          <h3>{forecastData.metrics.rmse.toFixed(2)}</h3>
                          <p>Root Mean Square Error</p>
                        </Card.Body>
                      </Card>
                    </Col>
                    <Col md={4}>
                      <Card className="metric-card">
                        <Card.Body>
                          <h5>MAPE</h5>
                          <h3>{forecastData.metrics.mape.toFixed(2)}%</h3>
                          <p>Mean Absolute Percentage Error</p>
                        </Card.Body>
                      </Card>
                    </Col>
                  </Row>
                </>
              ) : (
                <div className="text-center py-5">
                  <p>
                    Select parameters and click Analyze Data to generate a
                    forecast
                  </p>
                </div>
              )}
            </Tab>

            <Tab eventKey="clusters" title="Demand Clusters">
              {clusterData ? (
                <>
                  <Card className="mb-4">
                    <Card.Header>
                      <h4>Electricity Demand Clusters</h4>
                      <h6>
                        {clusterData.city.charAt(0).toUpperCase() +
                          clusterData.city.slice(1)}{" "}
                        -{clusterData.numClusters} Clusters
                      </h6>
                    </Card.Header>
                    <Card.Body>
                      <Row>
                        <Col md={8}>
                          <ResponsiveContainer width="100%" height={400}>
                            <ScatterChart
                              margin={{
                                top: 20,
                                right: 20,
                                bottom: 20,
                                left: 20,
                              }}
                            >
                              <CartesianGrid />
                              <XAxis
                                type="number"
                                dataKey="x"
                                name="PCA Component 1"
                                label={{
                                  value: "PCA Component 1",
                                  position: "bottom",
                                }}
                              />
                              <YAxis
                                type="number"
                                dataKey="y"
                                name="PCA Component 2"
                                label={{
                                  value: "PCA Component 2",
                                  angle: -90,
                                  position: "left",
                                }}
                              />
                              <ZAxis
                                type="number"
                                dataKey="demand"
                                range={[50, 500]}
                                name="Demand"
                              />
                              <Tooltip
                                cursor={{ strokeDasharray: "3 3" }}
                                formatter={(value, name, props) => {
                                  if (
                                    name === "PCA Component 1" ||
                                    name === "PCA Component 2"
                                  ) {
                                    return [value.toFixed(2), name];
                                  } else if (name === "Demand") {
                                    return [value.toFixed(2) + " MW", "Demand"];
                                  }
                                  return [value, name];
                                }}
                                labelFormatter={(label, payload) => {
                                  if (payload && payload.length > 0) {
                                    return `Cluster ${payload[0].payload.cluster}`;
                                  }
                                  return label;
                                }}
                              />
                              <Legend />
                              {Array.from(
                                { length: clusterData.numClusters },
                                (_, i) => (
                                  <Scatter
                                    key={i}
                                    name={`Cluster ${i}`}
                                    data={clusterData.chartData.filter(
                                      (d) => d.cluster === i
                                    )}
                                    fill={
                                      clusterColors[i % clusterColors.length]
                                    }
                                  />
                                )
                              )}
                            </ScatterChart>
                          </ResponsiveContainer>
                        </Col>
                        <Col md={4}>
                          <h5>Cluster Summary</h5>
                          <div className="cluster-stats">
                            {Array.from(
                              { length: clusterData.numClusters },
                              (_, i) => {
                                const clusterPoints =
                                  clusterData.chartData.filter(
                                    (d) => d.cluster === i
                                  );
                                const avgDemand =
                                  clusterPoints.reduce(
                                    (sum, point) => sum + point.demand,
                                    0
                                  ) / clusterPoints.length;

                                return (
                                  <div key={i} className="mb-3">
                                    <div className="d-flex justify-content-between align-items-center mb-1">
                                      <div>
                                        <span
                                          className="color-dot"
                                          style={{
                                            backgroundColor:
                                              clusterColors[
                                                i % clusterColors.length
                                              ],
                                          }}
                                        ></span>
                                        <strong>Cluster {i}</strong>
                                      </div>
                                      <span>{clusterPoints.length} points</span>
                                    </div>
                                    <div>
                                      <small>
                                        Avg. Demand: {avgDemand.toFixed(2)} MW
                                      </small>
                                    </div>
                                  </div>
                                );
                              }
                            )}
                          </div>

                          <h5 className="mt-4">Interpretation</h5>
                          <p>
                            Clusters represent similar electricity demand
                            patterns. Points close together in the visualization
                            have similar characteristics in terms of daily load
                            profiles, weather response, and temporal features.
                          </p>
                          <p>
                            Larger clusters may represent common demand patterns
                            (e.g., typical workdays), while smaller clusters may
                            indicate unusual events or outliers.
                          </p>
                        </Col>
                      </Row>
                    </Card.Body>
                  </Card>

                  <Card>
                    <Card.Header>Temporal Distribution by Cluster</Card.Header>
                    <Card.Body>
                      <ResponsiveContainer width="100%" height={300}>
                        <LineChart
                          data={clusterData.chartData}
                          margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                        >
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis
                            dataKey="timestamp"
                            tickFormatter={(tick) =>
                              new Date(tick).toLocaleDateString()
                            }
                            interval={Math.floor(
                              clusterData.chartData.length / 10
                            )}
                          />
                          <YAxis
                            label={{
                              value: "Demand (MW)",
                              angle: -90,
                              position: "insideLeft",
                            }}
                          />
                          <Tooltip
                            formatter={(value) => [
                              Number(value).toFixed(2),
                              "",
                            ]}
                            labelFormatter={(label) =>
                              new Date(label).toLocaleString()
                            }
                          />
                          <Legend />
                          <Line
                            type="monotone"
                            dataKey="demand"
                            stroke="#8884d8"
                            name="Demand"
                            dot={false}
                          />
                        </LineChart>
                      </ResponsiveContainer>
                    </Card.Body>
                  </Card>
                </>
              ) : (
                <div className="text-center py-5">
                  <p>
                    Select parameters and click Analyze Data to generate
                    clusters
                  </p>
                </div>
              )}
            </Tab>

            <Tab eventKey="comparison" title="Model Comparison">
              {forecastData ? (
                <Card>
                  <Card.Header>Model Performance Comparison</Card.Header>
                  <Card.Body>
                    <Row>
                      <Col md={6}>
                        <h5>Performance Metrics</h5>
                        <table className="table table-striped">
                          <thead>
                            <tr>
                              <th>Model</th>
                              <th>MAE</th>
                              <th>RMSE</th>
                              <th>MAPE</th>
                            </tr>
                          </thead>
                          <tbody>
                            <tr>
                              <td>
                                {forecastData.model
                                  .replace("_", " ")
                                  .charAt(0)
                                  .toUpperCase() +
                                  forecastData.model.replace("_", " ").slice(1)}
                              </td>
                              <td>{forecastData.metrics.mae.toFixed(2)}</td>
                              <td>{forecastData.metrics.rmse.toFixed(2)}</td>
                              <td>{forecastData.metrics.mape.toFixed(2)}%</td>
                            </tr>
                            <tr>
                              <td>Baseline (Previous Day)</td>
                              <td>TBD</td>
                              <td>TBD</td>
                              <td>TBD</td>
                            </tr>
                          </tbody>
                        </table>
                      </Col>
                      <Col md={6}>
                        <h5>Model Insights</h5>
                        <div className="model-info">
                          <h6>Input Features</h6>
                          <ul>
                            <li>Previous demand values</li>
                            <li>Weather data (temperature, humidity)</li>
                            <li>
                              Calendar features (hour, day of week, holidays)
                            </li>
                          </ul>

                          <h6>Model Architecture</h6>
                          <p>
                            {forecastData.model
                              .toLowerCase()
                              .includes("xgboost")
                              ? "XGBoost: Gradient boosting with tree-based models"
                              : forecastData.model
                                  .toLowerCase()
                                  .includes("lstm")
                              ? "LSTM: Long Short-Term Memory neural network"
                              : forecastData.model
                                  .toLowerCase()
                                  .includes("ensemble")
                              ? "Ensemble: Combination of multiple prediction models"
                              : "Machine learning model trained on historical demand patterns"}
                          </p>
                        </div>
                      </Col>
                    </Row>
                  </Card.Body>
                </Card>
              ) : (
                <div className="text-center py-5">
                  <p>
                    Select parameters and click Analyze Data to view model
                    comparison
                  </p>
                </div>
              )}
            </Tab>

            <Tab eventKey="data" title="Raw Data">
              {forecastData ? (
                <Card>
                  <Card.Header>Raw Data Explorer</Card.Header>
                  <Card.Body>
                    <div className="table-responsive">
                      <table className="table table-striped table-sm">
                        <thead>
                          <tr>
                            <th>Timestamp</th>
                            <th>Actual Demand (MW)</th>
                            <th>Forecast Demand (MW)</th>
                            <th>Error</th>
                            <th>Error %</th>
                          </tr>
                        </thead>
                        <tbody>
                          {forecastData.chartData
                            .slice(0, 100)
                            .map((point, index) => {
                              const error = point.actual - point.forecast;
                              const errorPct = (error / point.actual) * 100;

                              return (
                                <tr key={index}>
                                  <td>
                                    {new Date(point.timestamp).toLocaleString()}
                                  </td>
                                  <td>{point.actual.toFixed(2)}</td>
                                  <td>{point.forecast.toFixed(2)}</td>
                                  <td>{error.toFixed(2)}</td>
                                  <td>{errorPct.toFixed(2)}%</td>
                                </tr>
                              );
                            })}
                        </tbody>
                      </table>
                      {forecastData.chartData.length > 100 && (
                        <p className="text-center mt-2">
                          <em>
                            Showing first 100 rows of{" "}
                            {forecastData.chartData.length} total
                          </em>
                        </p>
                      )}
                    </div>
                  </Card.Body>
                </Card>
              ) : (
                <div className="text-center py-5">
                  <p>
                    Select parameters and click Analyze Data to view raw data
                  </p>
                </div>
              )}
            </Tab>
          </Tabs>
        </Col>
      </Row>

      <footer className="mt-5 mb-3 text-center">
        <p>
          &copy; 2025 Electricity Demand Analysis Tool | All rights reserved
        </p>
      </footer>
    </Container>
  );
}

export default App;

/* App.css - Add these styles to your CSS file */
