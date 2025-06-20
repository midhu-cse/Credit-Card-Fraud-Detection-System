import pytest
import requests
import json
import time

BASE_URL = "http://127.0.0.1:8000"

class TestFraudDetectionAPI:
    """Test suite for Fraud Detection API"""
    
    @pytest.fixture(scope="class")
    def api_client(self):
        """Setup API client and check if server is running"""
        try:
            response = requests.get(f"{BASE_URL}/health", timeout=5)
            if response.status_code != 200:
                pytest.skip("API server not running")
        except requests.exceptions.RequestException:
            pytest.skip("API server not accessible")
        return requests.Session()
    
    def test_health_endpoint(self, api_client):
        """Test health check endpoint"""
        response = api_client.get(f"{BASE_URL}/health")
        assert response.status_code == 200
        data = response.json()
        assert "api_status" in data
        assert data["api_status"] == "healthy"
    
    def test_root_endpoint(self, api_client):
        """Test root endpoint"""
        response = api_client.get(f"{BASE_URL}/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "status" in data
    
    def test_model_info_endpoint(self, api_client):
        """Test model info endpoint"""
        response = api_client.get(f"{BASE_URL}/model/info")
        assert response.status_code == 200
        data = response.json()
        assert "model_type" in data
        assert "n_estimators" in data
        assert "features_count" in data
    
    def test_valid_prediction(self, api_client):
        """Test valid fraud prediction"""
        # Normal transaction data
        transaction_data = {
            "V1": 0.1, "V2": -0.5, "V3": 0.3, "V4": -0.1, "V5": 0.2,
            "V6": 0.1, "V7": -0.2, "V8": 0.0, "V9": 0.1, "V10": -0.1,
            "V11": 0.0, "V12": 0.1, "V13": -0.1, "V14": 0.2, "V15": 0.0,
            "V16": -0.1, "V17": 0.1, "V18": 0.0, "V19": 0.1, "V20": -0.1,
            "V21": 0.0, "V22": 0.1, "V23": -0.1, "V24": 0.0, "V25": 0.1,
            "V26": -0.1, "V27": 0.0, "V28": 0.1,
            "Amount": 100.0
        }
        
        response = api_client.post(f"{BASE_URL}/predict", json=transaction_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "is_fraud" in data
        assert "fraud_probability" in data
        assert "risk_level" in data
        assert isinstance(data["is_fraud"], bool)
        assert 0 <= data["fraud_probability"] <= 1
        assert data["risk_level"] in ["Low", "Medium", "High"]
    
    def test_suspicious_transaction(self, api_client):
        """Test suspicious transaction prediction"""
        # Suspicious transaction data
        suspicious_data = {
            "V1": -2.5, "V2": 3.1, "V3": -1.8, "V4": 2.2, "V5": -3.0,
            "V6": 1.9, "V7": -2.1, "V8": 2.8, "V9": -1.5, "V10": 2.0,
            "V11": -2.3, "V12": 1.7, "V13": -2.9, "V14": 2.4, "V15": -1.6,
            "V16": 2.1, "V17": -2.7, "V18": 1.8, "V19": -2.0, "V20": 2.6,
            "V21": -1.9, "V22": 2.3, "V23": -2.8, "V24": 1.5, "V25": -2.2,
            "V26": 2.9, "V27": -1.7, "V28": 2.5,
            "Amount": 5000.0
        }
        
        response = api_client.post(f"{BASE_URL}/predict", json=suspicious_data)
        assert response.status_code == 200
        
        data = response.json()
        # Suspicious transactions should have higher fraud probability
        assert data["fraud_probability"] > 0.1  # Should be somewhat suspicious
    
    def test_invalid_data_missing_fields(self, api_client):
        """Test prediction with missing required fields"""
        incomplete_data = {
            "V1": 0.1,
            "Amount": 100.0
            # Missing V2-V28
        }
        
        response = api_client.post(f"{BASE_URL}/predict", json=incomplete_data)
        assert response.status_code == 422  # Validation error
    
    def test_invalid_data_negative_amount(self, api_client):
        """Test prediction with negative amount"""
        invalid_data = {
            "V1": 0.1, "V2": -0.5, "V3": 0.3, "V4": -0.1, "V5": 0.2,
            "V6": 0.1, "V7": -0.2, "V8": 0.0, "V9": 0.1, "V10": -0.1,
            "V11": 0.0, "V12": 0.1, "V13": -0.1, "V14": 0.2, "V15": 0.0,
            "V16": -0.1, "V17": 0.1, "V18": 0.0, "V19": 0.1, "V20": -0.1,
            "V21": 0.0, "V22": 0.1, "V23": -0.1, "V24": 0.0, "V25": 0.1,
            "V26": -0.1, "V27": 0.0, "V28": 0.1,
            "Amount": -100.0  # Invalid negative amount
        }
        
        response = api_client.post(f"{BASE_URL}/predict", json=invalid_data)
        assert response.status_code == 422  # Validation error
    
    def test_api_response_time(self, api_client):
        """Test API response time"""
        transaction_data = {
            "V1": 0.1, "V2": -0.5, "V3": 0.3, "V4": -0.1, "V5": 0.2,
            "V6": 0.1, "V7": -0.2, "V8": 0.0, "V9": 0.1, "V10": -0.1,
            "V11": 0.0, "V12": 0.1, "V13": -0.1, "V14": 0.2, "V15": 0.0,
            "V16": -0.1, "V17": 0.1, "V18": 0.0, "V19": 0.1, "V20": -0.1,
            "V21": 0.0, "V22": 0.1, "V23": -0.1, "V24": 0.0, "V25": 0.1,
            "V26": -0.1, "V27": 0.0, "V28": 0.1,
            "Amount": 100.0
        }
        
        start_time = time.time()
        response = api_client.post(f"{BASE_URL}/predict", json=transaction_data)
        end_time = time.time()
        
        assert response.status_code == 200
        assert (end_time - start_time) < 1.0  # Should respond within 1 second
    
    def test_batch_predictions(self, api_client):
        """Test multiple predictions in sequence"""
        test_cases = [
            {
                "V1": 0.1, "V2": -0.5, "V3": 0.3, "V4": -0.1, "V5": 0.2,
                "V6": 0.1, "V7": -0.2, "V8": 0.0, "V9": 0.1, "V10": -0.1,
                "V11": 0.0, "V12": 0.1, "V13": -0.1, "V14": 0.2, "V15": 0.0,
                "V16": -0.1, "V17": 0.1, "V18": 0.0, "V19": 0.1, "V20": -0.1,
                "V21": 0.0, "V22": 0.1, "V23": -0.1, "V24": 0.0, "V25": 0.1,
                "V26": -0.1, "V27": 0.0, "V28": 0.1,
                "Amount": 50.0
            },
            {
                "V1": -1.5, "V2": 2.1, "V3": -1.8, "V4": 1.2, "V5": -2.0,
                "V6": 1.9, "V7": -1.1, "V8": 1.8, "V9": -1.5, "V10": 1.0,
                "V11": -1.3, "V12": 1.7, "V13": -1.9, "V14": 1.4, "V15": -1.6,
                "V16": 1.1, "V17": -1.7, "V18": 1.8, "V19": -1.0, "V20": 1.6,
                "V21": -1.9, "V22": 1.3, "V23": -1.8, "V24": 1.5, "V25": -1.2,
                "V26": 1.9, "V27": -1.7, "V28": 1.5,
                "Amount": 1000.0
            }
        ]
        
        for i, test_case in enumerate(test_cases):
            response = api_client.post(f"{BASE_URL}/predict", json=test_case)
            assert response.status_code == 200, f"Test case {i} failed"
            
            data = response.json()
            assert "is_fraud" in data
            assert "fraud_probability" in data
            assert "risk_level" in data