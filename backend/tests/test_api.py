from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_and_demo():
    assert client.get('/api/health').json()['status'] == 'ok'
    result = client.get('/api/demo/demo-n2')
    assert result.status_code == 200
    body = result.json()
    assert body['prediction']['label'] in {'Wake', 'N1', 'N2', 'N3', 'REM', 'Unusable'}
    assert set(body['bandpowers']) == {'delta', 'theta', 'alpha', 'beta'}


def test_csv_upload():
    csv = b'12\n14\n13\n15\n'
    response = client.post('/api/analyze', files={'file': ('epoch.csv', csv, 'text/csv')})
    assert response.status_code == 200


def test_model_card_contract():
    response = client.get('/api/model-card')
    assert response.status_code == 200
    assert response.json()['status'] in {'trained', 'not_trained'}
