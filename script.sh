cd ~/fastapi-cicd
git pull origin main
screen -dmS api_session sh -c 'python3 api.py'
cd ~/mlflow
screen -dmS mlflow_session sh -c 'mlflow ui -h 0.0.0.0 -p 5000'