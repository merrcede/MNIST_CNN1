.PHONY: install train serve dashboard monitor lint test_model test_preprocess clean

# Install all required dependencies
install:
	pip install -r requirements.txt

# Train the model
train: install
	python model/train_model.py

# Run the Flask API
serve: train
	python api/api.py

# Run the Streamlit dashboard
dashboard: serve
	streamlit run streamlit_app/app.py

# Monitor the model with Evidently
monitor: serve
	python monitoring/monitor.py

# Run Linter to check the quality of the code
lint:
	pylint api/ model/ streamlit_app/ monitoring/

# Run data preprocessing tests
test_preprocess:
	python -m unittest tests/test_preprocess.py

# Run model-related tests
test_model:
	python -m unittest tests/test_model.py

# Clean temporary files
clean:
	rm -rf __pycache__
	rm -rf *.pyc
