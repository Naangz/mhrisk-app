install:
	pip install -r app/requirements.txt

train:
	python train.py

eval:
	python monitor.py
	echo "## Model Metrics" > report.md
	cat Results/metrics.txt >> report.md
	echo "![Confusion Matrix](./Results/confusion_matrix.png)" >> report.md
	echo "[Data Drift Report](./Results/data_drift_report.html)" >> report.md
	echo "[Data Test Suite](./Results/data_tests.html)" >> report.md

update-branch:
	git config --global user.name $(USER_NAME)
	git config --global user.email $(USER_EMAIL)
	git commit -am "Update results"
	git push --force origin HEAD:update

monitor:
	python monitor.py
